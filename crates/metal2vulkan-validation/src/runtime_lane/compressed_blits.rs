//! Blits through textures Metal keeps losslessly compressed on the device the cfg selects, under
//! the validation layer. Metal's blit moves such a texture's texels through its compression, which
//! writes `BGRA10_XR`'s, `RG11B10Float`'s and `RGB9E5Float`'s texels in their canonical encoding,
//! the same value (our probes on an Apple M4 and the package share-cblit on an Apple M3 Max); the
//! runtime's answer is the device's copy, which keeps the bytes (`compressed_blit.rs`):
//!
//! - A GPU blit into, out of, or between textures of `ShaderRead` usage at least 16 texels on a
//!   side keeps every byte, as do the same blits of a texture without
//!   `allowGPUOptimizedContents`, 8 texels high, or over a buffer; where the device's own copy of
//!   a compressed image is Metal's blit (`RuntimeDriverDefect::CompressedImageCopiesRewriteTexels`:
//!   MoltenVK's `RG11B10Float` and `RGB9E5Float` images) a `Private` compressed texture's rows are
//!   the device's, printed and not held.
//! - CPU copies keep the bytes of every texture, and so does a CPU copy in followed by a blit out,
//!   or a blit in followed by a CPU copy out.
//! - A copy out into a buffer writes only the texels' bytes at the buffer's rows.
//! - Each format's blits announce the note of the encoding they keep: `extended-range-bits-kept`,
//!   `device-nan-encoding`, `shared-exponent-kept`.
//! - A heap answers a `Shared` compressed texture's size and alignment as a `Private` one's, and
//!   two `Shared` textures placed back to back in a heap at that size keep their CPU bytes (on
//!   MoltenVK, whose copies of a compressed `RG11B10Float` or `RGB9E5Float` image rewrite texels,
//!   the image of a texture a CPU copy reaches is uncompressed, in the space of Metal's compressed
//!   size).

use super::degrade_notices::counts;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    BlitCommandEncoder, BlitOption, Buffer, CommandQueue, CpuCacheMode, Degradation, Device,
    HazardTrackingMode, HeapDescriptor, HeapType, Origin, Region, ResourceOptions, Size,
    StorageMode, Texture, TextureDescriptor,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(60);

/// The textures' side.
const SIDE: u64 = 64;

/// The formats whose blits through Metal's compression change bytes, their texel bytes and the
/// note a blit through one announces.
const FORMATS: [(PixelFormat, u64, Degradation); 4] = [
    (PixelFormat::BGRA10XR, 8, Degradation::ExtendedRangeBitsKept),
    (
        PixelFormat::BGRA10XRSRGB,
        8,
        Degradation::ExtendedRangeBitsKept,
    ),
    (PixelFormat::RG11B10Float, 4, Degradation::DeviceNanEncoding),
    (PixelFormat::RGB9E5Float, 4, Degradation::SharedExponentKept),
];

/// A splitmix64 stream: every bit of each byte random.
struct Rng(u64);

impl Rng {
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// Whether `device`'s own copy of a compressed image of `pf` is Metal's blit, which writes the
/// canonical encoding (`RuntimeDriverDefect::CompressedImageCopiesRewriteTexels`: MoltenVK's
/// `RG11B10Float` and `RGB9E5Float` images).
fn copies_rewrite(device: &Device, pf: PixelFormat) -> bool {
    device
        .properties()
        .driver_defects
        .contains(&RuntimeDriverDefect::CompressedImageCopiesRewriteTexels)
        && matches!(pf, PixelFormat::RG11B10Float | PixelFormat::RGB9E5Float)
}

/// Hold `out` to `bytes`, kept by the device's copy, unless the device's own copy rewrites a
/// compressed image (`rewritten`): then print how many of its words differ.
fn kept(out: &[u8], bytes: &[u8], rewritten: bool, what: &str) {
    if rewritten {
        let moved = (out.chunks(4).zip(bytes.chunks(4)))
            .filter(|(o, b)| o != b)
            .count();
        eprintln!("COMPRESSED-BLITS {what}: the device's copy rewrote {moved} words");
        return;
    }
    assert_eq!(out, bytes, "{what}");
}

struct Lane {
    device: Device,
    queue: CommandQueue,
}

/// A 2D texture of `pf`, `ShaderRead` usage and `storage`, `SIDE` x `height`, with `levels`
/// levels, with `allowGPUOptimizedContents` where `optimized`.
fn descriptor(
    pf: PixelFormat,
    optimized: bool,
    storage: StorageMode,
    height: u64,
    levels: u64,
) -> TextureDescriptor {
    let mut d = TextureDescriptor::texture_2d(pf, SIDE, height, false);
    d.usage = TextureUsage::SHADER_READ;
    d.allow_gpu_optimized_contents = optimized;
    d.mipmap_level_count = levels;
    d.resource_options = ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    );
    d
}

impl Lane {
    /// A texture of [`descriptor`]'s.
    fn texture(
        &self,
        pf: PixelFormat,
        optimized: bool,
        storage: StorageMode,
        height: u64,
        levels: u64,
    ) -> Texture {
        let d = descriptor(pf, optimized, storage, height, levels);
        self.device.new_texture(&d).unwrap()
    }

    /// A Shared buffer of `bytes`.
    fn buffer(&self, bytes: &[u8]) -> Buffer {
        let buffer = self
            .device
            .new_buffer(bytes.len() as u64, ResourceOptions::shared())
            .unwrap();
        // SAFETY: the buffer holds `bytes.len()` bytes at its contents; no command uses it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                buffer.contents().unwrap().as_ptr(),
                bytes.len(),
            );
        }
        buffer
    }

    /// Run `record` in one blit encoder, and wait for it.
    fn run(&self, record: impl FnOnce(&mut BlitCommandEncoder)) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        record(&mut blit);
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// `bytes` blitted into `texture`'s `level` (`w` x `h`, `per` bytes a texel).
    fn blit_in(&self, texture: &Texture, level: u64, [w, h]: [u64; 2], per: u64, bytes: &[u8]) {
        let source = self.buffer(bytes);
        self.run(|blit| {
            let size = Size::new(w, h, 1);
            let (row, at) = (w * per, Origin::new(0, 0, 0));
            blit.copy_from_buffer_to_texture(
                &source,
                0,
                row,
                0,
                size,
                texture,
                0,
                level,
                at,
                BlitOption::None,
            )
            .unwrap();
        });
    }

    /// `texture`'s `level` blitted out into a buffer of 0xee bytes at `offset`, rows
    /// `row_bytes` apart: the buffer's bytes.
    fn blit_out(
        &self,
        texture: &Texture,
        level: u64,
        [w, h]: [u64; 2],
        (offset, row_bytes): (u64, u64),
    ) -> Vec<u8> {
        let n = (offset + row_bytes * h) as usize;
        let out = self.buffer(&vec![0xee; n]);
        self.run(|blit| {
            let (size, at) = (Size::new(w, h, 1), Origin::new(0, 0, 0));
            blit.copy_from_texture_to_buffer(
                texture,
                0,
                level,
                at,
                size,
                &out,
                offset,
                row_bytes,
                0,
                BlitOption::None,
            )
            .unwrap();
        });
        // SAFETY: the command buffer completed; the buffer holds `n` bytes.
        unsafe { std::slice::from_raw_parts(out.contents().unwrap().as_ptr(), n) }.to_vec()
    }

    /// `source` copied whole into `destination`.
    fn copy(&self, source: &Texture, destination: &Texture, height: u64) {
        self.run(|blit| {
            let (size, at) = (Size::new(SIDE, height, 1), Origin::new(0, 0, 0));
            blit.copy_from_texture_to_texture(source, 0, 0, at, size, destination, 0, 0, at)
                .unwrap();
        });
    }
}

fn put(texture: &Texture, height: u64, per: u64, bytes: &[u8]) {
    let region = Region::new_2d(0, 0, SIDE, height);
    texture
        .replace_region(region, 0, 0, bytes, SIDE * per, 0)
        .unwrap();
}

fn get(texture: &Texture, height: u64, per: u64) -> Vec<u8> {
    let mut bytes = vec![0xee; (SIDE * height * per) as usize];
    let region = Region::new_2d(0, 0, SIDE, height);
    texture
        .get_bytes(&mut bytes, SIDE * per, 0, region, 0, 0)
        .unwrap();
    bytes
}

#[test]
fn blits_through_a_compressed_texture_keep_their_bytes_and_announce_it() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |_| {}).unwrap();
    let name = device.name().to_owned();
    let lane = Lane {
        queue: device.new_command_queue().unwrap(),
        device,
    };
    let mut rng = Rng(0xfc5_b117);
    let mut announced = std::collections::BTreeSet::new();
    for (pf, per, note) in FORMATS {
        let rewrites = copies_rewrite(&lane.device, pf);
        let whole = [SIDE, SIDE];
        let n = (SIDE * SIDE * per) as usize;
        // A buffer into a texture and back: compressed, without GPU-optimized contents, and
        // 8 high.
        for (optimized, height, compressed) in
            [(true, SIDE, true), (false, SIDE, false), (true, 8, false)]
        {
            let bytes = rng.bytes((SIDE * height * per) as usize);
            let texture = lane.texture(pf, optimized, StorageMode::Private, height, 1);
            lane.blit_in(&texture, 0, [SIDE, height], per, &bytes);
            let out = lane.blit_out(&texture, 0, [SIDE, height], (0, SIDE * per));
            let what = format!("{pf:?} optimized {optimized}, {height} high on {name}");
            kept(&out, &bytes, compressed && rewrites, &what);
        }
        // Level 3 of a mipmapped compressed texture, out into a buffer at an offset of 4 and
        // rows 12 bytes longer than the texels': only the texels' bytes are written.
        let texture = lane.texture(pf, true, StorageMode::Private, SIDE, 4);
        let small = [SIDE >> 3, SIDE >> 3];
        let bytes = rng.bytes((small[0] * small[1] * per) as usize);
        lane.blit_in(&texture, 3, small, per, &bytes);
        let row = small[0] * per + 12;
        let out = lane.blit_out(&texture, 3, small, (4, row));
        for y in 0..small[1] as usize {
            let at = 4 + y * row as usize;
            let texels = (small[0] * per) as usize;
            kept(
                &out[at..at + texels],
                &bytes[y * texels..(y + 1) * texels],
                rewrites,
                &format!("{pf:?} level 3 row {y} on {name}"),
            );
            assert!(out[at + texels..at + row as usize]
                .iter()
                .all(|&b| b == 0xee));
        }
        assert!(out[..4].iter().all(|&b| b == 0xee));
        // CPU copies keep the bytes of a compressed shared texture, and so does a CPU copy in and
        // a blit out, or a blit in and a CPU copy out (its image uncompressed where the device's
        // copies of a compressed one rewrite texels).
        let shared = lane.texture(pf, true, StorageMode::Shared, SIDE, 1);
        let bytes = rng.bytes(n);
        put(&shared, SIDE, per, &bytes);
        assert_eq!(get(&shared, SIDE, per), bytes, "{pf:?} CPU on {name}");
        let out = lane.blit_out(&shared, 0, whole, (0, SIDE * per));
        assert_eq!(out, bytes, "{pf:?} CPU in, blit out on {name}");
        let bytes = rng.bytes(n);
        lane.blit_in(&shared, 0, whole, per, &bytes);
        assert_eq!(
            get(&shared, SIDE, per),
            bytes,
            "{pf:?} blit in, CPU out on {name}"
        );
        // A heap answers the Shared texture's size as the Private one's, and two Shared
        // textures placed back to back in a heap at that size keep their CPU bytes, each
        // blitting out as it holds them: the image bound at each offset fits the space the answer
        // reserves (on MoltenVK an uncompressed image in the compressed one's space).
        let size = |storage| {
            let d = descriptor(pf, true, storage, SIDE, 1);
            lane.device.heap_texture_size_and_align(&d).unwrap()
        };
        let answer = size(StorageMode::Shared);
        assert_eq!(
            answer,
            size(StorageMode::Private),
            "{pf:?} heap size on {name}"
        );
        let second = answer.size.next_multiple_of(answer.align);
        let options = ResourceOptions::shared();
        let heap_descriptor =
            HeapDescriptor::new(HeapType::Placement, second + answer.size, options);
        let heap = lane.device.new_heap(&heap_descriptor).unwrap();
        let shared_descriptor = descriptor(pf, true, StorageMode::Shared, SIDE, 1);
        let placed = [0, second].map(|offset| {
            heap.new_texture_at_offset(&shared_descriptor, offset)
                .unwrap()
        });
        let bytes = [rng.bytes(n), rng.bytes(n)];
        for (texture, bytes) in placed.iter().zip(&bytes) {
            put(texture, SIDE, per, bytes);
        }
        for (k, (texture, bytes)) in placed.iter().zip(&bytes).enumerate() {
            assert_eq!(
                get(texture, SIDE, per),
                *bytes,
                "{pf:?} heap texture {k} CPU on {name}"
            );
            let out = lane.blit_out(texture, 0, whole, (0, SIDE * per));
            assert_eq!(
                out, *bytes,
                "{pf:?} heap texture {k} CPU in, blit out on {name}"
            );
        }
        // Texture to texture: out of a compressed texture's CPU bytes into a plain one, and
        // from a plain one's into a compressed one.
        let plain = lane.texture(pf, false, StorageMode::Shared, SIDE, 1);
        let bytes = rng.bytes(n);
        put(&shared, SIDE, per, &bytes);
        lane.copy(&shared, &plain, SIDE);
        assert_eq!(
            get(&plain, SIDE, per),
            bytes,
            "{pf:?} out of compressed on {name}"
        );
        let bytes = rng.bytes(n);
        put(&plain, SIDE, per, &bytes);
        lane.copy(&plain, &shared, SIDE);
        assert_eq!(
            get(&shared, SIDE, per),
            bytes,
            "{pf:?} into compressed on {name}"
        );
        // Into a texture over a buffer, its rows 256 bytes past the texels': the texels at its
        // rows, the bytes between them kept.
        let row = SIDE * per + 256;
        let backing = lane.buffer(&vec![0xee; (row * SIDE) as usize]);
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        d.usage = TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let over = backing.new_texture(&d, 0, row).unwrap();
        let bytes = rng.bytes(n);
        put(&shared, SIDE, per, &bytes);
        lane.copy(&shared, &over, SIDE);
        // SAFETY: every command buffer completed; the buffer holds `row * SIDE` bytes.
        let held = unsafe {
            std::slice::from_raw_parts(backing.contents().unwrap().as_ptr(), (row * SIDE) as usize)
        };
        let want = bytes;
        let texels = (SIDE * per) as usize;
        for y in 0..SIDE as usize {
            let at = y * row as usize;
            assert_eq!(
                held[at..at + texels],
                want[y * texels..(y + 1) * texels],
                "{pf:?} over a buffer row {y} on {name}"
            );
            assert!(held[at + texels..at + row as usize]
                .iter()
                .all(|&b| b == 0xee));
        }
        // The blits through the compressed textures announced the encoding they keep, once per
        // device and kind.
        let notices = counts(&messages.take_notices());
        eprintln!("COMPRESSED-BLITS {pf:?} on {name}: announced {notices:?}");
        announced.extend(notices.into_keys());
        assert!(announced.contains(&note), "{pf:?} on {name}: {announced:?}");
    }
    layer_reported_nothing_at_all(&messages);
}
