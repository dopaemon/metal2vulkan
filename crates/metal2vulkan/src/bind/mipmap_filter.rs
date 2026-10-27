//! `generateMipmapsForTexture:` where the device's blits cannot filter the format: a compute
//! filter, each level a plain bilinear (trilinear in 3D) filter of the level above at each texel's
//! centre, in `float`, the answer a linear sampler gives there.
//!
//! The runtime makes every level by the device's linear-filter blit where it can; this filter
//! serves the formats it cannot: an extended-range format, held raw in an integer format whose
//! codes must be decoded and encoded again, an 8-bit sRGB texture held as its UNORM twin, and any
//! format whose Vulkan format lacks a blit or a linear filter in its tiling.
//!
//! **The taps.** Along each axis, destination texel `i` of `w` from `W` reads source texels `x0`
//! and `min(x0 + 1, W - 1)` with weights `1 - t` and `t`, where `x0 + t = (i + 1/2) W / w - 1/2`
//! (`t` the fraction rounded once to `f32`): a halving axis averages two texels, an axis of one
//! texel reads it alone. A 2D level's texel weighs `wx wy` of four taps, a 3D level's `wx wy wz`
//! of eight (four where the level above is one texel deep). Each layer of an array or cube is
//! filtered alone.
//!
//! **The value.** Each tap's texel decodes to its `float` channels by its format's rule: a
//! normalized code `c` times `1 / (2^b - 1)` (snorm `max(c` times `1 / (2^(b-1) - 1), -1)`), an
//! 8-bit sRGB code through Metal's twelve-bit tables (`lower::arith::srgb`), a float as it is (a
//! half and a packed float widened exactly, `RGB9E5Float`'s mantissas at their shared exponent),
//! an extended-range code `c - 384` times `1 / 510` or its sRGB twin's table step, each
//! reciprocal an `f32` constant. Each channel is the sum of its taps' weighted values in tap
//! order (x fastest, then y, then z), a tap of weight 0 left out, every product and sum rounded
//! to `f32` on its own: operations Vulkan's precision rules have a device round correctly (no
//! division, whose precision is the device's), so they leave a device that keeps `f32`
//! subnormals no other bytes than the CPU's. The sum encodes back by the format's rule: a
//! normalized code rounded to nearest even from the clamped value, a float to nearest even (a NaN
//! the format's quiet NaN), `RGB9E5Float` by Vulkan's shared-exponent conversion. `layout.rs`
//! states each rule; [`MipmapFilterChain::expected_levels`] is the same chain on the CPU.
//!
//! **The consumer** computes the levels on the device in one scratch buffer laid out by
//! [`MipmapFilterChain::plan`]: it copies the first level of every layer in (unless it reads it in
//! place, [`MipmapFilterChain::fetch`]), fills every later level with zeros where the chain says
//! so ([`MipmapFilterChain::zeroed`]: texels narrower than a word), dispatches
//! [`mipmap_filter_module`] of each later level's kernel ([`MipmapFilterLevel::kernel`]) once for
//! the level, each reading the level above, and copies the later levels back.
//!
//! **The module** (one per [`MipmapFilterKernel`] and device: the level's format built in). One
//! `GLCompute` entry `main` with local size ([`MipmapFilterModule::LOCAL_SIZE`], 1, 1), no push
//! constants, and three storage buffers in set 0, each a whole number of 4-byte words:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`MipmapFilterModule::SOURCE_BINDING`] | the level above, as [`MipmapFilterLevel`] lays it |
//! | [`MipmapFilterModule::DESTINATION_BINDING`] | the level the dispatch writes |
//! | [`MipmapFilterModule::PARAMS_BINDING`] | the level's words ([`MipmapFilterLevel::params`]) |
//!
//! A level holds each layer's texels in turn, each layer's images, rows and texels tightly
//! packed, in the bytes of the Vulkan format the runtime holds the pixel format in. Invocation
//! `(x, y)` computes destination texel `y * row + x`: a texel narrower than a word is ORed into
//! its word (atomically, the level zeroed first), a wider one stores its words (every word of
//! the level, which needs no zeros), and one past the level's texels writes nothing.
//!
//! **Synchronization** (consumer): the copy in, the fills and the params' upload →
//! `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the first dispatch;
//! each dispatch's `SHADER_STORAGE_WRITE` → `SHADER_STORAGE_READ` before the next; the last →
//! `TRANSFER_READ` before the copies back.

mod cpu;
mod fetch;
mod kernel;
mod layout;
#[cfg(test)]
pub(crate) mod tests;

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::metal::PixelFormat;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::vk::Format;
use layout::{axis_tap, header, Layout, HEADER_WORDS, TAP_WORDS};

/// The crate-owned mipmap filter module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MipmapFilterModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device.
    pub requirements: Requirements,
}

impl MipmapFilterModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The level above, a storage buffer the dispatch reads; or, in the kernel of the first
    /// later level of a chain that reads its first level in place
    /// ([`MipmapFilterChain::fetch`]), a sampled image: the first level's view in that format,
    /// its one level the first level, a 2D array of every layer (3D for a 3D texture).
    pub const SOURCE_BINDING: u32 = 0;
    /// The level the dispatch writes, a storage buffer zeroed before it where the chain says so
    /// ([`MipmapFilterChain::zeroed`]).
    pub const DESTINATION_BINDING: u32 = 1;
    /// The level's words, a storage buffer the dispatch reads.
    pub const PARAMS_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one destination texel each.
    pub const LOCAL_SIZE: u32 = kernel::LOCAL_SIZE;
}

/// One level of a [`MipmapFilterChain`] in its scratch buffer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MipmapFilterLevel {
    /// Where the level's texels start in the scratch buffer: a multiple of the chain's alignment.
    pub offset: u64,
    /// The level's bytes: every layer's texels, tightly packed, layer after layer.
    pub bytes: u64,
    /// The level's width, height and depth.
    pub extent: [u32; 3],
    /// For every level after the first: where its words start in the scratch buffer (a
    /// multiple of the chain's alignment), and the words, to upload there as little-endian
    /// words and bind at [`MipmapFilterModule::PARAMS_BINDING`] over their bytes: the level's
    /// shape and its tap tables.
    pub params: Option<(u64, Vec<u32>)>,
    /// For every level after the first: the kernel its dispatch runs
    /// ([`mipmap_filter_module`]).
    pub kernel: Option<MipmapFilterKernel>,
    /// For every level after the first: the workgroups to dispatch along x and y.
    pub groups: [u32; 2],
}

/// The scratch buffer and the dispatches of one `generateMipmapsForTexture:` (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MipmapFilterChain {
    /// Every level, the first (which the consumer copies in) first. Level `i` after the first
    /// is one dispatch of [`mipmap_filter_module`] of its kernel: level `i - 1` bound at
    /// [`MipmapFilterModule::SOURCE_BINDING`] and level `i` at
    /// [`MipmapFilterModule::DESTINATION_BINDING`], each over its bytes rounded up to 4, and its
    /// params at [`MipmapFilterModule::PARAMS_BINDING`].
    pub levels: Vec<MipmapFilterLevel>,
    /// Bytes of the scratch buffer.
    pub bytes: u64,
    /// The pixel format of the texture.
    pub pixel_format: PixelFormat,
    /// The layers of every level (an array's slices, a cube's faces; 1 in 3D).
    pub layers: u32,
    /// Whether the consumer fills each later level with zeros before its dispatch: where the
    /// format's texels are narrower than a word, which a dispatch ORs into their words. A wider
    /// texel's dispatch writes every word of its level.
    pub zeroed: bool,
    /// The format of the sampled view the first later level fetches the first level through,
    /// where the chain reads its first level in place (`plan`'s `fetch`): the first level then
    /// has no bytes in the scratch buffer (its offset is 0 and nothing is copied in), and the
    /// first later level's kernel binds the view at
    /// [`MipmapFilterModule::SOURCE_BINDING`]. `None` where the first level is copied in.
    pub fetch: Option<Format>,
}

/// The kernel a level after the first dispatches ([`MipmapFilterLevel::kernel`]): the module
/// [`mipmap_filter_module`] builds for it computes the level's format's rules, and takes the depth
/// taps of a 3D level or not and fetches the first level from its view or not, each fixed in the
/// module. Levels of one kernel share a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MipmapFilterKernel {
    pixel_format: PixelFormat,
    depth: bool,
    fetch: Option<FetchView>,
}

/// The sampled view a kernel fetches the first level from ([`MipmapFilterChain::fetch`]): its
/// format, and whether it is a 3D view (a 3D texture's) rather than a 2D array of every layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct FetchView {
    format: Format,
    volume: bool,
}

impl MipmapFilterKernel {
    /// Whether the kernel fetches the first level from its sampled view, bound at
    /// [`MipmapFilterModule::SOURCE_BINDING`] ([`MipmapFilterChain::fetch`]), rather than
    /// reading the level above from a storage buffer there.
    pub fn fetches(&self) -> bool {
        self.fetch.is_some()
    }

    /// What the kernel is built with: its format's layout, its depth taps and its fetch.
    fn spec(&self) -> Option<kernel::Spec> {
        Some(kernel::Spec {
            layout: layout::layout(self.pixel_format)?,
            depth: self.depth,
            fetch: match self.fetch {
                Some(view) => Some(kernel::Fetch {
                    components: fetch::components(view.format)?,
                    volume: view.volume,
                }),
                None => None,
            },
        })
    }
}

/// The words of one later level of `small` texels from `big` (the module docs; `layout::header`):
/// its shape and its tap tables.
fn params(big: [u32; 3], small: [u32; 3], texels: u32, row: u32) -> Option<Vec<u32>> {
    let mut words = vec![0u32; HEADER_WORDS as usize];
    words[header::TEXELS as usize] = texels;
    words[header::ROW as usize] = row;
    for a in 0..3 {
        words[(header::EXTENT + a) as usize] = small[a as usize];
        words[(header::ABOVE + a) as usize] = big[a as usize];
    }
    for a in 0..3u32 {
        words[(header::TAPS + a) as usize] = u32::try_from(words.len()).ok()?;
        for i in 0..small[a as usize] {
            let (x0, t) = axis_tap(i, big[a as usize], small[a as usize]);
            words.extend([x0, t.to_bits()]);
        }
    }
    debug_assert_eq!(
        words.len(),
        (HEADER_WORDS + TAP_WORDS * small.iter().sum::<u32>()) as usize
    );
    Some(words)
}

impl MipmapFilterChain {
    /// The chain of `levels` levels of `layers` layers of `pixel_format` from a first level of
    /// `extent` (its depth 1 unless `volume`, a 3D texture's), in a scratch buffer whose bound
    /// ranges start at multiples of `alignment` (`minStorageBufferOffsetAlignment`, at least 16
    /// here, as image copies take) and span at most `max_range` bytes (`maxStorageBufferRange`),
    /// each dispatch at most `max_groups` workgroups along x and y (`maxComputeWorkGroupCount`).
    ///
    /// `fetch` asks to read the first level in place, through a sampled view of that Vulkan
    /// format: the image's own format, or the UNORM twin of an 8-bit sRGB one (the bytes of the
    /// same channels). The chain takes it ([`MipmapFilterChain::fetch`]) where a fetch of that
    /// format recovers the texel's bytes (`fetch.rs`: the 8-bit unorm and snorm
    /// formats, the 10-bit unorm ones, the 4- and 5-bit packed unorm ones, the 16-bit unorm,
    /// snorm and float ones, and the integer carriers of the extended-range formats), and copies
    /// the first level in otherwise. A fetch of a normalized or float format reads the device's
    /// conversion, which Vulkan does not state: the consumer asks for one only on a device
    /// measured to return values whose recovery reads every code back.
    ///
    /// `None` for a format the filter does not compute (the module docs), and where the
    /// device's limits do not fit: a level past `max_range`, a dispatch past `max_groups`, or
    /// offsets past `u64`. Also `None` for fewer than two levels, no layers, or an empty extent.
    pub fn plan(
        pixel_format: PixelFormat,
        extent: [u32; 3],
        layers: u32,
        levels: u32,
        volume: bool,
        alignment: u64,
        max_range: u32,
        max_groups: [u32; 2],
        fetch: Option<Format>,
    ) -> Option<MipmapFilterChain> {
        let layout = layout::layout(pixel_format)?;
        if levels < 2 || layers == 0 || extent.contains(&0) || !alignment.is_power_of_two() {
            return None;
        }
        let alignment = alignment.max(16);
        let fetch = fetch.filter(|&format| {
            fetch::components(format).is_some_and(|components| fetch::fits(&layout, &components))
        });
        let extent_of = |level: u32| extent.map(|axis| (axis >> level.min(31)).max(1));
        let mut out: Vec<MipmapFilterLevel> = Vec::with_capacity(levels as usize);
        let mut at = 0u64;
        for level in 0..levels {
            let e = extent_of(level);
            let texels =
                u64::from(layers).checked_mul(e.iter().map(|&a| u64::from(a)).product())?;
            let bytes = texels.checked_mul(u64::from(layout.bytes))?;
            let texels = u32::try_from(texels).ok()?;
            // A first level read in place has no bytes in the scratch buffer.
            if level == 0 {
                if fetch.is_none() {
                    if bytes.next_multiple_of(4) > u64::from(max_range) {
                        return None;
                    }
                    at = bytes.checked_next_multiple_of(alignment)?;
                }
                out.push(MipmapFilterLevel {
                    offset: 0,
                    bytes,
                    extent: e,
                    params: None,
                    kernel: None,
                    groups: [0, 0],
                });
                continue;
            }
            if bytes.next_multiple_of(4) > u64::from(max_range) {
                return None;
            }
            let offset = at;
            at = at.checked_add(bytes)?.checked_next_multiple_of(alignment)?;
            let big = out[level as usize - 1].extent;
            let groups = texels.div_ceil(MipmapFilterModule::LOCAL_SIZE);
            let x = groups.min(max_groups[0]);
            if x == 0 {
                return None;
            }
            let y = groups.div_ceil(x);
            if y > max_groups[1] {
                return None;
            }
            let row = x.checked_mul(MipmapFilterModule::LOCAL_SIZE)?;
            let words = params(big, e, texels, row)?;
            if 4 * words.len() as u64 > u64::from(max_range) {
                return None;
            }
            let params_at = at;
            at = at
                .checked_add(4 * words.len() as u64)?
                .checked_next_multiple_of(alignment)?;
            let kernel = MipmapFilterKernel {
                pixel_format,
                depth: big[2] > 1,
                fetch: fetch
                    .filter(|_| level == 1)
                    .map(|format| FetchView { format, volume }),
            };
            out.push(MipmapFilterLevel {
                offset,
                bytes,
                extent: e,
                params: Some((params_at, words)),
                kernel: Some(kernel),
                groups: [x, y],
            });
        }
        Some(MipmapFilterChain {
            levels: out,
            bytes: at,
            pixel_format,
            layers,
            zeroed: layout.bytes < 4,
            fetch,
        })
    }

    /// The bytes of one texel of `pixel_format` in the chain's levels, where the filter computes
    /// its mipmaps (the module docs: every colour format Metal filters, and `BGR10_XR`,
    /// `BGRA10_XR` and their sRGB twins): the texel of the Vulkan format the runtime holds it in.
    /// `None` for a format the filter does not compute.
    pub fn texel_bytes(pixel_format: PixelFormat) -> Option<u32> {
        layout::layout(pixel_format).map(|l| l.bytes)
    }

    /// The format of the sampled view through which `plan`'s `fetch` reads the first level of an
    /// image of Vulkan format `image`: the image's own, or the UNORM twin of an 8-bit sRGB format
    /// (the same bytes, its codes undecoded). `None` where no fetch recovers the texels.
    pub fn fetch_format(image: Format) -> Option<Format> {
        let view = crate::metal::texel_format::srgb_raw_view(image).unwrap_or(image);
        fetch::components(view).map(|_| view)
    }

    /// Whether a fetch of a sampled view of `format` reads the device's conversion of each texel
    /// (a normalized or 16-bit float format: `plan`'s `fetch` there needs a device measured to
    /// return values whose recovery reads every code back) rather than its bits (the
    /// extended-range formats' integer carriers, read in place on any device). `false` too for a
    /// format no fetch reads.
    pub fn fetch_converts(format: Format) -> bool {
        fetch::components(format).is_some_and(|components| !fetch::integer(&components))
    }

    /// The levels of this chain on the CPU, from the bytes of the first level (every layer, as
    /// the scratch buffer holds them): what the copy in and the dispatches leave in the scratch
    /// buffer on a device that rounds each `f32` operation as IEEE does, every level in order,
    /// the first included. `None` for too few bytes.
    pub fn expected_levels(&self, first: &[u8]) -> Option<Vec<Vec<u8>>> {
        let layout: Layout = layout::layout(self.pixel_format)?;
        let bytes = usize::try_from(self.levels[0].bytes).ok()?;
        let mut out = vec![first.get(..bytes)?.to_vec()];
        for level in 1..self.levels.len() {
            let (big, small) = (self.levels[level - 1].extent, self.levels[level].extent);
            let next = cpu::level(&layout, &out[level - 1], big, small, self.layers);
            out.push(next);
        }
        Some(out)
    }
}

/// The binding symbol of the level above in the typed module.
const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the first level's sampled view in the typed module of a kernel that
/// fetches it ([`MipmapFilterChain::fetch`]).
const SOURCE_IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the level written in the typed module.
const DESTINATION: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the params in the typed module.
const PARAMS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(MipmapFilterModule::SET),
        Symbol::Binding(s) if *s == SOURCE || *s == SOURCE_IMAGE => {
            Some(MipmapFilterModule::SOURCE_BINDING)
        }
        Symbol::Binding(s) if *s == DESTINATION => Some(MipmapFilterModule::DESTINATION_BINDING),
        Symbol::Binding(s) if *s == PARAMS => Some(MipmapFilterModule::PARAMS_BINDING),
        _ => None,
    }
}

/// The mipmap filter module of `kernel` on `device`: built with the typed builder, checked by
/// the owned verifier against the device's profile (requirements included), serialized, and
/// numbered with the fixed bindings of [`MipmapFilterModule`].
pub fn mipmap_filter_module(
    device: &Device,
    kernel: MipmapFilterKernel,
) -> Result<MipmapFilterModule, Error> {
    let profile = device.profile();
    let spec = kernel.spec().ok_or_else(|| {
        Error::Internal(InternalError::new(
            InternalKind::Invariant,
            format!("a mipmap filter kernel of {:?}", kernel.pixel_format),
        ))
    })?;
    let m = kernel::typed_module(&spec)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the mipmap filter needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(MipmapFilterModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}
