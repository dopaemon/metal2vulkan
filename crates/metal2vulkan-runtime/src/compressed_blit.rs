//! Blits through a texture Metal keeps losslessly compressed.
//!
//! **Metal's answers** (an Apple M4 under macOS 27.0.1, plain; random bytes through every blit
//! direction, every level, slice and texture type, heap textures and views; the package share-cblit
//! on an Apple M3 Max under 26.7 and the M3 Max under 27.0.1 alike). Metal keeps a texture
//! losslessly compressed where `texture/heap_tiled.rs`'s rule holds (`metal_compresses`:
//! `allowGPUOptimizedContents`, the shorter side of level 0 16 texels or more, a colour format of
//! neither `ShaderWrite` nor `PixelFormatView` usage). A GPU blit that reads or writes such a
//! texture (a buffer into it, it into a buffer, a texture into another either way) moves its texels
//! through the compression, which writes three families' texels in their canonical encoding, the
//! same value: `BGRA10_XR` (and its `_sRGB` twin) each 16-bit field's low six bits cleared and the
//! alpha code clamped to `[384, 894]`, `RG11B10Float` a NaN field made the quiet NaN, `RGB9E5Float`
//! the shared exponent lowered to the least that holds the texel's value. The 70 other formats the
//! census tried keep their bytes, and so do a texture over a buffer or of an IOSurface and CPU
//! copies (`replace_region`, `get_bytes`) of any texture.
//!
//! **The runtime's answer** is the device's copy, which keeps the bytes it read: the canonical
//! encoding is Metal's compressor's, not the format's (Metal's own uncompressed texture of the
//! format keeps the bytes through the same blit), and a texel whose bytes no Metal write left
//! (uploaded or copied bytes) reads the same value either way: alpha reads clamped and the low six
//! bits unread for `BGRA10_XR`, a NaN for `RG11B10Float`, the value for `RGB9E5Float`. A blit through
//! such a texture announces the note of its format ([`kept_encoding`]: `extended-range-bits-kept`,
//! `device-nan-encoding`, `shared-exponent-kept`), once per device, except where the device's own
//! copy is Metal's blit of a compressed image ([`copy_rewrites`]).
//!
//! **MoltenVK** (`RuntimeDriverDefect::CompressedImageCopiesRewriteTexels`): its copies of an
//! image are Metal's blits, so a copy of a compressed `RG11B10Float` or `RGB9E5Float` image writes
//! the canonical encoding. A CPU copy, which the runtime makes a Vulkan copy, would make it too, so
//! the image of a `Shared` or `Managed` texture that Metal compresses is made uncompressed there
//! (`PixelFormatView` usage, [`Device::copy_exact_usage`]), and so is a device's own texture's
//! without `allowGPUOptimizedContents`, whose blits Metal makes on an uncompressed texture: its
//! image of the texture's own usage would be compressed (a heap's texture still takes the
//! compressed image's bytes of the heap).

use super::error::Error;
use super::texture::heap_tiled::metal_compresses;
use super::texture::Texture;
use super::texture_descriptor::{TextureDescriptor, TextureLayout};
use super::values::StorageMode;
use super::{Degradation, Device};
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan::vk::{Format, ImageCreateFlags, ImageUsageFlags};

/// The note a blit through a compressed texture of `pf` announces, whose texels Metal's
/// compression writes in their canonical encoding (the module docs); `None` for a format whose
/// blits keep their bytes on Metal too.
fn kept_encoding(pf: PixelFormat) -> Option<Degradation> {
    match pf {
        PixelFormat::BGRA10XR | PixelFormat::BGRA10XRSRGB => {
            Some(Degradation::ExtendedRangeBitsKept)
        }
        PixelFormat::RG11B10Float => Some(Degradation::DeviceNanEncoding),
        PixelFormat::RGB9E5Float => Some(Degradation::SharedExponentKept),
        _ => None,
    }
}

/// Whether Metal keeps a texture of `pixel_format` losslessly compressed and its blits rewrite
/// its texels ([`kept_encoding`]): a texture (a view's first) of `texture_type`, its level 0
/// `width` by `height`, of `samples` samples and the `(usage, allowGPUOptimizedContents)` of
/// `optimized`.
fn rewritten(
    pixel_format: PixelFormat,
    texture_type: TextureType,
    [width, height]: [u64; 2],
    samples: u64,
    (usage, allow_gpu_optimized_contents): (TextureUsage, bool),
) -> bool {
    let one_row = matches!(texture_type, TextureType::Type1D | TextureType::Type1DArray);
    let level0 = [width, if one_row { 1 } else { height }];
    let colour = !is_depth_stencil(pixel_format);
    samples == 1
        && kept_encoding(pixel_format).is_some()
        && metal_compresses(colour, usage, allow_gpu_optimized_contents, level0)
}

/// Whether a driver of `RuntimeDriverDefect::CompressedImageCopiesRewriteTexels` rewrites texels
/// in its copies of a compressed image of `format`: the Vulkan formats MoltenVK makes Metal's
/// `RG11B10Float` and `RGB9E5Float` textures of. `BGRA10_XR`'s images are of a format whose
/// copies keep their bytes.
fn copies_rewrite(format: Format) -> bool {
    matches!(
        format,
        Format::B10G11R11_UFLOAT_PACK32 | Format::E5B9G9R9_UFLOAT_PACK32
    )
}

/// Whether `texture`'s own copies write Metal's canonical encoding: on a driver whose copies of a
/// compressed image rewrite texels, an image of such a format that Metal keeps compressed (neither
/// `STORAGE` usage nor `MUTABLE_FORMAT`).
fn copy_rewrites(texture: &Texture) -> Result<bool, Error> {
    let defects = &texture.device().properties().driver_defects;
    if !defects.contains(&RuntimeDriverDefect::CompressedImageCopiesRewriteTexels) {
        return Ok(false);
    }
    let plan = texture.image_plan()?;
    Ok(copies_rewrite(plan.format)
        && !plan.usage.contains(ImageUsageFlags::STORAGE)
        && !plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT))
}

/// Announce, for a blit (`call`) that reads or writes `texture`, the note of its format's
/// canonical encoding where Metal keeps the texture losslessly compressed and the device's copy
/// keeps the bytes (the module docs).
pub(super) fn degrade_compressed_blit(call: &'static str, texture: &Texture) -> Result<(), Error> {
    if texture.over_buffer() {
        return Ok(());
    }
    let mut root = texture;
    while let Some(parent) = root.parent_texture() {
        root = parent;
    }
    let pf = texture.pixel_format();
    let compressed = rewritten(
        pf,
        root.texture_type(),
        [root.width(), root.height()],
        texture.sample_count(),
        (root.usage(), root.allow_gpu_optimized_contents()),
    );
    let Some(kind) = kept_encoding(pf).filter(|_| compressed) else {
        return Ok(());
    };
    if copy_rewrites(texture)? {
        return Ok(());
    }
    let what = match kind {
        Degradation::ExtendedRangeBitsKept => {
            "clears each field's low six bits and clamps the alpha code to [384, 894]"
        }
        Degradation::DeviceNanEncoding => "makes a NaN field the quiet NaN",
        _ => "lowers the shared exponent to the least that holds the texel's value",
    };
    texture.device().degrade(kind, || {
        (
            format!("{call} through a {pf:?} texture Metal keeps losslessly compressed"),
            format!(
                "the device's copy keeps the texels' bytes, where Metal's blit through its \
                 compression {what}: the same value in another encoding"
            ),
        )
    });
    Ok(())
}

impl Device {
    /// `usage`, the usage of the image of `format` of a texture of `descriptor` (checked as
    /// `layout`), with `PixelFormatView` where this device's copies of the image would rewrite
    /// texels Metal keeps (`RuntimeDriverDefect::CompressedImageCopiesRewriteTexels`): an image
    /// of a format whose copies rewrite, which the device's Metal keeps compressed (an image of
    /// `usage` with GPU-optimized contents, [`rewritten`]), of a texture whose bytes Metal keeps
    /// through such a copy: a `Shared` or `Managed` texture, which CPU copies read and write, or
    /// one without `allowGPUOptimizedContents`, which Metal keeps uncompressed, so that its GPU
    /// blits keep the bytes too. Metal keeps an image of that usage uncompressed (MoltenVK makes a
    /// `PixelFormatView` Metal texture of a `MUTABLE_FORMAT` image with no list of formats), so its
    /// copies keep the bytes. `None` elsewhere: a `Private` texture with GPU-optimized contents,
    /// whose bytes no CPU copy reads or writes, keeps the compression.
    pub(super) fn copy_exact_usage(
        &self,
        descriptor: &TextureDescriptor,
        layout: &TextureLayout,
        format: Format,
        usage: TextureUsage,
    ) -> Option<TextureUsage> {
        let properties = self.properties();
        let defect = RuntimeDriverDefect::CompressedImageCopiesRewriteTexels;
        let reachable = matches!(
            descriptor.resource_options.storage_mode(),
            Ok(StorageMode::Shared | StorageMode::Managed)
        );
        let [width, height, _] = layout.extent;
        let compressed = rewritten(
            layout.pixel_format,
            layout.texture_type,
            [u64::from(width), u64::from(height)],
            u64::from(layout.samples),
            (usage, true),
        );
        let kept = reachable || !descriptor.allow_gpu_optimized_contents;
        let rewrites = properties.driver_defects.contains(&defect)
            && kept
            && copies_rewrite(format)
            && compressed;
        rewrites.then_some(usage | TextureUsage::PIXEL_FORMAT_VIEW)
    }
}
