//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{GuestIsolation, HandleDomain, PointerDomain};

/// Policy inputs of `Device::new`. Every field but `decoded_texels`, `heap_compression`,
/// `completion`, `batch_order`, `render_lowering` and `host_page_textures` can change emitted
/// code, so the profile carries it (containment as the encoding it selects) and its fingerprint
/// covers it; those six are the runtime's alone (the texels of a carried texture's decoded twin, a
/// heap's compressed texture's image, when the host sees a command buffer complete, how a batch
/// begins after the earlier ones of its queue, where a render encoder's calls are lowered, and
/// what holds a sampled texture over host pages), and no profile carries them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources (v1 ships `ResidentTable` only).
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve (v1 ships `Table` only).
    pub handle_domain: HandleDomain,
    /// Whether accesses are contained to the resources the binder supplied.
    pub containment: ContainmentPolicy,
    /// Whether guest shaders are isolated from each other's resources.
    pub isolation: GuestIsolation,
    /// Whether an 8-bit unsigned normalized texture's linear sample may be the device's own
    /// filter on a device measured within one code of Metal; by default it is Metal's filter,
    /// emulated exactly.
    pub unorm8_linear_filter: Unorm8LinearFilterPolicy,
    /// Whether a device that does not sample through Metal's own sampler samples natively (the
    /// default) or emulates Metal's sampler exactly.
    pub sampling: SamplingPolicy,
    /// Whether, under native device sampling, a linear sample of a texture slot bound to a
    /// two-plane YCbCr texture is the device's own filter of its texels (the default) or Metal's
    /// filter of the format, emulated in a variant of the pipeline the draw binding it makes.
    pub ycbcr_filter: YCbCrFilterPolicy,
    /// Whether the math library's functions (`sin`, `cos`, `exp`, `log`, `pow` and the rest), a
    /// precise division and Metal's fast reciprocal, and a fused multiply-add are the device's
    /// own instructions (native device maths, the default) or the translation's routines of
    /// Metal's answers.
    pub float_maths: FloatMathsPolicy,
    /// Whether a texel's `float` or `half` value is the device's conversion of its codes (native
    /// device sampling, the default off Apple's GPUs) or Metal's, converted exactly in the
    /// shader.
    pub texel_conversion: TexelConversionPolicy,
    /// Whether a normalized vertex attribute's value is the device's fixed-function conversion of
    /// its codes (the default) or Metal's, divided exactly in the shader.
    pub vertex_normalized: VertexNormalizedPolicy,
    /// Whether a NaN result's bits, and the operand Metal's minimum and maximum functions
    /// (`fmin`, `fmax`, `min3`, `max3`, `clamp`, `saturate`) pick at a NaN or a signed zero, are
    /// the device's own (the default) or Metal's.
    pub nan_and_min_max: NanMinMaxPolicy,
    /// Whether Metal's `dot`, `mix` and `fast::sqrt` round in Metal's order (its fused chain of
    /// the products, its two fused multiply-adds, its `x · rsqrt(x)`) or are the device's single
    /// instructions (the default).
    pub rounding_order: RoundingOrderPolicy,

    /// Whether a texture whose format the device lacks, held in a carrier (an ASTC LDR texture,
    /// or an EAC 11-bit one), samples a decoded twin of compact codes (the default) or of Metal's
    /// values exactly. The runtime's choice alone: no emitted code changes.
    pub decoded_texels: DecodedTexelsPolicy,
    /// Whether a heap's colour texture that Metal keeps losslessly compressed, on a driver whose
    /// compressed images hide writes through an alias, is the driver's compressed image (the
    /// default) or one the driver keeps uncompressed in the heap's bytes. The runtime's choice
    /// alone: no emitted code changes.
    pub heap_compression: HeapCompressionPolicy,
    /// Whether command buffers committed on one `VkQueue` that the runtime's scheduler finds ready
    /// together go to the driver in one kernel submission, each seen complete once the last of
    /// them completed (the default, on a driver that merges such submissions), or each in a
    /// submission of its own, seen complete at its own end. The runtime's choice alone: no
    /// emitted code changes.
    pub completion: CompletionPolicy,
    /// Whether a batch of command buffers on one `VkQueue` begins after the batches before it
    /// there by their ends (the default, on a driver measured to run no two kernel submissions of
    /// one queue side by side), or by a barrier for each access that depends on theirs. The
    /// runtime's choice alone: no emitted code changes.
    pub batch_order: BatchOrderPolicy,
    /// Whether a render encoder's call is lowered inside the call, which returns every refusal
    /// (the default), or after it returned, a refusal only the lowering finds failing its command
    /// buffer. The runtime's choice alone: no emitted code changes.
    pub render_lowering: RenderLoweringPolicy,
    /// Whether a texture that only samples or reads, over a no-copy buffer's host pages or over a
    /// buffer the runtime placed in fresh host pages, is an image aliasing those pages in place
    /// (the default, where the device's own check shows its first move keeps every byte), or a
    /// shadow the encoders copy the texels into and back. The runtime's choice alone: no emitted
    /// code changes.
    pub host_page_textures: HostPageTexturePolicy,
}

impl DevicePolicy {
    /// The untrusted-guest production policy: containment and isolation on, Metal's 8-bit linear
    /// filter emulated exactly where the emulation runs, native device sampling (a two-plane YCbCr texture's samples among them),
    /// native device maths, the device's own texel
    /// conversions, the device's NaN and minimum and maximum, the device's rounding order, compact
    /// codes in a carried texture's decoded twin, the driver's compressed image of a heap's texture
    /// Metal keeps compressed, command buffers ready together in one kernel submission, batches of
    /// one queue ordered by their ends where the driver runs them one after another, and a sampled
    /// texture over host pages in place where the device's check passed.
    pub const UNTRUSTED: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
        containment: ContainmentPolicy::Contained,
        isolation: GuestIsolation::On,
        unorm8_linear_filter: Unorm8LinearFilterPolicy::Exact,
        sampling: SamplingPolicy::NativeDevice,
        ycbcr_filter: YCbCrFilterPolicy::Device,
        float_maths: FloatMathsPolicy::Native,
        texel_conversion: TexelConversionPolicy::Device,
        vertex_normalized: VertexNormalizedPolicy::Device,
        nan_and_min_max: NanMinMaxPolicy::Native,
        rounding_order: RoundingOrderPolicy::Native,
        decoded_texels: DecodedTexelsPolicy::Compact,
        heap_compression: HeapCompressionPolicy::Compressed,
        completion: CompletionPolicy::Grouped,
        batch_order: BatchOrderPolicy::Ends,
        render_lowering: RenderLoweringPolicy::AtCall,
        host_page_textures: HostPageTexturePolicy::InPlace,
    };

    /// This policy with Metal's answer wherever the choices offer it: its maths functions,
    /// divisions and fused multiply-adds ([`FloatMathsPolicy::Exact`]), its texel conversions
    /// ([`TexelConversionPolicy::Exact`]), its normalized vertex attributes
    /// ([`VertexNormalizedPolicy::Exact`]), its
    /// NaN and minimum and maximum ([`NanMinMaxPolicy::Exact`]), its rounding order
    /// ([`RoundingOrderPolicy::Exact`]), its filter of a
    /// two-plane YCbCr texture
    /// ([`YCbCrFilterPolicy::Exact`]), its decode of a carried texture's texels
    /// ([`DecodedTexelsPolicy::Exact`]), a heap's texture whose bytes alias the heap's
    /// as written ([`HeapCompressionPolicy::Uncompressed`]), each command buffer seen complete
    /// at its own end ([`CompletionPolicy::Exact`]), each batch ordered after the earlier ones of
    /// its queue by
    /// the accesses it depends on alone ([`BatchOrderPolicy::Exact`]), each render encoder
    /// call lowered inside the call, which returns every refusal
    /// ([`RenderLoweringPolicy::AtCall`]), and every texture over host pages a shadow, whose
    /// copies touch its texels alone ([`HostPageTexturePolicy::Shadow`]). Sampling and every
    /// other field are kept.
    pub const fn exact_answers(self) -> DevicePolicy {
        DevicePolicy {
            ycbcr_filter: YCbCrFilterPolicy::Exact,
            float_maths: FloatMathsPolicy::Exact,
            texel_conversion: TexelConversionPolicy::Exact,
            vertex_normalized: VertexNormalizedPolicy::Exact,
            nan_and_min_max: NanMinMaxPolicy::Exact,
            rounding_order: RoundingOrderPolicy::Exact,
            decoded_texels: DecodedTexelsPolicy::Exact,
            heap_compression: HeapCompressionPolicy::Uncompressed,
            completion: CompletionPolicy::Exact,
            batch_order: BatchOrderPolicy::Exact,
            render_lowering: RenderLoweringPolicy::AtCall,
            host_page_textures: HostPageTexturePolicy::Shadow,
            ..self
        }
    }
}

impl Default for DevicePolicy {
    /// [`DevicePolicy::UNTRUSTED`].
    fn default() -> DevicePolicy {
        DevicePolicy::UNTRUSTED
    }
}

/// The containment choice. The encoding (robustness or in-shader checks) is selected by
/// `Device::new` from the enabled robustness features; the policy only says whether to contain.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContainmentPolicy {
    /// Every access stays inside the resources the binder supplied (the default).
    Contained,
    /// No containment. Only valid with `GuestIsolation::Off`; recorded in the fingerprint.
    UnsafeOff,
}

/// How an 8-bit unsigned normalized texture's linear sample is filtered: a 2D texture's or 2D
/// array's sample of a slot bound in `R8Unorm`, `RG8Unorm`, `RGBA8Unorm` or `BGRA8Unorm`, through a
/// constexpr sampler with normalized coordinates or a runtime sampler slot, without comparison or
/// offset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Unorm8LinearFilterPolicy {
    /// Metal's filter, emulated exactly: the default (the owner's ruling of 2026-10-11; on the
    /// consumer's fragments the device's filter bought no GPU time and cost module size and
    /// making time).
    #[default]
    Exact,
    /// The device's own filter at the centre of Metal's 1/256 footprint step, on a device
    /// measured to filter the class within one code (1/255) of Metal
    /// ([`DriverDeviation::Unorm8LinearFilter`](crate::error::DriverDeviation::Unorm8LinearFilter)):
    /// such a sample records
    /// [`PrecisionNote::DeviceUnorm8LinearFilter`](crate::pipeline::PrecisionNote::DeviceUnorm8LinearFilter)
    /// and requires `VkProperty::Unorm8LinearFilter`. An unmeasured device keeps Metal's filter.
    DeviceWithinOneCode,
}

/// How a device that does not sample through Metal's own sampler (MoltenVK on Apple's GPU,
/// `DriverDeviation::MetalSampler`) answers a module's samples, gathers and comparisons: natively,
/// its own sampler's answer (the default: the owner's ruling of 2026-10-11, making time before
/// exactness), or Metal's sampler, emulated exactly. A device that samples through Metal's own
/// sampler answers as Metal under either.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplingPolicy {
    /// Native device sampling: each sample, gather and comparison goes to the device's sampler
    /// as on a device whose sampler is Metal's, with no emulation of Metal's filters, footprint,
    /// level selection, anisotropy or comparisons in the module. Such a module records
    /// [`PrecisionNote::DeviceSampling`](crate::pipeline::PrecisionNote::DeviceSampling), whose
    /// docs state the bound measured against Metal, and requires `VkProperty::DeviceSampling`.
    #[default]
    NativeDevice,
    /// Metal's sampler, emulated exactly where the device's differs: the module's size and the
    /// driver's making time grow with each sample.
    Exact,
}

/// How a linear sample of a texture slot bound to a two-plane YCbCr texture
/// (`metal::format::ycbcr_format`) is filtered where the module leaves the slot's samples to the
/// device's sampler: on a device that samples natively ([`SamplingPolicy::NativeDevice`], the
/// module requiring `VkProperty::DeviceSampling`), and on one that samples through Metal's own
/// sampler (MoltenVK on Apple's GPU) under native sampling. The texture's image holds Metal's reads
/// of the planes exactly, steps of `1/2048` or `1/8192`; Metal's filter sums those steps on their
/// grid, each term truncated, at its own weights, so the device's filter of the same texels lands
/// off Metal's answer: natively within the bound `PrecisionNote::DeviceSampling` states, through
/// Metal's own sampler within one step of the grid (`ycbcr-filter-not-exact`). Under
/// [`SamplingPolicy::Exact`] every sample is Metal's, through the slot's variant on MoltenVK.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum YCbCrFilterPolicy {
    /// The device's filter of the texture's texels, as for every other format (the default: the
    /// owner's ruling, making time before exactness, recorded and announced): the pipeline a draw
    /// binding such a texture takes is the one a draw binding any other texture takes, so the draw
    /// makes none. A natively sampling module records
    /// [`PrecisionNote::DeviceSampling`](crate::pipeline::PrecisionNote::DeviceSampling), whose
    /// docs state the bound measured against Metal; through Metal's own sampler the module records
    /// its filter note under the bound format, which the runtime announces as
    /// `ycbcr-filter-not-exact` where a draw binds such a texture, and a `half` sample's one texel
    /// on the grid `8192` through a runtime sampler rounds to nearest even
    /// ([`PrecisionNote::HalfTexelConversion`](crate::pipeline::PrecisionNote::HalfTexelConversion)).
    #[default]
    Device,
    /// Metal's filter of the format, emulated: a draw that binds such a texture at a slot the
    /// function samples linearly takes the slot's YCbCr variant (`RuntimeFact::YCbCrTexture`), a
    /// pipeline of its own, made at that draw.
    Exact,
}

/// How the `float` and `half` maths functions, divisions and fused multiply-adds Metal answers
/// beyond Vulkan's precision table are computed on a device whose functions are not Metal's own
/// (the owner's "Maths functions" of 2026-10-11, native by default). On a device whose functions
/// are Metal's (MoltenVK on an Apple GPU, `DriverDeviation::MetalFloatFunctions`), the device's
/// instructions are Metal's own whatever the policy. A NaN such a function answers is
/// [`NanMinMaxPolicy`]'s, `dot`, `mix` and `fast::sqrt` are [`RoundingOrderPolicy`]'s, and a `half`
/// result's rounding is the device's own (`lower::arith::narrow`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FloatMathsPolicy {
    /// Native device maths (the default): the math library's functions are the device's
    /// `GLSL.std.450` instructions (or Metal's compositions of them), a precise division and
    /// Metal's fast reciprocal the device's `OpFDiv`, and a fused multiply-add without
    /// `OpFmaKHR` the device's `Fma`, each within Vulkan's precision table rather than Metal's
    /// answer, and each recording its kind
    /// ([`PrecisionNote::NativeTranscendentals`](crate::pipeline::PrecisionNote::NativeTranscendentals),
    /// [`NativeDivision`](crate::pipeline::PrecisionNote::NativeDivision),
    /// [`NativeFma`](crate::pipeline::PrecisionNote::NativeFma)) with the bound in its docs.
    #[default]
    Native,
    /// The translation's routines, which answer as Metal does on every device: larger modules and
    /// longer pipeline creation.
    Exact,
}

/// How a texel's value reaches a read, sample or gather with a `float` or `half` result on a
/// device that does not sample through Metal's own texture unit (`VkProperty::MetalSampler`). On
/// such a device Metal's conversion of a texel is emulated by a routine that tells the bound
/// format's class apart at every access, and the class is data, so no driver folds it: on the
/// consumer's fragments those routines were most of the driver's making time (296 calls
/// in `Xgc`, 311 in `A2Xghfc`).
///
/// The choice is made per access. A read's texel is this policy's. A sample's or gather's is too
/// where Metal's sampler is emulated ([`SamplingPolicy::Exact`]); under native device sampling
/// ([`SamplingPolicy::NativeDevice`], the module requiring `VkProperty::DeviceSampling`) it is the
/// device's under either policy: the device's sampler decodes its samples and gathers, and an exact
/// conversion of a sample's texel reads the runtime sampler's filter to tell one texel from a
/// filtered sum, which splits the pipeline by the bound sampler state. Measured on the RTX 5080
/// Laptop GPU, `Exact`'s reads cost the consumer's `Xgc` 3 to 5% of its making and 4 to 18% of its
/// GPU time per draw.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TexelConversionPolicy {
    /// Native device sampling, the default (the owner's ruling of 2026-10-11: making time before
    /// exactness off Apple's GPUs, with the bound recorded and announced): the device's `float`
    /// value of each texel, and its `half` rounded to nearest even. Such an access records
    /// [`PrecisionNote::DeviceTexelConversion`](crate::pipeline::PrecisionNote::DeviceTexelConversion)
    /// (`float`) or
    /// [`PrecisionNote::HalfTexelConversion`](crate::pipeline::PrecisionNote::HalfTexelConversion)
    /// (`half`), whose docs state the bound.
    #[default]
    Device,
    /// Metal's conversion, emulated exactly by the bound format's class, of every read, and of
    /// every sample and gather where Metal's sampler is emulated: what validation compares against
    /// Metal's answers.
    Exact,
}

/// Who converts a **normalized** vertex attribute: the Metal vertex and attribute formats whose
/// fetch asks for a code divided by its own maximum (`UCharNormalized` through
/// `Short4Normalized`, `UChar4Normalized_BGRA`, `Int1010102Normalized`, `UInt1010102Normalized`).
///
/// Metal's conversion is `c / (2^b - 1)` unsigned and `max(c / (2^(b-1) - 1), -1)` signed, each
/// correctly rounded, which is what Vulkan's `UNORM` and `SNORM` fixed-function vertex input
/// states as well. A driver whose conversion multiplies by the reciprocal instead is one `f32`
/// ulp off at some codes, and nothing in the attribute's own Vulkan format says to correct it
/// after the load, so the exact form moves the fetch into the shader, where
/// `lower::stage_io::fetch`'s `div_exact` computes the quotient in integer arithmetic.
///
/// The choice is per attribute, and it is a variant count as much as a cost: every format the
/// device fetches in fixed function shares one module, while a pulled attribute's decode is its
/// own program. Over `cases::vertex_formats`' 45-format sweep on an Apple M3 Max
/// (`moltenvk-m3`, `tests::engine_vertex_variants`) the `float4` entry compiles 2 modules under
/// [`VertexNormalizedPolicy::Device`] and 21 under [`VertexNormalizedPolicy::Exact`], and the
/// `half4` entry 1 against 20. The runtime's own counter reads the same over the whole case on an
/// Apple M3 Max: `shader_modules_made` 2 of 45 creations against 21, and 1 against 20.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VertexNormalizedPolicy {
    /// The device's, by fixed-function vertex input of the attribute's own `UNORM` or `SNORM`
    /// format (the default): one module for every format the device fetches, and the conversion
    /// the driver implements.
    #[default]
    Device,
    /// Metal's, computed exactly in the shader: the attribute is pulled from its vertex buffer
    /// and its codes divided by `lower::stage_io::fetch`'s integer sequence, as a format the
    /// fixed-function path cannot fetch at all already is. One module per format.
    Exact,
}

/// Whose NaN a `float` or `half` result answers, and which operand Metal's minimum and maximum
/// functions (`fmin`, `fmax`, `min3`, `max3`, `clamp`, and `saturate`, Metal's
/// `clamp(x, +0, 1)`) pick at a NaN, a signed zero or an `f32` denormal, on a device not measured
/// to answer Metal's (the owner's "the bit pattern of NaN values, and min and max when NaN or
/// signed zero is involved" of 2026-10-11). A device measured to answer Metal's canonical NaN
/// (`DriverDeviation::MetalCanonicalNans`) keeps it, and one measured to compute Metal's minimum
/// and maximum (`DriverDeviation::MetalMinMax`) or whose functions are Metal's own
/// (`DriverDeviation::MetalFloatFunctions`, MoltenVK on an Apple GPU) keeps Metal's picks, under
/// either setting.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NanMinMaxPolicy {
    /// The device's own (the default): every NaN Metal answers is the device's (its sign and
    /// payload; no select of Metal's bits), and those functions are the device's `NMin`, `NMax`
    /// and `NClamp`. On a device that rounds a `half` through its conversion under the float
    /// controls (ANV, RADV), a `half` read from its bits is the device's widening of them, and a
    /// `half` texel takes that conversion's NaN. Such a module records
    /// [`PrecisionNote::NativeNanEncoding`](crate::pipeline::PrecisionNote::NativeNanEncoding)
    /// or [`NativeMinMax`](crate::pipeline::PrecisionNote::NativeMinMax), whose docs state the
    /// bound.
    #[default]
    Native,
    /// Metal's: the translation selects Metal's NaN where the device may answer another, and
    /// Metal's operand where the device's instructions may pick another; a `half` widened to
    /// `float` answers `0x7fc00000` for every NaN, as Metal does. A `half` texel read, sampled or
    /// gathered keeps the NaN the device decoded through native half rounding too and through the
    /// conversion under the float controls, and a
    /// `half` read from its bits is decoded in integer arithmetic where the default takes the
    /// device's widening (a `half` stored back unchanged keeps its bits under either).
    Exact,
}

/// The order in which Metal's `dot`, `mix` and `fast::sqrt` on `float` and `half` round, on a
/// device whose functions are not Metal's own (the owner's "rounding order in dot products, blends
/// between two values (mix) and fast square root" of 2026-10-11). On a device whose functions are
/// Metal's (MoltenVK on an Apple GPU, `DriverDeviation::MetalFloatFunctions`) they are Metal's
/// under either setting.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RoundingOrderPolicy {
    /// The device's single instructions (the default): `dot` the device's `OpDot`, `mix` its
    /// `FMix` (a `half` result summed in `float` and rounded once), and `fast::sqrt` its `Sqrt`.
    /// Such a module records
    /// [`PrecisionNote::NativeDot`](crate::pipeline::PrecisionNote::NativeDot),
    /// [`NativeMix`](crate::pipeline::PrecisionNote::NativeMix) or
    /// [`NativeTranscendentals`](crate::pipeline::PrecisionNote::NativeTranscendentals)
    /// (`fast::sqrt`, with `DeviceDenormals`), whose docs state the bound.
    #[default]
    Native,
    /// Metal's order: `dot` the fused chain of its products, `mix` `fma(t, y, fma(-t, x, x))`, and
    /// `fast::sqrt` `x · rsqrt(x)` with Metal's answers at a zero, a denormal and `+inf`.
    Exact,
}

/// What texels the runtime holds in the decoded twin of a texture whose blocks it holds in a
/// carrier on a device without the format (the runtime's `texture/decoded_twin.rs`), which the
/// texture's samples, reads and gathers read: an ASTC LDR texture's (`bind::AstcDecode`), and an
/// EAC 11-bit texture's (`bind::EtcDecode`). Metal reads an ASTC LDR channel as a 12-bit value,
/// `(C >> 4) / 4095` of the decode's 16-bit result, which no 8- or 16-bit texel holds; it reads
/// an EAC 11-bit channel as `c / 2047` (signed `c / 1023`), which no 16-bit normalized code
/// holds. (An ASTC sRGB texture's twin holds the 8-bit codes `C >> 8` under either, which its
/// slots read raw and decode through Metal's table, exact; so does a PVRTC texture's twin, and an
/// 8-bit ETC2 or EAC texture's.) The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DecodedTexelsPolicy {
    /// Compact texels (the default): an ASTC LDR texture's `R8G8B8A8_UNORM` codes, 4 bytes a
    /// texel, each channel the code nearest Metal's value, at most `1/510` off it. An EAC 11-bit
    /// texture's 16-bit normalized codes (`R16_UNORM`, `R16_SNORM` and their two-channel forms),
    /// 2 bytes a channel, each nearest Metal's value, at most half a 16-bit step off it.
    /// Announced `decoded-texel-precision` where such a texture is made.
    #[default]
    Compact,
    /// Metal's values exactly, an ASTC LDR texture's in `R32G32B32A32_SFLOAT`, 16 bytes a texel
    /// (an EAC 11-bit texture's in `R32_SFLOAT` or `R32G32_SFLOAT`, 4 bytes a channel), where the
    /// device samples, filters and copies into the format (else the compact texels, announced).
    Exact,
}

/// The image of a heap's colour texture that Metal keeps losslessly compressed (with
/// `allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, and a
/// level 0 of 16 texels or more a side), on a driver whose compressed images may keep their texels
/// apart from the memory they are bound to (`DriverFact::CompressedImagesHideAliasedWrites`: ANV
/// on Intel's GPUs). Metal's own texture shows no aliased write as written either: an M3 Max read
/// opaque magenta (`0xffff00ff`) in every texel of drawn content a buffer's fill over the
/// texture's bytes covered, and the fill's bytes where the texture held only a clear. A texture
/// Metal keeps uncompressed takes the uncompressed image under either value. The runtime's choice
/// alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeapCompressionPolicy {
    /// The driver's compressed optimal image (the default), its fast clears, compression and
    /// tiling kept: a write through a buffer or texture over its bytes may leave its texels as
    /// they were or change them otherwise than Metal's, and its writes may show in those bytes
    /// compressed. Announced `heap-texture-compressed` where such a texture is made.
    #[default]
    Compressed,
    /// An image the driver keeps uncompressed in the heap's bytes, as for a texture Metal keeps
    /// uncompressed: host-transfer usage where the device enabled `hostImageCopy`, else a linear
    /// image, else the optimal one (each announced as such), so a write through a buffer over the
    /// texture's bytes shows in its texels as written.
    Uncompressed,
}

/// When the host sees a command buffer complete, on a driver that merges a submission signalling
/// nothing into the next one in the same `vkQueueSubmit` (Mesa's common queue,
/// `vk_queue_submits_merge`: ANV, RADV, lavapipe, NVK, Turnip and the other Mesa drivers of a kernel
/// submission), where each kernel submission costs CPU time for every `VkDeviceMemory` of the
/// device (ANV on i915 lists each in every execbuffer). Elsewhere every command buffer goes in a
/// submission of its own under either value. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompletionPolicy {
    /// Up to four batches of command buffers on one `VkQueue` that the scheduler finds ready
    /// together, none of which signals a binary semaphore, takes a fence, moves the caller's
    /// images or does first work, and none but the first of which waits for anything not seen
    /// complete, go in one `vkQueueSubmit`, and only the last signals, the queue's timeline and
    /// every event they signal (the default): one kernel submission for all of them. The commands,
    /// their barriers and their GPU times are unchanged; each but the last is seen complete (its
    /// status, handlers and waits) once the last completed, and an event's values they signal are
    /// reached then, late by the GPU time of the batches after it, never early and in commit
    /// order. The submission checks of up to three batches right behind a queue's next batch run
    /// with its own, in commit order, so checked command buffers are found ready together: such a
    /// check runs before the previous batch's submission rather than after it. Announced
    /// `grouped-completion` at the first such submission.
    #[default]
    Grouped,
    /// Each batch in a submission of its own that signals its own values, so each command buffer
    /// is seen complete, and its event signals land, at its own end, as Metal's are; each batch's
    /// submission checks run once the batch before it was submitted.
    Exact,
}

/// How a batch of command buffers on one `VkQueue` (the work the runtime submits for a command
/// buffer between its event waits and signals) begins after the earlier batches there. Metal
/// orders a command buffer after the earlier ones of its queue only where its work depends on
/// theirs, so an independent one may run beside the one before it; it writes its `GPUStartTime`
/// as its work begins. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BatchOrderPolicy {
    /// By their ends (the default) on a driver measured to run no two kernel submissions of one
    /// queue side by side, with or without a barrier between them, where it runs the same work
    /// recorded in one command buffer side by side (ANV, RADV and NVIDIA's driver): every batch
    /// ends with a barrier that orders all its commands before every command after it on its
    /// queue, so a later batch plans no barrier for the accesses of earlier ones, and its first
    /// encoder's GPU start time is written once the work before it ended, at the bottom of the
    /// pipe. Each batch then still starts once the one before it ended, as those drivers start it
    /// anyway. On any other driver, as [`Exact`](Self::Exact).
    #[default]
    Ends,
    /// A barrier at a batch's start for each access that depends on an earlier batch's, as for
    /// the accesses within a batch, and its first encoder's GPU start time written as it begins,
    /// at the top of the pipe, where the driver may run it beside the batch before it.
    Exact,
}

/// Where a render encoder's calls are lowered: where each call's commands, descriptors, pipelines
/// and resources are worked out and recorded. Metal's encoder checks a call as it is made and
/// returns; its GPU work, and any failure found there, comes after. The runtime's choice alone: no
/// emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderLoweringPolicy {
    /// After the call returned: the call makes Metal's checks of its arguments and
    /// of the encoder's state, copies what it was lent and holds what it names, and returns; the
    /// encoder lowers it later, in order, before its command buffer's submission (on the
    /// runtime's lowering thread while the program goes on, the encoder's end too, and the rest
    /// at what follows the end in the command buffer; `commit` makes its checks and returns, its
    /// own work run after the end, and later commits behind it). A refusal only
    /// the lowering finds fails the command buffer
    /// (`CommandBufferError::CallRefused`, naming the call), which then runs nothing of that
    /// encoder or of any after it.
    Worker,
    /// Inside the call, as each call is made (the default): every refusal, the lowering's too,
    /// returns from the call itself.
    #[default]
    AtCall,
}

/// What holds a texture over host pages whose declared usage only samples or reads it (no
/// `RenderTarget`, `ShaderWrite` or `ShaderAtomic`): over a no-copy buffer's pages (imported by
/// their pointer or as dma-bufs) or over a buffer the runtime placed in fresh host pages it
/// imported. Vulkan makes an image bound to imported memory start `UNDEFINED`
/// (`VUID-VkImageCreateInfo-pNext-01443`), and lets its first layout transition read and write
/// every byte bound to it, where Metal keeps every CPU write to those bytes. The runtime's choice
/// alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HostPageTexturePolicy {
    /// A linear image aliasing the pages (the default), on a device whose own check, run once per
    /// driver and release, shows that this first move of such an image leaves every byte alone
    /// while the CPU writes beside it; and where the device lays the image out at the buffer's
    /// row pitch. Elsewhere, and for a texture that renders or writes, the shadow below. A
    /// texture made before the check's verdict is known is a shadow too.
    #[default]
    InPlace,
    /// A shadow on every device: an image of the runtime's own memory the encoders copy the
    /// texels into before each use and merge back after each write, whose copies touch the
    /// texels alone.
    Shadow,
}
