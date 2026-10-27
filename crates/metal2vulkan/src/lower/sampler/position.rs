//! Metal's texel-space position of a normalized coordinate, and the coordinate that makes a
//! device's gather or linear sample of a pixel sampler read the texels Metal reads.
//!
//! **The position** (Apple M3 Max, macOS 26.7, and Apple M4, macOS 27.0.1, byte-identical; the
//! tie, gather, and bicubic probes). Per axis of size `N` and normalized coordinate `u`, Apple's
//! texture unit first wraps `u` into one period, in `f32`, where the axis repeats: `u' = u -
//! floor(u)` under `repeat`, `u' = u - 2 floor(u / 2)` under `mirrored_repeat` (a period of two),
//! the subtraction rounded toward positive infinity (`wrap`; it rounds at all only where `-1 < u <
//! 0`), or exactly where `N` is a power of two (below). Then it scales: `X = u' N` rounded to `f32`
//! with ties away from zero. A clamping axis scales `u` itself. Every filter reads its footprint
//! from `X`: the linear and bicubic filters and a gather the quantized `q = floor(256 X + 1/2) -
//! 128`, a nearest filter `floor(X)`, and the taps then wrap modulo `N` or mirror with period `2N`.
//! The probes measured the wrap on the linear, nearest, gather, and bicubic filters (every lane,
//! both chips), and the ties on the linear, gather, and bicubic filters under `clamp_to_zero`,
//! where a product exactly half a unit in the last place from two `f32`s decides the footprint. The
//! wrap's rounding comes from the position cases (`wrg_sampler_positions` at 13x11, 7x5, and 255x3
//! on the M3, whose outputs the M4's equal byte for byte): rounded up, every linear, nearest,
//! gather, and bicubic result of their repeating and mirrored axes is Metal's; rounded to nearest
//! even, with ties away, or not at all, it misses lanes with `-1 < u < 0` whose wrapped `u'` needs
//! more bits than an `f32` holds.
//!
//! **A power-of-two size wraps exactly**. On a repeating or mirrored axis whose size is a power of
//! two, Metal reads the footprints of the exact `u'`, per axis: the linear-filter probes (M4 and
//! M3, byte-identical) at 2D 16x8 and 4x32 and 3D 16x16x16, 8x4x2 and 16x8x4, and on the 4-texel
//! axis of a 9x4x11 texture whose other axes round up, and the position cases at 16x8, 4x32 and
//! 16x11 (linear, nearest, and gathers, `gather_compare` included; M3 and M4). `wrap` rounds the
//! subtraction down there, which reads the same footprints: `u' N` is then exact, and every step a
//! footprint reads (an integer, or `i + 1/2` plus a multiple of `1/512`) lies on the `1/512` grid,
//! which an `f32` holds exactly below `2^15`, so rounding `u'` down never carries `X` across one.
//! Rounded up, it reads one `1/256` step, sometimes a texel, high where `-1 < u < 0` lands just
//! below a step.
//!
//! Multiplying first and wrapping the taps after (as a Vulkan device does) reads another footprint wherever `u` lies outside `[0, 1)` on a size that is not a
//! power of two and `u N` lands near a step: the product rounds at the magnitude of `u N`, not of
//! `u' N`. At a power-of-two size `u N` is exact, and multiplying first reads Metal's footprints.
//!
//! **Ties away from zero, in integers** ([`position`]). `p = u' N` rounded to nearest even is a tie
//! rounded toward zero exactly when the bits of the exact product below `p`'s last place are
//! `100…0` and `p`'s significand is even; then `X` is the next `f32` away from zero, `p`'s bits
//! plus one. The exact product's low bits are those of `bits(u') * N` (an `OpIMul`, modulo `2^32`):
//! `u'`'s exponent and implicit bit sit at bit 23 and up, so they never reach the bits the test
//! reads. Those are bits `0` to `s`, where `s` is `p`'s biased exponent less `u'`'s (the sign bits
//! cancel, `p` having `u'`'s sign); a shift left by `31 - s` puts them at the top of the word,
//! where the tie reads `0x4000_0000`. This is exact for every finite normal `u'` and every
//! `N < 2^22`, where `s <= 22` and the bits stay below bit 23; Metal's sizes are at most `2^14`. A
//! denormal `u'` may be adjusted wrongly, by one unit in the last place of an `X` below `2^-100`,
//! which moves no footprint. A tie moves a footprint only at a negative position, or at one of
//! `2^14` or more. Where the two neighbours of a tie lie on either side of a footprint's step (an
//! integer, or a gather's `i + 1/2 - 1/512`), the upper neighbour is the step, whose significand is
//! even below `2^14`: the even rounding picks it, as ties away from zero does for a positive
//! position, while at a negative one ties away from zero picks the lower. That is why the probes'
//! deciding lanes were clamped axes' negative coordinates, and a repeating or mirrored axis, whose
//! position is never negative, differs only at sizes of at least `2^13` (mirrored) or `2^14`
//! (repeating). A product past `f32`'s range gives an unspecified `X`, as before (the footprints
//! read no non-finite position: `bicubic`). The test is 12 integer instructions on every device;
//! `OpFmaKHR` for the product's error would take 11, and only where `shaderFmaFloat32` is enabled.
//!
//! **A texel offset on a repeating or mirrored axis** ([`offset_dropped`]). Metal adds a constant
//! texel offset to the tap and then wraps it, as Vulkan places it, but on a repeating or mirrored
//! axis whose level size `n` lies in `[2, 7]` and whose texture's base-level size on that axis is
//! not a power of two: there it drops the offset on that axis. At the base level that is a size of
//! 3, 5, 6, or 7. The offset probes:
//! - round 1, Apple M4 on macOS 27.0.1 and Apple M3 Max on macOS 26.7, byte-identical, each run
//!   with and without Metal's validation layer: every 2D size from 1 to 17 per axis and eight
//!   larger controls, offsets -8 to 7 per axis, nearest and linear samples, gathers of each
//!   component, and a depth texture's gathers; a clamping axis places the offset as Vulkan does
//!   on every size;
//! - round 2, the M4: base levels up to 16384, 2D arrays and 3D textures, per axis;
//! - round 3, the M4: every level of 18 bases at an explicit level, nearest and linear samples
//!   (level sizes 2 and 4 drop it under a base that is not a power of two and keep it under one
//!   that is; sizes 3, 5 and 6 drop it, and 8 and above keep it, under either), and each level
//!   above the base through a view of that one level, which reads exactly as the level does: the
//!   view's parent's base-level size decides.
//!
//! An emulated footprint applies the rule to its taps (`bicubic::axis`), and a snapped axis that
//! repeats or mirrors subtracts the offset the device will add where Metal drops it
//! ([`snapped_axis`]). The base-level size the rule reads is a view's parent's, which a shader
//! cannot query: a view whose first level lies above its parent's base reads the bound view's
//! own, so on an axis of size 2 or 4 under a parent base that is not a power of two (where the
//! view's base is) the rule would keep the offset Metal drops. The binder states those axes in the
//! slot's `SpecKey::OffsetBaseAxes` specialization (0 for every texture that is not such a view),
//! whose tests (`texture::OffsetBase`) the rule reads; a heap element's access reads none, and
//! records `PrecisionNote::ViewBaseTexelOffset` (the residue there).
//!
//! **A subnormal coordinate** ([`subnormal_flush`]). Apple's texture unit reads a subnormal
//! normalized coordinate as the zero of its sign before the address mode decides (measured on an
//! Apple M3 Max), so a border axis reads texel 0 just outside the level where a device that keeps
//! the subnormal reads the border. The border axes of a constexpr sampler flush it before anything
//! reads it (`emit::plane_coord`), so the emulations and the device's own sampler alike are handed
//! Metal's coordinate; every other address mode reads the same texel either way.
//!
//! **The snapped coordinate** ([`snapped`]). A pixel sampler's coordinate, scaled for the
//! device, reads whichever texels the device's own arithmetic gives near a texel edge. Where
//! Metal's texels follow from the coordinate as given, a snapped axis instead hands the device a
//! coordinate half a step from every edge of them: a gather's `(i0 + 1) / N`, the centre of its
//! 2x2 block ([`AxisSnap::Gather`]), and a linear sample's footprint cell ([`AxisSnap::Cell`]),
//! with `i0` and the cell from Metal's position. Any device's rounding then reads Metal's block or
//! step, and the device keeps everything else: the texel conversion, the filter's weights, the
//! border, the comparison, the offset (added to the snapped index, then wrapped or clamped, as
//! Metal adds it), and the layer. A gather's `i0` is computed in `f32`, `floor(floor(256 X + 1/2)
//! / 256 - 1/2)`, exact for `|X| < 2^15` (every position of a Metal texture's repeating or mirrored
//! axis, at most `2N`, included). On a clamping axis `i0` is first clamped into `[-10, N + 8]`:
//! past either end, with any offset in Metal's gather range `[-8, 7]`, both texels lie outside the
//! level on the same side, so they read the same edge texels or border as the exact `i0` would,
//! and the coordinate stays small. On a repeating or mirrored axis whose size drops the offset
//! ([`offset_dropped`]), the snapped index is first moved back by the offset, which the device then
//! adds: the device reads the texel or block Metal reads there, the one without the offset.
//!
//! **A runtime or argument buffer's sampler** names its state in its word, read in the function:
//! where the word states pixel coordinates, its gather of a 2D texture or array snaps behind it
//! ([`gathered_by_word`]), each axis at the coordinate as given, with `i0` clamped into the wider
//! `[-2N - 10, 2N + 8]`, which holds every position of a repeating or mirrored axis and reads a
//! clamping axis's same edge texels or border; and where it also states linear min and mag
//! filters, its sample snaps onto its footprint cell ([`celled_by_word`]), as a constexpr pixel
//! sampler's does (that sample reads Metal's rule at the coordinate as given: our probe's 10,250
//! lanes of a runtime pixel sampler on both chips). A normalized word's gather and every other
//! word's sample keep the device's coordinate. On a `MirrorClampToEdge` axis (which no constexpr
//! sampler states) Metal's position is a clamping axis's, `u N` unwrapped, its taps mirrored once
//! and clamped as Vulkan's `MIRROR_CLAMP_TO_EDGE` mirrors and clamps them ([`position_by_mode`]),
//! and a texel offset is dropped by the sizes that drop it on a repeating axis (every gather of
//! 13x11, 7x5 and 16x8 textures through a slot's and an argument buffer's runtime sampler, with
//! offsets of 2 and -3, on an Apple M3 Max and an Apple M4 alike), a pixel coordinate's as a
//! normalized one's, and a pixel word's texel centre moves back by a dropped offset
//! (`emit::pixel_centre`).

use crate::air::abi::{AddressMode, Coordinates, Filter};
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::texture::OffsetBase;
use crate::metal::SamplerAddressMode;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, GlslOp, ImageQuery, Type as SType};

use super::bicubic::TapAddress;
use super::{CombinedImage, SampleOp};

/// The census words of one axis's [`position`]: the conversion of the size, the wrap (a mirrored
/// axis's four instructions, and the rounding's thirteen, which choose up or down by the size), the
/// product, and the ties' twelve integer instructions, with their constants (186 words on a
/// mirrored axis). Every estimate that emits a position adds it per axis.
pub const POSITION_WORDS: u32 = 200;

/// How the texels a snapped plane axis hands the device follow from the position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisSnap {
    /// A gather's 2x2 block: the coordinate `(i0 + 1) / N`.
    Gather,
    /// A linear sample's footprint cell: the coordinate `(floor(256 X - 127.5) + 128) / 256 / N`,
    /// the centre of the `1/256` step `q` Metal's filter reads `X` as, half a step from either edge
    /// of it.
    Cell,
}

/// What a snap hands the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SnapKind {
    /// The plane coordinate with [`Snap::axes`]' axes snapped.
    Axes(AxisSnap),
    /// A runtime or argument buffer's sampler's gather of a 2D texture or 2D array: behind the
    /// sampler's word, where it states pixel coordinates, each plane axis takes
    /// [`AxisSnap::Gather`]'s coordinate, the centre of the 2x2 block Metal reads at the coordinate
    /// as given ([`gathered_by_word`]); a normalized coordinate is the device's gather's.
    /// [`Snap::axes`] is empty: the word names the modes.
    WordGather,
    /// A runtime or argument buffer's sampler's sample of a texture of `axes` plane axes: where
    /// the word states pixel coordinates
    /// and linear min and mag filters, each plane axis takes [`AxisSnap::Cell`]'s coordinate, the
    /// centre of the footprint step Metal reads the coordinate as given at, as a constexpr linear
    /// pixel sampler's sample does there ([`celled_by_word`]); every other word keeps the scaled
    /// coordinate. [`Snap::axes`] is empty.
    WordCell { axes: u8 },
}

/// A gather or linear sample whose coordinate is snapped onto Metal's texels (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Snap {
    pub kind: SnapKind,
    /// Per plane axis (`s`, then `t`, then `r`), its addressing where the axis is snapped, or
    /// `None` where the device reads the coordinate as given.
    pub axes: Vec<Option<TapAddress>>,
}

impl Snap {
    /// A runtime or argument buffer's sampler's snap behind its word ([`SnapKind::WordGather`],
    /// [`SnapKind::WordCell`]).
    pub fn by_word(&self) -> bool {
        matches!(self.kind, SnapKind::WordGather | SnapKind::WordCell { .. })
    }

    /// The census words: the image and its size query; per snapped plane axis the extractions, the
    /// [`position`], the index and its clamp, and the coordinate, and the composite.
    pub fn words(&self) -> u32 {
        // The coordinate is in texels: Metal's position `X` is the coordinate as given.
        let per_axis = 8 + match self.kind {
            SnapKind::Axes(AxisSnap::Gather) => GATHER_AXIS_WORDS,
            SnapKind::Axes(AxisSnap::Cell) => CELL_AXIS_WORDS,
            SnapKind::WordGather => {
                return 32 + WORD_REACH_WORDS + 2 * (8 + GATHER_AXIS_WORDS + WORD_AXIS_WORDS);
            }
            SnapKind::WordCell { axes } => {
                return 32
                    + WORD_REACH_WORDS
                    + u32::from(axes) * (8 + CELL_AXIS_WORDS + WORD_AXIS_WORDS);
            }
        };
        let snapped = self.axes.iter().filter(|a| a.is_some()).count() as u32;
        32 + snapped * per_axis + self.axes.len() as u32 * 8
    }

    /// The census words of the offset rule ([`offset_dropped`]) on each axis this snap may move
    /// back, for an access with a texel offset: each snapped plane axis's, and each axis of a word
    /// gather.
    pub fn offset_words(&self) -> u32 {
        let axes = match self.kind {
            // The device's sampler places a pixel cell's.
            SnapKind::WordCell { .. } => 0,
            SnapKind::WordGather => 2,
            SnapKind::Axes(_) => self.axes.iter().filter(|a| a.is_some()).count() as u32,
        };
        axes * OFFSET_RULE_WORDS
    }
}

/// A gather axis's words past its position: the two extractions, `i0` (six instructions and their
/// constants), its clamp, and the coordinate.
const GATHER_AXIS_WORDS: u32 = 96;
/// A footprint cell's words past its position: the two extractions, the step (seven instructions
/// and their constants), and the coordinate.
const CELL_AXIS_WORDS: u32 = 96;
/// A word snap's test of its word (`SamplerWord::PIXEL_COORDINATES`, and a cell's linear filters):
/// the mask, the comparison, and their constants.
const WORD_REACH_WORDS: u32 = 24;
/// A word snap's axis words past its snapped coordinate: the extractions of the coordinate as
/// given and the device's, the size's conversion, a gather's widened clamp, and the select.
const WORD_AXIS_WORDS: u32 = 64;
/// The tap addressing of address mode `m`.
pub(super) fn address(m: AddressMode) -> TapAddress {
    match m {
        AddressMode::ClampToEdge => TapAddress::Edge,
        AddressMode::Repeat => TapAddress::Repeat,
        AddressMode::MirroredRepeat => TapAddress::Mirror,
        AddressMode::ClampToZero | AddressMode::ClampToBorder => TapAddress::Border,
    }
}

/// The census words of the subnormal flush ([`subnormal_flush`]): the bitcast, the magnitude, the
/// comparison with each axis's threshold, the sign, the select, and the bitcast back, with their
/// constant vectors.
pub const FLUSH_WORDS: u32 = 56;

/// The plane axes of a sample of `op` on a texture of `shape` through the static sampler `stat`
/// (`None`: an argument buffer's) whose coordinate the module flushes a subnormal of to the zero
/// of its sign before the sample, as a bit per axis ([`emit_flush`]).
///
/// **Metal flushes a subnormal texture coordinate.** Apple's texture unit reads a subnormal
/// normalized coordinate as the zero of its sign before the address mode decides, so a clamping
/// axis that reads the border outside the level reads texel 0 there, where a device that keeps
/// the subnormal reads the border. Measured on an Apple M3 Max (macOS 27.0.1) through a
/// 13x11 `R32Float` texture and constexpr clamp-to-zero samplers: at `-1.1754942e-38`
/// (bits `0x007f_ffff`, the largest subnormal magnitude) and at `-1e-40`, `-1e-42` and
/// `-1e-44` the nearest sample reads texel 0 and the linear one averages it with the border,
/// while at `f32::MIN_POSITIVE` negated (bits `0x0080_0000`) and at every larger magnitude down
/// to `-1.2e-38`, `-1e-30` and `-1e-37` both read the border: the boundary is `f32`'s own
/// subnormal edge, one unit in the last place wide, and not a quantization of the coordinate
/// (two ladders of 40 rungs each walking up to a clamp-to-zero edge from outside it, from
/// `2^-1` down to `2^-40` of a texel, read the border on every rung, which no fixed-point
/// quantum of any width would).
///
/// The flush is emitted where the measurement reaches: a constexpr sampler with normalized
/// coordinates, of a single-sampled 1D, 2D, or 3D texture or array, on each axis the sampler
/// addresses through the border (`clamp_to_zero`, and `clamp_to_border_color`, which the
/// lowering does not distinguish: the border test is the only address decision a subnormal can
/// flip, and the two modes differ only in the border colour). A repeating, mirrored or
/// edge-clamping axis reads texel 0 at a subnormal coordinate with the flush or without it, so
/// none is emitted for one. A cube's coordinate is a direction, a pixel sampler's is in texels,
/// and a runtime or argument buffer's sampler's addressing is not known when the function
/// compiles: none is measured, and none takes the flush.
pub(super) fn subnormal_flush(
    stat: Option<(SamplerRef, Coordinates)>,
    shape: ImageShape,
) -> Result<u8, Error> {
    let Some((SamplerRef::Constexpr(words), Coordinates::Normalized)) = stat else {
        return Ok(0);
    };
    let plane = match (shape.dim, shape.ms) {
        (ImageDim::D1, false) => 1,
        (ImageDim::D2, false) => 2,
        (ImageDim::D3, false) => 3,
        _ => return Ok(0),
    };
    let state = super::state_of(words)?;
    let modes = [state.address_s, state.address_t, state.address_r];
    let mut axes = 0u8;
    for (axis, mode) in modes.into_iter().take(plane).enumerate() {
        if address(mode) == TapAddress::Border {
            axes |= 1 << axis;
        }
    }
    Ok(axes)
}

/// `coord` (`lanes` floats) with the subnormal lanes of `axes` ([`subnormal_flush`]) replaced by
/// the zero of their sign, as Metal's texture unit reads them.
///
/// The test and the replacement are integer, so a device that flushes subnormals in its own float
/// arithmetic and one that keeps them emit the same answer: a lane whose magnitude bits lie below
/// `f32::MIN_POSITIVE`'s (`0x0080_0000`) becomes its sign bit alone, and every other lane is kept.
/// An axis outside `axes` takes the threshold 0, which no magnitude is below, so it is kept
/// whatever it holds (a zero included, which the replacement would leave unchanged anyway).
pub(super) fn emit_flush(
    fb: &mut FunctionBuilder<'_>,
    coord: SValue,
    lanes: u8,
    axes: u8,
) -> BResult<SValue> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let float = fb.module().ty(SType::Float { width: 32 })?;
    let vector = |fb: &mut FunctionBuilder<'_>, ty, n: u8| match n {
        1 => Ok(ty),
        n => fb.module().ty(SType::Vector { elem: ty, len: n }),
    };
    let uvec = vector(fb, u32_, lanes)?;
    let fvec = vector(fb, float, lanes)?;
    let splat = |fb: &mut FunctionBuilder<'_>, bits: Vec<u32>| -> BResult<SValue> {
        let mut parts = Vec::new();
        for b in &bits {
            let c = fb.module().constant(Constant::Scalar {
                ty: u32_,
                bits: ConstBits(u64::from(*b)),
            })?;
            parts.push(c);
        }
        let c = match parts.as_slice() {
            [one] => *one,
            many => fb.module().constant(Constant::Composite {
                ty: uvec,
                parts: many.to_vec().into(),
            })?,
        };
        fb.module().const_value(c)
    };
    let magnitudes = splat(fb, vec![0x7fff_ffff; usize::from(lanes)])?;
    let signs = splat(fb, vec![0x8000_0000; usize::from(lanes)])?;
    let thresholds = splat(
        fb,
        (0..lanes)
            .map(|axis| match axes & (1 << axis) {
                0 => 0,
                _ => 0x0080_0000,
            })
            .collect(),
    )?;
    let bits = fb.convert(ConvOp::Bitcast, coord, uvec)?;
    let magnitude = fb.binary(BinOp::BitwiseAnd, bits, magnitudes)?;
    let small = fb.binary(BinOp::ULessThan, magnitude, thresholds)?;
    let sign = fb.binary(BinOp::BitwiseAnd, bits, signs)?;
    let flushed = fb.select(small, sign, bits)?;
    fb.convert(ConvOp::Bitcast, flushed, fvec)
}

/// The snap of `op` on a texture of `shape` through the static sampler `stat` (`None`: an
/// argument buffer's), where the module docs snap it. A pixel sampler's coordinate is Metal's
/// position as given, which the device's sampler, reading the coordinate over the size, does not
/// reproduce, its own or Metal's alike. So on every device:
///
/// - a constexpr pixel sampler's gather of a 2D texture or 2D array snaps both plane axes from the
///   coordinate as given;
/// - a runtime or argument buffer's sampler's gather of a 2D texture or 2D array snaps behind the
///   sampler's word, where it states pixel coordinates ([`SnapKind::WordGather`]);
/// - a constexpr pixel sampler's linear sample of a 1D, 2D, or 3D texture or array with a
///   floating-point texel (an `integer` one's reads one texel) snaps every plane axis onto the
///   centre of its footprint cell, unless the sampler is anisotropic;
/// - a runtime or argument buffer's sampler's sample of such a texture, a comparison's included,
///   snaps onto its footprint cell where its word states a linear pixel sampler
///   ([`SnapKind::WordCell`]).
///
/// Every other gather and sample is the device's, and the module requires the device's sampler
/// (`SamplerFamily::metal_sampler`). A cube's coordinate is a direction through a pixel sampler
/// too, so a cube takes no snap.
pub(super) fn plan(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    stat: Option<(SamplerRef, Coordinates)>,
    (shape, integer): (ImageShape, bool),
) -> Result<Option<Snap>, Error> {
    // A constexpr sampler's state, and whether its coordinate is in texels (not a cube's).
    let state = match stat {
        Some((SamplerRef::Constexpr(words), c)) => Some((
            super::state_of(words)?,
            c == Coordinates::Pixel && shape.dim != ImageDim::Cube,
        )),
        _ => None,
    };
    // The plane axes of a 1D, 2D, or 3D texture or array, single-sampled.
    let plane = match (shape.dim, shape.ms) {
        (ImageDim::D1, false) => 1,
        (ImageDim::D2, false) => 2,
        (ImageDim::D3, false) => 3,
        _ => 0,
    };
    match op {
        SampleOp::Gather { .. } => match state {
            Some((s, true)) if shape.dim == ImageDim::D2 && !shape.ms => Ok(Some(Snap {
                kind: SnapKind::Axes(AxisSnap::Gather),
                axes: vec![Some(address(s.address_s)), Some(address(s.address_t))],
            })),
            // A runtime or argument buffer's sampler's gather, behind its word. The device's own
            // gather of a normalized coordinate is its block: the word snaps a pixel coordinate
            // alone, and the module requires the device's sampler.
            None if shape.dim == ImageDim::D2 && !shape.ms => {
                cx.families.sampler.metal_sampler = true;
                Ok(Some(Snap {
                    kind: SnapKind::WordGather,
                    axes: Vec::new(),
                }))
            }
            _ => {
                cx.families.sampler.metal_sampler = true;
                Ok(None)
            }
        },
        // A linear pixel sample's footprint cell.
        SampleOp::Sample { dref: None, .. }
            if state.as_ref().is_some_and(|(s, pixel)| {
                *pixel
                    && plane > 0
                    && !integer
                    && s.min_filter == Filter::Linear
                    && s.mag_filter == Filter::Linear
            }) =>
        {
            let Some((s, _)) = state else {
                return Ok(None);
            };
            if s.max_anisotropy > 1 {
                return Ok(None);
            }
            Ok(Some(Snap {
                kind: SnapKind::Axes(AxisSnap::Cell),
                axes: [s.address_s, s.address_t, s.address_r]
                    .into_iter()
                    .take(plane)
                    .map(|m| Some(address(m)))
                    .collect(),
            }))
        }
        // A runtime or argument buffer's sampler's sample, a comparison's included, onto its
        // footprint cell where its word states a linear pixel sampler, as a constexpr one's. An
        // integer texture's sample reads one texel whatever the filters say, its pixel sampler's
        // texel centre.
        SampleOp::Sample { .. }
            if plane > 0 && !integer && !matches!(stat, Some((SamplerRef::Constexpr(_), _))) =>
        {
            cx.families.sampler.metal_sampler = true;
            Ok(Some(Snap {
                kind: SnapKind::WordCell { axes: plane as u8 },
                axes: Vec::new(),
            }))
        }
        SampleOp::Sample { dref: None, .. } => {
            cx.families.sampler.metal_sampler = true;
            Ok(None)
        }
        _ => Ok(None),
    }
}

fn f32_const(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<SValue> {
    let ty = fb.module().ty(SType::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(x.to_bits())),
    })?;
    fb.module().const_value(c)
}

fn u32_const(fb: &mut FunctionBuilder<'_>, x: u32) -> BResult<SValue> {
    let c = fb.u32_const(x)?;
    fb.module().const_value(c)
}

/// The census words of [`offset_dropped`] and of a snapped axis's compensation by it: the
/// subtractions, the comparisons, the mask and their conjunction, the select, the subtraction from
/// the index, and their constants, in the leveled form (the larger).
pub const OFFSET_RULE_WORDS: u32 = 64;

/// Whether Metal drops a constant texel offset on a repeating or mirrored axis whose level size is
/// `n` (a `u32`), a `bool` (module docs): `n` lies in `[2, 7]` and the axis's base-level size
/// `base` (a `u32`; `None`: `n` is the base level's) is not a power of two, `n - 2 <= 5` unsigned
/// and `base & (base - 1) != 0`. At the base level that is `n` of 3, 5, 6, or 7, `n - 3 <= 4`
/// unsigned and `n != 4`. Every other size, and every clamping axis, takes the offset. The
/// base-level size Metal reads is a view's parent's: where `parent` (a `bool`, the axis's
/// `texture::OffsetBase` test) holds, it is not a power of two though the bound view's is, so the
/// offset drops at every `n` in `[2, 7]`.
pub fn offset_dropped(
    fb: &mut FunctionBuilder<'_>,
    n: SValue,
    base: Option<SValue>,
    parent: Option<SValue>,
) -> BResult<SValue> {
    let (base, parent) = match (base, parent) {
        (None, None) => {
            let three = u32_const(fb, 3)?;
            let four = u32_const(fb, 4)?;
            let t = fb.binary(BinOp::ISub, n, three)?;
            let small = fb.binary(BinOp::ULessThanEqual, t, four)?;
            let odd_one = fb.binary(BinOp::INotEqual, n, four)?;
            return fb.binary(BinOp::LogicalAnd, small, odd_one);
        }
        (base, parent) => (base.unwrap_or(n), parent),
    };
    let two = u32_const(fb, 2)?;
    let five = u32_const(fb, 5)?;
    let t = fb.binary(BinOp::ISub, n, two)?;
    let small = fb.binary(BinOp::ULessThanEqual, t, five)?;
    let one = u32_const(fb, 1)?;
    let below = fb.binary(BinOp::ISub, base, one)?;
    let mask = fb.binary(BinOp::BitwiseAnd, base, below)?;
    let zero = u32_const(fb, 0)?;
    let uneven = fb.binary(BinOp::INotEqual, mask, zero)?;
    let uneven = match parent {
        Some(p) => fb.binary(BinOp::LogicalOr, uneven, p)?,
        None => uneven,
    };
    fb.binary(BinOp::LogicalAnd, small, uneven)
}

/// The census words a slot's `SpecKey::OffsetBaseAxes` test adds to [`offset_dropped`] on each
/// plane axis an offset moves, per footprint, snap or centre that places it: the general rule's
/// words where they replace the base level's shorter ones, the disjunction, and their constants,
/// and a share of the slot's tests' declaration (a specialization constant and three
/// `OpSpecConstantOp` tests per axis, `texture::OffsetBase`).
pub const PARENT_RULE_WORDS: u32 = 64;

/// Metal's position `X` (an `f32`, module docs) of normalized coordinate `u` (an `f32`) on an axis
/// of size `n` (a `u32`, below `2^22`) addressed by `address`, with the size as an `f32`.
pub fn position(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    address: TapAddress,
) -> BResult<(SValue, SValue)> {
    let one = u32_const(fb, 1)?;
    let zero = u32_const(fb, 0)?;
    // The wrap into one period, in `f32`, its subtraction rounded up, or down at a power-of-two
    // size.
    let u = match address {
        TapAddress::Edge | TapAddress::Border => u,
        TapAddress::Repeat => {
            let fl = fb.glsl(GlslOp::Floor, &[u])?;
            wrap(fb, u, fl, n, (one, zero))?
        }
        TapAddress::Mirror => {
            let twice = mirror_start(fb, u)?;
            wrap(fb, u, twice, n, (one, zero))?
        }
    };
    scale(fb, u, n, (one, zero))
}

/// The census words [`position_by_mode`] adds to [`position`]'s: both periods' starts, the
/// mode's two tests and selects, and their constants.
pub const MODE_POSITION_WORDS: u32 = 64;

/// [`position`] on an axis whose address mode is `mode`, a `u32` holding `MTLSamplerAddressMode`'s
/// value (a runtime sampler's, read from its `SamplerWord` in the function): the period starts at
/// `floor(u)` under `Repeat` and at `2 floor(u / 2)` under `MirrorRepeat`, and at 0 under every
/// other mode, where the wrap is `u` itself (`u - 0` and its sum with 0 are `u`, so the rounding's
/// step is 0) and the product scales `u`, as a clamping axis's does.
pub fn position_by_mode(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    mode: SValue,
) -> BResult<(SValue, SValue)> {
    let one = u32_const(fb, 1)?;
    let zero = u32_const(fb, 0)?;
    let fl = fb.glsl(GlslOp::Floor, &[u])?;
    let twice = mirror_start(fb, u)?;
    let none = f32_const(fb, 0.0)?;
    let repeat = u32_const(fb, SamplerAddressMode::Repeat.as_raw())?;
    let mirror = u32_const(fb, SamplerAddressMode::MirrorRepeat.as_raw())?;
    let repeats = fb.binary(BinOp::IEqual, mode, repeat)?;
    let mirrors = fb.binary(BinOp::IEqual, mode, mirror)?;
    let k = fb.select(mirrors, twice, none)?;
    let k = fb.select(repeats, fl, k)?;
    let u = wrap(fb, u, k, n, (one, zero))?;
    scale(fb, u, n, (one, zero))
}

/// The start `2 floor(u / 2)` of a mirrored axis's period of two around `u` (an `f32`).
fn mirror_start(fb: &mut FunctionBuilder<'_>, u: SValue) -> BResult<SValue> {
    let half = f32_const(fb, 0.5)?;
    let h = fb.binary(BinOp::FMul, u, half)?;
    let fl = fb.glsl(GlslOp::Floor, &[h])?;
    fb.binary(BinOp::FAdd, fl, fl)
}

/// Metal's position `X` of the wrapped coordinate `u` (an `f32`) on an axis of size `n` (a `u32`):
/// the product rounded with ties away from zero (module docs), with the size as an `f32`;
/// `(one, zero)` are `u32`s.
fn scale(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    n: SValue,
    (one, zero): (SValue, SValue),
) -> BResult<(SValue, SValue)> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let w = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f)?;
    // The product rounded to nearest even, on its own (in the profile's form, never fused into what
    // follows).
    let p = fb.binary_no_contraction(BinOp::FMul, u, nf)?;
    // A tie rounded toward zero: bits `0..=s` of `bits(u) * n` are `0100…0` (module docs).
    let bu = fb.convert(ConvOp::Bitcast, u, w)?;
    let bp = fb.convert(ConvOp::Bitcast, p, w)?;
    let lo = fb.binary(BinOp::IMul, bu, n)?;
    let k23 = u32_const(fb, 23)?;
    let eu = fb.binary(BinOp::ShiftRightLogical, bu, k23)?;
    let ep = fb.binary(BinOp::ShiftRightLogical, bp, k23)?;
    let s = fb.binary(BinOp::ISub, ep, eu)?;
    let k31 = u32_const(fb, 31)?;
    let sh = fb.binary(BinOp::ISub, k31, s)?;
    let top = fb.binary(BinOp::ShiftLeftLogical, lo, sh)?;
    let tie_bits = u32_const(fb, 0x4000_0000)?;
    let tie = fb.binary(BinOp::IEqual, top, tie_bits)?;
    let up = fb.select(tie, one, zero)?;
    let bx = fb.binary(BinOp::IAdd, bp, up)?;
    Ok((fb.convert(ConvOp::Bitcast, bx, f)?, nf))
}

/// `u - k` rounded toward positive infinity, or toward negative infinity where the size `n` (a
/// `u32`) is a power of two, for an `f32` coordinate `u` and the integer `k` (an `f32`) of its
/// period's start (`floor(u)`, or `2 floor(u / 2)` mirrored), with `(1, 0)` as `u32`s (module
/// docs). The difference rounded to nearest even, `s`, is inexact only where `-1 < u < 0` (`k` is
/// `-1`, or `-2` mirrored, and `s` lies in `[1/2, 1]`, or `[1, 2]`), and there `z = s + k` is
/// exact by Sterbenz's lemma; elsewhere `s` is exact and `z` is `u`. So the exact difference
/// `u - k` exceeds `s` exactly when `u > z`, and then the next `f32` up from `s` (which is not
/// negative) is its bits plus one; it lies below `s` exactly when `u < z`, and then the next `f32`
/// down is its bits minus one. The three float operations are `NoContraction`: a device rounds each
/// on its own and never folds `(u - k) + k` to `u`.
fn wrap(
    fb: &mut FunctionBuilder<'_>,
    u: SValue,
    k: SValue,
    n: SValue,
    (one, zero): (SValue, SValue),
) -> BResult<SValue> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let w = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let s = fb.binary_no_contraction(BinOp::FSub, u, k)?;
    let z = fb.binary_no_contraction(BinOp::FAdd, s, k)?;
    let below = fb.binary(BinOp::FOrdLessThan, z, u)?;
    let above = fb.binary(BinOp::FOrdLessThan, u, z)?;
    let up = fb.select(below, one, zero)?;
    let minus_one = u32_const(fb, u32::MAX)?;
    let down = fb.select(above, minus_one, zero)?;
    // A power of two, `n & (n - 1) == 0`, wraps exactly (module docs).
    let m = fb.binary(BinOp::ISub, n, one)?;
    let m = fb.binary(BinOp::BitwiseAnd, n, m)?;
    let power_of_two = fb.binary(BinOp::IEqual, m, zero)?;
    let step = fb.select(power_of_two, down, up)?;
    let bs = fb.convert(ConvOp::Bitcast, s, w)?;
    let bs = fb.binary(BinOp::IAdd, bs, step)?;
    fb.convert(ConvOp::Bitcast, bs, f)
}

/// A gather's base texel `i0` (an `f32`) at position `x`: `floor(floor(256 x + 1/2) / 256 - 1/2)`,
/// the `q >> 8`, exact for `|x| < 2^15` (module docs).
fn gather_base(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let k256 = f32_const(fb, 256.0)?;
    let y = fb.binary(BinOp::FMul, x, k256)?;
    let half = f32_const(fb, 0.5)?;
    let y = fb.binary(BinOp::FAdd, y, half)?;
    let z = fb.glsl(GlslOp::Floor, &[y])?;
    let step = f32_const(fb, 1.0 / 256.0)?;
    let z = fb.binary(BinOp::FMul, z, step)?;
    let z = fb.binary(BinOp::FSub, z, half)?;
    fb.glsl(GlslOp::Floor, &[z])
}

/// Component `i` of `v`, or `v` itself for a one-lane value.
fn lane(fb: &mut FunctionBuilder<'_>, v: SValue, lanes: u8, i: u32) -> BResult<SValue> {
    if lanes == 1 {
        Ok(v)
    } else {
        fb.composite_extract(v, &[i])
    }
}

/// The snapped axis coordinate (an `f32`) of coordinate `u` (in texels, Metal's position as
/// given) on an axis of size `n` (a `u32`) addressed by `address`,
/// for `kind` (module docs), whose access the device moves by the constant texel offset `offset`
/// (an `f32` holding the axis's offset; `None`: none): on a repeating or mirrored axis whose size
/// drops it ([`offset_dropped`], whose `parent` is the axis's `texture::OffsetBase` test), the
/// index is moved back by it first.
pub fn snapped_axis(
    fb: &mut FunctionBuilder<'_>,
    kind: AxisSnap,
    x: SValue,
    n: SValue,
    (address, offset, parent): (TapAddress, Option<SValue>, Option<SValue>),
) -> BResult<SValue> {
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let nf = fb.convert(ConvOp::ConvertUToF, n, f)?;
    let at = match kind {
        AxisSnap::Gather => {
            let i0 = gather_base(fb, x)?;
            let i0 = match address {
                TapAddress::Edge | TapAddress::Border => {
                    let lo = f32_const(fb, -10.0)?;
                    let eight = f32_const(fb, 8.0)?;
                    let hi = fb.binary(BinOp::FAdd, nf, eight)?;
                    fb.glsl(GlslOp::FClamp, &[i0, lo, hi])?
                }
                TapAddress::Repeat | TapAddress::Mirror => i0,
            };
            let one = f32_const(fb, 1.0)?;
            fb.binary(BinOp::FAdd, i0, one)?
        }
        // `q + 128 = floor(256 X) + [frac(256 X) >= 1/2]`, as `bicubic::quantize` reads `q`:
        // `256 X`, its floor, and its fraction are exact, where `256 X - 127.5` can round (a
        // negative `X` whose sum carries into the next binade). The centre `(q + 128) / 256` is
        // exact below `|X| = 2^15`.
        AxisSnap::Cell => {
            let k256 = f32_const(fb, 256.0)?;
            let y = fb.binary(BinOp::FMul, x, k256)?;
            let fl = fb.glsl(GlslOp::Floor, &[y])?;
            let r = fb.binary(BinOp::FSub, y, fl)?;
            let half = f32_const(fb, 0.5)?;
            let up = fb.binary(BinOp::FOrdGreaterThanEqual, r, half)?;
            let one = f32_const(fb, 1.0)?;
            let zero = f32_const(fb, 0.0)?;
            let up = fb.select(up, one, zero)?;
            let c = fb.binary(BinOp::FAdd, fl, up)?;
            let step = f32_const(fb, 1.0 / 256.0)?;
            fb.binary(BinOp::FMul, c, step)?
        }
    };
    let at = match (address, offset) {
        (TapAddress::Repeat | TapAddress::Mirror, Some(o)) => {
            let dropped = offset_dropped(fb, n, None, parent)?;
            let none = f32_const(fb, 0.0)?;
            let back = fb.select(dropped, o, none)?;
            fb.binary(BinOp::FSub, at, back)?
        }
        _ => at,
    };
    fb.binary(BinOp::FDiv, at, nf)
}

/// The coordinate `coord` (`lanes` floats, in texels) of a sample or gather of sampled image `si` (image facts `image`) snapped by `snap` (module docs) on
/// the base level's size: a normalized coordinate. `offsets` holds, per plane axis, the constant
/// texel offset the device adds to the access (an `f32`, or `None`; [`snapped_axis`]), and
/// `parent` the slot's `SpecKey::OffsetBaseAxes` tests (`texture::OffsetBase`).
pub fn snapped(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    snap: &Snap,
    (coord, offsets, parent): (SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (si, image): (SValue, CombinedImage),
) -> BResult<SValue> {
    let img = fb.image(si)?;
    let zero = u32_const(fb, 0)?;
    let size = fb.query(ImageQuery::SizeLod(zero), img)?;
    snapped_on(cx, fb, snap, (coord, offsets, parent), lanes, (size, image))
}

/// The coordinate `coord` (`lanes` floats) snapped by `snap` (module docs) on the base level's
/// size `size` of an image with facts `image`: a plane coordinate's axes. `offsets` holds each
/// plane axis's constant texel offset the device adds ([`snapped_axis`]), and `parent` the slot's
/// `SpecKey::OffsetBaseAxes` tests (`texture::OffsetBase`).
pub fn snapped_on(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    snap: &Snap,
    (coord, offsets, parent): (SValue, &[Option<SValue>], Option<OffsetBase>),
    lanes: u8,
    (size, image): (SValue, CombinedImage),
) -> BResult<SValue> {
    let size_lanes = image.plane + u8::from(image.arrayed);
    let kind = match snap.kind {
        SnapKind::Axes(kind) => kind,
        SnapKind::WordGather | SnapKind::WordCell { .. } => {
            return Err(cx.fail("a word snap without its sampler's word"))
        }
    };
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for (a, axis) in (0u32..).zip(&snap.axes).take(usize::from(lanes)) {
        let u = lane(fb, coord, lanes, a)?;
        parts.push(match axis {
            Some(address) => {
                let n = lane(fb, size, size_lanes, a)?;
                let offset = offsets.get(a as usize).copied().flatten();
                let parent = OffsetBase::axis(parent, a);
                snapped_axis(fb, kind, u, n, (*address, offset, parent))?
            }
            None => u,
        });
    }
    if let [x] = parts[..] {
        return Ok(x);
    }
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector {
        elem: f,
        len: lanes,
    })?;
    fb.composite_construct(ty, &parts)
}

/// The plane coordinate of `parts` (one `f32` per plane axis).
fn composite(fb: &mut FunctionBuilder<'_>, parts: Vec<SValue>) -> BResult<SValue> {
    if let [x] = parts[..] {
        return Ok(x);
    }
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let ty = fb.module().ty(SType::Vector {
        elem: f,
        len: parts.len() as u8,
    })?;
    fb.composite_construct(ty, &parts)
}

/// Whether sampler word `word` (a `u32`) masked by `mask` holds every bit of `mask`, a `bool`.
fn word_holds(fb: &mut FunctionBuilder<'_>, word: SValue, mask: u32) -> BResult<SValue> {
    let m = u32_const(fb, mask)?;
    let have = fb.binary(BinOp::BitwiseAnd, word, m)?;
    fb.binary(BinOp::IEqual, have, m)
}

/// The coordinate of a runtime or argument buffer's sampler's 2D gather, its sampler's word
/// `word`, snapped by [`SnapKind::WordGather`] on the base level's size `size` (`size_lanes`
/// components): where the word states pixel coordinates, per plane axis the centre `(i0 + 1) / N`
/// of the 2x2 block Metal reads ([`AxisSnap::Gather`]), its base texel `i0` at Metal's position,
/// the coordinate as given (`given`, `lanes` floats); for a normalized word the coordinate the
/// device reads otherwise (`scaled`, `lanes` floats). `i0` is clamped into `[-2N - 10, 2N + 8]`,
/// past which, with any offset in Metal's gather range `[-8, 7]`, both texels of a clamping or
/// mirror-clamped axis read the same edge texel or border, as the exact `i0`'s do. The device's
/// sampler places an offset of the normalized centre (Metal's own drops it where it drops the
/// pixel coordinate's), so no offset moves it.
pub fn gathered_by_word(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    (given, scaled): (SValue, SValue),
    lanes: u8,
    (size, size_lanes): (SValue, u8),
) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let pixel = word_holds(fb, word, SamplerWord::PIXEL_COORDINATES)?;
    let f = fb.module().ty(SType::Float { width: 32 })?;
    let lo = f32_const(fb, -10.0)?;
    let eight = f32_const(fb, 8.0)?;
    let one = f32_const(fb, 1.0)?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for a in 0..u32::from(lanes) {
        let x = lane(fb, given, lanes, a)?;
        let device = lane(fb, scaled, lanes, a)?;
        let n = lane(fb, size, size_lanes, a)?;
        let nf = fb.convert(ConvOp::ConvertUToF, n, f)?;
        let i0 = gather_base(fb, x)?;
        let twice = fb.binary(BinOp::FAdd, nf, nf)?;
        let hi = fb.binary(BinOp::FAdd, twice, eight)?;
        let low = fb.binary(BinOp::FSub, lo, twice)?;
        let i0 = fb.glsl(GlslOp::FClamp, &[i0, low, hi])?;
        let at = fb.binary(BinOp::FAdd, i0, one)?;
        let centre = fb.binary(BinOp::FDiv, at, nf)?;
        parts.push(fb.select(pixel, centre, device)?);
    }
    composite(fb, parts)
}

/// The coordinate of a runtime or argument buffer's sampler's sample, its sampler's word `word`,
/// snapped by [`SnapKind::WordCell`] on the
/// base level's size `size` (`size_lanes` components): where the word states pixel coordinates and
/// linear min and mag filters, each plane axis of the coordinate as given (`given`, `lanes` floats)
/// onto the centre of its footprint step ([`AxisSnap::Cell`]); for every other word the coordinate
/// the device reads otherwise (`scaled`). A pixel sampler clamps every axis or mirrors it once and
/// clamps it (`runtime::sampler`'s rule), and the device's sampler places a texel offset of the
/// normalized cell (Metal's own as Metal places the pixel coordinate's: kept on a clamping axis,
/// dropped by a mirror-clamped one's sizes alike), so no offset moves the cell.
pub fn celled_by_word(
    fb: &mut FunctionBuilder<'_>,
    word: SValue,
    (given, scaled): (SValue, SValue),
    lanes: u8,
    (size, size_lanes): (SValue, u8),
) -> BResult<SValue> {
    use crate::metal::sampler::SamplerWord;
    let reach = word_holds(
        fb,
        word,
        SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR,
    )?;
    let mut parts = Vec::with_capacity(usize::from(lanes));
    for a in 0..u32::from(lanes) {
        let x = lane(fb, given, lanes, a)?;
        let device = lane(fb, scaled, lanes, a)?;
        let n = lane(fb, size, size_lanes, a)?;
        let cell = snapped_axis(fb, AxisSnap::Cell, x, n, (TapAddress::Edge, None, None))?;
        parts.push(fb.select(reach, cell, device)?);
    }
    composite(fb, parts)
}

#[cfg(test)]
mod tests;
