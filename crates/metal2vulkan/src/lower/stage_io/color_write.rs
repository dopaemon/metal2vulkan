//! A fragment function's color outputs as the device converts them: which conversions each
//! output's channels take, from `LoweringInputs::color_writes`, and the rounding Metal's
//! `textureWriteRoundingMode` asks for (`arith::color_write`).
//!
//! The function records the map it compiled under for each attachment it writes
//! (`StageIo::color_writes`, which link checks against the pipeline's), and writes each lane as
//! computed, the device's conversion storing it, but for a half-float channel under
//! `TextureWriteRoundingMode::NearestEven` (`color_writes::HALF_NEAREST`), which no device's
//! conversion of Metal's default does: the function rounds it to nearest even on every profile, and
//! the device's conversion of the rounded half is exact. A lane the device converts otherwise than
//! Metal may records a note:
//!
//! - a channel whose conversion the device makes where it is not measured to make Metal's (a
//!   blend other than one and zero, alpha-to-coverage's alpha, a format Metal was not measured
//!   with, or a unorm, snorm, sRGB or small float channel on a device that rounds it otherwise)
//!   records `PrecisionNote::DeviceColorConversion`; one whose blend Metal rounds to nearest even
//!   ([`ChannelWrite::BlendedNearestEven`], a half-float channel under
//!   `TextureWriteRoundingMode::NearestEven`) records `PrecisionNote::BlendedWriteRounding`;
//! - a `half` output (held widened) to a half-float channel is already a half: the device's
//!   conversion of it is exact;
//! - a `bfloat` output never arrives: Metal's compiler rejects one, and the stage's outputs refuse
//!   it before (`legal::refuse_bfloat_color`);
//! - an integer output to a channel with a rule is Metal's pipeline error, which link refuses
//!   (`ColorAttachment`): nothing converts it;
//! - a NaN written to a half-float channel is stored as the device's conversion encodes it
//!   (`PrecisionNote::DeviceNanEncoding`) unless the device is measured to store Metal's: Metal
//!   stores a `float` NaN as `0x7e00` and a `half` NaN's own bits. A `float` output's NaN is
//!   Metal's where the profile takes `DriverDeviation::MetalColorWrites`, and a lane's the function
//!   rounds to nearest even (its NaN `0x7fc00000`) where it takes `MetalFloatColorNans`; a `half`
//!   output's (held widened, its payload in the significand's top bits) where it takes
//!   `MetalHalfColorNans`, or where the function writes it through a 16-bit output (below);
//!   elsewhere the module records the note (`runtime_lane::color_writes`: ANV and RADV store
//!   `0xfe00`, from a 16-bit output too, lavapipe quiets or canonicalizes a `half` NaN, MoltenVK
//!   canonicalizes it from a 32-bit output).
//!
//! On a profile that stores Metal's codes itself (`VkProperty::MetalColorWrites`) the map holds
//! only half-float channels (`pipeline::color_writes`): only the NaN note is recorded, and the
//! rounding to nearest even under `NearestEven`.
//!
//! Where the device is measured to convert one kind of channel as Metal does, a lane of an output
//! variable to such a channel records no `DeviceColorConversion`, and the module requires the
//! deviation's property:
//! - a `float` output to a half-float or unsigned small float channel
//!   (`DriverDeviation::MetalSmallFloatColorWrites`: RADV on the Radeon 780M, ANV on the Iris Xe
//!   of RPL-P, NVIDIA's on the RTX 3090), which stores every value but a NaN as Metal does: the
//!   NaN, as computed, is the device's (`PrecisionNote::DeviceNanEncoding`);
//! - an output to a 16-bit unorm or snorm channel but a blended snorm one, whose NaN Metal stores
//!   as -1.0 (`DriverDeviation::MetalNorm16ColorWrites`: RADV on the Radeon 780M).
//!
//! An attachment no lane of which rounds is not listed as rounded: it keeps its Vulkan blend of
//! one and zero, through which the facts were measured too. A write in the storage form, or
//! through a 16-bit output, records the device's conversion: the facts are of the device's
//! conversion of an output variable. A storage write the shader blends into a 16-bit float
//! attachment gives every NaN as `0x7fc00000`, which the storage image's store converts: Metal's
//! `0x7e00` where the profile takes `DriverDeviation::MetalFloatStorageNans` (the module requires
//! its property), else the device's NaN, noted `PrecisionNote::DeviceNanEncoding` (ANV stores
//! `0xfe00`).
//!
//! Where the device does not store the output's NaN as Metal does, but is measured to store a
//! 16-bit output's bits unchanged (`DriverDeviation::HalfColorOutputBits`: MoltenVK 1.4.1 on the
//! M3 Max and the M4), its conversion to 16 bits is Metal's `half(float)`
//! (`DriverDeviation::MetalHalfConversion`), and the profile declares 16-bit outputs
//! (`storageInputOutput16`, with `shaderInt16` and `shaderFloat16`), an output to an attachment
//! whose every channel written is half-float is written as the bits of its half, through a 16-bit
//! output variable ([`half_output`]): each lane converted to 16 bits by that conversion, a NaN as
//! its own payload's code by integer operations (`arith::color_write::half_bits`), so no
//! conversion of the device's chooses the NaN, the NaN note is not recorded, and the module
//! requires `VkProperty::HalfColorOutputBits` and `VkProperty::MetalHalfConversion`. An attachment
//! a lane of which rounds is listed as rounded (`rounded_colors`), and one whose output is written
//! so but rounds nothing as written as a half's bits (`half_bit_colors`), its own notice; either
//! takes no Vulkan blend of one and zero. On a device not measured so the output stays 32-bit: ANV
//! and RADV store every NaN of a 16-bit output as `0xfe00` too, so the route would change nothing
//! there.

use crate::air::abi::{AbiInt, OutputRole};
use crate::device::requirements::VkProperty;
use crate::error::DriverDeviation;
use crate::lower::arith::color_write;
use crate::lower::arith::{FloatFmt, Form};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::color_writes;
use crate::pipeline::lower_inputs::ChannelWrite;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use std::collections::BTreeSet;

/// Plan the write of a fragment function's output of form `f` to color attachment `color`
/// (dual-source index 0), through an output variable or in the storage form (see the module
/// docs).
pub(crate) fn plan(cx: &mut SelectCx<'_>, color: u32, f: Form) {
    if cx.stage != Stage::Fragment {
        return;
    }
    let Some(rules) = cx.inputs.color_writes.get(&color).copied() else {
        return;
    };
    cx.families
        .stage_io
        .interface
        .color_writes
        .insert(color, rules);
    let metal = cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalColorWrites);
    let half = half_output(cx, color, f);
    let storage = crate::lower::fragment::writes_through_storage(cx, color);
    // A storage write the shader blends into a 16-bit float attachment stores the blend's `half`
    // in every channel, which the device converts exactly, but for a NaN: no conversion of an
    // output variable's bears on it, and its NaN is the storage image's store's ([`storage_nan`]).
    // One the shader blends into an 8-bit unorm attachment stores the code Metal's unorm rule
    // gives the blend's result, which the shader rounds itself (`fetch::unorm8_blended`).
    let shader_blended = crate::lower::fragment::blends_in_storage(cx, color);
    let sixteen_blended = crate::lower::fragment::blends_sixteen_in_storage(cx, color);
    let mut lanes = [None; 4];
    let mut device = false;
    let mut blended = false;
    let mut nans = false;
    let mut relied = BTreeSet::new();
    for (slot, rule) in lanes.iter_mut().zip(rules).take(usize::from(f.lanes)) {
        let Some(rule) = rule else { continue };
        nans |= !shader_blended && color_writes::half_float(rule) && !nan_kept(cx, rule, f, half);
        if metal && rule == color_writes::HALF {
            continue;
        }
        match (rule, f.float_fmt()) {
            (_, None | Some(FloatFmt::F64 | FloatFmt::BFloat)) => {}
            // A `half` is a half already, whichever way the channel rounds.
            (
                ChannelWrite::FloatTowardZero {
                    mantissa: 10..,
                    signed: true,
                }
                | ChannelWrite::FloatNearestEven {
                    mantissa: 10..,
                    signed: true,
                },
                Some(FloatFmt::Half),
            ) => {}
            // The pipeline's rounding mode, which the function applies.
            (rule @ ChannelWrite::FloatNearestEven { .. }, Some(_)) => *slot = Some(rule),
            (ChannelWrite::BlendedNearestEven { .. }, _) => blended |= !shader_blended,
            (ChannelWrite::Device, _) => device |= !shader_blended,
            (rule, Some(fmt)) => match conversion(cx, rule, fmt, half || storage, &mut relied) {
                Conversion::Metal => {}
                Conversion::MetalButNan => nans |= !shader_blended,
                Conversion::Device => device |= !shader_blended,
            },
        }
    }
    if device {
        cx.precision.insert(PrecisionNote::DeviceColorConversion);
    }
    if blended {
        cx.precision.insert(PrecisionNote::BlendedWriteRounding);
    }
    if sixteen_blended {
        nans |= !storage_nan(cx, &mut relied);
    }
    if nans {
        cx.precision.insert(PrecisionNote::DeviceNanEncoding);
    }
    if half {
        cx.families.stage_io.halves.insert(color);
        relied.extend([
            VkProperty::HalfColorOutputBits,
            VkProperty::MetalHalfConversion,
        ]);
    }
    cx.families.stage_io.color_properties.extend(relied);
    let rounds = lanes.iter().any(Option::is_some);
    let sp = &mut cx.families.stage_io;
    if rounds {
        sp.rounded.insert(color, lanes);
    }
    let listed = match (rounds, half) {
        (true, _) => &mut sp.interface.rounded_colors,
        (false, true) => &mut sp.interface.half_bit_colors,
        (false, false) => return,
    };
    if let Err(at) = listed.binary_search(&color) {
        listed.insert(at, color);
    }
}

/// Whether a storage write the shader blends into a 16-bit float attachment stores its NaN as
/// Metal does: the blend gives every NaN as `0x7fc00000` (`fetch::half_blended`; Metal stores
/// every NaN of that blend as `0x7e00`), which the storage image's store converts to Metal's
/// `0x7e00` where the profile takes `DriverDeviation::MetalFloatStorageNans`, whose property the
/// module then relies on (`relied`). Elsewhere the NaN stored is the device's (ANV stores every NaN
/// as `0xfe00`), which the module notes (`PrecisionNote::DeviceNanEncoding`): no value written to a
/// 16-bit float storage image stores another there.
fn storage_nan(cx: &SelectCx<'_>, relied: &mut BTreeSet<VkProperty>) -> bool {
    let metal = cx
        .profile
        .takes_driver_deviation(DriverDeviation::MetalFloatStorageNans);
    if metal {
        relied.insert(VkProperty::MetalFloatStorageNans);
    }
    metal
}

/// How the device converts a lane of an output of `fmt` (`F32` or `Half`) to a channel of rule
/// `rule` (a rule of Metal's, not the device's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Conversion {
    /// As Metal stores it.
    Metal,
    /// As Metal stores it but for a NaN, which the device encodes.
    MetalButNan,
    /// By the device's own rule, a near-equivalent (`PrecisionNote::DeviceColorConversion`).
    Device,
}

/// How the device converts a lane of an output of `fmt` to a channel of rule `rule` (`own`: the
/// function writes the output otherwise than as a value of its output variable the device
/// converts, as a half's bits through a 16-bit output, [`half_output`], or as a texel of a storage
/// image, which no fact measures): as Metal does where the profile takes the deviation measuring
/// it, whose property the module then relies on (`relied`), else the device's own:
///
/// - a `float` output to a half-float or unsigned small float channel where the profile takes
///   `DriverDeviation::MetalSmallFloatColorWrites`: Metal's but for a NaN;
/// - an output to a 16-bit unorm or snorm channel where the profile takes
///   `DriverDeviation::MetalNorm16ColorWrites`: Metal's, but for a blended snorm channel, whose NaN
///   Metal stores as -1.0.
fn conversion(
    cx: &SelectCx<'_>,
    rule: ChannelWrite,
    fmt: FloatFmt,
    own: bool,
    relied: &mut BTreeSet<VkProperty>,
) -> Conversion {
    let takes = |d| !own && cx.profile.takes_driver_deviation(d);
    match rule {
        ChannelWrite::FloatTowardZero { .. }
            if fmt == FloatFmt::F32 && takes(DriverDeviation::MetalSmallFloatColorWrites) =>
        {
            relied.insert(VkProperty::MetalSmallFloatColorWrites);
            Conversion::MetalButNan
        }
        ChannelWrite::Unorm { bits: 16 }
        | ChannelWrite::Snorm {
            bits: 16,
            blended: false,
        } if takes(DriverDeviation::MetalNorm16ColorWrites) => {
            relied.insert(VkProperty::MetalNorm16ColorWrites);
            Conversion::Metal
        }
        _ => Conversion::Device,
    }
}

/// Whether the device stores the NaN a fragment function writes to a half-float channel of rule
/// `rule` from its output of form `f` as Metal does (the module docs; `half`: written as a half's
/// bits, [`half_output`]): a `float` output's where the profile takes
/// `DriverDeviation::MetalColorWrites`, or where the function rounds the lane
/// (`color_writes::HALF_NEAREST`), whose NaN `0x7fc00000` it then stores as Metal's `0x7e00`
/// under `MetalFloatColorNans` (the output's own NaN, unrounded, it is not measured to); a `half`
/// output's where it takes `MetalHalfColorNans`, or written as a half's bits. Another output stores
/// no NaN of its own there.
fn nan_kept(cx: &SelectCx<'_>, rule: ChannelWrite, f: Form, half: bool) -> bool {
    let takes = |d| cx.profile.takes_driver_deviation(d);
    match f.float_fmt() {
        Some(FloatFmt::F32) => {
            takes(DriverDeviation::MetalColorWrites)
                || (rule == color_writes::HALF_NEAREST
                    && takes(DriverDeviation::MetalFloatColorNans))
        }
        Some(FloatFmt::Half) => half || takes(DriverDeviation::MetalHalfColorNans),
        _ => true,
    }
}

/// Whether the device stores the NaN of a fragment function's output of form `f` to a
/// half-float channel as Metal does, for [`half_output`]: a `float` output's where the profile
/// takes `DriverDeviation::MetalFloatColorNans` or `MetalColorWrites`, a `half` output's where it
/// takes `MetalHalfColorNans`. Another output stores no NaN of its own there.
fn metal_nans(cx: &SelectCx<'_>, f: Form) -> bool {
    let takes = |d| cx.profile.takes_driver_deviation(d);
    match f.float_fmt() {
        Some(FloatFmt::F32) => {
            takes(DriverDeviation::MetalColorWrites) || takes(DriverDeviation::MetalFloatColorNans)
        }
        Some(FloatFmt::Half) => takes(DriverDeviation::MetalHalfColorNans),
        _ => true,
    }
}

/// Whether a fragment function writes its output of form `f` (dual-source index 0) to color
/// attachment `color` as the bits of the half Metal stores, through a 16-bit output variable
/// (the module docs): the output is `float` or `half` and the device does not store its NaN as
/// Metal does, but stores a 16-bit output's bits unchanged
/// (`DriverDeviation::HalfColorOutputBits`); its conversion to 16 bits is Metal's `half(float)`
/// (`narrow::QuantizeForm::Converted`: `DriverDeviation::MetalHalfConversion` and `shaderFloat16`,
/// not native half rounding's conversion); every channel of the attachment it writes is half-float
/// (`color_writes::HALF`), none the device's; the profile declares 16-bit outputs
/// (`storageInputOutput16`) and the `Int16` arithmetic that builds a NaN's code (a channel that
/// rounds to nearest even, `color_writes::HALF_NEAREST`, is half-float too); and the
/// function writes the attachment through an output variable, with no dual-source second
/// output beside it.
pub(super) fn half_output(cx: &SelectCx<'_>, color: u32, f: Form) -> bool {
    let Some(rules) = cx.inputs.color_writes.get(&color) else {
        return false;
    };
    let leaf = cx.profile.fields().leaf;
    let written = || rules.iter().take(usize::from(f.lanes));
    let dual = cx.entry.ep.outputs.iter().any(|o| {
        matches!(
            o.role,
            OutputRole::RenderTarget {
                attachment: AbiInt::Const(a),
                index: 1,
                ..
            } if a == color
        )
    });
    cx.stage == Stage::Fragment
        && matches!(f.float_fmt(), Some(FloatFmt::F32 | FloatFmt::Half))
        && !metal_nans(cx, f)
        && cx
            .profile
            .takes_driver_deviation(DriverDeviation::HalfColorOutputBits)
        && crate::lower::arith::narrow::quantize_form(cx.profile)
            == crate::spirv::builder::QuantizeForm::Converted
        // The measured conversion, not native half rounding's.
        && cx
            .profile
            .takes_driver_deviation(DriverDeviation::MetalHalfConversion)
        && leaf.input_output_16bit
        && leaf.int16
        && written().any(Option::is_some)
        && written().flatten().all(|&r| color_writes::half_float(r))
        && !dual
        && !crate::lower::fragment::writes_through_storage(cx, color)
}

/// The words of the planned roundings, each emitted once per return.
pub(super) fn words(rounded: impl IntoIterator<Item = [Option<ChannelWrite>; 4]>) -> u64 {
    rounded.into_iter().map(|r| color_write::words(&r)).sum()
}

/// Output `x` (form `f`) to color attachment `color`, each lane the plan rounds rounded to nearest
/// even (`color_writes::HALF_NEAREST`), the others as computed.
pub(crate) fn round(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    color: u32,
    f: Form,
    x: Value,
) -> BResult<Value> {
    let Some(rules) = cx.plan.families.stage_io.rounded.get(&color).copied() else {
        return Ok(x);
    };
    color_write::round(fb, f, x, &rules)
}
