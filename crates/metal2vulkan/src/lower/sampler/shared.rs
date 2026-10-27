//! The shared emulation of a switch-at-use (SEMANTICS 1012).
//!
//! A switch-at-use whose every arm takes the same emulated sampling on its own texture slot
//! ([`Sampling::shared_shape`]): a bicubic sample, or a snapped gather or sample. Emitted per arm, each arm repeats the whole
//! emulation (its position, weights, sums, and routine calls), so the switch's words grow with arms × emulation, and so does
//! the driver's work once the bound formats make every arm's emulation live (an Intel ANV pipeline
//! of 32 such arms took 6 s to create).
//!
//! **The shared form** runs the emulation once, outside the dispatch, on the named candidate's
//! facts, and dispatches only what reads a candidate's own image:
//!
//! - a facts dispatch, whose arms yield their level-0 size (`OpImageQuerySizeLod` 0), the bound
//!   format's bits (the slot's `BicubicTexels` specialization) and `true`, and
//!   whose null arm yields a size of ones, bits 0 and `false`;
//! - the emulation on those facts (its footprint, weights and routine calls once), whose taps a
//!   dispatch reads from each arm's image (the null arm's are zero);
//! - a snapped sample's coordinate snapped once on the named size, and the device's access and
//!   texel conversion in each arm, the null arm yielding the switch's null value.
//!
//! Where a handle names a candidate the facts are that candidate's, so every value is the one its
//! own arm computed; elsewhere the switch's null value is selected. The emulation of a null handle
//! runs on a size of ones and zero taps, whose value is discarded.

use crate::lower::emit::EmitCx;
use crate::lower::texture::{dispatch, with_word, NullLeaf};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{ConstId, Value as SValue};
use crate::spirv::ir::{ConstBits, Constant, ImageOperands, ImageQuery, Type as SType};

use super::emit::{
    layered, offset_operands, op_offset, pixel_coordinates, plane_and_gradients, sample_access,
    sample_operands, scaled, snap_offsets, texel_value,
};
use super::{Binding, CombinedImage, SampleOp, Sampling, Source};

/// The candidates of a switch-at-use that emulates once: the handles' values dispatched on,
/// outermost first, and each arm's case literals and sampling.
pub(crate) struct Candidates<'a> {
    pub(crate) sels: Vec<SValue>,
    pub(crate) arms: Vec<(&'a [u32], &'a Sampling)>,
}

impl Candidates<'_> {
    /// One dispatch on the handles: `read` yields each arm's values from its sampling in the arm,
    /// `null` the null arm's; the merged values, in order.
    pub(super) fn each(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        read: &dyn Fn(&mut EmitCx<'_>, &mut FunctionBuilder<'_>, &Sampling) -> BResult<Vec<SValue>>,
        null: &NullLeaf<'_>,
    ) -> BResult<Vec<SValue>> {
        let leaf =
            |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| read(cx, fb, a);
        dispatch(cx, fb, &self.sels, &self.arms, &leaf, null)
    }

    /// The sampled image of arm sampling `a`'s binding, made in the open block.
    pub(super) fn sampled_image(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        a: &Sampling,
    ) -> BResult<SValue> {
        binding(cx, a)?.0.sampled_image(fb)
    }

    /// The image of arm sampling `a`'s binding, loaded in the open block.
    pub(super) fn image(
        &self,
        cx: &mut EmitCx<'_>,
        fb: &mut FunctionBuilder<'_>,
        a: &Sampling,
    ) -> BResult<SValue> {
        binding(cx, a)?.0.image(fb)
    }
}

/// The combined binding sampling `a` reads, and its image facts.
fn binding(cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<(Binding, CombinedImage)> {
    let Source::Combined(key) = a.source else {
        return Err(cx.fail("a shared sampling of a heap element"));
    };
    match cx.decl.families.sampler.combined.get(&key) {
        Some(&b) => Ok(b),
        None => Err(cx.fail(format!("combined binding {key:?} is not declared"))),
    }
}

/// The emulation every arm of a shared switch takes.
#[derive(Clone, Copy)]
enum Emulation<'a> {
    Bicubic(&'a super::bicubic::Bicubic),
    Snap(&'a super::position::Snap),
}

impl<'a> Emulation<'a> {
    fn of(s: &'a Sampling) -> Option<Emulation<'a>> {
        match (&s.op, &s.snap) {
            (SampleOp::Bicubic(b), None) => Some(Emulation::Bicubic(b)),
            (SampleOp::Sample { .. } | SampleOp::Gather { .. }, Some(p)) if !p.by_word() => {
                Some(Emulation::Snap(p))
            }
            _ => None,
        }
    }

    /// The specialization holding the bound format's bits of arm sampling `a`'s slot, where the
    /// emulation reads them: a bicubic sample's `BicubicTexels`.
    fn bits(self, cx: &mut EmitCx<'_>, a: &Sampling) -> BResult<Option<ConstId>> {
        let spec = match (self, Emulation::of(a)) {
            (Emulation::Bicubic(_), Some(Emulation::Bicubic(b))) => {
                cx.decl.families.sampler.bicubic.get(&b.texture).copied()
            }
            (Emulation::Snap(_), Some(Emulation::Snap(_))) => return Ok(None),
            _ => return Err(cx.fail("a shared switch's arms of different emulations")),
        };
        match spec {
            Some(c) => Ok(Some(c)),
            None => Err(cx.fail("a shared switch's arm without its format specialization")),
        }
    }
}

/// The value of the shared switch over `c` (module docs): its sampling's value where a handle
/// names a candidate, else `null`'s; `grads` are the gradients its rule-2 prelude hoisted.
pub(crate) fn shared_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    grads: Option<(SValue, SValue)>,
    null: &NullLeaf<'_>,
) -> BResult<SValue> {
    let Some(&(_, first)) = c.arms.first() else {
        return Err(cx.fail("a shared switch without arms"));
    };
    let Some(kind) = Emulation::of(first) else {
        return Err(cx.fail("a shared switch without an emulation"));
    };
    let (_, image) = binding(cx, first)?;
    for &(_, a) in &c.arms {
        if binding(cx, a)?.1 != image {
            return Err(cx.fail("a shared switch's candidates of different images"));
        }
    }
    let facts = facts(cx, fb, c, kind, image)?;
    let (size, rest) = match &facts[..] {
        [size, rest @ ..] => (*size, rest),
        _ => return Err(cx.fail("a shared switch's facts without their sizes")),
    };
    let (bits, named) = match (kind, rest) {
        (Emulation::Snap(_), &[]) => (None, None),
        (Emulation::Bicubic(_), &[bits, named]) => (Some(bits), Some(named)),
        _ => return Err(cx.fail("a shared switch's facts without their values")),
    };
    let x = match (kind, bits) {
        (Emulation::Bicubic(b), Some(bits)) => {
            super::bicubic::emit_shared(cx, fb, first, (size, image), b, bits, c)?
        }
        (Emulation::Snap(p), _) => {
            return snapped(cx, fb, c, (first, p), grads, (size, image), null);
        }
        _ => return Err(cx.fail("a shared emulation without its format bits")),
    };
    let x = with_word(cx, fb, &first.result, x, None)?;
    let Some(named) = named else {
        return Err(cx.fail("a shared emulation without its named test"));
    };
    let [n] = null(cx, fb)?[..] else {
        return Err(cx.fail("a shared switch without its null value"));
    };
    fb.select(named, x, n)
}

/// The facts dispatch (module docs): each arm's level-0 size, and where `kind` reads them its
/// format's bits and `true`; the null arm's ones, 0, and `false`.
fn facts(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    kind: Emulation<'_>,
    image: CombinedImage,
) -> BResult<Vec<SValue>> {
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let bool_ = fb.module().ty(SType::Bool)?;
    let scalar = |fb: &mut FunctionBuilder<'_>, ty, bits: u64| {
        fb.module().constant(Constant::Scalar {
            ty,
            bits: ConstBits(bits),
        })
    };
    let one = scalar(fb, u32_, 1)?;
    let lanes = image.plane + u8::from(image.arrayed);
    let ones = if lanes == 1 {
        one
    } else {
        let ty = fb.module().ty(SType::Vector {
            elem: u32_,
            len: lanes,
        })?;
        let parts = vec![one; usize::from(lanes)].into_boxed_slice();
        fb.module().constant(Constant::Composite { ty, parts })?
    };
    let zero = scalar(fb, u32_, 0)?;
    let yes = scalar(fb, bool_, 1)?;
    let no = scalar(fb, bool_, 0)?;
    let with_bits = !matches!(kind, Emulation::Snap(_));
    let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
        let img = c.image(cx, fb, a)?;
        let level = fb.module().const_value(zero)?;
        let mut v = vec![fb.query(ImageQuery::SizeLod(level), img)?];
        if let Some(bits) = kind.bits(cx, a)? {
            v.push(fb.module().const_value(bits)?);
            v.push(fb.module().const_value(yes)?);
        }
        Ok(v)
    };
    let null_facts = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
        let mut v = vec![fb.module().const_value(ones)?];
        if with_bits {
            v.push(fb.module().const_value(zero)?);
            v.push(fb.module().const_value(no)?);
        }
        Ok(v)
    };
    c.each(cx, fb, &read, &null_facts)
}

/// The plane coordinate of sampling `s` as the device reads it, on the named candidate's level-0
/// size `size` (image facts `image`): the plane coordinate scaled from pixel coordinates over the
/// level-0 size, then snapped by `snap` (a pixel sampler's snap from the coordinate as given), and
/// with the layer appended; and its gradients. The first of the pair is the plane coordinate before
/// the snap, which a bias past the device's limit differentiates.
fn device_coordinate(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    grads: Option<(SValue, SValue)>,
    (size, image): (SValue, CombinedImage),
    snap: Option<&super::position::Snap>,
) -> BResult<((SValue, SValue), Option<(SValue, SValue)>)> {
    let (mut coord, mut grad) = plane_and_gradients(cx, fb, s, grads)?;
    // A pixel sampler's snap reads the coordinate as given.
    let given = coord;
    if let Some(spec) = pixel_coordinates(cx, fb, s)? {
        // Pixel coordinates scale by the level-0 size.
        let size_lanes = image.plane + u8::from(image.arrayed);
        (coord, grad) = scaled(fb, s.lanes, (size, size_lanes), spec, (coord, grad))?;
    }
    let plane = coord;
    if let Some(snap) = snap {
        // The device adds the texel offset to the snapped coordinate; each snap moves its index
        // back where Metal drops it.
        let offsets = snap_offsets(cx, fb, op_offset(&s.op), s.lanes)?;
        let at = (given, offsets.as_slice(), None);
        coord = super::position::snapped_on(cx, fb, snap, at, s.lanes, (size, image))?;
    }
    let (coord, _) = layered(cx, fb, s, coord)?;
    Ok(((plane, coord), grad))
}

/// A shared snapped gather or nearest sample (module docs): the coordinate snapped once on the
/// named size, and in each arm the device's access and the texel's conversion; the null arm
/// yields `null`'s value.
fn snapped(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: &Candidates<'_>,
    (first, snap): (&Sampling, &super::position::Snap),
    grads: Option<(SValue, SValue)>,
    (size, image): (SValue, CombinedImage),
    null: &NullLeaf<'_>,
) -> BResult<SValue> {
    let ((plane, coord), grad) =
        device_coordinate(cx, fb, first, grads, (size, image), Some(snap))?;
    let values = match &first.op {
        SampleOp::Sample {
            lod,
            dref,
            min_lod,
            offset,
        } => {
            let (ops, dref, beyond) = sample_operands(
                cx,
                fb,
                first,
                (plane, grad, false),
                (lod, *min_lod, offset, *dref),
            )?;
            let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
                let si = c.sampled_image(cx, fb, a)?;
                let x = sample_access(coord, dref, ops)(fb, si, false)?;
                // Past the device's limit the sample with the scaled derivatives replaces the
                // implicit one; the implicit sample stays where it was, unconditional.
                let x = match beyond {
                    None => x,
                    Some((test, grads)) => {
                        let mut grad_ops = ops;
                        grad_ops.bias = None;
                        grad_ops.grad = Some(grads);
                        let past: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
                            // A sampled image is consumed in the block that makes it.
                            let si = c.sampled_image(cx, fb, a)?;
                            let x = sample_access(coord, dref, grad_ops)(fb, si, false)?;
                            Ok(ArmEnd::FallThrough(vec![x]))
                        });
                        let within: ArmFn<'_, EmitCx<'_>> =
                            Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![x])));
                        let Merged(merged) = fb.if_(cx, test, past, Some(within))?;
                        match merged[..] {
                            [x] => x,
                            _ => return Err(cx.fail("a biased sample without its merged texel")),
                        }
                    }
                };
                // A depth sample without comparison reads the first component.
                let x = if a.depth && dref.is_none() {
                    fb.composite_extract(x, &[0])?
                } else {
                    x
                };
                Ok(vec![texel_value(
                    cx,
                    fb,
                    a,
                    image,
                    (x, crate::lower::texture::half::TexelOrigin::Computed),
                    None,
                )?])
            };
            c.each(cx, fb, &read, null)?
        }
        SampleOp::Gather {
            component,
            dref,
            offset,
        } => {
            let mut ops = ImageOperands::default();
            offset_operands(cx, fb, offset, first.lanes, &mut ops)?;
            let component = *component;
            let dref = match dref {
                Some(r) => Some(super::emit::reference(cx, fb, first, *r)?),
                None => None,
            };
            let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
                let si = c.sampled_image(cx, fb, a)?;
                let x = match dref {
                    Some(r) => fb.dref_gather(si, coord, r, &ops)?,
                    None => fb.gather(si, coord, component, &ops)?,
                };
                Ok(vec![texel_value(
                    cx,
                    fb,
                    a,
                    image,
                    (x, crate::lower::texture::half::TexelOrigin::Computed),
                    None,
                )?])
            };
            c.each(cx, fb, &read, null)?
        }
        _ => return Err(cx.fail("a shared snapped access without its sample or gather")),
    };
    match values[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a shared snapped access without its value")),
    }
}
