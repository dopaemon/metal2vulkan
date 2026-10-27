//! A `half` gather's texels the shader fetches itself, where Metal's footprint is known
//! ([`Plain`]).
//!
//! **Why.** Under the NaN switch Metal's gather hands back a 16-bit float texel's own NaN; a
//! device's gather need not (NVIDIA's hands back every NaN as `0x7fffffff`), and its fetch does
//! where its read does ([`half_gather`]).
//!
//! **How.** The shader fetches a gather's four taps at Metal's quantized position
//! (`TapBase::Quantized`) of the base level, in gather order (`texels`), through
//! `bicubic::footprint`. Each tap is addressed as the sampler's axes say, and a border axis's
//! outside tap reads the border color as the bound format holds it: a constexpr sampler's
//! addressing is the function's own ([`FetchAddress::Modes`]), a runtime sampler's its slot's
//! word's, read in the function ([`FetchAddress::Word`]), and an argument buffer's sampler's its
//! table entry's word's ([`FetchAddress::Entry`]). The fetch sits behind the slot's `TexelFormat`
//! specialization (`branch`): the half-float class takes the fetches, and any other the device's
//! own gather. A heap texture's gather fetches where the class its handle carries is the
//! half-float one ([`heap`]), after the device's access. An arm of a switch-at-use keeps the
//! device's read through a word, so its arms share one emulation.
//!
//! **The device reads the texels** elsewhere: a dynamic offset, an observed residency, a raw sRGB
//! view.

use crate::air::abi::Coordinates;
use crate::air::intrinsics::{ImageDim, ImageShape};
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::texture::{ImageRef, TexelConv, TexelForm};
use crate::metal::bicubic::BicubicTexels;
use crate::pipeline::sym::SamplerRef;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, Merged};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::Constant;

use super::bicubic::{
    flag, footprint_from, footprint_on, value, Addressing, At, Bits, Footprint, Tap, TapAddress,
    TapBase, Taps, Texels, Tys,
};
use super::linear::LinearImage;
use super::{BindingKey, CombinedImage, Offset, Pixel, SampleOp, Sampling, Source};

/// How the shader fetches the texels of a gather (module docs): how its taps are addressed, and
/// Metal's constant texel offset.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fetch {
    pub address: FetchAddress,
    pub offset: Option<Vec<i32>>,
}

/// How a fetch addresses its taps (`bicubic::Addressing`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FetchAddress {
    /// A constexpr sampler's: each plane axis's addressing, the sampler's border color (which
    /// every axis clamping to zero or to the border reads), and whether the coordinate is in
    /// pixels.
    Modes {
        address: Vec<TapAddress>,
        border: crate::air::abi::BorderColor,
        pixel: bool,
    },
    /// Runtime sampler slot `sampler`'s word on `axes` plane axes: each axis's address mode, the
    /// border color and the coordinates, read in the function from the slot's `SamplerState`
    /// specialization.
    Word { sampler: u32, axes: u8 },
    /// An argument buffer's sampler's word on `axes` plane axes, as [`FetchAddress::Word`], read
    /// in the function from the sampler's table entry.
    Entry { axes: u8 },
}

/// The estimated words of a fetch's footprint beyond its taps: the size query, each of two axes'
/// position ([`super::position::POSITION_WORDS`]) and base, the layer's clamp, the border texel,
/// and the format's branch and the device's arm.
pub(super) const FETCH_WORDS: u32 = 2 * super::position::POSITION_WORDS + 320;

/// The estimated words of one fetched tap: its index under its addressing on each axis, the fetch,
/// the border's select, and the component's extraction.
pub(super) const FETCH_TAP_WORDS: u32 = 96;

impl Fetch {
    /// The estimated words of fetching a gather's four texels: the footprint and each tap, and for
    /// a runtime sampler's word the word's tests and each axis's position and taps under every
    /// mode (`linear::WORD_WORDS`), and Metal's offset rule on each axis a constant offset moves
    /// (`position::OFFSET_RULE_WORDS`).
    pub fn words(&self) -> u32 {
        let word = match &self.address {
            FetchAddress::Modes { .. } => 0,
            FetchAddress::Word { axes, .. } | FetchAddress::Entry { axes } => {
                super::linear::WORD_WORDS
                    + u32::from(*axes)
                        * (super::position::MODE_POSITION_WORDS + 2 * super::linear::WORD_TAP_WORDS)
            }
        };
        let offset = self
            .offset
            .as_deref()
            .map_or(0, |o| o.iter().filter(|o| **o != 0).count() as u32)
            * super::position::OFFSET_RULE_WORDS;
        FETCH_WORDS + 4 * FETCH_TAP_WORDS + word + offset
    }

    /// The taps of a gather fetched as `self` says from the base level, a runtime sampler's
    /// addressed by its word `word` (a `u32`): a footprint of two taps per axis from Metal's
    /// quantized position. `None` for a word-addressed fetch without its word.
    pub(super) fn taps(&self, word: Option<SValue>) -> Option<Taps<'_>> {
        let addressing = match (&self.address, word) {
            (
                FetchAddress::Modes {
                    address,
                    border,
                    pixel,
                },
                _,
            ) => Addressing::Modes {
                address,
                border: *border,
                pixel: *pixel,
            },
            (FetchAddress::Word { axes, .. } | FetchAddress::Entry { axes }, Some(word)) => {
                Addressing::Word {
                    word,
                    axes: usize::from(*axes),
                }
            }
            (FetchAddress::Word { .. } | FetchAddress::Entry { .. }, None) => return None,
        };
        Some(Taps {
            addressing,
            offset: self.offset.as_deref(),
            span: &[0, 1],
            base: TapBase::Quantized,
            level: None,
        })
    }
}

/// A `half` gather whose texels the shader fetches (module docs): where its texel class is read
/// (a texture slot's `TexelFormat` specialization, or the class a heap texture's handle carries),
/// and how the fetches are made.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Plain {
    pub image: LinearImage,
    pub fetch: Fetch,
}

impl Plain {
    /// The estimated words of fetching the texels ([`Fetch::words`]), and a heap texture's class
    /// from its handle, its element loaded again in the fetch's arm, an argument buffer's sampler's
    /// word from its entry, and the branch (`linear::HEAP_WORDS`).
    pub fn words(&self) -> u32 {
        let heap = match (self.image, &self.fetch.address) {
            (LinearImage::Slot(_), _) => 0,
            (LinearImage::Heap, FetchAddress::Entry { .. }) => {
                super::linear::HEAP_WORDS + crate::memory::handles::ENTRY_FIELD_WORDS
            }
            (LinearImage::Heap, _) => super::linear::HEAP_WORDS,
        };
        self.fetch.words() + heap
    }
}

/// The fetch of `op`, a gather without comparison of a 2D texture or array with texel form `form`
/// (module docs, [`half_gather`]), of `image` (shape `shape`) through `stat` (or, where
/// `argument`, its argument buffer's sampler): through a constexpr sampler addressed by its modes,
/// through a runtime or argument buffer's sampler by its word. A heap texture's fetch reads the
/// class its handle carries.
pub(super) fn select(
    cx: &mut SelectCx<'_>,
    op: &SampleOp,
    form: Option<TexelForm>,
    (stat, argument): (Option<(SamplerRef, Coordinates)>, bool),
    (image, shape): (ImageRef, ImageShape),
) -> Result<Option<Plain>, Error> {
    if !half_gather(cx, op, form) || shape.dim != ImageDim::D2 || shape.ms {
        return Ok(None);
    }
    let SampleOp::Gather { offset, .. } = op else {
        return Ok(None);
    };
    let offset = match offset {
        None => None,
        Some(Offset::Const(k)) => Some(k.clone()),
        Some(Offset::Dynamic(..)) => return Ok(None),
    };
    let address = match (stat.map(|(r, _)| r), argument) {
        (Some(SamplerRef::Constexpr(words)), _) => {
            let s = super::state_of(words)?;
            FetchAddress::Modes {
                address: [s.address_s, s.address_t]
                    .into_iter()
                    .map(super::position::address)
                    .collect(),
                border: s.border_color,
                pixel: s.coordinates == Coordinates::Pixel,
            }
        }
        (Some(SamplerRef::Slot(sampler)), _) => FetchAddress::Word { sampler, axes: 2 },
        (None, true) => FetchAddress::Entry { axes: 2 },
        (None, false) => return Ok(None),
    };
    let image = match image {
        ImageRef::Slot(n) => LinearImage::Slot(n),
        ImageRef::Heap { .. } => LinearImage::Heap,
    };
    cx.families.access.heap_class |= image == LinearImage::Heap;
    Ok(Some(Plain {
        image,
        fetch: Fetch { address, offset },
    }))
}

/// Whether `op` is a gather with a `half` result whose texels the shader fetches where the bound
/// class is a 16-bit float one: where the NaN switch asks for Metal's NaN
/// (`NanMinMaxPolicy::Exact`, the profile's `TexelNanRoute::Metal`) on a device without Metal's
/// own sampler. Metal's gather hands back a 16-bit float texel's own NaN; a device's gather need
/// not (NVIDIA's hands back every NaN as `0x7fffffff`), and its fetch does
/// where its read does.
fn half_gather(cx: &SelectCx<'_>, op: &SampleOp, form: Option<TexelForm>) -> bool {
    use crate::device::profile::TexelNanRoute;
    use crate::device::requirements::VkProperty;
    matches!(op, SampleOp::Gather { dref: None, .. })
        && form.map(|f| f.conv) == Some(TexelConv::F16)
        && cx.profile.fields().texel_nans == TexelNanRoute::Metal
        && !cx
            .profile
            .fields()
            .enabled
            .properties
            .contains(&VkProperty::MetalSampler)
}

/// The value of gather `s` from the texels fetched as `f` says through combined binding `key` (one
/// without comparison), where the slot `texture`'s `TexelFormat` specialization states the
/// half-float class: [`plain_value`] of [`texels`]' texels. Elsewhere `device`'s access of the
/// binding's sampled image makes it. Each arm makes the sampled image it reads, so once a device
/// specializes the module the other arm is dead.
pub(super) fn branch(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    (s, key): (&Sampling, BindingKey),
    (texture, f): (u32, &Fetch),
    image: CombinedImage,
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let Some(spec) = crate::lower::texture::texel_format_const(cx, texture) else {
        return Err(cx.fail(format!(
            "texture slot {texture} has no `TexelFormat` specialization"
        )));
    };
    let Some(&(binding, _)) = cx.decl.families.sampler.combined.get(&key) else {
        return Err(cx.fail(format!("combined binding {key:?} is not declared")));
    };
    // A runtime sampler's word, which addresses its taps.
    let word = match &f.address {
        FetchAddress::Word { sampler, .. } => {
            Some(super::emit::sampler_word(cx, fb, Pixel::Spec(*sampler))?)
        }
        FetchAddress::Modes { .. } => None,
        FetchAddress::Entry { .. } => {
            return Err(cx.fail("a slot's fetch addressed by an argument buffer's sampler"))
        }
    };
    let t = Tys::new(fb)?;
    let measured = flag(fb, &t, spec, BicubicTexels::HALF)?;
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let si = binding.sampled_image(fb)?;
        let Some(taps) = f.taps(word) else {
            return Err(cx.fail("a word-addressed fetch without its word"));
        };
        let x = texels(cx, fb, t, s, (si, image), Bits::Spec(spec), &taps)?;
        Ok(ArmEnd::FallThrough(vec![plain_value(fb, s, &x)?]))
    });
    let read: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, _| {
        let si = binding.sampled_image(fb)?;
        let x = device(fb, si)?;
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let Merged(merged) = fb.if_(cx, measured, fetch, Some(read))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("fetched texels' branch without its value")),
    }
}

/// The value of fetch `p` of heap texture gather `s` (image facts `image`): `device`, the device's
/// gather already made, in uniform control flow; and where the class the texture's handle carries
/// is the half-float one, a component of the four texels the shader fetches from the element as a
/// slot of that class would ([`texels`]). The fetch arm holds fetches alone, so a divergent handle
/// puts no gather in divergent flow.
pub(super) fn heap(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    image: CombinedImage,
    p: &Plain,
    device: SValue,
) -> BResult<SValue> {
    let handle = match s.source {
        Source::Heap { handle, .. } | Source::HeapSampler { handle, .. } => handle,
        Source::Combined(_) => return Err(cx.fail("a heap texture's fetch of a slot")),
    };
    let t = Tys::new(fb)?;
    let h = cx.operand(fb, handle)?;
    let bits = Bits::Value(crate::lower::texture::heap_texel_class(cx, fb, h)?);
    let word = match (&p.fetch.address, super::emit::word_pixel(s)) {
        (FetchAddress::Modes { .. }, _) => None,
        (_, Some(pixel)) => Some(super::emit::sampler_word(cx, fb, pixel)?),
        (_, None) => return Err(cx.fail("a word-addressed heap fetch without its word")),
    };
    let take = bits.test(fb, &t, BicubicTexels::HALF)?;
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let (si, _, _) = super::emit::sampled_image(cx, fb, s.source)?;
        let Some(taps) = p.fetch.taps(word) else {
            return Err(cx.fail("a word-addressed heap fetch without its word"));
        };
        let x = texels(cx, fb, t, s, (si, image), bits, &taps)?;
        Ok(ArmEnd::FallThrough(vec![plain_value(fb, s, &x)?]))
    });
    let kept: ArmFn<'_, EmitCx<'_>> = Box::new(move |_, _| Ok(ArmEnd::FallThrough(vec![device])));
    let Merged(merged) = fb.if_(cx, take, fetch, Some(kept))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a heap texture's fetch without its texel")),
    }
}

/// The value of gather `s` of its fetched `texels`, as the device's gather returns it: its
/// component of each of its four (a `vec4`).
pub(super) fn plain_value(
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    texels: &[SValue],
) -> BResult<SValue> {
    let SampleOp::Gather { component, .. } = s.op else {
        return Err(super::bicubic::missing_tap());
    };
    let vec4 = Tys::new(fb)?.vec4;
    let mut parts = Vec::with_capacity(texels.len());
    for &texel in texels {
        parts.push(fb.composite_extract(texel, &[component])?);
    }
    fb.composite_construct(vec4, &parts)
}

/// The `vec4` texels of `s` at `taps` (any [`Addressing`]: a constexpr sampler's, or a runtime
/// sampler's word) from sampled image `si` (image facts `image`), a border tap's texel the border
/// color seen through format bits `bits` (the slot's `TexelFormat` specialization, or a heap
/// element's class's): a gather's four in gather order, `x` `(i, j + 1)`, `y` `(i + 1, j + 1)`,
/// `z` `(i + 1, j)` and `w` `(i, j)`, of a span of two taps per axis; else the one of a span of
/// one tap per axis.
pub(super) fn texels(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    s: &Sampling,
    (si, image): (SValue, CombinedImage),
    bits: Bits,
    taps: &Taps<'_>,
) -> BResult<Vec<SValue>> {
    let at = At::of(cx, fb, s)?;
    texels_at(cx, fb, t, &at, (si, image), bits, taps)
}

/// [`texels`] of a sampling whose plain data is `at`.
fn texels_at(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    t: &Tys,
    at: &At,
    (si, image): (SValue, CombinedImage),
    bits: Bits,
    taps: &Taps<'_>,
) -> BResult<Vec<SValue>> {
    let (fp, img) = footprint_from(fb, t, at, (si, image), bits, taps)?;
    let Some(sites) = sites(&fp) else {
        return Err(cx.fail("fetched texels' footprint without its taps"));
    };
    let mut out = Vec::with_capacity(sites.len());
    for site in &sites {
        out.push(fp.fetch(fb, img, site)?);
    }
    Ok(out)
}

/// The sites [`texels`] fetches over footprint `fp`: a span of two taps per axis's four in gather
/// order, `(i, j + 1)`, `(i + 1, j + 1)`, `(i + 1, j)` and `(i, j)`, else a span of one tap per
/// axis's one; `None` for another footprint.
fn sites(fp: &Footprint) -> Option<Vec<Vec<Tap>>> {
    let rows: Vec<&[Tap]> = fp.axes.iter().map(|(_, row)| &row[..]).collect();
    match &rows[..] {
        [&[s0, s1], &[t0, t1]] => {
            Some(vec![vec![s0, t1], vec![s1, t1], vec![s1, t0], vec![s0, t0]])
        }
        _ => rows
            .iter()
            .map(|r| match **r {
                [tap] => Some(tap),
                _ => None,
            })
            .collect::<Option<Vec<Tap>>>()
            .map(|site| vec![site]),
    }
}

/// The shared form of fetch `p` of gather `s` in a switch-at-use (SEMANTICS 1012;
/// `shared::shared_value`): the taps once on the named candidate's level-0 size `size` (image facts
/// `image`), fetched in one dispatch over the candidates `c` from each arm's image where the named
/// format's bits `bits` are of the half-float class, and [`plain_value`] of their texels;
/// elsewhere `device`'s access, dispatched the same way from each arm's sampled image.
pub(super) fn emit_shared(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Sampling,
    (size, image): (SValue, CombinedImage),
    p: &Plain,
    (bits, c): (SValue, &super::Candidates<'_>),
    device: &dyn Fn(&mut FunctionBuilder<'_>, SValue) -> BResult<SValue>,
) -> BResult<SValue> {
    let t = Tys::new(fb)?;
    let bits = Bits::Value(bits);
    let measured = bits.test(fb, &t, BicubicTexels::HALF)?;
    let t = &t;
    let fetch: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let Some(taps) = p.fetch.taps(None) else {
            return Err(cx.fail("a shared fetch without its taps"));
        };
        let fp = footprint_on(cx, fb, t, s, (size, image), bits, &taps)?;
        let Some(sites) = sites(&fp) else {
            return Err(cx.fail("a shared fetch's footprint without its taps"));
        };
        let x = Texels::Candidates(c).fetch(cx, fb, (t, &fp), &sites, &|_, _, v| Ok(v))?;
        Ok(ArmEnd::FallThrough(vec![plain_value(fb, s, &x)?]))
    });
    let read: ArmFn<'_, EmitCx<'_>> = Box::new(move |fb, cx| {
        let read = |cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Sampling| {
            let si = c.sampled_image(cx, fb, a)?;
            Ok(vec![device(fb, si)?])
        };
        let null = |_: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>| {
            let z = fb.module().constant(Constant::Null { ty: t.vec4 })?;
            Ok(vec![value(fb, z)?])
        };
        let [x] = c.each(cx, fb, &read, &null)?[..] else {
            return Err(cx.fail("a shared fetch's device texel"));
        };
        Ok(ArmEnd::FallThrough(vec![x]))
    });
    let Merged(merged) = fb.if_(cx, measured, fetch, Some(read))?;
    match merged[..] {
        [x] => Ok(x),
        _ => Err(cx.fail("a shared fetch's branch without its texel")),
    }
}
