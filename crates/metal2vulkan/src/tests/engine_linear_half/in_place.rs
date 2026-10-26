//! A two-plane YCbCr texture's planes read in place (`lower::sampler::in_place`), on the CPU
//! interpreter against Metal's texels and Metal's filter of them, bit for bit.
//!
//! The slot under its in-place variant (`LoweringInputs::ycbcr_in_place`) is a uniform texel
//! buffer of the planes' codes (`R8G8_UINT` elements of 8-bit codes, `R16G16_UINT` ones of 10-bit
//! codes, each 10-bit code the high bits of its word over junk low bits), and its push field
//! states the texture's size, its planes' places and rows, and its conversion
//! (`bind::YCbCrInPlace::words`). Each texel the samples read is Metal's (`ycbcr_convert`'s module
//! docs: the luma code, the chroma pairs' 9/3/3/1 filter, the range's and the matrix's
//! conversion), and a linear sample is Metal's filter of those steps on the grid `2048`
//! (`super::ycbcr`'s rule); a nearest sample is the texel at Metal's position. The texel buffer
//! traps every fetch outside it, so each tap's elements lie within the planes.

use super::ycbcr::{metal_step, result_bits};
use super::*;
use crate::bind::ycbcr_convert::{ycbcr_read, YCbCrCodes, YCbCrMatrix, YCbCrPlanes, YCbCrRange};
use crate::bind::YCbCrInPlace;
use crate::lower::select::LowerInputs;
use crate::metal::format::ycbcr_format;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::{compile_with, typed_module_with};

/// The texture's width and height: odd, so the chroma plane's last pair serves one column and one
/// row alone.
const W: u32 = 7;
const H: u32 = 5;

/// Lowering inputs with texture slot 0 under its in-place variant, or none.
fn inputs(in_place: bool) -> LowerInputs {
    let mut i = LowerInputs::new();
    if in_place {
        i.ycbcr_in_place.insert(TEXTURE);
    }
    i
}

/// One texture's planes: their codes by texel, where they lie, and the texel buffer's elements.
struct Planes {
    planes: YCbCrPlanes,
    words: YCbCrInPlace,
    /// Each element's two lanes, as the texel buffer's format reads them.
    elements: Vec<[u32; 2]>,
    luma: Vec<u16>,
    chroma: Vec<[u16; 2]>,
}

impl Planes {
    /// Random planes of `pf` (`YCBCR8_420_2P` or `YCBCR10_420_2P`) under `conversion`: the chroma
    /// plane first, three elements in, then the luma plane, each row padded past its codes.
    fn random(pf: PixelFormat, conversion: (YCbCrRange, YCbCrMatrix), rng: &mut Lcg) -> Planes {
        let codes = ycbcr_format(pf).and_then(YCbCrCodes::of).expect("codes");
        let unit = codes.bytes;
        let element = 2 * unit;
        let [cw, ch] = [W.div_ceil(2), H.div_ceil(2)];
        let chroma = [3 * element, (cw + 2) * element];
        let luma_at = (chroma[0] + ch * chroma[1]).next_multiple_of(element) + element;
        let luma = [luma_at, (W.div_ceil(2) + 1) * element];
        let planes = YCbCrPlanes::new([W, H], luma, chroma, conversion, codes);
        let words = YCbCrInPlace::of(&planes).expect("in place");
        let max = (1u64 << codes.bits) - 1;
        let mut code = || (rng.next() % (max + 1)) as u16;
        let luma_codes: Vec<u16> = (0..W * H).map(|_| code()).collect();
        let chroma_codes: Vec<[u16; 2]> = (0..cw * ch).map(|_| [code(), code()]).collect();
        // Every lane a junk value, then each code's place written: a 10-bit code is the high bits
        // of its 16-bit word.
        let n = words.elements() as usize;
        let mut lanes: Vec<u32> = (0..2 * n)
            .map(|_| (rng.next() % (1 << (8 * unit))) as u32)
            .collect();
        let shift = 8 * unit - codes.bits;
        let mut put = |byte: u32, code: u16, junk: u64| {
            let lane = (byte / unit) as usize;
            lanes[lane] = (u32::from(code) << shift) | (junk as u32 & ((1 << shift) - 1));
        };
        for y in 0..H {
            for x in 0..W {
                let c = luma_codes[(y * W + x) as usize];
                put(luma[0] + y * luma[1] + x * unit, c, rng.next());
            }
        }
        for y in 0..ch {
            for x in 0..cw {
                let [cb, cr] = chroma_codes[(y * cw + x) as usize];
                let at = chroma[0] + y * chroma[1] + 2 * x * unit;
                put(at, cb, rng.next());
                put(at + unit, cr, rng.next());
            }
        }
        let elements = lanes.as_chunks::<2>().0.to_vec();
        Planes {
            planes,
            words,
            elements,
            luma: luma_codes,
            chroma: chroma_codes,
        }
    }

    /// Metal's texel `(x, y)`: each channel's step on the grid `2048`.
    fn texel(&self, x: i64, y: i64) -> [i64; 3] {
        let [cw, ch] = [W.div_ceil(2), H.div_ceil(2)].map(i64::from);
        // An axis's chroma texels weighted 3 and 1: its own, and the one after it for an odd
        // index, before it for an even one, clamped to the plane.
        let taps = |q: i64, n: i64| -> [(i64, u32); 2] {
            let own = q / 2;
            let other = if q % 2 == 1 { own + 1 } else { own - 1 };
            [(own, 3), (other.clamp(0, n - 1), 1)]
        };
        let mut sums = [0u32; 2];
        for (cy, wy) in taps(y, ch) {
            for (cx, wx) in taps(x, cw) {
                let pair = self.chroma[(cy * cw + cx) as usize];
                for (sum, c) in sums.iter_mut().zip(pair) {
                    *sum += wx * wy * u32::from(c);
                }
            }
        }
        let [cb, cr] = sums.map(|s| ((s + 8) >> 4) as u16);
        let luma = self.luma[(y * i64::from(W) + x) as usize];
        let p = &self.planes;
        ycbcr_read(p.codes, (p.range, p.matrix), [luma, cb, cr]).map(i64::from)
    }

    /// The texel buffer bound at `slot`.
    fn image(&self, slot: ResourceSlot) -> Image {
        let n = self.elements.len() as u32;
        let mut image = Image::new(slot, [n, 1, 1], 1, 1, 1);
        for (x, [a, b]) in (0u32..).zip(&self.elements) {
            let i = image.index(0, 0, 0, [x, 0, 0]).expect("element");
            image.texels[i] = [*a, *b, 0, 1];
        }
        image
    }
}

/// The texel buffer binding a module declares for texture slot 0.
fn texel_buffer(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(
                s @ ResourceSlot {
                    kind: ResourceKind::Texture,
                    index: 0,
                    ..
                },
            ) => Some(s),
            _ => None,
        })
        .expect("the planes' binding")
}

/// Resources of a dispatch of one-thread threadgroups: `[[buffer(i)]]` holding `init[i]`, and
/// the slot's push field holding `words`.
fn resources(plan: &Plan, init: &[Vec<u8>], words: &YCbCrInPlace) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let mut push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let mut written = false;
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::YCbCrInPlace(TEXTURE), PushLoc::Push { offset }) => {
                let at = offset as usize;
                for (i, w) in words.words().into_iter().enumerate() {
                    push[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
                }
                written = true;
            }
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            (PushField::Specialized(_), _) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    assert!(written, "the planes' push field");
    let buffers = init
        .iter()
        .enumerate()
        .map(|(i, b)| Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: i as u32,
            }),
            address: None,
            bytes: b.clone(),
        })
        .collect();
    Resources {
        buffers,
        push,
        spec,
    }
}

/// A sampler a test samples through: a constexpr sampler's words, or a runtime sampler's word.
#[derive(Clone, Copy, Debug)]
enum Through {
    Constexpr(u64),
    Runtime(u32),
}

/// The word of a runtime sampler of `filter` (linear, else nearest), every axis addressed by
/// `mode`, normalized coordinates, transparent black.
fn word(linear: bool, mode: SamplerAddressMode) -> u32 {
    let mut w = if linear { SamplerWord::LINEAR } else { 0 };
    for shift in SamplerWord::ADDRESS_SHIFT {
        w |= mode.as_raw() << shift;
    }
    w
}

/// Metal's sample of `planes` at footprint position `q` (`256 (x - 1/2)` per axis, the texel
/// coordinate `x`) offset by `offset` texels, clamped to the edge: linear, the rule's steps of
/// the four taps; nearest, the texel at `⌊x⌋`.
fn metal(planes: &Planes, linear: bool, (qx, qy): (i64, i64), offset: (i64, i64)) -> [i64; 3] {
    let edge = |i: i64, n: u32| i.clamp(0, i64::from(n) - 1);
    if !linear {
        let x = edge(((qx + 128) >> 8) + offset.0, W);
        let y = edge(((qy + 128) >> 8) + offset.1, H);
        return planes.texel(x, y);
    }
    let (i, j) = ((qx >> 8) + offset.0, (qy >> 8) + offset.1);
    let (i0, i1) = (edge(i, W), edge(i + 1, W));
    let (j0, j1) = (edge(j, H), edge(j + 1, H));
    let a = if i0 == i1 { 0 } else { qx & 255 };
    let b = if j0 == j1 { 0 } else { qy & 255 };
    let taps = [
        planes.texel(i0, j0),
        planes.texel(i1, j0),
        planes.texel(i0, j1),
        planes.texel(i1, j1),
    ];
    std::array::from_fn(|c| metal_step(taps.map(|t| t[c]), a, b))
}

/// Every sample of each format's random planes under each conversion, through constexpr and
/// runtime samplers (nearest and linear; a runtime sampler's repeat sampled as clamp-to-edge, as
/// a 4:2:0 texture's), with a constant offset or none, `float` and `half` results, on lavapipe and
/// on MoltenVK, is Metal's: a quarter of the positions past an edge.
#[test]
fn in_place_samples_are_metals() {
    let conversions = [
        (YCbCrRange::Video, YCbCrMatrix::Bt601),
        (YCbCrRange::Video, YCbCrMatrix::Bt709),
        (YCbCrRange::Video, YCbCrMatrix::Bt2020),
        (YCbCrRange::Full, YCbCrMatrix::Bt601),
        (YCbCrRange::Full, YCbCrMatrix::Bt709),
        (YCbCrRange::Full, YCbCrMatrix::Bt2020),
    ];
    let mut rng = Lcg(0x1c_bc_7a_1e);
    let planes: Vec<Planes> = [PixelFormat::YCBCR84202P, PixelFormat::YCBCR104202P]
        .into_iter()
        .flat_map(|pf| conversions.map(|c| (pf, c)))
        .map(|(pf, c)| Planes::random(pf, c, &mut rng))
        .collect();
    // Footprint positions `q = 256 (x - 1/2)`, each axis within one and a half texels of the
    // texture; a nearest sample's off the texel edges, where `⌊x⌋` holds for any rounding.
    let positions: Vec<(i64, i64)> = (0..64)
        .map(|_| {
            let mut q = |n: u32| {
                let q = (rng.next() % (256 * u64::from(n) + 768)) as i64 - 384;
                match (q + 128) & 255 {
                    0..8 => q + 8,
                    248.. => q - 8,
                    _ => q,
                }
            };
            (q(W), q(H))
        })
        .collect();
    let out = |q: i64, n: u32| q < 0 || q >= 256 * (i64::from(n) - 1);
    let edges = positions
        .iter()
        .filter(|&&(qx, qy)| out(qx, W) || out(qy, H))
        .count();
    assert!(edges >= positions.len() / 4, "{edges} past an edge");
    let coords: Vec<u8> = positions
        .iter()
        .flat_map(|&(qx, qy)| {
            let n = |q: i64, n: u32| (q + 128) as f32 / (256 * i64::from(n)) as f32;
            [n(qx, W).to_le_bytes(), n(qy, H).to_le_bytes()]
        })
        .flatten()
        .collect();
    let edge = SamplerAddressMode::ClampToEdge;
    let cases = [
        (
            "constexpr linear",
            Through::Constexpr(linear(1, false, 0)),
            true,
            (0, 0),
        ),
        (
            "constexpr nearest",
            Through::Constexpr(sampler_words(0, 0, 1, false, 0)),
            false,
            (0, 0),
        ),
        (
            "constexpr offset",
            Through::Constexpr(linear(1, false, 0)),
            true,
            (2, -1),
        ),
        (
            "runtime linear",
            Through::Runtime(word(true, edge)),
            true,
            (0, 0),
        ),
        (
            "runtime nearest",
            Through::Runtime(word(false, edge)),
            false,
            (0, 0),
        ),
        (
            "runtime repeat",
            Through::Runtime(word(true, SamplerAddressMode::Repeat)),
            true,
            (0, 0),
        ),
    ];
    for (name, through, linear, offset) in cases {
        for half in [false, true] {
            let sampler = match through {
                Through::Constexpr(w) => Some(w),
                Through::Runtime(_) => None,
            };
            let form = Form::Plain {
                offset: (offset.0 as i32, offset.1 as i32),
            };
            let Some(bc) = assemble(name, &kernel_of(sampler, form, !half)) else {
                return;
            };
            for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
                let what = format!("{name} half={half} on {stem}");
                let a = compile_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    a.strategies
                        .entries
                        .keys()
                        .any(|k| k.starts_with("sampler.ycbcr_in_place.")),
                    "{what}: {:?}",
                    a.strategies.entries.keys().collect::<Vec<_>>()
                );
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let slot = texel_buffer(&m);
                let n = positions.len();
                let size = if half { 8 } else { 16 };
                for planes in &planes {
                    let init = [coords.clone(), vec![0xab; size * n]];
                    let res = resources(&plan, &init, &planes.words);
                    let word = match through {
                        Through::Runtime(w) => Some(w),
                        Through::Constexpr(_) => None,
                    };
                    let res = with_word(res, &a, word);
                    let got = run(&m, &res, &planes.image(slot), n as u32, half);
                    for (q, g) in positions.iter().zip(got) {
                        let want = metal(planes, linear, *q, offset);
                        let p = &planes.planes;
                        assert_eq!(
                            g,
                            result_bits(want, half, 2048),
                            "{what} {:?} {}-bit q {q:?}: {g:x?}",
                            (p.range, p.matrix),
                            p.codes.bits,
                        );
                    }
                }
            }
        }
    }
}

/// A slot whose every access the form reads states `RuntimeFact::YCbCrInPlace` with the runtime
/// samplers it samples through, on a profile that takes the form (lavapipe, a CPU device; MoltenVK
/// on the M3, an integrated one) and on no other (the RTX 5080, a discrete one); a gather, a
/// dynamic offset, `min_lod_clamp` and a 3D texture keep it from the form. Its compile under the
/// variant states the same fact, so its plan names the same variant.
#[test]
fn a_slot_states_the_in_place_form_where_it_reads_only_it() {
    let fact = |samplers| RuntimeFact::YCbCrInPlace {
        slot: TEXTURE,
        samplers,
    };
    let reads = [
        (
            "linear",
            kernel(Some(linear(1, false, 0)), Form::Float),
            Some(0),
        ),
        ("runtime", kernel(None, Form::Float), Some(1)),
        (
            "dynamic offset",
            kernel(Some(linear(1, false, 0)), Form::DynamicOffset),
            None,
        ),
        (
            "min_lod_clamp",
            kernel(Some(linear(1, false, 0)), Form::MinLodClamp),
            None,
        ),
        (
            "3d",
            kernel(
                Some(linear(1, false, 0)),
                Form::ThreeD { offset: (0, 0, 0) },
            ),
            None,
        ),
    ];
    for (name, body, samplers) in reads {
        let Some(bc) = assemble(name, &body) else {
            return;
        };
        for (stem, takes) in [
            ("lavapipe", true),
            ("moltenvk-m3", true),
            ("nvidia-rtx5080-laptop", false),
        ] {
            let p = profile(stem, EXACT_SAMPLING).expect("profile");
            assert_eq!(p.fields().ycbcr_in_place, takes, "{stem}");
            let what = format!("{name} on {stem}");
            let (plan, _) = typed_module_with(&bc, "k", &p, &inputs(false))
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let stated = plan.facts.iter().find_map(|f| match f {
                RuntimeFact::YCbCrInPlace { slot, samplers } if *slot == TEXTURE => Some(*samplers),
                _ => None,
            });
            assert_eq!(stated, samplers.filter(|_| takes), "{what}");
            if let (Some(s), true) = (samplers, takes) {
                let (variant, _) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what}, in place: {e}"));
                assert!(variant.facts.contains(&fact(s)), "{what}, in place");
            }
        }
    }
}
