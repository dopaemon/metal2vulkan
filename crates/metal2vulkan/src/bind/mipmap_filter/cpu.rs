//! The filter on the CPU ([`super::MipmapFilterChain::expected_levels`]): the same `f32` operations
//! in the same order as the kernel (`kernel.rs`), each rounded on its own, so that a device that
//! rounds each operation as IEEE does (the kernel's sums and products are not contracted) writes
//! these bytes.

use super::layout::{axis_tap, Field, Layout, Rule, Texel, EXTENDED_SCALE, SRGB_STEP_SCALE};
use crate::lower::arith::srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::extended_srgb;

/// The texel's bytes as one little-endian integer.
fn word_of(texel: &[u8]) -> u128 {
    texel
        .iter()
        .rev()
        .fold(0u128, |word, &byte| (word << 8) | u128::from(byte))
}

/// Field `f`'s bits of a texel's `word`.
fn bits(word: u128, f: &Field) -> u32 {
    ((word >> f.offset) as u32) & f.mask()
}

/// `2^e` for `-126 <= e <= 127`.
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// The value of float bits `b` of a 5-bit exponent and `mantissa` mantissa bits (signed or not):
/// a half's or an unsigned packed float's, exactly.
pub(super) fn small_float_value(b: u32, mantissa: u32, signed: bool) -> f32 {
    let e = (b >> mantissa) & 31;
    let m = b & ((1 << mantissa) - 1);
    let sign = if signed { (b >> (mantissa + 5)) & 1 } else { 0 };
    let magnitude = match e {
        31 => 0x7f80_0000 | (m << (23 - mantissa)),
        0 => (m as f32 * pow2(-14 - mantissa as i32)).to_bits(),
        _ => ((e + 112) << 23) | (m << (23 - mantissa)),
    };
    f32::from_bits(magnitude | (sign << 31))
}

/// The bits of `v` as a float of a 5-bit exponent and `mantissa` mantissa bits (signed or not),
/// to nearest even: an overflow infinity, a NaN the quiet NaN, a negative value of an unsigned
/// format 0.
pub(super) fn small_float_bits(v: f32, mantissa: u32, signed: bool) -> u32 {
    let b = v.to_bits();
    let sign = b >> 31;
    let a = b & 0x7fff_ffff;
    let inf = 31 << mantissa;
    if a > 0x7f80_0000 {
        return inf | (1 << (mantissa - 1));
    }
    if !signed && sign == 1 {
        return 0;
    }
    let k = 23 - mantissa;
    let e = a >> 23;
    let r = if e >= 113 {
        let v = a - (112 << 23);
        ((v + (1 << (k - 1)) - 1 + ((v >> k) & 1)) >> k).min(inf)
    } else {
        let shift = k + 113 - e;
        if shift >= 32 {
            0
        } else {
            let sig = (a & 0x7f_ffff) | 0x80_0000;
            (sig + (1 << (shift - 1)) - 1 + ((sig >> shift) & 1)) >> shift
        }
    };
    if signed {
        r | (sign << (mantissa + 5))
    } else {
        r
    }
}

/// `v` with a NaN read as 0.
fn number(v: f32) -> f32 {
    if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// The extended-range `_sRGB` offset of encode index `i`: the largest whose first index is at
/// most `i`.
fn extended_offset(i: u32) -> u32 {
    (extended_srgb::SRGB_ENCODE_FIRST.partition_point(|&first| u32::from(first) <= i) - 1) as u32
}

/// The value of field `f`'s code `c` (`layout::Rule`).
fn value(f: &Field, c: u32) -> f32 {
    match f.rule {
        Rule::Unorm => c as f32 * f.scale(),
        Rule::Snorm => {
            let shift = 32 - f.width;
            let c = ((c << shift) as i32) >> shift;
            (c as f32 * f.scale()).max(-1.0)
        }
        Rule::Srgb8 => f32::from(SRGB_DECODE_STEPS[c as usize]) * SRGB_STEP_SCALE,
        Rule::Float { exponent: 8, .. } => f32::from_bits(c),
        Rule::Float {
            mantissa, signed, ..
        } => small_float_value(c, mantissa, signed),
        Rule::Extended { alpha } => {
            let c = if alpha { c.clamp(384, 894) } else { c };
            (c as i32 - 384) as f32 * EXTENDED_SCALE
        }
        Rule::ExtendedSrgb => {
            let n = (c as i32 - 384).unsigned_abs();
            let v = f32::from(extended_srgb::SRGB_DECODE_STEPS[n as usize]) * pow2(-13);
            if c < 384 {
                -v
            } else {
                v
            }
        }
        Rule::Padding(_) => 0.0,
    }
}

/// Field `f`'s code of value `v` (`layout::Rule`), in its low `width` bits.
fn code(f: &Field, v: f32) -> u32 {
    match f.rule {
        Rule::Unorm => (number(v).clamp(0.0, 1.0) * f.den() as f32).round_ties_even() as u32,
        Rule::Snorm => {
            let c = (number(v).clamp(-1.0, 1.0) * f.den() as f32).round_ties_even() as i32;
            c as u32 & f.mask()
        }
        Rule::Srgb8 => {
            let i = (number(v).clamp(0.0, 1.0) * 4095.0).round_ties_even() as u32;
            (SRGB_ENCODE_FIRST.partition_point(|&first| u32::from(first) <= i) - 1) as u32
        }
        Rule::Float { exponent: 8, .. } => {
            if v.is_nan() {
                0x7fc0_0000
            } else {
                v.to_bits()
            }
        }
        Rule::Float {
            mantissa, signed, ..
        } => small_float_bits(v, mantissa, signed),
        Rule::Extended { alpha } => {
            let (lo, hi) = if alpha { (0.0, 1.0) } else { (-1.0, 1.5) };
            let c = (number(v).clamp(lo, hi) * 510.0).round_ties_even() as i32 + 384;
            c.clamp(0, 1023) as u32
        }
        Rule::ExtendedSrgb => {
            let v = number(v);
            let i = (v.abs().min(2.0) * 8192.0).round_ties_even() as u32;
            let n = extended_offset(i);
            if v < 0.0 {
                384 - n.min(384)
            } else {
                384 + n
            }
        }
        Rule::Padding(bits) => bits & f.mask(),
    }
}

/// The `float` channels of a texel of `layout` from its bytes.
pub(super) fn decode(layout: &Layout, texel: &[u8]) -> Vec<f32> {
    let word = word_of(texel);
    match layout.texel {
        Texel::Fields(fields) => fields
            .iter()
            .filter(|f| f.channel())
            .map(|f| value(f, bits(word, f)))
            .collect(),
        Texel::Rgb9e5 => {
            let w = word as u32;
            let scale = pow2((w >> 27) as i32 - 24);
            (0..3)
                .map(|c| ((w >> (9 * c)) & 511) as f32 * scale)
                .collect()
        }
    }
}

/// `2^(24 - e)`: a shared exponent's mantissa scale.
fn rgb9e5_scale(e: i32) -> f32 {
    pow2(24 - e)
}

/// The bytes of a texel of `layout` from its `float` channels.
pub(super) fn encode(layout: &Layout, v: &[f32]) -> Vec<u8> {
    let word: u128 = match layout.texel {
        Texel::Fields(fields) => {
            let mut values = v.iter();
            fields.iter().fold(0u128, |word, f| {
                let c = match f.rule {
                    Rule::Padding(_) => code(f, 0.0),
                    _ => code(f, *values.next().expect("a channel per field")),
                };
                word | (u128::from(c) << f.offset)
            })
        }
        Texel::Rgb9e5 => {
            let c: Vec<f32> = v.iter().map(|&x| number(x).clamp(0.0, 65408.0)).collect();
            let most = c[0].max(c[1]).max(c[2]);
            let floor = (((most.to_bits() >> 23) & 0xff) as i32 - 127).max(-16);
            let mut e = floor + 16;
            if (most * rgb9e5_scale(e) + 0.5).floor() == 512.0 {
                e += 1;
            }
            let scale = rgb9e5_scale(e);
            let mut w = (e as u32) << 27;
            for (i, x) in c.iter().enumerate() {
                let m = ((x * scale + 0.5).floor() as u32).min(511);
                w |= m << (9 * i);
            }
            u128::from(w)
        }
    };
    word.to_le_bytes()[..layout.bytes as usize].to_vec()
}

/// One level of every layer of `layout` (`small` texels a layer) from the level above (`big`):
/// each destination texel the weighted sum of its taps' channels (`axis_tap` along each axis,
/// the depth taps where the level above is more than one texel deep), a tap of weight 0 left
/// out, the sum taken in tap order (x fastest, then y, then z).
pub(super) fn level(
    layout: &Layout,
    above: &[u8],
    big: [u32; 3],
    small: [u32; 3],
    layers: u32,
) -> Vec<u8> {
    let bytes = layout.bytes as usize;
    let depth = big[2] > 1;
    let taps = |axis: usize, i: u32| {
        let (x0, t) = axis_tap(i, big[axis], small[axis]);
        [(x0, 1.0 - t), ((x0 + 1).min(big[axis] - 1), t)]
    };
    let count = if depth { 8 } else { 4 };
    let texels = (layers * small[0] * small[1] * small[2]) as usize;
    let mut out = Vec::with_capacity(texels * bytes);
    for layer in 0..layers {
        for z in 0..small[2] {
            for y in 0..small[1] {
                for x in 0..small[0] {
                    let (ax, ay) = (taps(0, x), taps(1, y));
                    let az = taps(2, z);
                    let mut sum = vec![0.0f32; layout.channels()];
                    for t in 0..count {
                        let (sx, wx) = ax[t & 1];
                        let (sy, wy) = ay[(t >> 1) & 1];
                        let (image, w) = if depth {
                            let (sz, wz) = az[(t >> 2) & 1];
                            (layer * big[2] + sz, wx * wy * wz)
                        } else {
                            (layer * small[2] + z, wx * wy)
                        };
                        let index = ((image * big[1] + sy) * big[0] + sx) as usize;
                        let v = decode(layout, &above[index * bytes..(index + 1) * bytes]);
                        for (s, v) in sum.iter_mut().zip(v) {
                            let term = if w == 0.0 { 0.0 } else { w * v };
                            *s = if t == 0 { term } else { *s + term };
                        }
                    }
                    out.extend(encode(layout, &sum));
                }
            }
        }
    }
    out
}
