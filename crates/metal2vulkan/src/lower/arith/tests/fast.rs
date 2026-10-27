//! The fast functions Metal composes, on the interpreter against Metal's recorded answers (Apple
//! M3 Max, probe case 08bc6611, compiled with the default fast math): `fast::fmod` as
//! `fma(-trunc(x / y), y, x)` of the device's quotient on every `float` fused multiply-add route,
//! and `fast::sqrt` of Metal's own `rsqrt`.

use super::*;
use crate::lower::arith::fma::{self, Fma, Route};
use crate::lower::arith::libm::{Lib, Mode};

/// `(x, y, fast::fmod(x, y))`: the lanes of case 08bc6611 whose fast reciprocal of `y` is the
/// correctly rounded `1 / y` (917 of 1,024; these are its special lanes and 44 others drawn at
/// random).
const FMOD: [(u32, u32, u32); 108] = [
    (0xBFFE18C9, 0xC15FD2B5, 0xBFFE18C9),
    (0x3E57C427, 0x400109AA, 0x3E57C427),
    (0x3FFEC76F, 0xC11ADCA2, 0x3FFEC76F),
    (0xBF7A8F38, 0x40D82500, 0xBF7A8F38),
    (0x3FDACF22, 0xBF9E5534, 0x3EF1E7B8),
    (0x40370A25, 0x41351B59, 0x40370A25),
    (0x40333E0F, 0xC070368B, 0x40333E0F),
    (0x3F87699E, 0xBDCA9F43, 0x3D8C6142),
    (0x400F6B18, 0x404C13A5, 0x400F6B18),
    (0xC044DE3B, 0xC1067FAD, 0xC044DE3B),
    (0xBFC1C9F0, 0xC1596163, 0xBFC1C9F0),
    (0x3F455453, 0x40E5F119, 0x3F455453),
    (0xC004006C, 0xC1174EE9, 0xC004006C),
    (0xC033AD6C, 0xC1581FB8, 0xC033AD6C),
    (0xBF9346C4, 0xC0DA6A55, 0xBF9346C4),
    (0xBEABFA9E, 0x411845D4, 0xBEABFA9E),
    (0xAC7CF4DE, 0x3FC6FCE4, 0xAC7CF4DE),
    (0xAD71D04C, 0x401803AA, 0xAD71D04C),
    (0x3D24224C, 0x401C03CB, 0x3D24224C),
    (0x313F63AA, 0x405E6F2F, 0x313F63AA),
    (0xC3E60FEB, 0x40BE90BB, 0xBFC9CA04),
    (0x4808DD1C, 0xC0388B55, 0x3FBE1A92),
    (0xCB225FBF, 0x3E2F7E58, 0xBD97D9C0),
    (0x4AFD7F51, 0x4093C524, 0x3F760C00),
    (0x48354100, 0xC0B62F00, 0x408ED800),
    (0x5A900799, 0xBF634297, 0x4ECA3738),
    (0x5D4E041C, 0xC05F3547, 0xD0752700),
    (0x404AE76D, 0x3F9536FC, 0x3F56C1C4),
    (0x7CED6A0D, 0x412F29DB, 0x6E499AC0),
    (0x084A6E20, 0x4167E3DE, 0x084A6E20),
    (0x5785B164, 0xBE805703, 0xCA344400),
    (0x735934AF, 0xC10AA805, 0x66CD7F78),
    (0x584741F4, 0xBF6C5E08, 0xCC3E24C8),
    (0x7D0F380B, 0xC0FCCC48, 0x707E50E0),
    (0x47D98FB0, 0x411B6F4C, 0x403C7820),
    (0xC1C1650E, 0xB78404E2, 0xB47FF500),
    (0x42919FFE, 0x35DF6EE2, 0x365423D0),
    (0xC15B3515, 0x804230F6, 0x7FC00000),
    (0xC24E4367, 0xE310DC76, 0xC24E4367),
    (0x41A18EC6, 0xD1497DA4, 0x41A18EC6),
    (0xC285F86E, 0x61DE79B2, 0xC285F86E),
    (0x42831361, 0x804188FC, 0x7FC00000),
    (0x3F800020, 0xC10FA238, 0x3F800020),
    (0x3F7FFFF8, 0x417D4D92, 0x3F7FFFF8),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
    (0x7F800000, 0x3FC90FDB, 0x7FC00000),
    (0xFF800000, 0x7F800000, 0x7FC00000),
    (0x7FC00000, 0x7F7FFFFF, 0x7FC00000),
    (0x3F800000, 0x00000001, 0x7FC00000),
    (0xBF800000, 0x3F800000, 0x00000000),
    (0x7F7FFFFF, 0x00000000, 0x7FC00000),
    (0x40490FDB, 0xFF800000, 0x7FC00000),
    (0x3FC90FDB, 0x40490FDB, 0x3FC90FDB),
    (0x00000000, 0x80400000, 0x7FC00000),
    (0x80000000, 0xBF800000, 0x00000000),
    (0x00000001, 0x80000000, 0x7FC00000),
    (0x80400000, 0x7FC00000, 0x7FC00000),
];

/// `fast::fmod` (`math::fast_fmod`) on every `float` route of its fused multiply-add, at the
/// lanes of [`FMOD`], bit for bit: Metal's answer is the expression on its own quotient, its fast
/// division's, where the module takes the device's division (`PrecisionNote::NativeDivision`),
/// here the interpreter's correctly rounded one. A lane whose finite answer depends on the
/// quotient's last bits (a quotient of `2^22` or more in magnitude, whose `trunc` they decide) is
/// the device's division's and not asserted. The device's division, `OpFmaKHR` and `Fma` keep
/// denormals (here the interpreter's IEEE ones), so a lane with an `f32` denormal operand or a
/// zero or denormal answer is the device's and not asserted either.
#[test]
fn fast_fmod_answers_as_metal_on_every_route() {
    let f = float(32);
    let cases: Vec<Case> = FMOD
        .iter()
        .map(|(x, y, _)| vec![vec![u64::from(*x)], vec![u64::from(*y)]])
        .collect();
    for route in [Route::Khr, Route::Native] {
        let fma = Fma { form: f, route };
        check_accept(
            &format!("fast fmod {route:?} against Metal"),
            &[f, f],
            f,
            &|fb, x| {
                let divide = |fb: &mut FunctionBuilder<'_>, p: Value, q: Value| {
                    fb.binary_with(BinOp::FDiv, p, q, Contraction::Forbidden)
                };
                let call = |fb: &mut FunctionBuilder<'_>, args: &[Value]| fma::emit(fb, fma, args);
                math::fast_fmod(fb, x[0], x[1], &divide, &call)
            },
            &cases,
            &|c| {
                let (x, y) = (arg(c, 0) as u32, arg(c, 1) as u32);
                let want = FMOD
                    .iter()
                    .find(|(a, b, _)| *a == x && *b == y)
                    .map(|(_, _, w)| *w)?;
                let denormal = |b: u32| b & 0x7F80_0000 == 0 && b & 0x007F_FFFF != 0;
                if denormal(x) || denormal(y) || want & 0x7F80_0000 == 0 {
                    return None;
                }
                let q = f32::from_bits(x) / f32::from_bits(y);
                if q.is_finite() && q.abs() >= 4_194_304.0 && !f32::from_bits(want).is_nan() {
                    return None;
                }
                Some(vec![u64::from(want)])
            },
            &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
        );
    }
}

/// `(x, rsqrt(x), fast::sqrt(x))`: Metal's fast `rsqrt` and `sqrt` at the special lanes of case
/// 08bc6611 and 40 others drawn at random.
const SQRT: [(u32, u32, u32); 52] = [
    (0x00000000, 0x7F800000, 0x00000000),
    (0x00000001, 0x7F800000, 0x00000000),
    (0x03EB9990, 0x5D3CB198, 0x21ADA837),
    (0x0AD36389, 0x59C734EB, 0x25247E12),
    (0x10BF38EE, 0x56D1729C, 0x281C7316),
    (0x2813F70D, 0x4B285D4D, 0x33C2A01C),
    (0x30125D7E, 0x47294838, 0x37C19205),
    (0x3252A670, 0x460D1B6F, 0x38E8387D),
    (0x382DB84C, 0x431B6233, 0x3BD2E272),
    (0x392CDDDE, 0x429BC440, 0x3C525DB5),
    (0x3CB5ECBD, 0x40D6BBAE, 0x3E189946),
    (0x3CD3FD00, 0x40C6ECC3, 0x3E24B9BD),
    (0x3CE1B503, 0x40C0C8DC, 0x3E29F8DA),
    (0x3CFEE21B, 0x40B56A5C, 0x3E349FC2),
    (0x3EB47B35, 0x3FD79713, 0x3F17FDFC),
    (0x3F800000, 0x3F800000, 0x3F800000),
    (0x3F800040, 0x3F7FFFC0, 0x3F800020),
    (0x3FA97BAA, 0x3F5E79AD, 0x3F9349CB),
    (0x3FB5E18D, 0x3F56C249, 0x3F989495),
    (0x3FC90FDB, 0x3F4C422A, 0x3FA06C9A),
    (0x3FCFB1F9, 0x3F48F861, 0x3FA30C8E),
    (0x3FFFC4B1, 0x3F3519EF, 0x3FB4EFFA),
    (0x4020E4ED, 0x3F217543, 0x3FCAF358),
    (0x403F7694, 0x3F14023B, 0x3FDD6471),
    (0x40490FDB, 0x3F106EBA, 0x3FE2DFC4),
    (0x41A18EC6, 0x3E63DDE9, 0x408FCDA5),
    (0x42B06B8F, 0x3DDA0EB3, 0x411645B9),
    (0x46331409, 0x3C190A90, 0x42D61CBA),
    (0x4A1C15F4, 0x3A23ED1A, 0x44C7E515),
    (0x527736DD, 0x3602412F, 0x48FB919D),
    (0x56A91948, 0x33DEBA5C, 0x4B131F06),
    (0x71A27292, 0x26633DEC, 0x589032E3),
    (0x7F7FFFFF, 0x1F800000, 0x5F7FFFFF),
    (0x7F800000, 0x00000000, 0x7F800000),
    (0x7FC00000, 0x7FC00000, 0x7FC00000),
    (0x80000000, 0xFF800000, 0x80000000),
    (0x80400000, 0xFF800000, 0x80000000),
    (0xBDFF9977, 0x7FC00000, 0x7FC00000),
    (0xBF20930B, 0x7FC00000, 0x7FC00000),
    (0xBF46AC52, 0x7FC00000, 0x7FC00000),
    (0xBF6EF8AD, 0x7FC00000, 0x7FC00000),
    (0xBF800000, 0x7FC00000, 0x7FC00000),
    (0xBF9B0F44, 0x7FC00000, 0x7FC00000),
    (0xBF9E0E31, 0x7FC00000, 0x7FC00000),
    (0xBFF56F61, 0x7FC00000, 0x7FC00000),
    (0xC017738C, 0x7FC00000, 0x7FC00000),
    (0xC0269500, 0x7FC00000, 0x7FC00000),
    (0xC22E7F33, 0x7FC00000, 0x7FC00000),
    (0xC36B16DC, 0x7FC00000, 0x7FC00000),
    (0xC48E08E1, 0x7FC00000, 0x7FC00000),
    (0xC59BB47B, 0x7FC00000, 0x7FC00000),
    (0xFF800000, 0x7FC00000, 0x7FC00000),
];

/// `fast::sqrt` (`math::fast_sqrt_of`) from Metal's own `rsqrt` is Metal's answer bit for bit:
/// `x · rsqrt(x)`, with the zero of `x`'s sign for a zero or an `f32` denormal and `+inf` for
/// `+inf`, where the product is NaN.
#[test]
fn fast_sqrt_answers_as_metal_from_its_rsqrt() {
    let f = float(32);
    let cases: Vec<Case> = SQRT
        .iter()
        .map(|(x, r, _)| vec![vec![u64::from(*x)], vec![u64::from(*r)]])
        .collect();
    check_accept(
        "fast sqrt f32 against Metal",
        &[f, f],
        f,
        &|fb, a| math::fast_sqrt_of(fb, f, a[0], a[1]),
        &cases,
        &|c| {
            let (x, r) = (arg(c, 0) as u32, arg(c, 1) as u32);
            SQRT.iter()
                .find(|(a, b, _)| *a == x && *b == r)
                .map(|(_, _, w)| vec![u64::from(*w)])
        },
        &|g, w| g == w || (is_nan(f, g) && is_nan(f, w)),
    );
}

/// The fast functions Metal composes are emitted as its compositions, read from the module:
/// `fast::exp` is one `Exp2` of a product with `float(log2 e)`, `fast::log` one `Log2` times
/// `float(ln 2)`, `fast::powr` and `fast::pow` one `Exp2` of a product with one `Log2`; none reads
/// `Exp`, `Log`, or `Pow`. `fast::sqrt` is the device's `Sqrt` where its functions are not Metal's,
/// and Metal's `x · rsqrt(x)` of one `InverseSqrt` where they are.
#[test]
fn fast_functions_are_metals_compositions() {
    const OP_CONSTANT: u32 = 43;
    const OP_EXT_INST: u32 = 12;
    const OP_FMUL: u32 = 133;
    // GLSL.std.450: Pow, Exp, Log, Exp2, Log2, Sqrt, InverseSqrt.
    const GLSL: [u32; 7] = [26, 27, 28, 29, 30, 31, 32];
    let f = float(32);
    let read_in = |op: FloatFn, lib: Lib| {
        let m = kernel(&vec![f; op.arity()], f, &|fb, a| {
            math::float_fn(fb, op, f, Contraction::Forbidden, NanTest::IsNan, a, &lib)
        })
        .expect("kernel");
        let words = crate::spirv::serialize::serialize(m).expect("words").words;
        let (mut ext, mut fmul, mut consts) = ([0u32; 7], 0, Vec::new());
        let mut i = 5;
        while i < words.len() {
            let (n, code) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
            if code == OP_EXT_INST {
                if let Some(k) = GLSL.iter().position(|g| words.get(i + 4) == Some(g)) {
                    ext[k] += 1;
                }
            }
            if code == OP_FMUL {
                fmul += 1;
            }
            if code == OP_CONSTANT && n == 4 {
                consts.push(words[i + 3]);
            }
            i += n.max(1);
        }
        (ext, fmul, consts)
    };
    let read = |op: FloatFn| read_in(op, Lib::instructions());
    let log2e = (std::f64::consts::LOG2_E as f32).to_bits();
    let ln2 = (std::f64::consts::LN_2 as f32).to_bits();
    assert_eq!((log2e, ln2), (0x3FB8_AA3B, 0x3F31_7218));
    let (ext, fmul, consts) = read(FloatFn::FastExp);
    assert_eq!((ext, fmul), ([0, 0, 0, 1, 0, 0, 0], 1));
    assert!(consts.contains(&log2e));
    let (ext, fmul, consts) = read(FloatFn::FastLog);
    assert_eq!((ext, fmul), ([0, 0, 0, 0, 1, 0, 0], 1));
    assert!(consts.contains(&ln2));
    let (ext, fmul, _) = read(FloatFn::FastPowr);
    assert_eq!((ext, fmul), ([0, 0, 0, 1, 1, 0, 0], 1));
    let (ext, _, _) = read(FloatFn::FastPow);
    assert_eq!(ext, [0, 0, 0, 1, 1, 0, 0]);
    let (ext, _, _) = read(FloatFn::Pow);
    assert_eq!(ext, [1, 0, 0, 0, 0, 0, 0]);
    let (ext, _, _) = read(FloatFn::FastSqrt);
    assert_eq!(ext, [0, 0, 0, 0, 0, 1, 0]);
    let (ext, _, _) = read_in(FloatFn::FastSqrt, Lib::new(Some(Mode::Fast), None));
    assert_eq!(ext, [0, 0, 0, 0, 0, 0, 1]);
}
