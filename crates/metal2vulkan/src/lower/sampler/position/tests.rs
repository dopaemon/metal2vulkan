//! Metal's position and the snapped coordinates, run on the interpreter against the rule computed
//! exactly. Each kernel reads `(bits(u), n)` per invocation and writes the bits of [`position`]'s
//! `X`, for one addressing, and of [`snapped_axis`]'s gather and footprint-cell coordinates reading
//! `u` as a pixel coordinate.
//!
//! The reference wraps `u` with the same `f32` operations Metal's rule names (they are exact or
//! rounded once, as on the device), forms the product `u' n` exactly in `f64` (at most 46
//! significant bits), and rounds it to `f32` with ties away from zero.

use super::{celled_by_word, gathered_by_word, offset_dropped, position, snapped_axis, AxisSnap};
use crate::air::abi::Coordinates;
use crate::air::intrinsics::ImageDim;
use crate::lower::sampler::bicubic::TapAddress;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn storage_buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

fn u32_const(f: &mut FunctionBuilder<'_>, x: u32) -> BResult<Value> {
    let c = f.u32_const(x)?;
    f.module().const_value(c)
}

/// Word `k` of invocation `i`'s record of `per` words in `buf`.
fn word(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, per: u32, k: u32) -> BResult<Value> {
    let per = u32_const(f, per)?;
    let at = f.binary(crate::spirv::ir::BinOp::IMul, i, per)?;
    let k = u32_const(f, k)?;
    let at = f.binary(crate::spirv::ir::BinOp::IAdd, at, k)?;
    let zero = u32_const(f, 0)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads `(bits(u), n)` from buffer 0 and writes `X` and, reading `u`
/// as a pixel coordinate, the gather's and the footprint cell's coordinates, as bits, to buffer 1,
/// for an axis addressed by `address` whose access the device moves by the constant texel offset
/// `offset` (0: none).
fn kernel(address: TapAddress, offset: i32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 2, 0)?;
    let ub = f.load(at, MemAccess::default())?;
    let u = f.convert(ConvOp::Bitcast, ub, f32_)?;
    let at = word(&mut f, input, i, 2, 1)?;
    let n = f.load(at, MemAccess::default())?;
    let (x, _) = position(&mut f, u, n, address)?;
    let o = if offset == 0 {
        None
    } else {
        // Metal's offsets lie in [-8, 7], each exact as an `f32`.
        Some(super::f32_const(&mut f, offset as f32)?)
    };
    let at = (address, o, None);
    let pg = snapped_axis(&mut f, AxisSnap::Gather, u, n, at)?;
    let cell = snapped_axis(&mut f, AxisSnap::Cell, u, n, at)?;
    for (k, v) in [x, pg, cell].into_iter().enumerate() {
        let bits = f.convert(ConvOp::Bitcast, v, u32_)?;
        let at = word(&mut f, output, i, 3, k as u32)?;
        f.store(at, bits, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The kernel's three words per lane `(u, n)`.
fn run(address: TapAddress, lanes: &[(f32, u32)]) -> Vec<[u32; 3]> {
    run_offset(address, 0, lanes)
}

/// The kernel's three words per lane `(u, n)`, under the constant texel offset `offset`.
fn run_offset(address: TapAddress, offset: i32, lanes: &[(f32, u32)]) -> Vec<[u32; 3]> {
    let m = kernel(address, offset).expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(u, n)| [u.to_bits().to_le_bytes(), n.to_le_bytes()])
        .flatten()
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; lanes.len() * 12],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<12>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// How [`wrap`] rounds the subtraction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounding {
    Up,
    Down,
    Even,
}

/// `u` wrapped into one period under `address`, its subtraction rounded by `rounding`.
fn wrap(u: f32, address: TapAddress, rounding: Rounding) -> f32 {
    let k = match address {
        TapAddress::Edge | TapAddress::Border => return u,
        TapAddress::Repeat => u.floor(),
        TapAddress::Mirror => {
            let fl = (u * 0.5).floor();
            fl + fl
        }
    };
    // Exact in `f64` but where `|u| < 2^-29`, where it lies just below `-k` and rounds to `-k`
    // (to nearest) either way. So `s` is a neighbour of the exact difference, which `s + k` (exact
    // in `f64`) against `u` places on one side of it or on it.
    let exact = f64::from(u) - f64::from(k);
    let s = exact as f32;
    let z = f64::from(s) + f64::from(k);
    match rounding {
        Rounding::Up if z < f64::from(u) => f32::from_bits(s.to_bits() + 1),
        Rounding::Down if z > f64::from(u) => f32::from_bits(s.to_bits() - 1),
        _ => s,
    }
}

/// `u` wrapped into one period under `address` on an axis of size `n` as Metal wraps it: in
/// `f32`, the subtraction rounded toward positive infinity, or toward negative infinity (as the
/// exact wrap reads, module docs) where `n` is a power of two.
fn wrapped(u: f32, n: u32, address: TapAddress) -> f32 {
    wrap(
        u,
        address,
        if n.is_power_of_two() {
            Rounding::Down
        } else {
            Rounding::Up
        },
    )
}

/// The exact real `x` (an `f64` holding it exactly) rounded to `f32` with ties away from zero.
fn ties_away(x: f64) -> f32 {
    let r = x as f32;
    let d = x - f64::from(r);
    if d == 0.0 || !r.is_finite() {
        return r;
    }
    // The other neighbour, on `x`'s side of `r`.
    let other = if (d > 0.0) == (r >= 0.0) {
        f32::from_bits(r.to_bits() + 1)
    } else if r == 0.0 {
        return r;
    } else {
        f32::from_bits(r.to_bits() - 1)
    };
    let e = (f64::from(other) - x).abs();
    if e == d.abs() && f64::from(other).abs() > f64::from(r).abs() {
        other
    } else {
        r
    }
}

/// Metal's position (module docs).
fn metal_x(u: f32, n: u32, address: TapAddress) -> f32 {
    ties_away(f64::from(wrapped(u, n, address)) * f64::from(n))
}

/// The position were the product rounded to nearest even (the rule before the ties).
fn even_x(u: f32, n: u32, address: TapAddress) -> f32 {
    (f64::from(wrapped(u, n, address)) * f64::from(n)) as f32
}

/// The position were the wrap rounded by `rounding` on every size.
fn x_rounded(u: f32, n: u32, address: TapAddress, rounding: Rounding) -> f32 {
    ties_away(f64::from(wrap(u, address, rounding)) * f64::from(n))
}

/// Whether positions `a` and `b` read another footprint: a gather's `i0`, a linear or bicubic
/// filter's `q`, or a nearest texel.
fn footprints_differ(a: f32, b: f32) -> bool {
    let q = |x: f32| (256.0 * f64::from(x) + 0.5).floor() as i64;
    q(a) != q(b) || a.floor() != b.floor()
}

/// The `i0` at position `x`.
fn gather_i0(x: f32) -> i64 {
    let q = (256.0 * f64::from(x) + 0.5).floor() as i64 - 128;
    q >> 8
}

/// The gather's snapped coordinate (module docs), computed from the exact `i0`.
fn gather_coordinate(x: f32, n: u32, address: TapAddress) -> f32 {
    let mut i0 = gather_i0(x);
    if matches!(address, TapAddress::Edge | TapAddress::Border) {
        i0 = i0.clamp(-10, i64::from(n) + 8);
    }
    (i0 as f32 + 1.0) / n as f32
}

const ADDRESSES: [TapAddress; 4] = [
    TapAddress::Edge,
    TapAddress::Border,
    TapAddress::Repeat,
    TapAddress::Mirror,
];

/// `j 2^k` for odd `j` with `n`'s odd part times `j` in `[2^24, 2^25)`: the product `u n` is then
/// exactly halfway between two `f32`s. Several `j` per size, at magnitudes from a fraction of a
/// period to a few periods, both signs.
fn ties(n: u32) -> Vec<f32> {
    let odd = n >> n.trailing_zeros();
    if odd == 1 {
        return Vec::new();
    }
    let lo = (1u64 << 24).div_ceil(u64::from(odd));
    let hi = ((1u64 << 25) - 1) / u64::from(odd);
    let mut js: Vec<u64> = Vec::new();
    for j in [
        lo,
        lo + 1,
        lo + 2,
        lo + 3,
        (lo + hi) / 2,
        (lo + hi) / 2 + 1,
        hi - 1,
        hi,
    ] {
        let j = j | 1;
        if j >= lo && j <= hi && !js.contains(&j) && j < (1 << 24) {
            js.push(j);
        }
    }
    let mut out = Vec::new();
    for j in js {
        for target in [0.3f64, 0.99, 1.7, 2.6] {
            // `u = j 2^k` near `target`, an exact `f32` (`j` has at most 24 bits).
            let k = (target / j as f64).log2().floor() as i32;
            let u = (j as f64 * 2f64.powi(k)) as f32;
            assert_eq!(f64::from(u), j as f64 * 2f64.powi(k));
            out.push(u);
            out.push(-u);
        }
    }
    out
}

/// Coordinates within three ulps of each footprint step from 6 texels below the level to 2 above
/// it (a gather's `i0` steps where `X` crosses `i + 1/2 - 1/512`, a nearest texel at each
/// integer) whose product `u n` is a tie the even and ties-away roundings put on either side of
/// the step (the tie probe's lanes).
fn step_ties(n: u32) -> Vec<f32> {
    let nf = f64::from(n);
    let mut steps: Vec<f64> = (-6 * 256..(i64::from(n) + 2) * 256)
        .map(|z| (2 * z - 1) as f64 / 512.0)
        .collect();
    steps.extend((-6..i64::from(n) + 2).map(|k| k as f64));
    let mut out = Vec::new();
    for step in steps {
        if step == 0.0 {
            continue;
        }
        let u0 = (step / nf) as f32;
        for d in -3i32..=3 {
            let u = f32::from_bits(u0.to_bits().wrapping_add_signed(d));
            let exact = f64::from(u) * nf;
            let (even, away) = ((exact as f32), ties_away(exact));
            if even != away
                && (gather_i0(even) != gather_i0(away) || even.floor() != away.floor())
                && !out.contains(&u)
            {
                out.push(u);
            }
        }
    }
    out
}

/// Coordinates for size `n`: texel edges and centres and a ulp either side of each, through
/// several periods both ways, every tie [`ties`] makes, and for the small sizes every tie at a
/// footprint step ([`step_ties`]).
fn lanes(n: u32) -> Vec<(f32, u32)> {
    let mut us: Vec<f32> = Vec::new();
    let nf = f64::from(n);
    for k in -2 * i64::from(n) - 3..=3 * i64::from(n) + 3 {
        for frac in [0.0, 0.5, 0.25, 1.0 / 512.0, 0.5 - 1.0 / 512.0] {
            let u = ((k as f64 + frac) / nf) as f32;
            us.push(u);
            us.push(f32::from_bits(u.to_bits().wrapping_add(1)));
            us.push(f32::from_bits(u.to_bits().wrapping_sub(1)));
        }
    }
    us.extend(ties(n));
    if n <= 255 {
        us.extend(step_ties(n));
    }
    us.retain(|u| u.is_finite());
    us.into_iter().map(|u| (u, n)).collect()
}

/// On every addressing and every size from 1 to Metal's largest, `X` is Metal's position bit for
/// bit (ties included), and the lanes include, on the clamping axes, ties where the even product
/// reads another footprint (a gather's `i0` or a nearest texel), and on the repeating and mirrored
/// axes coordinates outside one period where multiplying first reads another footprint, and where
/// the wrap's rounding does: up at a size that is not a power of two, where rounding to nearest
/// even reads another footprint, and down at a power of two, where rounding up does.
#[test]
fn the_position_is_metals_wrap_first_and_ties_away() {
    let sizes = [
        1u32, 2, 3, 5, 7, 8, 10, 12, 13, 16, 255, 256, 1000, 4095, 16383, 16384,
    ];
    for address in ADDRESSES {
        let (mut ties_seen, mut wraps_seen) = (0usize, 0usize);
        let (mut ups_seen, mut downs_seen) = (0usize, 0usize);
        for n in sizes {
            // Every tie, and up to 200 lanes where the wrap's rounding up or down decides; the
            // other lanes sampled to at most about 1500 per size.
            let special: Vec<f32> = ties(n)
                .into_iter()
                .chain(if n <= 255 { step_ties(n) } else { Vec::new() })
                .collect();
            let all = lanes(n);
            let step = (all.len() / 1500).max(1);
            let decides = |u: f32| {
                footprints_differ(
                    x_rounded(u, n, address, Rounding::Up),
                    x_rounded(u, n, address, Rounding::Down),
                )
            };
            let mut deciding = 0;
            let picked: Vec<(f32, u32)> = all
                .iter()
                .enumerate()
                .filter(|(k, (u, _))| {
                    let take = deciding < 200 && decides(*u);
                    deciding += usize::from(take);
                    take || k % step == 0 || special.contains(u)
                })
                .map(|(_, l)| *l)
                .collect();
            let out = run(address, &picked);
            for (&(u, n), got) in picked.iter().zip(&out) {
                let want = metal_x(u, n, address);
                assert_eq!(
                    got[0],
                    want.to_bits(),
                    "{address:?} n {n} u {u:e} ({:#010x}): X {:e} vs {want:e}",
                    u.to_bits(),
                    f32::from_bits(got[0])
                );
                let even = even_x(u, n, address);
                if gather_i0(want) != gather_i0(even) || want.floor() != even.floor() {
                    ties_seen += 1;
                }
                if n.is_power_of_two() {
                    let up = x_rounded(u, n, address, Rounding::Up);
                    downs_seen += usize::from(footprints_differ(want, up));
                } else {
                    let nearest_even = x_rounded(u, n, address, Rounding::Even);
                    ups_seen += usize::from(footprints_differ(want, nearest_even));
                }
                let first = (f64::from(u) * f64::from(n)) as f32;
                let period = match address {
                    TapAddress::Mirror => 2 * i64::from(n),
                    _ => i64::from(n),
                };
                if matches!(address, TapAddress::Repeat | TapAddress::Mirror)
                    && (gather_i0(first) - gather_i0(want)).rem_euclid(period) != 0
                {
                    wraps_seen += 1;
                }
            }
        }
        // A tie moves a footprint only at a negative position, or at one of `2^14` or more (where
        // a step of `i0` is odd in `f32`): elsewhere the even neighbour is the step itself
        // (module docs). A repeating or mirrored axis's position is never negative.
        if matches!(address, TapAddress::Repeat | TapAddress::Mirror) {
            assert!(
                wraps_seen > 0,
                "{address:?}: no lane where the wrap decides"
            );
            assert!(
                ups_seen > 0,
                "{address:?}: no lane where the wrap's rounding up decides"
            );
            assert!(
                downs_seen > 0,
                "{address:?}: no lane where a power of two's rounding down decides"
            );
        } else {
            assert!(ties_seen > 0, "{address:?}: no lane where the ties decide");
        }
    }
}

/// The integer tie test's bound, `n < 2^22`: at `n = 2^22 - 1` every tie on every addressing is
/// Metal's.
#[test]
fn the_ties_hold_up_to_the_size_bound() {
    let n = (1u32 << 22) - 1;
    let us: Vec<(f32, u32)> = ties(n).into_iter().map(|u| (u, n)).collect();
    assert!(!us.is_empty());
    for address in ADDRESSES {
        let out = run(address, &us);
        let mut decided = 0;
        for (&(u, n), got) in us.iter().zip(&out) {
            let want = metal_x(u, n, address);
            assert_eq!(got[0], want.to_bits(), "{address:?} n {n} u {u:e}");
            decided += usize::from(want != even_x(u, n, address));
        }
        assert!(decided > 0, "{address:?}: no tie rounded the other way");
    }
}

/// A pixel coordinate `x` is Metal's position as given. Its gather's snapped coordinate is the
/// gather's at `X = x`, and its footprint cell's, `(q + 128) / 256 / n` for `q = floor(256 x -
/// 127.5)`, reads as a normalized coordinate (Metal's own sampler's position of it, ties away from
/// zero) the same `q`: every lane a ulp either side of each step and texel edge and centre, on the
/// clamping axes (a pixel sampler only clamps) of Metal's sizes.
#[test]
fn pixel_coordinate_snaps_read_metals_footprint() {
    let q = |x: f32| (256.0 * f64::from(x) - 127.5).floor() as i64;
    for address in [TapAddress::Edge, TapAddress::Border] {
        for n in [1u32, 2, 3, 13, 255, 1000, 16383, 16384] {
            let mut xs = Vec::new();
            let top = i64::from(n).min(40) + 12;
            for k in -12..=top {
                for base in [
                    k as f32,
                    k as f32 + 0.5 - 1.0 / 512.0,
                    k as f32 + 0.5 - 1.0 / 256.0,
                ] {
                    xs.extend([base, base.next_down(), base.next_up(), base + 1.0 / 1024.0]);
                }
            }
            for k in [n - 1, n, n + 1] {
                let b = k as f32 - 0.5 - 1.0 / 512.0;
                xs.extend([b, b.next_down(), b.next_up(), k as f32 - 0.25]);
            }
            let lanes: Vec<(f32, u32)> = xs.iter().map(|x| (*x, n)).collect();
            let out = run(address, &lanes);
            for (&(x, n), got) in lanes.iter().zip(&out) {
                let g = gather_coordinate(x, n, address);
                assert_eq!(got[1], g.to_bits(), "{address:?} n {n} x {x:e}: gather");
                let c = f32::from_bits(got[2]);
                assert_eq!(
                    q(metal_x(c, n, address)),
                    q(x),
                    "{address:?} n {n} x {x:e}: cell {c:e}"
                );
            }
        }
    }
}

/// A kernel whose invocation `i` reads `(n, base, parent)` from buffer 0 and writes
/// [`offset_dropped`]'s answer for a level of size `n` under a base level of size `base`, and its
/// base-level form's for a base level of size `n`, each without a parent test and under the parent
/// test `parent != 0`, as `0` or `1`, to buffer 1.
fn rule_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 3, 0)?;
    let n = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 1)?;
    let base = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 2)?;
    let parent = f.load(at, MemAccess::default())?;
    let one = u32_const(&mut f, 1)?;
    let zero = u32_const(&mut f, 0)?;
    let parent = f.binary(crate::spirv::ir::BinOp::INotEqual, parent, zero)?;
    let forms = [None, Some(parent)]
        .into_iter()
        .flat_map(|p| [(Some(base), p), (None, p)]);
    for (k, (base, parent)) in forms.enumerate() {
        let dropped = offset_dropped(&mut f, n, base, parent)?;
        let bit = f.select(dropped, one, zero)?;
        let at = word(&mut f, output, i, 4, k as u32)?;
        f.store(at, bit, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The rule kernel's four answers per lane `(n, base)`, under no parent test.
fn run_rule(lanes: &[(u32, u32)]) -> Vec<[u32; 4]> {
    let lanes: Vec<(u32, u32, u32)> = lanes.iter().map(|&(n, b)| (n, b, 0)).collect();
    run_parent_rule(&lanes)
}

/// The rule kernel's four answers per lane `(n, base, parent)`.
fn run_parent_rule(lanes: &[(u32, u32, u32)]) -> Vec<[u32; 4]> {
    let m = rule_kernel().expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(n, b, p)| [n.to_le_bytes(), b.to_le_bytes(), p.to_le_bytes()])
        .flatten()
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; lanes.len() * 16],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// Whether Metal dropped a constant texel offset on a repeating or mirrored axis whose level size
/// is `.0` under a base level of size `.1`: the offset probe, round 3 (Apple M4,
/// macOS 27.0.1), every level of 18 bases at an explicit level and through a view of one level,
/// under `repeat` and `mirrored_repeat` alike, offsets -8 to 7.
const MEASURED_RULE: [(u32, u32, bool); 85] = [
    (2, 2, false),
    (2, 4, false),
    (2, 5, true),
    (2, 8, false),
    (2, 9, true),
    (2, 10, true),
    (2, 11, true),
    (2, 16, false),
    (2, 17, true),
    (2, 18, true),
    (2, 20, true),
    (2, 32, false),
    (2, 33, true),
    (2, 36, true),
    (2, 40, true),
    (2, 64, false),
    (2, 66, true),
    (3, 3, true),
    (3, 6, true),
    (3, 7, true),
    (3, 12, true),
    (3, 13, true),
    (3, 24, true),
    (3, 48, true),
    (4, 4, false),
    (4, 8, false),
    (4, 9, true),
    (4, 16, false),
    (4, 17, true),
    (4, 18, true),
    (4, 32, false),
    (4, 33, true),
    (4, 36, true),
    (4, 64, false),
    (4, 66, true),
    (5, 5, true),
    (5, 10, true),
    (5, 11, true),
    (5, 20, true),
    (5, 40, true),
    (6, 6, true),
    (6, 12, true),
    (6, 13, true),
    (6, 24, true),
    (6, 48, true),
    (7, 7, true),
    (8, 8, false),
    (8, 16, false),
    (8, 17, false),
    (8, 32, false),
    (8, 33, false),
    (8, 64, false),
    (8, 66, false),
    (9, 9, false),
    (9, 18, false),
    (9, 36, false),
    (10, 10, false),
    (10, 20, false),
    (10, 40, false),
    (11, 11, false),
    (12, 12, false),
    (12, 24, false),
    (12, 48, false),
    (13, 13, false),
    (16, 16, false),
    (16, 32, false),
    (16, 33, false),
    (16, 64, false),
    (16, 66, false),
    (17, 17, false),
    (18, 18, false),
    (18, 36, false),
    (20, 20, false),
    (20, 40, false),
    (24, 24, false),
    (24, 48, false),
    (32, 32, false),
    (32, 64, false),
    (33, 33, false),
    (33, 66, false),
    (36, 36, false),
    (40, 40, false),
    (48, 48, false),
    (64, 64, false),
    (66, 66, false),
];

/// [`offset_dropped`] is Metal's rule, a level size from 2 to 7 under a base-level size that is not
/// a power of two: on every pair round 3 measured, on every pair of a level size up to 64 and a
/// base up to 600, and its base-level form (sizes 3, 5, 6 and 7) wherever the level is the base.
#[test]
fn the_offset_rule_is_metals_by_level_and_base() {
    let rule = |n: u32, base: u32| (2..=7).contains(&n) && !base.is_power_of_two();
    for &(n, base, dropped) in &MEASURED_RULE {
        assert_eq!(rule(n, base), dropped, "the measured pair ({n}, {base})");
    }
    let mut lanes: Vec<(u32, u32)> = MEASURED_RULE.iter().map(|&(n, b, _)| (n, b)).collect();
    for base in 1u32..=600 {
        let mut n = base;
        loop {
            lanes.push((n, base));
            if n == 1 {
                break;
            }
            n /= 2;
        }
    }
    lanes.extend((0u32..=70).map(|n| (n, n)));
    lanes.extend([(u32::MAX, u32::MAX), (1 << 14, 1 << 14), (16383, 16383)]);
    let out = run_rule(&lanes);
    for (&(n, base), got) in lanes.iter().zip(&out) {
        assert_eq!(got[0], u32::from(rule(n, base)), "level {n} of base {base}");
        let at_base = matches!(n, 3 | 5 | 6 | 7);
        assert_eq!(got[1], u32::from(at_base), "base level {n}");
        if n == base && n != 0 {
            assert_eq!(rule(n, n), at_base, "the base-level form of {n}");
        }
    }
}

/// The residue: a view whose first level lies `level` levels above its parent's base reads Metal's
/// rule by its parent's base-level size. Under the bound texture's `TextureFacts::offset_base_axes`
/// bit (the binder's `SpecKey::OffsetBaseAxes`), [`offset_dropped`] of each of the view's levels,
/// by the view's own base-level size and in its base-level form, answers as Metal's rule by the
/// parent's, on every parent size up to 300 and view level up to 9; without the bit it reads the
/// view's own, as before.
#[test]
fn the_offset_rule_reads_the_parent_s_base_through_the_view_bit() {
    use crate::bind::TextureFacts;
    use crate::metal::{PixelFormat, TextureType};
    let rule = |n: u32, base: u32| (2..=7).contains(&n) && !base.is_power_of_two();
    let mut lanes = Vec::new();
    let mut cases = Vec::new();
    for parent in 1u32..=300 {
        for level in 0..10u32 {
            let own = (parent >> level).max(1);
            let mut facts = TextureFacts::new(TextureType::Type1D, PixelFormat::RGBA8Unorm);
            facts.parent_relative_level = level;
            facts.parent_base_size = [parent, 0, 0];
            facts.mipmap_level_count = 32 - own.leading_zeros();
            let bit = u32::from(facts.offset_base_axes() & 1);
            let mut n = own;
            loop {
                lanes.push((n, own, bit));
                cases.push((parent, level));
                if n == 1 {
                    break;
                }
                n /= 2;
            }
        }
    }
    let out = run_parent_rule(&lanes);
    for ((&(n, own, bit), &(parent, level)), got) in lanes.iter().zip(&cases).zip(&out) {
        let at = format!("level {n} of a view at {level} of {parent} (own base {own}, bit {bit})");
        let metal = u32::from(rule(n, parent));
        assert_eq!(got[2], metal, "{at}");
        if n == own {
            assert_eq!(got[3], metal, "{at}: the base-level form");
        }
        assert_eq!(got[0], u32::from(rule(n, own)), "{at}: no parent test");
    }
}

/// On a repeating or mirrored axis whose size is 3, 5, 6 or 7, Metal drops a texel offset the
/// device adds, so each snapped coordinate (reading `u` as a pixel coordinate) moves back by the
/// offset there: the device's index at a gather's (`floor(c n - 1/2)`), plus the offset, is the
/// unmoved snap's, and the footprint cell moves back the same way. On every other size, and on
/// every clamping axis, the offset leaves the snap as it is (Metal adds it as the device does).
#[test]
fn a_snap_moves_back_by_the_offset_metal_drops() {
    let mut moved = 0usize;
    for address in ADDRESSES {
        for n in (1u32..=10).chain([13, 16, 255]) {
            let all = lanes(n);
            let step = (all.len() / 200).max(1);
            let picked: Vec<(f32, u32)> = all.into_iter().step_by(step).collect();
            let plain = run(address, &picked);
            let dropped = matches!(address, TapAddress::Repeat | TapAddress::Mirror)
                && matches!(n, 3 | 5 | 6 | 7);
            for o in [-8i32, -1, 1, 7] {
                let out = run_offset(address, o, &picked);
                let back = if dropped { o as f32 } else { 0.0 };
                let nf = n as f32;
                for ((&(u, _), got), was) in picked.iter().zip(&out).zip(&plain) {
                    let at = format!("{address:?} n {n} o {o} u {u:e}");
                    if !dropped {
                        assert_eq!(got, was, "{at}: moved where Metal adds the offset");
                        continue;
                    }
                    assert_eq!(got[0], was[0], "{at}: X");
                    let pg = (gather_i0(u) as f32 + 1.0 - back) / nf;
                    assert_eq!(got[1], pg.to_bits(), "{at}: pixel gather");
                    let y = 256.0 * u;
                    let fl = y.floor();
                    let cell = (fl + if y - fl >= 0.5 { 1.0 } else { 0.0 }) * (1.0 / 256.0);
                    let cell = (cell - back) / nf;
                    assert_eq!(got[2], cell.to_bits(), "{at}: cell");
                    // The device's index at the moved coordinate, plus the offset, is the
                    // unmoved one's.
                    let index = |bits: u32| {
                        let y = f64::from(f32::from_bits(bits)) * f64::from(n);
                        (y - 0.5).floor() as i64
                    };
                    assert_eq!(index(got[1]) + i64::from(o), index(was[1]), "{at}");
                    moved += 1;
                }
            }
        }
    }
    assert!(moved > 0, "no lane moved back");
}

/// The coordinate the word kernels hand the device where their snap does not reach a lane: a
/// marker no snap yields.
const DEVICE: f32 = -12345.0;

/// A kernel whose invocation `i` reads `(bits(u), n, word)` from buffer 0 and writes the bits of
/// [`gathered_by_word`]'s coordinate of a one-axis gather of size `n` through a runtime sampler of
/// word `word`, and of [`celled_by_word`]'s, each handing [`DEVICE`] as the coordinate the device
/// reads otherwise, to buffer 1.
fn word_gather_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 3, 0)?;
    let ub = f.load(at, MemAccess::default())?;
    let u = f.convert(ConvOp::Bitcast, ub, f32_)?;
    let at = word(&mut f, input, i, 3, 1)?;
    let n = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 2)?;
    let w = f.load(at, MemAccess::default())?;
    let device = super::f32_const(&mut f, DEVICE)?;
    let g = gathered_by_word(&mut f, w, (u, device), 1, (n, 1))?;
    let c = celled_by_word(&mut f, w, (u, device), 1, (n, 1))?;
    for (k, v) in [g, c].into_iter().enumerate() {
        let bits = f.convert(ConvOp::Bitcast, v, u32_)?;
        let at = word(&mut f, output, i, 2, k as u32)?;
        f.store(at, bits, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The word gather kernel's two coordinates' bits per lane `(u, n, word)`.
fn run_word_gather(lanes: &[(f32, u32, u32)]) -> Vec<[u32; 2]> {
    let m = word_gather_kernel().expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(u, n, w)| [u.to_bits().to_le_bytes(), n.to_le_bytes(), w.to_le_bytes()])
        .flatten()
        .collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; lanes.len() * 8],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// A runtime sampler's gather hands the device the centre of the 2x2 block Metal reads at the
/// coordinate as given where its word states pixel coordinates, and the device's coordinate for a
/// normalized word; a word's other fields leave it as it is. The footprint-cell snap takes the
/// centre of the step Metal reads the coordinate as given at where the word states a linear pixel
/// sampler, and the device's coordinate for every other word.
#[test]
fn a_runtime_samplers_gather_centres_metals_block_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let modes = [
        M::ClampToEdge,
        M::MirrorClampToEdge,
        M::Repeat,
        M::MirrorRepeat,
        M::ClampToZero,
        M::ClampToBorderColor,
    ];
    // Fields the gather ignores: the filters, the mip filter, the `t` and `r` modes, the border,
    // the comparison, the reduction, anisotropy and the level clamps.
    let ignored = SamplerWord::MIN_LINEAR
        | SamplerWord::MAG_LINEAR
        | SamplerWord::MIP_FILTER
        | SamplerWord::ADDRESS[1]
        | SamplerWord::ADDRESS[2]
        | SamplerWord::BORDER
        | SamplerWord::COMPARE
        | SamplerWord::REDUCTION
        | SamplerWord::ANISOTROPIC
        | SamplerWord::LEVEL_MIN
        | SamplerWord::LEVEL_MAX;
    let linear_pixel =
        SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR;
    let mut checked = 0usize;
    for n in [1u32, 3, 7, 13, 16, 255, 16383] {
        let all = lanes(n);
        let step = (all.len() / 60).max(1);
        let us: Vec<f32> = all.into_iter().step_by(step).map(|(u, _)| u).collect();
        let mut picked = Vec::new();
        for mode in modes {
            let m = mode.as_raw() << SamplerWord::ADDRESS_SHIFT[0];
            for w in [
                m,
                m | ignored,
                m | SamplerWord::PIXEL_COORDINATES,
                m | linear_pixel,
                m | SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR,
            ] {
                picked.extend(us.iter().map(|&u| (u, n, w)));
            }
        }
        // Pixel coordinates around the steps and edges of the texels near the level.
        for k in -12i64..=i64::from(n.min(40)) + 12 {
            for base in [k as f32, k as f32 + 0.5 - 1.0 / 512.0, k as f32 + 0.25] {
                for x in [base, base.next_down(), base.next_up()] {
                    for w in [SamplerWord::PIXEL_COORDINATES, linear_pixel] {
                        picked.push((x, n, w));
                    }
                }
            }
        }
        let out = run_word_gather(&picked);
        let nf = n as f32;
        for (&(u, n, w), got) in picked.iter().zip(&out) {
            let at = format!("n {n} word {w:#x} u {u:e}");
            let pixel = w & SamplerWord::PIXEL_COORDINATES != 0;
            let snapped = pixel.then_some(u);
            let want = match snapped {
                Some(x) => {
                    let i0 = gather_i0(x).clamp(-2 * i64::from(n) - 10, 2 * i64::from(n) + 8);
                    (i0 as f32 + 1.0) / nf
                }
                None => DEVICE,
            };
            assert_eq!(
                got[0],
                want.to_bits(),
                "{at}: gather {:e} vs {want:e}",
                f32::from_bits(got[0])
            );
            if let Some(x) = snapped {
                // The device reads the block at `i0`, as Metal does.
                let c = f64::from(f32::from_bits(got[0])) * f64::from(n);
                let i0 = gather_i0(x).clamp(-2 * i64::from(n) - 10, 2 * i64::from(n) + 8);
                assert!((c - (i0 + 1) as f64).abs() <= 2f64.powi(-7) + 8.0, "{at}");
            }
            let cell = if w & linear_pixel == linear_pixel {
                let y = 256.0 * u;
                let fl = y.floor();
                ((fl + if y - fl >= 0.5 { 1.0 } else { 0.0 }) * (1.0 / 256.0)) / nf
            } else {
                DEVICE
            };
            assert_eq!(
                got[1],
                cell.to_bits(),
                "{at}: cell {:e} vs {cell:e}",
                f32::from_bits(got[1])
            );
            checked += 1;
        }
    }
    assert!(checked > 15_000, "{checked}");
}

/// A kernel whose invocation `i` reads the bits of a two-lane coordinate from buffer 0 and writes
/// the bits of [`super::emit_flush`]'s result of `axes` to buffer 1.
fn flush_kernel(axes: u8) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let fvec2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let mut lanes = Vec::new();
    for lane in 0..2 {
        let at = word(&mut f, input, i, 2, lane)?;
        let bits = f.load(at, MemAccess::default())?;
        lanes.push(f.convert(ConvOp::Bitcast, bits, f32_)?);
    }
    let coord = f.composite_construct(fvec2, &lanes)?;
    let flushed = super::emit_flush(&mut f, coord, 2, axes)?;
    for lane in 0..2u32 {
        let c = f.composite_extract(flushed, &[lane])?;
        let bits = f.convert(ConvOp::Bitcast, c, u32_)?;
        let at = word(&mut f, output, i, 2, lane)?;
        f.store(at, bits, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// `bits` with a subnormal magnitude replaced by the zero of its sign, Metal's reading of a
/// subnormal texture coordinate.
fn flushed_bits(bits: u32) -> u32 {
    match bits & 0x7fff_ffff < 0x0080_0000 {
        true => bits & 0x8000_0000,
        false => bits,
    }
}

/// The flush replaces every subnormal magnitude by the zero of its sign on the axes it names and
/// keeps every other lane and axis exactly, for each of the four masks of two axes. The lanes
/// cover the whole deciding region exhaustively: every subnormal whose significand is a power of
/// two or one off it, of both signs, the two floats either side of `f32::MIN_POSITIVE`, the
/// magnitudes the Metal measurement read (`1e-30`, `1e-37`, `1.2e-38`, `1e-40`, `1e-42`,
/// `1e-44`), both zeros, and normals that must be kept.
#[test]
fn the_flush_reads_a_subnormal_coordinate_as_the_zero_of_its_sign() {
    let mut magnitudes: Vec<u32> = Vec::new();
    for e in 0..24u32 {
        for d in [0u32, 1] {
            let one = 1u32 << e;
            magnitudes.push(one + d);
            magnitudes.push(one.saturating_sub(d));
        }
    }
    magnitudes.extend([
        0x0000_0000,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        1.2e-38f32.to_bits(),
        1e-30f32.to_bits(),
        1e-37f32.to_bits(),
        1e-40f32.to_bits(),
        1e-42f32.to_bits(),
        1e-44f32.to_bits(),
        1.0f32.to_bits(),
        0.375f32.to_bits(),
        f32::MAX.to_bits(),
    ]);
    magnitudes.sort_unstable();
    magnitudes.dedup();
    let lanes: Vec<u32> = magnitudes
        .iter()
        .flat_map(|&m| [m, m | 0x8000_0000])
        .collect();
    let mut checked = 0usize;
    for axes in 0..4u8 {
        let m = flush_kernel(axes).expect("kernel");
        // Each invocation takes one lane on axis 0 and its reverse on axis 1, so both axes see
        // every value under every mask.
        let pairs: Vec<(u32, u32)> = lanes
            .iter()
            .copied()
            .zip(lanes.iter().rev().copied())
            .collect();
        let bytes: Vec<u8> = pairs
            .iter()
            .flat_map(|&(s, t)| [s.to_le_bytes(), t.to_le_bytes()])
            .flatten()
            .collect();
        let res = Resources {
            buffers: vec![
                Buffer {
                    slot: Some(slot(0)),
                    address: None,
                    bytes,
                },
                Buffer {
                    slot: Some(slot(1)),
                    address: None,
                    bytes: vec![0xAB; pairs.len() * 8],
                },
            ],
            ..Resources::default()
        };
        let cfg = Config {
            workgroups: [u32::try_from(pairs.len()).expect("lanes"), 1, 1],
            ..Config::default()
        };
        let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
        for (&(s, t), got) in pairs.iter().zip(out.buffers[1].as_chunks::<8>().0.iter()) {
            let read = |k: usize| u32::from_le_bytes(got[4 * k..4 * k + 4].try_into().unwrap());
            for (axis, given) in [(0usize, s), (1, t)] {
                let want = match axes & (1 << axis) {
                    0 => given,
                    _ => flushed_bits(given),
                };
                assert_eq!(
                    read(axis),
                    want,
                    "axes {axes:#b} axis {axis} given {given:#010x}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 700, "only {checked} lanes checked");
}

/// The plan names exactly the border axes of a constexpr sampler with normalized coordinates, of
/// a single-sampled 1D, 2D or 3D texture or array: a repeating, mirrored or edge-clamping axis,
/// an axis past the texture's dimensions, a pixel sampler, a cube, a multisampled texture and an
/// argument buffer's sampler take no flush.
#[test]
fn the_flush_names_the_border_axes_of_a_constexpr_normalized_sampler() {
    use crate::air::abi::{AddressMode as M, SamplerState};
    use crate::air::intrinsics::ImageShape;
    use crate::lower::sampler::constexpr::state_words;
    use crate::pipeline::sym::SamplerRef;
    let modes = [
        (M::ClampToEdge, false),
        (M::Repeat, false),
        (M::MirroredRepeat, false),
        (M::ClampToZero, true),
        (M::ClampToBorder, true),
    ];
    let shape = |dim, array, ms| ImageShape {
        dim,
        array,
        ms,
        depth: false,
    };
    let mut checked = 0usize;
    for (s_mode, s_border) in modes {
        for (t_mode, t_border) in modes {
            for (r_mode, r_border) in modes {
                let mut state =
                    SamplerState::from_words([0, 0]).expect("a decodable sampler state");
                state.address_s = s_mode;
                state.address_t = t_mode;
                state.address_r = r_mode;
                let words = state_words(&state);
                let constexpr = SamplerRef::Constexpr(words);
                let borders = [s_border, t_border, r_border];
                let mask = |axes: usize| {
                    borders
                        .iter()
                        .take(axes)
                        .enumerate()
                        .fold(0u8, |m, (a, &b)| m | u8::from(b) << a)
                };
                for (dim, axes) in [(ImageDim::D1, 1usize), (ImageDim::D2, 2), (ImageDim::D3, 3)] {
                    for array in [false, true] {
                        let got = super::subnormal_flush(
                            Some((constexpr, Coordinates::Normalized)),
                            shape(dim, array, false),
                        )
                        .expect("flush");
                        assert_eq!(got, mask(axes), "{dim:?} array {array} {s_mode:?}");
                        checked += 1;
                    }
                    // A multisampled texture of the same modes takes none.
                    let ms = super::subnormal_flush(
                        Some((constexpr, Coordinates::Normalized)),
                        shape(dim, false, true),
                    )
                    .expect("flush");
                    assert_eq!(ms, 0, "{dim:?} multisampled");
                }
                // A pixel sampler's coordinate is in texels, a cube's is a direction, and an
                // argument buffer's sampler states no mode in the function.
                let pixel = super::subnormal_flush(
                    Some((constexpr, Coordinates::Pixel)),
                    shape(ImageDim::D2, false, false),
                )
                .expect("flush");
                assert_eq!(pixel, 0, "pixel coordinates");
                let cube = super::subnormal_flush(
                    Some((constexpr, Coordinates::Normalized)),
                    shape(ImageDim::Cube, false, false),
                )
                .expect("flush");
                assert_eq!(cube, 0, "a cube");
                let heap =
                    super::subnormal_flush(None, shape(ImageDim::D2, false, false)).expect("flush");
                assert_eq!(heap, 0, "an argument buffer's sampler");
                let slot = super::subnormal_flush(
                    Some((SamplerRef::Slot(0), Coordinates::Normalized)),
                    shape(ImageDim::D2, false, false),
                )
                .expect("flush");
                assert_eq!(slot, 0, "a runtime sampler");
            }
        }
    }
    assert!(checked > 500, "only {checked} shapes checked");
}
