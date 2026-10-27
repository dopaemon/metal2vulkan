//! `snapped` on the interpreter against Metal's position (the cube rounding probes, Apple M3 Max
//! and M4), and its wrap of a repeating face axis (the cube probes, Apple M4). Each kernel
//! invocation reads a direction's bits and a face size, hands the direction through `metal_ties`
//! and then `snapped`, and writes the handed direction's bits; a device reading it in exact
//! arithmetic, or rounding each operation to `f32`, must read the texel Metal's position names with
//! the device's reciprocal (here the interpreter's correctly rounded one).

use super::super::{snapped, FaceRoutines, FaceWrap, SnapRoutines, SCALAR, SNAP_WORDS};
use super::{
    kernel_on, lanes, metal_axis, metal_ties, read_as_zero, run_records, selecting, SIZES,
};
use crate::lower::arith::fma::{self, Fma, Route};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::TypedModule;

/// The device's reciprocal of a major magnitude `top`, as the interpreter divides: `RN(1 / top)`.
fn device_recip(top: f32) -> f32 {
    1.0 / top
}

/// A kernel whose invocation reads a direction and a face size and writes the direction that
/// `metal_ties` and then (`snap`) `snapped` hand the device, by `fma`'s route, wrapping the face
/// axes `wrap` names.
fn kernel(fma: Fma, snap: bool, wrap: FaceWrap) -> BResult<TypedModule> {
    let b = ModuleBuilder::new();
    let r = SnapRoutines { fma };
    kernel_on(b, 4, &|f, p, n| {
        let p = metal_ties(f, p)?;
        match snap {
            true => snapped(f, p, n, (r, wrap)),
            false => Ok(p),
        }
    })
}

/// The `float` fused multiply-add by `route`.
fn by(route: Route) -> Fma {
    let routes = fma::Routes {
        float: route,
        half: Route::Native,
    };
    Fma::by(routes, SCALAR).expect("float")
}

/// The handed direction of each lane, a direction and a face size, by `fma`'s route, wrapping
/// the face axes `wrap` names.
fn run(fma: Fma, lanes: &[([f32; 3], u32)], wrap: FaceWrap) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes
        .iter()
        .flat_map(|(d, n)| [d[0].to_bits(), d[1].to_bits(), d[2].to_bits(), *n])
        .collect();
    run_records(
        &kernel(fma, true, wrap).expect("kernel"),
        &records,
        lanes.len(),
    )
}

/// Vulkan's face coordinates `(s_c, t_c)` of direction `d` on the face of major axis `major`
/// (`positive`: its sign), by its "Cube Map Face Selection" table.
fn face_minors(d: [f32; 3], major: usize, positive: bool) -> [f32; 2] {
    match (major, positive) {
        (0, true) => [-d[2], -d[1]],
        (0, false) => [d[2], -d[1]],
        (1, true) => [d[0], d[2]],
        (1, false) => [d[0], -d[2]],
        (2, true) => [d[0], -d[1]],
        _ => [-d[0], -d[1]],
    }
}

/// A face (its major axis and sign) and the texel on each of its axes.
type Texel = (usize, bool, [u32; 2]);

/// The texel `floor(x)`, clamped into a face of size `n`.
fn clamped(x: f64, n: u32) -> u32 {
    x.floor().clamp(0.0, f64::from(n - 1)) as u32
}

/// Metal's position `X` on a face axis of size `n` of face coordinate `c` beside a major
/// magnitude whose reciprocal is `r`: `RN(RN(1 + c * r) / 2 * n)`, the sum fused and rounded once,
/// ties to even; on a repeating axis (`wraps`) of a size that is not a power of two, a face
/// coordinate `RN(1 + c * r) / 2` of 1 or more less 1 first.
fn metal_x(c: f32, r: f32, n: u32, wraps: bool) -> f32 {
    let u = c.mul_add(r, 1.0) * 0.5;
    let u = match wraps && !n.is_power_of_two() && u >= 1.0 {
        true => u - 1.0,
        false => u,
    };
    u * n as f32
}

/// Metal's nearest texel along a finite direction `d` on faces of size `n`, its reciprocal of the
/// major magnitude the device's ([`device_recip`]): the face `metal_reading` reads, and on each
/// face axis `floor(X)` of [`metal_x`] (wrapped on the axes `wrap` names), at most `n - 1`, with
/// each component Metal reads as zero `0`; a direction left zero at face `+x`'s texel `(0, 0)`.
fn metal_texel(d: [f32; 3], n: u32, wrap: FaceWrap) -> Texel {
    let m = selecting(d);
    let major = metal_axis(m);
    let top = m[major];
    if top == 0.0 {
        return (0, true, [0, 0]);
    }
    let read = std::array::from_fn(|k| match k != major && read_as_zero(d[k], top) {
        true => 0.0,
        false => d[k],
    });
    let positive = d[major] > 0.0;
    let r = device_recip(top);
    let [s, t] = face_minors(read, major, positive);
    let texels = [(s, wrap.s), (t, wrap.t)].map(|(c, w)| clamped(metal_x(c, r, n, w).into(), n));
    (major, positive, texels)
}

/// What a device reads along the handed direction `q`, whose major magnitude is `1` and alone the
/// largest: its face, and on each face axis the texel of `(s_c / |m_a| + 1) / 2 * n`, in exact
/// arithmetic (`exact`) or with each operation rounded to `f32`.
fn device_texel(q: [f32; 3], n: u32, exact: bool) -> Texel {
    let a = q.map(f32::abs);
    let major = metal_axis(a);
    for k in (0..3).filter(|&k| k != major) {
        assert!(a[k] < a[major], "{q:?}: axis {k} ties {major}");
    }
    assert_eq!(a[major], 1.0, "{q:?}: the major magnitude");
    let positive = q[major] > 0.0;
    let texels = face_minors(q, major, positive).map(|c| {
        let x = match exact {
            true => (f64::from(c) + 1.0) / 2.0 * f64::from(n),
            false => f64::from((c / a[major] + 1.0) / 2.0 * n as f32),
        };
        clamped(x, n)
    });
    (major, positive, texels)
}

/// A deterministic stream of `u32`s (a 64-bit linear congruential generator's high half).
fn stream(mut seed: u64) -> impl FnMut() -> u32 {
    move || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (seed >> 32) as u32
    }
}

/// Directions whose face coordinate lies at a texel step of Metal's position or a few ulps either
/// side, on several majors and sizes: `s_c = (2k / n - 1) * |m_a|` per step `k`, moved by up to
/// three ulps, on face `+x` (`s_c = -z`) with `t_c` the same on `-y`.
fn steps() -> Vec<([f32; 3], u32)> {
    let mut next = stream(1017);
    let mut majors = vec![
        1.0f32,
        f32::from_bits(0x3f80_0001),
        3.0,
        0.885,
        1.0e-30,
        1.0e30,
        f32::from_bits(0x7e00_0000),
    ];
    majors.extend((0..24).map(|_| f32::from_bits(0x3f80_0000 | (next() >> 9))));
    let mut out = Vec::new();
    for n in [5u32, 7, 8, 13, 255, 1024, 16_384] {
        let ks: Vec<u32> = (1..n).step_by((n as usize / 6).max(1)).collect();
        for &ma in &majors {
            for &k in &ks {
                let c = (2.0 * f64::from(k) / f64::from(n) - 1.0) * f64::from(ma);
                let c = c as f32;
                for d in -3i32..=3 {
                    let moved = f32::from_bits(c.to_bits().wrapping_add_signed(d));
                    if moved.abs() < ma {
                        out.push(([ma, moved, -moved], n));
                    }
                }
            }
        }
    }
    out
}

/// The lane: along `(1, 1 + 2^-23, 0.25)` Metal reads texel `(7, 5)` of face `+y` of an 8x8 cube,
/// where ANV and NVIDIA read `(7, 4)`.
const TODO_1_14: ([f32; 3], u32) = ([1.0, f32::from_bits(0x3f80_0001), 0.25], 8);

/// Each finite lane `(d, n)` and the direction `q` handed for it: one major magnitude `1`, alone
/// the largest, and Metal's face and texel read along `q` both in exact arithmetic and in `f32`.
fn check(lanes: &[([f32; 3], u32)], got: &[[f32; 3]], wrap: FaceWrap, what: &str) {
    for ((d, n), q) in lanes.iter().zip(got) {
        if d.iter().any(|x| !x.is_finite()) {
            assert_eq!(q.map(f32::to_bits), d.map(f32::to_bits), "{what}: {d:?}");
            continue;
        }
        let want = metal_texel(*d, *n, wrap);
        for exact in [true, false] {
            assert_eq!(
                device_texel(*q, *n, exact),
                want,
                "{what}: direction {d:?} on {n} handed {q:?} (exact: {exact})"
            );
        }
    }
}

/// On every lane of `metal_ties`'s tests, at every size, through `OpFmaKHR`, and on the texel-step
/// lanes through the device's `Fma` too, the device reads the face and texel Metal's position
/// names with the device's reciprocal along the handed direction; a direction with an infinite or
/// NaN component passes unchanged.
#[test]
fn the_device_reads_metals_nearest_texel() {
    let mut all: Vec<([f32; 3], u32)> = lanes()
        .into_iter()
        .enumerate()
        .map(|(i, d)| (d, SIZES[i % SIZES.len()]))
        .collect();
    let steps = steps();
    all.extend(steps.iter().copied());
    all.push(TODO_1_14);
    let none = FaceWrap::default();
    assert_eq!(metal_texel(TODO_1_14.0, 8, none), (1, true, [7, 5]));
    check(&all, &run(by(Route::Khr), &all, none), none, "Khr");
    let mut few: Vec<([f32; 3], u32)> = steps.iter().step_by(5).copied().collect();
    few.push(TODO_1_14);
    check(&few, &run(by(Route::Native), &few, none), none, "Native");
}

/// The step lanes reach the hazards Metal's position decides: on many of them the texel differs
/// from an unfused sum's (`RN(RN(c * r) + 1)`), from a quotient's (`RN(c / |m_a|)`), and from one
/// ulp of the reciprocal either way (Apple's texture unit's residue,
/// `PrecisionNote::CubeFaceReciprocal`).
#[test]
fn the_step_lanes_reach_the_rounding_hazards() {
    let steps = steps();
    let (mut unfused, mut quotient, mut residue) = (0, 0, 0);
    for ([ma, c, _], n) in &steps {
        let r = device_recip(*ma);
        let texel = |x: f32| clamped(x.into(), *n);
        let want = texel(metal_x(*c, r, *n, false));
        unfused += usize::from(texel((c * r + 1.0) * 0.5 * *n as f32) != want);
        quotient += usize::from(texel((c / ma + 1.0) * 0.5 * *n as f32) != want);
        let off = |d: i32| {
            texel(metal_x(
                *c,
                f32::from_bits(r.to_bits().wrapping_add_signed(d)),
                *n,
                false,
            ))
        };
        residue += usize::from(off(1) != want || off(-1) != want);
    }
    assert!(unfused > 100, "{unfused} of {} lanes", steps.len());
    assert!(quotient > 100, "{quotient} of {} lanes", steps.len());
    assert!(residue > 1000, "{residue} of {} lanes", steps.len());
}

/// `SNAP_WORDS`, the reciprocal's division, two fused multiply-adds', and the wrap's bound the
/// words `snapped` adds beside its routines, in a module valid on every committed profile, without
/// a wrap and wrapping either face axis or both.
#[test]
fn the_estimate_bounds_the_emitted_words() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    let wraps = [(false, false), (true, false), (false, true), (true, true)]
        .map(|(s, t)| FaceWrap { s, t });
    for (stem, profile) in &profiles {
        let fma = Fma::on(profile, SCALAR).expect("float");
        let without = crate::spirv::testing::assert_valid(
            kernel(fma, false, FaceWrap::default()).expect("kernel"),
            profile,
        );
        for wrap in wraps {
            let with = crate::spirv::testing::assert_valid(
                kernel(fma, true, wrap).expect("kernel"),
                profile,
            );
            let added = with.len() - without.len();
            let estimate = SNAP_WORDS + FaceRoutines { fma }.use_words() + wrap.words();
            assert!(
                added <= estimate as usize,
                "{stem} {wrap:?}: snapped adds {added} words, estimated {estimate}"
            );
        }
    }
}

/// Directions whose face coordinate on either axis of every face is exactly 1 or -1, or an ulp or
/// two inside, beside a major magnitude of 1, 3, or 0.75 (a tie among them), on faces of sizes
/// that are and are not powers of two.
fn edges() -> Vec<([f32; 3], u32)> {
    let below = |k: u32| f32::from_bits(1.0f32.to_bits() - k);
    let minors = [-1.0, -below(1), -below(2), 0.5, below(2), below(1), 1.0];
    let mut out = Vec::new();
    for n in [3u32, 5, 6, 7, 4, 8, 16] {
        for m in [1.0f32, 3.0, 0.75] {
            for face in 0..6 {
                for s in minors {
                    for t in minors {
                        let d = match face {
                            0 => [1.0, -t, -s],
                            1 => [-1.0, -t, s],
                            2 => [s, 1.0, t],
                            3 => [s, -1.0, -t],
                            4 => [s, -t, 1.0],
                            _ => [-s, -t, -1.0],
                        };
                        out.push((d.map(|c| c * m), n));
                    }
                }
            }
        }
    }
    out
}

/// Under `repeat` on a face axis whose size is not a power of two, a face coordinate of exactly 1
/// (a tie with the major magnitude included) reads texel 0 of that axis, the face's other edge, on
/// each axis the wrap names and on no other; at a size that is a power of two it reads the last
/// texel, as a clamping axis does. Every lane of [`edges`] is Metal's texel through `OpFmaKHR`, and
/// the wrap moves the texel on some lanes at each size that is not a power of two and on none at a
/// power of two.
#[test]
fn a_repeating_face_axis_wraps_a_coordinate_of_one() {
    let lanes = edges();
    let none = FaceWrap::default();
    for wrap in [(true, false), (false, true), (true, true)].map(|(s, t)| FaceWrap { s, t }) {
        check(
            &lanes,
            &run(by(Route::Khr), &lanes, wrap),
            wrap,
            &format!("{wrap:?}"),
        );
        for n in [3u32, 5, 6, 7, 4, 8, 16] {
            let moved = lanes
                .iter()
                .filter(|(d, m)| *m == n && metal_texel(*d, n, wrap) != metal_texel(*d, n, none))
                .count();
            match n.is_power_of_two() {
                true => assert_eq!(moved, 0, "{wrap:?} n={n}"),
                false => assert!(moved > 0, "{wrap:?} n={n}: no lane wraps"),
            }
        }
    }
}
