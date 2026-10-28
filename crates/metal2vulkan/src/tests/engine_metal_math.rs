//! Metal's math functions end to end: a call's `air` symbol variant and fast-math flags select
//! what it computes, the device's instruction where one is the function and a definition of the
//! device's instructions where none is. On every committed M1 profile each kernel compiles, links,
//! and passes L3, and its module holds the instructions of the form it takes:
//!
//! - `air.fast_tanh` on `float` is the MSL fast-math formula `(t - 1) / (t + 1)`,
//!   `t = exp2(x * 2 log2 e)` of the device's `Exp2` (no `Tanh`), every step of it
//!   `NoContraction`, its quotient the device's division (`OpFDiv`); the precise `air.tanh` is the
//!   device's `Tanh`. A `half` `fast_tanh` is Metal's `half` `tanh` (Metal's `half` functions are
//!   the same in both math modes, and MSL's fast-math formulas are its `float` ones), so it is the
//!   device's `Tanh` too (no `Exp2`, no division), noted `NativeTranscendentals` wherever the
//!   device's functions are not Metal's own, and on MoltenVK MSL's `float` `tanh` rounded to
//!   `half` (`DeviceTranscendentals`);
//! - `air.sign` is the device's `FSign` with and without fast math (it keeps the sign of a zero
//!   where Metal answers `+0`), noted `NativeTranscendentals` wherever the device's functions are
//!   not Metal's own (on MoltenVK it is MSL's own `sign`), and both answer `-1` for `-3` on the
//!   interpreter;
//! - `air.sinpi` and `air.fast_sinpi` reduce their argument (`RoundEven`) and take the sine of the
//!   residue from the device's `Sin` on every profile.
//!
//! The values of these functions are the unit tests' (`lower::arith::tests`); no GPU is used.
//!
//! The sRGB conversions of `unorm4x8_srgb` compile end to end: A1's committed fixtures
//! `a1_srgb_decode_all_bytes` (every byte in every channel, `float` and `half` results) and
//! `a1_srgb_encode_sweep` (every multiple of 2⁻¹⁴ in `[0, 1.0078]` and every `half` up to 2.25)
//! compile on every profile above, take the MSL transfer functions by the device's `Pow` around
//! the 8-bit instruction, declare no table, and note `DevicePackConversion`. The interpreter models
//! no `Pow`, so their values are the device lanes' (the validation crate's authored cases of the
//! same fixtures).
//!
//! `llvm-as` assembles the modules; without it the tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::profile::FloatFunctions;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, exact_maths, link_and_validate, llvm_assemble, meter, profile,
    typed_module, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::StorageClass;

const PROFILES: [&str; 3] = ["moltenvk-m3", "lunarg-desktop-baseline-2023", "vk12-floor"];

/// `GLSL.std.450` instruction numbers.
const FSIGN: u32 = 6;
const SIN: u32 = 13;
const TANH: u32 = 21;
const EXP2: u32 = 29;
const ROUND_EVEN: u32 = 2;
const POW: u32 = 26;
const PACK_UNORM_4X8: u32 = 55;
const UNPACK_UNORM_4X8: u32 = 64;

/// One kernel: its body defines `%r` (`float`), which it stores.
struct Kernel {
    name: &'static str,
    body: &'static str,
    /// `GLSL.std.450` instructions the module must and must not contain.
    has: &'static [u32],
    lacks: &'static [u32],
    /// Whether the module divides (`OpFDiv`): `fast::tanh`'s quotient.
    divides: bool,
    /// Whether the module takes the device's instruction for a function Metal defines otherwise,
    /// recording `NativeTranscendentals` wherever the device's functions are not Metal's own.
    native: bool,
    /// The stored word on the interpreter, when it models the kernel.
    want: Option<u32>,
}

const KERNELS: [Kernel; 8] = [
    Kernel {
        name: "fast_tanh_float",
        body: "  %a = fadd fast float 0x4046000000000000, -0.0\n  %r = call fast float @air.fast_tanh.f32(float %a)\n",
        has: &[EXP2],
        lacks: &[TANH],
        divides: true,
        native: false,
        want: None,
    },
    // RC3 cases e72f550f and 2d5f3949's GELU tail at x = -8: 0.5 * x * (1 + tanh(...)).
    Kernel {
        name: FAST_TANH_IN_GELU,
        body: "  %a = fadd fast float 0xC038A66660000000, -0.0\n  %x = fadd fast float 0xC020000000000000, -0.0\n  %t = call fast float @air.fast_tanh.f32(float %a)\n  %s = fadd fast float %t, 1.000000e+00\n  %h = fmul fast float %x, 5.000000e-01\n  %r = fmul fast float %h, %s\n",
        has: &[EXP2],
        lacks: &[TANH],
        divides: true,
        native: false,
        want: None,
    },
    Kernel {
        name: "precise_tanh_float",
        body: "  %a = fadd fast float 0x4046000000000000, -0.0\n  %r = call fast float @air.tanh.f32(float %a)\n",
        has: &[TANH],
        lacks: &[EXP2],
        divides: false,
        native: false,
        want: None,
    },
    Kernel {
        name: "fast_tanh_half",
        body: "  %a = fadd fast float 0x4000000000000000, -0.0\n  %h = fptrunc float %a to half\n  %t = call fast half @air.fast_tanh.f16(half %h)\n  %r = fpext half %t to float\n",
        has: &[TANH],
        lacks: &[EXP2],
        divides: false,
        native: true,
        want: None,
    },
    Kernel {
        name: "fast_sign",
        body: "  %a = fadd fast float 0xC008000000000000, -0.0\n  %r = call fast float @air.sign.f32(float %a)\n",
        has: &[FSIGN],
        lacks: &[],
        divides: false,
        native: true,
        want: Some(0xBF80_0000),
    },
    Kernel {
        name: "precise_sign",
        body: "  %a = fadd float 0xC008000000000000, -0.0\n  %r = call float @air.sign.f32(float %a)\n",
        has: &[FSIGN],
        lacks: &[],
        divides: false,
        native: true,
        want: Some(0xBF80_0000),
    },
    Kernel {
        name: "precise_sinpi",
        body: "  %a = fadd fast float 0x3FF0000000000000, -0.0\n  %r = call fast float @air.sinpi.f32(float %a)\n",
        has: &[SIN, ROUND_EVEN],
        lacks: &[],
        divides: false,
        native: false,
        want: None,
    },
    Kernel {
        name: "fast_sinpi",
        body: "  %a = fadd fast float 0x3FF0000000000000, -0.0\n  %r = call fast float @air.fast_sinpi.f32(float %a)\n",
        has: &[SIN, ROUND_EVEN],
        lacks: &[],
        divides: false,
        native: false,
        want: None,
    },
];

/// The kernel whose `fast::tanh` feeds the caller's `1 + tanh` ([`fast_tanh_is_one_operation`]).
const FAST_TANH_IN_GELU: &str = "fast_tanh_in_gelu";

/// The `GLSL.std.450 Fma` instruction number.
const FMA: u32 = 50;

fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for k in &KERNELS {
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n{}  store float %r, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n",
            k.name, k.body
        ));
    }
    s.push_str("declare float @air.fast_tanh.f32(float)\ndeclare float @air.tanh.f32(float)\ndeclare half @air.fast_tanh.f16(half)\ndeclare float @air.sign.f32(float)\ndeclare float @air.sinpi.f32(float)\ndeclare float @air.fast_sinpi.f32(float)\n");
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-metal-math");
    let dir = scratch.path();
    let ll = dir.join("metal_math.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// The `GLSL.std.450` instruction numbers of every `OpExtInst` in `words`, and whether an
/// `OpFDiv` occurs.
fn ext_insts(words: &[u32]) -> (Vec<u32>, bool) {
    let (mut ext, mut div) = (Vec::new(), false);
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == crate::spv::Op::ExtInst as u32 {
            ext.push(words[i + 4]);
        }
        div |= op == crate::spv::Op::FDiv as u32;
        i += len;
    }
    (ext, div)
}

/// The stored word of `entry` on the interpreter.
fn interpret(bitcode: &[u8], entry: &str, stem: &str) -> u32 {
    let p = profile(stem, exact_maths()).expect("profile");
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, &p, &mut m).expect("memory");
    let plan = select(&spec, &memory, &p, &LowerInputs::new(), &mut m).expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; 4],
        }],
        push: vec![0; 4],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let b = out.buffers.first().expect("buffer");
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[test]
fn metal_math_variants_select_their_forms() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for k in &KERNELS {
            let what = format!("{} on {stem}", k.name);
            let a = compile(&bitcode, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, div) = ext_insts(&words);
            for i in k.has {
                assert!(
                    ext.contains(i),
                    "{what}: GLSL.std.450 {i} missing ({ext:?})"
                );
            }
            for i in k.lacks {
                assert!(
                    !ext.contains(i),
                    "{what}: GLSL.std.450 {i} present ({ext:?})"
                );
            }
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            assert_eq!(div, k.divides, "{what}: OpFDiv");
            // The fast-math entry's `fast_tanh` notes its steps' precision (the device's
            // `Exp2` and division), except where every step is Metal's own function on the device
            // (`FloatFunctions::Metal`), which notes nothing in such an entry.
            if k.divides {
                assert_eq!(
                    a.precision.contains(&PrecisionNote::DeviceArithmetic)
                        && a.precision.contains(&PrecisionNote::NativeTranscendentals),
                    !metal,
                    "{what}: {:?}",
                    a.precision
                );
            }
            // The device's instruction: noted `NativeTranscendentals` wherever the device's
            // functions are not Metal's own. On MoltenVK `sign` is MSL's own; a `half` `tanh` is
            // MSL's `float` one rounded to `half`, within Apple's bound.
            if k.native {
                assert_eq!(
                    a.precision.contains(&PrecisionNote::NativeTranscendentals),
                    !metal,
                    "{what}: {:?}",
                    a.precision
                );
                let half = k.name.ends_with("_half");
                assert_eq!(
                    a.precision.contains(&PrecisionNote::DeviceTranscendentals),
                    metal && half,
                    "{what}: {:?}",
                    a.precision
                );
            }
            if let Some(want) = k.want {
                let got = interpret(&bitcode, k.name, stem);
                assert_eq!(got, want, "{what}: {got:#x} != {want:#x}");
            }
        }
    }
}

/// Each float arithmetic result of `words` (`OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv`, and
/// `GLSL.std.450 Fma`), as its opcode and whether it is decorated `NoContraction`.
fn float_arithmetic(words: &[u32]) -> Vec<(u32, bool)> {
    let (mut results, mut exact) = (Vec::new(), Vec::new());
    let arith = [
        crate::spv::Op::FAdd,
        crate::spv::Op::FSub,
        crate::spv::Op::FMul,
        crate::spv::Op::FDiv,
    ]
    .map(|op| op as u32);
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == crate::spv::Op::Decorate as u32
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            exact.push(words[i + 1]);
        }
        if arith.contains(&op) || (op == crate::spv::Op::ExtInst as u32 && words[i + 4] == FMA) {
            results.push((op, words[i + 2]));
        }
        i += len;
    }
    results
        .into_iter()
        .map(|(op, id)| (op, exact.contains(&id)))
        .collect()
}

/// `air.fast_tanh` is one operation to Metal, so the device may not fuse or regroup any of its
/// steps with the caller's arithmetic. Every step is `NoContraction` whatever the call's flags:
/// `x · 2 log2 e` (in the profile's form: `GLSL.std.450 Fma(a, b, -0.0)` where its `float`
/// functions are Metal's, an `OpFMul` elsewhere), `t - 1`, `t + 1`, and the quotient. The caller's
/// `fast` arithmetic stays undecorated. Without the decorations, Mesa's `lower_flrp` rule turned
/// GELU's `1 + tanh` into `t` on RADV and x86 lavapipe (RC3 cases e72f550f and 2d5f3949,
/// `gelu(-8)`).
#[test]
fn fast_tanh_is_one_operation() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES.into_iter().chain(["amd-rdna3-radv", "lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        let what = format!("{FAST_TANH_IN_GELU} on {stem}");
        let a =
            compile(&bitcode, FAST_TANH_IN_GELU, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let ops = float_arithmetic(&words);
        let count = |op: crate::spv::Op, exact: bool| {
            ops.iter()
                .filter(|(o, e)| *o == op as u32 && *e == exact)
                .count()
        };
        let fma = |exact: bool| {
            ops.iter()
                .filter(|(o, e)| *o == crate::spv::Op::ExtInst as u32 && *e == exact)
                .count()
        };
        // The tanh's product is in the profile's form.
        let metal = usize::from(p.fields().float_functions == FloatFunctions::Metal);
        assert_eq!(
            (
                count(crate::spv::Op::FMul, true),
                count(crate::spv::Op::FSub, true),
                count(crate::spv::Op::FAdd, true),
                count(crate::spv::Op::FDiv, true),
                fma(true)
            ),
            (1 - metal, 1, 1, 1, metal),
            "{what}: the tanh's steps, NoContraction: {ops:?}"
        );
        assert_eq!(
            (
                fma(false),
                count(crate::spv::Op::FSub, false),
                count(crate::spv::Op::FDiv, false),
                count(crate::spv::Op::FMul, false),
            ),
            (0, 0, 0, 2),
            "{what}: only the caller's multiplies are left free: {ops:?}"
        );
        assert!(
            count(crate::spv::Op::FAdd, false) >= 1,
            "{what}: the caller's 1 + tanh stays free: {ops:?}"
        );
    }
}

/// End to end: the sRGB fixtures through select, emit, and link on every profile above: L3 passes,
/// and each module takes the MSL transfer functions by the device's `Pow` around the 8-bit
/// instruction (`a1_srgb_decode_all_bytes`: `UnpackUnorm4x8`, its `float` and `half` results;
/// `a1_srgb_encode_sweep`: `PackUnorm4x8`, from `float` and `half` colours), declares no table
/// (no initialized `Private` variable), and records `DevicePackConversion`.
#[test]
fn srgb_fixtures_take_the_transfer_by_the_devices_pow() {
    let assembled = |stem: &str| match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {stem}: {e}"),
    };
    let (Some(decode_bc), Some(encode_bc)) = (
        assembled("a1_srgb_decode_all_bytes"),
        assembled("a1_srgb_encode_sweep"),
    ) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (bc, entry, instruction) in [
            (&decode_bc, "a1_srgb_decode_all_bytes", UNPACK_UNORM_4X8),
            (&encode_bc, "a1_srgb_encode_sweep", PACK_UNORM_4X8),
        ] {
            let what = format!("{entry} on {stem}");
            let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, _) = ext_insts(&words);
            for i in [POW, instruction] {
                assert!(
                    ext.contains(&i),
                    "{what}: GLSL.std.450 {i} missing ({ext:?})"
                );
            }
            assert!(
                a.precision.contains(&PrecisionNote::DevicePackConversion),
                "{what}: {:?}",
                a.precision
            );
            let (_, m) = typed_module(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let tables = m
                .globals()
                .iter()
                .filter(|g| g.class == StorageClass::Private && g.init.is_some())
                .count();
            assert_eq!(tables, 0, "{what}: a table");
        }
    }
}
