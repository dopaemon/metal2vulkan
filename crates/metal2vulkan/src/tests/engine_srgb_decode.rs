//! An 8-bit sRGB texel read as Metal reads it, by the device's decode wherever the device samples
//! the format, and through Metal's twelve-bit table where it holds the format as its UNORM twin.
//!
//! - Without a slot's `SrgbTexture` variant every access reads the device's decode through the
//!   slot's sRGB view, declaring no table: `PrecisionNote::SrgbTexelDecode` under the bound
//!   format's class off Metal's own texture unit, `VkProperty::MetalSampler` on it.
//! - Under the variant (a format the device cannot sample, held as its UNORM twin), on the CPU
//!   interpreter, a read and a pixel-coordinate fetch of every code, with a `half` and a `float`
//!   result, are the reference bit for bit: a color channel's step
//!   `k = round(decode(c / 255) · 4095)` (the MSL transfer, which A1's every-byte reading matches)
//!   and its value `k / 4095`, alpha's `c / 255`, each correctly rounded to the result's type. A
//!   filtered sample decodes the device's filter of the codes (`SrgbTexelDecode`).
//! - The binder joins a slot to the plan's variant exactly where the device holds the bound
//!   format's twin, and its views then view the twin; elsewhere they view the sRGB format.
//! - The plan's word estimate bounds the emission of every form on every committed profile.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bind::{
    BindState, BoundBuffer, BoundTexture, ImageSource, StageTable, TextureFacts, WriteSource,
};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::{Device, DeviceDescription, DevicePolicy, FormatUsage};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::format::bound_texel_format;
use crate::metal::texel_format::srgb_twin_on;
use crate::metal::{self, PixelFormat, TextureType};
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{
    Axis, PushField, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage,
};
use crate::pipeline::testing::{
    link_and_validate, llvm_assemble, meter, profile, typed_module_with, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, SpecializedNote,
    StageSource,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, TypedModule};
use crate::spirv::testing::committed_profiles;

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The texture's side: 16 x 16 texels, one per code.
const SIDE: u32 = 16;

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-srgb-raw-{}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        name.replace(' ', "-")
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

/// The words of a constexpr sampler (the `hlf_half_address` fixture's fields): minification and
/// magnification `filter` (0 nearest, 1 linear), mip filter `mip` (0 none, 1 nearest, 2
/// linear), every axis clamped to its edge, `pixel` coordinates.
fn sampler_words(filter: u64, mip: u64, pixel: bool) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    let edge = 1;
    BASE | edge
        | edge << 3
        | edge << 6
        | filter << 9
        | filter << 11
        | mip << 13
        | u64::from(pixel) << 15
}

/// How a test kernel accesses `texture2d` slot 0 at its thread's texel `gid`.
#[derive(Clone, Copy, Debug)]
enum Access {
    /// `t.read(gid)` of an `access::read` texture.
    Read,
    /// `t.sample(s, coord)` through a constexpr sampler of these words, at the texel's centre
    /// (pixel coordinates, or normalized ones where the sampler's are).
    Sample(u64),
    /// `t.sample(s, coord)` through the runtime `[[sampler(0)]]`, at the texel's normalized
    /// centre.
    Runtime,
    /// `t.gather(s, coord, 0, component)` through a nearest constexpr sampler.
    Gather(u32),
}

/// A kernel accessing slot 0 (`half` texels, or `float`) as `access` at thread `gid` (a 16 x 16
/// grid), storing the `half4` (`float4`) to `out[gid.x + 16 gid.y]` (buffer 0).
fn kernel(access: Access, half: bool) -> String {
    let (texel, vec4, size, out_ty) = if half {
        ("half", "<4 x half>", 8, "half4")
    } else {
        ("float", "<4 x float>", 16, "float4")
    };
    let suffix = if half { "v4f16" } else { "v4f32" };
    let (state_words, runtime) = match access {
        Access::Read => (None, false),
        Access::Sample(w) => (Some(w), false),
        Access::Runtime => (None, true),
        Access::Gather(_) => (Some(sampler_words(0, 0, false)), false),
    };
    let pixel = matches!(access, Access::Sample(w) if w & 1 << 15 != 0);
    let scale = if pixel { "1.0" } else { "0.0625" };
    let s = if runtime {
        "%s"
    } else {
        "@__air_sampler_state"
    };
    let (tex_access, call, decl) = match access {
        Access::Read => (
            "read",
            format!(
                "%rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"
            ),
        ),
        Access::Sample(_) | Access::Runtime => (
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"
            ),
        ),
        Access::Gather(component) => (
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {s}, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 {component}, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"
            ),
        ),
    };
    let mut params = String::new();
    let mut mds = Vec::new();
    let mut next = 2;
    if runtime {
        params.push_str(", ptr addrspace(2) %s");
        mds.push(format!("!{{i32 {next}, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}}"));
        next += 1;
    }
    mds.push(format!("!{{i32 {next}, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"gid\"}}"));
    let first = 6;
    let arg_mds: Vec<String> = (0..mds.len()).map(|i| format!("!{}", first + i)).collect();
    let tail: String = mds
        .iter()
        .enumerate()
        .map(|(i, m)| format!("!{} = {m}\n", first + i))
        .collect();
    let (state, states, state_md) = match state_words {
        Some(w) => (
            format!("@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {w}, i64 0], align 8\n\n"),
            "!air.sampler_states = !{!5}\n",
            "!5 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        None => (String::new(), "", ""),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out{params}, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = shl i32 %y, 4
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %cx = fadd float %fx, 0.5
  %cy = fadd float %fy, 0.5
  %c0 = insertelement <2 x float> zeroinitializer, float %cx, i64 0
  %c1 = insertelement <2 x float> %c0, float %cy, i64 1
  %uv = fmul <2 x float> %c1, <float {scale}, float {scale}>
  {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

{decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, {arg_mds}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{tex_access}", !"air.arg_type_name", !"texture2d<{texel}, {tex_access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
{state_md}{tail}"#,
        arg_mds = arg_mds.join(", "),
    )
}

/// Lowering inputs with slot 0 in the `SrgbTexture` variant (bound to a format the device holds
/// as its UNORM twin), or none.
fn inputs(variant: bool) -> LowerInputs {
    let mut i = LowerInputs::new();
    if variant {
        i.srgb_textures.insert(TEXTURE);
    }
    i
}

fn lavapipe() -> crate::device::profile::DeviceProfile {
    profile("lavapipe", DevicePolicy::default()).expect("profile")
}

fn m3() -> crate::device::profile::DeviceProfile {
    profile("moltenvk-m3", DevicePolicy::default()).expect("profile")
}

// ---- The reference ----------------------------------------------------------------------------

/// The four codes of texel `n` (0 to 255): every code in each channel, alpha's permuted.
fn codes(n: u32) -> [u32; 4] {
    [n, 255 - n, (n * 7 + 3) % 256, (n * 13 + 5) % 256]
}

/// Metal's decode step of code `c`: `round(decode(c / 255) · 4095)` for the MSL transfer.
pub(super) fn step(c: u32) -> u64 {
    let x = f64::from(c) / 255.0;
    let lin = if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    };
    (lin * 4095.0).round() as u64
}

/// `k / d` (`0 <= k <= d`, `d` odd, so no ties) correctly rounded to a binary format of
/// `bits` significand bits: its significand `q` and exponent `e` (`value = q · 2^-e`), `None` for
/// zero.
fn ratio(k: u64, d: u64, bits: u32) -> Option<(u128, i32)> {
    if k == 0 {
        return None;
    }
    let (k, d) = (u128::from(k), u128::from(d));
    let mut e = 0;
    while k << e < d << (bits - 1) {
        e += 1;
    }
    let n = k << e;
    let (q, r) = (n / d, n % d);
    Some((q + u128::from(2 * r > d), e))
}

/// `f32` bits of `k / d` correctly rounded.
pub(super) fn f32_bits(k: u64, d: u64) -> u32 {
    ratio(k, d, 24)
        .map_or(0.0f32, |(q, e)| ((q as f64) * 2f64.powi(-e)) as f32)
        .to_bits()
}

/// `half` bits of `k / d` correctly rounded (every nonzero quotient here is a normal `half`).
pub(super) fn half_bits(k: u64, d: u64) -> u16 {
    let Some((mut q, mut e)) = ratio(k, d, 11) else {
        return 0;
    };
    if q == 1 << 11 {
        q >>= 1;
        e -= 1;
    }
    let biased = 10 - e + 15;
    assert!(biased > 0, "a half denormal");
    ((biased as u16) << 10) | (q as u16 - (1 << 10))
}

/// Channel `c`'s exact quotient: a color channel's step over 4095, alpha's code over 255.
fn quotient(channel: usize, code: u32) -> (u64, u64) {
    if channel < 3 {
        (step(code), 4095)
    } else {
        (u64::from(code), 255)
    }
}

/// The twin's image at `slot`: texel `(x, y)` holds codes `codes(x + 16 y)`, each read as
/// `c / 255` correctly rounded to `f32`, as a UNORM view reads it.
fn codes_image(slot: ResourceSlot) -> Image {
    let mut img = Image::new(slot, [SIDE, SIDE, 1], 1, 1, 1);
    for n in 0..SIDE * SIDE {
        let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
        img.texels[at] = codes(n).map(|c| f32_bits(u64::from(c), 255));
    }
    img
}

/// The module's resource of `kind`'s shape at texture slot 0.
fn resource(m: &TypedModule, kind: impl Fn(&ResourceKind) -> bool) -> Option<ResourceSlot> {
    m.globals().iter().find_map(|g| match g.kind {
        GlobalKind::Resource(s) if s.index == 0 && kind(&s.kind) => Some(s),
        _ => None,
    })
}

/// A sampled texture's binding: a texture, separate or combined with its sampler.
fn texture(k: &ResourceKind) -> bool {
    matches!(
        k,
        ResourceKind::Texture
            | ResourceKind::SeparateTexture(_)
            | ResourceKind::CombinedTextureSampler(_)
    )
}

/// Resources of a dispatch of one-thread threadgroups over the 16 x 16 grid, `out` sized for
/// `bytes` per texel.
fn resources(plan: &Plan, bytes: usize) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    for f in &plan.push.fields {
        match (f.field, f.loc) {
            (PushField::SlotBindOffset(_) | PushField::DispatchRegion, PushLoc::Push { .. }) => {}
            // A bound value read as data, which the interpreter writes from `spec`.
            (PushField::Specialized(_), PushLoc::Push { .. }) => {}
            // No slot's writes dropped: the word stays 0.
            (PushField::Specialized(crate::pipeline::sym::SpecKey::DroppedStoreSlots(_)), _) => {}
            other => panic!("unexpected push field {other:?}"),
        }
    }
    let out = Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: vec![0xab; bytes * (SIDE * SIDE) as usize],
    };
    Resources {
        buffers: vec![out],
        push,
        spec,
    }
}

/// `m` over the grid with `images` bound, in both scheduling orders, which must agree: `out`.
fn run(m: &TypedModule, res: &Resources, images: &[Image]) -> Vec<u8> {
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [SIDE, SIDE, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(m, &cfg, res.clone(), images.to_vec())
            .unwrap_or_else(|e| panic!("interpreter: {e:?}"));
        outs.push(o.buffers);
    }
    assert_eq!(outs[0], outs[1], "the scheduling orders disagree");
    outs.swap_remove(0).swap_remove(0)
}

/// Under the variant (a slot bound in a format the device holds as its UNORM twin, whose view
/// reads each code as `c / 255`), a read and a pixel-coordinate fetch of every code, `half` and
/// `float`, are Metal's decode bit for bit, through the slot's own binding, noting nothing.
#[test]
fn twin_reads_and_fetches_decode_every_code_as_metal_does() {
    let fetch = sampler_words(0, 0, true);
    for (name, access) in [("read", Access::Read), ("fetch", Access::Sample(fetch))] {
        for half in [true, false] {
            let what = format!("{name} {}", if half { "half" } else { "float" });
            let Some(bc) = assemble(&what, &kernel(access, half)) else {
                return;
            };
            for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
                let (plan, m) = typed_module_with(&bc, "k", &p, &inputs(true))
                    .unwrap_or_else(|e| panic!("{what} on {stem}: {e}"));
                assert!(
                    !plan.precision.contains(&PrecisionNote::SrgbTexelDecode)
                        && !plan
                            .specialized
                            .iter()
                            .any(|s| s.note == PrecisionNote::SrgbTexelDecode),
                    "{what} on {stem}"
                );
                let own = resource(&m, texture)
                    .unwrap_or_else(|| panic!("{what} on {stem}: no texture binding"));
                let bytes = if half { 8 } else { 16 };
                let out = run(&m, &resources(&plan, bytes), &[codes_image(own)]);
                for n in 0..SIDE * SIDE {
                    for (c, code) in codes(n).into_iter().enumerate() {
                        let (k, d) = quotient(c, code);
                        let at = n as usize * bytes;
                        let (got, want) = if half {
                            let i = at + 2 * c;
                            (
                                u32::from(u16::from_le_bytes([out[i], out[i + 1]])),
                                u32::from(half_bits(k, d)),
                            )
                        } else {
                            let i = at + 4 * c;
                            (
                                u32::from_le_bytes([out[i], out[i + 1], out[i + 2], out[i + 3]]),
                                f32_bits(k, d),
                            )
                        };
                        assert_eq!(
                            got, want,
                            "{what} on {stem}: texel {n} channel {c} code {code}: {got:#x} vs {want:#x}"
                        );
                    }
                }
            }
        }
    }
}

/// Under the variant, a read of a texture whose swizzle moves a code between a color lane and
/// alpha (`abgr`) decodes each code by its own channel and then swizzles, as plain Metal does,
/// under the slot's `swizzled_storage` variant (the view takes no swizzle): each lane is its
/// source channel's quotient, a color channel's step over 4095 on alpha's lane and alpha's code
/// over 255 on red's, bit for bit, on the interpreter.
#[test]
fn a_swizzled_twin_read_decodes_before_its_swizzle() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels};
    let Some(bc) = assemble("read float", &kernel(Access::Read, false)) else {
        return;
    };
    let swizzle = TextureSwizzleChannels {
        red: S::Alpha,
        green: S::Blue,
        blue: S::Green,
        alpha: S::Red,
    };
    let source = [3usize, 2, 1, 0];
    for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
        let mut under = inputs(true);
        under.swizzled_storage.insert(TEXTURE);
        let (plan, m) =
            typed_module_with(&bc, "k", &p, &under).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let own = resource(&m, texture).unwrap_or_else(|| panic!("{stem}: no texture binding"));
        let mut res = resources(&plan, 16);
        res.spec
            .insert(SpecKey::TextureSwizzle(TEXTURE), u64::from(swizzle.word()));
        let out = run(&m, &res, &[codes_image(own)]);
        for n in 0..SIDE * SIDE {
            let codes = codes(n);
            for (lane, &from) in source.iter().enumerate() {
                let (k, d) = quotient(from, codes[from]);
                let i = n as usize * 16 + 4 * lane;
                let got = u32::from_le_bytes([out[i], out[i + 1], out[i + 2], out[i + 3]]);
                assert_eq!(
                    got,
                    f32_bits(k, d),
                    "{stem}: texel {n} lane {lane} (channel {from}, code {})",
                    codes[from]
                );
            }
        }
    }
}

/// The plan of a slot bound with a swizzle in an 8-bit sRGB format the device holds as its UNORM
/// twin names the slot in `VariantKey::swizzled_storage`, and its views take no swizzle, exactly
/// where the swizzle moves a code between a color lane and alpha (the function decodes, then
/// swizzles); a swizzle among the color lanes or to constants keeps the views' component mapping,
/// which reads the same. A format the device samples is decoded by its sRGB view before the
/// view's component mapping swizzles, so its views take every swizzle.
#[test]
fn an_srgb_swizzle_across_alpha_swizzles_in_the_function_on_a_twin() {
    use crate::metal::{TextureSwizzle as S, TextureSwizzleChannels};
    let Some(bc) = assemble("read float", &kernel(Access::Read, false)) else {
        return;
    };
    let channels = |red, green, blue, alpha| TextureSwizzleChannels {
        red,
        green,
        blue,
        alpha,
    };
    let cases = [
        (channels(S::Alpha, S::Blue, S::Green, S::Red), true),
        (channels(S::Red, S::Green, S::Blue, S::Red), true),
        (channels(S::Alpha, S::Green, S::Blue, S::Alpha), true),
        (channels(S::Blue, S::Green, S::Red, S::Alpha), false),
        (channels(S::Red, S::Red, S::Red, S::One), false),
        (channels(S::Zero, S::Green, S::One, S::Alpha), false),
    ];
    let (mut planned, mut swizzled) = (0, 0);
    for (stem, dev) in devices() {
        let plain = pipeline(&dev, &bc, &inputs(false));
        for pf in [PixelFormat::RGBA8UnormSRGB, PixelFormat::RG8UnormSRGB] {
            let twin = srgb_twin_on(pf, dev.profile());
            for (swizzle, crosses) in cases {
                let what = format!("{stem} {pf:?} {swizzle:?}");
                let mut state = state(Some(pf));
                if let Some(BoundTexture::Texture { facts, .. }) = state
                    .stages
                    .get_mut(&Stage::Kernel)
                    .and_then(|t| t.textures.get_mut(&0))
                {
                    facts.swizzle = swizzle;
                }
                let Ok(plan) = plain.plan(&dev, &state.shape()) else {
                    continue;
                };
                let in_function = crosses && twin;
                assert_eq!(
                    plan.variant.swizzled_storage.contains(&TEXTURE),
                    in_function,
                    "{what}"
                );
                let views = slot_views(&plan, TEXTURE);
                assert!(!views.is_empty(), "{what}");
                for v in views {
                    assert_eq!(v.shader_swizzle, in_function, "{what}");
                }
                planned += 1;
                swizzled += usize::from(in_function);
            }
        }
    }
    assert!(planned > 0, "no committed profile planned the texture");
    assert!(swizzled > 0, "no committed profile holds a twin");
}

/// Every access of an 8-bit sRGB texture slot records `RuntimeFact::SrgbTexture`. Without the
/// variant every one of them (a read, a fetch, a nearest, linear, mipmapped or bicubic sample, a
/// gather, a runtime sampler's sample) reads the device's own decode and declares no Metal table:
/// on lavapipe it notes `PrecisionNote::SrgbTexelDecode` under the bound format's class (an
/// `RGBA8Unorm_sRGB` or ASTC sRGB texture's, not an `RGBA8Unorm` one's), and on MoltenVK, whose
/// decode is Metal's own, it requires `VkProperty::MetalSampler` instead. Under the variant every
/// access declares the table, and only a filtered sample (a linear filter, a linear mip filter, a
/// runtime sampler), the device's filter of the codes, notes `SrgbTexelDecode`, outright. On
/// MoltenVK a filtered sample records `PrecisionNote::HalfSampleFilter` or
/// `PrecisionNote::FloatSampleFilter` (`lower::sampler::linear::metal_filtered_ycbcr`), a `half`
/// access that converts the device's texel by the bound format's class states
/// `RuntimeFact::TexelFormat`, and every sample and gather the device makes requires the
/// property. Every form validates on both devices.
#[test]
fn the_device_decodes_every_access_but_a_twins() {
    use crate::device::requirements::VkProperty;
    let nearest = sampler_words(0, 0, false);
    let linear = sampler_words(1, 0, false);
    let bicubic = sampler_words(2, 0, false);
    let fetch = sampler_words(0, 0, true);
    // Each access, and whether it reads one texel's codes under the variant (not a filter's).
    let cases = [
        ("read half", Access::Read, true, true),
        ("read float", Access::Read, false, true),
        ("fetch half", Access::Sample(fetch), true, true),
        ("fetch float", Access::Sample(fetch), false, true),
        ("nearest half", Access::Sample(nearest), true, true),
        ("nearest float", Access::Sample(nearest), false, true),
        (
            "nearest level",
            Access::Sample(sampler_words(0, 1, false)),
            false,
            true,
        ),
        ("gather red half", Access::Gather(0), true, true),
        ("gather alpha half", Access::Gather(3), true, true),
        ("gather float", Access::Gather(0), false, true),
        ("bicubic half", Access::Sample(bicubic), true, true),
        ("bicubic float", Access::Sample(bicubic), false, true),
        ("linear half", Access::Sample(linear), true, false),
        ("linear float", Access::Sample(linear), false, false),
        (
            "mip linear",
            Access::Sample(sampler_words(0, 2, false)),
            false,
            false,
        ),
        ("runtime half", Access::Runtime, true, false),
        ("runtime float", Access::Runtime, false, false),
    ];
    let consts = FunctionConstantValues::defaults();
    for (name, access, half, codes) in cases {
        let Some(bc) = assemble(name, &kernel(access, half)) else {
            return;
        };
        for (stem, p) in [("lavapipe", lavapipe()), ("moltenvk-m3", m3())] {
            let metal = stem == "moltenvk-m3";
            for variant in [false, true] {
                let what = format!("{name} on {stem}, variant {variant}");
                let (plan, _) = typed_module_with(&bc, "k", &p, &inputs(variant))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(
                    plan.facts
                        .contains(&RuntimeFact::SrgbTexture { slot: TEXTURE }),
                    "{what}"
                );
                assert_eq!(
                    plan.precision.contains(&PrecisionNote::SrgbTexelDecode),
                    !variant && !metal || variant && !codes,
                    "{what}"
                );
                let noted = |pf| bound_note(&plan.specialized, pf);
                for pf in [PixelFormat::RGBA8UnormSRGB, PixelFormat::ASTC4x4SRGB] {
                    assert_eq!(noted(pf), !variant && !metal, "{what} {pf:?}");
                }
                assert!(!noted(PixelFormat::RGBA8Unorm), "{what}");
                // On MoltenVK a `half` access outside the codes converts its texel by the
                // format's class, whose rounding to `half` is Metal's there only through Metal's
                // own texture unit; a bicubic sample's taps are fetched.
                let converts = metal && half && !(variant && codes) && name != "bicubic half";
                assert_eq!(
                    plan.facts
                        .contains(&RuntimeFact::TexelFormat { slot: TEXTURE }),
                    converts,
                    "{what}"
                );
                assert_eq!(
                    plan.precision.contains(&PrecisionNote::FloatSampleFilter),
                    !codes && !half && metal,
                    "{what}"
                );
                assert_eq!(
                    plan.precision.contains(&PrecisionNote::HalfSampleFilter),
                    !codes && half && metal,
                    "{what}"
                );
                let a = compile_bitcode(
                    &bc,
                    EntryRef::Name("k"),
                    &consts,
                    &p,
                    &inputs(variant),
                    &mut meter(),
                    SPIRV_WORDS,
                )
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(
                    super::engine_fragment::holds_srgb_table(&a.module.words),
                    variant,
                    "{what}"
                );
                // Every sample and gather the device makes (not a read, a pixel fetch or a
                // bicubic sample's fetched taps), and every decode of the device's.
                let sampled = !matches!(
                    name,
                    "read half"
                        | "read float"
                        | "fetch half"
                        | "fetch float"
                        | "bicubic half"
                        | "bicubic float"
                );
                assert_eq!(
                    a.requirements
                        .properties
                        .contains(&VkProperty::MetalSampler),
                    metal && (!variant || sampled || converts),
                    "{what}"
                );
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            }
        }
    }
}

/// Whether `specialized` notes `PrecisionNote::SrgbTexelDecode` where texture slot 0 binds a
/// texture of `pf`: a clause testing the bound format's class alone, as the runtime reads it.
fn bound_note(specialized: &[SpecializedNote], pf: PixelFormat) -> bool {
    specialized
        .iter()
        .filter(|s| s.note == PrecisionNote::SrgbTexelDecode)
        .flat_map(|s| &s.when)
        .any(|c| {
            c.tests_bound_formats()
                && c.holds_of_bound(&|slot| {
                    assert_eq!(slot, TEXTURE);
                    Some(bound_texel_format(pf))
                })
        })
}

/// A kernel sampling a `dims`-dimensional texture (`texture1d`, `texture2d`, or `texture3d`) slot 0
/// of `half` texels (`float`) through the constexpr sampler of `words` at buffer 1's lane `tid`
/// (a `float2`, its `x` for 1D; a `float4`'s `xyz` for 3D), storing the `half4` (`float4`) to
/// buffer 0.
fn filter_kernel(words: u64, dims: u8, half: bool) -> String {
    filter_kernel_through(Some(words), dims, half)
}

/// [`filter_kernel`] through the constexpr sampler of `words`, or with none through the runtime
/// `[[sampler(0)]]`.
fn filter_kernel_through(words: Option<u64>, dims: u8, half: bool) -> String {
    let (texel, vec4, size, out_ty, suffix) = if half {
        ("half", "<4 x half>", 8, "half4", "v4f16")
    } else {
        ("float", "<4 x float>", 16, "float4", "v4f32")
    };
    let (tex_ty, name, args, call_args) = match dims {
        1 => (
            format!("texture1d<{texel}, sample>"),
            "1d",
            "ptr addrspace(1), ptr addrspace(2), float, i1, i32, i1, float, float, i32",
            "float %u, i1 false, i32 0, i1 false, float 0.0, float 0.0, i32 0",
        ),
        2 => (
            format!("texture2d<{texel}, sample>"),
            "2d",
            "ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32",
            "<2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0",
        ),
        _ => (
            format!("texture3d<{texel}, sample>"),
            "3d",
            "ptr addrspace(1), ptr addrspace(2), <3 x float>, i1, <3 x i32>, i1, float, float, i32",
            "<3 x float> %uvw, i1 true, <3 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0",
        ),
    };
    let (load, coord_size, coord_ty) = if dims == 3 {
        (
            "  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %c, i64 %i\n  %c4 = load <4 x float>, ptr addrspace(1) %p, align 16\n  %uvw = shufflevector <4 x float> %c4, <4 x float> zeroinitializer, <3 x i32> <i32 0, i32 1, i32 2>\n",
            16,
            "float4",
        )
    } else {
        (
            "  %p = getelementptr inbounds <2 x float>, ptr addrspace(1) %c, i64 %i\n  %uv = load <2 x float>, ptr addrspace(1) %p, align 8\n  %u = extractelement <2 x float> %uv, i64 0\n",
            8,
            "float2",
        )
    };
    // The constexpr sampler's state and its metadata, or the runtime sampler's parameter.
    let (state, sampler, param, states, args_md, mds) = match words {
        Some(words) => (
            format!("@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {words}, i64 0], align 8\n\n"),
            "@__air_sampler_state",
            "",
            "!air.sampler_states = !{!7}\n",
            "!3, !4, !5, !6",
            "!6 = !{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}\n!7 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
        None => (
            String::new(),
            "%s",
            ", ptr addrspace(2) %s",
            "",
            "!3, !4, !5, !6, !7",
            "!6 = !{i32 3, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n!7 = !{i32 4, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tid\"}\n",
        ),
    };
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, ptr addrspace(1) %c{param}, i32 %tid) {{
  %i = zext i32 %tid to i64
{load}  %r = call {{ {vec4}, i8 }} @air.sample_texture_{name}.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) {sampler}, {call_args})
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align {size}
  ret void
}}

declare {{ {vec4}, i8 }} @air.sample_texture_{name}.{suffix}({args})

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{{args_md}}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"{tex_ty}", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 {size}, !"air.arg_type_align_size", i32 {size}, !"air.arg_type_name", !"{out_ty}", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 {coord_size}, !"air.arg_type_align_size", i32 {coord_size}, !"air.arg_type_name", !"{coord_ty}", !"air.arg_name", !"c"}}
{mds}"#
    )
}

/// The plan's word estimate bounds the emitted module of every access, with the slot in the
/// variant (decoding the twin's codes) and without (the device's decode), on every committed
/// profile, and every form validates there.
#[test]
fn the_srgb_estimate_bounds_its_emission() {
    let consts = FunctionConstantValues::defaults();
    let forms = [
        ("read half", Access::Read, true),
        ("read float", Access::Read, false),
        (
            "fetch half",
            Access::Sample(sampler_words(0, 0, true)),
            true,
        ),
        (
            "nearest float",
            Access::Sample(sampler_words(0, 0, false)),
            false,
        ),
        ("gather alpha half", Access::Gather(3), true),
        ("gather red float", Access::Gather(0), false),
        ("gather alpha float", Access::Gather(3), false),
        (
            "linear half",
            Access::Sample(sampler_words(1, 0, false)),
            true,
        ),
        (
            "linear float",
            Access::Sample(sampler_words(1, 0, false)),
            false,
        ),
        (
            "bicubic half",
            Access::Sample(sampler_words(2, 0, false)),
            true,
        ),
        (
            "bicubic float",
            Access::Sample(sampler_words(2, 0, false)),
            false,
        ),
        ("runtime half", Access::Runtime, true),
        ("runtime float", Access::Runtime, false),
    ];
    let mut kernels: Vec<(&str, String)> = forms
        .iter()
        .map(|(name, access, half)| (*name, kernel(*access, *half)))
        .collect();
    for (name, half) in [("linear 3d half", true), ("linear 3d float", false)] {
        kernels.push((name, filter_kernel(sampler_words(1, 0, true), 3, half)));
    }
    for (name, half) in [("runtime 3d half", true), ("runtime 3d float", false)] {
        kernels.push((name, filter_kernel_through(None, 3, half)));
    }
    for (name, k) in kernels {
        let Some(bc) = assemble(name, &k) else {
            return;
        };
        for ((stem, p), (twin, under)) in committed_profiles()
            .expect("profiles")
            .into_iter()
            .flat_map(|sp| {
                [
                    (sp.clone(), ("", inputs(false))),
                    (sp, (" twin", inputs(true))),
                ]
            })
        {
            let what = format!("{name}{twin} on {stem}");
            let entry = EntryRef::Name("k");
            let plan = plan_bitcode(&bc, entry, &consts, &p, &under, &mut meter())
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let a = compile_bitcode(&bc, entry, &consts, &p, &under, &mut meter(), SPIRV_WORDS)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = a.module.words.len() as u64;
            eprintln!("{what}: estimate {} emitted {words}", plan.estimate.words);
            assert!(
                plan.estimate.words >= words,
                "{what}: estimate {} < emitted {words}",
                plan.estimate.words
            );
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
        }
    }
}

// ---- The binder -------------------------------------------------------------------------------

fn devices() -> Vec<(String, Device)> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out = Vec::new();
    for e in std::fs::read_dir(&dir).expect("profiles") {
        let path = e.expect("entry").path();
        if path.extension().is_none_or(|x| x != "json") {
            continue;
        }
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = std::fs::read_to_string(&path).expect("profile");
        let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
        out.push((
            stem,
            Device::new(desc, DevicePolicy::default()).expect("device"),
        ));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// The kernel compiled into a compute pipeline under `inputs`.
fn pipeline(dev: &Device, bc: &[u8], inputs: &LowerInputs) -> ComputePipeline {
    let lib = Library::from_air(bc).expect("library");
    let f = lib.function("k").expect("function");
    let compiled = dev
        .compile_function(&FunctionDescriptor::new(f), inputs, &Budget::standard())
        .expect("compile");
    dev.compile_compute_pipeline(
        &ComputePipelineDescriptor::new(StageSource::Compiled(compiled)),
        &Budget::standard(),
    )
    .expect("pipeline")
}

/// A kernel's state: texture slot 0 bound in `pf` (or nothing), buffer 0 bound.
fn state(pf: Option<PixelFormat>) -> BindState {
    let mut t = StageTable::default();
    if let Some(pf) = pf {
        let mut facts = TextureFacts::new(TextureType::Type2D, pf);
        facts.usage = metal::TextureUsage::SHADER_READ;
        t.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
    }
    t.buffers.insert(
        0,
        BoundBuffer::Buffer {
            length: 4096,
            offset: 0,
            gpu_address: 0x1_0000,
            device_address: None,
            attribute_stride: None,
        },
    );
    let mut s = BindState::default();
    s.stages.insert(Stage::Kernel, t);
    s
}

/// The image views `plan` writes for texture slot `slot`, through a texture or a combined write.
fn slot_views(plan: &crate::bind::BindingPlan, slot: SlotRef) -> Vec<crate::bind::ImageViewDesc> {
    plan.descriptors
        .iter()
        .filter_map(|d| match &d.source {
            WriteSource::Texture { slot: s, view } if *s == slot => Some(*view),
            WriteSource::CombinedImageSampler {
                image: ImageSource::Texture { slot: s, view },
                ..
            } if *s == slot => Some(*view),
            _ => None,
        })
        .collect()
}

/// Whether `profile`'s device decodes through Metal's own texture unit.
fn metal_sampler(dev: &Device) -> bool {
    dev.profile()
        .fields()
        .enabled
        .properties
        .contains(&crate::device::requirements::VkProperty::MetalSampler)
}

/// Whether `p`'s stage notes `PrecisionNote::SrgbTexelDecode` where slot 0 binds a texture of
/// `pf`, as the runtime reads it at a dispatch (`StageModule::bound_format_clauses`).
fn notes_bound(p: &ComputePipeline, pf: PixelFormat) -> bool {
    !p.stage
        .bound_format_clauses(PrecisionNote::SrgbTexelDecode, &|_| {
            Some(bound_texel_format(pf))
        })
        .is_empty()
}

/// A slot bound in an 8-bit sRGB format joins the plan's variant exactly where the device holds
/// the format as its UNORM twin (`metal::texel_format::srgb_twin_on`), from the device's format
/// features alone. Where the device samples the format the slot joins nothing, every binding of
/// the slot views the image in that sRGB format (the device decodes it), and off Metal's own
/// texture unit the pipeline notes `PrecisionNote::SrgbTexelDecode` under the bound format, which
/// an `RGBA8Unorm` texture does not take. Where the device holds the twin, the variant's pipeline
/// plans the same key, every binding of the slot views the twin, and only a runtime sampler's
/// sample (the device's filter of the codes) notes `SrgbTexelDecode`, outright; each variant
/// validates. A device that samples neither form refuses the format. Every committed profile, for
/// a read, a nearest sample and a runtime sampler's sample.
#[test]
fn a_slot_bound_in_srgb_joins_the_variant_only_where_the_device_holds_its_twin() {
    use crate::device::requirements::FormatUsage as U;
    let forms = [
        ("read", Access::Read, false),
        ("nearest", Access::Sample(sampler_words(0, 0, false)), false),
        ("runtime", Access::Runtime, true),
    ];
    let (mut decoded, mut twinned) = (0, 0);
    for (name, access, filtered) in forms {
        let Some(bc) = assemble(name, &kernel(access, false)) else {
            return;
        };
        for (stem, dev) in devices() {
            let metal = metal_sampler(&dev);
            let plain = pipeline(&dev, &bc, &inputs(false));
            assert!(!notes_bound(&plain, PixelFormat::RGBA8Unorm), "{stem}");
            for pf in [
                PixelFormat::RGBA8UnormSRGB,
                PixelFormat::BGRA8UnormSRGB,
                PixelFormat::R8UnormSRGB,
                PixelFormat::RG8UnormSRGB,
            ] {
                let what = format!("{name} on {stem} {pf:?}");
                let shape = state(Some(pf)).shape();
                let sampled =
                    crate::metal::format::vk_format(pf, &[U::SampledImage].into(), dev.profile());
                let twin = srgb_twin_on(pf, dev.profile());
                assert!(!(twin && sampled.is_ok()), "{what}");
                let plan = match plain.plan(&dev, &shape) {
                    Ok(plan) => plan,
                    Err(crate::error::BindError::Unsupported(u))
                        if u.feature == crate::error::Feature::Format
                            && sampled.is_err()
                            && !twin =>
                    {
                        continue
                    }
                    Err(e) => panic!("{what}: {e}"),
                };
                let views = slot_views(&plan, TEXTURE);
                assert!(!views.is_empty(), "{what}");
                if let Ok(s) = sampled {
                    assert!(plan.variant.srgb_textures.is_empty(), "{what}");
                    for v in &views {
                        assert_eq!(v.format, s.format, "{what}");
                        assert_eq!(
                            v.format.block().and_then(|b| b.color),
                            Some(crate::vk::NumericFormat::Srgb),
                            "{what}"
                        );
                    }
                    assert_eq!(notes_bound(&plain, pf), !metal, "{what}");
                    decoded += 1;
                    continue;
                }
                assert!(twin, "{what}");
                assert_eq!(
                    plan.variant.srgb_textures,
                    BTreeSet::from([TEXTURE]),
                    "{what}"
                );
                let held = crate::metal::format::srgb_raw_twin(pf)
                    .and_then(crate::metal::format::direct_format)
                    .expect(&what);
                for v in &views {
                    assert_eq!(v.format, held, "{what}");
                }
                let key = plan.variant.lowering_inputs(Stage::Kernel);
                assert_eq!(key.srgb_textures, BTreeSet::from([TEXTURE]), "{what}");
                let variant = pipeline(&dev, &bc, &key);
                assert_eq!(
                    variant
                        .stage
                        .precision_notes
                        .contains(&PrecisionNote::SrgbTexelDecode),
                    filtered,
                    "{what}"
                );
                assert!(!notes_bound(&variant, pf), "{what}");
                let a = compile_bitcode(
                    &bc,
                    EntryRef::Name("k"),
                    &FunctionConstantValues::defaults(),
                    dev.profile(),
                    &key,
                    &mut meter(),
                    SPIRV_WORDS,
                )
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                link_and_validate(&a, dev.profile()).unwrap_or_else(|e| panic!("{what}: {e}"));
                let again = variant
                    .plan(&dev, &shape)
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(again.variant, plan.variant, "{what}");
                for v in slot_views(&again, TEXTURE) {
                    assert_eq!(v.format, held, "{what}");
                }
                twinned += 1;
                // Another format, or nothing, leaves the variant.
                for other in [Some(PixelFormat::RGBA8Unorm), None] {
                    let shape = state(other).shape();
                    let plan = variant.plan(&dev, &shape).expect("plan");
                    assert!(plan.variant.srgb_textures.is_empty(), "{what} {other:?}");
                }
            }
        }
    }
    assert!(decoded > 0, "no committed profile samples an sRGB format");
    assert!(
        twinned > 0,
        "no committed profile samples an sRGB format only through its twin"
    );
}

/// An ASTC sRGB slot samples the device's decode wherever the device samples a format of it: its
/// own block where the device has ASTC (ANV, MoltenVK), its decoded twin's `R8G8B8A8_SRGB` view
/// elsewhere (`bind::AstcDecode`). It joins no variant, and off Metal's own texture unit the
/// pipeline notes `PrecisionNote::SrgbTexelDecode` under the bound format.
#[test]
fn an_astc_srgb_slot_samples_the_devices_decode() {
    let Some(bc) = assemble("astc read", &kernel(Access::Read, true)) else {
        return;
    };
    let (mut decoded, mut native) = (0, 0);
    for (stem, dev) in devices() {
        let metal = metal_sampler(&dev);
        let plain = pipeline(&dev, &bc, &inputs(false));
        for pf in [PixelFormat::ASTC4x4SRGB, PixelFormat::ASTC12x10SRGB] {
            let what = format!("{stem} {pf:?}");
            let sampled = crate::metal::format::vk_format(
                pf,
                &[FormatUsage::SampledImage].into(),
                dev.profile(),
            )
            .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert!(!srgb_twin_on(pf, dev.profile()), "{what}");
            let plan = plain.plan(&dev, &state(Some(pf)).shape()).expect(&what);
            assert!(plan.variant.srgb_textures.is_empty(), "{what}");
            let views = slot_views(&plan, TEXTURE);
            assert!(!views.is_empty(), "{what}");
            for v in &views {
                assert_eq!(v.format, sampled.format, "{what}");
            }
            if sampled.emulation == crate::metal::format::FormatEmulation::Decoded {
                assert_eq!(sampled.format, crate::vk::Format::R8G8B8A8_SRGB, "{what}");
                decoded += 1;
            } else {
                native += 1;
            }
            assert_eq!(notes_bound(&plain, pf), !metal, "{what}");
        }
    }
    assert!(decoded > 0, "no committed profile holds an ASTC sRGB twin");
    assert!(
        native > 0,
        "no committed profile samples an ASTC sRGB block"
    );
}

/// A `half` sample the device decodes and filters (a linear one with a linear mip filter)
/// converts the device's `f32` texel by the bound format's class, the conversion's `OTHER` arm
/// for an 8-bit sRGB format. On MoltenVK, whose decode and filter are Metal's own texture unit's,
/// that `f32` is Metal's `float` value, whose rounding to `half` is Metal's
/// (`metals_srgb_float_steps_round_to_their_half_steps`): the pipeline takes no
/// `PrecisionNote::HalfTexelConversion` under the format and requires `VkProperty::MetalSampler`,
/// and the note holds under a format of no measured family. On lavapipe, whose decode and
/// conversion are the device's, the note holds under every format.
#[test]
fn metals_own_srgb_decode_converts_to_half_as_metal_does() {
    use crate::device::requirements::VkProperty;
    let Some(bc) = assemble(
        "mip linear half",
        &kernel(Access::Sample(sampler_words(1, 2, false)), true),
    ) else {
        return;
    };
    let mut seen = 0;
    for (stem, dev) in devices() {
        let metal = match stem.as_str() {
            "lavapipe" => false,
            "moltenvk-m3" => true,
            _ => continue,
        };
        seen += 1;
        let p = pipeline(&dev, &bc, &inputs(false));
        assert_eq!(
            p.requirements
                .properties
                .contains(&VkProperty::MetalSampler),
            metal,
            "{stem}"
        );
        for pf in [PixelFormat::RGBA8UnormSRGB, PixelFormat::BGRA8UnormSRGB] {
            let plan = p.plan(&dev, &state(Some(pf)).shape()).expect("plan");
            assert!(plan.variant.srgb_textures.is_empty(), "{stem} {pf:?}");
            let notes = p.stage.precision_notes_under(&plan.specialization);
            assert_eq!(
                notes.contains(&PrecisionNote::HalfTexelConversion),
                !metal,
                "{stem} {pf:?}: {notes:?}"
            );
        }
        if !metal {
            assert!(
                p.stage
                    .precision_notes
                    .contains(&PrecisionNote::HalfTexelConversion),
                "{stem}: {:?}",
                p.stage.precision_notes
            );
            assert!(
                !p.stage
                    .specialized_notes
                    .iter()
                    .any(|s| s.note == PrecisionNote::HalfTexelConversion),
                "{stem}: {:?}",
                p.stage.specialized_notes
            );
            continue;
        }
        // The module keeps the note for a format of no measured family: its slot's
        // `UnmeasuredTexels` specialization, 1 for such a format.
        let other = crate::metal::texel_format::TexelFormat::OTHER.0;
        let value = |key: &SpecKey| match key {
            SpecKey::UnmeasuredTexels(_) => 1,
            _ => other,
        };
        let held = p.stage.specialized_notes.iter().any(|s| {
            s.note == PrecisionNote::HalfTexelConversion
                && s.when.iter().any(|c| {
                    c.all
                        .iter()
                        .all(|t| ((value(&t.key) & t.mask) == t.bits) == t.equal)
                })
        });
        assert!(held, "{stem}: {:?}", p.stage.specialized_notes);
    }
    assert_eq!(seen, 2);
}

/// A render pipeline: a vertex function placing vertex `vid`, and a fragment function returning
/// texel `(0, 0)` of its `texture2d<float, read>` slot 0.
const RENDER: &str = r#"define <{ <4 x float> }> @v(i32 noundef %vid) {
  %f = uitofp i32 %vid to float
  %p = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %p, 0
  ret <{ <4 x float> }> %o
}
define <4 x float> @f(<4 x float> noundef %pos, ptr addrspace(1) %t) {
  %s = tail call ptr addrspace(2) @air.get_read_sampler()
  %r = call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %t, ptr addrspace(2) %s, <2 x i32> zeroinitializer, <2 x i32> zeroinitializer, i32 0, i32 1)
  %c = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %c
}
declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
!air.vertex = !{!20}
!air.fragment = !{!0}
!air.version = !{!9}
!air.language_version = !{!10}
!0 = !{ptr @f, !1, !2}
!1 = !{!11}
!11 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!2 = !{!12, !3}
!12 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!3 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!9 = !{i32 2, i32 8, i32 0}
!10 = !{!"Metal", i32 4, i32 0, i32 0}
!20 = !{ptr @v, !21, !23}
!21 = !{!22}
!22 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!23 = !{!24}
!24 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
"#;

/// A render pipeline's fragment slot bound in an 8-bit sRGB format the device holds as its UNORM
/// twin (`RG8Unorm_sRGB` on lavapipe) joins the plan's variant; `compile_render_variant` compiles
/// it, its plan states the same key, and the fragment's binding views the twin. A format the
/// device samples (`RGBA8Unorm_sRGB` everywhere, `RG8Unorm_sRGB` on MoltenVK and RADV) joins
/// nothing, and the binding views the image in that sRGB format.
#[test]
fn a_fragment_slot_bound_in_a_twin_joins_the_render_variant() {
    use crate::pipeline::descriptor::ColorAttachmentDescriptor;
    use crate::pipeline::RenderPipelineDescriptor;
    let Some(bc) = assemble("render", RENDER) else {
        return;
    };
    let fragment = SlotRef {
        stage: Stage::Fragment,
        ..TEXTURE
    };
    let state = |pf: PixelFormat| {
        let mut f = StageTable::default();
        let mut facts = TextureFacts::new(TextureType::Type2D, pf);
        facts.usage = metal::TextureUsage::SHADER_READ;
        f.textures.insert(
            0,
            BoundTexture::Texture {
                facts,
                resource_id: None,
            },
        );
        let mut s = BindState::default();
        s.stages.insert(Stage::Fragment, f);
        s
    };
    let (mut ran, mut twinned) = (0, 0);
    for (stem, dev) in devices() {
        if !["lavapipe", "moltenvk-m3", "amd-rdna3-radv"].contains(&stem.as_str()) {
            continue;
        }
        ran += 1;
        let lib = Library::from_air(&bc).expect("library");
        let v = lib.function("v").expect("vertex");
        let f = lib.function("f").expect("fragment");
        let mut d =
            RenderPipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(v)));
        d.fragment = Some(StageSource::Function(FunctionDescriptor::new(f)));
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        for pf in [PixelFormat::RGBA8UnormSRGB, PixelFormat::RG8UnormSRGB] {
            let what = format!("{stem} {pf:?}");
            let shape = state(pf).shape();
            let plan = p.plan(&dev, &shape).expect("plan");
            let views = slot_views(&plan, fragment);
            assert_eq!(views.len(), 1, "{what}: {views:?}");
            if !srgb_twin_on(pf, dev.profile()) {
                assert!(plan.variant.srgb_textures.is_empty(), "{what}");
                let sampled = crate::metal::format::vk_format(
                    pf,
                    &[FormatUsage::SampledImage].into(),
                    dev.profile(),
                )
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(views[0].format, sampled.format, "{what}");
                continue;
            }
            assert_eq!(
                plan.variant.srgb_textures,
                BTreeSet::from([fragment]),
                "{what}"
            );
            let held = crate::metal::format::srgb_raw_twin(pf)
                .and_then(crate::metal::format::direct_format);
            assert_eq!(Some(views[0].format), held, "{what}");
            let variant = dev
                .compile_render_variant(&d, &plan.variant, &Budget::standard())
                .unwrap_or_else(|e| panic!("{what} variant: {e}"));
            let again = variant.plan(&dev, &shape).expect("variant plan");
            assert_eq!(again.variant, plan.variant, "{what}: a fixed point");
            assert_eq!(slot_views(&again, fragment), views, "{what}");
            twinned += 1;
        }
    }
    assert_eq!(ran, 3);
    assert!(twinned > 0, "lavapipe holds no twin");
}
