//! `DevicePolicy`: the consumer's product choices that, together with a `DeviceDescription`,
//! decide a `DeviceProfile`.

use crate::device::profile::{GuestIsolation, HandleDomain, PointerDomain};

/// Policy inputs of `Device::new`. Every field but `heap_compression`, `completion`,
/// `batch_order`, `render_lowering` and `host_page_textures` can change emitted code, so the
/// profile carries it (containment as the encoding it selects) and its fingerprint covers it; those
/// five are the runtime's alone (a heap's compressed texture's image, when the host sees a command
/// buffer complete, how a batch begins after the earlier ones of its queue, where a render
/// encoder's calls are lowered, and what holds a sampled texture over host pages), and no profile
/// carries them.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DevicePolicy {
    /// How pointer values resolve to resources (v1 ships `ResidentTable` only).
    pub pointer_domain: PointerDomain,
    /// How resource handles resolve (v1 ships `Table` only).
    pub handle_domain: HandleDomain,
    /// Whether accesses are contained to the resources the binder supplied.
    pub containment: ContainmentPolicy,
    /// Whether guest shaders are isolated from each other's resources.
    pub isolation: GuestIsolation,

    /// Whether a heap's colour texture that Metal keeps losslessly compressed, on a driver whose
    /// compressed images hide writes through an alias, is the driver's compressed image (the
    /// default) or one the driver keeps uncompressed in the heap's bytes. The runtime's choice
    /// alone: no emitted code changes.
    pub heap_compression: HeapCompressionPolicy,
    /// Whether command buffers committed on one `VkQueue` that the runtime's scheduler finds ready
    /// together go to the driver in one kernel submission, each seen complete once the last of
    /// them completed (the default, on a driver that merges such submissions), or each in a
    /// submission of its own, seen complete at its own end. The runtime's choice alone: no
    /// emitted code changes.
    pub completion: CompletionPolicy,
    /// Whether a batch of command buffers on one `VkQueue` begins after the batches before it
    /// there by their ends (the default, on a driver measured to run no two kernel submissions of
    /// one queue side by side), or by a barrier for each access that depends on theirs. The
    /// runtime's choice alone: no emitted code changes.
    pub batch_order: BatchOrderPolicy,
    /// Whether a render encoder's call is lowered inside the call, which returns every refusal
    /// (the default), or after it returned, a refusal only the lowering finds failing its command
    /// buffer. The runtime's choice alone: no emitted code changes.
    pub render_lowering: RenderLoweringPolicy,
    /// Whether a texture that only samples or reads, over a no-copy buffer's host pages or over a
    /// buffer the runtime placed in fresh host pages, is an image aliasing those pages in place
    /// (the default, where the device's own check shows its first move keeps every byte), or a
    /// shadow the encoders copy the texels into and back. The runtime's choice alone: no emitted
    /// code changes.
    pub host_page_textures: HostPageTexturePolicy,
}

impl DevicePolicy {
    /// The untrusted-guest production policy: containment and isolation on, the driver's
    /// compressed image of a heap's texture Metal keeps compressed, command buffers ready together
    /// in one kernel submission, batches of one queue ordered by their ends where the driver runs
    /// them one after another, and a sampled
    /// texture over host pages in place where the device's check passed.
    pub const UNTRUSTED: DevicePolicy = DevicePolicy {
        pointer_domain: PointerDomain::ResidentTable,
        handle_domain: HandleDomain::Table,
        containment: ContainmentPolicy::Contained,
        isolation: GuestIsolation::On,
        heap_compression: HeapCompressionPolicy::Compressed,
        completion: CompletionPolicy::Grouped,
        batch_order: BatchOrderPolicy::Ends,
        render_lowering: RenderLoweringPolicy::AtCall,
        host_page_textures: HostPageTexturePolicy::InPlace,
    };

    /// This policy with Metal's answer wherever the choices offer it: a heap's texture whose bytes
    /// alias the heap's
    /// as written ([`HeapCompressionPolicy::Uncompressed`]), each command buffer seen complete
    /// at its own end ([`CompletionPolicy::Exact`]), each batch ordered after the earlier ones of
    /// its queue by
    /// the accesses it depends on alone ([`BatchOrderPolicy::Exact`]), each render encoder
    /// call lowered inside the call, which returns every refusal
    /// ([`RenderLoweringPolicy::AtCall`]), and every texture over host pages a shadow, whose
    /// copies touch its texels alone ([`HostPageTexturePolicy::Shadow`]). Every other field is
    /// kept.
    pub const fn exact_answers(self) -> DevicePolicy {
        DevicePolicy {
            heap_compression: HeapCompressionPolicy::Uncompressed,
            completion: CompletionPolicy::Exact,
            batch_order: BatchOrderPolicy::Exact,
            render_lowering: RenderLoweringPolicy::AtCall,
            host_page_textures: HostPageTexturePolicy::Shadow,
            ..self
        }
    }
}

impl Default for DevicePolicy {
    /// [`DevicePolicy::UNTRUSTED`].
    fn default() -> DevicePolicy {
        DevicePolicy::UNTRUSTED
    }
}

/// The containment choice. The encoding (robustness or in-shader checks) is selected by
/// `Device::new` from the enabled robustness features; the policy only says whether to contain.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContainmentPolicy {
    /// Every access stays inside the resources the binder supplied (the default).
    Contained,
    /// No containment. Only valid with `GuestIsolation::Off`; recorded in the fingerprint.
    UnsafeOff,
}

/// The image of a heap's colour texture that Metal keeps losslessly compressed (with
/// `allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView` or `ShaderAtomic` usage, and a
/// level 0 of 16 texels or more a side), on a driver whose compressed images may keep their texels
/// apart from the memory they are bound to (`DriverFact::CompressedImagesHideAliasedWrites`: ANV
/// on Intel's GPUs). Metal's own texture shows no aliased write as written either: an M3 Max read
/// opaque magenta (`0xffff00ff`) in every texel of drawn content a buffer's fill over the
/// texture's bytes covered, and the fill's bytes where the texture held only a clear. A texture
/// Metal keeps uncompressed takes the uncompressed image under either value. The runtime's choice
/// alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HeapCompressionPolicy {
    /// The driver's compressed optimal image (the default), its fast clears, compression and
    /// tiling kept: a write through a buffer or texture over its bytes may leave its texels as
    /// they were or change them otherwise than Metal's, and its writes may show in those bytes
    /// compressed. Announced `heap-texture-compressed` where such a texture is made.
    #[default]
    Compressed,
    /// An image the driver keeps uncompressed in the heap's bytes, as for a texture Metal keeps
    /// uncompressed: host-transfer usage where the device enabled `hostImageCopy`, else a linear
    /// image, else the optimal one (each announced as such), so a write through a buffer over the
    /// texture's bytes shows in its texels as written.
    Uncompressed,
}

/// When the host sees a command buffer complete, on a driver that merges a submission signalling
/// nothing into the next one in the same `vkQueueSubmit` (Mesa's common queue,
/// `vk_queue_submits_merge`: ANV, RADV, lavapipe, NVK, Turnip and the other Mesa drivers of a kernel
/// submission), where each kernel submission costs CPU time for every `VkDeviceMemory` of the
/// device (ANV on i915 lists each in every execbuffer). Elsewhere every command buffer goes in a
/// submission of its own under either value. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CompletionPolicy {
    /// Up to four batches of command buffers on one `VkQueue` that the scheduler finds ready
    /// together, none of which signals a binary semaphore, takes a fence, moves the caller's
    /// images or does first work, and none but the first of which waits for anything not seen
    /// complete, go in one `vkQueueSubmit`, and only the last signals, the queue's timeline and
    /// every event they signal (the default): one kernel submission for all of them. The commands,
    /// their barriers and their GPU times are unchanged; each but the last is seen complete (its
    /// status, handlers and waits) once the last completed, and an event's values they signal are
    /// reached then, late by the GPU time of the batches after it, never early and in commit
    /// order. The submission checks of up to three batches right behind a queue's next batch run
    /// with its own, in commit order, so checked command buffers are found ready together: such a
    /// check runs before the previous batch's submission rather than after it. Announced
    /// `grouped-completion` at the first such submission.
    #[default]
    Grouped,
    /// Each batch in a submission of its own that signals its own values, so each command buffer
    /// is seen complete, and its event signals land, at its own end, as Metal's are; each batch's
    /// submission checks run once the batch before it was submitted.
    Exact,
}

/// How a batch of command buffers on one `VkQueue` (the work the runtime submits for a command
/// buffer between its event waits and signals) begins after the earlier batches there. Metal
/// orders a command buffer after the earlier ones of its queue only where its work depends on
/// theirs, so an independent one may run beside the one before it; it writes its `GPUStartTime`
/// as its work begins. The runtime's choice alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BatchOrderPolicy {
    /// By their ends (the default) on a driver measured to run no two kernel submissions of one
    /// queue side by side, with or without a barrier between them, where it runs the same work
    /// recorded in one command buffer side by side (ANV, RADV and NVIDIA's driver): every batch
    /// ends with a barrier that orders all its commands before every command after it on its
    /// queue, so a later batch plans no barrier for the accesses of earlier ones, and its first
    /// encoder's GPU start time is written once the work before it ended, at the bottom of the
    /// pipe. Each batch then still starts once the one before it ended, as those drivers start it
    /// anyway. On any other driver, as [`Exact`](Self::Exact).
    #[default]
    Ends,
    /// A barrier at a batch's start for each access that depends on an earlier batch's, as for
    /// the accesses within a batch, and its first encoder's GPU start time written as it begins,
    /// at the top of the pipe, where the driver may run it beside the batch before it.
    Exact,
}

/// Where a render encoder's calls are lowered: where each call's commands, descriptors, pipelines
/// and resources are worked out and recorded. Metal's encoder checks a call as it is made and
/// returns; its GPU work, and any failure found there, comes after. The runtime's choice alone: no
/// emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RenderLoweringPolicy {
    /// After the call returned: the call makes Metal's checks of its arguments and
    /// of the encoder's state, copies what it was lent and holds what it names, and returns; the
    /// encoder lowers it later, in order, before its command buffer's submission (on the
    /// runtime's lowering thread while the program goes on, the encoder's end too, and the rest
    /// at what follows the end in the command buffer; `commit` makes its checks and returns, its
    /// own work run after the end, and later commits behind it). A refusal only
    /// the lowering finds fails the command buffer
    /// (`CommandBufferError::CallRefused`, naming the call), which then runs nothing of that
    /// encoder or of any after it.
    Worker,
    /// Inside the call, as each call is made (the default): every refusal, the lowering's too,
    /// returns from the call itself.
    #[default]
    AtCall,
}

/// What holds a texture over host pages whose declared usage only samples or reads it (no
/// `RenderTarget`, `ShaderWrite` or `ShaderAtomic`): over a no-copy buffer's pages (imported by
/// their pointer or as dma-bufs) or over a buffer the runtime placed in fresh host pages it
/// imported. Vulkan makes an image bound to imported memory start `UNDEFINED`
/// (`VUID-VkImageCreateInfo-pNext-01443`), and lets its first layout transition read and write
/// every byte bound to it, where Metal keeps every CPU write to those bytes. The runtime's choice
/// alone: no emitted code changes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HostPageTexturePolicy {
    /// A linear image aliasing the pages (the default), on a device whose own check, run once per
    /// driver and release, shows that this first move of such an image leaves every byte alone
    /// while the CPU writes beside it; and where the device lays the image out at the buffer's
    /// row pitch. Elsewhere, and for a texture that renders or writes, the shadow below. A
    /// texture made before the check's verdict is known is a shadow too.
    #[default]
    InPlace,
    /// A shadow on every device: an image of the runtime's own memory the encoders copy the
    /// texels into before each use and merge back after each write, whose copies touch the
    /// texels alone.
    Shadow,
}
