//! The compiler-word-bound fit: a pipeline whose largest stage module passes the device's
//! compiler word bound ([`DeviceProfile::module_word_bound`], where the driver faults compiling a
//! module past it), translated again from its functions with the device policy's module-growing
//! exact forms made native, recording the forms dropped
//! (`RuntimeFact::ExactFormsOverCompilerLimit`). Every pipeline kind is fitted by the one rule: a
//! render pipeline of vertex and fragment stages, a mesh render pipeline (object, mesh and
//! fragment stages, or the compute and synthesized vertex stages of an emulated mesh), a compute
//! pipeline and a tile render pipeline. Each kind's compile from `Function` stages fits itself
//! ([`Device::compile_render_variant`], [`Device::compile_mesh_variant`],
//! [`Device::compile_compute_pipeline`], [`Device::compile_tile_pipeline`]); a pipeline linked from
//! `Compiled` stages is fitted from the descriptor of its functions by the kind's `fit_*` call.

use crate::budget::{Budget, Meter};
use crate::cache::VariantKey;
use crate::device::policy::{
    DevicePolicy,
};
use crate::device::{mechanism, Device, DeviceProfile};
use crate::error::Error;
use crate::pipeline::artifact::{ComputePipeline, RenderPipeline, RuntimeDependency};
use crate::pipeline::compile::guarded;
use crate::pipeline::descriptor::{
    ComputePipelineDescriptor, MeshPipelineDescriptor, RenderPipelineDescriptor, StageSource,
    TileRenderPipelineDescriptor,
};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;

#[cfg(test)]
mod tests;

impl Device {
    /// `pipeline`, [`Device::compile_render_variant`]'s compile of `d` under `v` (whose stages may
    /// be `Compiled`), held to the device's compiler word bound
    /// ([`DeviceProfile::module_word_bound`]): where the driver faults compiling a module past it
    /// and `pipeline`'s largest stage passes it, `d`'s stages translated again with the device
    /// policy's module-growing exact answers made native, recording the forms dropped
    /// (`RuntimeFact::ExactFormsOverCompilerLimit`) and keeping `pipeline`'s key and profile
    /// fingerprint, so a cache finds it where it would find `pipeline`. The stages carry the
    /// reduced forms' own precision notes. `pipeline` itself where the device has no bound, it
    /// fits, or a stage of `d` is `Compiled` (only a `Function` stage can be translated again):
    /// a caller that compiled its stages ahead must pass the descriptor of their functions here,
    /// or the module goes to the driver unreduced.
    pub fn fit_render_variant(
        &self,
        d: &RenderPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
        pipeline: RenderPipeline,
    ) -> Result<RenderPipeline, Error> {
        guarded(|| self.fit_render_within(d, v, b, pipeline, self.profile().module_word_bound()))
    }

    /// `pipeline`, [`Device::compile_mesh_variant`]'s compile of `d` under `v` (whose stages may be
    /// `Compiled`), held to the device's compiler word bound as
    /// [`Device::fit_render_variant`] holds a render pipeline: its object, mesh and fragment
    /// stages translated again from `d`'s functions with the exact forms dropped where its largest
    /// stage (the synthesized vertex stage of an emulated mesh among them) passes the bound, under
    /// `pipeline`'s key. `pipeline` itself where the device has no bound, it fits, or a stage of
    /// `d` is `Compiled`.
    pub fn fit_mesh_variant(
        &self,
        d: &MeshPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
        pipeline: RenderPipeline,
    ) -> Result<RenderPipeline, Error> {
        guarded(|| self.fit_mesh_within(d, v, b, pipeline, self.profile().module_word_bound()))
    }

    /// `pipeline`, [`Device::compile_compute_pipeline`]'s compile of `d` (whose kernel may be
    /// `Compiled`), its kernel compiled under `inputs`, held to the device's compiler word bound as
    /// [`Device::fit_render_variant`] holds a render pipeline: `d`'s kernel function translated
    /// again under `inputs` with the exact forms dropped where its module passes the bound, under
    /// `pipeline`'s key. `inputs` are those the kernel was compiled under: a `Function` kernel's
    /// in [`Device::compile_compute_pipeline`] are `d`'s stage-input descriptor and
    /// `maxCallStackDepth`, and a kernel compiled ahead ([`Device::compile_function`]) is fitted
    /// under the inputs given there. `pipeline` itself where the device has no bound, it fits, or
    /// `d`'s kernel is `Compiled`.
    pub fn fit_compute_pipeline(
        &self,
        d: &ComputePipelineDescriptor,
        inputs: &LoweringInputs,
        b: &Budget,
        pipeline: ComputePipeline,
    ) -> Result<ComputePipeline, Error> {
        let bound = self.profile().module_word_bound();
        guarded(|| self.fit_compute_within(d, inputs, b, pipeline, bound))
    }

    /// `pipeline`, [`Device::compile_tile_pipeline`]'s compile of `d` (whose tile function may be
    /// `Compiled`), its tile function compiled under `inputs`, held to the device's compiler word
    /// bound as [`Device::fit_compute_pipeline`] holds a compute pipeline: `inputs` are those the
    /// tile function was compiled under (in [`Device::compile_tile_pipeline`], `d`'s color
    /// attachment formats, `threadgroupSizeMatchesTileSize` and `maxCallStackDepth`). `pipeline`
    /// itself where the device has no bound, it fits, or `d`'s tile function is `Compiled`.
    pub fn fit_tile_pipeline(
        &self,
        d: &TileRenderPipelineDescriptor,
        inputs: &LoweringInputs,
        b: &Budget,
        pipeline: ComputePipeline,
    ) -> Result<ComputePipeline, Error> {
        let bound = self.profile().module_word_bound();
        guarded(|| self.fit_tile_within(d, inputs, b, pipeline, bound))
    }

    /// [`Device::fit_render_variant`] under `bound` in place of the device's own: a test pins a
    /// low bound without a device that carries the workaround.
    pub(crate) fn fit_render_within(
        &self,
        d: &RenderPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
        pipeline: RenderPipeline,
        bound: Option<u32>,
    ) -> Result<RenderPipeline, Error> {
        let translatable = function(&d.vertex) && d.fragment.as_ref().is_none_or(function);
        self.held_to(pipeline, bound, translatable, |device| {
            device.render_pipeline(d, v, &mut Meter::new(b))
        })
    }

    /// [`Device::fit_mesh_variant`] under `bound` in place of the device's own.
    pub(crate) fn fit_mesh_within(
        &self,
        d: &MeshPipelineDescriptor,
        v: &VariantKey,
        b: &Budget,
        pipeline: RenderPipeline,
        bound: Option<u32>,
    ) -> Result<RenderPipeline, Error> {
        let translatable = d.object.as_ref().is_none_or(function)
            && function(&d.mesh)
            && d.fragment.as_ref().is_none_or(function);
        self.held_to(pipeline, bound, translatable, |device| {
            device.mesh_pipeline(d, v, &mut Meter::new(b))
        })
    }

    /// [`Device::fit_compute_pipeline`] under `bound` in place of the device's own.
    pub(crate) fn fit_compute_within(
        &self,
        d: &ComputePipelineDescriptor,
        inputs: &LoweringInputs,
        b: &Budget,
        pipeline: ComputePipeline,
        bound: Option<u32>,
    ) -> Result<ComputePipeline, Error> {
        self.held_to(pipeline, bound, function(&d.kernel), |device| {
            device.compute_pipeline(d, inputs, &mut Meter::new(b))
        })
    }

    /// [`Device::fit_tile_pipeline`] under `bound` in place of the device's own.
    pub(crate) fn fit_tile_within(
        &self,
        d: &TileRenderPipelineDescriptor,
        inputs: &LoweringInputs,
        b: &Budget,
        pipeline: ComputePipeline,
        bound: Option<u32>,
    ) -> Result<ComputePipeline, Error> {
        self.held_to(pipeline, bound, function(&d.tile), |device| {
            device.tile_pipeline(d, inputs, &mut Meter::new(b))
        })
    }

    /// `pipeline` held to `bound`, every kind by the one rule: where its stages can be translated
    /// again (`translatable`) and its largest stage passes `bound`, `again` (the pipeline's compile
    /// on a device of a reduced policy) under every module-growing exact answer native, with the
    /// forms it dropped recorded and `pipeline`'s key and profile fingerprint kept. A reduction
    /// that changes nothing of the policy is skipped.
    fn held_to<P: Fitted>(
        &self,
        pipeline: P,
        bound: Option<u32>,
        translatable: bool,
        again: impl Fn(&Device) -> Result<P, Error>,
    ) -> Result<P, Error> {
        if !bound.is_some_and(|w| translatable && pipeline.largest_stage_words() > w) {
            return Ok(pipeline);
        }
        let base = *self.policy();
        let policy = native_forms(base);
        if policy == base {
            return Ok(pipeline);
        }
        // The lowering reads the profile, never the policy, and `Device::new` is the profile's
        // one producer: the description kept, the policy reduced.
        let device = Device::new(self.description().clone(), policy)?;
        let mut reduced = again(&device)?;
        reduced
            .dependencies_mut()
            .push(dropped_forms(base, &policy, device.profile()));
        Ok(reduced.named_as(&pipeline))
    }
}

/// Whether a stage can be translated again: a `Function` stage, not one taken `Compiled`.
fn function(source: &StageSource) -> bool {
    matches!(source, StageSource::Function(_))
}

/// A compiled pipeline the compiler word bound holds: a render pipeline (vertex or mesh) or a
/// compute pipeline (kernel or tile).
trait Fitted: Sized {
    /// The word count of its largest stage module: what a device's compiler word bound holds
    /// ([`DeviceProfile::module_word_bound`]). A pipeline with no stage counts as 0.
    fn largest_stage_words(&self) -> u32;
    /// Its runtime dependencies, where the fit records the forms it dropped.
    fn dependencies_mut(&mut self) -> &mut Vec<RuntimeDependency>;
    /// `self`, named as `compiled` is (its key and profile fingerprint), so a cache stores and
    /// finds it where it would `compiled`.
    fn named_as(self, compiled: &Self) -> Self;
}

impl Fitted for RenderPipeline {
    fn largest_stage_words(&self) -> u32 {
        self.stages
            .iter()
            .map(|s| words(&s.spirv))
            .max()
            .unwrap_or(0)
    }

    fn dependencies_mut(&mut self) -> &mut Vec<RuntimeDependency> {
        &mut self.dependencies
    }

    fn named_as(self, compiled: &RenderPipeline) -> RenderPipeline {
        RenderPipeline {
            key: compiled.key,
            profile: compiled.profile,
            ..self
        }
    }
}

impl Fitted for ComputePipeline {
    fn largest_stage_words(&self) -> u32 {
        words(&self.stage.spirv)
    }

    fn dependencies_mut(&mut self) -> &mut Vec<RuntimeDependency> {
        &mut self.dependencies
    }

    fn named_as(self, compiled: &ComputePipeline) -> ComputePipeline {
        ComputePipeline {
            key: compiled.key,
            profile: compiled.profile,
            ..self
        }
    }
}

/// A module's word count, saturated.
fn words(spirv: &[u32]) -> u32 {
    u32::try_from(spirv.len()).unwrap_or(u32::MAX)
}

/// The compiler-word-bound fallback's policy: `base` with every module-growing exact answer (the
/// fast default's forms) set to its native, module-smaller answer. It does not touch the policy's
/// other choices (pointer and handle domains, containment, isolation, framebuffer fetch, or the
/// runtime's choices, which change no module), so a reduced device keeps them.
fn native_forms(base: DevicePolicy) -> DevicePolicy {
    DevicePolicy {
        nan_and_min_max: NanMinMaxPolicy::Native,
        ..base
    }
}

/// Which of the consumer's exact forms the fallback dropped to fit the bound, as the dependency a
/// fitted pipeline records: each field of [`RuntimeFact::ExactFormsOverCompilerLimit`] is set
/// where `base`'s form differs from the reduced `policy`'s. The consumer announces it.
fn dropped_forms(
    base: DevicePolicy,
    policy: &DevicePolicy,
    reduced: &DeviceProfile,
) -> RuntimeDependency {
    let fact = RuntimeFact::ExactFormsOverCompilerLimit {
    };
    RuntimeDependency {
        fact,
        mechanism: mechanism(&fact, reduced),
    }
}
