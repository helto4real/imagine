//! Qwen-specific budgets, inventory and compatibility names.
pub use crate::components::offload::*;
use crate::{
    checkpoint::PreparedParameters,
    native::{Error, Kitchen},
};
use tch::Device;
pub type StaticOffload<B = super::FluxSampling> = crate::components::StaticOffload<B>;
/// BaseModel.memory_required, QwenImage21 factor=6, BF16 + PyTorch flash SDPA,
/// no reference latents. The batch doubled by estimate_memory is explicit.
pub fn qwen_sampling_memory(batch: i64, height: i64, width: i64) -> f64 {
    ((batch * height * width) as f64 * 2. * 0.01 * 6.) * (1024 * 1024) as f64
}
/// Original estimate_memory + QwenImage.extra_conds_shapes. Each entry is one
/// conditioning's ordered reference tensor shapes, including positive/negative.
/// CFG=1's later denoising shortcut does not remove them from this estimate.
pub fn qwen_sampling_memory_with_references(
    noise_shape: &[i64],
    conditions: &[Vec<Vec<i64>>],
) -> (f64, f64) {
    let area = noise_shape[0] * noise_shape[2..].iter().product::<i64>();
    let reference_areas = conditions
        .iter()
        .map(|refs| refs.iter().map(|s| s.iter().product::<i64>()).sum::<i64>() / 64)
        .collect::<Vec<_>>();
    let bytes = |a: i64| (a as f64 * 2. * 0.01 * 6.) * (1024 * 1024) as f64;
    (
        bytes(area * 2 + reference_areas.iter().sum::<i64>()),
        bytes(area + reference_areas.iter().copied().max().unwrap_or(0)),
    )
}
pub fn bf16_dit_modules() -> Result<Vec<OffloadModule>, Error> {
    serde_json::from_str(include_str!(
        "../../../resources/qwen-image21-offload-modules.json"
    ))
    .map_err(|e| Error::Native(e.to_string()))
}
impl ModelBuffers for super::FluxSampling {
    fn buffer_bytes(&self) -> u64 {
        self.buffer_bytes()
    }
    fn to_device(&mut self, device: Device) -> Result<(), Error> {
        self.to_device(device)
    }
}
impl crate::components::StaticOffload<super::FluxSampling> {
    pub fn new(parameters: PreparedParameters, native: Kitchen) -> Result<Self, Error> {
        Self::from_modules(
            parameters,
            native,
            bf16_dit_modules()?,
            super::FluxSampling::qwen21()?,
        )
    }
}
