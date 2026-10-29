//! Reusable Qwen BaseModel/CFG execution, independent of development profiles
//! and artifact I/O. The caller owns the model, cache policy and noise stream.
use super::*;
use crate::{
    execution::Runtime,
    native::{Error, Generator},
};
use tch::{Device, Kind, Tensor};

pub type SamplingObserver<'a> = Box<dyn FnMut(EulerStep<'_>) -> Result<(), Error> + 'a>;
pub struct QwenDenoiserConfig<'a, C: CacheServices> {
    pub runtime: &'a Runtime,
    pub model: &'a DiT,
    /// Source model_conds remain on their intermediate device until the model
    /// call. Reference latents have already passed process_latent_in.
    pub positive: FullAreaCondition,
    pub negative: Option<FullAreaCondition>,
    pub cfg: f64,
    pub disable_cfg1_optimization: bool,
    pub compute_dtype: Kind,
    /// Original xformers_enabled OR pytorch_attention_flash_attention input.
    pub flash_attention_memory: bool,
    pub cache: PrefixCache<'a, C>,
    pub execution: ResidentExecution,
    /// Euler uses its original randn_like stream. ER-SDE uses the independent
    /// default_noise_sampler stream (CUDA seed; CPU seed+1).
    pub generator: Generator,
    pub observer: Option<SamplingObserver<'a>>,
}
pub struct QwenDenoiser<'a, C: CacheServices> {
    config: QwenDenoiserConfig<'a, C>,
}
impl<'a, C: CacheServices> QwenDenoiser<'a, C> {
    /// # Safety
    /// The model/execution pair and referenced weight/layout storage must obey
    /// DiT::forward's constructor/kernel contracts throughout this runner's use.
    /// Its input tensors must satisfy the same model's valid inference shapes.
    pub unsafe fn bind(config: QwenDenoiserConfig<'a, C>) -> Self {
        Self { config }
    }
    pub fn cache(&self) -> &PrefixCache<'a, C> {
        &self.config.cache
    }
    pub fn cache_mut(&mut self) -> &mut PrefixCache<'a, C> {
        &mut self.config.cache
    }
}
struct Forward<'a, 'b, C: CacheServices> {
    runtime: &'a Runtime,
    model: &'a DiT,
    cache: &'b mut PrefixCache<'a, C>,
    execution: &'b mut ResidentExecution,
    compute_dtype: Kind,
    flash_attention_memory: bool,
}
impl<C: CacheServices> FullAreaExecution for Forward<'_, '_, C> {
    fn free_memory(&mut self, device: Device) -> Result<f64, Error> {
        match device {
            Device::Cpu => Ok(host_available_memory()? as f64),
            Device::Cuda(index) => {
                let [free, _, active, reserved] = self.runtime.native.cuda_memory(index)?;
                // As ModelPatcher.get_free_memory: what dynamic VRAM could
                // release counts as free, so CFG batching wins over residency.
                let dynamic = crate::checkpoint::dynamic_vbar::reclaimable(index);
                Ok((free + reserved - active + dynamic) as f64)
            }
            _ => Err(Error::BindingGap {
                backend: "qwen_cfg".into(),
                operation: "Non-initial-platform memory query".into(),
            }),
        }
    }
    fn memory_required(&self, input: &[i64], conditions: &[&FullAreaCondition]) -> f64 {
        let mut area = input[0] * input[2..].iter().product::<i64>();
        for condition in conditions {
            if let Some(refs) = &condition.reference_latents {
                area += refs.iter().map(|r| r.numel() as i64).sum::<i64>() / 64;
            }
        }
        if self.flash_attention_memory {
            (area as f64 * self.compute_dtype.elt_size_in_bytes() as f64 * 0.01 * 6.)
                * (1024 * 1024) as f64
        } else {
            (area as f64 * 0.15 * 6.) * (1024 * 1024) as f64
        }
    }
    fn apply_model(
        &mut self,
        x: &Tensor,
        timestep: &Tensor,
        condition: &FullAreaCondition,
        _branches: &[usize],
    ) -> Result<Tensor, Error> {
        let input = x.f_to_kind(self.compute_dtype)?;
        let cast = |t: &Tensor, kind| {
            crate::patching::cast_to_device(&self.runtime.native, t, x.device(), kind, false)
        };
        let context = cast(&condition.context, self.compute_dtype)?;
        // Qwen21's forward ignores this extra kwarg, but BaseModel still stages
        // and retains it for the call. Integer tensors keep their source dtype.
        let _mask = condition
            .attention_mask
            .as_ref()
            .map(|mask| {
                cast(
                    mask,
                    if matches!(mask.kind(), Kind::Int | Kind::Int64) {
                        mask.kind()
                    } else {
                        self.compute_dtype
                    },
                )
            })
            .transpose()?;
        let refs = condition
            .reference_latents
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|reference| cast(reference, self.compute_dtype))
            .collect::<Result<Vec<_>, Error>>()?;
        let output = unsafe {
            self.model.forward(
                self.runtime,
                &input,
                &timestep.f_to_kind(Kind::Float)?,
                &context,
                &refs,
                condition.image_slots.as_deref().unwrap_or_default(),
                Some(self.cache),
                self.execution,
            )
        }?
        .f_to_kind(Kind::Float)?;
        flow_denoised(timestep, &output, x)
    }
}
impl<C: CacheServices> EulerExecution for QwenDenoiser<'_, C> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor) -> Result<Tensor, Error> {
        let config = &mut self.config;
        full_area_cfg(
            x,
            sigma,
            &config.positive,
            config.negative.as_ref(),
            config.cfg,
            config.disable_cfg1_optimization,
            &mut Forward {
                runtime: config.runtime,
                model: config.model,
                cache: &mut config.cache,
                execution: &mut config.execution,
                compute_dtype: config.compute_dtype,
                flash_attention_memory: config.flash_attention_memory,
            },
        )
    }
    fn noise_like(&mut self, x: &Tensor) -> Result<Tensor, Error> {
        self.config.generator.randn(&x.size(), x.kind())
    }
    fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error> {
        if let Some(observer) = &mut self.config.observer {
            observer(step)?;
        }
        Ok(())
    }
}
