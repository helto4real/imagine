//! Explicit static MOSS services; source compiler/vbars/async offload are disabled.
use super::{
    MossControl, MossDiT, MossExecution, MossInputs, MossLifecycle, MossReferences, MossRope,
    moss_memory_required,
};
use crate::models::qwen_image21::{
    EulerExecution, EulerStep, FullAreaCondition, FullAreaExecution, SamplingObserver,
    TextEncoderExecution, VaeExecution, flow_denoised, full_area_cfg, host_available_memory,
};
use crate::{
    attention::{Attention, AttentionInput},
    checkpoint::PreparedParameters,
    execution::Runtime,
    native::Error,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tch::{Device, Kind, Tensor};

pub struct MossStaticExecution {
    pub cancelled: Arc<AtomicBool>,
    pub operations: usize,
    pub lifecycle: Vec<MossLifecycle>,
    attention: Attention,
    preferences: BTreeMap<String, Attention>,
    text_block: Option<usize>,
}
impl MossStaticExecution {
    pub fn bind(rt: &Runtime, p: &PreparedParameters, attention: Attention) -> Result<Self, Error> {
        let mut preferences = BTreeMap::new();
        let names = p
            .values
            .keys()
            .chain(p.attention_configs.keys())
            .collect::<std::collections::BTreeSet<_>>();
        for name in names {
            if let Some(module) = name.strip_suffix(".comfy_attention.config") {
                let config = p
                    .attention_config(name)?
                    .ok_or_else(|| Error::Native("Missing attention config".into()))?;
                let mut preferred = attention.clone();
                preferred.method = attention.preferred(rt, Some(&config))?;
                preferences.insert(module.into(), preferred);
            }
        }
        Ok(Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            operations: 0,
            lifecycle: vec![],
            attention,
            preferences,
            text_block: None,
        })
    }
    fn interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        if self.cancelled.load(Ordering::Relaxed) {
            Err(Error::ReferenceException {
                class: "InterruptProcessingException",
                message: String::new(),
            })
        } else {
            Ok(())
        }
    }
    fn preferred(&self, module: &str) -> &Attention {
        self.preferences.get(module).unwrap_or(&self.attention)
    }
}
impl MossExecution for MossStaticExecution {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.interrupt()
    }
    fn lifecycle(&mut self, event: MossLifecycle) -> Result<(), Error> {
        self.lifecycle.push(event);
        Ok(())
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: i64,
        module: &str,
    ) -> Result<Tensor, Error> {
        unsafe {
            self.preferred(module).run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask: None,
                    skip_reshape: false,
                    skip_output_reshape: false,
                    enable_gqa: false,
                    scale: None,
                },
            )
        }
    }
}
impl TextEncoderExecution for MossStaticExecution {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.interrupt()
    }
    fn queue_begin(&mut self, _layers: usize, _device: Device) -> Result<(), Error> {
        Ok(())
    }
    fn queue_end(&mut self, _device: Device) -> Result<(), Error> {
        Ok(())
    }
    fn block(
        &mut self,
        index: usize,
        core: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        let previous = self.text_block.replace(index);
        let result = core(self);
        self.text_block = previous;
        result
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        heads: i64,
        gqa: bool,
    ) -> Result<Tensor, Error> {
        let module = format!(
            "model.layers.{}.self_attn",
            self.text_block.ok_or_else(|| Error::Native(
                "Qwen3.5 attention outside its source block".into()
            ))?
        );
        unsafe {
            self.preferred(&module).run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    skip_reshape: true,
                    skip_output_reshape: false,
                    enable_gqa: gqa,
                    scale: None,
                },
            )
        }
    }
}
impl VaeExecution for MossStaticExecution {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.interrupt()
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, Error> {
        super::moss_vae_attention(rt, q, k, v)
    }
}

pub struct MossDenoiserConfig<'a> {
    pub runtime: &'a Runtime,
    pub model: &'a MossDiT,
    pub positive: FullAreaCondition,
    pub negative: Option<FullAreaCondition>,
    pub cfg: f64,
    pub compute_dtype: Kind,
    pub flash_attention_memory: bool,
    pub multiplier: f64,
    pub execution: MossStaticExecution,
    pub generator: crate::native::Generator,
    pub observer: Option<SamplingObserver<'a>>,
    /// Edit references both conditions attend to, as the original gives the
    /// negative prompt the references too.
    pub references: Option<MossReferences<'a>>,
    /// Scale ROPE's options for the generated image.
    pub rope: MossRope,
    /// Color/depth control for a widened video embedder.
    pub control: Option<MossControl>,
}
pub struct MossDenoiser<'a> {
    config: MossDenoiserConfig<'a>,
}
impl<'a> MossDenoiser<'a> {
    /// # Safety
    /// The model, original prepared parameters and selected static services
    /// obey their source layout, kernel and device/stream contracts.
    pub unsafe fn bind(config: MossDenoiserConfig<'a>) -> Self {
        Self { config }
    }
}
struct Forward<'a, 'b> {
    rt: &'a Runtime,
    model: &'a MossDiT,
    exec: &'b mut MossStaticExecution,
    dtype: Kind,
    flash: bool,
    multiplier: f64,
    references: Option<MossReferences<'a>>,
    rope: MossRope,
}
impl FullAreaExecution for Forward<'_, '_> {
    fn free_memory(&mut self, device: Device) -> Result<f64, Error> {
        match device {
            Device::Cuda(i) => {
                let [free, _, active, reserved] = self.rt.native.cuda_memory(i)?;
                // As ModelPatcher.get_free_memory: what dynamic VRAM could
                // release counts as free, so CFG batching wins over residency.
                let dynamic = crate::checkpoint::dynamic_vbar::reclaimable(i);
                Ok((free + reserved - active + dynamic) as f64)
            }
            Device::Cpu => Ok(host_available_memory()? as f64),
            _ => Err(Error::BindingGap {
                backend: "moss_cfg".into(),
                operation: "Non-initial-platform memory query".into(),
            }),
        }
    }
    fn memory_required(&self, input: &[i64], _conditions: &[&FullAreaCondition]) -> f64 {
        moss_memory_required(input, self.dtype, self.flash)
    }
    fn apply_model(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        condition: &FullAreaCondition,
        _branches: &[usize],
    ) -> Result<Tensor, Error> {
        // BaseModel concatenates c_concat in the input's dtype, then casts both.
        let input = match &condition.concat {
            Some(concat) => Tensor::f_cat(
                &[
                    x.shallow_clone(),
                    crate::patching::cast_to_device(
                        &self.rt.native,
                        concat,
                        x.device(),
                        x.kind(),
                        false,
                    )?,
                ],
                1,
            )?,
            None => x.shallow_clone(),
        }
        .f_to_kind(self.dtype)?;
        let context = crate::patching::cast_to_device(
            &self.rt.native,
            &condition.context,
            x.device(),
            self.dtype,
            false,
        )?;
        let timestep = sigma
            .f_mul_scalar(self.multiplier)?
            .f_to_kind(Kind::Float)?;
        let output = unsafe {
            self.model.forward(
                self.rt,
                &MossInputs {
                    latent: &input,
                    timestep: &timestep,
                    context: &context,
                    references: self.references,
                    rope: self.rope,
                },
                self.exec,
            )
        }?
        .f_to_kind(Kind::Float)?;
        flow_denoised(sigma, &output, x)
    }
}
impl EulerExecution for MossDenoiser<'_> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor) -> Result<Tensor, Error> {
        let config = &mut self.config;
        if let Some(control) = &config.control {
            let concat = control.concat(sigma.f_double_value(&[0])? as f32)?;
            config.positive.concat = Some(concat.shallow_clone());
            if let Some(negative) = &mut config.negative {
                negative.concat = Some(concat.shallow_clone());
            }
        }
        full_area_cfg(
            x,
            sigma,
            &config.positive,
            config.negative.as_ref(),
            config.cfg,
            false,
            &mut Forward {
                rt: config.runtime,
                model: config.model,
                exec: &mut config.execution,
                dtype: config.compute_dtype,
                flash: config.flash_attention_memory,
                multiplier: config.multiplier,
                references: config.references,
                rope: config.rope,
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
