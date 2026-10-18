//! Product entry point over the already validated static inference components.
//! No development profiles, tensor dumps, prompts or images are written to disk.
mod moss;
mod qwen;
use crate::{
    attention::{Attention, AttentionMethod},
    checkpoint::{Aimdo, PreparedParameters, Safetensors},
    execution::Runtime,
    memory::Manager,
    models::{moss_image as m, qwen_image21 as q},
    native::Error,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
use tch::{Device, Kind, Tensor};
#[derive(Deserialize, Serialize)]
pub struct Sampling {
    pub steps: i64,
    pub cfg_scale: f64,
    pub sampler: String,
    pub scheduler: String,
    #[serde(default)]
    pub custom_sigmas: Vec<f32>,
}
#[derive(Deserialize, Serialize)]
pub struct Lora {
    pub path: PathBuf,
    pub strength: f64,
}
#[derive(Deserialize, Serialize)]
pub struct Request {
    pub model: String,
    pub diffusion_model: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
    pub prompt: String,
    pub negative_prompt: String,
    pub width: i64,
    pub height: i64,
    pub seed: u64,
    pub sampling: Sampling,
    #[serde(default)]
    pub loras: Vec<Lora>,
    #[serde(default)]
    pub zero_negative_conditioning: bool,
}
pub enum Event {
    Phase(&'static str),
    Step(i64),
}
pub type Observer<'a> = dyn FnMut(Event) -> Result<(), Error> + 'a;
pub(super) fn attention() -> Attention {
    Attention {
        method: AttentionMethod::Pytorch,
        priority_wrapper: true,
        attention_precision: None,
        low_precision_attention: false,
        model_device: Device::Cuda(0),
    }
}
pub(super) fn load(
    rt: &Runtime,
    aimdo: &Aimdo,
    path: &std::path::Path,
    role: &str,
    moss: bool,
) -> Result<PreparedParameters, Error> {
    let state = Safetensors::load_aimdo(path, aimdo, &rt.native)?;
    if state.metadata.get("_quantization_metadata").is_some() {
        return Err(Error::Native(
            "Unsupported legacy checkpoint metadata".into(),
        ));
    }
    let state = state
        .tensors
        .into_iter()
        .map(|(n, t)| (n, t.tensor))
        .collect::<Vec<_>>();
    if moss {
        m::prepare_moss_component(
            rt,
            role,
            state,
            Device::Cpu,
            q::ComponentLoad {
                dtype: Kind::BFloat16,
                freeze: role != "vae",
            },
        )
    } else {
        let mut state = state.into_iter().collect::<BTreeMap<_, _>>();
        if role == "text" {
            state = q::normalize_text_keys(state)
        }
        q::prepare_initial_component(rt, role, state, Device::Cpu)
    }
}
/// All calls run on the private worker's single inference thread. Cancellation
/// terminates/reaps that worker before the server releases its GPU lease.
pub fn generate(
    rt: &Runtime,
    aimdo: &Aimdo,
    r: &Request,
    images: &[Tensor],
    observer: &mut Observer<'_>,
) -> Result<Tensor, Error> {
    let moss = r.model == "moss-image-sft-int8";
    if !(moss
        || matches!(
            r.model.as_str(),
            "qwen-image-2.1-int8" | "qwen-image-2.1-turbo"
        ))
    {
        return Err(Error::Native("Unsupported model".into()));
    }
    if !(1..=100).contains(&r.sampling.steps)
        || r.width < 32
        || r.height < 32
        || r.width > 8192
        || r.height > 8192
        || r.width % 32 != 0
        || r.height % 32 != 0
        || !r.sampling.cfg_scale.is_finite()
        || !(0.0..=20.0).contains(&r.sampling.cfg_scale)
    {
        return Err(Error::Native("Invalid inference request".into()));
    }
    if !matches!(r.sampling.sampler.as_str(), "euler" | "er_sde")
        || !matches!(
            r.sampling.scheduler.as_str(),
            "simple" | "beta" | "normal" | "sgm_uniform"
        )
        || moss
            && (r.sampling.sampler != "euler"
                || !matches!(r.sampling.scheduler.as_str(), "normal" | "sgm_uniform")
                || !images.is_empty()
                || !r.loras.is_empty())
    {
        return Err(Error::Native("Unsupported inference selection".into()));
    }
    let s = &r.sampling.custom_sigmas;
    if !s.is_empty()
        && (s.len() != r.sampling.steps as usize + 1
            || s.last() != Some(&0.)
            || s[0] <= 0.
            || s.iter().any(|v| !v.is_finite() || *v < 0.)
            || s.windows(2).any(|v| v[1] > v[0]))
    {
        return Err(Error::Native("Invalid sigma schedule".into()));
    }
    let max = [&r.diffusion_model, &r.text_encoder, &r.vae]
        .into_iter()
        .map(std::fs::metadata)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| Error::Native(e.to_string()))?
        .into_iter()
        .map(|m| m.len())
        .max()
        .unwrap();
    // Same conservative host guard as the selected trained development controls.
    if q::host_available_memory()? < 2 * max + 4 * 1024u64.pow(3) {
        return Err(Error::Native(
            "Insufficient host RAM for selected model".into(),
        ));
    }
    let _grad = tch::no_grad_guard();
    let _inference = crate::native::inference_mode(true)?;
    let out = if moss {
        moss::generate(rt, aimdo, r, observer)
    } else {
        qwen::generate(rt, aimdo, r, images, observer)
    }?;
    rt.native.cuda_soft_empty_cache(0)?;
    Ok(out)
}
pub(super) fn full_condition(
    c: &q::TextConditioning,
    refs: Option<&q::ReferenceImages>,
    rt: &Runtime,
) -> Result<q::FullAreaCondition, Error> {
    Ok(q::FullAreaCondition {
        context: c.context.shallow_clone(),
        attention_mask: c.attention_mask.as_ref().map(Tensor::shallow_clone),
        reference_latents: refs
            .map(|r| {
                r.reference_latents
                    .iter()
                    .map(|v| q::LatentNormalization::new(rt)?.process_in(v))
                    .collect()
            })
            .transpose()?,
        image_slots: Some(c.image_slots.clone()),
    })
}
/// Same managed VAE path for reference encoding and final decoding.
pub(super) fn qwen_vae<'a>(
    rt: &'a Runtime,
    model: &std::rc::Rc<q::StaticModel<'a, q::ComponentBuffers>>,
    input: &Tensor,
    op: q::VaeOperation,
    manager: &mut Manager<'a>,
    memory: &mut q::StaticMemoryServices,
) -> Result<Tensor, Error> {
    let patcher = model.patcher();
    manager.load(
        std::slice::from_ref(&patcher),
        0.,
        None,
        false,
        false,
        memory,
    )?;
    let schema = q::constructor_schema()?;
    let mut exec = q::ResidentExecution::bind(rt, &PreparedParameters::default(), 0, attention())?;
    let mut service = q::ManagedImageVae {
        runtime: rt,
        model: model.clone(),
        patcher,
        manager,
        memory,
        execution: &mut exec,
        config: q::ImageVaeConfig {
            dim: schema.vae_config.dim,
            dec_dim: schema.vae_config.dec_dim,
            ..Default::default()
        },
        device: Device::Cuda(0),
        force_full_load: false,
    };
    let policy = q::ImageVaePolicy {
        device: Device::Cuda(0),
        dtype: Kind::BFloat16,
        output_device: Device::Cpu,
        output_kind: Kind::Float,
        default_kind: Kind::Float,
        output_channels: 4,
        latent_channels: 64,
    };
    let out = policy.execute(op, input, &mut service)?;
    drop(service);
    Ok(out)
}
