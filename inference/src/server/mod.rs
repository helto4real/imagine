//! Product entry point over the already validated static inference components.
//! No development profiles, tensor dumps, prompts or images are written to disk.
#[path = "../../../shared/component_error.rs"]
pub mod component_error;
mod image_preview;
mod load_error;
pub mod maps;
#[path = "../../../shared/model_spec.rs"]
pub mod model_spec;
mod moss;
mod preview_callback;
mod qwen;
mod sampling_values;
pub use crate::h3_product::protocol::preview;
use crate::{
    attention::{Attention, AttentionMethod},
    checkpoint::{Aimdo, PreparedParameters, Safetensors},
    execution::Runtime,
    memory::Manager,
    models::{moss_image as m, qwen_image21 as q},
    native::Error,
};
use load_error::{observe_load, report as load_failure};
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
fn active_loras(loras: &[Lora]) -> impl Iterator<Item = &Lora> {
    // Match ComfyUI's loader bypass before reading or binding disabled LoRAs.
    loras.iter().filter(|lora| lora.strength != 0.0)
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
    #[serde(default)]
    pub sampling_preview: Option<preview::Settings>,
    #[serde(default)]
    pub control: Control,
}
/// The MOSS color/depth control adapter's settings, as `MOSSImageControl` takes
/// them: the maps' strength and the sampling percents it spans.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct Control {
    pub strength: f64,
    pub start: f64,
    pub end: f64,
}
impl Default for Control {
    fn default() -> Self {
        Self {
            strength: 1.,
            start: 0.,
            end: 1.,
        }
    }
}
/// MOSS control maps, RGBA in [0, 1] `[1, h, w, 4]`: one flat color per region and
/// depth with near white.
#[derive(Default)]
pub struct ControlMaps {
    pub color: Option<Tensor>,
    pub depth: Option<Tensor>,
}
impl ControlMaps {
    pub fn is_empty(&self) -> bool {
        self.color.is_none() && self.depth.is_none()
    }
}
pub enum Event {
    Phase(&'static str),
    Step(i64),
    LoadFailure(component_error::LoadDiagnostic),
    Preview {
        metadata: preview::Metadata,
        pixels: Vec<u8>,
    },
    PreviewWarning,
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
    model: &str,
) -> Result<PreparedParameters, Error> {
    let moss = model_spec::is_moss(model);
    let state = Safetensors::load_aimdo(path, aimdo, &rt.native)?;
    if state.metadata.get("_quantization_metadata").is_some() {
        return Err(Error::Native(
            "Unsupported legacy checkpoint metadata".into(),
        ));
    }
    if role == "dit" {
        let header = state
            .tensors
            .iter()
            .map(|(name, value)| {
                let dtype = match value.tensor.kind() {
                    Kind::BFloat16 => "BF16",
                    Kind::Float => "F32",
                    Kind::Int8 => "I8",
                    _ => "OTHER",
                };
                (
                    name.clone(),
                    serde_json::json!({"shape":value.tensor.size(),"dtype":dtype}),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let detected = model_spec::detect(&serde_json::Value::Object(header))
            .map_err(|e| Error::Native(e.into()))?;
        let selected =
            model_spec::model(model).ok_or_else(|| Error::Native("Unsupported model".into()))?;
        if detected.model(selected.profile) != Ok(model) {
            return Err(Error::Native(
                "Checkpoint does not match selected architecture or weight format".into(),
            ));
        }
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
    maps: &ControlMaps,
    observer: &mut Observer<'_>,
) -> Result<Tensor, Error> {
    let moss = model_spec::is_moss(&r.model);
    if model_spec::model(&r.model).is_none() {
        return Err(Error::Native("Unsupported model".into()));
    }
    if !sampling_values::valid(r.sampling.steps, r.sampling.cfg_scale)
        || r.width < 32
        || r.height < 32
        || r.width > 8192
        || r.height > 8192
        || r.width % 32 != 0
        || r.height % 32 != 0
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
                || images.len() > 4)
        || !maps.is_empty() && (!moss || !images.is_empty())
    {
        return Err(Error::Native("Unsupported inference selection".into()));
    }
    let c = r.control;
    if ![c.strength, c.start, c.end].iter().all(|v| v.is_finite())
        || !(0.0..=10.0).contains(&c.strength)
        || !(0.0..=1.0).contains(&c.start)
        || !(0.0..=1.0).contains(&c.end)
        || c.start > c.end
    {
        return Err(Error::Native("Invalid control settings".into()));
    }
    let s = &r.sampling.custom_sigmas;
    if !s.is_empty()
        && (usize::try_from(r.sampling.steps)
            .ok()
            .and_then(|n| n.checked_add(1))
            != Some(s.len())
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
        moss::generate(rt, aimdo, r, images, maps, observer)
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
        concat: None,
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
    diagnostic: (&std::path::Path, &mut Observer<'_>),
) -> Result<Tensor, Error> {
    let patcher = model.patcher();
    manager
        .load(
            std::slice::from_ref(&patcher),
            0.,
            None,
            false,
            false,
            memory,
        )
        .map_err(|error| load_failure("vae", diagnostic.0, diagnostic.1, error))?;
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
#[cfg(test)]
mod lora_selection_tests {
    use super::*;

    fn selection(entries: &[(&str, f64)]) -> Vec<Lora> {
        entries
            .iter()
            .map(|(path, strength)| Lora {
                path: path.into(),
                strength: *strength,
            })
            .collect()
    }

    #[test]
    fn zero_strength_never_reaches_the_reader() {
        let loras = selection(&[
            ("missing-positive-zero", 0.0),
            ("missing-negative-zero", -0.0),
        ]);
        let loaded: Result<Vec<()>, &str> = active_loras(&loras)
            .map(|_| Err("reader must not be called"))
            .collect();
        assert_eq!(loaded, Ok(vec![]));
    }

    #[test]
    fn mixed_selection_preserves_order_duplicates_and_nonzero_strengths() {
        let loras = selection(&[
            ("disabled", 0.0),
            ("same", 0.5),
            ("disabled", -0.0),
            ("negative", -1.0),
            ("same", f64::MIN_POSITIVE),
        ]);
        let loaded: Vec<_> = active_loras(&loras)
            .map(|l| (l.path.to_str().unwrap(), l.strength))
            .collect();
        assert_eq!(
            loaded,
            vec![
                ("same", 0.5),
                ("negative", -1.0),
                ("same", f64::MIN_POSITIVE)
            ]
        );
    }
}
