//! Schema-driven preparation extracted without arithmetic or lifecycle changes.
use crate::{
    checkpoint::{
        LoadOptions, PreparedParameters, load_quantized_embedding, load_quantized_module,
    },
    execution::Runtime,
    execution_value::Value,
    native::Error,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
#[derive(Deserialize, Clone)]
pub struct ParameterSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub kind: String,
    pub requires_grad: Option<bool>,
}
#[derive(Clone, Copy)]
pub struct ComponentLoad {
    pub dtype: Kind,
    /// BaseModel freezes the DiT, and CLIP freezes its transformer. The VAE
    /// keeps constructor flags; its inference caller disables autograd.
    pub freeze: bool,
}
#[derive(Deserialize)]
pub struct ComponentSchema {
    pub parameters: Vec<ParameterSpec>,
    pub buffers: Vec<BufferSpec>,
}
#[derive(Deserialize)]
pub struct BufferSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
}
#[derive(Deserialize)]
pub struct ConstructorSchema {
    pub models: BTreeMap<String, ComponentSchema>,
    pub vae_config: VaeWidths,
}
#[derive(Deserialize)]
pub struct VaeWidths {
    pub dim: i64,
    pub dec_dim: i64,
}
fn gap(op: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "qwen21_initial_checkpoint".into(),
        operation: op.into(),
    }
}
/// Same source preparation with an explicit original constructor schema. Tiny
/// constructor fixtures validate the whole loader without trained-model runs.
pub fn prepare_resident_component(
    rt: &Runtime,
    component: &str,
    spec: &ComponentSchema,
    state: BTreeMap<String, Tensor>,
    device: Device,
) -> Result<PreparedParameters, Error> {
    prepare_resident_component_with_options(
        rt,
        component,
        spec,
        state,
        device,
        ComponentLoad {
            dtype: Kind::BFloat16,
            freeze: true,
        },
    )
}
pub fn prepare_resident_component_with_options(
    rt: &Runtime,
    component: &str,
    spec: &ComponentSchema,
    mut state: BTreeMap<String, Tensor>,
    device: Device,
    options: ComponentLoad,
) -> Result<PreparedParameters, Error> {
    for p in &spec.parameters {
        let t = state
            .get(&p.name)
            .ok_or_else(|| gap(format!("Missing checkpoint parameter {}", p.name)))?;
        if t.size() != p.shape {
            return Err(gap(format!(
                "Alternate constructor shape {}: {:?} != {:?}",
                p.name,
                t.size(),
                p.shape
            )));
        }
    }
    let mut prepared = PreparedParameters::default();
    let quantized_ops = state.keys().any(|k| k.ends_with(".comfy_quant"));
    for p in &spec.parameters {
        let module = p.name.rsplit_once('.').unwrap().0;
        if p.kind == "linear" && p.name.ends_with(".weight") && quantized_ops {
            let loaded = load_quantized_module(
                &mut state,
                &format!("{module}."),
                LoadOptions {
                    device: Device::Cpu,
                    compute_kind: options.dtype,
                    original_shape: p.shape.clone(),
                    full_precision: component == "text",
                    disabled_formats: &[],
                    load_extra_params: true,
                },
                &rt.native,
            )?;
            prepared.insert_linear(module, loaded);
        } else if p.kind == "embedding" && p.name.ends_with(".weight") && quantized_ops {
            let value = load_quantized_embedding(
                &mut state,
                &format!("{module}."),
                p.shape.clone(),
                options.dtype,
                false,
                &rt.native,
            )?
            .ok_or_else(|| gap(format!("Missing {}", p.name)))?;
            prepared.values.insert(p.name.clone(), value);
            state.remove(&p.name);
        } else {
            let t = state
                .remove(&p.name)
                .ok_or_else(|| gap(format!("Consumed {} twice", p.name)))?;
            // Plain super-loaders copy parameters. VAE constructs in Float32,
            // loads there, then the original VAE wrapper casts to vae_dtype.
            let kind = if component == "vae" {
                Kind::Float
            } else {
                options.dtype
            };
            let value = rt.native.clone_tensor(&t.f_to_kind(kind)?)?.f_detach()?;
            prepared.values.insert(p.name.clone(), Value::tensor(value));
        }
    }
    // Nonpersistent source vision rotary buffer is constructed even for T2I.
    for b in &spec.buffers {
        if let Some(value) = state.remove(&b.name) {
            if value.size() != b.shape {
                return Err(gap(format!(
                    "Alternate constructor buffer shape {}",
                    b.name
                )));
            }
            prepared.values.insert(
                b.name.clone(),
                Value::tensor(rt.native.clone_tensor(&value)?.f_detach()?),
            );
        } else if component == "text"
            && b.name == "visual.rotary_pos_emb.inv_freq"
            && b.shape == [18]
        {
            let inv = Tensor::f_arange_start_step(0, 36, 2, (Kind::Float, Device::Cpu))?
                .f_div_scalar(36)?;
            let inv = Tensor::f_pow_scalar(10000., &inv)?
                .f_reciprocal()?
                .f_mul_scalar(1.)?;
            prepared.values.insert(b.name.clone(), Value::tensor(inv));
        } else {
            return Err(gap(format!("Unbound constructor buffer {}", b.name)));
        }
    }
    // ComfyAttention decodes JSON into Python attributes, outside Module.to.
    let metadata = state
        .keys()
        .filter(|k| k.ends_with(".comfy_attention.config"))
        .cloned()
        .collect::<Vec<_>>();
    for key in metadata {
        prepared
            .insert_attention_config(key.clone(), Value::tensor(state.remove(&key).unwrap()))?;
    }
    if !state.is_empty() {
        return Err(gap(format!(
            "Unbound checkpoint keys: {:?}",
            state.keys().collect::<Vec<_>>()
        )));
    }
    for (name, value) in &mut prepared.values {
        if name.ends_with(".comfy_attention.config") {
            continue;
        }
        *value = match value {
            Value::Quantized(_) => Value::quantized(
                value
                    .as_quantized()?
                    .to(&rt.native, Some(device), None, false)?
                    .as_ref()
                    .shallow_parts(),
            ),
            _ => {
                let t = value.as_tensor()?;
                let k = if !t.f_is_floating_point()? && !t.f_is_complex()? {
                    t.kind()
                } else if name == "visual.rotary_pos_emb.inv_freq" {
                    Kind::Float
                } else {
                    options.dtype
                };
                Value::tensor(t.f_to_device_(device, k, false, false)?)
            }
        };
    }
    for policy in prepared.linear_policies.values_mut() {
        for v in [&mut policy.input_scale, &mut policy.pre_quant_scale]
            .into_iter()
            .flatten()
        {
            *v = Value::tensor(v.as_tensor()?.f_to_device(device)?);
        }
    }
    for parameter in &spec.parameters {
        if let Some(Value::Tensor(_)) = prepared.values.get(&parameter.name) {
            let tensor = prepared[&parameter.name].as_tensor()?;
            prepared.values.insert(
                parameter.name.clone(),
                Value::tensor(tensor.f_set_requires_grad(
                    !options.freeze && parameter.requires_grad.unwrap_or(true),
                )?),
            );
        }
    }
    Ok(prepared)
}
