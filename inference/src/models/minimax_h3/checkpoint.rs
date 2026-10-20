//! Selected H3 constructors and quantized CPU preparation. The source's common
//! component normalization must precede this explicit consumer; no Qwen Image
//! constructor schema or checkpoint-wide dtype conversion is used.
use crate::{
    checkpoint::{
        LoadOptions, PreparedParameters, load_quantized_embedding, load_quantized_module,
    },
    execution::Runtime,
    execution_value::Value,
    models::qwen_image21::{
        ComponentBuffers, ComponentInventory, ComponentSchema, StaticOffload,
        static_component_from_inventory,
    },
    native::Error,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    Dit,
    Text,
    VideoVae,
    AudioVae,
}
impl Component {
    pub fn name(self) -> &'static str {
        match self {
            Self::Dit => "dit",
            Self::Text => "text",
            Self::VideoVae => "video_vae",
            Self::AudioVae => "audio_vae",
        }
    }
    fn compute_kind(self) -> Kind {
        match self {
            Self::Dit | Self::Text => Kind::BFloat16,
            Self::VideoVae => Kind::Half,
            Self::AudioVae => Kind::Float,
        }
    }
}
#[derive(Deserialize)]
pub struct LoadedParameter {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub requires_grad: bool,
    pub load_kind: String,
}
#[derive(Deserialize)]
pub struct LoadedBuffer {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: String,
    pub persistent: bool,
    pub initial_bytes: Option<String>,
}
#[derive(Deserialize)]
pub struct LoadedSchema {
    pub parameters: Vec<LoadedParameter>,
    pub buffers: Vec<LoadedBuffer>,
}
#[derive(Deserialize)]
pub struct ConstructorSchema {
    pub models: BTreeMap<String, ComponentSchema>,
    pub loaded_models: BTreeMap<String, LoadedSchema>,
}
pub fn constructor_schema() -> Result<ConstructorSchema, Error> {
    serde_json::from_str(include_str!("../../../resources/h3-schema.json"))
        .map_err(|e| Error::Native(e.to_string()))
}
fn gap(message: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "minimax_h3_checkpoint".into(),
        operation: message.into(),
    }
}
fn kind(name: &str) -> Result<Kind, Error> {
    match name {
        "torch.float32" => Ok(Kind::Float),
        "torch.bfloat16" => Ok(Kind::BFloat16),
        "torch.float16" => Ok(Kind::Half),
        _ => Err(gap(format!("Unbound constructor dtype {name}"))),
    }
}
fn initial_buffer(b: &LoadedBuffer) -> Result<Tensor, Error> {
    let hex = b
        .initial_bytes
        .as_deref()
        .ok_or_else(|| gap(format!("Unbound constructor buffer {}", b.name)))?;
    let bytes = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| Error::Native(e.to_string())))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::f_from_data_size(&bytes, &b.shape, kind(&b.dtype)?)?)
}
/// Prepare a complete selected component on CPU. Caller retains the normalized
/// metadata owner and source annotations until its original loading scope ends.
/// No trained GPU placement occurs at this boundary.
pub fn prepare(
    rt: &Runtime,
    component: Component,
    state: BTreeMap<String, Tensor>,
) -> Result<PreparedParameters, Error> {
    let schema = constructor_schema()?;
    prepare_with_schema(
        rt,
        component,
        &schema.models[component.name()],
        &schema.loaded_models[component.name()],
        state,
    )
}

/// Consume a typed view of the common normalized component while leaving its
/// ordered dictionary and metadata owners in the caller's original scope.
pub fn prepare_normalized(
    rt: &Runtime,
    component: Component,
    state: &crate::checkpoint::component::ComponentState,
) -> Result<PreparedParameters, Error> {
    let Value::Dict(weights) = state.weights.snapshot()? else {
        return Err(gap(
            "Typed H3 component requires an ordered tensor dictionary",
        ));
    };
    let tensors = weights
        .into_iter()
        .map(|(name, value)| Ok((name, value.as_tensor()?)))
        .collect::<Result<BTreeMap<_, _>, Error>>()?;
    prepare(rt, component, tensors)
}
/// Same source loading operations with an explicit tiny constructor schema for
/// bounded differential acceptance. Only complete typed tensor state is bound.
pub fn prepare_with_schema(
    rt: &Runtime,
    component: Component,
    spec: &ComponentSchema,
    loaded: &LoadedSchema,
    mut state: BTreeMap<String, Tensor>,
) -> Result<PreparedParameters, Error> {
    let mut prepared = PreparedParameters::default();
    for p in &spec.parameters {
        let actual = state
            .get(&p.name)
            .ok_or_else(|| gap(format!("Missing checkpoint parameter {}", p.name)))?;
        if actual.size() != p.shape {
            return Err(gap(format!("Alternate constructor shape {}", p.name)));
        }
        let output = loaded
            .parameters
            .iter()
            .find(|v| v.name == p.name)
            .ok_or_else(|| gap(format!("Missing original load specification {}", p.name)))?;
        if output.shape != p.shape {
            return Err(gap(format!("Original load specification shape {}", p.name)));
        }
        let module = p
            .name
            .rsplit_once('.')
            .ok_or_else(|| gap("Unbound parameter name"))?
            .0;
        if output.load_kind == "mixed_linear" && p.name.ends_with(".weight") {
            let weight = load_quantized_module(
                &mut state,
                &format!("{module}."),
                LoadOptions {
                    device: Device::Cpu,
                    compute_kind: component.compute_kind(),
                    original_shape: p.shape.clone(),
                    full_precision: component == Component::Text,
                    disabled_formats: &[],
                    load_extra_params: true,
                },
                &rt.native,
            )?;
            prepared.insert_linear(module, weight);
        } else if output.load_kind == "mixed_embedding" && p.name.ends_with(".weight") {
            let weight = load_quantized_embedding(
                &mut state,
                &format!("{module}."),
                p.shape.clone(),
                component.compute_kind(),
                false,
                &rt.native,
            )?
            .ok_or_else(|| gap(format!("Missing {}", p.name)))?;
            prepared.values.insert(p.name.clone(), weight);
            state.remove(&p.name);
        } else {
            let value = state
                .remove(&p.name)
                .ok_or_else(|| gap(format!("Consumed {} twice", p.name)))?;
            // Ordinary Module copies use the constructor dtype, then VAE.to
            // casts to the chosen execution dtype. Mixed Linear biases are
            // constructed directly in the mixed-operation compute dtype.
            let constructor_kind = if output.load_kind == "mixed_linear" {
                component.compute_kind()
            } else {
                kind(&p.dtype)?
            };
            let value = rt
                .native
                .clone_tensor(&value.f_to_kind(constructor_kind)?)?
                .f_detach()?
                .f_to_kind(kind(&output.dtype)?)?
                .f_set_requires_grad(output.requires_grad)?;
            prepared.values.insert(p.name.clone(), Value::tensor(value));
        }
    }
    for b in &loaded.buffers {
        let value = if b.persistent {
            let t = state
                .remove(&b.name)
                .ok_or_else(|| gap(format!("Missing checkpoint buffer {}", b.name)))?;
            if t.size() != b.shape {
                return Err(gap(format!(
                    "Alternate constructor buffer shape {}",
                    b.name
                )));
            }
            rt.native
                .clone_tensor(
                    &t.f_to_kind(kind(
                        &spec
                            .buffers
                            .iter()
                            .find(|x| x.name == b.name)
                            .ok_or_else(|| gap(format!("Missing constructor buffer {}", b.name)))?
                            .dtype,
                    )?)?,
                )?
                .f_to_kind(kind(&b.dtype)?)?
        } else {
            initial_buffer(b)?
        };
        prepared.values.insert(b.name.clone(), Value::tensor(value));
    }
    let attention_keys = state
        .keys()
        .filter(|k| k.ends_with(".comfy_attention.config"))
        .cloned()
        .collect::<Vec<_>>();
    for key in attention_keys {
        prepared
            .insert_attention_config(key.clone(), Value::tensor(state.remove(&key).unwrap()))?;
    }
    if !state.is_empty() {
        return Err(gap(format!(
            "Unbound checkpoint keys: {:?}",
            state.keys().collect::<Vec<_>>()
        )));
    }
    Ok(prepared)
}

pub fn static_component(
    rt: &Runtime,
    component: Component,
    mut parameters: PreparedParameters,
    auxiliary_buffers: BTreeMap<String, Value>,
) -> Result<StaticOffload<ComponentBuffers>, Error> {
    for (name, value) in auxiliary_buffers {
        if parameters.values.contains_key(&name) {
            return Err(gap(format!("Duplicate auxiliary buffer {name}")));
        }
        parameters.values.insert(name, value);
    }
    #[derive(Deserialize)]
    struct Inventories {
        models: BTreeMap<String, ComponentInventory>,
    }
    let mut inventories: Inventories =
        serde_json::from_str(include_str!("../../../resources/h3-component-modules.json"))
            .map_err(|e| Error::Native(e.to_string()))?;
    let inventory = inventories
        .models
        .remove(component.name())
        .ok_or_else(|| gap("Unbound H3 component inventory"))?;
    static_component_from_inventory(rt, parameters, inventory)
}
