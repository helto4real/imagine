//! Original constructor inventories for static encoder/VAE ownership. The same
//! physical offload and coordinator implementation is used for the DiT.
use super::{ComponentBuffers, ModelCastDtype, OffloadModule, StaticOffload};
use crate::{checkpoint::PreparedParameters, execution::Runtime, native::Error};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
pub struct ComponentInventory {
    pub modules: Vec<OffloadModule>,
    pub model_size: u64,
    #[serde(default)]
    pub force_cast_weights: bool,
    pub manual_cast_dtype: String,
    pub buffers: Vec<ComponentBuffer>,
}
#[derive(Deserialize)]
pub struct ComponentBuffer {
    pub name: String,
    pub persistent: bool,
    pub initial_value: Option<f64>,
    #[serde(default)]
    pub default_dtype: bool,
    pub kind: Option<String>,
    #[serde(default)]
    pub requires_grad: bool,
}
pub fn component_inventory(component: &str) -> Result<ComponentInventory, Error> {
    #[derive(Deserialize)]
    struct Inventories {
        models: BTreeMap<String, ComponentInventory>,
    }
    let mut inventories: Inventories = serde_json::from_str(include_str!(
        "../../../resources/qwen-image21-component-modules.json"
    ))
    .map_err(|e| Error::Native(e.to_string()))?;
    inventories
        .models
        .remove(component)
        .ok_or_else(|| Error::BindingGap {
            backend: "component_inventory".into(),
            operation: component.into(),
        })
}
pub fn static_component(
    rt: &Runtime,
    component: &str,
    parameters: PreparedParameters,
) -> Result<StaticOffload<ComponentBuffers>, Error> {
    static_component_from_inventory(rt, parameters, component_inventory(component)?)
}
pub fn static_component_from_inventory(
    rt: &Runtime,
    mut parameters: PreparedParameters,
    inventory: ComponentInventory,
) -> Result<StaticOffload<ComponentBuffers>, Error> {
    let mut values = BTreeMap::new();
    let mut persistent = BTreeSet::new();
    let mut nonleaf_parameters = BTreeSet::new();
    for buffer in inventory.buffers {
        if let Some(value) = buffer.initial_value {
            // Source torch.tensor(4.6055) uses the current default dtype; the
            // constructor parameter is independent of checkpoint storage.
            let dtype = if buffer.default_dtype {
                rt.native.default_dtype()?
            } else {
                tch::Kind::Float
            };
            parameters.values.insert(
                buffer.name.clone(),
                crate::execution_value::Value::tensor(
                    tch::Tensor::f_full([], value, (dtype, tch::Device::Cpu))?
                        .f_set_requires_grad(buffer.requires_grad)?,
                ),
            );
        }
        let value = parameters
            .values
            .remove(&buffer.name)
            .ok_or_else(|| Error::BindingGap {
                backend: "component_buffers".into(),
                operation: buffer.name.clone(),
            })?;
        if buffer.persistent {
            persistent.insert(buffer.name.clone());
        }
        if buffer.kind.as_deref() == Some("nonleaf_parameter") {
            nonleaf_parameters.insert(buffer.name.clone());
        }
        values.insert(buffer.name, value);
    }
    let cast = match inventory.manual_cast_dtype.as_str() {
        "missing" => ModelCastDtype::Missing,
        "None" => ModelCastDtype::None,
        "torch.bfloat16" => ModelCastDtype::Kind(tch::Kind::BFloat16),
        "torch.float16" => ModelCastDtype::Kind(tch::Kind::Half),
        "torch.float32" => ModelCastDtype::Kind(tch::Kind::Float),
        _ => {
            return Err(Error::BindingGap {
                backend: "component_cast_dtype".into(),
                operation: inventory.manual_cast_dtype,
            });
        }
    };
    let mut owner = StaticOffload::from_modules(
        parameters,
        rt.native.clone(),
        inventory.modules,
        ComponentBuffers {
            values,
            persistent,
            nonleaf_parameters,
        },
    )?
    .with_cast_dtype(cast)?;
    owner.force_cast_weights = inventory.force_cast_weights;
    Ok(owner)
}
impl StaticOffload<ComponentBuffers> {
    /// Snapshot references after a coordinator transition, without copying any
    /// weight payload or retaining stale CUDA parameters across later loads.
    pub fn execution_parameters(&self) -> PreparedParameters {
        let mut parameters = self.parameters.clone();
        parameters.values.extend(self.sampling.values.clone());
        parameters
    }
}
