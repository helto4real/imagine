//! Original constructor/schema and static component inventory for local MOSS.
use crate::{
    checkpoint::PreparedParameters,
    execution::Runtime,
    models::qwen_image21::{
        ComponentBuffers, ComponentInventory, ComponentLoad, ConstructorSchema, StaticOffload,
        normalized_text_key, prepare_resident_component_with_options,
        static_component_from_inventory,
    },
    native::Error,
};
use std::collections::BTreeMap;
use tch::{Device, Tensor};
pub fn static_moss_component(
    rt: &Runtime,
    component: &str,
    mut parameters: PreparedParameters,
) -> Result<StaticOffload<ComponentBuffers>, Error> {
    if component == "dit" {
        parameters.values.insert(
            "model_sampling.sigmas".into(),
            crate::execution_value::Value::tensor(super::MossSampling::moss()?.sigmas),
        );
    }
    static_component_from_inventory(rt, parameters, moss_component_inventory(component)?)
}
pub fn moss_constructor_schema() -> Result<ConstructorSchema, Error> {
    serde_json::from_str(include_str!("../../../resources/moss-schema.json"))
        .map_err(|e| Error::Native(e.to_string()))
}
pub fn moss_component_inventory(component: &str) -> Result<ComponentInventory, Error> {
    #[derive(serde::Deserialize)]
    struct Inventories {
        models: BTreeMap<String, ComponentInventory>,
    }
    let mut inventories: Inventories = serde_json::from_str(include_str!(
        "../../../resources/moss-component-modules.json"
    ))
    .map_err(|e| Error::Native(e.to_string()))?;
    inventories
        .models
        .remove(component)
        .ok_or_else(|| Error::BindingGap {
            backend: "moss_component_inventory".into(),
            operation: component.into(),
        })
}
pub fn prepare_moss_component(
    rt: &Runtime,
    component: &str,
    mut state: Vec<(String, Tensor)>,
    device: Device,
    options: ComponentLoad,
) -> Result<PreparedParameters, Error> {
    let schema = moss_constructor_schema()?;
    let spec = schema
        .models
        .get(component)
        .ok_or_else(|| Error::BindingGap {
            backend: "moss_constructor".into(),
            operation: component.into(),
        })?;
    if component == "dit" {
        // supported_models.MOSSImage.process_unet_state_dict discards the
        // serialized legacy rope bands; the original model constructs RoPE.
        state.retain(|(name, _)| name != "rope.bands");
    }
    let state = if component == "text" {
        state
            .into_iter()
            .map(|(name, value)| (normalized_text_key(&name), value))
            .collect()
    } else if component == "vae" {
        super::flux2_vae_checkpoint(state)?
    } else {
        state.into_iter().collect()
    };
    prepare_resident_component_with_options(rt, component, spec, state, device, options)
}
