use crate::models::qwen_image21::normalized_text_key;
// Original constructor/schema and static component inventory for local MOSS.
use crate::{
    checkpoint::PreparedParameters,
    components::{
        ComponentBuffers, ComponentInventory, ComponentLoad, ConstructorSchema, StaticOffload,
        prepare_resident_component_with_options, static_component_from_inventory,
    },
    execution::Runtime,
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
    let fused = component != "dit" || parameters.contains_key("layers.0.proj.qkv.weight");
    static_component_from_inventory(
        rt,
        parameters,
        moss_component_inventory_for_layout(component, fused)?,
    )
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
    let fused = component != "dit"
        || state
            .iter()
            .any(|(name, _)| name == "layers.0.proj.qkv.weight");
    let schema = moss_constructor_schema_for_layout(fused)?;
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

/// Alternate constructor metadata only; checkpoint tensors are never fused or split.
pub fn moss_constructor_schema_for_layout(fused_qkv: bool) -> Result<ConstructorSchema, Error> {
    let mut schema = moss_constructor_schema()?;
    if !fused_qkv {
        let dit = schema
            .models
            .get_mut("dit")
            .ok_or_else(|| Error::Native("Missing MOSS DiT schema".into()))?;
        dit.parameters = std::mem::take(&mut dit.parameters)
            .into_iter()
            .flat_map(|p| {
                if let Some(stem) = p.name.strip_suffix(".proj.qkv.weight") {
                    ["q", "k", "v"]
                        .iter()
                        .map(|projection| {
                            let mut separate = p.clone();
                            separate.name = format!("{stem}.proj.{projection}.weight");
                            separate.shape[0] /= 3;
                            separate
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![p]
                }
            })
            .collect();
    }
    Ok(schema)
}
pub fn moss_component_inventory_for_layout(
    component: &str,
    fused_qkv: bool,
) -> Result<ComponentInventory, Error> {
    let mut inventory = moss_component_inventory(component)?;
    if component == "dit" && !fused_qkv {
        inventory.modules = inventory
            .modules
            .into_iter()
            .flat_map(|m| {
                if let Some(stem) = m.name.strip_suffix(".proj.qkv") {
                    ["q", "k", "v"]
                        .iter()
                        .map(|projection| {
                            let name = format!("{stem}.proj.{projection}");
                            let mut separate = m.clone();
                            separate.parameters = vec![format!("{name}.weight")];
                            separate.name = name;
                            separate.module_bytes /= 3;
                            separate.offload_bytes /= 3;
                            separate
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![m]
                }
            })
            .collect();
    }
    Ok(inventory)
}
