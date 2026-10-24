//! Qwen component inventory; generic construction belongs to components.
use super::{ComponentBuffers, StaticOffload};
pub use crate::components::{ComponentBuffer, ComponentInventory, static_component_from_inventory};
use crate::{checkpoint::PreparedParameters, execution::Runtime, native::Error};
use serde::Deserialize;
use std::collections::BTreeMap;
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
