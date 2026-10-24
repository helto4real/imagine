//! Constructor-locked initial resident test path. The full Comfy model-family
//! loader, missing-key recovery and alternate constructors remain separate work.
pub use crate::components::{
    BufferSpec, ComponentLoad, ComponentSchema, ConstructorSchema, ParameterSpec, VaeWidths,
    prepare_resident_component, prepare_resident_component_with_options,
};
use crate::{checkpoint::PreparedParameters, execution::Runtime, native::Error};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
pub fn constructor_schema() -> Result<ConstructorSchema, Error> {
    serde_json::from_str(include_str!("../../../resources/qwen-image21-schema.json"))
        .map_err(|e| Error::Native(e.to_string()))
}
pub fn normalize_text_keys(state: BTreeMap<String, Tensor>) -> BTreeMap<String, Tensor> {
    state
        .into_iter()
        .map(|(name, t)| (normalized_text_key(&name), t))
        .collect()
}
pub fn normalized_text_key(name: &str) -> String {
    if let Some(suffix) = name.strip_prefix("model.language_model.") {
        format!("model.{suffix}")
    } else if let Some(suffix) = name.strip_prefix("model.visual.") {
        format!("visual.{suffix}")
    } else if let Some(suffix) = name.strip_prefix("lm_head.") {
        format!("model.lm_head.{suffix}")
    } else {
        name.into()
    }
}
fn gap(op: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "qwen21_initial_checkpoint".into(),
        operation: op.into(),
    }
}
/// Load all constructor parameters, including unused visual/lm-head parameters.
/// The selected initial source configuration uses strict complete checkpoints,
/// static patchers, resident phases, BF16 compute and no dynamic vbars/compiler.
/// Unsupported loader branches return an explicit gap before trained execution.
pub fn prepare_initial_component(
    rt: &Runtime,
    component: &str,
    state: BTreeMap<String, Tensor>,
    device: Device,
) -> Result<PreparedParameters, Error> {
    prepare_initial_component_with_options(
        rt,
        component,
        state,
        device,
        ComponentLoad {
            dtype: Kind::BFloat16,
            freeze: component != "vae",
        },
    )
}
pub fn prepare_initial_component_with_options(
    rt: &Runtime,
    component: &str,
    state: BTreeMap<String, Tensor>,
    device: Device,
    options: ComponentLoad,
) -> Result<PreparedParameters, Error> {
    let schema = constructor_schema()?;
    let spec = schema
        .models
        .get(component)
        .ok_or_else(|| gap("Unknown constructor component"))?;
    prepare_resident_component_with_options(rt, component, spec, state, device, options)
}
