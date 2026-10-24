//! Qwen's original model/CLIP alias owners; no model or adapter fallback.
use crate::{
    checkpoint::PreparedParameters,
    execution_value::Value,
    native::Error,
    patching::{self, PatchOwner},
};
use std::collections::BTreeMap;
pub fn component_patch_owner(
    component: &str,
    state: &PreparedParameters,
    loras: &[(BTreeMap<String, Value>, f64, f64)],
) -> Result<PatchOwner, Error> {
    let prefix = match component {
        "dit" => "diffusion_model.",
        "text" => "qwen3vl_8b.transformer.",
        _ => {
            return Err(patching::gap(
                "Only original model/CLIP LoRA owners are defined",
            ));
        }
    };
    let schema = super::constructor_schema()?;
    let weights = schema.models["dit"]
        .parameters
        .iter()
        .map(|p| (format!("diffusion_model.{}", p.name), p.shape.clone()))
        .collect::<Vec<_>>();
    let mut mapping = patching::qwen_key_map(&weights);
    mapping.extend(patching::text_key_map(
        &schema.models["text"]
            .parameters
            .iter()
            .map(|p| format!("qwen3vl_8b.transformer.{}", p.name))
            .collect::<Vec<_>>(),
    ));
    patching::bind_component(prefix, state, loras, &mapping, component == "text")
}
