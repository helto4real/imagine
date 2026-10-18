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
    let mut owner = PatchOwner::new();
    owner.key_prefix = prefix.into();
    for (lora, model_strength, text_strength) in loras {
        let mut loaded = patching::load_lora(lora, &mapping)?;
        loaded.patches.retain_mut(|(target, _)| {
            if let Some(key) = target.key.strip_prefix(prefix) {
                target.key = key.to_owned();
                true
            } else {
                false
            }
        });
        owner.add(
            state,
            &loaded,
            if component == "dit" {
                *model_strength
            } else {
                *text_strength
            },
        );
    }
    Ok(owner)
}
