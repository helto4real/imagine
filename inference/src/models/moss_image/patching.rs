//! MOSS aliases from the pinned ComfyUI model_lora_keys_unet, using logical shapes.
use crate::{
    checkpoint::PreparedParameters,
    native::Error,
    patching::{self, PatchOwner, SelectedLora, Target},
};
pub fn moss_key_map(weights: &[(String, Vec<i64>)]) -> Vec<(String, Target)> {
    let mut mapping = Vec::new();
    for (key, _) in weights {
        if let Some(name) = key.strip_prefix("diffusion_model.") {
            let target = Target {
                key: key.clone(),
                offset: None,
            };
            if let Some(stem) = name.strip_suffix(".weight") {
                mapping.push((
                    format!("lora_unet_{}", stem.replace('.', "_")),
                    target.clone(),
                ));
                mapping.push((key.trim_end_matches(".weight").into(), target));
            } else {
                mapping.push((key.clone(), target));
            }
        }
    }
    for (key, shape) in weights {
        let Some(name) = key.strip_prefix("diffusion_model.") else {
            continue;
        };
        if let Some(stem) = name.strip_suffix(".weight") {
            // Bare module paths, as LoRAs trained outside ComfyUI name them.
            mapping.push((
                stem.into(),
                Target {
                    key: key.clone(),
                    offset: None,
                },
            ));
        }
        if let Some(stem) = key.strip_suffix(".proj.qkv.weight") {
            // Fused q/k/v files; LoRAs address the separate projections.
            let third = shape[0] / 3;
            for (i, projection) in ["q", "k", "v"].iter().enumerate() {
                let name = format!("{stem}.proj.{projection}");
                let bare = name.trim_start_matches("diffusion_model.");
                let target = Target {
                    key: key.clone(),
                    offset: Some([0, i as i64 * third, third]),
                };
                mapping.push((name.clone(), target.clone()));
                mapping.push((bare.into(), target.clone()));
                mapping.push((format!("lora_unet_{}", bare.replace('.', "_")), target));
            }
        }
    }
    mapping
}
pub fn component_patch_owner(
    state: &PreparedParameters,
    loras: &[SelectedLora],
) -> Result<PatchOwner, Error> {
    let fused = state.contains_key("layers.0.proj.qkv.weight");
    let schema = super::moss_constructor_schema_for_layout(fused)?;
    let weights = schema.models["dit"]
        .parameters
        .iter()
        .map(|p| (format!("diffusion_model.{}", p.name), p.shape.clone()))
        .collect::<Vec<_>>();
    patching::bind_component(
        "diffusion_model.",
        state,
        loras,
        &moss_key_map(&weights),
        false,
    )
}
