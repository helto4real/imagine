//! MiniMaxH3 aliases and original static ModelPatcher ownership, model-only.
use super::protocol::options::Lora;
use crate::{
    checkpoint::PreparedParameters,
    execution_value::Value,
    native::Error,
    patching::{self, PatchOwner, Target},
};
use std::{collections::BTreeMap, path::Path};
pub fn key_map(keys: impl IntoIterator<Item = String>) -> Vec<(String, Target)> {
    let mut mapping = vec![];
    for key in keys {
        let target = Target {
            key: key.clone(),
            offset: None,
        };
        if let Some(stem) = key.strip_suffix(".weight") {
            mapping.push((
                format!("lora_unet_{}", stem.replace('.', "_")),
                target.clone(),
            ));
            mapping.push((format!("diffusion_model.{stem}"), target.clone()));
            mapping.push((stem.to_owned(), target));
        } else {
            mapping.push((format!("diffusion_model.{key}"), target));
        }
    }
    mapping
}
pub fn owner(state: &PreparedParameters, root: &Path, rows: &[Lora]) -> Result<PatchOwner, Error> {
    let mapping = key_map(state.values.keys().cloned());
    let mut owner = PatchOwner::new();
    owner.key_prefix = "diffusion_model.".into();
    for row in rows.iter().filter(|row| row.strength != 0.) {
        let path = root.join("loras").join(&row.name);
        let values = tch::Tensor::read_safetensors(path)?
            .into_iter()
            .map(|(key, t)| (key, Value::tensor(t)))
            .collect::<BTreeMap<_, _>>();
        let loaded = patching::load_lora(&values, &mapping)?;
        owner.diagnostics.warnings.extend(
            loaded
                .missing_keys
                .iter()
                .map(|key| format!("lora key not loaded: {key}")),
        );
        owner.add(state, &loaded, row.strength);
    }
    Ok(owner)
}
