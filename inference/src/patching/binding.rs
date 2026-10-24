//! Shared LoRA binding. Parsing and application order are owned by the caller.
use super::{PatchOwner, Target, load_lora};
use crate::{
    checkpoint::{Aimdo, PreparedParameters, Safetensors},
    execution::Runtime,
    execution_value::Value,
    native::Error,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
pub type LoraState = BTreeMap<String, Value>;
pub type SelectedLora = (LoraState, f64, f64);
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FilePatchReport {
    pub consumed: BTreeSet<String>,
    pub unmatched: BTreeSet<String>,
    pub parsed_targets: Vec<Target>,
    pub accepted_targets: Vec<Target>,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PatchReport {
    pub files: Vec<FilePatchReport>,
}
impl PatchReport {
    /// Components refer to the same ordered selection; never classify per-component misses as file failures.
    pub fn merge(&mut self, other: &Self) {
        self.files
            .resize_with(self.files.len().max(other.files.len()), Default::default);
        for (a, b) in self.files.iter_mut().zip(&other.files) {
            a.consumed.extend(b.consumed.iter().cloned());
            a.unmatched.extend(b.unmatched.iter().cloned());
            a.unmatched.retain(|k| !a.consumed.contains(k));
            for (targets, more) in [
                (&mut a.parsed_targets, &b.parsed_targets),
                (&mut a.accepted_targets, &b.accepted_targets),
            ] {
                for target in more {
                    if !targets.contains(target) {
                        targets.push(target.clone());
                    }
                }
            }
        }
    }
}
pub fn read_lora(rt: &Runtime, aimdo: &Aimdo, path: &Path) -> Result<LoraState, Error> {
    Ok(Safetensors::load_aimdo(path, aimdo, &rt.native)?
        .tensors
        .into_iter()
        .map(|(n, t)| (n, Value::tensor(t.tensor)))
        .collect())
}
pub fn bind_component(
    prefix: &str,
    state: &PreparedParameters,
    loras: &[SelectedLora],
    mapping: &[(String, Target)],
    text_strength: bool,
) -> Result<PatchOwner, Error> {
    let mut owner = PatchOwner::new();
    owner.key_prefix = prefix.into();
    for (lora, model_strength, clip_strength) in loras {
        let mut loaded = load_lora(lora, mapping)?;
        let mut report = FilePatchReport {
            consumed: loaded.loaded_keys.clone(),
            unmatched: loaded.missing_keys.iter().cloned().collect(),
            parsed_targets: loaded.patches.iter().map(|(t, _)| t.clone()).collect(),
            ..Default::default()
        };
        loaded.patches.retain_mut(|(target, _)| {
            if let Some(key) = target.key.strip_prefix(prefix) {
                target.key = key.to_owned();
                true
            } else {
                false
            }
        });
        let accepted = owner.add(
            state,
            &loaded,
            if text_strength {
                *clip_strength
            } else {
                *model_strength
            },
        );
        report
            .accepted_targets
            .extend(accepted.iter().map(|t| Target {
                key: format!("{prefix}{}", t.key),
                offset: t.offset,
            }));
        owner.report.files.push(report);
    }
    Ok(owner)
}
