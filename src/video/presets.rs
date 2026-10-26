//! Video presets: named scene recipes (H3 variant, settings overrides and
//! LoRAs) read from a local YAML file on each use, for MCP and the UI alike.
//! The two defaults are built in.
use super::{contract::*, editing::Edit, planning};
use imagine_video_contract::loras;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Component, Path};

pub const DEFAULT_REFERENCE: &str = "Default ref2v";
pub const DEFAULT_I2V: &str = "Default i2v";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema_version: u32,
    #[serde(default)]
    presets: Vec<Spec>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Spec {
    key: String,
    variant: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    settings: Option<Value>,
    #[serde(default)]
    loras: Vec<loras::Row>,
    #[serde(default)]
    lora_merge: Option<loras::Merge>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub key: String,
    pub variant: String,
    pub description: String,
    /// Overrides on the variant's settings; none keeps the project's settings.
    pub settings: Option<Value>,
    pub loras: Vec<loras::Row>,
    pub lora_merge: Option<loras::Merge>,
    /// Why the preset cannot be used, as a stable code.
    pub problem: Option<&'static str>,
}
impl Preset {
    fn builtin(key: &str, variant: &str, description: &str) -> Self {
        Self {
            key: key.into(),
            variant: variant.into(),
            description: description.into(),
            settings: None,
            loras: vec![],
            lora_merge: None,
            problem: None,
        }
    }
    pub fn is_default(&self) -> bool {
        self.key == default_key(&self.variant)
    }
    /// The scene's H3 settings: `base` (the project's settings for this model, or
    /// the variant's defaults) with this preset's overrides; lists replace.
    pub fn resolve(&self, base: Option<&Value>) -> Result<Option<Value>, &'static str> {
        let Some(overrides) = &self.settings else {
            return Ok(None);
        };
        let mut value = base.cloned().unwrap_or_else(|| {
            json!(imagine_video_contract::generation::h3_defaults_for(
                &self.variant
            ))
        });
        merge(&mut value, overrides);
        let settings: planning::H3Settings =
            serde_json::from_value(value.clone()).map_err(|_| "preset_settings_invalid")?;
        settings.validate().map_err(|_| "preset_settings_invalid")?;
        Ok(Some(value))
    }
    /// The scene LoRA override, when the preset names LoRAs or a merge mode.
    pub fn lora_override(&self) -> Option<loras::Override> {
        (!self.loras.is_empty() || self.lora_merge.is_some()).then(|| loras::Override {
            enabled: true,
            // Scene LoRAs add to the global stack unless the preset says otherwise.
            merge_mode: self.lora_merge.unwrap_or(loras::Merge::Add),
            main: loras::Stack {
                loras: self.loras.clone(),
                ..Default::default()
            },
        })
    }
    /// The model selection for this preset's variant.
    pub fn model(&self) -> Result<ModelSelection, &'static str> {
        super::adapters::h3::variants()
            .into_iter()
            .map(|(model, _)| model)
            .find(|m| m.variant == self.variant)
            .ok_or("preset_variant_invalid")
    }
    /// What applying this preset makes of one scene in `project`.
    pub fn scene_change(&self, project: &VideoProject) -> Result<SceneChange, &'static str> {
        let model = self.model()?;
        let base = project
            .generation
            .as_ref()
            .and_then(|g| g.settings.iter().find(|s| s.model == model))
            .map(|s| &s.values);
        Ok(SceneChange {
            choice: if model == project.default_model {
                ModelChoice::ProjectDefault
            } else {
                ModelChoice::Override(model.clone())
            },
            settings: self.resolve(base)?.map(|values| SettingsSlot {
                model: model.clone(),
                schema_version: 1,
                values,
            }),
            model,
            loras: self.lora_override(),
        })
    }
    /// One undoable edit batch setting `scenes` to this preset and recording it.
    pub fn scene_edits(
        &self,
        project: &VideoProject,
        scenes: &[SceneId],
    ) -> Result<Vec<Edit>, &'static str> {
        let change = self.scene_change(project)?;
        let mut creative = project.creative.clone();
        let mut edits = vec![];
        for id in scenes {
            project.scene(id).map_err(|_| "scene_not_found")?;
            edits.push(Edit::SetSceneModel {
                scene_id: id.clone(),
                model: change.choice.clone(),
            });
            edits.push(match &change.settings {
                Some(slot) => Edit::SetModelSettings {
                    scene_id: id.clone(),
                    settings: slot.clone(),
                },
                None => Edit::InheritModelSettings {
                    scene_id: id.clone(),
                    model: change.model.clone(),
                },
            });
            match &change.loras {
                Some(loras) => creative.scene_loras.insert(id.clone(), loras.clone()),
                None => creative.scene_loras.remove(id),
            };
            creative
                .scene_presets
                .insert(id.clone(), change.applied(&self.key));
        }
        edits.push(Edit::SetCreative { creative });
        Ok(edits)
    }
    /// One undoable edit batch making this preset the project default: its mode,
    /// the project's settings for that mode and, when it names LoRAs, the global
    /// LoRA stack.
    pub fn project_edits(&self, project: &VideoProject) -> Result<Vec<Edit>, &'static str> {
        let model = self.model()?;
        let mut generation = project.generation.clone().unwrap_or_default();
        let values = self.resolve(None)?.unwrap_or_else(|| {
            json!(imagine_video_contract::generation::h3_defaults_for(
                &self.variant
            ))
        });
        generation.settings.retain(|slot| slot.model != model);
        generation.settings.push(SettingsSlot {
            model: model.clone(),
            schema_version: 1,
            values,
        });
        let mut edits = vec![
            Edit::SetDefaultModel { model },
            Edit::SetGenerationDefaults {
                generation: Some(generation),
            },
        ];
        if !self.loras.is_empty() || self.lora_merge.is_some() {
            let mut creative = project.creative.clone();
            creative.h3_loras = Some(loras::Stack {
                loras: self.loras.clone(),
                ..Default::default()
            });
            edits.push(Edit::SetCreative { creative });
        }
        Ok(edits)
    }
    /// Model and LoRA files this preset names that are not files under `root`.
    pub fn missing_files(&self, root: &Path) -> Vec<String> {
        let mut files = vec![];
        if let Some(model) = self
            .settings
            .as_ref()
            .and_then(|s| s["components"]["model"].as_str())
        {
            files.push(model.to_owned());
        }
        files.extend(self.loras.iter().map(|row| format!("loras/{}", row.name)));
        files.retain(|file| !relative(file) || !root.join(file).is_file());
        files
    }
}
/// A scene's mode, own settings slot and LoRA override after applying a preset.
pub struct SceneChange {
    pub model: ModelSelection,
    pub choice: ModelChoice,
    /// None: the scene inherits the project's settings for `model`.
    pub settings: Option<SettingsSlot>,
    /// None: the scene has no LoRA override and uses the global stack.
    pub loras: Option<loras::Override>,
}
impl SceneChange {
    pub fn applied(&self, key: &str) -> imagine_video_contract::creative::AppliedPreset {
        imagine_video_contract::creative::AppliedPreset {
            key: key.into(),
            fingerprint: imagine_video_contract::creative::preset_fingerprint(
                &self.choice,
                self.settings.as_ref().map(|s| &s.values),
                self.loras.as_ref(),
            ),
        }
    }
}
pub fn default_key(variant: &str) -> &'static str {
    if variant == "i2v" {
        DEFAULT_I2V
    } else {
        DEFAULT_REFERENCE
    }
}
fn relative(path: &str) -> bool {
    let path = Path::new(path);
    !path.as_os_str().is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))
}
fn merge(target: &mut Value, overrides: &Value) {
    match (target, overrides) {
        (Value::Object(target), Value::Object(overrides)) => {
            for (key, value) in overrides {
                match target.get_mut(key) {
                    Some(slot) if slot.is_object() && value.is_object() => merge(slot, value),
                    _ => {
                        target.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (target, overrides) => *target = overrides.clone(),
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Catalog {
    pub presets: Vec<Preset>,
    /// Why the file could not be used; the built-in defaults remain.
    pub problem: Option<&'static str>,
}
impl Catalog {
    pub fn builtin() -> Self {
        Self::from_specs(vec![], None)
    }
    /// A missing file means the built-in defaults only.
    pub fn load(path: Option<&Path>) -> Self {
        let Some(path) = path.filter(|p| p.exists()) else {
            return Self::builtin();
        };
        match std::fs::read(path)
            .map_err(|_| "video_presets_unreadable")
            .and_then(|bytes| {
                serde_yaml::from_slice::<File>(&bytes).map_err(|_| "video_presets_invalid")
            }) {
            Ok(file) if file.schema_version == 1 => Self::from_specs(file.presets, None),
            Ok(_) => Self::from_specs(vec![], Some("video_presets_unsupported_schema")),
            Err(code) => Self::from_specs(vec![], Some(code)),
        }
    }
    fn from_specs(specs: Vec<Spec>, problem: Option<&'static str>) -> Self {
        let mut presets: Vec<Preset> = vec![];
        for spec in specs {
            let key = spec.key.trim().to_owned();
            let problem = if key.is_empty() || key.len() > 64 || key.chars().any(char::is_control) {
                Some("preset_key_invalid")
            } else if presets.iter().any(|p| p.key == key) {
                Some("preset_key_duplicate")
            } else if !["reference", "i2v"].contains(&spec.variant.as_str()) {
                Some("preset_variant_invalid")
            } else if spec.loras.iter().any(|row| {
                !relative(&row.name)
                    || !row.strength_model.is_finite()
                    || !row.strength_clip.is_finite()
            }) {
                Some("preset_lora_invalid")
            } else {
                None
            };
            let mut preset = Preset {
                key,
                variant: spec.variant,
                description: spec.description.trim().to_owned(),
                settings: spec.settings,
                loras: spec.loras,
                lora_merge: spec.lora_merge,
                problem,
            };
            if preset.problem.is_none() {
                preset.problem = preset.resolve(None).err();
            }
            presets.push(preset);
        }
        let builtins = [
            (
                DEFAULT_REFERENCE,
                "reference",
                "Standard MiniMax H3 reference-to-video",
            ),
            (DEFAULT_I2V, "i2v", "Standard MiniMax H3 image-to-video"),
        ]
        .into_iter()
        .filter(|(key, ..)| !presets.iter().any(|p| p.key == *key))
        .map(|(key, variant, description)| Preset::builtin(key, variant, description))
        .collect::<Vec<_>>();
        presets.splice(0..0, builtins);
        // Defaults first, reference before I2V; the file's order otherwise.
        presets.sort_by_key(|p| (!p.is_default(), p.is_default() && p.variant == "i2v"));
        Self { presets, problem }
    }
    pub fn get(&self, key: &str) -> Option<&Preset> {
        self.presets.iter().find(|p| p.key == key)
    }
    /// What an agent sees: keys, variants and availability, never file paths.
    pub fn summary(&self, root: Option<&Path>) -> Value {
        let presets = self
            .presets
            .iter()
            .map(|p| {
                let problem = p.problem.or_else(|| {
                    root.filter(|root| !p.missing_files(root).is_empty())
                        .map(|_| "preset_file_missing")
                });
                json!({"key":p.key,"variant":p.variant,"description":p.description,
                    "default":p.is_default(),"available":problem.is_none(),"problem":problem})
            })
            .collect::<Vec<_>>();
        json!({"presets":presets,"problem":self.problem})
    }
}

/// The full catalog for the local UI, including model and LoRA paths.
pub fn details(file: Option<&Path>, root: Option<&Path>) -> Value {
    let catalog = Catalog::load(file);
    let summary = catalog.summary(root);
    let presets = catalog
        .presets
        .iter()
        .zip(summary["presets"].as_array().cloned().unwrap_or_default())
        .map(|(p, mut entry)| {
            entry["settings"] = p.settings.clone().unwrap_or(Value::Null);
            entry["loras"] = json!(p.loras);
            entry["lora_merge"] = json!(p.lora_merge.unwrap_or(loras::Merge::Add));
            entry
        })
        .collect::<Vec<_>>();
    json!({"schema_version":1,"file":file,"problem":catalog.problem,"presets":presets})
}
/// Authoring limits with the preset catalog, as capabilities, profile
/// descriptions and bundles all report them.
pub fn limits(file: Option<&Path>, root: Option<&Path>) -> Value {
    let mut limits = super::authoring::steps::limits();
    let mut summary = Catalog::load(file).summary(root);
    limits["presets"] = summary["presets"].take();
    limits["presets_problem"] = summary["problem"].take();
    limits
}

#[cfg(test)]
mod tests {
    use super::*;
    fn catalog(yaml: &str) -> Catalog {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("video-presets.yaml");
        std::fs::write(&path, yaml).unwrap();
        Catalog::load(Some(&path))
    }
    #[test]
    fn built_in_defaults_remain_and_presets_override_only_what_they_name() {
        let builtin = Catalog::load(Some(Path::new("/nonexistent/video-presets.yaml")));
        assert_eq!(
            builtin
                .presets
                .iter()
                .map(|p| p.key.as_str())
                .collect::<Vec<_>>(),
            [DEFAULT_REFERENCE, DEFAULT_I2V]
        );
        let presets = catalog(
            "schema_version: 1\npresets:\n  - key: Own tuned\n    variant: reference\n    description: Own fine-tune\n    settings:\n      components:\n        model: diffusion_models/own_tuned.safetensors\n      stage_one:\n        steps: 24\n    loras:\n      - name: h3/own_style.safetensors\n        strength_model: 0.8\n",
        );
        assert_eq!(presets.problem, None);
        let tuned = presets.get("Own tuned").unwrap();
        assert_eq!(tuned.problem, None);
        let settings = tuned.resolve(None).unwrap().unwrap();
        assert_eq!(
            settings["components"]["model"],
            "diffusion_models/own_tuned.safetensors"
        );
        assert_eq!(settings["stage_one"]["steps"], 24);
        // Unnamed values keep the variant's defaults.
        assert_eq!(settings["stage_one"]["sampler"], "res_multistep");
        let lora = tuned.lora_override().unwrap();
        assert_eq!(lora.merge_mode, loras::Merge::Add);
        assert_eq!(lora.main.loras[0].strength_model, 0.8);
        assert!(
            presets
                .get(DEFAULT_REFERENCE)
                .unwrap()
                .resolve(None)
                .unwrap()
                .is_none()
        );
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            tuned.missing_files(dir.path()),
            [
                "diffusion_models/own_tuned.safetensors",
                "loras/h3/own_style.safetensors"
            ]
        );
    }
    #[test]
    fn the_tracked_example_file_is_valid() {
        let example = Catalog::load(Some(Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/config/video/presets.example.yaml"
        ))));
        assert_eq!(example.problem, None);
        assert!(example.presets.iter().all(|p| p.problem.is_none()));
        assert_eq!(example.presets.len(), 2);
    }
    #[test]
    fn invalid_presets_are_listed_with_a_reason_and_bad_files_keep_the_defaults() {
        let presets = catalog(
            "schema_version: 1\npresets:\n  - key: Own a\n    variant: video\n  - key: Own b\n    variant: reference\n    settings:\n      stage_one:\n        steps: 0\n  - key: Own b\n    variant: i2v\n  - key: Own c\n    variant: reference\n    loras:\n      - name: ../outside.safetensors\n",
        );
        let problem = |key: &str| {
            presets
                .presets
                .iter()
                .find(|p| p.key == key)
                .unwrap()
                .problem
        };
        assert_eq!(problem("Own a"), Some("preset_variant_invalid"));
        assert_eq!(problem("Own b"), Some("preset_settings_invalid"));
        assert_eq!(problem("Own c"), Some("preset_lora_invalid"));
        assert_eq!(
            presets
                .presets
                .iter()
                .filter(|p| p.key == "Own b")
                .nth(1)
                .unwrap()
                .problem,
            Some("preset_key_duplicate")
        );
        for (yaml, code) in [
            ("schema_version: 2\n", "video_presets_unsupported_schema"),
            ("presets: [", "video_presets_invalid"),
            (
                "schema_version: 1\nunknown: true\n",
                "video_presets_invalid",
            ),
        ] {
            let bad = catalog(yaml);
            assert_eq!(bad.problem, Some(code));
            assert_eq!(bad.presets.len(), 2);
        }
        let summary = presets.summary(None);
        assert!(!summary.to_string().contains("safetensors"));
    }
}
