//! Server-owned YAML defaults. Resolve once before queue admission; workers only
//! see the validated, persisted request, never mutable configuration files.
use crate::{
    contract::{JobRequest, Lora, ResolvedProfile},
    scheduler::Fault,
    storage::{Store, hash},
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema_version: u32,
    pub id: String,
    pub version: u32,
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub prompt_guidance: Option<String>,
    pub defaults: Value,
    #[serde(default)]
    pub references: References,
    #[serde(default)]
    pub sizing: Option<Sizing>,
    #[serde(default)]
    pub nsfw_loras: Vec<Lora>,
    #[serde(skip)]
    sha256: String,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct References {
    pub min: usize,
    pub max: usize,
}
impl Default for References {
    fn default() -> Self {
        Self { min: 0, max: 10 }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Sizing {
    pub max_side: u32,
    pub reference_aspect: bool,
}
#[derive(Clone, Default)]
pub struct Catalog {
    profiles: BTreeMap<String, Profile>,
}

fn invalid(code: &'static str) -> Fault {
    Fault::new(422, code)
}
fn merge(base: &mut Value, patch: Value) {
    if let (Some(base), Value::Object(patch)) = (base.as_object_mut(), &patch) {
        for (key, value) in patch {
            merge(
                base.entry(key.clone()).or_insert(Value::Null),
                value.clone(),
            );
        }
    } else {
        *base = patch;
    }
}
impl Catalog {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut catalog = Self::default();
        let mut paths = std::fs::read_dir(dir)
            .context("profile directory")?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        paths.sort();
        for path in paths {
            if !matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("yaml" | "yml")
            ) {
                continue;
            }
            if path.metadata()?.len() > 1024 * 1024 {
                bail!("profile file is too large: {}", path.display());
            }
            let bytes = std::fs::read(&path)?;
            // Parse via YAML Value first so duplicate mapping keys fail, including defaults.
            let yaml: serde_yaml::Value = serde_yaml::from_slice(&bytes)
                .with_context(|| format!("invalid YAML: {}", path.display()))?;
            let mut profile: Profile = serde_yaml::from_value(yaml)
                .with_context(|| format!("invalid profile: {}", path.display()))?;
            profile.sha256 = hash(&bytes);
            profile
                .check()
                .with_context(|| format!("invalid profile: {}", path.display()))?;
            if catalog
                .profiles
                .insert(profile.id.clone(), profile)
                .is_some()
            {
                bail!("duplicate profile ID");
            }
        }
        if catalog.profiles.is_empty() {
            bail!("no YAML profiles found");
        }
        Ok(catalog)
    }
    pub fn list(&self) -> Value {
        json!({"profiles":self.profiles.values().map(Profile::summary).collect::<Vec<_>>()})
    }
    pub fn describe(&self, id: &str) -> Result<Value, Fault> {
        let p = self
            .profiles
            .get(id)
            .ok_or(Fault::new(404, "profile_not_found"))?;
        let mut value = p.summary();
        value["defaults"] = p.defaults.clone();
        value["has_default_prompt"] = json!(p.defaults.get("prompt").is_some());
        for key in ["prompt", "negative_prompt"] {
            value["defaults"].as_object_mut().unwrap().remove(key);
        }
        value["references"] = json!(p.references);
        value["sizing"] = json!(p.sizing);
        value["nsfw_available"] = json!(!p.nsfw_loras.is_empty());
        value["prompt_guidance"] = json!(p.prompt_guidance);
        let mut schema = serde_json::to_value(schemars::schema_for!(JobRequest)).unwrap();
        schema.as_object_mut().unwrap().remove("required");
        schema["properties"]
            .as_object_mut()
            .unwrap()
            .remove("resolved_profile");
        for (name, spec) in [
            ("nsfw", json!({"type":"boolean","default":false})),
            (
                "max_side",
                json!({"type":"integer","minimum":256,"maximum":4096,"multipleOf":32,"description":"Longest image edge in pixels; maximum 4096. For image_upscale, 3200 is only the default when omitted, not the maximum."}),
            ),
            (
                "aspect_ratio",
                json!({"type":"string","description":"Positive width:height ratio; requires max_side, conflicts with explicit dimensions."}),
            ),
        ] {
            schema["properties"][name] = spec;
        }
        // Every nested object is a partial override; final merged request is validated normally.
        fn partial(v: &mut Value) {
            if let Some(o) = v.as_object_mut() {
                o.remove("required");
                o.remove("default");
                for value in o.values_mut() {
                    partial(value);
                }
            } else if let Some(a) = v.as_array_mut() {
                for value in a {
                    partial(value);
                }
            }
        }
        partial(&mut schema);
        // Arrays replace as a whole: each LoRA still needs both fields.
        schema["$defs"]["Lora"]["required"] = json!(["path", "strength"]);
        if p.defaults.get("prompt").is_none() {
            schema["required"] = json!(["prompt"]);
        }
        value["parameters_schema"] = schema;
        Ok(value)
    }
    pub async fn resolve(
        &self,
        mut value: Value,
        store: &Store,
        owner: &str,
    ) -> Result<JobRequest, Fault> {
        let object = value.as_object_mut().ok_or(invalid("invalid_request"))?;
        if object.contains_key("resolved_profile") {
            return Err(invalid("read_only_profile_metadata"));
        }
        let selected = object.remove("profile");
        let version = object.remove("profile_version");
        let Some(selected) = selected else {
            if version.is_some() {
                return Err(invalid("profile_required"));
            }
            let request: JobRequest =
                serde_json::from_value(value).map_err(|_| invalid("invalid_request"))?;
            request.validate().map_err(invalid)?;
            return Ok(request);
        };
        let id = selected.as_str().ok_or(invalid("invalid_profile"))?;
        let profile = self.profiles.get(id).ok_or(invalid("unknown_profile"))?;
        if let Some(version) = version
            && version.as_u64() != Some(u64::from(profile.version))
        {
            return Err(Fault::new(409, "profile_version_mismatch"));
        }
        let nsfw = object
            .remove("nsfw")
            .map(|v| v.as_bool().ok_or(invalid("invalid_nsfw")))
            .transpose()?
            .unwrap_or(false);
        if nsfw && profile.nsfw_loras.is_empty() {
            return Err(invalid("nsfw_not_configured"));
        }
        let max_side = object.remove("max_side");
        let aspect = object.remove("aspect_ratio");
        let explicit_dimensions = object.contains_key("width") || object.contains_key("height");
        if explicit_dimensions && (max_side.is_some() || aspect.is_some()) {
            return Err(invalid("conflicting_dimensions"));
        }
        let mut merged = profile.defaults.clone();
        merge(&mut merged, value);
        let refs = merged["inputs"]["reference_images"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        if refs.len() < profile.references.min || refs.len() > profile.references.max {
            return Err(invalid("invalid_reference_count"));
        }
        if merged.get("operation").is_none() {
            merged["operation"] = json!(if refs.is_empty() {
                "text_to_image"
            } else {
                "image_edit"
            });
        }
        if !explicit_dimensions
            && (max_side.is_some() || profile.sizing.is_some() || aspect.is_some())
        {
            let max = max_side
                .as_ref()
                .map(|v| v.as_u64().ok_or(invalid("invalid_max_side")))
                .transpose()?
                .or_else(|| profile.sizing.as_ref().map(|s| u64::from(s.max_side)))
                .ok_or(invalid("max_side_required"))?;
            if !(256..=4096).contains(&max) || max % 32 != 0 {
                return Err(invalid("invalid_max_side"));
            }
            let reference_aspect =
                profile.sizing.as_ref().is_some_and(|s| s.reference_aspect) && aspect.is_none();
            let (w, h) = if let Some(aspect) = aspect {
                let (a, b) = aspect
                    .as_str()
                    .and_then(|s| s.split_once(':'))
                    .ok_or(invalid("invalid_aspect_ratio"))?;
                (
                    a.parse::<f64>()
                        .map_err(|_| invalid("invalid_aspect_ratio"))?,
                    b.parse::<f64>()
                        .map_err(|_| invalid("invalid_aspect_ratio"))?,
                )
            } else if reference_aspect {
                let id = refs
                    .first()
                    .and_then(Value::as_str)
                    .ok_or(invalid("invalid_reference_count"))?;
                let asset = store
                    .asset(id, owner)
                    .await?
                    .ok_or(Fault::new(404, "asset_not_found"))?;
                (
                    asset["width"].as_f64().unwrap(),
                    asset["height"].as_f64().unwrap(),
                )
            } else {
                (
                    merged["width"].as_f64().unwrap_or(1600.),
                    merged["height"].as_f64().unwrap_or(896.),
                )
            };
            if !w.is_finite() || !h.is_finite() || w <= 0. || h <= 0. {
                return Err(invalid("invalid_aspect_ratio"));
            }
            let aligned = |n: f64| {
                let units = n / 32.;
                (if reference_aspect {
                    units.floor()
                } else {
                    units.round_ties_even()
                } as u32
                    * 32)
                    .max(32)
            };
            merged["width"] = json!(aligned(max as f64 * (w / w.max(h))));
            merged["height"] = json!(aligned(max as f64 * (h / w.max(h))));
        }
        let mut request: JobRequest =
            serde_json::from_value(merged).map_err(|_| invalid("invalid_request"))?;
        if nsfw {
            request
                .model_options
                .qwen_image_2_1
                .workflow
                .as_mut()
                .ok_or(invalid("nsfw_not_configured"))?
                .loras
                .extend(profile.nsfw_loras.clone());
        }
        request.prompt = crate::privacy::decrypt(&request.prompt)
            .map_err(|_| invalid("private_prompt_unavailable"))?;
        request.negative_prompt = crate::privacy::decrypt(&request.negative_prompt)
            .map_err(|_| invalid("private_prompt_unavailable"))?;
        request.validate().map_err(invalid)?;
        request.resolved_profile = Some(ResolvedProfile {
            id: profile.id.clone(),
            version: profile.version,
            sha256: profile.sha256.clone(),
        });
        Ok(request)
    }
}
impl Profile {
    fn summary(&self) -> Value {
        json!({"id":self.id,"version":self.version,"sha256":self.sha256,"name":self.name,"description":self.description,"model":self.defaults["model"]})
    }
    fn check(&self) -> Result<()> {
        if self.schema_version != 1
            || self.version == 0
            || self.id.is_empty()
            || self.id.len() > 128
            || !self
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
            || self.references.min > self.references.max
            || self.references.max > 10
        {
            bail!("invalid profile metadata");
        }
        let mut value = self.defaults.clone();
        let object = value
            .as_object_mut()
            .context("defaults must be an object")?;
        if object.contains_key("resolved_profile") {
            bail!("defaults cannot set resolved_profile");
        }
        object
            .entry("prompt")
            .or_insert(json!("profile validation"));
        object.entry("operation").or_insert(json!("text_to_image"));
        let mut request: JobRequest = serde_json::from_value(value).context("invalid defaults")?;
        // Validate reference-dependent defaults without reading assets or private prompts.
        if self.references.min > 0 {
            request.operation = crate::contract::Operation::ImageEdit;
            request.inputs.reference_images = vec!["validation".into(); self.references.min];
        }
        request.validate().map_err(anyhow::Error::msg)?;
        if !self.nsfw_loras.is_empty() {
            request
                .model_options
                .qwen_image_2_1
                .workflow
                .as_mut()
                .context("nsfw_loras require generation settings")?
                .loras
                .extend(self.nsfw_loras.clone());
            request.validate().map_err(anyhow::Error::msg)?;
        }
        if let Some(s) = &self.sizing
            && (!(256..=4096).contains(&s.max_side)
                || s.max_side % 32 != 0
                || (s.reference_aspect && self.references.min == 0))
        {
            bail!("invalid profile sizing");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_broken_yaml_and_defaults_without_loading_models() {
        let dir = tempfile::tempdir().unwrap();
        let good = include_str!("../profiles/qwen.yaml");
        for bad in [
            good.replace("steps: 40", "steps: 0"),
            good.replace("steps: 40", "steps: 40\n    steps: 12"),
            good.replace("schema_version: 1", "schema_version: 99"),
            good.replace("sampler: er_sde", "sampler: unknown"),
            good.replace("strength: 1.0", "strength: .nan"),
            good.replace("  model: qwen", "  unknown: 1\n  model: qwen"),
        ] {
            std::fs::write(dir.path().join("qwen.yaml"), bad).unwrap();
            assert!(Catalog::load(dir.path()).is_err());
        }
        std::fs::write(dir.path().join("qwen.yaml"), good).unwrap();
        let catalog = Catalog::load(dir.path()).unwrap();
        std::fs::write(dir.path().join("copy.yaml"), good).unwrap();
        assert!(Catalog::load(dir.path()).is_err());
        // A loaded catalog is immutable until the server restarts.
        assert_eq!(
            catalog.describe("qwen").unwrap()["defaults"]["sampling"]["steps"],
            40
        );
    }
}
