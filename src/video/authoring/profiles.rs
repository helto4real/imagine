//! Director's current v3 prompt profiles, including operation/default inheritance.
//! Reads only the caller-selected source-owned profile root; snapshots own their text.
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};

pub const OPERATIONS: [&str; 6] = [
    "text_enhancement",
    "image_analysis",
    "story_generation",
    "story_outline",
    "story_scene_expansion",
    "animate_image",
];
pub const PROVIDERS: [&str; 2] = ["ollama", "koboldcpp"];
pub const PROMPTS: [(&str, &str); 8] = [
    ("animate_image", "animate_image.txt"),
    ("enhance_scene_field", "enhance_scene_field.txt"),
    ("enhance_section", "enhance_section.txt"),
    ("story_image_inventory", "story_image_inventory.txt"),
    ("story_outline", "story_outline.txt"),
    ("story_outline_dialogue", "story_outline_dialogue.txt"),
    ("story_scene", "story_scene.txt"),
    ("story_scene_dialogue", "story_scene_dialogue.txt"),
];
type Parameters = Map<String, Value>;
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Generation {
    #[serde(default)]
    pub defaults: Parameters,
    #[serde(default)]
    pub operations: BTreeMap<String, Parameters>,
    #[serde(default)]
    pub providers: BTreeMap<String, Provider>,
}
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provider {
    #[serde(default)]
    pub defaults: Parameters,
    #[serde(default)]
    pub operations: BTreeMap<String, Parameters>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    id: String,
    label: String,
    default: bool,
    order: i64,
    generation: Value,
}
fn legacy_generation() -> Result<Generation> {
    normalize(
        json!({"defaults":{"temperature":0.05,"top_p":0.9,"top_k":40,"repeat_penalty":1.05,"max_tokens":4500},"operations":{"text_enhancement":{},"image_analysis":{},"story_generation":{}},"providers":{"ollama":{"defaults":{"repeat_penalty":1.1,"max_tokens":1024,"context_tokens":0},"operations":{}},"koboldcpp":{"defaults":{"min_p":0.05,"reasoning":{"include":true,"effort":"medium","enable_thinking":true}},"operations":{"image_analysis":{"max_tokens":4500},"story_generation":{"max_tokens":8000}}}}}),
        true,
    )
}
fn manifest(bytes: &[u8]) -> Result<Manifest> {
    let mut raw: Value = serde_json::from_slice(bytes).map_err(|_| "invalid_profile")?;
    if raw
        .get("schema_version")
        .is_some_and(|v| !matches!(v.as_u64(), Some(2 | 3)))
    {
        return Err("invalid_profile");
    }
    if raw.get("schema_version").is_none() {
        let map = raw.as_object_mut().ok_or("invalid_profile")?;
        if map.len() != 5
            || !["id", "label", "temperature", "default", "order"]
                .iter()
                .all(|key| map.contains_key(*key))
        {
            return Err("invalid_profile");
        }
        let temperature = map
            .remove("temperature")
            .filter(|value| {
                value
                    .as_f64()
                    .is_some_and(|t| t.is_finite() && (0. ..=2.).contains(&t))
            })
            .ok_or("invalid_profile_parameters")?;
        map.insert("schema_version".into(), json!(1));
        map.insert("generation".into(),json!({"defaults":{"temperature":temperature},"operations":{"text_enhancement":{"temperature":temperature},"image_analysis":{"temperature":temperature},"story_generation":{"temperature":temperature}},"providers":{}}));
    }
    serde_json::from_value(raw).map_err(|_| "invalid_profile")
}
fn legacy_animation(mut generation: Generation) -> Result<Generation> {
    for provider in PROVIDERS {
        let settings = resolve(&generation, provider, "story_scene_expansion")?;
        generation
            .providers
            .entry(provider.into())
            .or_default()
            .operations
            .insert("animate_image".into(), settings);
    }
    Ok(generation)
}
#[derive(Clone, Serialize)]
pub struct PromptFile {
    pub key: String,
    pub file: String,
    pub source_profile_id: String,
    pub inherited: bool,
    pub digest: String,
}
#[derive(Clone, Serialize)]
pub struct Profile {
    pub schema_version: u32,
    pub id: String,
    pub label: String,
    pub default: bool,
    pub order: i64,
    pub temperature: f64,
    pub generation: Generation,
    pub revision: String,
    pub prompt_files: Vec<PromptFile>,
}
#[derive(Clone, Serialize)]
pub struct Catalog {
    pub profiles: Vec<Profile>,
    pub default_id: String,
    pub warnings: Vec<String>,
    #[serde(skip)]
    pub prompts: BTreeMap<String, BTreeMap<String, String>>,
}
/// Freeze at admission; later Reload never changes an in-flight authoring request.
#[derive(Clone, Serialize)]
pub struct Snapshot {
    pub profile: Profile,
    pub requested_id: String,
    pub resolved_id: String,
    pub fallback: bool,
    pub warning: String,
    pub prompts: BTreeMap<String, String>,
}
fn fallback(operation: &str) -> Option<&'static str> {
    matches!(operation, "story_outline" | "story_scene_expansion").then_some("story_generation")
}
fn update(target: &mut Parameters, values: &Parameters) {
    for (key, value) in values {
        if key == "reasoning" {
            let reasoning = target.entry(key.clone()).or_insert_with(|| json!({}));
            reasoning
                .as_object_mut()
                .unwrap()
                .extend(value.as_object().unwrap().clone());
        } else {
            target.insert(key.clone(), value.clone());
        }
    }
}
pub fn resolve(generation: &Generation, provider: &str, operation: &str) -> Result<Parameters> {
    let provider = provider.trim().to_lowercase();
    let operation = operation.trim().to_lowercase();
    if !PROVIDERS.contains(&provider.as_str()) || !OPERATIONS.contains(&operation.as_str()) {
        return Err("unsupported_profile_generation_target");
    }
    let mut resolved = generation.defaults.clone();
    if let Some(values) = fallback(&operation).and_then(|v| generation.operations.get(v)) {
        resolved.extend(values.clone());
    }
    if let Some(values) = generation.operations.get(&operation) {
        resolved.extend(values.clone());
    }
    if let Some(block) = generation.providers.get(&provider) {
        update(&mut resolved, &block.defaults);
        if let Some(values) = fallback(&operation).and_then(|v| block.operations.get(v)) {
            update(&mut resolved, values);
        }
        if let Some(values) = block.operations.get(&operation) {
            update(&mut resolved, values);
        }
    }
    Ok(resolved)
}
fn inherited(base: &Generation, overlay: Generation) -> Generation {
    let mut result = base.clone();
    for operation in ["story_outline", "story_scene_expansion"] {
        if !overlay.operations.contains_key(operation) {
            result.operations.remove(operation);
        }
        for provider in PROVIDERS {
            if !overlay
                .providers
                .get(provider)
                .is_some_and(|p| p.operations.contains_key(operation))
                && let Some(block) = result.providers.get_mut(provider)
            {
                block.operations.remove(operation);
            }
        }
    }
    result.defaults.extend(overlay.defaults.clone());
    for operation in OPERATIONS.into_iter().filter(|o| fallback(o).is_none()) {
        result
            .operations
            .entry(operation.into())
            .or_default()
            .extend(overlay.defaults.clone());
    }
    for (operation, values) in overlay.operations {
        result
            .operations
            .entry(operation)
            .or_default()
            .extend(values);
    }
    for (provider, overlay) in overlay.providers {
        let target = result.providers.entry(provider).or_default();
        update(&mut target.defaults, &overlay.defaults);
        for operation in OPERATIONS.into_iter().filter(|o| fallback(o).is_none()) {
            update(
                target.operations.entry(operation.into()).or_default(),
                &overlay.defaults,
            );
        }
        for (operation, values) in overlay.operations {
            update(target.operations.entry(operation).or_default(), &values);
        }
    }
    result
}
fn parameters(values: &mut Parameters, provider: Option<&str>, complete: bool) -> Result<()> {
    for (key, value) in values.iter_mut() {
        match key.as_str() {
            "temperature" | "top_p" | "min_p" | "repeat_penalty" => {
                if key == "min_p" && provider != Some("koboldcpp") {
                    return Err("invalid_profile_parameters");
                }
                let max = match key.as_str() {
                    "temperature" => 2.,
                    "repeat_penalty" => 10.,
                    _ => 1.,
                };
                let n = value
                    .as_f64()
                    .filter(|v| v.is_finite() && (0. ..=max).contains(v))
                    .ok_or("invalid_profile_parameters")?;
                *value = json!(n);
            }
            "top_k" | "max_tokens" | "context_tokens" => {
                if key == "context_tokens" && provider != Some("ollama") {
                    return Err("invalid_profile_parameters");
                }
                let (min, max) = match key.as_str() {
                    "top_k" => (0, 10000),
                    "context_tokens" => (0, 1048576),
                    _ => (if provider == Some("ollama") { 1 } else { 4500 }, 131072),
                };
                if !value.as_u64().is_some_and(|n| (min..=max).contains(&n)) {
                    return Err("invalid_profile_parameters");
                }
            }
            "reasoning" if provider == Some("koboldcpp") => {
                let r = value.as_object_mut().ok_or("invalid_profile_reasoning")?;
                if !r.keys().all(|k| {
                    ["include", "effort", "enable_thinking", "budget_tokens"].contains(&k.as_str())
                }) || (complete
                    && !["include", "effort", "enable_thinking"]
                        .iter()
                        .all(|k| r.contains_key(*k)))
                {
                    return Err("invalid_profile_reasoning");
                }
                for key in ["include", "enable_thinking"] {
                    if r.get(key).is_some_and(|v| !v.is_boolean()) {
                        return Err("invalid_profile_reasoning");
                    }
                }
                if let Some(value) = r.get_mut("effort") {
                    let raw = value.as_str().ok_or("invalid_profile_reasoning")?;
                    let effort = raw.trim().to_lowercase();
                    if (!raw.is_empty() && effort.is_empty())
                        || !["", "low", "medium", "high"].contains(&effort.as_str())
                    {
                        return Err("invalid_profile_reasoning");
                    }
                    *value = json!(effort);
                }
                if r.get("budget_tokens")
                    .is_some_and(|v| v.as_u64().is_none_or(|n| n > 131072))
                {
                    return Err("invalid_profile_reasoning");
                }
            }
            _ => return Err("invalid_profile_parameters"),
        }
    }
    Ok(())
}
fn normalize(value: Value, complete: bool) -> Result<Generation> {
    if complete {
        let raw = value.as_object().ok_or("invalid_profile_generation")?;
        if !["defaults", "operations", "providers"]
            .iter()
            .all(|k| raw.contains_key(*k))
        {
            return Err("incomplete_profile_generation");
        }
        for provider in PROVIDERS {
            let p = raw["providers"][provider]
                .as_object()
                .ok_or("incomplete_profile_generation")?;
            if !p.contains_key("defaults") || !p.contains_key("operations") {
                return Err("incomplete_profile_generation");
            }
        }
    }
    let mut generation: Generation =
        serde_json::from_value(value).map_err(|_| "invalid_profile_generation")?;
    parameters(&mut generation.defaults, None, false)?;
    let required = [
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "max_tokens",
    ];
    if complete
        && !required
            .iter()
            .all(|k| generation.defaults.contains_key(*k))
    {
        return Err("incomplete_profile_generation");
    }
    for (operation, values) in &mut generation.operations {
        if !OPERATIONS.contains(&operation.as_str()) {
            return Err("invalid_profile_operation");
        }
        parameters(values, None, false)?;
    }
    for (provider, block) in &mut generation.providers {
        if !PROVIDERS.contains(&provider.as_str()) {
            return Err("invalid_profile_provider");
        }
        parameters(&mut block.defaults, Some(provider), complete)?;
        if complete
            && !if provider == "ollama" {
                block.defaults.contains_key("context_tokens")
            } else {
                block.defaults.contains_key("min_p") && block.defaults.contains_key("reasoning")
            }
        {
            return Err("incomplete_profile_generation");
        }
        for (operation, values) in &mut block.operations {
            if !OPERATIONS.contains(&operation.as_str()) {
                return Err("invalid_profile_operation");
            }
            parameters(values, Some(provider), false)?;
        }
    }
    Ok(generation)
}
fn validate_effective(generation: &Generation) -> Result<()> {
    for provider in PROVIDERS {
        for operation in OPERATIONS {
            let mut values = resolve(generation, provider, operation)?;
            parameters(&mut values, Some(provider), provider == "koboldcpp")?;
            let mut keys = vec![
                "temperature",
                "top_p",
                "top_k",
                "repeat_penalty",
                "max_tokens",
            ];
            keys.push(if provider == "ollama" {
                "context_tokens"
            } else {
                "min_p"
            });
            if !keys.iter().all(|k| values.contains_key(*k))
                || (provider == "koboldcpp" && !values.contains_key("reasoning"))
            {
                return Err("incomplete_profile_generation");
            }
        }
    }
    Ok(())
}
impl Catalog {
    pub fn load(root: &Path) -> Result<Self> {
        let root = root
            .canonicalize()
            .map_err(|_| "prompt_profiles_unavailable")?;
        let mut directories = std::fs::read_dir(&root)
            .map_err(|_| "prompt_profiles_unavailable")?
            .filter_map(|v| v.ok())
            .filter(|v| v.path().is_dir())
            .collect::<Vec<_>>();
        directories.sort_by_key(|v| v.file_name().to_string_lossy().to_lowercase());
        let mut manifests = vec![];
        let mut warnings = vec![];
        for dir in directories {
            let id = dir.file_name().to_string_lossy().into_owned();
            let parsed = (|| {
                let file = dir
                    .path()
                    .join("profile.json")
                    .canonicalize()
                    .map_err(|_| "invalid_profile")?;
                if !file.starts_with(&root) {
                    return Err("invalid_profile");
                }
                let bytes = std::fs::read(file).map_err(|_| "invalid_profile")?;
                let mut manifest = manifest(&bytes)?;
                manifest.id = manifest.id.trim().into();
                manifest.label = manifest.label.trim().into();
                if !matches!(manifest.schema_version, 1..=3)
                    || manifest.id != id
                    || !valid_id(&id)
                    || manifest.label.is_empty()
                    || manifest.label.chars().count() > 80
                {
                    return Err("invalid_profile");
                }
                let mut overlay = normalize(
                    manifest.generation.clone(),
                    manifest.default && manifest.schema_version != 1,
                )?;
                if manifest.default && manifest.schema_version == 1 {
                    overlay = inherited(&legacy_generation()?, overlay);
                }
                Ok((manifest, overlay))
            })();
            match parsed {
                Ok(value) => manifests.push(value),
                Err(_) => warnings.push(format!("Ignored invalid Prompt Studio profile '{id}'.")),
            }
        }
        let defaults = manifests
            .iter()
            .filter(|(m, _)| m.default)
            .collect::<Vec<_>>();
        if defaults.len() != 1 {
            return Err("prompt_profile_default_ambiguous_or_missing");
        }
        let (default, base) = defaults[0];
        let base = if default.schema_version < 3 {
            legacy_animation(base.clone())?
        } else {
            base.clone()
        };
        validate_effective(&base)?;
        let mut profiles = vec![];
        let mut prompts = BTreeMap::new();
        for (manifest, overlay) in &manifests {
            let mut generation = if manifest.default {
                base.clone()
            } else {
                inherited(&base, overlay.clone())
            };
            if manifest.schema_version < 3 {
                generation = legacy_animation(generation)?;
            }
            let loaded = (|| {
                validate_effective(&generation)?;
                let mut texts = BTreeMap::new();
                let mut files = vec![];
                for (key, filename) in PROMPTS {
                    let own = root.join(&manifest.id).join(filename);
                    let is_inherited =
                        !manifest.default && std::fs::symlink_metadata(&own).is_err();
                    let source_id = if is_inherited {
                        &default.id
                    } else {
                        &manifest.id
                    };
                    let source = root
                        .join(source_id)
                        .join(filename)
                        .canonicalize()
                        .map_err(|_| "invalid_prompt_file")?;
                    if !source.starts_with(&root) || source.extension().is_none_or(|s| s != "txt") {
                        return Err("invalid_prompt_file");
                    }
                    let text = std::fs::read_to_string(source)
                        .map_err(|_| "invalid_prompt_file")?
                        .trim()
                        .to_owned();
                    if text.is_empty()
                        || text.chars().count() > 100000
                        || text.contains("{{EXPANSION_PROFILE_INSTRUCTIONS}}")
                        || (matches!(key, "story_outline" | "story_outline_dialogue")
                            && !text.contains("{{STORY_OUTLINE_JSON_SCHEMA}}"))
                    {
                        return Err("invalid_prompt_file");
                    }
                    files.push(PromptFile {
                        key: key.into(),
                        file: filename.into(),
                        source_profile_id: source_id.clone(),
                        inherited: is_inherited,
                        digest: format!("{:x}", Sha256::digest(text.as_bytes()))[..12].into(),
                    });
                    texts.insert(key.to_owned(), text);
                }
                Ok((texts, files))
            })();
            let (texts, files) = match loaded {
                Ok(value) => value,
                Err(error) if manifest.default => return Err(error),
                Err(_) => {
                    warnings.push(format!("Ignored Prompt Studio profile '{}' because its prompts or settings are invalid.", manifest.id));
                    continue;
                }
            };
            let mut digest = Sha256::new();
            let version = if manifest.schema_version == 1 { 1 } else { 3 };
            let metadata = json!({"schema_version":version,"id":manifest.id,"label":manifest.label,"default":manifest.default,"order":manifest.order,"generation":generation});
            // Python's revision JSON uses sorted keys, compact separators and ensure_ascii.
            let serialized = canonical_json(&metadata);
            let mut ascii = String::new();
            for ch in serialized.chars() {
                if ch.is_ascii() {
                    ascii.push(ch);
                } else {
                    for unit in ch.encode_utf16(&mut [0; 2]) {
                        ascii.push_str(&format!("\\u{unit:04x}"));
                    }
                }
            }
            digest.update(ascii);
            for (key, _) in PROMPTS {
                digest.update(b"\0");
                digest.update(key);
                digest.update(b"\0");
                digest.update(&texts[key]);
            }
            profiles.push(Profile {
                schema_version: version,
                id: manifest.id.clone(),
                label: manifest.label.clone(),
                default: manifest.default,
                order: manifest.order,
                temperature: resolve(&generation, "koboldcpp", "text_enhancement")?["temperature"]
                    .as_f64()
                    .unwrap(),
                generation,
                revision: format!("{:x}", digest.finalize())[..16].into(),
                prompt_files: files,
            });
            prompts.insert(manifest.id.clone(), texts);
        }
        profiles.sort_by_key(|p| (p.order, p.label.to_lowercase(), p.id.clone()));
        Ok(Self {
            profiles,
            default_id: default.id.clone(),
            warnings,
            prompts,
        })
    }
    pub fn snapshot(&self, requested_id: &str) -> Result<Snapshot> {
        let requested_id = requested_id.trim();
        let found = self.profiles.iter().find(|p| p.id == requested_id);
        let fallback = !requested_id.is_empty() && found.is_none();
        let profile = found
            .or_else(|| self.profiles.iter().find(|p| p.id == self.default_id))
            .ok_or("prompt_profiles_unavailable")?
            .clone();
        Ok(Snapshot {
            requested_id: requested_id.into(),
            resolved_id: profile.id.clone(),
            fallback,
            warning: if fallback {
                format!(
                    "The selected Prompt Studio profile is unavailable; using {}.",
                    profile.label
                )
            } else {
                String::new()
            },
            prompts: self
                .prompts
                .get(&profile.id)
                .ok_or("prompt_profiles_unavailable")?
                .clone(),
            profile,
        })
    }
}
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(values) => {
            let sorted = values.iter().collect::<BTreeMap<_, _>>();
            let fields = sorted
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).unwrap(),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        _ => serde_json::to_string(value).unwrap(),
    }
}
fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value.as_bytes()[0].is_ascii_lowercase()
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}
