//! Server-owned model discovery. Reads weight headers only, never media or sidecars.
use crate::{
    contract::{self, Lora, WorkflowOptions},
    scheduler::Fault,
    storage::hash,
};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

#[path = "../shared/model_catalog.rs"]
mod types;
pub use types::*;
const DIFFUSION: &str = "qwen_image_2.1_int8_convrot.safetensors";
const TURBO_DIFFUSION: &str = "qwen_image_2.1_turbo_bf16.safetensors";
const ENCODER: &str = "qwen3vl_8b_int8_convrot.safetensors";
const VAE: &str = "qwen_image_2.1_vae_bf16.safetensors";

#[derive(Default)]
pub struct Library {
    settings: LibrarySettings,
    files: Vec<ModelFile>,
    scanned: BTreeMap<String, FileStamp>,
    warnings: Vec<String>,
    settings_path: PathBuf,
}
// Private execution snapshot, persisted only in the existing in-memory job store.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResolvedModels {
    pub root: PathBuf,
    pub files: Vec<FileStamp>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FileStamp {
    path: PathBuf,
    device: u64,
    inode: u64,
    size: u64,
    modified: (i64, i64),
}
impl FileStamp {
    fn read(path: &Path) -> Result<Self> {
        let m = std::fs::metadata(path)?;
        if !m.is_file() {
            bail!("Not a model file");
        }
        Ok(Self {
            path: path.into(),
            device: m.dev(),
            inode: m.ino(),
            size: m.len(),
            modified: (m.mtime(), m.mtime_nsec()),
        })
    }
}
impl ResolvedModels {
    pub fn verify(&self) -> Result<()> {
        for expected in &self.files {
            let actual = FileStamp::read(&expected.path)?;
            if actual.device != expected.device
                || actual.inode != expected.inode
                || actual.size != expected.size
                || actual.modified != expected.modified
            {
                bail!("Model file changed after submission");
            }
        }
        Ok(())
    }
}
impl Library {
    pub fn load(default_root: PathBuf, settings_path: PathBuf) -> Result<Self> {
        let settings = if settings_path.exists() {
            serde_json::from_slice(&std::fs::read(&settings_path)?)
                .context("Invalid model-library settings")?
        } else {
            LibrarySettings {
                root: default_root,
                ..Default::default()
            }
        };
        let mut library = Self {
            settings,
            settings_path,
            ..Default::default()
        };
        if !library.settings.root.is_absolute() {
            bail!("The model folder must be an absolute path");
        }
        if library.scan().is_err() {
            library.warnings.push(
                "Could not scan the model folder. Check permissions and refresh in Settings."
                    .into(),
            );
        }
        Ok(library)
    }
    pub fn snapshot(&self) -> LibrarySnapshot {
        let models = self
            .files
            .iter()
            .filter(|f| f.kind == "diffusion_models" && f.model.is_some())
            .map(|file| {
                let model = file.model.as_deref().unwrap();
                let mut meta = model_metadata(model);
                let components = self.components(&file.id);
                let problem = if file.supported {
                    components.as_ref().err().copied()
                } else {
                    Some("unsupported_model_file")
                };
                meta["id"] = json!(file.id);
                meta["model"] = json!(model);
                meta["group"] = json!(file.group);
                meta["name"] = json!(if model == contract::MOSS {
                    "MOSS Image · SFT INT8"
                } else if model == contract::TURBO {
                    "Qwen Image 2.1 · Turbo"
                } else {
                    "Qwen Image 2.1 · INT8"
                });
                meta["path"] = json!(file.path);
                meta["available"] = json!(problem.is_none());
                meta["problem"] = json!(problem);
                meta["components"] = json!(components.ok());
                meta
            })
            .collect();
        LibrarySnapshot {
            settings: self.settings.clone(),
            files: self.files.clone(),
            models,
            warnings: self.warnings.clone(),
        }
    }
    pub fn update(&mut self, settings: LibrarySettings) -> Result<()> {
        if !settings.root.is_absolute() || !settings.root.is_dir() {
            bail!("Choose an existing absolute model folder");
        }
        let mut next = Self {
            settings,
            settings_path: self.settings_path.clone(),
            ..Default::default()
        };
        next.scan()?;
        let parent = next.settings_path.parent().context("Settings directory")?;
        std::fs::create_dir_all(parent)?;
        let temp = parent.join(format!("model-library-{}.tmp", uuid::Uuid::new_v4()));
        let saved = (|| -> Result<()> {
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&temp)?;
            f.write_all(&serde_json::to_vec_pretty(&next.settings)?)?;
            f.sync_all()?;
            std::fs::rename(&temp, &next.settings_path)?;
            Ok(())
        })();
        if saved.is_err() {
            let _ = std::fs::remove_file(temp);
        }
        saved?;
        *self = next;
        Ok(())
    }
    pub fn scan(&mut self) -> Result<()> {
        let mut files = Vec::new();
        let mut warnings = Vec::new();
        if !self.settings.root.is_dir() {
            warnings.push("Model folder is missing. Choose a folder in Settings.".into());
        } else {
            for (kind, folders) in [
                ("diffusion_models", &["diffusion_models", "unet"][..]),
                ("text_encoders", &["text_encoders", "clip"][..]),
                ("vae", &["vae"][..]),
                ("loras", &["loras"][..]),
                ("checkpoints", &["checkpoints"][..]),
            ] {
                let mut visited = BTreeSet::new();
                for folder in folders {
                    let path = self.settings.root.join(folder);
                    if path.exists() {
                        scan_dir(
                            &self.settings.root,
                            &path,
                            kind,
                            &mut visited,
                            &mut files,
                            &mut warnings,
                            0,
                        )?;
                    }
                }
            }
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let scanned = files
            .iter()
            .map(|file| {
                Ok((
                    file.id.clone(),
                    FileStamp::read(&self.settings.root.join(&file.path))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        self.files = files;
        self.scanned = scanned;
        self.warnings = warnings;
        Ok(())
    }
    fn file(&self, id: &str, kind: &str) -> std::result::Result<&ModelFile, &'static str> {
        self.files
            .iter()
            .find(|f| f.id == id && f.kind == kind && f.supported)
            .ok_or("model_file_missing")
    }
    fn components(&self, diffusion: &str) -> std::result::Result<Components, &'static str> {
        let group = self
            .file(diffusion, "diffusion_models")?
            .group
            .as_deref()
            .ok_or("incompatible_component")?;
        if let Some(choice) = self.settings.components.get(diffusion) {
            for (id, kind) in [
                (&choice.text_encoder, "text_encoders"),
                (&choice.vae, "vae"),
            ] {
                if self.file(id, kind)?.group.as_deref() != Some(group) {
                    return Err("incompatible_component");
                }
            }
            return Ok(choice.clone());
        }
        let one = |kind: &str| {
            let found: Vec<_> = self
                .files
                .iter()
                .filter(|f| f.kind == kind && f.supported && f.group.as_deref() == Some(group))
                .collect();
            match found.as_slice() {
                [file] => Ok(file.id.clone()),
                [] => Err("model_component_missing"),
                _ => Err("choose_model_components"),
            }
        };
        Ok(Components {
            text_encoder: one("text_encoders")?,
            vae: one("vae")?,
        })
    }
    /// Explicit legacy profiles keep their component choices, but share the configured root.
    pub fn pin_explicit(
        &self,
        request: &contract::JobRequest,
    ) -> std::result::Result<Option<ResolvedModels>, Fault> {
        if self.settings.root.as_os_str().is_empty() {
            return Ok(None);
        }
        let paths = if let Some(w) = request.model_options.generation(&request.model) {
            let mut paths = vec![
                w.diffusion_model.clone(),
                w.text_encoder.clone(),
                w.vae.clone(),
            ];
            paths.extend(w.loras.iter().map(|l| format!("loras/{}", l.path)));
            paths
        } else {
            let profile: Value = serde_yaml::from_str(include_str!("../profiles/qwen.yaml"))
                .expect("shipped profile");
            let w = &profile["defaults"]["model_options"]["qwen_image_2_1"]["generation"];
            ["diffusion_model", "text_encoder", "vae"]
                .iter()
                .map(|key| w[key].as_str().expect("shipped component").to_owned())
                .collect()
        };
        let files = paths
            .iter()
            .map(|path| FileStamp::read(&self.settings.root.join(path)))
            .collect::<Result<Vec<_>>>()
            .map_err(|_| Fault::new(422, "model_file_missing"))?;
        Ok(Some(ResolvedModels {
            root: self.settings.root.clone(),
            files,
        }))
    }
    pub fn prepare(
        &self,
        mut value: Value,
    ) -> std::result::Result<(Value, Option<ResolvedModels>), Fault> {
        let object = value
            .as_object_mut()
            .ok_or(Fault::new(422, "invalid_request"))?;
        if object.contains_key("resolved_models") {
            return Err(Fault::new(422, "read_only_model_metadata"));
        }
        let Some(selection) = object.remove("model_selection") else {
            return Ok((value, None));
        };
        if object.contains_key("profile") {
            return Err(Fault::new(422, "model_selection_profile_conflict"));
        }
        let selection: Selection = serde_json::from_value(selection)
            .map_err(|_| Fault::new(422, "invalid_model_selection"))?;
        let invalid = |code| Fault::new(422, code);
        let diffusion = self
            .file(&selection.diffusion, "diffusion_models")
            .map_err(invalid)?;
        let model = diffusion
            .model
            .as_deref()
            .ok_or(invalid("unsupported_model_file"))?;
        if value["model"] != model {
            return Err(invalid("model_settings_mismatch"));
        }
        if !value["model_options"]["qwen_image_2_1"]["generation"].is_null()
            || !value["model_options"]["qwen_image_2_1"]["workflow"].is_null()
            || !value["model_options"]["moss_image"].is_null()
        {
            return Err(invalid("model_selection_settings_conflict"));
        }
        let components = self.components(&selection.diffusion).map_err(invalid)?;
        let encoder = self
            .file(&components.text_encoder, "text_encoders")
            .map_err(invalid)?;
        let vae = self.file(&components.vae, "vae").map_err(invalid)?;
        let mut files = vec![diffusion, encoder, vae];
        let mut loras = Vec::new();
        let mut seen = BTreeSet::new();
        if selection.loras.len() > 64 {
            return Err(invalid("too_many_loras"));
        }
        for lora in selection.loras {
            if !lora.strength.is_finite() || !seen.insert(lora.id.clone()) {
                return Err(invalid("invalid_lora_selection"));
            }
            let file = self.file(&lora.id, "loras").map_err(invalid)?;
            // Paths passed to native are relative to its explicit loras directory.
            let path = file
                .path
                .strip_prefix("loras/")
                .ok_or(invalid("invalid_model_path"))?;
            loras.push(Lora {
                path: path.into(),
                strength: lora.strength,
            });
            files.push(file);
        }
        let workflow = WorkflowOptions {
            diffusion_model: diffusion.path.clone(),
            text_encoder: encoder.path.clone(),
            vae: vae.path.clone(),
            loras,
            zero_negative_conditioning: model == contract::TURBO,
            color_match_strength: 0.,
            brightness_strength: 0.,
            contrast_strength: 0.,
        };
        let options = value
            .as_object_mut()
            .unwrap()
            .entry("model_options")
            .or_insert_with(|| json!({}));
        let options = options.as_object_mut().ok_or(invalid("invalid_request"))?;
        if model == contract::MOSS {
            options.insert("moss_image".into(), json!(workflow));
        } else {
            let qwen = options.entry("qwen_image_2_1").or_insert_with(|| json!({}));
            qwen.as_object_mut()
                .ok_or(invalid("invalid_request"))?
                .insert("generation".into(), json!(workflow));
        }
        let stamps = files
            .iter()
            .map(|f| {
                let current = FileStamp::read(&self.settings.root.join(&f.path))
                    .map_err(|_| invalid("model_file_missing"))?;
                if self.scanned.get(&f.id) != Some(&current) {
                    return Err(invalid("model_catalog_stale"));
                }
                Ok(current)
            })
            .collect::<std::result::Result<Vec<_>, Fault>>()?;
        Ok((
            value,
            Some(ResolvedModels {
                root: self.settings.root.clone(),
                files: stamps,
            }),
        ))
    }
}

pub fn model_metadata(model: &str) -> Value {
    if model == contract::MOSS {
        let mut v = contract::model(model).unwrap();
        v["defaults"]["model_options"]
            .as_object_mut()
            .unwrap()
            .remove("moss_image");
        return v;
    }
    let mut meta = contract::model(contract::MODEL).unwrap();
    meta["id"] = json!(model);
    meta["constraints"]["loras"] = json!(true);
    // Library selection resolves a workflow, whose validation accepts these choices.
    meta["constraints"]["samplers"] = json!(["er_sde", "euler", "res_multistep"]);
    meta["constraints"]["schedulers"] = json!(["beta", "simple"]);
    meta["request_schema"]["properties"]["model_selection"] = json!({
        "type":"object", "additionalProperties":false,"required":["diffusion"],
        "properties":{"diffusion":{"type":"string"},"loras":{"type":"array","maxItems":64,
        "items":{"type":"object","additionalProperties":false,"required":["id","strength"],
        "properties":{"id":{"type":"string"},"strength":{"type":"number"}}}}}
    });
    if model == contract::TURBO {
        meta["operations"] = json!(["text_to_image", "image_edit"]);
        meta.as_object_mut().unwrap().remove("character_sheet");
        // Existing shipped quick-draft defaults, not new inference settings.
        let profile: Value = serde_yaml::from_str(include_str!("../profiles/quick-draft.yaml"))
            .expect("shipped profile");
        meta["defaults"]["sampling"] = profile["defaults"]["sampling"].clone();
    }
    meta
}
fn scan_dir(
    root: &Path,
    dir: &Path,
    kind: &str,
    visited: &mut BTreeSet<PathBuf>,
    files: &mut Vec<ModelFile>,
    warnings: &mut Vec<String>,
    depth: usize,
) -> Result<()> {
    if depth > 32 || files.len() > 20_000 {
        bail!("Model folder scan limit exceeded");
    }
    let canonical = dir.canonicalize()?;
    if !visited.insert(canonical) {
        return Ok(());
    }
    let mut entries = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    for path in entries {
        if files.len() >= 20_000 {
            bail!("Model folder scan limit exceeded");
        }
        if path.is_dir() {
            scan_dir(root, &path, kind, visited, files, warnings, depth + 1)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !(["safetensors", "gguf", "ckpt", "pt", "pth", "bin", "sft"]
            .contains(&extension.as_str())
            || kind == "loras" && extension == "safetensor")
        {
            continue;
        }
        let Some(relative) = path.strip_prefix(root)?.to_str() else {
            continue;
        };
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let model = match (kind, name) {
            ("diffusion_models", DIFFUSION) => Some(contract::MODEL.into()),
            ("diffusion_models", TURBO_DIFFUSION) => Some(contract::TURBO.into()),
            ("diffusion_models", "moss_image_sft_int8_convrot.safetensors") => {
                Some(contract::MOSS.into())
            }
            _ => None,
        };
        let moss = model.as_deref() == Some(contract::MOSS)
            || kind == "text_encoders" && name == "qwen3.5_9b_int8_convrot.safetensors"
            || kind == "vae" && name == "flux2-vae.safetensors";
        let known = moss
            || model.is_some()
            || (kind == "text_encoders" && name == ENCODER)
            || (kind == "vae" && name == VAE);
        let mut group = known.then(|| if moss { MOSS_GROUP } else { QWEN_GROUP }.to_owned());
        let mut supported = known || kind == "loras";
        if known {
            supported = if moss {
                moss_header_matches(&path, kind)
            } else {
                model_header_matches(&path, kind)
            }
            .unwrap_or(false);
            if !supported {
                warnings.push(format!(
                    "Model header does not match the supported component: {relative}"
                ));
                group = None;
            }
        }
        let absolute = path.canonicalize()?;
        if files
            .iter()
            .any(|f| f.kind == kind && f.absolute == absolute)
        {
            continue;
        }
        files.push(ModelFile {
            id: hash(relative.as_bytes()),
            path: relative.into(),
            kind: kind.into(),
            group,
            model,
            supported,
            absolute,
        });
    }
    Ok(())
}
fn read_header(path: &Path) -> Result<Value> {
    let mut f = std::fs::File::open(path)?;
    let mut size = [0u8; 8];
    f.read_exact(&mut size)?;
    let size = u64::from_le_bytes(size);
    if size > 8 * 1024 * 1024 {
        bail!("Header exceeds the 8 MiB scan limit");
    }
    if size == 0 || size + 8 > f.metadata()?.len() {
        bail!("Invalid safetensors header");
    }
    let mut bytes = vec![0; size as usize];
    f.read_exact(&mut bytes)?;
    let header: Value = serde_json::from_slice(&bytes)?;
    let object = header.as_object().context("Invalid safetensors object")?;
    let data_size = f.metadata()?.len() - 8 - size;
    for (key, tensor) in object.iter().filter(|(k, _)| k.as_str() != "__metadata__") {
        let offsets = tensor["data_offsets"]
            .as_array()
            .context("Missing tensor offsets")?;
        if offsets.len() != 2
            || offsets[0].as_u64().is_none()
            || offsets[1].as_u64().is_none()
            || offsets[0].as_u64() > offsets[1].as_u64()
            || offsets[1].as_u64().unwrap() > data_size
            || !tensor["shape"].is_array()
            || !tensor["dtype"].is_string()
        {
            bail!("Invalid tensor header: {key}");
        }
    }
    Ok(header)
}
fn model_header_matches(path: &Path, kind: &str) -> Result<bool> {
    let header = read_header(path)?;
    let keys: Vec<_> = header.as_object().unwrap().keys().collect();
    let has = |suffix: &str| keys.iter().any(|k| k.ends_with(suffix));
    // Match native architecture indicators as well as the supported weight filenames.
    Ok(match kind {
        "diffusion_models" => has("txt_in.text_norm.weight") && has("img_in.weight"),
        "text_encoders" => has("embed_tokens.weight") && has("layers.0.self_attn.q_norm.weight"),
        "vae" => {
            has("encoder.conv1.weight")
                && header
                    .as_object()
                    .unwrap()
                    .iter()
                    .any(|(k, v)| k.ends_with("decoder.conv1.weight") && v["shape"][1] == 64)
        }
        _ => false,
    })
}
fn moss_header_matches(path: &Path, kind: &str) -> Result<bool> {
    let h = read_header(path)?;
    let has = |suffix: &str| h.as_object().unwrap().keys().any(|k| k.ends_with(suffix));
    Ok(match kind {
        "diffusion_models" => has("video_embedder.weight") && has("layers.41.proj.qkv.weight"),
        "text_encoders" => {
            has("layers.0.linear_attn.in_proj_qkv.weight")
                && has("layers.3.self_attn.q_proj.weight")
        }
        "vae" => has("bn.running_mean") && has("decoder.conv_in.weight"),
        _ => false,
    })
}

#[cfg(test)]
#[path = "model_library_tests.rs"]
mod tests;
