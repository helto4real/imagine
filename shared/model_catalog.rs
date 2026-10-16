//! Public model catalog wire types shared by the native UI and server.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};
pub const QWEN_GROUP: &str = "qwen-image-2.1";
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelFile {
    pub id: String,
    pub path: String,
    pub kind: String,
    pub group: Option<String>,
    pub model: Option<String>,
    pub supported: bool,
    #[serde(skip)]
    pub(crate) absolute: PathBuf,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Components {
    pub text_encoder: String,
    pub vae: String,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LibrarySettings {
    pub root: PathBuf,
    pub components: BTreeMap<String, Components>,
    // Retained for reading older settings; LoRA lists are owned by UI model groups.
    pub lora_groups: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LibrarySnapshot {
    pub settings: LibrarySettings,
    pub files: Vec<ModelFile>,
    pub models: Vec<Value>,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub diffusion: String,
    #[serde(default)]
    pub loras: Vec<SelectedLora>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SelectedLora {
    pub id: String,
    pub strength: f32,
}
