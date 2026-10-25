//! Optional LoRA discovery and information; never an inference allow-list.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    pub file: String,
    pub size: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Catalog {
    pub files: Vec<Entry>,
    pub warnings: Vec<String>,
    /// The LoRA folder the file names are relative to, so the local UI can
    /// turn a picked file inside it into the same relative name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Info {
    pub file: String,
    pub name: String,
    pub base_model: String,
    pub model_type: String,
    pub format: String,
    pub sha256: String,
    pub trained_words: Vec<String>,
}
