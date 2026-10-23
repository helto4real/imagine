//! Per-request component selections and AV model-sampling parameters.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Components {
    pub model: Option<String>,
    pub text_encoder: Option<String>,
    pub video_vae: Option<String>,
    pub audio_vae: Option<String>,
}
impl Components {
    /// Absolute selections stay absolute; relative selections use the configured model root.
    /// No filename, directory, extension or model-identity allowlist is applied here.
    pub fn path(root: &Path, selected: Option<&str>, fallback: &str) -> PathBuf {
        root.join(
            selected
                .filter(|p| !p.trim().is_empty())
                .unwrap_or(fallback),
        )
    }
}
pub fn video_shift() -> f64 {
    12.
}
pub fn audio_shift() -> f64 {
    3.
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<Lora>,
    pub components: Components,
    pub shift_video: f64,
    pub shift_audio: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            loras: vec![],
            components: Components::default(),
            shift_video: video_shift(),
            shift_audio: audio_shift(),
        }
    }
}
impl Options {
    pub fn stage_two(&self) -> Self {
        let mut options = self.clone();
        options.loras.clear();
        options
    }
    pub fn valid(&self) -> bool {
        self.loras
            .iter()
            .all(|l| !l.name.is_empty() && l.strength.is_finite())
            && self.shift_video.is_finite()
            && self.shift_video > 0.
            && self.shift_audio.is_finite()
            && self.shift_audio > 0.
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lora {
    pub name: String,
    pub strength: f64,
}
