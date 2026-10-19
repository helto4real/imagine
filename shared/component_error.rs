//! Bounded model-loading diagnostics. Never carry raw exceptions or user inputs.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    TextEncoder,
    Vae,
    DiffusionModel,
}
impl Component {
    pub fn field(self) -> &'static str {
        match self {
            Self::TextEncoder => "text_encoder",
            Self::Vae => "vae",
            Self::DiffusionModel => "diffusion_model",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::TextEncoder => "Text encoder",
            Self::Vae => "VAE",
            Self::DiffusionModel => "Diffusion model",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    FileNotFound,
    PermissionDenied,
    UnsupportedFormat,
    InvalidCheckpoint,
    UnsupportedArchitecture,
    UnsupportedQuantization,
    MissingTensor,
    TensorShapeMismatch,
    OutOfMemory,
    LoadFailed,
}
impl Reason {
    pub fn message(self) -> &'static str {
        match self {
            Self::FileNotFound => {
                "The file was not found. Refresh Models and select an existing file."
            }
            Self::PermissionDenied => "The backend cannot read the file. Check its permissions.",
            Self::UnsupportedFormat => "The backend's loader does not support this file format.",
            Self::InvalidCheckpoint => "The checkpoint is invalid, incomplete or unreadable.",
            Self::UnsupportedArchitecture => {
                "The backend cannot recognize this component's model architecture."
            }
            Self::UnsupportedQuantization => {
                "The backend does not support this checkpoint's quantization or weight type."
            }
            Self::MissingTensor => {
                "Required model weights are missing. This file may use a different architecture."
            }
            Self::TensorShapeMismatch => {
                "Weight dimensions do not match the selected model's architecture."
            }
            Self::OutOfMemory => "There is not enough RAM or VRAM to load this component.",
            Self::LoadFailed => "The component loader failed without a recognized cause.",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadDiagnostic {
    pub component: Component,
    pub reason: Reason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor: Option<String>,
}
impl LoadDiagnostic {
    pub fn valid(&self) -> bool {
        self.tensor.as_deref().is_none_or(valid_tensor)
    }
}
pub fn valid_tensor(tensor: &str) -> bool {
    !tensor.is_empty()
        && tensor.len() <= 200
        && tensor
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'.'))
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComponentLoadError {
    #[serde(flatten)]
    pub diagnostic: LoadDiagnostic,
    pub file: String,
}
impl ComponentLoadError {
    pub fn valid(&self) -> bool {
        self.diagnostic.valid()
            && !self.file.is_empty()
            && self.file.len() <= 4096
            && !self.file.chars().any(char::is_control)
            && std::path::Path::new(&self.file)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_)))
    }
}
impl std::fmt::Display for ComponentLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} could not be loaded: {}. {}",
            self.diagnostic.component.label(),
            self.file,
            self.diagnostic.reason.message()
        )?;
        if let Some(tensor) = &self.diagnostic.tensor {
            write!(f, " Tensor: {tensor}.")?;
        }
        Ok(())
    }
}
impl std::error::Error for ComponentLoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_errors_round_trip_and_only_expose_relative_paths_and_tensor_names() {
        let details = ComponentLoadError {
            diagnostic: LoadDiagnostic {
                component: Component::TextEncoder,
                reason: Reason::MissingTensor,
                tensor: Some("layers.0.weight".into()),
            },
            file: "text_encoders/nested/selected.gguf".into(),
        };
        let json = serde_json::to_value(&details).unwrap();
        assert_eq!(json["component"], "text_encoder");
        assert_eq!(
            serde_json::from_value::<ComponentLoadError>(json).unwrap(),
            details
        );
        assert!(details.valid());
        for file in ["/private/absolute", "../outside", "encoder\nprivate", ""] {
            assert!(
                !ComponentLoadError {
                    file: file.into(),
                    ..details.clone()
                }
                .valid()
            );
        }
        assert!(!valid_tensor("layers.0.weight\nprivate"));
        assert!(!valid_tensor("/private/path"));
    }
}
