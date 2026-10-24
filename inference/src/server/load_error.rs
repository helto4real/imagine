//! Diagnostics at model-loading boundaries, before user text is evaluated.
use super::{
    Event, Observer,
    component_error::{Component, LoadDiagnostic, Reason, valid_tensor},
};
use crate::native::Error;
use std::path::Path;

pub(super) fn observe_load<T>(
    role: &str,
    path: &Path,
    observer: &mut Observer<'_>,
    operation: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    operation().map_err(|error| report(role, path, observer, error))
}
pub(super) fn report(role: &str, path: &Path, observer: &mut Observer<'_>, error: Error) -> Error {
    let diagnostic = classify(role, path, &error);
    if let Err(transport) = observer(Event::LoadFailure(diagnostic)) {
        return transport;
    }
    error
}
pub fn classify(role: &str, path: &Path, error: &Error) -> LoadDiagnostic {
    let component = match role {
        "text" => Component::TextEncoder,
        "vae" => Component::Vae,
        _ => Component::DiffusionModel,
    };
    let text = error.to_string();
    let lower = text.to_lowercase();
    let mut tensor = None;
    let missing = [
        "Missing checkpoint parameter ",
        "Missing prepared ",
        "Missing ",
    ]
    .iter()
    .find_map(|prefix| text.strip_prefix(prefix))
    .filter(|value| valid_tensor(value));
    let reason = if matches!(
        error,
        Error::ReferenceException {
            class: "OutOfMemoryError",
            ..
        }
    ) || lower.contains("out of memory")
        || lower.contains("cannot allocate memory")
    {
        Reason::OutOfMemory
    } else if matches!(
        path.metadata().map_err(|e| e.kind()),
        Err(std::io::ErrorKind::NotFound)
    ) {
        Reason::FileNotFound
    } else if lower.contains("permission denied") {
        Reason::PermissionDenied
    } else if text == "unsupported_model_architecture" {
        Reason::UnsupportedArchitecture
    } else if matches!(
        text.as_str(),
        "unsupported_model_configuration" | "unsupported_qkv_layout"
    ) {
        Reason::TensorShapeMismatch
    } else if text == "unsupported_weight_format" {
        Reason::UnsupportedQuantization
    } else if text == "invalid_checkpoint" {
        Reason::InvalidCheckpoint
    } else if let Some(key) = missing {
        tensor = Some(key.to_owned());
        Reason::MissingTensor
    } else if lower.contains("constructor shape")
        || lower.contains("buffer shape")
        || lower.contains("size mismatch")
        || lower.contains("shape mismatch")
    {
        Reason::TensorShapeMismatch
    } else if lower.contains("quantization")
        || lower.contains("unsupported dtype")
        || lower.contains("unsupported legacy checkpoint metadata")
    {
        Reason::UnsupportedQuantization
    } else if matches!(
        error,
        Error::ReferenceException {
            class: "KeyError",
            ..
        }
    ) {
        Reason::InvalidCheckpoint
    } else if lower.contains("safetensors") || lower.contains("header") {
        let mut magic = [0; 4];
        let recognized_other_format = std::fs::File::open(path).ok().is_some_and(|mut file| {
            std::io::Read::read_exact(&mut file, &mut magic).is_ok()
                && (magic == *b"GGUF" || magic == *b"PK\x03\x04" || magic[0] == 0x80)
        });
        if recognized_other_format {
            Reason::UnsupportedFormat
        } else {
            Reason::InvalidCheckpoint
        }
    } else {
        Reason::LoadFailed
    };
    LoadDiagnostic {
        component,
        reason,
        tensor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_loader_cause_and_safe_weight_keys_are_emitted() {
        let path = std::env::temp_dir().join(format!(
            "imagine-load-diagnostic-{}.safetensors",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut file, b"synthetic checkpoint fixture").unwrap();
        for (message, reason, tensor) in [
            (
                "Missing checkpoint parameter layers.0.weight",
                Reason::MissingTensor,
                Some("layers.0.weight"),
            ),
            (
                "Alternate constructor shape layers.0.weight: [2] != [3]",
                Reason::TensorShapeMismatch,
                None,
            ),
            (
                "Unsupported legacy checkpoint metadata",
                Reason::UnsupportedQuantization,
                None,
            ),
            (
                "The safetensors header is incomplete. private sentinel",
                Reason::InvalidCheckpoint,
                None,
            ),
            (
                "CUDA out of memory. private sentinel",
                Reason::OutOfMemory,
                None,
            ),
            (
                "permission denied private sentinel",
                Reason::PermissionDenied,
                None,
            ),
            ("private sentinel", Reason::LoadFailed, None),
            (
                "unsupported_model_architecture",
                Reason::UnsupportedArchitecture,
                None,
            ),
            (
                "unsupported_model_configuration",
                Reason::TensorShapeMismatch,
                None,
            ),
            ("unsupported_qkv_layout", Reason::TensorShapeMismatch, None),
            (
                "unsupported_weight_format",
                Reason::UnsupportedQuantization,
                None,
            ),
            ("invalid_checkpoint", Reason::InvalidCheckpoint, None),
        ] {
            let mut emitted = None;
            let mut observer = |event| {
                if let Event::LoadFailure(diagnostic) = event {
                    emitted = Some(diagnostic);
                }
                Ok(())
            };
            let result: Result<(), Error> = observe_load("text", &path, &mut observer, || {
                Err(Error::Native(message.into()))
            });
            assert!(result.is_err());
            let diagnostic = emitted.unwrap();
            assert_eq!(diagnostic.component, Component::TextEncoder);
            assert_eq!(diagnostic.reason, reason);
            assert_eq!(diagnostic.tensor.as_deref(), tensor);
            assert!(
                !serde_json::to_string(&diagnostic)
                    .unwrap()
                    .contains("private sentinel")
            );
        }
        drop(file);
        assert_eq!(
            classify(
                "text",
                &path,
                &Error::ReferenceException {
                    class: "OutOfMemoryError",
                    message: "opaque allocator failure".into()
                }
            )
            .reason,
            Reason::OutOfMemory
        );
        std::fs::write(&path, b"GGUFsynthetic non-safetensors checkpoint").unwrap();
        assert_eq!(
            classify(
                "text",
                &path,
                &Error::Native("The safetensors header is invalid".into())
            )
            .reason,
            Reason::UnsupportedFormat
        );
        std::fs::remove_file(&path).unwrap();
        assert_eq!(
            classify("vae", &path, &Error::Native("missing".into())).reason,
            Reason::FileNotFound
        );
    }
}
