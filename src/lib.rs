pub mod api;
pub mod comfy_engine;
#[path = "../shared/component_error.rs"]
pub mod component_error;
pub mod config;
#[path = "../shared/config_files.rs"]
pub mod config_files;
/// Configuration files the server reads at startup; see `config/README.md`.
pub const CONFIG_FILES: &[&str] = &[
    "image/character-sheet/portrait.txt",
    "image/character-sheet/turnaround.txt",
    "image/character-sheet/moss.txt",
    "image/profiles/qwen.yaml",
    "image/profiles/quick-draft.yaml",
    "image/profiles/moss.yaml",
    "video/agent-guidance.txt",
];
pub mod contract;
pub mod engine;
pub mod enhancement;
#[path = "../shared/enhancement.rs"]
pub mod enhancement_types;
pub(crate) mod image_sampling_preview;
#[path = "../shared/private_storage.rs"]
pub mod private_storage;
pub mod progress;
pub mod sampling_preview;
pub mod scheduler;
#[path = "../shared/settings_store.rs"]
pub mod settings_store;
pub mod storage;
#[path = "../shared/storage_layout.rs"]
pub mod storage_layout;
#[cfg(test)]
#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;
#[path = "../shared/workspace_ids.rs"]
pub mod workspace_ids;
pub mod workspaces;

pub mod coordination;

pub mod workflow_image;

pub mod color_match;

#[path = "../shared/privacy.rs"]
pub mod privacy;
pub mod profiles;

pub mod session;

pub mod character_sheet;
pub mod generation;
pub mod model_adapter;

pub mod model_library;

pub mod rust_runtime;

pub mod video;

#[path = "../shared/h3_protocol.rs"]
pub mod h3_protocol;

#[path = "../shared/short_av.rs"]
pub mod short_av;

#[path = "../shared/video_defaults.rs"]
pub mod video_defaults;
#[path = "../shared/video_destination.rs"]
pub mod video_destination;

#[path = "../shared/model_spec.rs"]
pub mod model_spec;
pub mod moss_layout;
