pub mod app;
mod async_runtime;
pub mod client;
mod clipboard;
#[path = "../../shared/component_error.rs"]
pub mod component_error;
#[path = "../../shared/config_files.rs"]
pub mod config_files;
/// The Director prompt catalog folder below the configuration root: every
/// `.yaml`/`.yml` file in it, loaded by Director's rules.
pub fn video_catalog_folder() -> std::path::PathBuf {
    config_files::path("video/prompt-catalog")
}
#[path = "../../shared/enhancement.rs"]
pub mod enhancement_types;
pub mod helto;
mod icons;
#[path = "../../shared/private_storage.rs"]
pub mod private_storage;
pub mod settings;
#[path = "../../shared/settings_store.rs"]
pub mod settings_store;
#[path = "../../shared/storage_layout.rs"]
pub mod storage_layout;
#[cfg(test)]
#[path = "../tests/support/destination.rs"]
pub(crate) mod test_destination;

#[path = "../../shared/model_catalog.rs"]
pub mod model_catalog;

#[path = "../../shared/h3_protocol.rs"]
pub mod h3_protocol;
#[path = "../../shared/short_av.rs"]
pub mod short_av;

#[path = "../../shared/video_defaults.rs"]
pub mod video_defaults;
#[path = "../../shared/video_destination.rs"]
pub mod video_destination;
