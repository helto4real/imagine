//! Explicit project Capture attachment and removal, separate from authoring Undo.
use crate::contract::{AssetId, SceneId, TakeId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub expected_project_revision: u64,
    pub scene_id: SceneId,
    pub take_id: TakeId,
    pub action: Action,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Attach,
    AttachAndAccept,
    Detach,
    Delete,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Capture {
    #[serde(default)]
    pub detached: bool,
    #[serde(default)]
    pub removed: bool,
    /// Original generated receipt when attached to another Scene. Its recipe is retained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_capture: Option<TakeId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from: Option<Source>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub asset_id: AssetId,
    pub revision: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Import {
    pub schema_version: u32,
    pub expected_project_revision: u64,
    pub scene_id: SceneId,
    pub asset_id: AssetId,
    #[serde(default)]
    pub accept: bool,
}
