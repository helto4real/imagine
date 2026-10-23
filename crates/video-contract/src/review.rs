//! Explicit manual review is separate from undoable authoring and automatic Run policy.
use crate::contract::{SceneId, TakeId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReviewAction {
    Accept,
    Reject,
    Candidate,
    Outdated,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReviewRequest {
    #[schemars(range(min = 1, max = 1))]
    pub schema_version: u32,
    pub expected_project_revision: u64,
    pub scene_id: SceneId,
    pub take_id: TakeId,
    pub action: ReviewAction,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TakeState {
    pub accepted: bool,
    pub outdated: bool,
}
