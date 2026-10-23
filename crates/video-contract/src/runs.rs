//! Session-local ordered generation intent. No durable queue or automatic restart.
use crate::contract::{ProjectId, SceneId, TakeId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    MissingOnly,
    AllSameSeeds,
    AllFreshSeeds,
    SelectedWithDependents,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub project_id: ProjectId,
    pub expected_project_revision: u64,
    pub mode: Mode,
    pub selected_scene: Option<SceneId>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Running,
    Stopping,
    Cancelling,
    Completed,
    Stopped,
    Cancelled,
    Failed,
    Interrupted,
}
impl Status {
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Stopped | Self::Cancelled | Self::Failed | Self::Interrupted
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Target {
    pub scene_id: SceneId,
    pub seed: u64,
    pub seed_source: String,
    pub seed_take_id: Option<TakeId>,
    pub dependency: Option<SceneId>,
    pub job_id: Option<String>,
    pub completed_take_id: Option<TakeId>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Run {
    pub schema_version: u32,
    pub id: String,
    pub project_id: ProjectId,
    pub mode: Mode,
    pub status: Status,
    pub targets: Vec<Target>,
    pub current_job: Option<String>,
    pub failure: Option<String>,
}
impl Run {
    pub fn completed(&self) -> usize {
        self.targets
            .iter()
            .filter(|t| t.completed_take_id.is_some())
            .count()
    }
}
