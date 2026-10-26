//! Session-local ordered generation intent. No durable queue or automatic restart.
use crate::contract::{ProjectId, SceneId, SectionId, TakeId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    MissingOnly,
    AllSameSeeds,
    AllFreshSeeds,
    SelectedWithDependents,
    /// Selected Scenes with their own seed and prompt choices, plus every Scene
    /// that continues from them.
    Rerun,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub project_id: ProjectId,
    pub expected_project_revision: u64,
    pub mode: Mode,
    #[serde(default)]
    pub selected_scene: Option<SceneId>,
    /// Only for `Mode::Rerun`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerun: Option<Rerun>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Rerun {
    pub scenes: Vec<RerunScene>,
    /// Seed of the Scenes rendered again because they continue from a selected one.
    #[serde(default = "same")]
    pub dependents_seed: Seed,
}
fn same() -> Seed {
    Seed::Same
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RerunScene {
    pub scene_id: SceneId,
    #[serde(default = "fresh")]
    pub seed: Seed,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptChange>,
}
fn fresh() -> Seed {
    Seed::Fresh
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Seed {
    /// A new random seed.
    Fresh,
    /// The seed of the Scene's current, or last locked, Take.
    Same,
    /// The seed of an earlier Take of the same Scene.
    FromTake(TakeId),
}
/// Prompt text applied in the same revision as the run; timing never changes.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PromptChange {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sections: Vec<SectionPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambient_sound: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub non_diegetic_music: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SectionPrompt {
    pub section_id: SectionId,
    pub prompt: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    Selected,
    ContinuesFrom,
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
    /// Why a rerun renders this Scene.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<Reason>,
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
