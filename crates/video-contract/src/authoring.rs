//! Editable source proposals. These types never contain credentials or model-family syntax.
use crate::contract::{AssetId, ProjectId, SceneId, SectionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Outline,
    EnhanceScene,
    EnhanceSection,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Destination {
    #[default]
    Append,
    Replace,
    Continue,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub expected_project_revision: u64,
    pub provider_profile_id: String,
    pub provider: String,
    pub prompt_profile_id: String,
    pub operation: Operation,
    #[serde(default)]
    pub scene_id: Option<SceneId>,
    #[serde(default)]
    pub section_id: Option<SectionId>,
    #[serde(default)]
    pub brief: String,
    #[serde(default)]
    pub instruction: String,
    pub duration_seconds: f64,
    #[serde(default)]
    pub destination: Destination,
    #[serde(default)]
    pub cast: Vec<String>,
    #[serde(default)]
    pub images: Vec<StoryImage>,
    #[serde(default)]
    pub add_dialogue: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageUse {
    #[default]
    SettingReference,
    RequiredMoment,
    StartFrame,
    EndFrame,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoryImage {
    pub id: String,
    pub asset: AssetId,
    pub label: String,
    pub usage: ImageUse,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryNote {
    pub id: String,
    pub note: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorySection {
    pub id: String,
    pub prompt: String,
    pub weight: f64,
    #[serde(default)]
    pub image_ids: Vec<String>,
    #[serde(default)]
    pub first_frame_id: String,
    #[serde(default)]
    pub last_frame_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialogue_intent: Option<crate::creative::DialogueIntent>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoryScene {
    pub id: String,
    pub title: String,
    pub summary: String,
    pub description: String,
    pub ambient_sound: String,
    pub non_diegetic_music: String,
    pub weight: f64,
    pub boundary_from_previous: String,
    pub sections: Vec<StorySection>,
    #[serde(default)]
    pub image_ids: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Storyboard {
    pub scenes: Vec<StoryScene>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Inventory,
    Running,
    Outline,
    Expanding,
    Review,
    Failed,
    Cancelled,
    Applied,
}
impl Phase {
    pub fn busy(self) -> bool {
        matches!(self, Self::Inventory | Self::Running | Self::Expanding)
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    pub schema_version: u32,
    pub id: String,
    pub project_id: ProjectId,
    pub request: Request,
    pub phase: Phase,
    pub outline: Option<Storyboard>,
    pub scenes: Vec<Option<StoryScene>>,
    pub text: Option<String>,
    pub failure: Option<String>,
    pub failed_index: Option<usize>,
    pub prompt_profile_id: String,
    pub prompt_profile_revision: String,
    pub profile_warning: String,
    pub provider_snapshot: serde_json::Value,
    #[serde(default)]
    pub planned_durations: Vec<f64>,
    #[serde(default)]
    pub inventory: Vec<InventoryNote>,
    #[serde(default)]
    pub selected_cast: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub schema_version: u32,
    pub outline: Option<Storyboard>,
    pub scenes: Option<Vec<Option<StoryScene>>>,
    pub text: Option<String>,
}
