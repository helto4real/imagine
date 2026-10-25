//! Editable source proposals. These types never contain credentials or model-family syntax.
use crate::contract::{
    AssetId, ContinuityMethod, ProjectId, Scene, SceneId, SectionId, VideoProject,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Outline,
    EnhanceScene,
    EnhanceSection,
    /// Every Scene field and Section prompt of one Scene, reviewed side by
    /// side, as Director's Enhance mode.
    EnhanceSceneAll,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneField {
    #[default]
    Description,
    Summary,
    AmbientSound,
    NonDiegeticMusic,
}
impl SceneField {
    pub fn label(self) -> &'static str {
        match self {
            Self::Description => "Description",
            Self::Summary => "Summary",
            Self::AmbientSound => "Ambient sound",
            Self::NonDiegeticMusic => "Audience-only music",
        }
    }
    pub fn source(self, project: &VideoProject, scene: &Scene) -> String {
        if self == Self::Description {
            return scene.description.clone();
        }
        let Some(source) = project.creative.scene_prompts.get(&scene.id) else {
            return String::new();
        };
        match self {
            Self::Description => unreachable!(),
            Self::Summary => source.summary.clone(),
            Self::AmbientSound => source.ambient_sound.clone(),
            Self::NonDiegeticMusic => source.non_diegetic_music.clone(),
        }
    }
}
/// The text one row of a whole-Scene enhancement proposes for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum FieldTarget {
    Scene { field: SceneField },
    Section { section_id: SectionId },
}
/// One row of a whole-Scene enhancement: the text when it was proposed and
/// the proposal, editable, applied only when included.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldProposal {
    pub target: FieldTarget,
    pub original: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub failure: Option<String>,
    #[serde(default = "included")]
    pub include: bool,
}
fn included() -> bool {
    true
}
impl FieldProposal {
    /// The rows for `scene`: its four Scene fields, then each Section prompt.
    pub fn rows(project: &VideoProject, scene: &Scene) -> Vec<Self> {
        [
            SceneField::Description,
            SceneField::Summary,
            SceneField::AmbientSound,
            SceneField::NonDiegeticMusic,
        ]
        .into_iter()
        .map(|field| (FieldTarget::Scene { field }, field.source(project, scene)))
        .chain(scene.sections.iter().map(|s| {
            (
                FieldTarget::Section {
                    section_id: s.id.clone(),
                },
                s.prompt.clone(),
            )
        }))
        .map(|(target, original)| Self {
            target,
            original,
            text: None,
            failure: None,
            include: true,
        })
        .collect()
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_field: Option<SceneField>,
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
    /// Audience-only music in the Story; off clears what the model writes, as
    /// in Director.
    #[serde(default)]
    pub add_music: bool,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair_with: Option<String>,
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
    /// The rows of a whole-Scene enhancement.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<FieldProposal>,
    /// Model and continuity per outline Scene, chosen by an external agent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub production: Option<Production>,
}
/// Choices the prompt profiles do not write, aligned to the outline's Scenes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Production {
    pub scenes: Vec<ProductionScene>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductionModel {
    Reference,
    I2v,
}
impl ProductionModel {
    pub fn variant(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::I2v => "i2v",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductionScene {
    /// A preset key from the MCP preset catalog; it fixes the model variant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preset: Option<String>,
    /// Without a preset, the variant's default preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ProductionModel>,
    /// Only for a Scene whose boundary is a continuous shot.
    #[serde(default)]
    pub continuity: Option<ContinuityChoice>,
    /// A planning note; never compiled into a prompt.
    #[serde(default)]
    pub rationale: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuityChoice {
    /// `auto` (none) resolves by the target variant.
    #[serde(default, with = "auto_method")]
    pub method: Option<ContinuityMethod>,
    #[serde(default)]
    pub include_audio: bool,
}
mod auto_method {
    use super::ContinuityMethod;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    #[derive(Serialize, Deserialize)]
    #[serde(untagged)]
    enum Choice {
        Auto(Auto),
        Method(ContinuityMethod),
    }
    #[derive(Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Auto {
        Auto,
    }
    pub fn serialize<S: Serializer>(
        value: &Option<ContinuityMethod>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            None => Choice::Auto(Auto::Auto),
            Some(method) => Choice::Method(method.clone()),
        }
        .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<ContinuityMethod>, D::Error> {
        Ok(match Choice::deserialize(deserializer)? {
            Choice::Auto(_) => None,
            Choice::Method(method) => Some(method),
        })
    }
}
/// An external agent's complete Story: inventory notes, the outline answer,
/// one expansion answer per Scene and the production block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Submission {
    pub schema_version: u32,
    pub expected_project_revision: u64,
    #[serde(default)]
    pub inventory: Vec<InventoryNote>,
    pub outline: String,
    pub expansions: Vec<String>,
    #[serde(default)]
    pub production: Option<Production>,
    pub dry_run: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Review {
    pub schema_version: u32,
    pub outline: Option<Storyboard>,
    pub scenes: Option<Vec<Option<StoryScene>>>,
    pub text: Option<String>,
    #[serde(default)]
    pub fields: Option<Vec<FieldProposal>>,
}
/// Director ordinary-UI allocator: 250ms minimum when possible; weighted remainder and cap redistribution.
pub fn durations(weights: &[f64], duration: f64, cap: f64) -> Result<Vec<f64>, &'static str> {
    if weights.is_empty()
        || !duration.is_finite()
        || duration <= 0.
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.)
    {
        return Err("invalid_story_duration");
    }
    if duration > weights.len() as f64 * cap + 1e-9 {
        return Err("story_scene_duration_capacity");
    }
    let minimum = if duration >= weights.len() as f64 * 0.25 {
        0.25
    } else {
        duration / weights.len() as f64
    };
    let mut out = vec![minimum; weights.len()];
    let mut active = (0..weights.len()).collect::<Vec<_>>();
    let mut remaining = (duration - minimum * weights.len() as f64).max(0.);
    while remaining > 1e-12 && !active.is_empty() {
        let sum = active.iter().map(|&i| weights[i].max(0.001)).sum::<f64>();
        let saturated = active
            .iter()
            .copied()
            .filter(|&i| out[i] + remaining * weights[i].max(0.001) / sum > cap + 1e-12)
            .collect::<Vec<_>>();
        if saturated.is_empty() {
            for &i in &active {
                out[i] += remaining * weights[i].max(0.001) / sum;
            }
            remaining = 0.;
        } else {
            for i in saturated {
                remaining -= (cap - out[i]).max(0.);
                out[i] = cap;
                active.retain(|&j| j != i);
            }
            remaining = remaining.max(0.);
        }
    }
    if remaining > 1e-9 {
        return Err("story_scene_duration_capacity");
    }
    let last = out.len() - 1;
    out[last] = duration - out[..last].iter().sum::<f64>();
    Ok(out)
}
