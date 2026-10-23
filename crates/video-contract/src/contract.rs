pub use crate::audio::{
    AudioClip, AudioPurpose, AudioReference, AudioRelationship, AudioSettings, AudioTrack,
    AudioUsage, NormalizationMode, NormalizationOverrides,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SCHEMA_VERSION: u32 = 1;
pub const MAX_INTEGER: u64 = 9_007_199_254_740_991;

macro_rules! identity {
    ($($name:ident),+) => {$ (
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
        #[serde(transparent)]
        pub struct $name(pub String);
    )+};
}
identity!(
    ProjectId,
    SceneId,
    SectionId,
    AssetId,
    BindingId,
    TakeId,
    AudioClipId,
    AudioTrackId
);

/// Exact, nonnegative timeline time. Validation belongs to the document boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Time {
    #[schemars(range(max = 9_007_199_254_740_991u64))]
    pub num: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991u64))]
    pub den: u64,
}
impl Time {
    pub fn valid(self) -> bool {
        self.num <= MAX_INTEGER && self.den > 0 && self.den <= MAX_INTEGER
    }
    pub fn normalized(self) -> Self {
        if !self.valid() {
            return self;
        }
        let (mut a, mut b) = (self.num, self.den);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        Self {
            num: self.num / a,
            den: self.den / a,
        }
    }
    pub fn positive(self) -> bool {
        self.valid() && self.num > 0
    }
    pub fn checked_add(self, other: Self) -> Option<Self> {
        if !self.valid() || !other.valid() {
            return None;
        }
        let num = u128::from(self.num) * u128::from(other.den)
            + u128::from(other.num) * u128::from(self.den);
        let den = u128::from(self.den) * u128::from(other.den);
        let (mut a, mut b) = (num, den);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        let (num, den) = (num / a, den / a);
        (num <= u128::from(MAX_INTEGER) && den <= u128::from(MAX_INTEGER)).then_some(Self {
            num: num as u64,
            den: den as u64,
        })
    }
    pub fn equivalent(self, other: Self) -> bool {
        if !self.valid() || !other.valid() {
            return false;
        }
        u128::from(self.num) * u128::from(other.den) == u128::from(other.num) * u128::from(self.den)
    }
    pub fn fits(self, duration: Self, limit: Self) -> bool {
        if !self.valid() || !duration.valid() || !limit.valid() {
            return false;
        }
        // Each operand is <= 2^53-1; the sum fits in 107 bits.
        let a = u128::from(self.num) * u128::from(duration.den);
        let b = u128::from(duration.num) * u128::from(self.den);
        let den = u128::from(self.den) * u128::from(duration.den);
        // Compare ratios by Euclidean division: no 159-bit cross multiplication.
        ratio_cmp(a + b, den, u128::from(limit.num), u128::from(limit.den))
            != std::cmp::Ordering::Greater
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelSelection {
    pub adapter: String,
    pub model: String,
    pub variant: String,
    pub profile: String,
    pub profile_revision: String,
}
/// Settings are keyed by the entire model/profile identity, never copied between models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SettingsSlot {
    pub model: ModelSelection,
    pub schema_version: u32,
    pub values: serde_json::Value,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelChoice {
    ProjectDefault,
    Override(ModelSelection),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DeliverySettings {
    pub fps: Time,
    pub width: u32,
    pub height: u32,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub id: SectionId,
    pub start: Time,
    pub duration: Time,
    pub prompt: String,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub library_source: Option<LibraryMediaSource>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    pub id: AssetId,
    pub revision: u64,
    pub kind: MediaKind,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LibraryMediaSource {
    pub library_id: String,
    pub entry_id: String,
    pub resource_id: String,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum InputRole {
    FirstFrame,
    LastFrame,
    VisualReference,
    VideoReference,
    VoiceReference,
    AudioReference,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub id: BindingId,
    pub asset: AssetId,
    pub role: InputRole,
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<VideoReferenceSettings>,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct VideoReferenceSettings {
    pub role: String,
    pub custom_role: String,
    pub relationship: Option<crate::creative::ReferenceRelationship>,
    pub source_in: Option<Time>,
    pub source_out: Option<Time>,
    pub soundtrack_usage: String,
}
impl VideoReferenceSettings {
    pub fn valid_ranges(&self) -> bool {
        self.source_in.is_none_or(Time::valid) && self.source_out.is_none_or(Time::valid)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ContinuityMethod {
    LastFrameFirstFrame,
    Combo,
    Influence,
    MotionContext,
    MotionContextExp,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Continuity {
    pub source_scene: SceneId,
    pub method: ContinuityMethod,
    #[serde(default)]
    pub include_audio: bool,
}
/// A retained source identity; complete result/provenance publication belongs to P03.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AcceptedTake {
    pub id: TakeId,
    /// Imported video may have no known generation model.
    pub model: Option<ModelSelection>,
    pub source_fingerprint: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Scene {
    pub id: SceneId,
    pub start: Time,
    pub duration: Time,
    pub description: String,
    pub model: ModelChoice,
    pub settings: Vec<SettingsSlot>,
    pub sections: Vec<Section>,
    pub bindings: Vec<Binding>,
    pub continuity: Option<Continuity>,
    pub accepted_take: Option<AcceptedTake>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VideoProject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<crate::generation::Defaults>,
    #[schemars(range(min = 1, max = 1))]
    pub schema_version: u32,
    pub id: ProjectId,
    pub project_revision: u64,
    pub default_model: ModelSelection,
    pub delivery: DeliverySettings,
    pub global_prompt: String,
    #[serde(default, skip_serializing_if = "crate::creative::Creative::is_empty")]
    pub creative: crate::creative::Creative,
    pub assets: BTreeMap<AssetId, Asset>,
    pub scenes: Vec<Scene>,
    #[serde(default)]
    pub audio: AudioSettings,
    #[serde(default)]
    pub audio_tracks: Vec<AudioTrack>,
    #[serde(default, skip_serializing_if = "crate::assembly::Intent::is_empty")]
    pub assembly: crate::assembly::Intent,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CreateProject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<crate::generation::Defaults>,
    #[schemars(range(min = 1, max = 1))]
    pub schema_version: u32,
    pub default_model: ModelSelection,
    pub delivery: DeliverySettings,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Issue {
    pub code: String,
    pub field: String,
    pub scene_id: Option<SceneId>,
}
impl Issue {
    pub fn new(code: &str, field: &str) -> Self {
        Self {
            code: code.into(),
            field: field.into(),
            scene_id: None,
        }
    }
    pub fn scene(mut self, id: &SceneId) -> Self {
        self.scene_id = Some(id.clone());
        self
    }
}

fn ratio_cmp(mut a: u128, mut b: u128, mut c: u128, mut d: u128) -> std::cmp::Ordering {
    let mut reversed = false;
    loop {
        let order = (a / b).cmp(&(c / d));
        if order != std::cmp::Ordering::Equal {
            return if reversed { order.reverse() } else { order };
        }
        let (r, s) = (a % b, c % d);
        if r == 0 || s == 0 {
            let order = r.cmp(&s);
            return if reversed { order.reverse() } else { order };
        }
        (a, b, c, d) = (b, r, d, s);
        reversed = !reversed;
    }
}
