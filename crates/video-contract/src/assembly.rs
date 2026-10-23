//! Model-neutral assembly intent and immutable delivery plans; no media paths.
use crate::{contract::*, project::check};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ClipInstance {
    pub asset: AssetId,
    pub source_in: Time,
    pub source_out: Option<Time>,
    pub speed: f64,
    #[serde(default = "yes")]
    pub include_audio: bool,
}
fn yes() -> bool {
    true
}
fn is_yes(value: &bool) -> bool {
    *value
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AudioContinuity {
    #[default]
    NewSceneAudio,
    ReferencePreviousAudio,
}
impl AudioContinuity {
    fn is_default(&self) -> bool {
        *self == Self::NewSceneAudio
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryMode {
    #[default]
    HardCut,
    ContinuousShot,
    BlendSeam,
    Transition,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Boundary {
    pub left: SceneId,
    pub right: SceneId,
    pub mode: BoundaryMode,
    pub blend_frames: u32,
    /// None is the Director default: three seconds at the project's current fps.
    /// Some(0) is an authored zero, not a missing value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail_frames: Option<u64>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub transition_prompt: String,
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub reuse_visual_references: bool,
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub reuse_style: bool,
    #[serde(default, skip_serializing_if = "AudioContinuity::is_default")]
    pub audio_continuity: AudioContinuity,
}
impl Boundary {
    pub fn new(left: SceneId, right: SceneId) -> Self {
        Self {
            left,
            right,
            mode: BoundaryMode::HardCut,
            blend_frames: 3,
            tail_frames: None,
            transition_prompt: String::new(),
            reuse_visual_references: true,
            reuse_style: true,
            audio_continuity: AudioContinuity::NewSceneAudio,
        }
    }
    pub fn requested_tail_frames(&self, fps: Time) -> Result<u64, Issue> {
        match self.tail_frames {
            Some(frames) => Ok(frames),
            None => {
                check(fps.positive(), "invalid_delivery", "delivery.fps")?;
                // Director Math.round uses positive half-up, not ties-to-even.
                Ok((seconds(fps) * 3.).round() as u64)
            }
        }
    }
}
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Intent {
    #[serde(default)]
    pub clips: BTreeMap<SceneId, ClipInstance>,
    #[serde(default)]
    pub boundaries: Vec<Boundary>,
}
impl Intent {
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty() && self.boundaries.is_empty()
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MixMode {
    Generated,
    Timeline,
    #[default]
    Combined,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Options {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_filename: Option<String>,
    pub audio_mode: MixMode,
    pub generated_gain: f64,
    pub timeline_gain: f64,
    pub duck_generated: bool,
    pub ducking_strength: f64,
    pub normalize_mix: bool,
    pub continuous_tone_match: bool,
    pub bit_depth: u8,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            output_filename: None,
            audio_mode: MixMode::Combined,
            generated_gain: 1.,
            timeline_gain: 1.,
            duck_generated: false,
            ducking_strength: 0.5,
            normalize_mix: true,
            continuous_tone_match: true,
            bit_depth: 8,
        }
    }
}
/// A basename within the selected output destination, never a different path.
pub fn output_filename(value: Option<&str>) -> Result<Option<String>, Issue> {
    let Some(value) = value.filter(|v| !v.trim().is_empty()) else {
        return Ok(None);
    };
    let mut components = std::path::Path::new(value).components();
    if value.contains('\0')
        || value.contains('/')
        || !matches!(components.next(), Some(std::path::Component::Normal(_)))
        || components.next().is_some()
    {
        return Err(Issue::new(
            "invalid_output_filename",
            "options.output_filename",
        ));
    }
    Ok(Some(if value.to_ascii_lowercase().ends_with(".mp4") {
        value.to_owned()
    } else {
        format!("{value}.mp4")
    }))
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub project_id: ProjectId,
    pub expected_project_revision: u64,
    #[serde(default)]
    pub options: Options,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Source {
    pub resource_id: String,
    pub sha256: String,
    pub bytes: u64,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub duration: Time,
    pub fps: Option<Time>,
    pub frames: Option<u64>,
    pub audio: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClipPlan {
    pub scene_id: SceneId,
    pub take_id: Option<TakeId>,
    pub asset_id: Option<AssetId>,
    pub source: Source,
    pub source_in: Time,
    pub source_out: Option<Time>,
    pub timeline_start: Time,
    pub timeline_duration: Time,
    pub frames: u64,
    pub output_start: u64,
    pub overlap: u64,
    pub boundary: BoundaryMode,
    pub include_audio: bool,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AudioPlan {
    pub clip_id: AudioClipId,
    pub source: Source,
    pub start_sample: u64,
    pub samples: u64,
    pub source_in: Time,
    pub source_out: Time,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_start_sample: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_end_sample: Option<u64>,
    pub volume: f64,
    pub fade_in: f64,
    pub fade_out: f64,
    pub normalization: crate::audio::NormalizationOverrides,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    pub schema_version: u32,
    pub id: String,
    pub project_id: ProjectId,
    pub project_revision: u64,
    pub fingerprint: String,
    pub delivery: DeliverySettings,
    pub sample_rate: u32,
    pub frames: u64,
    pub samples: u64,
    pub options: Options,
    pub clips: Vec<ClipPlan>,
    pub audio: Vec<AudioPlan>,
    pub warnings: Vec<String>,
}
pub fn validate(project: &VideoProject) -> Result<(), Issue> {
    for (id, clip) in &project.assembly.clips {
        project.scene(id)?;
        check(
            project
                .assets
                .get(&clip.asset)
                .is_some_and(|a| a.kind == MediaKind::Video),
            "imported_video_required",
            "assembly.clips",
        )?;
        check(
            clip.source_in.valid()
                && clip
                    .source_out
                    .is_none_or(|t| t.valid() && crate::audio::compare(t, clip.source_in).is_gt())
                && clip.speed.is_finite()
                && clip.speed > 0.,
            "invalid_clip_range",
            "assembly.clips",
        )?;
    }
    let mut pairs = std::collections::BTreeSet::new();
    for boundary in &project.assembly.boundaries {
        project.scene(&boundary.left)?;
        project.scene(&boundary.right)?;
        check(
            boundary.left != boundary.right && pairs.insert((&boundary.left, &boundary.right)),
            "invalid_assembly_boundary",
            "assembly.boundaries",
        )?;
    }
    Ok(())
}
pub fn seconds(time: Time) -> f64 {
    time.num as f64 / time.den as f64
}
pub fn time(seconds: f64) -> Time {
    Time {
        num: (seconds.max(0.) * 1_000_000.).round_ties_even() as u64,
        den: 1_000_000,
    }
    .normalized()
}
pub fn round_samples(seconds: f64, rate: f64) -> u64 {
    (seconds * rate).round_ties_even().max(0.) as u64
}
pub fn fit_index(index: u64, source: u64, target: u64) -> u64 {
    if source <= 1 || target <= 1 {
        return 0;
    }
    // torch.linspace's FP32 CPU path anchors each half at its nearest endpoint.
    // Computing this in f64 changes tie decisions for otherwise ordinary clip lengths.
    let value = linspace_f32(index, 0., (source - 1) as f64, target);
    (value.round_ties_even() as u64).min(source - 1)
}
pub fn linspace_f32(index: u64, start: f64, end: f64, count: u64) -> f32 {
    let start = start as f32;
    let end = end as f32;
    if count <= 1 {
        return start;
    }
    let step = (end - start) / (count - 1) as f32;
    if index < count / 2 {
        (index as f32).mul_add(step, start)
    } else {
        (-((count - index - 1) as f32)).mul_add(step, end)
    }
}
