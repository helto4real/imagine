//! Model-neutral audio intent. Defaults mirror Director; no decoding or inference here.
use crate::{contract::*, project::check};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeSet};
#[path = "audio_references.rs"]
pub mod references;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
pub enum NormalizationMode {
    #[default]
    #[serde(rename = "Integrated LUFS")]
    IntegratedLufs,
    Peak,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioSettings {
    pub use_native_audio: bool,
    pub normalization_mode: NormalizationMode,
    pub target_lufs: f64,
    pub true_peak_limit_db: f64,
    pub default_volume: f64,
    pub default_fade_in_seconds: f64,
    pub default_fade_out_seconds: f64,
}
impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            use_native_audio: false,
            normalization_mode: NormalizationMode::IntegratedLufs,
            target_lufs: -16.,
            true_peak_limit_db: -1.,
            default_volume: 100.,
            default_fade_in_seconds: 0.,
            default_fade_out_seconds: 0.,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct NormalizationOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_mode: Option<NormalizationMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_lufs: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub true_peak_limit_db: Option<f64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioPurpose {
    VoiceTimbre,
    DialogueLyrics,
    #[default]
    AmbienceSfx,
    MusicStyle,
    BeatRhythm,
    Continuity,
    Custom,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioRelationship {
    #[default]
    Reference,
    WeakReference,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioReference {
    pub purpose: AudioPurpose,
    pub target_reference_id: Option<String>,
    pub relationship: AudioRelationship,
    pub description: String,
    pub strength: f64,
}
impl Default for AudioReference {
    fn default() -> Self {
        Self {
            purpose: Default::default(),
            target_reference_id: None,
            relationship: Default::default(),
            description: String::new(),
            strength: 1.,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioUsage {
    pub generation_reference: bool,
    pub final_mix: bool,
}
impl Default for AudioUsage {
    fn default() -> Self {
        Self {
            generation_reference: false,
            final_mix: true,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioClip {
    pub id: AudioClipId,
    pub asset: Option<AssetId>,
    pub start: Time,
    pub duration: Time,
    pub source_in: Time,
    pub source_out: Option<Time>,
    pub volume: f64,
    pub normalization: NormalizationOverrides,
    pub fade_in: f64,
    pub fade_out: f64,
    pub enabled: bool,
    pub locked: bool,
    pub name: String,
    pub lane: u32,
    pub usage: AudioUsage,
    pub reference: AudioReference,
}
impl AudioClip {
    pub fn new(id: AudioClipId, start: Time, duration: Time, settings: &AudioSettings) -> Self {
        Self {
            id,
            asset: None,
            start,
            duration,
            source_in: Time { num: 0, den: 1 },
            source_out: None,
            volume: settings.default_volume,
            normalization: Default::default(),
            fade_in: settings.default_fade_in_seconds,
            fade_out: settings.default_fade_out_seconds,
            enabled: true,
            locked: false,
            name: String::new(),
            lane: 0,
            usage: Default::default(),
            reference: Default::default(),
        }
    }
    pub fn overlaps(&self, start: Time, duration: Time) -> bool {
        !self.start.fits(self.duration, start) && !start.fits(duration, self.start)
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AudioTrack {
    pub id: AudioTrackId,
    pub clips: Vec<AudioClip>,
}
pub fn compare(a: Time, b: Time) -> Ordering {
    (u128::from(a.num) * u128::from(b.den)).cmp(&(u128::from(b.num) * u128::from(a.den)))
}
fn positive_or_zero(value: f64) -> bool {
    value.is_finite() && value >= 0.
}
fn volume(value: f64) -> bool {
    value.is_finite() && (0.0..=400.).contains(&value)
}
pub fn validate(project: &VideoProject) -> Result<(), Issue> {
    let a = &project.audio;
    check(
        a.target_lufs.is_finite()
            && a.true_peak_limit_db.is_finite()
            && volume(a.default_volume)
            && positive_or_zero(a.default_fade_in_seconds)
            && positive_or_zero(a.default_fade_out_seconds),
        "invalid_audio_settings",
        "audio",
    )?;
    let (mut tracks, mut clips) = (BTreeSet::new(), BTreeSet::new());
    for track in &project.audio_tracks {
        check(
            !track.id.0.trim().is_empty() && tracks.insert(&track.id),
            "duplicate_or_invalid_id",
            "audio_tracks.id",
        )?;
        for clip in &track.clips {
            check(
                !clip.id.0.trim().is_empty() && clips.insert(&clip.id),
                "duplicate_or_invalid_id",
                "audio_tracks.clips.id",
            )?;
            check(
                clip.start.valid()
                    && clip.duration.positive()
                    && clip.start.checked_add(clip.duration).is_some()
                    && clip.source_in.valid()
                    && clip.source_out.is_none_or(|out| {
                        out.valid() && compare(out, clip.source_in) != Ordering::Less
                    }),
                "invalid_audio_time",
                "audio_tracks.clips.time",
            )?;
            check(
                volume(clip.volume)
                    && positive_or_zero(clip.fade_in)
                    && positive_or_zero(clip.fade_out)
                    && clip.normalization.target_lufs.is_none_or(f64::is_finite)
                    && clip
                        .normalization
                        .true_peak_limit_db
                        .is_none_or(f64::is_finite),
                "invalid_audio_mix",
                "audio_tracks.clips.mix",
            )?;
            check(
                clip.reference.strength.is_finite()
                    && (0.0..=1.).contains(&clip.reference.strength)
                    && clip
                        .reference
                        .target_reference_id
                        .as_ref()
                        .is_none_or(|id| !id.trim().is_empty()),
                "invalid_audio_reference",
                "audio_tracks.clips.reference",
            )?;
            check(
                clip.asset.as_ref().is_none_or(|id| {
                    project
                        .assets
                        .get(id)
                        .is_some_and(|a| a.kind == MediaKind::Audio)
                }),
                "invalid_audio_asset",
                "audio_tracks.clips.asset",
            )?;
        }
    }
    Ok(())
}
/// Stable first-fit lanes, matching Director autoStackAudioLanes (including locked clips).
pub(crate) fn stack(tracks: &mut Vec<AudioTrack>) {
    for track in tracks.iter_mut() {
        track.clips.sort_by(|a, b| compare(a.start, b.start));
        let mut lanes: Vec<Time> = vec![];
        for clip in &mut track.clips {
            let lane = lanes
                .iter()
                .position(|end| compare(clip.start, *end) != Ordering::Less)
                .unwrap_or(lanes.len());
            clip.lane = lane as u32;
            let end = clip
                .start
                .checked_add(clip.duration)
                .expect("validated audio timing");
            if lane == lanes.len() {
                lanes.push(end);
            } else {
                lanes[lane] = end;
            }
        }
        track
            .clips
            .sort_by(|a, b| a.lane.cmp(&b.lane).then_with(|| compare(a.start, b.start)));
    }
    tracks.retain(|track| !track.clips.is_empty());
}
pub fn generation_clips<'a>(project: &'a VideoProject, scene: &Scene) -> Vec<&'a AudioClip> {
    project
        .audio_tracks
        .iter()
        .flat_map(|t| &t.clips)
        .filter(|c| {
            c.enabled && c.usage.generation_reference && c.overlaps(scene.start, scene.duration)
        })
        .collect()
}
#[cfg(test)]
#[path = "audio_tests.rs"]
mod tests;
