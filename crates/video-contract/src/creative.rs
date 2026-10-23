//! Project-owned cast and Story source bindings. No provider payloads or media bytes.
use crate::contract::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Creative {
    pub references: Vec<VisualReference>,
    pub sections: BTreeMap<SectionId, SectionSource>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scene_prompts: BTreeMap<SceneId, ScenePrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual_references_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_prompt: Option<GlobalPromptOptions>,
}
impl Creative {
    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
            && self.sections.is_empty()
            && self.scene_prompts.is_empty()
            && self.visual_references_enabled.is_none()
            && self.global_prompt.is_none()
    }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ScenePrompt {
    pub title: String,
    pub summary: String,
    pub ambient_sound: String,
    pub non_diegetic_music: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GlobalPromptOptions {
    pub enabled: bool,
    pub position: String,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ReferenceRelationship {
    pub marker: String,
    pub note: String,
    pub target_reference_id: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VisualReference {
    pub id: String,
    pub name: String,
    pub alias: String,
    pub kind: String,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub enabled: bool,
    pub strength: f64,
    pub retention: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationship: Option<ReferenceRelationship>,
    pub images: Vec<ReferenceImage>,
    pub voice: Option<Voice>,
    pub origin: Option<Origin>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReferenceImage {
    pub id: String,
    pub asset: AssetId,
    pub enabled: bool,
    pub preview: bool,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub contribution: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Voice {
    pub language: String,
    pub description: String,
    pub assets: Vec<AssetId>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub reference_settings: BTreeMap<AssetId, VoiceReferenceSettings>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceReferenceSettings {
    pub enabled: bool,
    pub voice_description: String,
    pub contribution: String,
    pub strength: Option<f64>,
    pub relationship: String,
}
impl Default for VoiceReferenceSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            voice_description: String::new(),
            contribution: String::new(),
            strength: None,
            relationship: "reference".into(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Origin {
    pub library_id: String,
    pub entry_id: String,
    pub revision: u64,
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SectionSource {
    pub images: Vec<ImageBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub videos: Vec<Binding>,
    pub dialogue_intent: Option<DialogueIntent>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageBinding {
    pub image_id: String,
    pub asset: AssetId,
    pub role: InputRole,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DialogueIntent {
    pub participants: Vec<String>,
    pub purpose: String,
    #[serde(default = "two_turns")]
    pub max_turns: u32,
}
pub fn two_turns() -> u32 {
    2
}
pub struct VideoPlacement<'a> {
    pub binding: &'a Binding,
    pub section_id: Option<&'a SectionId>,
    pub shot_index: usize,
}
pub fn video_bindings<'a>(project: &'a VideoProject, scene: &'a Scene) -> Vec<VideoPlacement<'a>> {
    let mut sections = scene.sections.iter().collect::<Vec<_>>();
    sections.sort_by(|a, b| crate::audio::compare(a.start, b.start));
    let scene_videos = scene.bindings.iter().filter(|b| {
        b.enabled
            && project
                .assets
                .get(&b.asset)
                .is_some_and(|a| a.kind == MediaKind::Video)
    });
    let mut result = vec![];
    if sections.is_empty() {
        result.extend(scene_videos.map(|binding| VideoPlacement {
            binding,
            section_id: None,
            shot_index: 0,
        }));
    } else {
        for (shot_index, section) in sections.iter().enumerate() {
            if let Some(source) = project.creative.sections.get(&section.id) {
                result.extend(source.videos.iter().filter(|b| b.enabled).map(|binding| {
                    VideoPlacement {
                        binding,
                        section_id: Some(&section.id),
                        shot_index,
                    }
                }));
            }
            if shot_index == 0 {
                result.extend(scene_videos.clone().map(|binding| VideoPlacement {
                    binding,
                    section_id: Some(&section.id),
                    shot_index,
                }));
            }
        }
    }
    result
}

/// Director selects used enabled subjects, then follows attribute-transfer targets.
pub fn selected_references<'a>(
    creative: &'a Creative,
    aliases: &BTreeSet<String>,
) -> Vec<&'a VisualReference> {
    if creative.visual_references_enabled == Some(false) {
        return vec![];
    }
    let enabled = creative
        .references
        .iter()
        .filter(|r| r.enabled)
        .collect::<Vec<_>>();
    let by_id = enabled
        .iter()
        .enumerate()
        .map(|(i, r)| (r.id.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    let mut selected = enabled
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            aliases
                .contains(&r.alias.trim_start_matches('@').to_lowercase())
                .then_some(i)
        })
        .collect::<BTreeSet<_>>();
    let mut pending = selected.iter().copied().collect::<Vec<_>>();
    while let Some(i) = pending.pop() {
        if let Some(relationship) = &enabled[i].relationship
            && relationship.marker == "attribute_transfer"
            && let Some(target) = relationship
                .target_reference_id
                .as_deref()
                .and_then(|id| by_id.get(id))
            && selected.insert(*target)
        {
            pending.push(*target);
        }
    }
    enabled
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| selected.contains(&i).then_some(r))
        .collect()
}

pub fn validate(project: &VideoProject) -> Result<(), Issue> {
    let error = || Issue::new("invalid_creative_reference", "creative");
    if project
        .creative
        .scene_prompts
        .keys()
        .any(|id| project.scene(id).is_err())
    {
        return Err(Issue::new("scene_not_found", "creative.scene_prompts"));
    }
    let mut ids = BTreeSet::new();
    let mut images = BTreeSet::new();
    for reference in &project.creative.references {
        if reference.id.is_empty() || !ids.insert(&reference.id) || !reference.strength.is_finite()
        {
            return Err(error());
        }
        for image in &reference.images {
            if image.id.is_empty()
                || !images.insert(&image.id)
                || project
                    .assets
                    .get(&image.asset)
                    .is_none_or(|a| a.kind != MediaKind::Image)
            {
                return Err(error());
            }
        }
        if reference.voice.iter().flat_map(|v| &v.assets).any(|id| {
            project
                .assets
                .get(id)
                .is_none_or(|a| a.kind != MediaKind::Audio)
        }) {
            return Err(error());
        }
    }
    let sections = project
        .scenes
        .iter()
        .flat_map(|s| &s.sections)
        .map(|s| &s.id)
        .collect::<BTreeSet<_>>();
    for (id, source) in &project.creative.sections {
        if !sections.contains(id)
            || source.videos.iter().any(|b| {
                b.id.0.is_empty()
                    || project
                        .assets
                        .get(&b.asset)
                        .is_none_or(|a| a.kind != MediaKind::Video)
                    || b.video.as_ref().is_some_and(|v| !v.valid_ranges())
            })
            || source.images.iter().any(|b| {
                project
                    .assets
                    .get(&b.asset)
                    .is_none_or(|a| a.kind != MediaKind::Image)
            })
        {
            return Err(error());
        }
        if source.dialogue_intent.as_ref().is_some_and(|d| {
            d.participants.len() != 2
                || d.participants.iter().any(String::is_empty)
                || d.purpose.trim().is_empty()
                || d.max_turns != 2
        }) {
            return Err(Issue::new("invalid_dialogue_intent", "creative.sections"));
        }
    }
    Ok(())
}
