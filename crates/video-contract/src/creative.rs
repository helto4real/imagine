//! Project-owned cast and Story source bindings. No provider payloads or media bytes.
use crate::contract::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Creative {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeline_duration: Option<Time>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub h3_loras: Option<crate::loras::Stack>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scene_loras: BTreeMap<SceneId, crate::loras::Override>,
    pub references: Vec<VisualReference>,
    pub sections: BTreeMap<SectionId, SectionSource>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scene_prompts: BTreeMap<SceneId, ScenePrompt>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scene_kinds: BTreeMap<SceneId, SceneKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visual_references_enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_prompt: Option<GlobalPromptOptions>,
    /// The video preset each scene was last set to; see `AppliedPreset`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub scene_presets: BTreeMap<SceneId, AppliedPreset>,
}
/// A preset applied to a scene, and a fingerprint of the scene's mode, own
/// settings slot and LoRA override right after applying it. A different
/// fingerprint now means the scene was changed since.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AppliedPreset {
    pub key: String,
    pub fingerprint: String,
}
/// Whether a scene still has the values its applied preset gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PresetState {
    Applied,
    Modified,
}
/// Object keys sorted at every level, independent of map ordering features.
fn canonical(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let fields = entries
                .into_iter()
                .map(|(k, v)| format!("{}:{}", serde_json::Value::String(k.clone()), canonical(v)))
                .collect::<Vec<_>>();
            format!("{{{}}}", fields.join(","))
        }
        serde_json::Value::Array(values) => {
            format!(
                "[{}]",
                values.iter().map(canonical).collect::<Vec<_>>().join(",")
            )
        }
        other => other.to_string(),
    }
}
/// A scene's mode, own settings slot for that mode and LoRA override.
pub fn preset_fingerprint(
    model: &ModelChoice,
    slot: Option<&serde_json::Value>,
    loras: Option<&crate::loras::Override>,
) -> String {
    use sha2::{Digest, Sha256};
    let value = serde_json::json!({"model":model,"settings":slot,"loras":loras});
    format!("{:x}", Sha256::digest(canonical(&value).as_bytes()))[..16].into()
}
pub fn scene_preset_fingerprint(project: &VideoProject, scene: &Scene) -> String {
    let model = project.effective_model(scene);
    let slot = scene
        .settings
        .iter()
        .find(|s| &s.model == model)
        .map(|s| &s.values);
    preset_fingerprint(
        &scene.model,
        slot,
        project.creative.scene_loras.get(&scene.id),
    )
}
pub fn preset_states(project: &VideoProject) -> BTreeMap<SceneId, PresetState> {
    project
        .creative
        .scene_presets
        .iter()
        .filter_map(|(id, applied)| {
            let scene = project.scenes.iter().find(|s| &s.id == id)?;
            Some((
                id.clone(),
                if scene_preset_fingerprint(project, scene) == applied.fingerprint {
                    PresetState::Applied
                } else {
                    PresetState::Modified
                },
            ))
        })
        .collect()
}
impl Creative {
    pub fn is_empty(&self) -> bool {
        self.timeline_duration.is_none()
            && self.h3_loras.is_none()
            && self.scene_loras.is_empty()
            && self.references.is_empty()
            && self.sections.is_empty()
            && self.scene_prompts.is_empty()
            && self.scene_kinds.is_empty()
            && self.visual_references_enabled.is_none()
            && self.global_prompt.is_none()
            && self.scene_presets.is_empty()
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SceneKind {
    #[default]
    Generated,
    Imported,
    Extended,
    Edited,
    Placeholder,
}
impl SceneKind {
    pub const ALL: [Self; 5] = [
        Self::Generated,
        Self::Imported,
        Self::Extended,
        Self::Edited,
        Self::Placeholder,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Generated => "Generated",
            Self::Imported => "Imported",
            Self::Extended => "Extended",
            Self::Edited => "Edited",
            Self::Placeholder => "Placeholder",
        }
    }
}
impl VideoProject {
    pub fn scene_kind(&self, id: &SceneId) -> SceneKind {
        self.creative
            .scene_kinds
            .get(id)
            .copied()
            .unwrap_or_else(|| {
                if self.assembly.clips.contains_key(id) {
                    SceneKind::Imported
                } else {
                    SceneKind::Generated
                }
            })
    }
    pub fn scene_clip(&self, id: &SceneId) -> Option<&crate::assembly::ClipInstance> {
        (self.scene_kind(id) == SceneKind::Imported)
            .then(|| self.assembly.clips.get(id))
            .flatten()
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media_order: Vec<SectionMediaId>,
    pub dialogue_intent: Option<DialogueIntent>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "id", rename_all = "snake_case")]
pub enum SectionMediaId {
    Image(String),
    Video(BindingId),
}
impl SectionSource {
    /// Older documents retain their deterministic image-then-video order.
    /// Deleted entries are skipped; newly unlisted entries follow that fallback.
    pub fn ordered_media(&self) -> Vec<(SectionMediaId, &AssetId)> {
        let mut remaining = self
            .images
            .iter()
            .map(|image| (SectionMediaId::Image(image.image_id.clone()), &image.asset))
            .chain(
                self.videos
                    .iter()
                    .map(|video| (SectionMediaId::Video(video.id.clone()), &video.asset)),
            )
            .collect::<Vec<_>>();
        let mut ordered = Vec::with_capacity(remaining.len());
        for id in &self.media_order {
            if let Some(index) = remaining.iter().position(|(key, _)| key == id) {
                ordered.push(remaining.remove(index));
            }
        }
        ordered.extend(remaining);
        ordered
    }
    pub fn primary_media_asset(&self) -> Option<&AssetId> {
        self.ordered_media()
            .into_iter()
            .next()
            .map(|(_, asset)| asset)
    }
    pub fn push_image(&mut self, image: ImageBinding) {
        self.media_order = self.ordered_media().into_iter().map(|(id, _)| id).collect();
        self.media_order
            .push(SectionMediaId::Image(image.image_id.clone()));
        self.images.push(image);
    }
    pub fn push_video(&mut self, video: Binding) {
        self.media_order = self.ordered_media().into_iter().map(|(id, _)| id).collect();
        self.media_order
            .push(SectionMediaId::Video(video.id.clone()));
        self.videos.push(video);
    }
}
#[cfg(test)]
mod section_media_tests {
    use super::*;
    fn image() -> ImageBinding {
        ImageBinding {
            anchor: None,
            image_id: "same".into(),
            asset: AssetId("own-image".into()),
            role: InputRole::VisualReference,
        }
    }
    fn video() -> Binding {
        Binding {
            id: BindingId("same".into()),
            asset: AssetId("own-video".into()),
            role: InputRole::VideoReference,
            enabled: true,
            video: None,
        }
    }
    fn assets(source: &SectionSource) -> Vec<&str> {
        source
            .ordered_media()
            .into_iter()
            .map(|(_, asset)| asset.0.as_str())
            .collect()
    }
    #[test]
    fn typed_binding_ids_preserve_both_append_orders() {
        let mut source = SectionSource::default();
        source.push_video(video());
        source.push_image(image());
        assert_eq!(assets(&source), ["own-video", "own-image"]);
        assert_eq!(
            source.media_order,
            [
                SectionMediaId::Video(BindingId("same".into())),
                SectionMediaId::Image("same".into())
            ]
        );
        let mut reverse = SectionSource::default();
        reverse.push_image(image());
        reverse.push_video(video());
        assert_eq!(assets(&reverse), ["own-image", "own-video"]);
    }
    #[test]
    fn removing_first_and_readding_it_keeps_surviving_order() {
        let mut source = SectionSource::default();
        source.push_video(video());
        source.push_image(image());
        source.videos.clear();
        assert_eq!(assets(&source), ["own-image"]);
        source.push_video(video());
        assert_eq!(assets(&source), ["own-image", "own-video"]);
        source.images.clear();
        assert_eq!(assets(&source), ["own-video"]);
    }
    #[test]
    fn legacy_arrays_and_json_roundtrip_keep_deterministic_order() {
        let mut source = SectionSource {
            images: vec![image()],
            videos: vec![video()],
            ..Default::default()
        };
        let legacy = serde_json::to_value(&source).unwrap();
        assert!(legacy.get("media_order").is_none());
        let restored: SectionSource = serde_json::from_value(legacy).unwrap();
        assert_eq!(assets(&restored), ["own-image", "own-video"]);
        source.media_order = vec![
            SectionMediaId::Video(BindingId("same".into())),
            SectionMediaId::Image("same".into()),
        ];
        let restored: SectionSource =
            serde_json::from_slice(&serde_json::to_vec(&source).unwrap()).unwrap();
        assert_eq!(restored, source);
        assert_eq!(assets(&restored), ["own-video", "own-image"]);
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageBinding {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<ImageAnchor>,
    pub image_id: String,
    pub asset: AssetId,
    pub role: InputRole,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ImageAnchor {
    pub role: String,
    pub custom_role: String,
    pub relationship: ReferenceRelationship,
}
impl Default for ImageAnchor {
    fn default() -> Self {
        Self {
            role: "auto".into(),
            custom_role: String::new(),
            relationship: ReferenceRelationship {
                marker: "fully_preserved".into(),
                ..Default::default()
            },
        }
    }
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
/// Shared spelling for the editable alias's prompt token. Do not change stored aliases.
pub fn reference_alias(alias: &str) -> String {
    let mut value = String::new();
    let mut invalid = false;
    for c in alias.trim_matches(|c| matches!(c,'\u{0009}'..='\u{000d}' | ' ' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')).to_lowercase().chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c=='_' || c=='-' { value.push(c); invalid=false; }
        else if !invalid { value.push('_'); invalid=true; }
    }
    let value = value.trim_matches(['_', '-']);
    if value.starts_with(|c: char| c.is_ascii_lowercase()) {
        value.into()
    } else {
        "ref1".into()
    }
}
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
            (aliases.contains(&reference_alias(&r.alias))
                || aliases.contains(&r.alias.trim_start_matches('@').to_lowercase()))
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
    if project
        .creative
        .h3_loras
        .as_ref()
        .is_some_and(|s| !s.valid())
        || project
            .creative
            .scene_loras
            .values()
            .any(|s| !s.main.valid())
    {
        return Err(Issue::new("invalid_lora_strength", "creative.loras"));
    }
    if project
        .creative
        .scene_loras
        .keys()
        .any(|id| project.scene(id).is_err())
    {
        return Err(Issue::new("scene_not_found", "creative.scene_loras"));
    }
    let error = || Issue::new("invalid_creative_reference", "creative");
    if project
        .creative
        .scene_prompts
        .keys()
        .any(|id| project.scene(id).is_err())
    {
        return Err(Issue::new("scene_not_found", "creative.scene_prompts"));
    }
    if project
        .creative
        .scene_kinds
        .keys()
        .any(|id| project.scene(id).is_err())
    {
        return Err(Issue::new("scene_not_found", "creative.scene_kinds"));
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

#[cfg(test)]
mod scene_kind_tests {
    use super::*;
    use crate::editing::{Edit, EditRequest};
    fn apply(p: &mut VideoProject, commands: Vec<Edit>) {
        crate::editing::apply(
            p,
            EditRequest {
                schema_version: 1,
                expected_project_revision: p.project_revision,
                commands,
            },
        )
        .unwrap();
    }
    #[test]
    fn director_kind_change_retains_clip_and_duplicate_remove_roundtrip() {
        let mut p = VideoProject::create(CreateProject {
            schema_version: 1,
            generation: None,
            default_model: crate::generation::h3_model("reference"),
            delivery: DeliverySettings {
                width: 416,
                height: 224,
                fps: Time { num: 30, den: 1 },
            },
        })
        .unwrap();
        let id = SceneId("own".into());
        let scene:Scene=serde_json::from_value(serde_json::json!({"id":"own","start":{"num":0,"den":1},"duration":{"num":2,"den":1},"description":"Own fixture","model":"project_default","settings":[],"sections":[],"bindings":[],"continuity":null,"accepted_take":null})).unwrap();
        let asset = Asset {
            id: AssetId("own-video".into()),
            revision: 1,
            kind: MediaKind::Video,
            name: String::new(),
            library_source: None,
        };
        apply(
            &mut p,
            vec![
                Edit::AddScene {
                    scene: Box::new(scene),
                },
                Edit::AddAsset { asset },
            ],
        );
        assert_eq!(p.scene_kind(&id), SceneKind::Generated);
        let clip = crate::assembly::ClipInstance {
            asset: AssetId("own-video".into()),
            source_in: Time { num: 0, den: 1 },
            source_out: None,
            speed: 1.,
            include_audio: true,
        };
        apply(
            &mut p,
            vec![Edit::SetClipInstance {
                scene_id: id.clone(),
                clip: Some(clip.clone()),
            }],
        );
        assert_eq!(p.scene_kind(&id), SceneKind::Imported);
        for kind in SceneKind::ALL {
            let scene = p.scene(&id).unwrap().clone();
            apply(
                &mut p,
                vec![Edit::SetSceneKind {
                    scene_id: id.clone(),
                    kind,
                }],
            );
            assert_eq!(p.scene(&id).unwrap(), &scene);
            assert_eq!(p.assembly.clips.get(&id), Some(&clip));
            assert_eq!(p.scene_kind(&id), kind);
            assert_eq!(p.scene_clip(&id).is_some(), kind == SceneKind::Imported);
            assert_eq!(
                serde_json::from_slice::<VideoProject>(&serde_json::to_vec(&p).unwrap()).unwrap(),
                p
            );
        }
        let copy = SceneId("copy".into());
        apply(
            &mut p,
            vec![Edit::DuplicateScene {
                scene_id: id.clone(),
                new_scene_id: copy.clone(),
                start: Time { num: 2, den: 1 },
            }],
        );
        assert_eq!(p.scene_kind(&copy), SceneKind::Placeholder);
        assert_eq!(p.assembly.clips.get(&copy), Some(&clip));
        apply(
            &mut p,
            vec![Edit::RemoveScene {
                scene_id: copy.clone(),
            }],
        );
        assert!(!p.creative.scene_kinds.contains_key(&copy));
        assert!(!p.assembly.clips.contains_key(&copy));
        // Old documents with only a clip binding retain their imported meaning.
        p.creative.scene_kinds.clear();
        assert_eq!(p.scene_kind(&id), SceneKind::Imported);
    }
}
