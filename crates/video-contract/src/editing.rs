use super::{contract::*, project::check};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EditRequest {
    #[schemars(range(min = 1, max = 1))]
    pub schema_version: u32,
    pub expected_project_revision: u64,
    pub commands: Vec<Edit>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
pub enum Edit {
    SetCreative {
        creative: crate::creative::Creative,
    },
    SetAssembly {
        intent: crate::assembly::Intent,
    },
    SetClipInstance {
        scene_id: SceneId,
        clip: Option<crate::assembly::ClipInstance>,
    },
    SetAssemblyBoundary {
        boundary: crate::assembly::Boundary,
    },
    SetDefaultModel {
        model: ModelSelection,
    },
    SetDelivery {
        delivery: DeliverySettings,
    },
    SetGlobalPrompt {
        prompt: String,
    },
    SetAudioSettings {
        settings: AudioSettings,
    },
    SetAudioTracks {
        tracks: Vec<AudioTrack>,
    },
    AddAudioClip {
        track_id: AudioTrackId,
        clip: Box<AudioClip>,
    },
    UpdateAudioClip {
        clip: Box<AudioClip>,
    },
    RemoveAudioClip {
        clip_id: AudioClipId,
    },
    AddAsset {
        asset: Asset,
    },
    AddScene {
        scene: Box<Scene>,
    },
    RemoveScene {
        scene_id: SceneId,
    },
    MoveScene {
        scene_id: SceneId,
        start: Time,
    },
    SetSceneOrder {
        scene_ids: Vec<SceneId>,
    },
    SetSceneModel {
        scene_id: SceneId,
        model: ModelChoice,
    },
    SetSceneDescription {
        scene_id: SceneId,
        description: String,
    },
    SetSceneDuration {
        scene_id: SceneId,
        duration: Time,
    },
    SetModelSettings {
        scene_id: SceneId,
        settings: SettingsSlot,
    },
    AddSection {
        scene_id: SceneId,
        section: Section,
    },
    UpdateSection {
        scene_id: SceneId,
        section: Section,
    },
    RemoveSection {
        scene_id: SceneId,
        section_id: SectionId,
    },
    SetBindings {
        scene_id: SceneId,
        bindings: Vec<Binding>,
    },
    SetContinuity {
        scene_id: SceneId,
        continuity: Option<Continuity>,
    },
    DuplicateScene {
        scene_id: SceneId,
        new_scene_id: SceneId,
        start: Time,
    },
}
fn scene<'a>(project: &'a mut VideoProject, id: &SceneId) -> Result<&'a mut Scene, Issue> {
    project
        .scenes
        .iter_mut()
        .find(|s| &s.id == id)
        .ok_or_else(|| Issue::new("scene_not_found", "scene_id"))
}
pub fn apply(project: &mut VideoProject, request: EditRequest) -> Result<(), Issue> {
    check(
        request.schema_version == SCHEMA_VERSION,
        "unsupported_schema",
        "schema_version",
    )?;
    check(
        request.expected_project_revision == project.project_revision,
        "revision_conflict",
        "expected_project_revision",
    )?;
    check(!request.commands.is_empty(), "empty_edit", "commands")?;
    // Validate the complete candidate, then swap once. Failed batches have no effect.
    let mut candidate = project.clone();
    for command in request.commands {
        match command {
            Edit::SetCreative { creative } => candidate.creative = creative,
            Edit::SetAssembly { intent } => candidate.assembly = intent,
            Edit::SetClipInstance { scene_id, clip } => {
                scene(&mut candidate, &scene_id)?;
                if let Some(clip) = clip {
                    candidate.assembly.clips.insert(scene_id, clip);
                } else {
                    candidate.assembly.clips.remove(&scene_id);
                }
            }
            Edit::SetAssemblyBoundary { boundary } => {
                candidate
                    .assembly
                    .boundaries
                    .retain(|b| b.left != boundary.left || b.right != boundary.right);
                candidate.assembly.boundaries.push(boundary);
            }
            Edit::SetDefaultModel { model } => candidate.default_model = model,
            Edit::SetDelivery { delivery } => candidate.delivery = delivery,
            Edit::SetGlobalPrompt { prompt } => candidate.global_prompt = prompt,
            Edit::SetAudioSettings { settings } => candidate.audio = settings,
            Edit::SetAudioTracks { tracks } => candidate.audio_tracks = tracks,
            Edit::AddAudioClip { track_id, clip } => {
                if !candidate.audio_tracks.iter().any(|t| t.id == track_id) {
                    candidate.audio_tracks.push(AudioTrack {
                        id: track_id.clone(),
                        clips: vec![],
                    });
                }
                candidate
                    .audio_tracks
                    .iter_mut()
                    .find(|t| t.id == track_id)
                    .unwrap()
                    .clips
                    .push(*clip);
            }
            Edit::UpdateAudioClip { clip } => {
                let old = candidate
                    .audio_tracks
                    .iter_mut()
                    .flat_map(|t| &mut t.clips)
                    .find(|c| c.id == clip.id)
                    .ok_or_else(|| Issue::new("audio_clip_not_found", "clip.id"))?;
                *old = *clip;
            }
            Edit::RemoveAudioClip { clip_id } => {
                check(
                    candidate
                        .audio_tracks
                        .iter()
                        .any(|t| t.clips.iter().any(|c| c.id == clip_id)),
                    "audio_clip_not_found",
                    "clip_id",
                )?;
                for track in &mut candidate.audio_tracks {
                    track.clips.retain(|c| c.id != clip_id);
                }
            }
            Edit::AddAsset { asset } => {
                check(
                    !candidate.assets.contains_key(&asset.id),
                    "asset_exists",
                    "asset.id",
                )?;
                candidate.assets.insert(asset.id.clone(), asset);
            }
            Edit::AddScene { scene } => {
                // Authoring cannot forge an inference result.
                check(
                    scene.accepted_take.is_none(),
                    "unexpected_take",
                    "scene.accepted_take",
                )?;
                candidate.scenes.push(*scene);
            }
            Edit::RemoveScene { scene_id } => {
                candidate.creative.scene_prompts.remove(&scene_id);
                let removed = scene(&mut candidate, &scene_id)?
                    .sections
                    .iter()
                    .map(|s| s.id.clone())
                    .collect::<Vec<_>>();
                candidate
                    .creative
                    .sections
                    .retain(|id, _| !removed.contains(id));
                candidate.assembly.clips.remove(&scene_id);
                candidate
                    .assembly
                    .boundaries
                    .retain(|b| b.left != scene_id && b.right != scene_id);
                candidate.scenes.retain(|s| s.id != scene_id);
            }
            Edit::MoveScene { scene_id, start } => scene(&mut candidate, &scene_id)?.start = start,
            Edit::SetSceneOrder { scene_ids } => {
                let requested: std::collections::BTreeSet<_> = scene_ids.iter().collect();
                check(
                    scene_ids.len() == candidate.scenes.len()
                        && requested.len() == scene_ids.len()
                        && candidate.scenes.iter().all(|s| requested.contains(&s.id)),
                    "invalid_scene_order",
                    "scene_ids",
                )?;
                candidate.scenes.sort_by_key(|s| {
                    scene_ids
                        .iter()
                        .position(|id| id == &s.id)
                        .expect("validated scene set")
                });
            }
            Edit::SetSceneModel { scene_id, model } => {
                scene(&mut candidate, &scene_id)?.model = model
            }
            Edit::SetSceneDescription {
                scene_id,
                description,
            } => scene(&mut candidate, &scene_id)?.description = description,
            Edit::SetSceneDuration { scene_id, duration } => {
                scene(&mut candidate, &scene_id)?.duration = duration
            }
            Edit::SetModelSettings { scene_id, settings } => {
                let slots = &mut scene(&mut candidate, &scene_id)?.settings;
                slots.retain(|slot| slot.model != settings.model);
                slots.push(settings);
            }
            Edit::AddSection { scene_id, section } => {
                scene(&mut candidate, &scene_id)?.sections.push(section)
            }
            Edit::UpdateSection { scene_id, section } => {
                let existing = scene(&mut candidate, &scene_id)?
                    .sections
                    .iter_mut()
                    .find(|s| s.id == section.id)
                    .ok_or_else(|| Issue::new("section_not_found", "section.id"))?;
                *existing = section;
            }
            Edit::RemoveSection {
                scene_id,
                section_id,
            } => {
                let sections = &mut scene(&mut candidate, &scene_id)?.sections;
                check(
                    sections.iter().any(|s| s.id == section_id),
                    "section_not_found",
                    "section_id",
                )?;
                sections.retain(|s| s.id != section_id);
                candidate.creative.sections.remove(&section_id);
            }
            Edit::SetBindings { scene_id, bindings } => {
                scene(&mut candidate, &scene_id)?.bindings = bindings
            }
            Edit::SetContinuity {
                scene_id,
                continuity,
            } => scene(&mut candidate, &scene_id)?.continuity = continuity,
            Edit::DuplicateScene {
                scene_id,
                new_scene_id,
                start,
            } => {
                let mut copy = scene(&mut candidate, &scene_id)?.clone();
                if let Some(source) = candidate.creative.scene_prompts.get(&scene_id).cloned() {
                    candidate
                        .creative
                        .scene_prompts
                        .insert(new_scene_id.clone(), source);
                }
                copy.id = new_scene_id;
                copy.start = start;
                copy.accepted_take = None;
                for section in &mut copy.sections {
                    let old = section.id.clone();
                    section.id = SectionId(format!("section_{}", uuid::Uuid::new_v4().simple()));
                    if let Some(source) = candidate.creative.sections.get(&old).cloned() {
                        candidate
                            .creative
                            .sections
                            .insert(section.id.clone(), source);
                    }
                }
                for binding in &mut copy.bindings {
                    binding.id = BindingId(format!("binding_{}", uuid::Uuid::new_v4().simple()));
                }
                candidate.scenes.push(copy);
            }
        }
    }
    candidate.project_revision = candidate
        .project_revision
        .checked_add(1)
        .filter(|n| *n <= MAX_INTEGER)
        .ok_or_else(|| Issue::new("revision_exhausted", "project_revision"))?;
    candidate.validate()?;
    crate::audio::stack(&mut candidate.audio_tracks);
    *project = candidate;
    Ok(())
}
