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
    SetDefaultModel {
        model: ModelSelection,
    },
    SetDelivery {
        delivery: DeliverySettings,
    },
    SetGlobalPrompt {
        prompt: String,
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
            Edit::SetDefaultModel { model } => candidate.default_model = model,
            Edit::SetDelivery { delivery } => candidate.delivery = delivery,
            Edit::SetGlobalPrompt { prompt } => candidate.global_prompt = prompt,
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
                scene(&mut candidate, &scene_id)?;
                candidate.scenes.retain(|s| s.id != scene_id);
            }
            Edit::MoveScene { scene_id, start } => scene(&mut candidate, &scene_id)?.start = start,
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
                copy.id = new_scene_id;
                copy.start = start;
                copy.accepted_take = None;
                for section in &mut copy.sections {
                    section.id = SectionId(format!("section_{}", uuid::Uuid::new_v4().simple()));
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
    *project = candidate;
    Ok(())
}
