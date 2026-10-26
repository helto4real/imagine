//! Frozen seeds and intent, with guarded automatic review. Scheduler owns advancement.
use super::{contract::*, persistence::*, planning};
pub use imagine_video_contract::runs::*;
use std::collections::{BTreeMap, BTreeSet};
pub struct Planned {
    pub run: Run,
    pub owner: String,
    pub request_hash: String,
    pub key: String,
    pub cancel_sent: bool,
    intents: BTreeMap<SceneId, String>,
    selections: BTreeMap<SceneId, Option<AcceptedTake>>,
    /// The prompt edits to publish with the run, applied to the planned document.
    pub edited: Option<ProjectDocument>,
    pub report: serde_json::Value,
}
fn error(code: &str) -> StorageError {
    StorageError::new(code)
}
/// The Scene this one continues from, if it is a continuous shot.
pub(crate) fn dependency(project: &VideoProject, scene: &Scene) -> Option<SceneId> {
    scene
        .continuity
        .as_ref()
        .map(|c| c.source_scene.clone())
        .or_else(|| {
            imagine_video_contract::boundary::resolve(project, &scene.id)
                .ok()
                .flatten()
                .map(|b| b.source_scene)
        })
}
fn authoring(project: &VideoProject) -> VideoProject {
    let mut project = project.clone();
    for scene in &mut project.scenes {
        scene.accepted_take = None;
    }
    project
}
/// The run's prompt edits applied to a copy of the document, or None without any.
fn prompt_edits(
    document: &ProjectDocument,
    rerun: &Rerun,
) -> Result<Option<ProjectDocument>, StorageError> {
    use imagine_video_contract::editing::{Edit, EditRequest};
    let project = &document.project;
    let mut commands = vec![];
    let mut creative = project.creative.clone();
    let mut creative_changed = false;
    for entry in &rerun.scenes {
        let Some(change) = &entry.prompt else {
            continue;
        };
        let scene = project.scene(&entry.scene_id)?;
        if change.sections.is_empty()
            && [
                &change.description,
                &change.summary,
                &change.ambient_sound,
                &change.non_diegetic_music,
            ]
            .iter()
            .all(|field| field.is_none())
        {
            return Err(error("rerun_prompt_empty"));
        }
        for text in &change.sections {
            let mut section = scene
                .sections
                .iter()
                .find(|s| s.id == text.section_id)
                .cloned()
                .ok_or_else(|| error("section_not_found"))?;
            section.prompt = text.prompt.clone();
            commands.push(Edit::UpdateSection {
                scene_id: scene.id.clone(),
                section,
            });
        }
        if let Some(description) = &change.description {
            commands.push(Edit::SetSceneDescription {
                scene_id: scene.id.clone(),
                description: description.clone(),
            });
        }
        let fields = [
            &change.summary,
            &change.ambient_sound,
            &change.non_diegetic_music,
        ];
        if fields.iter().any(|field| field.is_some()) {
            let prompt = creative.scene_prompts.entry(scene.id.clone()).or_default();
            for (target, value) in [
                &mut prompt.summary,
                &mut prompt.ambient_sound,
                &mut prompt.non_diegetic_music,
            ]
            .into_iter()
            .zip(fields)
            {
                if let Some(value) = value {
                    target.clone_from(value);
                }
            }
            creative_changed = true;
        }
    }
    if creative_changed {
        commands.push(Edit::SetCreative { creative });
    }
    if commands.is_empty() {
        return Ok(None);
    }
    document
        .edited(EditRequest {
            schema_version: 1,
            expected_project_revision: project.project_revision,
            commands,
        })
        .map(Some)
}
fn seconds(time: Time) -> f64 {
    time.num as f64 / time.den as f64
}
pub fn plan(
    document: &ProjectDocument,
    request: &Request,
    owner: String,
    key: String,
) -> Result<Planned, StorageError> {
    let original = document;
    let project = &original.project;
    if request.project_id != project.id {
        return Err(error("project_identity_changed"));
    }
    if request.schema_version != 1 {
        return Err(error("unsupported_schema"));
    }
    if project.project_revision != request.expected_project_revision {
        return Err(error("revision_conflict"));
    }
    project.validate()?;
    match (request.mode, &request.rerun) {
        (Mode::Rerun, None) => return Err(error("rerun_required")),
        (Mode::Rerun, Some(_)) | (_, None) => {}
        (_, Some(_)) => return Err(error("rerun_requires_rerun_mode")),
    }
    let edited = match &request.rerun {
        Some(rerun) => prompt_edits(original, rerun)?,
        None => None,
    };
    let document = edited.as_ref().unwrap_or(original);
    let project = &document.project;
    let accepted = |scene: &Scene| {
        scene.accepted_take.as_ref().and_then(|summary| {
            document
                .takes
                .iter()
                .find(|t| t.id == summary.id && t.status == TakeStatus::Accepted)
        })
    };
    let locked = |scene: &Scene| {
        scene
            .accepted_take
            .as_ref()
            .map(|t| &t.id)
            .or_else(|| document.locked_takes.get(&scene.id))
            .and_then(|id| {
                document.takes.iter().find(|t| {
                    &t.id == id && matches!(t.status, TakeStatus::Accepted | TakeStatus::Outdated)
                })
            })
    };
    let mut included = BTreeSet::new();
    let mut selected = BTreeMap::new();
    if request.mode == Mode::SelectedWithDependents {
        let selected = request
            .selected_scene
            .as_ref()
            .ok_or_else(|| error("selected_scene_required"))?;
        project.scene(selected)?;
        included.insert(selected.clone());
    }
    if let Some(rerun) = &request.rerun {
        if rerun.scenes.is_empty() {
            return Err(error("rerun_scenes_required"));
        }
        for entry in &rerun.scenes {
            project.scene(&entry.scene_id)?;
            if project.scene_kind(&entry.scene_id)
                == imagine_video_contract::creative::SceneKind::Imported
            {
                return Err(error("rerun_scene_imported"));
            }
            if selected.insert(entry.scene_id.clone(), entry).is_some() {
                return Err(error("rerun_scene_duplicate"));
            }
            included.insert(entry.scene_id.clone());
        }
    }
    // Scenes are in timeline order, so one pass collects every transitive continuation.
    for scene in &project.scenes {
        if dependency(project, scene).is_some_and(|source| included.contains(&source)) {
            included.insert(scene.id.clone());
        }
    }
    let mut reserved: BTreeSet<u64> = project
        .scenes
        .iter()
        .filter_map(|s| locked(s).and_then(|t| t.provenance.as_ref().map(|p| p.seed)))
        .collect();
    let mut fresh_seed = || {
        let mut value = uuid::Uuid::new_v4().as_u128() as u64 & MAX_INTEGER;
        while reserved.contains(&value) {
            value = (value + 1) & MAX_INTEGER;
        }
        reserved.insert(value);
        value
    };
    let mut targets = vec![];
    for scene in &project.scenes {
        if project.scene_kind(&scene.id) == imagine_video_contract::creative::SceneKind::Imported {
            continue;
        }
        let partial = matches!(request.mode, Mode::SelectedWithDependents | Mode::Rerun);
        if (request.mode == Mode::MissingOnly && accepted(scene).is_some())
            || (partial && !included.contains(&scene.id))
        {
            continue;
        }
        let previous = locked(scene);
        let locked_seed = previous.and_then(|t| t.provenance.as_ref().map(|p| p.seed));
        let source = dependency(project, scene);
        let (seed, seed_source, seed_take_id, reason) = if let Some(rerun) = &request.rerun {
            let entry = selected.get(&scene.id);
            let policy = entry.map_or(&rerun.dependents_seed, |e| &e.seed);
            let (seed, seed_source, seed_take_id) = match (policy, previous, locked_seed) {
                (Seed::Same, Some(take), Some(seed)) => {
                    (seed, "locked_take", Some(take.id.clone()))
                }
                (Seed::Fresh | Seed::Same, ..) => (fresh_seed(), "fresh", None),
                (Seed::FromTake(id), ..) => {
                    let take = document
                        .takes
                        .iter()
                        .find(|t| &t.id == id && t.scene_id == scene.id)
                        .and_then(|t| t.provenance.as_ref().map(|p| (t, p.seed)))
                        .ok_or_else(|| error("rerun_take_not_found"))?;
                    (take.1, "take", Some(take.0.id.clone()))
                }
            };
            // A selected Scene whose source is not rendered again would repeat its current Take.
            if entry.is_some()
                && source.as_ref().is_none_or(|s| !included.contains(s))
                && let Some(current) = accepted(scene).and_then(|t| t.provenance.as_ref())
                && current.seed == seed
                && current.input_revision == planning::input_revision(project, &scene.id)?
            {
                return Err(error("rerun_would_repeat_take"));
            }
            let reason = if entry.is_some() {
                Reason::Selected
            } else {
                Reason::ContinuesFrom
            };
            (seed, seed_source, seed_take_id, Some(reason))
        } else {
            if request.mode == Mode::AllSameSeeds && locked_seed.is_none() {
                return Err(error("all_scenes_require_locked_seeds"));
            }
            match locked_seed {
                Some(seed) if request.mode != Mode::AllFreshSeeds => {
                    (seed, "locked_take", previous.map(|t| t.id.clone()), None)
                }
                _ => (fresh_seed(), "fresh", None, None),
            }
        };
        targets.push(Target {
            scene_id: scene.id.clone(),
            seed,
            seed_source: seed_source.into(),
            seed_take_id,
            dependency: source,
            reason,
            job_id: None,
            completed_take_id: None,
        });
    }
    let clean = authoring(project);
    let intents = targets
        .iter()
        .map(|target| {
            Ok((
                target.scene_id.clone(),
                planning::input_revision(&clean, &target.scene_id)?,
            ))
        })
        .collect::<Result<_, StorageError>>()?;
    let selections = project
        .scenes
        .iter()
        .map(|s| (s.id.clone(), s.accepted_take.clone()))
        .collect();
    let report = report(original, document, &targets, &selected);
    let run = Run {
        schema_version: 1,
        id: format!("run_{}", uuid::Uuid::new_v4().simple()),
        project_id: project.id.clone(),
        mode: request.mode,
        status: if targets.is_empty() {
            Status::Completed
        } else {
            Status::Running
        },
        targets,
        current_job: None,
        failure: None,
    };
    Ok(Planned {
        run,
        owner,
        request_hash: crate::storage::hash(&serde_json::to_vec(request).unwrap()),
        key,
        cancel_sent: false,
        intents,
        selections,
        edited,
        report,
    })
}
/// What a run would render and why, for a plan before any GPU work.
fn report(
    original: &ProjectDocument,
    document: &ProjectDocument,
    targets: &[Target],
    selected: &BTreeMap<SceneId, &RerunScene>,
) -> serde_json::Value {
    let project = &document.project;
    let changed = |id: &SceneId| selected.get(id).is_some_and(|e| e.prompt.is_some());
    let rendered: BTreeSet<_> = targets.iter().map(|t| &t.scene_id).collect();
    let mut total = 0.0;
    let mut warnings = vec![];
    let rows = targets
        .iter()
        .map(|t| {
            let scene = project.scene(&t.scene_id).ok();
            let planned = scene.map_or(0.0, |s| seconds(s.duration));
            total += planned;
            if t.reason == Some(Reason::ContinuesFrom)
                && t.dependency.as_ref().is_some_and(&changed)
            {
                warnings.push(format!("dependent_opening_may_mismatch:{}", t.scene_id.0));
            }
            serde_json::json!({
                "scene_id": t.scene_id,
                "title": project.creative.scene_prompts.get(&t.scene_id).map(|p| &p.title),
                "reason": t.reason,
                "source": t.dependency,
                "seed": t.seed_source,
                "seed_take_id": t.seed_take_id,
                "prompt_changed": changed(&t.scene_id),
                "previous_take": scene.and_then(|s| s.accepted_take.as_ref()).map(|a| &a.id)
                    .or_else(|| original.locked_takes.get(&t.scene_id)),
                "planned_seconds": planned,
            })
        })
        .collect::<Vec<_>>();
    if original
        .deliveries
        .last()
        .is_some_and(|d| d.plan.clips.iter().any(|c| rendered.contains(&c.scene_id)))
    {
        warnings.insert(0, "final_video_outdated".into());
    }
    let unchanged = project
        .scenes
        .iter()
        .filter(|s| !rendered.contains(&s.id))
        .map(|s| &s.id)
        .collect::<Vec<_>>();
    serde_json::json!({
        "valid": true,
        "project_revision": original.project.project_revision,
        "targets": rows,
        "unchanged": unchanged,
        "total_seconds": total,
        "warnings": warnings,
    })
}
impl Planned {
    pub fn guard(&self, document: &ProjectDocument, scene: &SceneId) -> Result<(), StorageError> {
        let project = &document.project;
        if self.intents.get(scene) != Some(&planning::input_revision(&authoring(project), scene)?) {
            return Err(error("run_inputs_changed"));
        }
        let mut current = Some(scene.clone());
        while let Some(id) = current {
            let s = project.scene(&id)?;
            if self.selections.get(&id) != Some(&s.accepted_take) {
                return Err(error("run_take_selection_changed"));
            }
            current = dependency(project, s);
        }
        Ok(())
    }
    pub fn accept(
        &mut self,
        session: &mut ProjectSession,
        index: usize,
        id: TakeId,
    ) -> Result<(), StorageError> {
        let scene = self.run.targets[index].scene_id.clone();
        self.guard(session.document(), &scene)?;
        let take = session
            .document()
            .takes
            .iter()
            .find(|take| take.id == id && take.scene_id == scene)
            .ok_or_else(|| error("run_result_missing"))?;
        if take.status != TakeStatus::Candidate
            || planning::input_revision(&session.document().project, &scene)?
                != take.generated()?.input_revision
        {
            return Err(error("run_result_outdated"));
        }
        let before = session.document().project.clone();
        session.review(super::review::ReviewRequest {
            schema_version: 1,
            expected_project_revision: before.project_revision,
            scene_id: scene,
            take_id: id.clone(),
            action: super::review::ReviewAction::Accept,
        })?;
        for scene in &session.document().project.scenes {
            if self.selections.get(&scene.id)
                == before.scene(&scene.id).ok().map(|s| &s.accepted_take)
            {
                self.selections
                    .insert(scene.id.clone(), scene.accepted_take.clone());
            }
        }
        self.run.targets[index].completed_take_id = Some(id);
        self.run.current_job = None;
        if self.run.completed() == self.run.targets.len() {
            self.run.status = Status::Completed
        } else if self.run.status == Status::Stopping {
            self.run.status = Status::Stopped
        }
        Ok(())
    }
    pub fn fail(&mut self, code: &str) {
        self.run.status = Status::Failed;
        self.run.failure = Some(code.into());
    }
}
