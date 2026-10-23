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
}
fn error(code: &str) -> StorageError {
    StorageError::new(code)
}
fn dependency(project: &VideoProject, scene: &Scene) -> Option<SceneId> {
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
pub fn plan(
    document: &ProjectDocument,
    request: &Request,
    owner: String,
    key: String,
) -> Result<Planned, StorageError> {
    let project = &document.project;
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
    if request.mode == Mode::SelectedWithDependents {
        let selected = request
            .selected_scene
            .as_ref()
            .ok_or_else(|| error("selected_scene_required"))?;
        project.scene(selected)?;
        included.insert(selected.clone());
        for scene in &project.scenes {
            if dependency(project, scene).is_some_and(|source| included.contains(&source)) {
                included.insert(scene.id.clone());
            }
        }
    }
    let mut reserved: BTreeSet<u64> = project
        .scenes
        .iter()
        .filter_map(|s| locked(s).and_then(|t| t.provenance.as_ref().map(|p| p.seed)))
        .collect();
    let mut targets = vec![];
    for scene in &project.scenes {
        if project.scene_kind(&scene.id) == imagine_video_contract::creative::SceneKind::Imported {
            continue;
        }
        if (request.mode == Mode::MissingOnly && accepted(scene).is_some())
            || (request.mode == Mode::SelectedWithDependents && !included.contains(&scene.id))
        {
            continue;
        }
        let previous = locked(scene);
        let locked_seed = previous.and_then(|t| t.provenance.as_ref().map(|p| p.seed));
        if request.mode == Mode::AllSameSeeds && locked_seed.is_none() {
            return Err(error("all_scenes_require_locked_seeds"));
        }
        let use_fresh = request.mode == Mode::AllFreshSeeds || locked_seed.is_none();
        let seed = if use_fresh {
            let mut value = uuid::Uuid::new_v4().as_u128() as u64 & MAX_INTEGER;
            while reserved.contains(&value) {
                value = (value + 1) & MAX_INTEGER;
            }
            reserved.insert(value);
            value
        } else {
            locked_seed.unwrap()
        };
        targets.push(Target {
            scene_id: scene.id.clone(),
            seed,
            seed_source: if use_fresh { "fresh" } else { "locked_take" }.into(),
            seed_take_id: (!use_fresh).then(|| previous.unwrap().id.clone()),
            dependency: dependency(project, scene),
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
