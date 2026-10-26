//! Atomic manual selection and transitive invalidation of retained generation results.
use super::{bundle::Bundle, contract::*, persistence::*};
pub use imagine_video_contract::review::*;
use std::collections::{BTreeMap, BTreeSet};

pub fn states(document: &ProjectDocument) -> BTreeMap<TakeId, TakeState> {
    let revisions: BTreeMap<_, _> = document
        .project
        .scenes
        .iter()
        .filter_map(|scene| {
            imagine_video_contract::planning::input_revision(&document.project, &scene.id)
                .ok()
                .map(|revision| (&scene.id, revision))
        })
        .collect();
    document
        .takes
        .iter()
        .map(|take| {
            let accepted = document
                .project
                .scene(&take.scene_id)
                .ok()
                .and_then(|scene| scene.accepted_take.as_ref())
                .is_some_and(|selected| selected.id == take.id);
            let outdated = take.status == TakeStatus::Outdated;
            let inputs_current = take
                .provenance
                .as_ref()
                .map(|p| revisions.get(&take.scene_id) == Some(&p.input_revision));
            (
                take.id.clone(),
                TakeState {
                    accepted,
                    outdated,
                    inputs_current,
                },
            )
        })
        .collect()
}

/// Authoring changes preserve Accepted. Only a changed continuity relationship or frozen
/// source selection invalidates its target. Clearing that selection propagates downstream
/// in the project's validated topological order, matching Director boundary reconciliation.
pub fn invalidate_changed(before: &VideoProject, after: &mut ProjectDocument) -> Result<(), Issue> {
    after.project.validate()?;
    for index in 0..after.project.scenes.len() {
        let id = after.project.scenes[index].id.clone();
        let dependency = |project: &VideoProject| {
            project.scene(&id).ok().map(|scene| {
                let source = scene
                    .continuity
                    .as_ref()
                    .and_then(|continuity| project.scene(&continuity.source_scene).ok())
                    .and_then(|source| source.accepted_take.clone());
                (scene.continuity.clone(), source)
            })
        };
        let selected = after.project.scenes[index]
            .accepted_take
            .as_ref()
            .map(|t| t.id.clone());
        let tracked = selected
            .as_ref()
            .and_then(|id| after.takes.iter().find(|t| &t.id == id))
            .and_then(|t| t.provenance.as_ref())
            .and_then(|p| p.boundary.as_ref());
        let tracked_changed = if let Some(receipt) = tracked {
            let current =
                imagine_video_contract::boundary::resolve(&after.project, &id)?.map(|incoming| {
                    imagine_video_contract::boundary::dependency(
                        &incoming,
                        receipt["dependency"]["reference_previous_clip"] == true,
                    )
                });
            receipt["dependency"] != serde_json::json!(current)
        } else {
            false
        };
        let legacy_changed = dependency(before) != dependency(&after.project);
        let has_tracking = tracked.is_some();
        if legacy_changed || tracked_changed {
            if let Some(take) = &after.project.scenes[index].accepted_take {
                after.locked_takes.insert(id.clone(), take.id.clone());
            }
            after.project.scenes[index].accepted_take = None;
            for take in after.takes.iter_mut().filter(|take| take.scene_id == id) {
                if take.status != TakeStatus::Rejected
                    && (!has_tracking || selected.as_ref() == Some(&take.id))
                {
                    take.status = TakeStatus::Outdated;
                }
            }
        }
    }
    Ok(())
}

pub fn apply(
    document: &ProjectDocument,
    request: ReviewRequest,
) -> Result<ProjectDocument, StorageError> {
    if request.schema_version != SCHEMA_VERSION {
        return Err(StorageError::new("unsupported_schema"));
    }
    if request.expected_project_revision != document.project.project_revision {
        return Err(StorageError::new("revision_conflict"));
    }
    let index = document
        .takes
        .iter()
        .position(|take| take.id == request.take_id && take.scene_id == request.scene_id)
        .ok_or_else(|| StorageError::new("take_not_found"))?;
    if !super::captures::attached(document, &request.take_id) {
        return Err(StorageError::new("take_not_attached"));
    }
    if !document.resources.iter().any(|resource| matches!(&resource.kind, ResourceKind::TakeVideo { take_id } if take_id == &request.take_id)) {
        return Err(StorageError::new("take_resource_missing"));
    }
    let mut candidate = document.clone();
    if matches!(
        request.action,
        ReviewAction::Accept | ReviewAction::Outdated
    ) {
        candidate
            .locked_takes
            .insert(request.scene_id.clone(), request.take_id.clone());
    } else if candidate.locked_takes.get(&request.scene_id) == Some(&request.take_id) {
        candidate.locked_takes.remove(&request.scene_id);
    }
    if matches!(request.action, ReviewAction::Accept) {
        for take in candidate
            .takes
            .iter_mut()
            .filter(|take| take.scene_id == request.scene_id && take.status == TakeStatus::Accepted)
        {
            take.status = TakeStatus::Candidate;
        }
    }
    let take = &mut candidate.takes[index];
    take.status = match request.action {
        ReviewAction::Accept => TakeStatus::Accepted,
        ReviewAction::Reject => TakeStatus::Rejected,
        ReviewAction::Candidate => TakeStatus::Candidate,
        ReviewAction::Outdated => TakeStatus::Outdated,
    };
    let accepted = matches!(request.action, ReviewAction::Accept)
        .then(|| take.accepted_summary(&candidate.resources))
        .transpose()?;
    let scene = candidate
        .project
        .scenes
        .iter_mut()
        .find(|scene| scene.id == request.scene_id)
        .ok_or_else(|| StorageError::new("scene_not_found"))?;
    if accepted.is_some()
        || scene
            .accepted_take
            .as_ref()
            .is_some_and(|take| take.id == request.take_id)
    {
        scene.accepted_take = accepted;
    }
    invalidate_changed(&document.project, &mut candidate)?;
    candidate.project.project_revision = candidate
        .project
        .project_revision
        .checked_add(1)
        .filter(|revision| *revision <= MAX_INTEGER)
        .ok_or_else(|| StorageError::new("invalid_revision"))?;
    candidate.validate()?;
    Ok(candidate)
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectRequest {
    pub schema_version: u32,
    pub expected_project_revision: u64,
    /// Also restore the prompt text the Take was made with.
    #[serde(default)]
    pub restore_prompt: bool,
}
pub struct Selected {
    pub document: ProjectDocument,
    pub prompt_restored: bool,
    /// Continuations made current again, in timeline order.
    pub restored: Vec<(SceneId, TakeId)>,
    /// Continuations without a kept Take made from the selected chain.
    pub needs_render: Vec<SceneId>,
    pub final_video_outdated: bool,
}
fn restore(project: &mut VideoProject, scene_id: &SceneId, authored: &AuthoredPrompt) {
    if let Some(scene) = project.scenes.iter_mut().find(|s| &s.id == scene_id) {
        scene.description.clone_from(&authored.description);
        for section in &mut scene.sections {
            if let Some(text) = authored.sections.iter().find(|a| a.id == section.id) {
                section.prompt.clone_from(&text.prompt);
            }
        }
    }
    let texts = [
        &authored.summary,
        &authored.ambient_sound,
        &authored.non_diegetic_music,
    ];
    if !project.creative.scene_prompts.contains_key(scene_id) && texts.iter().all(|t| t.is_empty())
    {
        return;
    }
    let prompt = project
        .creative
        .scene_prompts
        .entry(scene_id.clone())
        .or_default();
    prompt.summary.clone_from(&authored.summary);
    prompt.ambient_sound.clone_from(&authored.ambient_sound);
    prompt
        .non_diegetic_music
        .clone_from(&authored.non_diegetic_music);
}
fn accept(document: &ProjectDocument, take: &StoredTake) -> Result<ProjectDocument, StorageError> {
    apply(
        document,
        ReviewRequest {
            schema_version: SCHEMA_VERSION,
            expected_project_revision: document.project.project_revision,
            scene_id: take.scene_id.clone(),
            take_id: take.id.clone(),
            action: ReviewAction::Accept,
        },
    )
}
/// Make an earlier Take current, then bring back, for every Scene that continues from
/// it, the kept Take made from the now-current chain.
pub fn select(
    document: &ProjectDocument,
    take_id: &TakeId,
    request: SelectRequest,
) -> Result<Selected, StorageError> {
    use imagine_video_contract::planning::input_revision;
    if request.schema_version != SCHEMA_VERSION {
        return Err(StorageError::new("unsupported_schema"));
    }
    if request.expected_project_revision != document.project.project_revision {
        return Err(StorageError::new("revision_conflict"));
    }
    let take = document
        .takes
        .iter()
        .find(|t| &t.id == take_id)
        .ok_or_else(|| StorageError::new("take_not_found"))?;
    let scene = document.project.scene(&take.scene_id)?;
    if scene
        .accepted_take
        .as_ref()
        .is_some_and(|a| &a.id == take_id)
    {
        return Err(StorageError::new("take_already_current"));
    }
    let mut candidate = document.clone();
    let mut prompt_restored = false;
    if let Some(provenance) = &take.provenance
        && input_revision(&candidate.project, &scene.id)? != provenance.input_revision
    {
        let mut restored = candidate.project.clone();
        if let Some(authored) = &provenance.authored {
            restore(&mut restored, &scene.id, authored);
        }
        let prompt_only = provenance.authored.is_some()
            && input_revision(&restored, &scene.id)? == provenance.input_revision;
        match (prompt_only, request.restore_prompt) {
            (false, _) if provenance.authored.is_none() => {
                return Err(StorageError::new("take_prompt_unknown"));
            }
            (false, _) => return Err(StorageError::new("take_inputs_changed")),
            (true, false) => return Err(StorageError::new("take_prompt_changed")),
            (true, true) => {
                candidate.project = restored;
                prompt_restored = true;
            }
        }
    }
    candidate = accept(&candidate, take)?;
    let mut changed = BTreeSet::from([scene.id.clone()]);
    let mut restored = vec![];
    let mut needs_render = vec![];
    let ids: Vec<_> = candidate
        .project
        .scenes
        .iter()
        .map(|s| s.id.clone())
        .collect();
    for id in ids {
        let scene = candidate.project.scene(&id)?;
        if !super::runs::dependency(&candidate.project, scene).is_some_and(|s| changed.contains(&s))
            || scene.accepted_take.is_some()
        {
            continue;
        }
        changed.insert(id.clone());
        let revision = input_revision(&candidate.project, &id)?;
        let kept = candidate
            .takes
            .iter()
            .rev()
            .find(|t| {
                t.scene_id == id
                    && t.status != TakeStatus::Rejected
                    && t.provenance
                        .as_ref()
                        .is_some_and(|p| p.input_revision == revision)
            })
            .cloned();
        let next = kept.as_ref().map(|t| accept(&candidate, t)).transpose()?;
        match (kept, next) {
            (Some(take), Some(next))
                if next
                    .project
                    .scene(&id)?
                    .accepted_take
                    .as_ref()
                    .is_some_and(|a| a.id == take.id) =>
            {
                candidate = next;
                restored.push((id, take.id));
            }
            _ => needs_render.push(id),
        }
    }
    let final_video_outdated = candidate.deliveries.last().is_some_and(|d| {
        d.plan.clips.iter().any(|clip| {
            clip.take_id.is_some()
                && candidate
                    .project
                    .scene(&clip.scene_id)
                    .ok()
                    .and_then(|s| s.accepted_take.as_ref())
                    .map(|a| &a.id)
                    != clip.take_id.as_ref()
        })
    });
    Ok(Selected {
        document: candidate,
        prompt_restored,
        restored,
        needs_render,
        final_video_outdated,
    })
}

#[cfg(test)]
#[path = "review_tests.rs"]
mod tests;
