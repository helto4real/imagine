//! Atomic manual selection and transitive invalidation of retained generation results.
use super::{bundle::Bundle, contract::*, persistence::*};
pub use imagine_video_contract::review::*;
use std::collections::BTreeMap;

pub fn states(document: &ProjectDocument) -> BTreeMap<TakeId, TakeState> {
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
            (take.id.clone(), TakeState { accepted, outdated })
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
        if dependency(before) != dependency(&after.project) {
            if let Some(take) = &after.project.scenes[index].accepted_take {
                after.locked_takes.insert(id.clone(), take.id.clone());
            }
            after.project.scenes[index].accepted_take = None;
            for take in after.takes.iter_mut().filter(|take| take.scene_id == id) {
                if take.status != TakeStatus::Rejected {
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
    let accepted = matches!(request.action, ReviewAction::Accept).then(|| AcceptedTake {
        id: take.id.clone(),
        model: take.provenance.model.clone(),
        source_fingerprint: take.provenance.input_revision.clone(),
    });
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

#[cfg(test)]
#[path = "review_tests.rs"]
mod tests;
