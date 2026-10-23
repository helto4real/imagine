//! Capture catalog mutations never unlink an input or a caller-selected original.
use super::{bundle::Bundle, contract::*, persistence::*, review};
pub use imagine_video_contract::captures::*;
use std::collections::BTreeMap;

pub fn attached(document: &ProjectDocument, id: &TakeId) -> bool {
    document
        .captures
        .get(id)
        .is_none_or(|c| !c.detached && !c.removed)
}

pub struct Change {
    pub document: ProjectDocument,
    pub attached_id: Option<TakeId>,
    pub retained_for_provenance: bool,
    pub cleanup_warning: Option<String>,
}

pub fn import(document: &ProjectDocument, request: Import) -> Result<Change, StorageError> {
    if request.schema_version != 1 {
        return Err(StorageError::new("unsupported_schema"));
    }
    if request.expected_project_revision != document.project.project_revision {
        return Err(StorageError::new("revision_conflict"));
    }
    document.project.scene(&request.scene_id)?;
    let asset = document
        .project
        .assets
        .get(&request.asset_id)
        .ok_or_else(|| StorageError::new("asset_not_found"))?;
    if asset.kind != MediaKind::Video {
        return Err(StorageError::new("take_requires_video"));
    }
    let source = Source {
        asset_id: asset.id.clone(),
        revision: asset.revision,
    };
    let existing = document.takes.iter().find(|t| {
        t.scene_id == request.scene_id
            && document
                .captures
                .get(&t.id)
                .is_some_and(|c| !c.removed && c.imported_from.as_ref() == Some(&source))
    });
    if let Some(take) = existing {
        return apply(
            document,
            Request {
                schema_version: 1,
                expected_project_revision: request.expected_project_revision,
                scene_id: request.scene_id,
                take_id: take.id.clone(),
                action: if request.accept {
                    Action::AttachAndAccept
                } else {
                    Action::Attach
                },
            },
        );
    }
    let reference = document.resources.iter().find(|r| matches!(&r.kind,ResourceKind::Reference {asset_id,revision} if asset_id == &asset.id && *revision == asset.revision))
        .ok_or_else(|| StorageError::new("reference_resource_missing"))?;
    let mut candidate = document.clone();
    let id = TakeId(format!("take_{}", uuid::Uuid::new_v4().simple()));
    candidate.takes.push(StoredTake {
        id: id.clone(),
        scene_id: request.scene_id.clone(),
        status: TakeStatus::Candidate,
        provenance: None,
    });
    candidate.resources.push(Resource {
        id: format!("take_video_{}", uuid::Uuid::new_v4().simple()),
        kind: ResourceKind::TakeVideo {
            take_id: id.clone(),
        },
        ..reference.clone()
    });
    candidate.captures.insert(
        id.clone(),
        Capture {
            imported_from: Some(source),
            ..Default::default()
        },
    );
    if request.accept {
        candidate = review::apply(
            &candidate,
            review::ReviewRequest {
                schema_version: 1,
                expected_project_revision: candidate.project.project_revision,
                scene_id: request.scene_id,
                take_id: id.clone(),
                action: review::ReviewAction::Accept,
            },
        )?;
    }
    candidate.project.project_revision = document
        .project
        .project_revision
        .checked_add(1)
        .filter(|n| *n <= MAX_INTEGER)
        .ok_or_else(|| StorageError::new("invalid_revision"))?;
    candidate.validate()?;
    Ok(Change {
        document: candidate,
        attached_id: Some(id),
        retained_for_provenance: false,
        cleanup_warning: None,
    })
}

pub fn apply(document: &ProjectDocument, request: Request) -> Result<Change, StorageError> {
    if request.schema_version != 1 {
        return Err(StorageError::new("unsupported_schema"));
    }
    if request.expected_project_revision != document.project.project_revision {
        return Err(StorageError::new("revision_conflict"));
    }
    let take = document
        .takes
        .iter()
        .find(|take| take.id == request.take_id)
        .ok_or_else(|| StorageError::new("take_not_found"))?
        .clone();
    if request.action != Action::Delete {
        document.project.scene(&request.scene_id)?;
    }
    let mut candidate = document.clone();
    let mut attached_id = None;
    let mut retained_for_provenance = false;
    match request.action {
        Action::Attach | Action::AttachAndAccept => {
            let existing = candidate
                .takes
                .iter()
                .find(|t| {
                    t.scene_id == request.scene_id
                        && (t.id == take.id
                            || candidate.captures.get(&t.id).is_some_and(|c| {
                                !c.removed && c.source_capture.as_ref() == Some(&take.id)
                            }))
                })
                .map(|t| t.id.clone());
            let id = if let Some(id) = existing {
                let state = candidate.captures.entry(id.clone()).or_default();
                state.detached = false;
                state.removed = false;
                id
            } else {
                // A generated receipt remains immutable when reused in another Scene.
                // Only its attachment identity changes, never the source files or recipe.
                let id = TakeId(format!("take_{}", uuid::Uuid::new_v4().simple()));
                let mut copy = take.clone();
                copy.id = id.clone();
                copy.scene_id = request.scene_id.clone();
                copy.status = TakeStatus::Candidate;
                candidate.takes.push(copy);
                let resources: Vec<_> = document
                    .resources
                    .iter()
                    .filter_map(|resource| {
                        let kind = match &resource.kind {
                            ResourceKind::TakeVideo { take_id } if take_id == &take.id => {
                                ResourceKind::TakeVideo {
                                    take_id: id.clone(),
                                }
                            }
                            ResourceKind::TakeAudio { take_id } if take_id == &take.id => {
                                ResourceKind::TakeAudio {
                                    take_id: id.clone(),
                                }
                            }
                            ResourceKind::H3Endpoint {
                                take_id,
                                schema_version,
                                model,
                                shape,
                            } if take_id == &take.id => ResourceKind::H3Endpoint {
                                take_id: id.clone(),
                                schema_version: *schema_version,
                                model: model.clone(),
                                shape: *shape,
                            },
                            ResourceKind::H3Motion {
                                take_id,
                                model,
                                descriptor,
                            } if take_id == &take.id => ResourceKind::H3Motion {
                                take_id: id.clone(),
                                model: model.clone(),
                                descriptor: descriptor.clone(),
                            },
                            ResourceKind::Continuation {
                                take_id,
                                format,
                                format_version,
                                model,
                                width,
                                height,
                            } if take_id == &take.id => ResourceKind::Continuation {
                                take_id: id.clone(),
                                format: format.clone(),
                                format_version: *format_version,
                                model: model.clone(),
                                width: *width,
                                height: *height,
                            },
                            _ => return None,
                        };
                        Some(Resource {
                            id: format!("resource_{}", uuid::Uuid::new_v4().simple()),
                            kind,
                            ..resource.clone()
                        })
                    })
                    .collect();
                candidate.resources.extend(resources);
                candidate.captures.insert(
                    id.clone(),
                    Capture {
                        source_capture: Some(take.id.clone()),
                        imported_from: document
                            .captures
                            .get(&take.id)
                            .and_then(|c| c.imported_from.clone()),
                        ..Default::default()
                    },
                );
                id
            };
            if request.action == Action::AttachAndAccept {
                candidate = review::apply(
                    &candidate,
                    review::ReviewRequest {
                        schema_version: 1,
                        expected_project_revision: candidate.project.project_revision,
                        scene_id: request.scene_id,
                        take_id: id.clone(),
                        action: review::ReviewAction::Accept,
                    },
                )?;
            }
            attached_id = Some(id);
        }
        Action::Detach | Action::Delete => {
            if request.action == Action::Detach && take.scene_id != request.scene_id {
                return Err(StorageError::new("take_scene_mismatch"));
            }
            if candidate
                .project
                .scenes
                .iter()
                .find(|s| s.id == take.scene_id)
                .and_then(|s| s.accepted_take.as_ref())
                .is_some_and(|t| t.id == take.id)
            {
                candidate
                    .project
                    .scenes
                    .iter_mut()
                    .find(|s| s.id == take.scene_id)
                    .unwrap()
                    .accepted_take = None;
            }
            if candidate.locked_takes.get(&take.scene_id) == Some(&take.id) {
                candidate.locked_takes.remove(&take.scene_id);
            }
            candidate
                .takes
                .iter_mut()
                .find(|t| t.id == take.id)
                .unwrap()
                .status = TakeStatus::Candidate;
            let state = candidate.captures.entry(take.id.clone()).or_default();
            state.detached = true;
            if request.action == Action::Delete {
                retained_for_provenance = candidate.takes.iter().any(|t| {
                    t.provenance
                        .as_ref()
                        .is_some_and(|p| p.source_take.as_ref() == Some(&take.id))
                }) || candidate
                    .captures
                    .values()
                    .any(|c| c.source_capture.as_ref() == Some(&take.id));
                if retained_for_provenance {
                    candidate.captures.get_mut(&take.id).unwrap().removed = true;
                } else {
                    candidate.takes.retain(|t| t.id != take.id);
                    candidate.captures.remove(&take.id);
                    candidate.resources.retain(|r| match &r.kind {
                        ResourceKind::Reference { .. } => true,
                        ResourceKind::TakeVideo { take_id }
                        | ResourceKind::TakeAudio { take_id }
                        | ResourceKind::H3Endpoint { take_id, .. }
                        | ResourceKind::H3Motion { take_id, .. }
                        | ResourceKind::Continuation { take_id, .. } => take_id != &take.id,
                    });
                }
            }
            review::invalidate_changed(&document.project, &mut candidate)?;
        }
    }
    candidate.project.project_revision = document
        .project
        .project_revision
        .checked_add(1)
        .filter(|n| *n <= MAX_INTEGER)
        .ok_or_else(|| StorageError::new("invalid_revision"))?;
    candidate.validate()?;
    Ok(Change {
        document: candidate,
        attached_id,
        retained_for_provenance,
        cleanup_warning: None,
    })
}

/// What clearing every Take and Capture would remove; see `clear`.
#[derive(Debug, serde::Serialize)]
pub struct CleanupPreview {
    pub project_revision: u64,
    pub takes: usize,
    /// Project-owned files that would be deleted.
    pub owned_files: usize,
    /// Files kept because they are user originals or still shared.
    pub kept_files: usize,
}

/// Director's project cleanup: removes every Take, Capture and Take-owned
/// resource and clears accepted and locked Takes. References and their
/// files stay, and a user's original files are never deleted.
pub fn clear(document: &ProjectDocument) -> Result<ProjectDocument, StorageError> {
    let mut candidate = document.clone();
    candidate.takes.clear();
    candidate.captures.clear();
    candidate.locked_takes.clear();
    candidate
        .resources
        .retain(|r| matches!(r.kind, ResourceKind::Reference { .. }));
    for scene in &mut candidate.project.scenes {
        scene.accepted_take = None;
    }
    review::invalidate_changed(&document.project, &mut candidate)?;
    candidate.project.project_revision = document
        .project
        .project_revision
        .checked_add(1)
        .filter(|n| *n <= MAX_INTEGER)
        .ok_or_else(|| StorageError::new("invalid_revision"))?;
    candidate.validate()?;
    Ok(candidate)
}

/// Owned blobs `after` no longer uses, and how many removed resources keep their file.
fn released(before: &ProjectDocument, after: &ProjectDocument) -> (Vec<BlobKey>, usize) {
    let retained: std::collections::BTreeSet<_> =
        after.resources.iter().map(Resource::key).collect();
    let removed: Vec<_> = before
        .resources
        .iter()
        .filter(|r| !after.resources.iter().any(|a| a.id == r.id))
        .collect();
    let mut owned: Vec<_> = removed
        .iter()
        .filter(|r| r.source.is_none() && !retained.contains(&r.key()))
        .map(|r| r.key())
        .collect();
    owned.sort();
    owned.dedup();
    let kept = removed
        .iter()
        .filter(|r| r.source.is_some() || retained.contains(&r.key()))
        .count();
    (owned, kept)
}

impl ProjectSession {
    pub fn cleanup_preview(&self) -> Result<CleanupPreview, StorageError> {
        let after = clear(self.document())?;
        let (owned, kept) = released(self.document(), &after);
        Ok(CleanupPreview {
            project_revision: self.document().project.project_revision,
            takes: self.document().takes.len(),
            owned_files: owned.len(),
            kept_files: kept,
        })
    }
    /// Clears every Take and Capture at `expected` revision; returns a warning
    /// when some owned files could not be deleted.
    pub fn clear_takes(
        &mut self,
        expected: u64,
        staging: &SessionStaging,
    ) -> Result<Option<String>, StorageError> {
        if self.document().project.project_revision != expected {
            return Err(StorageError::new("revision_conflict"));
        }
        let after = clear(self.document())?;
        let (owned, _) = released(self.document(), &after);
        let prepared = self.prepare_update(expected, after, BTreeMap::new(), staging)?;
        self.commit_atomic(prepared)?;
        Ok(self.prune_capture_files(&owned).err().map(|e| e.code))
    }
    pub fn import_take(
        &mut self,
        request: Import,
        staging: &SessionStaging,
    ) -> Result<Change, StorageError> {
        let change = import(self.document(), request.clone())?;
        let asset = self
            .document()
            .project
            .assets
            .get(&request.asset_id)
            .ok_or_else(|| StorageError::new("asset_not_found"))?;
        let source = self.document().resources.iter().find(|r| matches!(&r.kind,ResourceKind::Reference{asset_id,revision} if asset_id == &asset.id && *revision == asset.revision))
            .ok_or_else(|| StorageError::new("reference_resource_missing"))?;
        let id = change.attached_id.as_ref().unwrap();
        let video = change
            .document
            .resources
            .iter()
            .find(|r| matches!(&r.kind,ResourceKind::TakeVideo{take_id} if take_id == id))
            .unwrap();
        let inputs = if self
            .document()
            .resources
            .iter()
            .any(|r| r.key() == video.key())
        {
            BTreeMap::new()
        } else {
            BTreeMap::from([(video.key(), self.resource_file(&source.id)?)])
        };
        let prepared = self.prepare_update(
            request.expected_project_revision,
            change.document.clone(),
            inputs,
            staging,
        )?;
        self.commit_atomic(prepared)?;
        Ok(change)
    }
    pub fn capture(
        &mut self,
        request: Request,
        staging: &SessionStaging,
    ) -> Result<Change, StorageError> {
        let mut change = apply(self.document(), request)?;
        let retained: std::collections::BTreeSet<_> = change
            .document
            .resources
            .iter()
            .map(Resource::key)
            .collect();
        let released: Vec<_> = self
            .document()
            .resources
            .iter()
            .filter(|resource| resource.source.is_none())
            .map(Resource::key)
            .filter(|key| !retained.contains(key))
            .collect();
        let prepared = self.prepare_update(
            self.document().project.project_revision,
            change.document.clone(),
            BTreeMap::new(),
            staging,
        )?;
        self.commit_atomic(prepared)?;
        change.cleanup_warning = self.prune_capture_files(&released).err().map(|e| e.code);
        Ok(change)
    }
}
