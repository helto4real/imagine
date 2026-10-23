//! Prepare a complete candidate before the job's output transaction; autosave follows commit.
use super::{execution::Target, persistence::*, planning, service::Service};
use std::collections::BTreeMap;
pub struct Prepared {
    update: super::persistence::PreparedUpdate,
    pub project_id: super::contract::ProjectId,
    pub take_id: super::contract::TakeId,
    pub media: Option<std::fs::File>,
}
pub fn prepare(
    service: &Service,
    target: Target,
    bytes: &[u8],
    artifacts: &[super::artifacts::Produced],
    staging: &SessionStaging,
) -> Result<Prepared, StorageError> {
    let session = service.projects.session(&target.project_id)?;
    let mut document = session.document().clone();
    let mut take = target.take;
    for artifact in artifacts {
        if let super::artifacts::Produced::BoundaryApplied(receipt) = artifact {
            let provenance = take
                .provenance
                .as_mut()
                .ok_or_else(|| StorageError::new("boundary_receipt_mismatch"))?;
            if receipt["applied"] != true
                || provenance
                    .boundary
                    .as_ref()
                    .is_none_or(|expected| expected["dependency"] != receipt["dependency"])
            {
                return Err(StorageError::new("boundary_receipt_mismatch"));
            }
            provenance.boundary = Some(receipt.clone());
        }
    }
    // A concurrent scene edit retains the result, but never declares it current.
    if planning::input_revision(&document.project, &take.scene_id)?
        != take.generated()?.input_revision
    {
        take.status = TakeStatus::Outdated;
    }
    let mut inputs = BTreeMap::new();
    for (resource, file) in target.references {
        if !document.resources.iter().any(|r| r == &resource) {
            if document.resources.iter().any(|r| r.id == resource.id) {
                return Err(StorageError::new("resource_identity_conflict"));
            }
            inputs.insert(resource.key(), file);
            document.resources.push(resource);
        }
    }
    let mut file = super::execution::media_file(staging, bytes)?;
    let resource = Resource::inspect(
        format!("take_video_{}", uuid::Uuid::new_v4().simple()),
        ResourceKind::TakeVideo {
            take_id: take.id.clone(),
        },
        &mut file,
    )?;
    let media = file
        .try_clone()
        .map_err(|_| StorageError::new("storage_io_failed"))?;
    inputs.insert(resource.key(), file);
    document.resources.push(resource);
    for artifact in artifacts {
        if matches!(artifact, super::artifacts::Produced::BoundaryApplied(_)) {
            continue;
        }
        let (resource, file) = artifact.prepare(&take, staging)?;
        inputs.insert(resource.key(), file);
        document.resources.push(resource);
    }
    let take_id = take.id.clone();
    document.takes.push(take);
    let update =
        session.prepare_update(document.project.project_revision, document, inputs, staging)?;
    Ok(Prepared {
        update,
        project_id: target.project_id,
        take_id,
        media: Some(media),
    })
}
impl Prepared {
    /// Caller holds the service mutex across output SQL commit and this infallible
    /// session swap. Save failure leaves the complete candidate available for retry.
    pub fn commit(self, service: &mut Service, asset: String) {
        let session = service
            .projects
            .session_mut(&self.project_id)
            .expect("locked project");
        session.commit_prepared(self.update);
        let _ = session.autosave();
        service
            .runtime_takes
            .entry(self.project_id)
            .or_default()
            .insert(self.take_id, asset);
    }
}
