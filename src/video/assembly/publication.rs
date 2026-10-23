use crate::video::{assembly::Plan, contract::*, persistence::*, service::Service};
use std::{collections::BTreeMap, fs::File};
pub struct Prepared {
    update: PreparedUpdate,
    project_id: ProjectId,
    pub asset_id: AssetId,
    pub media: Option<File>,
}
pub fn prepare(
    service: &Service,
    plan: Plan,
    bytes: &[u8],
    staging: &SessionStaging,
) -> Result<Prepared, StorageError> {
    let session = service.projects.session(&plan.project_id)?;
    let mut document = session.document().clone();
    let asset_id = AssetId(crate::storage::id("asset"));
    let mut file = crate::video::execution::media_file(staging, bytes)?;
    let resource = Resource::inspect(
        crate::storage::id("delivery_video"),
        ResourceKind::Reference {
            asset_id: asset_id.clone(),
            revision: 1,
        },
        &mut file,
    )?;
    let media = file
        .try_clone()
        .map_err(|_| StorageError::new("storage_io_failed"))?;
    let key = resource.key();
    document.resources.push(resource);
    document.project.assets.insert(
        asset_id.clone(),
        Asset {
            id: asset_id.clone(),
            revision: 1,
            kind: MediaKind::Video,
        },
    );
    let project_id = plan.project_id.clone();
    document.deliveries.push(StoredDelivery {
        id: crate::storage::id("delivery"),
        asset_id: asset_id.clone(),
        plan,
    });
    let update = session.prepare_update(
        document.project.project_revision,
        document,
        BTreeMap::from([(key, file)]),
        staging,
    )?;
    Ok(Prepared {
        update,
        project_id,
        asset_id,
        media: Some(media),
    })
}
impl Prepared {
    pub fn commit(self, service: &mut Service) {
        let session = service
            .projects
            .session_mut(&self.project_id)
            .expect("locked project");
        session.commit_prepared(self.update);
        let _ = session.autosave();
        service
            .runtime_assets
            .entry(self.project_id)
            .or_default()
            .insert(self.asset_id.clone(), self.asset_id.0);
    }
}
