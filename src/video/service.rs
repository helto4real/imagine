//! Session-owned project handles and explicit local folder selections. No queue ownership.
use super::{bundle::Directory, contract::*, persistence::*, project::ProjectService};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

#[derive(Default)]
pub struct Service {
    pub assemblies: BTreeMap<String, super::assembly::Entry>,
    pub authoring: std::sync::Arc<super::authoring::drafts::Service>,
    pub libraries: BTreeMap<String, super::library::LibrarySession>,
    pub projects: ProjectService,
    locations: BTreeMap<String, Location>,
    pub runtime_assets: BTreeMap<ProjectId, BTreeMap<AssetId, String>>,
    pub runtime_takes: BTreeMap<ProjectId, BTreeMap<TakeId, String>>,
}
struct Location {
    path: PathBuf,
    directory: Directory,
}
#[derive(Serialize)]
pub struct Snapshot {
    #[serde(flatten)]
    pub document: ProjectDocument,
    pub persistence: PersistenceStatus,
    pub runtime_assets: BTreeMap<AssetId, String>,
    pub runtime_takes: BTreeMap<TakeId, String>,
    pub take_states: BTreeMap<TakeId, super::review::TakeState>,
}
impl Service {
    pub fn snapshot(&self, id: &ProjectId) -> Result<Snapshot, Issue> {
        let project = self.projects.session(id)?;
        Ok(Snapshot {
            document: project.document().clone(),
            persistence: project.status().clone(),
            runtime_assets: self.runtime_assets.get(id).cloned().unwrap_or_default(),
            runtime_takes: self.runtime_takes.get(id).cloned().unwrap_or_default(),
            take_states: super::review::states(project.document()),
        })
    }
    pub fn select_location(
        &mut self,
        path: &Path,
        session_data: &Path,
    ) -> Result<String, StorageError> {
        if !path.is_absolute() || path.starts_with(session_data) || session_data.starts_with(path) {
            return Err(StorageError::new("invalid_project_location"));
        }
        let directory = Directory::select(path)?;
        let id = format!("location_{}", uuid::Uuid::new_v4().simple());
        self.locations.insert(
            id.clone(),
            Location {
                path: path.into(),
                directory,
            },
        );
        Ok(id)
    }
    pub fn location(&self, id: &str) -> Result<PathBuf, StorageError> {
        let selected = self
            .locations
            .get(id)
            .ok_or_else(|| StorageError::new("location_not_found"))?;
        selected.directory.check()?;
        Ok(selected.path.clone())
    }
    pub fn copy_to(
        &mut self,
        id: &ProjectId,
        path: &Path,
        staging: &SessionStaging,
    ) -> Result<ProjectId, StorageError> {
        let source = self.projects.session(id)?;
        let mut copied =
            ProjectSession::from_copy(source.document().clone(), source.files()?, staging)?;
        copied.save_to(path)?;
        let new_id = self.projects.insert(copied)?;
        Ok(new_id)
    }
}
