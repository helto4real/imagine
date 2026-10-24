//! Session-owned project handles and explicit local folder selections. No queue ownership.
use super::{bundle::Directory, contract::*, persistence::*, project::ProjectService};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[derive(Default)]
pub struct Service {
    pub reference_cache: Option<std::sync::Arc<dyn super::reference_cache::Control>>,
    /// Where the deployed H3 worker resolves model files; none without H3.
    pub h3_model_root: Option<PathBuf>,
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
fn check_location(path: &Path, session_data: &Path) -> Result<(), StorageError> {
    if !path.is_absolute() || path.starts_with(session_data) || session_data.starts_with(path) {
        return Err(StorageError::new("invalid_project_location"));
    }
    Ok(())
}
impl Service {
    pub fn with_reference_cache(
        reference_cache: Option<std::sync::Arc<dyn super::reference_cache::Control>>,
    ) -> Self {
        Self {
            reference_cache,
            ..Default::default()
        }
    }
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
    /// Create only a missing leaf folder for an explicit location, never its
    /// parents. Private storage creates it only inside an approved root.
    pub fn create_location(
        &self,
        path: &Path,
        session_data: &Path,
        private: Option<&crate::private_storage::Root>,
    ) -> Result<(), StorageError> {
        check_location(path, session_data)?;
        if path
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
        {
            return Err(StorageError::new("invalid_project_location"));
        }
        self.projects.outside_projects(path)?;
        if std::fs::symlink_metadata(path).is_ok() {
            return Ok(());
        }
        if let Some(root) = private {
            return root.create_location(path).map_err(StorageError::new);
        }
        let parent = path
            .parent()
            .ok_or_else(|| StorageError::new("invalid_project_location"))?;
        if std::fs::symlink_metadata(parent)
            .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
        {
            return Err(StorageError::new("location_parent_missing"));
        }
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(path)
            .map_err(|_| StorageError::new("storage_io_failed"))
    }
    pub fn select_location(
        &mut self,
        path: &Path,
        session_data: &Path,
    ) -> Result<String, StorageError> {
        check_location(path, session_data)?;
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
    pub fn copy_private(
        &mut self,
        id: &ProjectId,
        root: &crate::private_storage::Root,
        staging: &SessionStaging,
    ) -> Result<ProjectId, StorageError> {
        let source = self.projects.session(id)?;
        let mut copied =
            ProjectSession::from_copy(source.document().clone(), source.files()?, staging)?;
        copied.save_private(root)?;
        Ok(self.projects.insert(copied)?)
    }
}
