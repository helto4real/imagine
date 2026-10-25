use super::contract::*;
pub use imagine_video_contract::project::check;
use std::collections::BTreeMap;

/// Session-owned document collection; each handle owns its persistence boundary.
#[derive(Default)]
pub struct ProjectService {
    projects: BTreeMap<ProjectId, super::persistence::ProjectSession>,
}
impl ProjectService {
    pub(crate) fn at_location(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<ProjectId>, super::persistence::StorageError> {
        for (id, project) in &self.projects {
            if project.is_at(path)? {
                return Ok(Some(id.clone()));
            }
        }
        Ok(None)
    }
    /// Each open session's bound folder with its current document.
    pub(crate) fn located(
        &self,
    ) -> Vec<(std::path::PathBuf, &super::persistence::ProjectDocument)> {
        self.projects
            .values()
            .filter_map(|session| session.location().map(|path| (path, session.document())))
            .collect()
    }
    /// A new project folder never lies inside another open project's folder.
    pub(crate) fn outside_projects(
        &self,
        path: &std::path::Path,
    ) -> Result<(), super::persistence::StorageError> {
        if self.projects.values().any(|p| {
            p.location()
                .is_some_and(|location| path != location && path.starts_with(&location))
        }) {
            return Err(super::persistence::StorageError::new(
                "project_location_nested",
            ));
        }
        Ok(())
    }
    pub fn create(&mut self, request: CreateProject) -> Result<VideoProject, Issue> {
        let project = VideoProject::create(request)?;
        self.projects.insert(
            project.id.clone(),
            super::persistence::ProjectSession::new(project.clone())
                .map_err(|e| Issue::new(&e.code, "project"))?,
        );
        Ok(project)
    }
    pub fn create_private(
        &mut self,
        request: CreateProject,
        root: &crate::private_storage::Root,
    ) -> Result<VideoProject, super::persistence::StorageError> {
        let project = VideoProject::create(request)?;
        let mut session = super::persistence::ProjectSession::new(project.clone())?;
        session.save_private(root)?;
        self.insert(session)?;
        Ok(project)
    }
    /// Create and save at once in an explicitly located folder, so every later
    /// edit autosaves there. The folder must be new or empty.
    pub fn create_private_at(
        &mut self,
        request: CreateProject,
        root: &crate::private_storage::Root,
        path: &std::path::Path,
    ) -> Result<VideoProject, super::persistence::StorageError> {
        let project = VideoProject::create(request)?;
        let mut session = super::persistence::ProjectSession::new(project.clone())?;
        session.save_private_at(root, path)?;
        self.insert(session)?;
        Ok(project)
    }
    pub fn create_at(
        &mut self,
        request: CreateProject,
        path: &std::path::Path,
    ) -> Result<VideoProject, super::persistence::StorageError> {
        let project = VideoProject::create(request)?;
        let mut session = super::persistence::ProjectSession::new(project.clone())?;
        session.save_to(path)?;
        self.insert(session)?;
        Ok(project)
    }
    pub fn read(&self, id: &ProjectId) -> Result<VideoProject, Issue> {
        self.projects
            .get(id)
            .map(|p| p.document().project.clone())
            .ok_or_else(|| Issue::new("project_not_found", "project_id"))
    }
    pub fn edit(
        &mut self,
        id: &ProjectId,
        request: super::editing::EditRequest,
    ) -> Result<VideoProject, Issue> {
        let project = self
            .projects
            .get_mut(id)
            .ok_or_else(|| Issue::new("project_not_found", "project_id"))?;
        project
            .edit(request)
            .map_err(|e| Issue::new(&e.code, "project"))?;
        Ok(project.document().project.clone())
    }
    pub fn session(&self, id: &ProjectId) -> Result<&super::persistence::ProjectSession, Issue> {
        self.projects
            .get(id)
            .ok_or_else(|| Issue::new("project_not_found", "project_id"))
    }
    pub fn session_mut(
        &mut self,
        id: &ProjectId,
    ) -> Result<&mut super::persistence::ProjectSession, Issue> {
        self.projects
            .get_mut(id)
            .ok_or_else(|| Issue::new("project_not_found", "project_id"))
    }
    pub fn insert(
        &mut self,
        project: super::persistence::ProjectSession,
    ) -> Result<ProjectId, Issue> {
        let id = project.document().project.id.clone();
        check(
            !self.projects.contains_key(&id),
            "project_already_open",
            "project_id",
        )?;
        self.projects.insert(id.clone(), project);
        Ok(id)
    }
}
