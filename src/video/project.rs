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
