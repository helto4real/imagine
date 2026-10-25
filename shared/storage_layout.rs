//! Approved storage classes. Path construction does not create or inspect files.
use std::path::{Component, Path, PathBuf};

pub const PRIVATE_ROOT: &str = "/run/media/veracrypt1/imagine";
pub const PRIVATE_MOUNT: &str = "/run/media/veracrypt1";
/// Approved roots for agent-selected project folders, besides the private root.
/// Older agent projects live here; new agent work uses the MCP root.
pub const DEFAULT_PROJECT_ROOTS: &[&str] = &["/run/media/veracrypt1/mcp-inputs"];
/// Agent workspaces, each in its creation date folder.
pub const MCP_ROOT: &str = "/run/media/veracrypt1/mcp-root";

/// The date folder named by a `project_YYYYMMDD_…` ID (see the video
/// contract's `ProjectId::generate`); older IDs have none.
pub fn date_folder(id: &str) -> Option<String> {
    let date = id.strip_prefix("project_")?.split('_').next()?;
    if date.len() != 8 || !date.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (month, day) = (
        date[4..6].parse::<u32>().ok()?,
        date[6..8].parse::<u32>().ok()?,
    );
    ((1..=12).contains(&month) && (1..=31).contains(&day))
        .then(|| format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8]))
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    root: PathBuf,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
    pub root: PathBuf,
    pub document: PathBuf,
    pub images: PathBuf,
    pub videos: PathBuf,
    pub audio: PathBuf,
    pub work: PathBuf,
    pub tmp: PathBuf,
    pub autosave: PathBuf,
}
impl Default for Layout {
    fn default() -> Self {
        Self::new(PathBuf::from(PRIVATE_ROOT))
    }
}
impl Layout {
    /// An explicit root supports source-owned synthetic test volumes. Production
    /// callers obtain the writable root through private_storage::Root first.
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn project(&self, id: &str) -> Result<Project, &'static str> {
        // IDs are one directory component, never a supplied relative path.
        let mut parts = Path::new(id).components();
        if !matches!(parts.next(), Some(Component::Normal(_))) || parts.next().is_some() {
            return Err("invalid_project_id");
        }
        // New IDs carry their creation date and live in that date folder;
        // older IDs keep their folder directly under `projects`.
        let projects = self.root.join("projects");
        Ok(Self::at(match date_folder(id) {
            Some(date) => projects.join(date).join(id),
            None => projects.join(id),
        }))
    }
    /// Whether `path` is the ID-named folder of a project in this root, with or
    /// without its date folder.
    pub fn is_named_project(&self, path: &Path) -> bool {
        path.file_name()
            .and_then(|id| id.to_str())
            .and_then(|id| self.project(id).ok())
            .is_some_and(|project| project.root == path)
    }
    /// The same project classes at a folder bound by an explicit location.
    pub fn project_at(&self, path: &Path) -> Result<Project, &'static str> {
        if !path.is_absolute()
            || path
                .components()
                .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
        {
            return Err("invalid_project_location");
        }
        Ok(Self::at(path.to_path_buf()))
    }
    fn at(root: PathBuf) -> Project {
        Project {
            document: root.join("project.json"),
            images: root.join("outputs/images"),
            videos: root.join("outputs/videos"),
            audio: root.join("outputs/audio"),
            work: root.join("work"),
            tmp: root.join("tmp"),
            autosave: root.join("autosave"),
            root,
        }
    }
    pub fn library(&self) -> PathBuf {
        self.root.join("library")
    }
}

/// These files may contain only the indicated non-private preferences/cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicPaths {
    pub ui: PathBuf,
    pub models: PathBuf,
    pub cache: PathBuf,
    pub logs: PathBuf,
}
impl PublicPaths {
    pub fn for_home(home: &Path) -> Self {
        let config = home.join(".config/imagine");
        Self {
            ui: config.join("ui.json"),
            models: config.join("models.json"),
            cache: home.join(".cache/imagine"),
            logs: home.join(".local/state/imagine/logs"),
        }
    }
}
pub fn runtime_directory(xdg: Option<&Path>) -> Result<PathBuf, &'static str> {
    xdg.filter(|path| path.is_absolute())
        .map(|path| path.join("imagine"))
        .ok_or("runtime_directory_unavailable")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_approved_classes_and_no_runtime_tmp_fallback() {
        let layout = Layout::default();
        let project = layout.project("project_own").unwrap();
        assert_eq!(
            project.document,
            Path::new(PRIVATE_ROOT).join("projects/project_own/project.json")
        );
        for path in [
            &project.images,
            &project.videos,
            &project.audio,
            &project.work,
            &project.tmp,
            &project.autosave,
        ] {
            assert!(path.starts_with(&project.root));
        }
        assert_eq!(layout.library(), Path::new(PRIVATE_ROOT).join("library"));
        let public = PublicPaths::for_home(Path::new("/own-home"));
        assert_eq!(public.ui, Path::new("/own-home/.config/imagine/ui.json"));
        assert_eq!(
            public.models,
            Path::new("/own-home/.config/imagine/models.json")
        );
        assert_eq!(public.cache, Path::new("/own-home/.cache/imagine"));
        assert_eq!(
            public.logs,
            Path::new("/own-home/.local/state/imagine/logs")
        );
        assert!(runtime_directory(None).is_err());
        assert!(runtime_directory(Some(Path::new("relative"))).is_err());
        assert_eq!(
            runtime_directory(Some(Path::new("/run/user/999"))).unwrap(),
            Path::new("/run/user/999/imagine")
        );
        for id in ["", ".", "..", "/absolute", "one/two", "one/../two"] {
            assert!(layout.project(id).is_err(), "{id}");
        }
        // New IDs carry their creation date and live in its folder.
        let dated = layout.project("project_20261025_own").unwrap();
        assert_eq!(
            dated.root,
            Path::new(PRIVATE_ROOT).join("projects/2026-10-25/project_20261025_own")
        );
        assert!(layout.is_named_project(&dated.root));
        assert!(layout.is_named_project(&project.root));
        assert!(
            !layout
                .is_named_project(&Path::new(PRIVATE_ROOT).join("projects/project_20261025_own"))
        );
        for id in [
            "project_2026102_own",
            "project_20261325_own",
            "project_image_own",
        ] {
            assert_eq!(date_folder(id), None, "{id}");
        }
        let elsewhere = Path::new("/own-volume/own-inputs/own-project");
        let named = layout.project("own-project").unwrap();
        let moved = layout.project_at(elsewhere).unwrap();
        for (at, by_id) in [
            (&moved.document, &named.document),
            (&moved.images, &named.images),
            (&moved.videos, &named.videos),
            (&moved.audio, &named.audio),
            (&moved.work, &named.work),
            (&moved.tmp, &named.tmp),
            (&moved.autosave, &named.autosave),
        ] {
            assert_eq!(
                at.strip_prefix(elsewhere).unwrap(),
                by_id.strip_prefix(&named.root).unwrap()
            );
        }
        for path in ["relative", "/own/../escape", "./own"] {
            assert!(layout.project_at(Path::new(path)).is_err(), "{path}");
        }
    }
}
