//! Private writes use the same mounted-volume/dirfd contract as media export.
//! Never creates the mount/root itself or falls back to a different filesystem.
use crate::{storage_layout as layout, video_destination as d};
use std::io::{Read, Write};
use std::{
    ffi::CString,
    fs::File,
    os::fd::{AsRawFd, FromRawFd},
    os::unix::fs::MetadataExt,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

#[derive(Clone)]
pub struct Root {
    config: d::Config,
    environment: Arc<dyn d::Environment>,
    runtime: Option<PathBuf>,
    project_roots: Vec<PathBuf>,
}
impl Root {
    pub fn system() -> d::Result<Self> {
        let environment: Arc<dyn d::Environment> = Arc::new(d::System);
        let mount = PathBuf::from(layout::PRIVATE_MOUNT);
        let root = PathBuf::from(layout::PRIVATE_ROOT);
        let text = environment.mounts()?;
        let filesystem = text
            .lines()
            .find_map(|line| {
                let (left, right) = line.split_once(" - ")?;
                let left: Vec<_> = left.split_whitespace().collect();
                let right: Vec<_> = right.split_whitespace().collect();
                (left.get(4) == Some(&layout::PRIVATE_MOUNT)
                    && right.get(1) == Some(&"/dev/mapper/veracrypt1"))
                .then(|| right.first().map(|s| s.to_string()))
                .flatten()
            })
            .ok_or("video_volume_not_mounted")?;
        let fsid = d::fsid(&d::directory(&mount)?)?;
        let config = d::Config {
            root,
            mount,
            filesystem,
            source: "/dev/mapper/veracrypt1".into(),
            fsid,
        };
        Self::with_environment(config, environment)
    }
    pub fn with_environment(
        config: d::Config,
        environment: Arc<dyn d::Environment>,
    ) -> d::Result<Self> {
        let root = Self {
            config,
            environment,
            runtime: None,
            project_roots: vec![],
        };
        root.proof()?;
        Ok(root)
    }
    pub fn proof(&self) -> d::Result<d::Proof> {
        let (proof, file) = d::inspect(&self.config, self.environment.as_ref())?;
        d::writable(&file, 0)?;
        Ok(proof)
    }
    pub fn layout(&self) -> layout::Layout {
        layout::Layout::new(self.config.root.clone())
    }
    /// Approve folders for explicitly located projects. Only roots strictly
    /// inside this verified mount are kept; the rest are returned to report.
    pub fn with_project_roots(mut self, roots: &[PathBuf]) -> (Self, Vec<PathBuf>) {
        let (kept, ignored) = roots.iter().cloned().partition(|root| {
            clean(root) && root.starts_with(&self.config.mount) && *root != self.config.mount
        });
        self.project_roots = kept;
        (self, ignored)
    }
    pub fn project_roots(&self) -> &[PathBuf] {
        &self.project_roots
    }
    /// One storage guard: the mount proof holds and the existing directory is
    /// inside the private root or an approved project root on the same volume.
    pub fn allows(&self, path: &Path) -> bool {
        self.approved(path) && self.verified(path).is_ok()
    }
    fn approved(&self, path: &Path) -> bool {
        clean(path)
            && (path.starts_with(&self.config.root)
                || self.project_roots.iter().any(|root| path.starts_with(root)))
    }
    fn verified(&self, path: &Path) -> d::Result<File> {
        let proof = self.proof()?;
        let file = d::directory(path)?;
        if file
            .metadata()
            .map_err(|_| "video_destination_unavailable")?
            .dev()
            != proof.device
            || d::fsid(&file)? != self.config.fsid
        {
            return Err("video_volume_mismatch");
        }
        Ok(file)
    }
    /// A folder an explicit location may bind a project to: strictly inside an
    /// approved root, and outside the ID-named `projects` and the library areas.
    pub fn project_location(&self, path: &Path) -> d::Result<()> {
        if !clean(path) {
            return Err("invalid_project_location");
        }
        let layout = self.layout();
        if path.starts_with(layout.root().join("projects")) || path.starts_with(layout.library()) {
            return Err("invalid_project_location");
        }
        let Some(root) = std::iter::once(&self.config.root)
            .chain(&self.project_roots)
            .find(|root| path != *root && path.starts_with(root))
        else {
            return Err("invalid_storage_root");
        };
        // Never inside another project's folder below the approved root.
        if path
            .ancestors()
            .skip(1)
            .take_while(|ancestor| ancestor != root)
            .any(|ancestor| std::fs::symlink_metadata(ancestor.join("project.json")).is_ok())
        {
            return Err("project_location_nested");
        }
        Ok(())
    }
    /// Create one missing folder whose parent already exists and is allowed;
    /// intermediate directories are never created.
    pub fn create_location(&self, path: &Path) -> d::Result<()> {
        self.project_location(path)?;
        let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
            return Err("invalid_project_location");
        };
        self.proof()?;
        if std::fs::symlink_metadata(parent)
            .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        {
            return Err("location_parent_missing");
        }
        let parent = self.verified(parent).map_err(|_| "invalid_storage_root")?;
        d::writable(&parent, 0)?;
        use std::os::unix::ffi::OsStrExt;
        let name = CString::new(name.as_bytes()).map_err(|_| "invalid_project_location")?;
        if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST)
        {
            return Err("video_destination_not_writable");
        }
        self.verified(path).map(|_| ())
    }
    /// The same verified volume rooted at one approved project root, for
    /// storage kept beside the private root (the MCP root). Creates that one
    /// folder when its parent exists; never intermediate folders.
    pub fn rooted_at(&self, path: &Path) -> d::Result<Self> {
        if !self.project_roots.iter().any(|root| root == path) {
            return Err("invalid_storage_root");
        }
        if std::fs::symlink_metadata(path).is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound)
        {
            let (Some(parent), Some(name)) = (path.parent(), path.file_name()) else {
                return Err("invalid_storage_root");
            };
            let parent = self.verified(parent).map_err(|_| "invalid_storage_root")?;
            d::writable(&parent, 0)?;
            use std::os::unix::ffi::OsStrExt;
            let name = CString::new(name.as_bytes()).map_err(|_| "invalid_storage_root")?;
            if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0
                && std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST)
            {
                return Err("video_destination_not_writable");
            }
        }
        let mut config = self.config.clone();
        config.root = path.to_path_buf();
        let mut root = Self::with_environment(config, self.environment.clone())?;
        root.runtime = self.runtime.clone();
        root.project_roots = self.project_roots.clone();
        Ok(root)
    }
    /// A new file at `relative`, written through the returned handle and
    /// published by `NewFile::commit` without replacing an existing file.
    pub fn create_new(&self, relative: &Path) -> d::Result<NewFile> {
        let (parent, name) = self.parent(relative)?;
        d::writable(&parent, 0)?;
        let temporary = CString::new(format!(
            ".private-{}.partial",
            uuid::Uuid::new_v4().simple()
        ))
        .unwrap();
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                temporary.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err("private_storage_write_failed");
        }
        Ok(NewFile {
            file: Some(unsafe { File::from_raw_fd(fd) }),
            parent,
            temporary,
            name,
            root: self.clone(),
        })
    }
    /// Share the same mount witness with disk owners; fixtures inject only their
    /// own volume policy instead of substituting production paths.
    pub fn environment(&self) -> Arc<dyn d::Environment> {
        self.environment.clone()
    }
    /// Explicit runtime injection is for an owned instance/fixture. Missing XDG
    /// runtime is an error, never permission to use a disk temporary directory.
    pub fn with_runtime_directory(mut self, path: PathBuf) -> d::Result<Self> {
        if !path.is_absolute() {
            return Err("runtime_directory_unavailable");
        }
        self.runtime = Some(path);
        Ok(self)
    }
    pub fn runtime_directory(&self) -> d::Result<PathBuf> {
        if let Some(path) = &self.runtime {
            return Ok(path.clone());
        }
        layout::runtime_directory(
            std::env::var_os("XDG_RUNTIME_DIR")
                .as_deref()
                .map(Path::new),
        )
    }
    /// Allocate only child directories after verifying the mounted root. Each
    /// child is opened without following symlinks and remains on that volume.
    pub fn directory(&self, relative: &Path) -> d::Result<File> {
        if relative.is_absolute()
            || relative
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err("video_destination_invalid");
        }
        let (proof, parent) = d::inspect(&self.config, self.environment.as_ref())?;
        d::writable(&parent, 0)?;
        let parent = self.descend(&proof, parent, relative)?;
        self.proof()?;
        Ok(parent)
    }
    fn descend(&self, proof: &d::Proof, mut parent: File, relative: &Path) -> d::Result<File> {
        for part in relative.components() {
            use std::os::unix::ffi::OsStrExt;
            let Component::Normal(part) = part else {
                return Err("video_destination_invalid");
            };
            let name = CString::new(part.as_bytes()).map_err(|_| "video_destination_invalid")?;
            if unsafe { libc::mkdirat(parent.as_raw_fd(), name.as_ptr(), 0o700) } != 0
                && std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST)
            {
                return Err("video_destination_not_writable");
            }
            let fd = unsafe {
                libc::openat(
                    parent.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                return Err("video_destination_invalid");
            }
            parent = unsafe { File::from_raw_fd(fd) };
            if parent
                .metadata()
                .map_err(|_| "video_destination_unavailable")?
                .dev()
                != proof.device
                || d::fsid(&parent)? != self.config.fsid
            {
                return Err("video_volume_mismatch");
            }
        }
        Ok(parent)
    }
    pub fn project(&self, id: &str) -> d::Result<layout::Project> {
        let project = self.layout().project(id)?;
        for path in [
            &project.images,
            &project.videos,
            &project.audio,
            &project.work,
            &project.tmp,
            &project.autosave,
        ] {
            self.directory(
                path.strip_prefix(&self.config.root)
                    .map_err(|_| "video_destination_invalid")?,
            )?;
        }
        Ok(project)
    }
    /// The project classes inside an allowed, already existing project folder.
    pub fn project_at(&self, path: &Path) -> d::Result<layout::Project> {
        let project = self.layout().project_at(path)?;
        if !self.approved(path) {
            return Err("invalid_storage_root");
        }
        let (proof, _) = d::inspect(&self.config, self.environment.as_ref())?;
        let base = self.verified(path)?;
        d::writable(&base, 0)?;
        for child in [
            &project.images,
            &project.videos,
            &project.audio,
            &project.work,
            &project.tmp,
            &project.autosave,
        ] {
            let relative = child
                .strip_prefix(path)
                .map_err(|_| "video_destination_invalid")?;
            let base = base
                .try_clone()
                .map_err(|_| "video_destination_unavailable")?;
            self.descend(&proof, base, relative)?;
        }
        self.proof()?;
        Ok(project)
    }
    pub fn library(&self) -> d::Result<PathBuf> {
        self.directory(Path::new("library"))?;
        Ok(self.layout().library())
    }
    /// Revalidated export destination for this exact private child directory.
    pub fn destination(&self, path: &Path) -> d::Result<d::Config> {
        if let Ok(relative) = path.strip_prefix(&self.config.root) {
            self.directory(relative)?;
        } else if self.approved(path) {
            self.verified(path)?;
        } else {
            return Err("video_destination_invalid");
        }
        let mut config = self.config.clone();
        config.root = path.into();
        d::inspect(&config, self.environment.as_ref())?;
        Ok(config)
    }
    fn parent(&self, relative: &Path) -> d::Result<(File, CString)> {
        if relative.is_absolute()
            || relative
                .components()
                .any(|p| !matches!(p, Component::Normal(_)))
        {
            return Err("video_destination_invalid");
        }
        let name = relative
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("video_destination_invalid")?;
        let parent = self.directory(relative.parent().unwrap_or(Path::new("")))?;
        Ok((
            parent,
            CString::new(name).map_err(|_| "video_destination_invalid")?,
        ))
    }
    pub fn read(&self, relative: &Path) -> d::Result<Option<Vec<u8>>> {
        let (parent, name) = self.parent(relative)?;
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
            )
        };
        if fd < 0 {
            return if std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOENT) {
                Ok(None)
            } else {
                Err("private_storage_read_failed")
            };
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        if !file
            .metadata()
            .map_err(|_| "private_storage_read_failed")?
            .is_file()
        {
            return Err("private_storage_read_failed");
        }
        let mut bytes = vec![];
        file.read_to_end(&mut bytes)
            .map_err(|_| "private_storage_read_failed")?;
        self.proof()?;
        Ok(Some(bytes))
    }
    /// All staging, the final rename and fsync stay on the verified volume.
    pub fn write(&self, relative: &Path, bytes: &[u8]) -> d::Result<()> {
        let (parent, name) = self.parent(relative)?;
        d::writable(&parent, bytes.len() as u64)?;
        let temporary = CString::new(format!(
            ".private-{}.partial",
            uuid::Uuid::new_v4().simple()
        ))
        .unwrap();
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                temporary.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err("private_storage_write_failed");
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let result = (|| {
            file.write_all(bytes)
                .map_err(|_| "private_storage_write_failed")?;
            file.sync_all()
                .map_err(|_| "private_storage_write_failed")?;
            self.proof()?;
            if unsafe {
                libc::renameat(
                    parent.as_raw_fd(),
                    temporary.as_ptr(),
                    parent.as_raw_fd(),
                    name.as_ptr(),
                )
            } != 0
            {
                return Err("private_storage_write_failed");
            }
            parent
                .sync_all()
                .map_err(|_| "private_storage_write_unconfirmed")
        })();
        if result.is_err() {
            unsafe {
                libc::unlinkat(parent.as_raw_fd(), temporary.as_ptr(), 0);
            }
        }
        result
    }
}

/// A file being written on the verified volume; removed unless committed.
pub struct NewFile {
    file: Option<File>,
    parent: File,
    temporary: CString,
    name: CString,
    root: Root,
}
impl NewFile {
    /// The open file to write into.
    pub fn file(&mut self) -> &mut File {
        self.file.as_mut().expect("uncommitted file")
    }
    /// Flushes and publishes under the requested name; fails with
    /// `private_storage_exists` when that name is taken.
    pub fn commit(mut self) -> d::Result<()> {
        let file = self.file.take().expect("uncommitted file");
        file.sync_all()
            .map_err(|_| "private_storage_write_failed")?;
        drop(file);
        self.root.proof()?;
        let mut renamed = unsafe {
            libc::renameat2(
                self.parent.as_raw_fd(),
                self.temporary.as_ptr(),
                self.parent.as_raw_fd(),
                self.name.as_ptr(),
                libc::RENAME_NOREPLACE,
            )
        };
        // Filesystems without RENAME_NOREPLACE: check, then rename.
        if renamed != 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINVAL) {
            let mut stat: libc::stat = unsafe { std::mem::zeroed() };
            let taken = unsafe {
                libc::fstatat(
                    self.parent.as_raw_fd(),
                    self.name.as_ptr(),
                    &mut stat,
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } == 0;
            renamed = if taken {
                unsafe { *libc::__errno_location() = libc::EEXIST };
                -1
            } else {
                unsafe {
                    libc::renameat(
                        self.parent.as_raw_fd(),
                        self.temporary.as_ptr(),
                        self.parent.as_raw_fd(),
                        self.name.as_ptr(),
                    )
                }
            };
        }
        if renamed != 0 {
            let exists = std::io::Error::last_os_error().raw_os_error() == Some(libc::EEXIST);
            unsafe {
                libc::unlinkat(self.parent.as_raw_fd(), self.temporary.as_ptr(), 0);
            }
            self.temporary = CString::default();
            return Err(if exists {
                "private_storage_exists"
            } else {
                "private_storage_write_failed"
            });
        }
        self.temporary = CString::default();
        self.parent
            .sync_all()
            .map_err(|_| "private_storage_write_unconfirmed")
    }
}
impl Drop for NewFile {
    fn drop(&mut self) {
        if !self.temporary.as_bytes().is_empty() {
            unsafe {
                libc::unlinkat(self.parent.as_raw_fd(), self.temporary.as_ptr(), 0);
            }
        }
    }
}

fn clean(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|part| matches!(part, Component::RootDir | Component::Normal(_)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    struct Environment(Mutex<String>);
    impl d::Environment for Environment {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            Ok("own-fixture".into())
        }
    }
    #[test]
    fn mounted_root_children_and_outage_never_fall_back() {
        let temp = tempfile::tempdir().unwrap();
        let mount = temp.path().join("volume");
        let root = mount.join("imagine");
        std::fs::create_dir_all(&root).unwrap();
        let config = d::Config {
            root: root.clone(),
            mount: mount.clone(),
            source: "own-fixture".into(),
            filesystem: "fixturefs".into(),
            fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
        };
        let mounts = format!(
            "1 0 0:1 / {} rw - fixturefs own-fixture rw\n",
            mount.display()
        );
        let env = Arc::new(Environment(Mutex::new(mounts.clone())));
        let policy = Root::with_environment(config, env.clone()).unwrap();
        let project = policy.project("own-project").unwrap();
        assert!(
            project.tmp.is_dir() && project.images.is_dir() && policy.library().unwrap().is_dir()
        );
        std::os::unix::fs::symlink(temp.path(), root.join("redirect")).unwrap();
        assert!(policy.directory(Path::new("redirect/private")).is_err());
        assert!(!temp.path().join("private").exists());
        *env.0.lock().unwrap() = mounts.replace(" rw", " ro");
        assert_eq!(
            policy.project("after-readonly").unwrap_err(),
            "video_destination_read_only"
        );
        assert!(!root.join("projects/after-readonly").exists());
        env.0.lock().unwrap().clear();
        assert_eq!(
            policy.project("after-unmount").unwrap_err(),
            "video_volume_not_mounted"
        );
        assert!(!root.join("projects/after-unmount").exists());
        assert!(policy.directory(Path::new("../outside")).is_err());
    }
    #[test]
    fn approved_project_roots_stay_inside_the_verified_mount() {
        let temp = tempfile::tempdir().unwrap();
        let mount = temp.path().join("volume");
        let root = mount.join("imagine");
        let inputs = mount.join("own-inputs");
        let outside = temp.path().join("outside");
        for path in [&root, &inputs, &outside] {
            std::fs::create_dir_all(path).unwrap();
        }
        let config = d::Config {
            root: root.clone(),
            mount: mount.clone(),
            source: "own-fixture".into(),
            filesystem: "fixturefs".into(),
            fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
        };
        let env = Arc::new(Environment(Mutex::new(format!(
            "1 0 0:1 / {} rw - fixturefs own-fixture rw\n",
            mount.display()
        ))));
        let (policy, ignored) = Root::with_environment(config, env.clone())
            .unwrap()
            .with_project_roots(&[
                inputs.clone(),
                outside.clone(),
                mount.clone(),
                "relative".into(),
                mount.join("own/../escape"),
            ]);
        assert_eq!(policy.project_roots(), std::slice::from_ref(&inputs));
        assert_eq!(
            ignored,
            [
                outside.clone(),
                mount.clone(),
                "relative".into(),
                mount.join("own/../escape")
            ]
        );
        assert!(policy.allows(&root) && policy.allows(&inputs));
        assert!(!policy.allows(&outside) && !policy.allows(&mount));
        assert!(!policy.allows(&inputs.join("missing")));
        std::os::unix::fs::symlink(&outside, inputs.join("redirect")).unwrap();
        assert!(!policy.allows(&inputs.join("redirect")));
        assert_eq!(
            policy.create_location(&inputs.join("redirect/own")),
            Err("invalid_storage_root")
        );
        assert!(!outside.join("own").exists());

        let located = inputs.join("own-project");
        policy.create_location(&located).unwrap();
        assert!(located.is_dir());
        assert_eq!(
            policy.create_location(&inputs.join("missing/own")),
            Err("location_parent_missing")
        );
        assert!(!inputs.join("missing").exists());
        for (path, code) in [
            (outside.join("own"), "invalid_storage_root"),
            (mount.join("own"), "invalid_storage_root"),
            (root.join("projects/own"), "invalid_project_location"),
            (root.join("library/own"), "invalid_project_location"),
        ] {
            assert_eq!(policy.create_location(&path), Err(code), "{path:?}");
            assert!(!path.exists());
        }
        assert_eq!(
            policy.project_location(&inputs),
            Err("invalid_storage_root")
        );
        assert_eq!(policy.project_location(&root), Err("invalid_storage_root"));
        assert_eq!(
            policy.project_location(Path::new("own-relative")),
            Err("invalid_project_location")
        );
        policy
            .project_location(&root.join("own-elsewhere"))
            .unwrap();

        let project = policy.project_at(&located).unwrap();
        assert!(project.images.is_dir() && project.videos.is_dir() && project.autosave.is_dir());
        assert_eq!(
            policy.destination(&project.videos).unwrap().root,
            project.videos
        );
        assert!(policy.project_at(&outside).is_err());
        assert!(policy.destination(&outside).is_err());
        assert!(!outside.join("outputs").exists());

        env.0.lock().unwrap().clear();
        assert!(!policy.allows(&located));
        assert_eq!(
            policy.create_location(&inputs.join("after-unmount")),
            Err("video_volume_not_mounted")
        );
        assert!(!inputs.join("after-unmount").exists());
    }
}
