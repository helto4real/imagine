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
        let (proof, mut parent) = d::inspect(&self.config, self.environment.as_ref())?;
        d::writable(&parent, 0)?;
        for part in relative.components() {
            use std::os::unix::ffi::OsStrExt;
            let name = CString::new(part.as_os_str().as_bytes())
                .map_err(|_| "video_destination_invalid")?;
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
        self.proof()?;
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
    pub fn library(&self) -> d::Result<PathBuf> {
        self.directory(Path::new("library"))?;
        Ok(self.layout().library())
    }
    /// Revalidated export destination for this exact private child directory.
    pub fn destination(&self, path: &Path) -> d::Result<d::Config> {
        let relative = path
            .strip_prefix(&self.config.root)
            .map_err(|_| "video_destination_invalid")?;
        self.directory(relative)?;
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
}
