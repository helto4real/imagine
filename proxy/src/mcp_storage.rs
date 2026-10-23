//! Disposable MCP copies on the verified media volume. Never scans historical directories.
use crate::media_destination::{self as d, Environment, Proof, System};
use anyhow::{Context, Result, anyhow, bail};
use fs2::FileExt;
use std::{
    collections::{HashMap, HashSet},
    ffi::CString,
    fs::File,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
struct OwnedDirectory {
    fd: File,
    files: HashSet<String>,
}
pub struct Storage {
    pub root: PathBuf,
    proof: Proof,
    env: Arc<dyn Environment>,
    cache: File,
    cache_path: PathBuf,
    directories: Mutex<HashMap<PathBuf, OwnedDirectory>>,
    _lock: File,
}
fn err(s: &'static str) -> anyhow::Error {
    anyhow!(s)
}
fn name(value: &str) -> Result<CString> {
    if value.is_empty() || value == "." || value == ".." || value.contains('/') {
        bail!("invalid MCP cache name");
    }
    Ok(CString::new(value)?)
}
fn open_dir(parent: &File, value: &str, create: bool) -> Result<File> {
    let n = name(value)?;
    if create && unsafe { libc::mkdirat(parent.as_raw_fd(), n.as_ptr(), 0o700) } != 0 {
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(e.into());
        }
    }
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            n.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(unsafe { File::from_raw_fd(fd) })
}
fn same(a: &File, b: &File) -> Result<bool> {
    let a = a.metadata()?;
    let b = b.metadata()?;
    Ok(a.dev() == b.dev() && a.ino() == b.ino())
}
fn unlink(fd: &File, n: &str, flags: i32) -> Result<()> {
    if unsafe { libc::unlinkat(fd.as_raw_fd(), name(n)?.as_ptr(), flags) } != 0 {
        let e = std::io::Error::last_os_error();
        if e.kind() != std::io::ErrorKind::NotFound {
            return Err(e.into());
        }
    }
    Ok(())
}
pub(crate) fn write_bytes(writer: &mut dyn Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes).map_err(|e| match e.raw_os_error() {
        Some(libc::ENOSPC | libc::EDQUOT) => anyhow!("mcp_destination_full"),
        Some(libc::EACCES) => anyhow!("mcp_destination_not_writable"),
        Some(libc::EROFS) => anyhow!("mcp_destination_read_only"),
        _ => e.into(),
    })
}
impl Storage {
    pub fn open(root: &Path, input: &Path, proof: Proof) -> Result<Self> {
        Self::with_environment(root, input, proof, Arc::new(System))
    }
    pub fn with_environment(
        root: &Path,
        input: &Path,
        proof: Proof,
        env: Arc<dyn Environment>,
    ) -> Result<Self> {
        // A single configured child of the trusted root; no arbitrary paths or fallback.
        let child = root
            .strip_prefix(&proof.config.root)
            .context("MCP output must be a direct child of the verified media destination")?;
        if child.components().count() != 1
            || !matches!(
                child.components().next(),
                Some(std::path::Component::Normal(_))
            )
        {
            bail!("MCP output must be a direct child of the verified media destination");
        }
        let child = child.to_str().context("invalid MCP output path")?;
        if input.starts_with(root) {
            bail!("MCP input must be outside disposable output storage");
        }
        let (actual, fd) = d::inspect(&proof.config, env.as_ref()).map_err(err)?;
        if actual != proof {
            bail!("mcp_destination_mismatch");
        }
        d::writable(&fd, 1).map_err(err)?;
        let cache = open_dir(&fd, child, true)?;
        if cache.metadata()?.dev() != proof.device
            || d::fsid(&cache).map_err(err)? != proof.config.fsid
        {
            bail!("mcp_volume_mismatch");
        }
        d::writable(&cache, 1).map_err(err)?;
        let lock_fd = unsafe {
            libc::openat(
                cache.as_raw_fd(),
                c".mcp.lock".as_ptr(),
                libc::O_CREAT | libc::O_RDWR | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if lock_fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let lock = unsafe { File::from_raw_fd(lock_fd) };
        lock.try_lock_exclusive()
            .context("MCP output directory is already in use")?;
        let session = format!("session-{}", uuid::Uuid::new_v4().simple());
        let root_fd = open_dir(&cache, &session, true)?;
        let cache_path = root.to_path_buf();
        let root = root.join(&session);
        let mut dirs = HashMap::new();
        dirs.insert(
            root.clone(),
            OwnedDirectory {
                fd: root_fd,
                files: HashSet::new(),
            },
        );
        Ok(Self {
            root,
            proof,
            env,
            cache,
            cache_path,
            directories: Mutex::new(dirs),
            _lock: lock,
        })
    }
    pub fn persistent_root(&self) -> &Path {
        &self.proof.config.root
    }
    pub fn preflight(&self, required: u64) -> Result<()> {
        let (actual, fd) = d::inspect(&self.proof.config, self.env.as_ref()).map_err(err)?;
        if actual != self.proof {
            bail!("mcp_destination_mismatch");
        }
        d::writable(&fd, required).map_err(err)?;
        if !same(&self.cache, &d::directory(&self.cache_path).map_err(err)?)? {
            bail!("mcp_cache_changed");
        }
        let dirs = self.directories.lock().unwrap();
        let root = &dirs[&self.root].fd;
        if !same(root, &d::directory(&self.root).map_err(err)?)? {
            bail!("mcp_cache_changed");
        }
        d::writable(root, required).map_err(err)
    }
    pub fn directory(&self, label: &str) -> Result<PathBuf> {
        self.preflight(1)?;
        name(label)?;
        let path = self.root.join(label);
        let mut dirs = self.directories.lock().unwrap();
        if let Some(existing) = dirs.get(&path) {
            if !same(&existing.fd, &d::directory(&path).map_err(err)?)? {
                bail!("mcp_cache_changed");
            }
            return Ok(path);
        }
        let fd = open_dir(&dirs[&self.root].fd, label, true)?;
        if fd.metadata()?.dev() != self.proof.device
            || d::fsid(&fd).map_err(err)? != self.proof.config.fsid
        {
            bail!("mcp_volume_mismatch");
        }
        dirs.insert(
            path.clone(),
            OwnedDirectory {
                fd,
                files: HashSet::new(),
            },
        );
        Ok(path)
    }
    pub fn save(&self, directory: &Path, filename: &str, bytes: &[u8]) -> Result<PathBuf> {
        self.preflight(bytes.len() as u64)?;
        name(filename)?;
        let mut dirs = self.directories.lock().unwrap();
        let dir = dirs
            .get_mut(directory)
            .context("unknown MCP cache directory")?;
        if !same(&dir.fd, &d::directory(directory).map_err(err)?)? {
            bail!("mcp_cache_changed");
        }
        d::writable(&dir.fd, bytes.len() as u64).map_err(err)?;
        let temp = format!(".{}.part", uuid::Uuid::new_v4().simple());
        let n = name(&temp)?;
        let fd = unsafe {
            libc::openat(
                dir.fd.as_raw_fd(),
                n.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let result: Result<()> = (|| {
            write_bytes(&mut file, bytes)?;
            file.sync_all()?;
            if unsafe {
                libc::renameat2(
                    dir.fd.as_raw_fd(),
                    n.as_ptr(),
                    dir.fd.as_raw_fd(),
                    name(filename)?.as_ptr(),
                    libc::RENAME_NOREPLACE,
                )
            } != 0
            {
                return Err(std::io::Error::last_os_error().into());
            }
            Ok(())
        })();
        if result.is_err() {
            let _ = unlink(&dir.fd, &temp, 0);
        } else {
            dir.files.insert(filename.into());
        }
        result?;
        Ok(directory.join(filename))
    }
    pub fn read(&self, path: &Path) -> Result<Vec<u8>> {
        self.preflight(0)?;
        let directory = path.parent().context("invalid MCP cache path")?;
        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("invalid MCP cache name")?;
        let dirs = self.directories.lock().unwrap();
        let dir = dirs.get(directory).context("unknown MCP cache directory")?;
        if !dir.files.contains(filename) || !same(&dir.fd, &d::directory(directory).map_err(err)?)?
        {
            bail!("unowned or changed MCP cache");
        }
        let fd = unsafe {
            libc::openat(
                dir.fd.as_raw_fd(),
                name(filename)?.as_ptr(),
                libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        if !file.metadata()?.is_file() {
            bail!("invalid MCP cache file");
        }
        let mut bytes = Vec::new();
        file.take((crate::references::LIMIT + 1) as u64)
            .read_to_end(&mut bytes)?;
        if bytes.is_empty() || bytes.len() > crate::references::LIMIT {
            bail!("invalid MCP cache size");
        }
        Ok(bytes)
    }
    pub fn remove(&self, directory: &Path, filename: &str) -> Result<()> {
        self.preflight(0)?;
        let mut dirs = self.directories.lock().unwrap();
        let Some(dir) = dirs.get_mut(directory) else {
            return Ok(());
        };
        if !same(&dir.fd, &d::directory(directory).map_err(err)?)? {
            bail!("mcp_cache_changed");
        }
        if !dir.files.contains(filename) {
            bail!("unowned MCP cache file");
        }
        unlink(&dir.fd, filename, 0)?;
        dir.files.remove(filename);
        if dir.files.is_empty() && directory != self.root {
            unlink(
                &dirs[&self.root].fd,
                directory.file_name().unwrap().to_str().unwrap(),
                libc::AT_REMOVEDIR,
            )?;
            dirs.remove(directory);
        }
        Ok(())
    }
    pub fn validate_saved_path(&self, path: &Path) -> Result<()> {
        self.preflight(0)?;
        if !path.starts_with(&self.proof.config.root) || path.starts_with(&self.cache_path) {
            bail!("saved_path must be on the verified media destination outside MCP cache");
        }
        let parent = d::directory(path.parent().context("invalid saved_path")?).map_err(err)?;
        if parent.metadata()?.dev() != self.proof.device
            || d::fsid(&parent).map_err(err)? != self.proof.config.fsid
        {
            bail!("mcp_volume_mismatch");
        }
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("invalid saved_path")?;
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name(filename)?.as_ptr(),
                libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let file = unsafe { File::from_raw_fd(fd) };
        if !file.metadata()?.is_file() || file.metadata()?.dev() != self.proof.device {
            bail!("saved_path must be a regular file on the verified volume");
        }
        Ok(())
    }
    pub fn clear(&self) -> Result<()> {
        // Only names recorded in this process; no read_dir, recursive deletion or old cache access.
        self.preflight(0)?;
        let mut dirs = self.directories.lock().unwrap();
        let paths = dirs
            .keys()
            .filter(|p| **p != self.root)
            .cloned()
            .collect::<Vec<_>>();
        for path in paths {
            let dir = &dirs[&path];
            if !same(&dir.fd, &d::directory(&path).map_err(err)?)? {
                bail!("mcp_cache_changed");
            }
            for file in &dir.files {
                unlink(&dir.fd, file, 0)?;
            }
            unlink(
                &dirs[&self.root].fd,
                path.file_name().unwrap().to_str().unwrap(),
                libc::AT_REMOVEDIR,
            )?;
            dirs.remove(&path);
        }
        Ok(())
    }
}
impl Drop for Storage {
    fn drop(&mut self) {
        if self.clear().is_ok() {
            let _ = unlink(
                &self.cache,
                self.root.file_name().unwrap().to_str().unwrap(),
                libc::AT_REMOVEDIR,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (tempfile::TempDir, Proof) {
        let dir = tempfile::tempdir().unwrap();
        assert!(
            dir.path()
                .to_string_lossy()
                .starts_with("/run/media/veracrypt1/imagine/.codex-")
        );
        let config = d::Config {
            root: dir.path().into(),
            mount: "/run/media/veracrypt1".into(),
            source: "/dev/mapper/veracrypt1".into(),
            filesystem: "ext3".into(),
            fsid: d::fsid(&d::directory(dir.path()).unwrap()).unwrap(),
        };
        let proof = d::inspect(&config, &System).unwrap().0;
        (dir, proof)
    }
    struct MutableEnv(Mutex<String>);
    impl Environment for MutableEnv {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            System.host()
        }
    }
    #[test]
    fn old_cache_and_originals_are_never_scanned_or_removed() {
        let (dir, proof) = fixture();
        let root = dir.path().join("cache");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("historical.png"), b"own historical fixture").unwrap();
        let storage = Storage::open(&root, &dir.path().join("input"), proof.clone()).unwrap();
        assert!(Storage::open(&root, &dir.path().join("input"), proof).is_err());
        let job = storage.directory("job-one").unwrap();
        let file = storage.save(&job, "0.png", b"own new fixture").unwrap();
        let session = storage.root.clone();
        storage.clear().unwrap();
        assert!(!file.exists());
        assert!(root.join("historical.png").exists());
        let again = storage.directory("job-two").unwrap();
        storage
            .save(&again, "0.png", b"own after session reset")
            .unwrap();
        drop(storage);
        assert!(!session.exists());
        assert!(root.join("historical.png").exists());
    }
    #[test]
    fn missing_wrong_readonly_mount_and_outside_config_create_nothing() {
        let (dir, proof) = fixture();
        let root = dir.path().join("cache");
        let env = Arc::new(MutableEnv(Mutex::new(String::new())));
        for mount in [
            "",
            "1 0 0:1 / /run/media/veracrypt1 rw - ext3 other rw",
            "1 0 0:1 / /run/media/veracrypt1 ro - ext3 /dev/mapper/veracrypt1 rw",
        ] {
            *env.0.lock().unwrap() = mount.into();
            assert!(
                Storage::with_environment(
                    &root,
                    &dir.path().join("input"),
                    proof.clone(),
                    env.clone()
                )
                .is_err()
            );
            assert!(!root.exists());
        }
        let outside = dir
            .path()
            .parent()
            .unwrap()
            .join(format!("outside-{}", uuid::Uuid::new_v4()));
        assert!(Storage::open(&outside, &dir.path().join("input"), proof.clone()).is_err());
        assert!(!outside.exists());
        let mut wrong = proof.clone();
        wrong.host.push_str("wrong");
        assert!(Storage::open(&root, dir.path(), wrong).is_err());
        assert!(!root.exists());
        let mut wrong = proof;
        wrong.config.fsid[0] ^= 1;
        assert!(Storage::open(&root, dir.path(), wrong).is_err());
        assert!(!root.exists());
    }
    #[test]
    fn detached_mount_blocks_new_writes_without_plaintext_fallback() {
        let (dir, proof) = fixture();
        let env = Arc::new(MutableEnv(Mutex::new(System.mounts().unwrap())));
        let storage =
            Storage::with_environment(&dir.path().join("cache"), dir.path(), proof, env.clone())
                .unwrap();
        let job = storage.directory("job-one").unwrap();
        *env.0.lock().unwrap() = String::new();
        assert!(storage.save(&job, "0.png", b"own blocked fixture").is_err());
        assert!(!job.join("0.png").exists());
        assert!(storage.directory("job-blocked").is_err());
        assert!(!storage.root.join("job-blocked").exists());
        *env.0.lock().unwrap() = System.mounts().unwrap();
        storage.clear().unwrap();
    }
    #[test]
    fn previews_retain_identity_and_changed_files_reject() {
        let (dir, proof) = fixture();
        let storage = Storage::open(&dir.path().join("cache"), dir.path(), proof).unwrap();
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(8, 8)
            .write_to(&mut png, image::ImageFormat::Png)
            .unwrap();
        let (bytes, w, h) = crate::references::normalize(png.get_ref()).unwrap();
        assert_eq!((w, h), (8, 8));
        let (path, digest) = crate::references::store_preview(&storage, &bytes).unwrap();
        assert_eq!(
            crate::references::store_preview(&storage, &bytes)
                .unwrap()
                .0,
            path
        );
        let reference = crate::references::Reference {
            workflow: "qwen".into(),
            path: path.clone(),
            asset: "fixture".into(),
            width: w,
            height: h,
            digest,
        };
        reference.verify(&storage).unwrap();
        std::fs::write(&path, b"own changed fixture").unwrap();
        assert!(reference.verify(&storage).is_err());
        assert!(crate::references::store_preview(&storage, &bytes).is_err());
        storage.clear().unwrap();
        assert!(!path.exists());
    }
    #[test]
    fn competing_outputs_never_overwrite_and_symlinks_fail_closed() {
        let (dir, proof) = fixture();
        let storage =
            Arc::new(Storage::open(&dir.path().join("cache"), dir.path(), proof).unwrap());
        let job = storage.directory("job-one").unwrap();
        let handles = (0..4)
            .map(|_| {
                let s = storage.clone();
                let j = job.clone();
                std::thread::spawn(move || s.save(&j, "0.png", b"synthetic").is_ok())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| h.join().unwrap() as u32)
                .sum::<u32>(),
            1
        );
        let moved = dir.path().join("moved");
        std::fs::rename(&job, &moved).unwrap();
        std::os::unix::fs::symlink(&moved, &job).unwrap();
        assert!(storage.save(&job, "1.png", b"synthetic").is_err());
        assert!(storage.remove(&job, "0.png").is_err());
        assert!(storage.clear().is_err());
        assert!(moved.join("0.png").exists());
        std::fs::remove_file(&job).unwrap();
        std::fs::rename(&moved, &job).unwrap();
        storage.clear().unwrap();
    }
    #[test]
    fn full_and_unwritable_destination_reject_before_output() {
        use std::os::unix::fs::PermissionsExt;
        let (dir, proof) = fixture();
        let storage = Storage::open(&dir.path().join("cache"), dir.path(), proof).unwrap();
        assert!(
            storage
                .preflight(u64::MAX)
                .unwrap_err()
                .to_string()
                .contains("full")
        );
        std::fs::set_permissions(&storage.root, std::fs::Permissions::from_mode(0o500)).unwrap();
        assert!(storage.directory("blocked").is_err());
        std::fs::set_permissions(&storage.root, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(!storage.root.join("blocked").exists());
    }
    #[test]
    fn simulated_write_failures_are_explicit() {
        struct Failure(i32);
        impl Write for Failure {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::from_raw_os_error(self.0))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for (code, label) in [
            (libc::ENOSPC, "mcp_destination_full"),
            (libc::EDQUOT, "mcp_destination_full"),
            (libc::EACCES, "mcp_destination_not_writable"),
            (libc::EROFS, "mcp_destination_read_only"),
        ] {
            assert_eq!(
                write_bytes(&mut Failure(code), b"synthetic")
                    .unwrap_err()
                    .to_string(),
                label
            );
        }
    }
}
