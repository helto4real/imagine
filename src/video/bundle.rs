//! Private filesystem mechanics shared by project and library storage.
use super::persistence::{BlobKey, Resource, StorageError};
use fs2::FileExt;
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::CString,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Component, Path, PathBuf},
};

fn io(_: std::io::Error) -> StorageError {
    StorageError::new("storage_io_failed")
}
fn c(value: &str) -> Result<CString, StorageError> {
    CString::new(value).map_err(|_| StorageError::new("invalid_storage_name"))
}
fn fd(result: libc::c_int) -> Result<File, StorageError> {
    if result < 0 {
        Err(io(std::io::Error::last_os_error()))
    } else {
        Ok(unsafe { File::from_raw_fd(result) })
    }
}
/// Root path identity is rechecked on every operation; all children use dirfd+NOFOLLOW.
#[derive(Debug)]
pub(crate) struct Directory {
    path: PathBuf,
    file: File,
    device: u64,
    inode: u64,
    #[cfg(test)]
    fail_sync_after_rename: bool,
}
/// End the transaction's flock explicitly. A descriptor inherited by a concurrently
/// spawned child must not keep a completed transaction locked until exec/exit.
pub(crate) struct DirectoryLock {
    file: File,
}
impl Drop for DirectoryLock {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.file);
    }
}
impl Directory {
    pub(crate) fn select(path: &Path) -> Result<Self, StorageError> {
        let path = std::path::absolute(path).map_err(io)?;
        let mut walked = PathBuf::new();
        for component in path.components() {
            match component {
                Component::RootDir | Component::Normal(_) => walked.push(component),
                _ => return Err(StorageError::new("invalid_storage_root")),
            }
            let meta = fs::symlink_metadata(&walked).map_err(io)?;
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(StorageError::new("invalid_storage_root"));
            }
        }
        let name = c(path
            .to_str()
            .ok_or_else(|| StorageError::new("invalid_storage_root"))?)?;
        let file = fd(unsafe {
            libc::open(
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        })?;
        let meta = file.metadata().map_err(io)?;
        let root = Self {
            path,
            file,
            device: meta.dev(),
            inode: meta.ino(),
            #[cfg(test)]
            fail_sync_after_rename: false,
        };
        root.check()?;
        Ok(root)
    }
    pub(crate) fn check(&self) -> Result<(), StorageError> {
        let mut walked = PathBuf::new();
        for component in self.path.components() {
            walked.push(component);
            let meta = fs::symlink_metadata(&walked).map_err(io)?;
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(StorageError::new("storage_root_changed"));
            }
        }
        let meta = fs::symlink_metadata(&self.path).map_err(io)?;
        if meta.dev() != self.device || meta.ino() != self.inode {
            return Err(StorageError::new("storage_root_changed"));
        }
        Ok(())
    }
    fn open(&self, name: &str, flags: i32, mode: libc::mode_t) -> Result<File, StorageError> {
        self.check()?;
        if name.is_empty() || name.contains('/') || name == "." || name == ".." {
            return Err(StorageError::new("invalid_storage_name"));
        }
        let name = c(name)?;
        let file = fd(unsafe {
            libc::openat(
                self.file.as_raw_fd(),
                name.as_ptr(),
                flags | libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK,
                mode,
            )
        })?;
        let meta = file.metadata().map_err(io)?;
        if !meta.is_file() || meta.nlink() != 1 {
            return Err(StorageError::new("invalid_storage_file"));
        }
        Ok(file)
    }
    pub(crate) fn read(&self, name: &str) -> Result<File, StorageError> {
        self.open(name, libc::O_RDONLY, 0)
    }
    fn create(&self, name: &str) -> Result<File, StorageError> {
        self.open(name, libc::O_RDWR | libc::O_CREAT | libc::O_EXCL, 0o600)
    }
    fn unlink(&self, name: &str) -> Result<(), StorageError> {
        self.check()?;
        let name = c(name)?;
        if unsafe { libc::unlinkat(self.file.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err(io(std::io::Error::last_os_error()));
        }
        Ok(())
    }
    fn sync(&self) -> Result<(), StorageError> {
        self.file.sync_all().map_err(io)
    }
    pub(crate) fn lock(&self) -> Result<DirectoryLock, StorageError> {
        self.lock_named(".imagine-lock")
    }
    fn lock_named(&self, name: &str) -> Result<DirectoryLock, StorageError> {
        let file = self.open(name, libc::O_RDWR | libc::O_CREAT, 0o600)?;
        file.try_lock_exclusive()
            .map_err(|_| StorageError::new("storage_busy"))?;
        if unsafe { libc::fchmod(self.file.as_raw_fd(), 0o700) } != 0 {
            return Err(io(std::io::Error::last_os_error()));
        }
        Ok(DirectoryLock { file })
    }
    fn folder(&self, name: &str) -> Result<Self, StorageError> {
        self.check()?;
        let cname = c(name)?;
        let result = unsafe { libc::mkdirat(self.file.as_raw_fd(), cname.as_ptr(), 0o700) };
        if result != 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST) {
            return Err(io(std::io::Error::last_os_error()));
        }
        let folder = Self::select(&self.path.join(name))?;
        self.check()?;
        Ok(folder)
    }
    pub(crate) fn temporary(&self) -> Result<File, StorageError> {
        let name = format!(".staging-{}", uuid::Uuid::new_v4().simple());
        let file = self.create(&name)?;
        self.unlink(&name)?;
        Ok(file)
    }
    fn atomic(&self, name: &str, bytes: &[u8]) -> Result<(), StorageError> {
        let temporary = format!(".manifest-{}", uuid::Uuid::new_v4().simple());
        let mut file = self.create(&temporary)?;
        let result = (|| {
            file.write_all(bytes).map_err(io)?;
            file.sync_all().map_err(io)?;
            self.check()?;
            let from = c(&temporary)?;
            let to = c(name)?;
            if unsafe {
                libc::renameat(
                    self.file.as_raw_fd(),
                    from.as_ptr(),
                    self.file.as_raw_fd(),
                    to.as_ptr(),
                )
            } != 0
            {
                return Err(io(std::io::Error::last_os_error()));
            }
            // The caller handles a durability failure after rename as an uncertain save.
            #[cfg(test)]
            if self.fail_sync_after_rename {
                return Err(StorageError::new("storage_durability_uncertain"));
            }
            self.sync()
                .map_err(|_| StorageError::new("storage_durability_uncertain"))
        })();
        if result.is_err() {
            let _ = self.unlink(&temporary);
        }
        result
    }
}

pub(crate) fn digest(file: &mut File) -> Result<(String, u64), StorageError> {
    let mut hash = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = std::os::unix::fs::FileExt::read_at(file, &mut buffer, size).map_err(io)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        size = size
            .checked_add(n as u64)
            .ok_or_else(|| StorageError::new("resource_too_large"))?;
    }
    Ok((format!("{:x}", hash.finalize()), size))
}
pub(crate) fn stream(source: &File, target: &mut impl Write) -> Result<(), StorageError> {
    let mut offset = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = std::os::unix::fs::FileExt::read_at(source, &mut buffer, offset).map_err(io)?;
        if n == 0 {
            break;
        }
        target.write_all(&buffer[..n]).map_err(io)?;
        offset = offset
            .checked_add(n as u64)
            .ok_or_else(|| StorageError::new("resource_too_large"))?;
    }
    Ok(())
}
pub(crate) fn copy(source: &mut File, target: &mut File) -> Result<(), StorageError> {
    target.seek(SeekFrom::Start(0)).map_err(io)?;
    target.set_len(0).map_err(io)?;
    stream(source, target)?;
    target.sync_all().map_err(io)
}
pub(crate) fn clone_files(
    files: &BTreeMap<BlobKey, File>,
) -> Result<BTreeMap<BlobKey, File>, StorageError> {
    files
        .iter()
        .map(|(k, v)| v.try_clone().map(|f| (k.clone(), f)).map_err(io))
        .collect()
}
pub(crate) trait Bundle: Serialize + DeserializeOwned + Clone {
    fn identity(&self) -> &str;
    fn revision(&self) -> u64;
    fn resources(&self) -> Vec<&Resource>;
    fn validate(&self) -> Result<(), StorageError>;
    fn resource_directory(&self, resource: &Resource) -> &'static str {
        resource.key().bucket.directory()
    }
    fn released_directories(&self, key: &BlobKey) -> Vec<&'static str> {
        vec![key.bucket.directory()]
    }
    fn prepared_directories(&self) -> &'static [&'static str] {
        &[]
    }
}
pub(crate) struct Store<T> {
    root: Directory,
    manifest: &'static str,
    expected: Option<String>,
    marker: std::marker::PhantomData<T>,
    private_root: Option<crate::private_storage::Root>,
    write_error: Option<&'static str>,
    #[cfg(test)]
    pub(crate) fail_commit: bool,
}
impl<T: Bundle> Store<T> {
    #[cfg(test)]
    pub(crate) fn inject_sync_failure_after_rename(&mut self, enabled: bool) {
        self.root.fail_sync_after_rename = enabled;
    }
    pub(crate) fn path(&self) -> &Path {
        &self.root.path
    }
    pub(crate) fn is_at(&self, path: &Path) -> Result<bool, StorageError> {
        if self.root.path != path {
            return Ok(false);
        }
        self.root.check()?;
        Ok(true)
    }
    pub(crate) fn select(path: &Path, manifest: &'static str) -> Result<Self, StorageError> {
        Ok(Self {
            root: Directory::select(path)?,
            manifest,
            expected: None,
            marker: std::marker::PhantomData,
            private_root: None,
            write_error: None,
            #[cfg(test)]
            fail_commit: false,
        })
    }
    pub(crate) fn select_private(
        path: &Path,
        manifest: &'static str,
        root: &crate::private_storage::Root,
    ) -> Result<Self, StorageError> {
        root.proof().map_err(StorageError::new)?;
        if !root.allows(path) {
            return Err(StorageError::new("invalid_storage_root"));
        }
        let mut store = Self::select(path, manifest)?;
        store.private_root = Some(root.clone());
        Ok(store)
    }
    pub(crate) fn guard_private(
        &mut self,
        root: &crate::private_storage::Root,
    ) -> Result<(), StorageError> {
        root.proof().map_err(StorageError::new)?;
        if !root.allows(&self.root.path) {
            return Err(StorageError::new("invalid_storage_root"));
        }
        self.private_root = Some(root.clone());
        self.write_error = None;
        Ok(())
    }
    fn check_private(&self) -> Result<(), StorageError> {
        self.check_writable()?;
        self.private_root.as_ref().map_or(Ok(()), |root| {
            root.proof().map(|_| ()).map_err(StorageError::new)
        })
    }
    pub(crate) fn block_writes(&mut self, code: &'static str) {
        self.write_error = Some(code);
    }
    pub(crate) fn check_writable(&self) -> Result<(), StorageError> {
        self.write_error
            .map_or(Ok(()), |code| Err(StorageError::new(code)))
    }
    fn resource_folder(&self, name: &str, create: bool) -> Result<Directory, StorageError> {
        let mut folder = Directory::select(&self.root.path)?;
        for part in Path::new(name).components() {
            let Component::Normal(part) = part else {
                return Err(StorageError::new("invalid_storage_name"));
            };
            let name = part
                .to_str()
                .ok_or_else(|| StorageError::new("invalid_storage_name"))?;
            folder = if create {
                folder.folder(name)?
            } else {
                Directory::select(&folder.path.join(name))?
            };
        }
        self.root.check()?;
        Ok(folder)
    }
    fn lock(&self) -> Result<DirectoryLock, StorageError> {
        if let Some(root) = &self.private_root {
            use std::os::unix::ffi::OsStrExt;
            let runtime = root.runtime_directory().map_err(StorageError::new)?;
            let directory = Directory::select(&runtime)?;
            let identity = Sha256::digest(self.root.path.as_os_str().as_bytes());
            directory.lock_named(&format!("bundle-{identity:x}.lock"))
        } else {
            self.root.lock()
        }
    }
    fn prepared_directories(&self, document: &T) -> &'static [&'static str] {
        if self
            .private_root
            .as_ref()
            .is_some_and(|root| root.layout().is_named_project(&self.root.path))
        {
            &["outputs", "work", "tmp", "autosave"]
        } else if self
            .private_root
            .as_ref()
            .is_some_and(|root| self.root.path == root.layout().library())
        {
            // Prompt editing and the media library share the decided private
            // metadata root; editing a catalog must not block first library Save.
            &["prompt-catalog"]
        } else {
            document.prepared_directories()
        }
    }
    fn prepared_metadata_files(&self) -> &'static [&'static str] {
        if self
            .private_root
            .as_ref()
            .is_some_and(|root| self.root.path == root.layout().library())
        {
            &[
                "image-workspace.json",
                "enhancement-profiles.json",
                "ui-preferences.json",
            ]
        } else {
            &[]
        }
    }
    pub(crate) fn open(path: &Path, manifest: &'static str) -> Result<(Self, T), StorageError> {
        Self::read_current(Self::select(path, manifest)?)
    }
    pub(crate) fn open_private(
        path: &Path,
        manifest: &'static str,
        root: &crate::private_storage::Root,
    ) -> Result<(Self, T), StorageError> {
        Self::read_current(Self::select_private(path, manifest, root)?)
    }
    pub(crate) fn open_readonly(
        path: &Path,
        manifest: &'static str,
        root: &crate::private_storage::Root,
    ) -> Result<(Self, T), StorageError> {
        root.proof().map_err(StorageError::new)?;
        let mut store = Self::select(path, manifest)?;
        store.private_root = Some(root.clone());
        let (mut store, document) = Self::read_current(store)?;
        store.block_writes("legacy_project_requires_private_copy");
        Ok((store, document))
    }
    fn read_current(mut store: Self) -> Result<(Self, T), StorageError> {
        let _lock = store.lock()?;
        let bytes = store.bytes()?;
        let document: T =
            serde_json::from_slice(&bytes).map_err(|_| StorageError::new("invalid_manifest"))?;
        document.validate()?;
        store.files(&document)?;
        store.expected = Some(format!("{:x}", Sha256::digest(&bytes)));
        Ok((store, document))
    }
    /// Release only named owned blobs after committing the new manifest.
    pub(crate) fn prune_released(&self, released: &[BlobKey]) -> Result<(), StorageError> {
        self.check_private()?;
        let _lock = self.lock()?;
        let bytes = self.bytes()?;
        if self.expected.as_ref() != Some(&format!("{:x}", Sha256::digest(&bytes))) {
            return Err(StorageError::new("storage_conflict"));
        }
        let current: T =
            serde_json::from_slice(&bytes).map_err(|_| StorageError::new("invalid_manifest"))?;
        let live: std::collections::BTreeSet<_> =
            current.resources().iter().map(|r| r.key()).collect();
        for key in released {
            if live.contains(key) || matches!(key.bucket, super::persistence::Bucket::Media) {
                continue;
            }
            for name in current.released_directories(key) {
                let folder = self.resource_folder(name, false)?;
                if folder.path.join(&key.sha256).try_exists().map_err(io)? {
                    folder.unlink(&key.sha256)?;
                    folder.sync()?;
                }
            }
        }
        Ok(())
    }
    fn bytes(&self) -> Result<Vec<u8>, StorageError> {
        let mut bytes = vec![];
        self.root
            .read(self.manifest)?
            .read_to_end(&mut bytes)
            .map_err(io)?;
        Ok(bytes)
    }
    pub(crate) fn files(&self, document: &T) -> Result<BTreeMap<BlobKey, File>, StorageError> {
        let mut files = BTreeMap::new();
        for resource in document.resources() {
            // A link is metadata, not an owned bundle blob. Missing originals do
            // not prevent opening the project to repair them with Relink.
            if resource.source.is_some() {
                continue;
            }
            let key = resource.key();
            let directory = self.resource_folder(document.resource_directory(resource), false)?;
            let mut file = directory.read(&key.sha256)?;
            if digest(&mut file)? != (key.sha256.clone(), resource.bytes) {
                return Err(StorageError::new("resource_integrity_failed"));
            }
            files.insert(key, file);
        }
        Ok(files)
    }
    pub(crate) fn save(
        &mut self,
        document: &T,
        sources: &mut BTreeMap<BlobKey, File>,
    ) -> Result<(), StorageError> {
        self.check_private()?;
        document.validate()?;
        let _lock = self.lock()?;
        match &self.expected {
            Some(expected) => {
                let bytes = self.bytes()?;
                if format!("{:x}", Sha256::digest(&bytes)) != *expected {
                    return Err(StorageError::new("storage_conflict"));
                }
                let previous: T = serde_json::from_slice(&bytes)
                    .map_err(|_| StorageError::new("invalid_manifest"))?;
                if previous.identity() != document.identity()
                    || previous.revision() > document.revision()
                {
                    return Err(StorageError::new("storage_conflict"));
                }
            }
            None => {
                // Explicit first Save requires an otherwise empty dedicated directory.
                for entry in fs::read_dir(&self.root.path).map_err(io)? {
                    let entry = entry.map_err(io)?;
                    let file_type = entry.file_type().map_err(io)?;
                    if entry.file_name() != ".imagine-lock"
                        && !(self
                            .prepared_directories(document)
                            .iter()
                            .any(|name| entry.file_name() == *name)
                            && file_type.is_dir())
                        && !(self
                            .prepared_metadata_files()
                            .iter()
                            .any(|name| entry.file_name() == *name)
                            && file_type.is_file())
                    {
                        return Err(StorageError::new("storage_not_empty"));
                    }
                }
            }
        }
        let mut created = vec![];
        let result = (|| {
            for resource in document.resources() {
                if resource.source.is_some() {
                    continue;
                }
                let key = resource.key();
                let name = document.resource_directory(resource);
                let folder = self.resource_folder(name, true)?;
                // Existing content-addressed resources are immutable and reverified.
                if let Ok(mut existing) = folder.read(&key.sha256) {
                    if digest(&mut existing)? == (key.sha256.clone(), resource.bytes) {
                        continue;
                    }
                    return Err(StorageError::new("resource_integrity_failed"));
                }
                let source = sources
                    .get_mut(&key)
                    .ok_or_else(|| StorageError::new("resource_missing"))?;
                let temporary = format!(".resource-{}", uuid::Uuid::new_v4().simple());
                let mut target = folder.create(&temporary)?;
                let copied = (|| {
                    copy(source, &mut target)?;
                    if digest(&mut target)? != (key.sha256.clone(), resource.bytes) {
                        return Err(StorageError::new("resource_integrity_failed"));
                    }
                    let from = c(&temporary)?;
                    let to = c(&key.sha256)?;
                    // No replacement of an existing blob, including symlinks.
                    if unsafe {
                        libc::linkat(
                            folder.file.as_raw_fd(),
                            from.as_ptr(),
                            folder.file.as_raw_fd(),
                            to.as_ptr(),
                            0,
                        )
                    } != 0
                    {
                        return Err(io(std::io::Error::last_os_error()));
                    }
                    Ok(())
                })();
                let cleaned = folder.unlink(&temporary);
                if copied.is_ok() {
                    created.push((name, key.sha256));
                }
                copied?;
                cleaned?;
                folder.sync()?;
            }
            #[cfg(test)]
            if self.fail_commit {
                return Err(StorageError::new("injected_disk_full"));
            }
            let bytes = serde_json::to_vec_pretty(document)
                .map_err(|_| StorageError::new("invalid_manifest"))?;
            self.check_private()?;
            let result = self.root.atomic(self.manifest, &bytes);
            if result.is_ok()
                || result
                    .as_ref()
                    .is_err_and(|e| e.code == "storage_durability_uncertain")
            {
                self.expected = Some(format!("{:x}", Sha256::digest(&bytes)));
            }
            result
        })();
        if result
            .as_ref()
            .is_err_and(|e| e.code != "storage_durability_uncertain")
        {
            for (directory, name) in created {
                if let Ok(folder) = self.resource_folder(directory, false) {
                    let _ = folder.unlink(&name);
                }
            }
            // Only empty directories created for this first-save attempt may be removed.
            if self.expected.is_none()
                && document.prepared_directories().is_empty()
                && self.root.check().is_ok()
            {
                for name in ["media", "takes", "continuity"] {
                    let _ = fs::remove_dir(self.root.path.join(name));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod lock_tests {
    use super::*;
    #[test]
    fn completed_transaction_releases_lock_while_inherited_descriptor_remains_open() {
        let root = tempfile::tempdir().unwrap();
        let directory = Directory::select(root.path()).unwrap();
        let held = directory.lock().unwrap();
        let inherited = held.file.try_clone().unwrap();
        assert!(directory.lock().is_err());
        drop(held);
        let next = directory
            .lock()
            .expect("a completed transaction must release its lock");
        assert!(directory.lock().is_err());
        drop(inherited);
        assert!(
            directory.lock().is_err(),
            "the new transaction must remain locked"
        );
        drop(next);
        assert!(directory.lock().is_ok());
    }
}
