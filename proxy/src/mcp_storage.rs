//! Exclusive, disposable MCP export storage. Never reads files during cleanup.
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use std::{
    fs::{File, OpenOptions},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
};

pub struct Storage {
    pub root: PathBuf,
    _lock: File,
}
impl Storage {
    pub fn open(root: &Path, input: &Path) -> Result<Self> {
        let absolute = if root.is_absolute() {
            root.to_path_buf()
        } else {
            std::env::current_dir()?.join(root)
        };
        let protected = [
            Some(std::env::current_dir()?),
            std::env::var_os("HOME").map(PathBuf::from),
        ];
        if protected
            .into_iter()
            .flatten()
            .any(|path| path.starts_with(&absolute))
        {
            bail!("MCP output cannot contain the working directory or home directory");
        }
        if root.is_symlink() {
            bail!("MCP output root must not be a symlink");
        }
        crate::references::private_dir(root)?;
        let root = root.canonicalize()?;
        if std::env::current_dir()?.canonicalize()?.starts_with(&root)
            || std::env::var_os("HOME")
                .map(PathBuf::from)
                .and_then(|p| p.canonicalize().ok())
                .is_some_and(|p| p.starts_with(&root))
        {
            bail!("MCP output cannot contain the working directory or home directory");
        }
        let input = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
        if input.starts_with(&root) {
            bail!("MCP input must be outside disposable output storage");
        }
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(root.join(".mcp.lock"))?;
        lock.try_lock_exclusive()
            .context("MCP output directory is already in use")?;
        let storage = Self { root, _lock: lock };
        storage.clear()?;
        Ok(storage)
    }
    pub fn clear(&self) -> Result<()> {
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_name() == ".mcp.lock" {
                continue;
            }
            if entry.file_type()?.is_dir() {
                std::fs::remove_dir_all(entry.path())?;
            } else {
                std::fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
}
impl Drop for Storage {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exclusive_cleanup_does_not_follow_links_or_delete_copies() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("output");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(dir.path().join("saved.png"), b"synthetic").unwrap();
        std::os::unix::fs::symlink(dir.path().join("saved.png"), root.join("ref.png")).unwrap();
        let storage = Storage::open(&root, &dir.path().join("input")).unwrap();
        assert!(!root.join("ref.png").exists());
        assert!(Storage::open(&root, &dir.path().join("input")).is_err());
        std::fs::write(root.join("temporary.png"), b"synthetic").unwrap();
        drop(storage);
        assert!(!root.join("temporary.png").exists());
        assert!(dir.path().join("saved.png").exists());
    }
}
