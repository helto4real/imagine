//! Session-only storage: in-memory history and disposable media files.
use anyhow::Result;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct Session {
    pub id: String,
    pub media: PathBuf,
}
impl Session {
    pub fn new(data: &Path) -> Result<Arc<Self>> {
        let id = uuid::Uuid::new_v4().to_string();
        let media = data.join("media").join(&id);
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&media)?;
        Ok(Arc::new(Self { id, media }))
    }
    pub fn clear(&self) -> Result<()> {
        remove_tree(&self.media)
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.clear();
    }
}
fn remove_tree(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            std::fs::remove_file(path)?;
        }
        Ok(_) => std::fs::remove_dir_all(path)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    Ok(())
}
/// Call only while holding the data-directory server lock. Never opens history.
pub fn purge(data: &Path) -> Result<()> {
    for name in [
        "imagine.sqlite",
        "imagine.sqlite-wal",
        "imagine.sqlite-shm",
        "imagine.sqlite-journal",
    ] {
        match std::fs::remove_file(data.join(name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    remove_tree(&data.join("media"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sessions_are_isolated_and_remove_only_their_own_media() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::new(dir.path()).unwrap();
        let other = Session::new(dir.path()).unwrap();
        assert_ne!(session.id, other.id);
        let path = session.media.clone();
        let kept = other.media.join("asset_b");
        std::fs::write(path.join("asset_a"), b"synthetic image").unwrap();
        std::fs::write(&kept, b"synthetic image").unwrap();
        drop(session);
        assert!(!path.exists());
        assert!(kept.exists());
        drop(other);
        assert!(!kept.exists());
    }
    #[test]
    fn purge_removes_history_without_following_links_or_removing_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("keep"), b"synthetic").unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("media")).unwrap();
        for file in [
            "imagine.sqlite",
            "imagine.sqlite-wal",
            "access.json",
            "privacy-keystore.json",
        ] {
            std::fs::write(dir.path().join(file), b"synthetic").unwrap();
        }
        purge(dir.path()).unwrap();
        assert!(!dir.path().join("imagine.sqlite").exists());
        assert!(!dir.path().join("imagine.sqlite-wal").exists());
        assert!(outside.path().join("keep").exists());
        assert!(dir.path().join("access.json").exists());
        assert!(dir.path().join("privacy-keystore.json").exists());
    }
}
