//! Exclusive ownership and stale-file recovery for the Imagine coordinator socket.
use anyhow::{Context, Result, bail};
use fs2::FileExt;
use std::{
    fs::{File, Metadata, OpenOptions},
    io::ErrorKind,
    os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    time::Duration,
};
use tokio::net::{UnixListener, UnixStream};

pub struct BoundSocket {
    pub listener: UnixListener,
    path: PathBuf,
    identity: (u64, u64),
    // Keep the sidecar inode on disk, and hold its lock until the listener drops.
    _lock: File,
}
fn identity(metadata: &Metadata) -> (u64, u64) {
    (metadata.dev(), metadata.ino())
}
impl BoundSocket {
    pub async fn bind(path: &Path) -> Result<Self> {
        let name = path
            .file_name()
            .context("Coordinator socket needs a filename")?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let path = parent.canonicalize()?.join(name);
        let mut lock_path = path.as_os_str().to_owned();
        lock_path.push(".lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(lock_path)?;
        let metadata = lock.metadata()?;
        if !metadata.is_file() || metadata.uid() != unsafe { libc::geteuid() } {
            bail!("Coordinator socket lock must be a regular file owned by this user");
        }
        lock.try_lock_exclusive()
            .context("Imagine coordinator socket is already owned by another proxy")?;
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if !metadata.file_type().is_socket() || metadata.uid() != unsafe { libc::geteuid() }
                {
                    bail!("Refusing to replace a non-socket or another user's coordinator socket");
                }
                match tokio::time::timeout(Duration::from_secs(1), UnixStream::connect(&path)).await
                {
                    Ok(Err(error)) if error.kind() == ErrorKind::ConnectionRefused => {}
                    _ => bail!("Imagine coordinator socket is active or cannot safely be checked"),
                }
                // A non-cooperating process may have changed the path during the probe.
                if identity(&std::fs::symlink_metadata(&path)?) != identity(&metadata) {
                    bail!("Imagine coordinator socket changed during startup; retry");
                }
                std::fs::remove_file(&path)?;
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let listener = UnixListener::bind(&path).with_context(|| {
            format!(
                "Could not bind Imagine coordinator socket {}",
                path.display()
            )
        })?;
        let identity = identity(&std::fs::symlink_metadata(&path)?);
        let bound = Self {
            listener,
            path,
            identity,
            _lock: lock,
        };
        std::fs::set_permissions(&bound.path, std::fs::Permissions::from_mode(0o600))?;
        Ok(bound)
    }
}
impl Drop for BoundSocket {
    fn drop(&mut self) {
        // Cancellation, startup errors and normal shutdown remove only our socket.
        if std::fs::symlink_metadata(&self.path)
            .is_ok_and(|m| m.file_type().is_socket() && identity(&m) == self.identity)
        {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn stale_socket_is_replaced_and_restart_works() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.sock");
        drop(UnixListener::bind(&path).unwrap());
        assert!(path.exists());
        let bound = BoundSocket::bind(&path).await.unwrap();
        let client = UnixStream::connect(&path).await.unwrap();
        let _ = bound.listener.accept().await.unwrap();
        drop(client);
        drop(bound);
        assert!(!path.exists());
        assert!(BoundSocket::bind(&path).await.is_ok());
    }
    #[tokio::test]
    async fn active_legacy_listener_is_never_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let before = identity(&std::fs::symlink_metadata(&path).unwrap());
        assert!(BoundSocket::bind(&path).await.is_err());
        assert_eq!(identity(&std::fs::symlink_metadata(&path).unwrap()), before);
        // The probe connected but sent no protocol data. The original still accepts.
        let _ = listener.accept().await.unwrap();
        let _client = UnixStream::connect(&path).await.unwrap();
        let _ = listener.accept().await.unwrap();
    }
    #[tokio::test]
    async fn simultaneous_proxies_cannot_steal_ownership() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.sock");
        let (first, second) = tokio::join!(BoundSocket::bind(&path), BoundSocket::bind(&path));
        assert_ne!(first.is_ok(), second.is_ok());
        let owner = first.or(second).unwrap();
        let _client = UnixStream::connect(&path).await.unwrap();
        let _ = owner.listener.accept().await.unwrap();
    }
    #[tokio::test]
    async fn regular_files_and_symlinks_are_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.sock");
        std::fs::write(&path, b"synthetic sentinel").unwrap();
        assert!(BoundSocket::bind(&path).await.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"synthetic sentinel");
        std::fs::remove_file(&path).unwrap();
        let target = dir.path().join("target.sock");
        drop(UnixListener::bind(&target).unwrap());
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(BoundSocket::bind(&path).await.is_err());
        assert!(path.is_symlink());
        assert!(target.exists());
    }
    #[tokio::test]
    async fn cleanup_does_not_remove_a_replacement_listener() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.sock");
        let old = BoundSocket::bind(&path).await.unwrap();
        std::fs::remove_file(&path).unwrap();
        let replacement = UnixListener::bind(&path).unwrap();
        drop(old);
        let _client = UnixStream::connect(&path).await.unwrap();
        let _ = replacement.accept().await.unwrap();
    }
    #[tokio::test]
    async fn aborted_listener_task_removes_its_socket() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gpu.sock");
        let bound = BoundSocket::bind(&path).await.unwrap();
        let task = tokio::spawn(async move {
            let _ = bound.listener.accept().await;
            drop(bound);
        });
        task.abort();
        let _ = task.await;
        assert!(!path.exists());
        assert!(BoundSocket::bind(&path).await.is_ok());
    }
}
