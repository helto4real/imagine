//! Session-only storage: no prompt database on disk, authenticated media encryption.
use aes_gcm::{
    Aes256Gcm, KeyInit, Nonce,
    aead::{Aead, OsRng, rand_core::RngCore},
};
use anyhow::{Result, bail};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use zeroize::Zeroizing;

pub struct Session {
    pub id: String,
    pub media: PathBuf,
    key: Option<Zeroizing<[u8; 32]>>,
}
impl Session {
    pub fn new(data: &Path, encrypted: bool) -> Result<Arc<Self>> {
        let id = uuid::Uuid::new_v4().to_string();
        let media = data.join("media").join(&id);
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&media)?;
        let key = encrypted.then(|| {
            let mut key = Zeroizing::new([0; 32]);
            OsRng.fill_bytes(key.as_mut());
            key
        });
        Ok(Arc::new(Self { id, media, key }))
    }
    pub fn encrypted(&self) -> bool {
        self.key.is_some()
    }
    pub fn encode(&self, id: &str, bytes: &[u8]) -> Result<Vec<u8>> {
        let Some(key) = &self.key else {
            return Ok(bytes.to_vec());
        };
        let cipher = Aes256Gcm::new_from_slice(key.as_ref()).unwrap();
        let mut nonce = [0; 12];
        OsRng.fill_bytes(&mut nonce);
        let payload = aes_gcm::aead::Payload {
            msg: bytes,
            aad: id.as_bytes(),
        };
        let encrypted = cipher
            .encrypt(Nonce::from_slice(&nonce), payload)
            .map_err(|_| anyhow::anyhow!("media_encrypt_failed"))?;
        Ok([b"IMAGINE1".as_slice(), &nonce, &encrypted].concat())
    }
    pub fn decode(&self, id: &str, bytes: &[u8]) -> Result<Vec<u8>> {
        let Some(key) = &self.key else {
            return Ok(bytes.to_vec());
        };
        if bytes.len() < 36 || &bytes[..8] != b"IMAGINE1" {
            bail!("invalid_media_envelope");
        }
        Aes256Gcm::new_from_slice(key.as_ref())
            .unwrap()
            .decrypt(
                Nonce::from_slice(&bytes[8..20]),
                aes_gcm::aead::Payload {
                    msg: &bytes[20..],
                    aad: id.as_bytes(),
                },
            )
            .map_err(|_| anyhow::anyhow!("media_decrypt_failed"))
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
    fn encrypted_media_authenticates_identity_and_session_and_cleans_up() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::new(dir.path(), true).unwrap();
        let bytes = session
            .encode("asset_a", b"synthetic private image")
            .unwrap();
        assert!(!bytes.windows(9).any(|w| w == b"synthetic"));
        assert_eq!(
            session.decode("asset_a", &bytes).unwrap(),
            b"synthetic private image"
        );
        assert!(session.decode("asset_b", &bytes).is_err());
        let other = Session::new(dir.path(), true).unwrap();
        assert!(other.decode("asset_a", &bytes).is_err());
        let mut corrupt = bytes.clone();
        corrupt[20] ^= 1;
        assert!(session.decode("asset_a", &corrupt).is_err());
        let path = session.media.clone();
        std::fs::write(path.join("asset_a"), bytes).unwrap();
        drop(session);
        assert!(!path.exists());
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
