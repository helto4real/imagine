//! Read-only proof of Imagine's persistent destination. MCP owns no media copies.
use crate::media_destination::{self as d, Environment, Proof, System};
use anyhow::{Context, Result, anyhow, bail};
use std::{
    ffi::CString,
    fs::File,
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Component, Path},
    sync::Arc,
};

pub struct Storage {
    proof: Proof,
    env: Arc<dyn Environment>,
}
fn err(code: &'static str) -> anyhow::Error {
    anyhow!(code)
}
impl Storage {
    pub fn open(proof: Proof) -> Result<Self> {
        Self::with_environment(proof, Arc::new(System))
    }
    pub fn with_environment(proof: Proof, env: Arc<dyn Environment>) -> Result<Self> {
        let storage = Self { proof, env };
        storage.preflight(0)?;
        Ok(storage)
    }
    pub fn persistent_root(&self) -> &Path {
        &self.proof.config.root
    }
    pub fn preflight(&self, required: u64) -> Result<()> {
        let (actual, file) = d::inspect(&self.proof.config, self.env.as_ref()).map_err(err)?;
        if actual != self.proof {
            bail!("mcp_destination_mismatch");
        }
        d::writable(&file, required).map_err(err)
    }
    /// Acknowledgements reference the already published server file. This only
    /// checks the existing destination contract; it never writes, copies or unlinks.
    pub fn validate_saved_path(&self, path: &Path) -> Result<()> {
        self.preflight(0)?;
        if !path.is_absolute()
            || !path.starts_with(self.persistent_root())
            || path
                .components()
                .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
        {
            bail!("saved_path must be on the verified media destination");
        }
        let parent = d::directory(path.parent().context("invalid saved_path")?).map_err(err)?;
        if parent.metadata()?.dev() != self.proof.device
            || d::fsid(&parent).map_err(err)? != self.proof.config.fsid
        {
            bail!("mcp_volume_mismatch");
        }
        let name = CString::new(
            path.file_name()
                .context("invalid saved_path")?
                .as_encoded_bytes(),
        )?;
        let fd = unsafe {
            libc::openat(
                parent.as_raw_fd(),
                name.as_ptr(),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    struct MutableEnv(Mutex<String>);
    impl Environment for MutableEnv {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            System.host()
        }
    }
    fn fixture() -> (tempfile::TempDir, Proof) {
        let dir = tempfile::tempdir().unwrap();
        let proof = serde_json::from_value(crate::mcp::test_destination(dir.path())).unwrap();
        (dir, proof)
    }
    #[test]
    fn existing_outputs_and_historical_cache_remain_untouched_without_new_cache() {
        let (dir, proof) = fixture();
        let historical = dir.path().join("historical-cache");
        std::fs::create_dir(&historical).unwrap();
        std::fs::write(historical.join("own.png"), b"own old fixture").unwrap();
        let output = dir.path().join("own-published.png");
        std::fs::write(&output, b"own published fixture").unwrap();
        let before = std::fs::read_dir(dir.path()).unwrap().count();
        let storage = Storage::open(proof.clone()).unwrap();
        storage.validate_saved_path(&output).unwrap();
        Storage::open(proof)
            .unwrap()
            .validate_saved_path(&output)
            .unwrap();
        drop(storage);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), before);
        assert_eq!(std::fs::read(&output).unwrap(), b"own published fixture");
        assert_eq!(
            std::fs::read(historical.join("own.png")).unwrap(),
            b"own old fixture"
        );
    }
    #[test]
    fn detached_readonly_or_wrong_volume_rejects_without_fallback() {
        let (dir, proof) = fixture();
        let mounts = System.mounts().unwrap();
        let env = Arc::new(MutableEnv(Mutex::new(mounts.clone())));
        let storage = Storage::with_environment(proof.clone(), env.clone()).unwrap();
        *env.0.lock().unwrap() = String::new();
        assert!(storage.preflight(0).is_err());
        *env.0.lock().unwrap() = mounts.replace(" rw", " ro");
        assert!(storage.preflight(0).is_err());
        *env.0.lock().unwrap() = mounts;
        let mut wrong = proof;
        wrong.config.fsid[0] ^= 1;
        assert!(Storage::with_environment(wrong, env).is_err());
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
    #[test]
    fn only_existing_regular_published_paths_are_accepted() {
        let (dir, proof) = fixture();
        let output = dir.path().join("own.png");
        std::fs::write(&output, b"own").unwrap();
        let link = dir.path().join("own-link");
        std::os::unix::fs::symlink(&output, &link).unwrap();
        let storage = Storage::open(proof).unwrap();
        assert!(storage.validate_saved_path(&output).is_ok());
        for invalid in [
            link,
            dir.path().to_path_buf(),
            dir.path().join("missing"),
            dir.path().join("../outside.png"),
        ] {
            assert!(storage.validate_saved_path(&invalid).is_err());
        }
    }
}
