//! Read-only proof that Imagine's encrypted destination is available. MCP owns no media copies.
use crate::media_destination::{self as d, Environment, Proof, System};
use anyhow::{Result, anyhow, bail};
use std::sync::Arc;

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
    pub fn preflight(&self, required: u64) -> Result<()> {
        let (actual, file) = d::inspect(&self.proof.config, self.env.as_ref()).map_err(err)?;
        if actual != self.proof {
            bail!("mcp_destination_mismatch");
        }
        d::writable(&file, required).map_err(err)
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
}
