use anyhow::{Context, Result, bail};
use fs2::FileExt;
use std::{
    fs::{File, OpenOptions},
    net::SocketAddr,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub struct Config {
    pub project: PathBuf,
    pub data: PathBuf,
    pub bind: SocketAddr,
    pub model_root: PathBuf,
}
impl Config {
    pub fn load(project: PathBuf, data: PathBuf, bind: SocketAddr) -> Result<(Self, File)> {
        if !bind.ip().is_loopback() {
            bail!("Imagine is a local API and requires a loopback bind address");
        }
        let project = project.canonicalize().context("project directory")?;
        std::fs::create_dir_all(&data)?;
        std::fs::set_permissions(&data, std::fs::Permissions::from_mode(0o700))?;
        let data = data.canonicalize()?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(data.join("server.lock"))?;
        lock.try_lock_exclusive()
            .context("Another Imagine server owns this data directory")?;
        let local: serde_json::Value =
            std::fs::read(project.join("experiments/qwen-image21-sage/local-settings.json"))
                .ok()
                .and_then(|v| serde_json::from_slice(&v).ok())
                .unwrap_or_default();
        let model_root = std::env::var_os("QWEN_MODEL_ROOT")
            .map(PathBuf::from)
            .or_else(|| local["model_root"].as_str().map(PathBuf::from))
            .unwrap_or_else(|| project.join("models"));
        let model_root = if model_root.is_absolute() {
            model_root
        } else {
            project.join(model_root)
        };
        Ok((
            Self {
                project,
                data,
                bind,
                model_root,
            },
            lock,
        ))
    }
}
pub fn baseline_dir(project: &Path) -> PathBuf {
    project.join("experiments/qwen-image21-sage/baselines/1600x896-v2")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_configuration_needs_no_credentials_and_keeps_single_instance() {
        let root = tempfile::tempdir().unwrap();
        let data = root.path().join("data");
        let (config, lock) = Config::load(
            root.path().into(),
            data.clone(),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        assert!(!data.join("access.json").exists());
        assert_eq!(
            std::fs::metadata(&data).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(
            Config::load(
                root.path().into(),
                data.clone(),
                "127.0.0.1:0".parse().unwrap()
            )
            .is_err()
        );
        drop(lock);
        // Legacy credentials are neither parsed nor changed by token-free startup.
        std::fs::write(data.join("access.json"), b"unused legacy fixture").unwrap();
        let (again, _) = Config::load(
            root.path().into(),
            data.clone(),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        assert_eq!(config.data, again.data);
        assert_eq!(
            std::fs::read(data.join("access.json")).unwrap(),
            b"unused legacy fixture"
        );
        assert!(Config::load(root.path().into(), data, "0.0.0.0:8080".parse().unwrap()).is_err());
    }
}
