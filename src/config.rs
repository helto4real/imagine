use anyhow::{Context, Result, bail};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    net::SocketAddr,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Access {
    pub admin_token: String,
    pub clients: BTreeMap<String, String>,
}
#[derive(Clone)]
pub struct Config {
    pub project: PathBuf,
    pub data: PathBuf,
    pub bind: SocketAddr,
    pub model_root: PathBuf,
    pub access: Access,
}
pub fn token() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}
impl Config {
    pub fn load(project: PathBuf, data: PathBuf, bind: SocketAddr) -> Result<(Self, File)> {
        if !bind.ip().is_loopback() {
            bail!(
                "The initial API is loopback-only; use an authenticated local reverse proxy for remote access"
            );
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
        let access_path = data.join("access.json");
        if !access_path.exists() {
            let access = Access {
                admin_token: token(),
                clients: BTreeMap::from([("local".into(), token())]),
            };
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&access_path)?;
            use std::io::Write;
            f.write_all(&serde_json::to_vec_pretty(&access)?)?;
            f.sync_all()?;
        }
        let access: Access = serde_json::from_slice(&std::fs::read(&access_path)?)?;
        if access.admin_token.len() < 32
            || access.clients.values().any(|v| v.len() < 32)
            || access
                .clients
                .keys()
                .any(|id| id == "admin" || id.is_empty() || id.len() > 64)
        {
            bail!(
                "access.json requires unique client identities and tokens of at least 32 characters"
            );
        }
        let mut tokens: Vec<_> = access
            .clients
            .values()
            .chain(std::iter::once(&access.admin_token))
            .collect();
        tokens.sort();
        tokens.dedup();
        if tokens.len() != access.clients.len() + 1 {
            bail!("Client/admin tokens must be distinct");
        }
        std::fs::set_permissions(&access_path, std::fs::Permissions::from_mode(0o600))?;
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
                access,
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
    fn credentials_are_private_stable_and_single_instance() {
        let root = tempfile::tempdir().unwrap();
        let data = root.path().join("data");
        let (config, lock) = Config::load(
            root.path().into(),
            data.clone(),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        assert_ne!(config.access.admin_token, config.access.clients["local"]);
        assert_eq!(
            std::fs::metadata(data.join("access.json"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
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
        let (again, _) = Config::load(
            root.path().into(),
            data.clone(),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        assert_eq!(config.access.admin_token, again.access.admin_token);
        assert!(Config::load(root.path().into(), data, "0.0.0.0:8080".parse().unwrap()).is_err());
    }
}
