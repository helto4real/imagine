//! Actual startup identities. Only filesystem bytes, serde and SHA-256; no CUDA.
use super::protocol::{self as p, Deployment, ErrorCode, Profile};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

pub const BRIDGE_SHA256: &str = "ab7c807cd5f9aac244f58a302d9500a6ab04b0314b7223404881ce407ed0d4a3";
pub const MAP_ENV: &str = "IMAGINE_H3_DEPENDENCY_MAP";
pub const MODEL_ENV: &str = "IMAGINE_H3_MODEL_ROOT";
pub struct Config {
    pub model_root: PathBuf,
    pub kitchen: PathBuf,
    pub aimdo: PathBuf,
    pub deployment: Deployment,
}
pub fn sha(path: &Path) -> Result<String, ErrorCode> {
    let mut file = File::open(path).map_err(|_| ErrorCode::DeploymentMismatch)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|_| ErrorCode::DeploymentMismatch)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn absolute(path: &Path) -> bool {
    path.is_absolute()
        && path
            .components()
            .all(|c| matches!(c, Component::RootDir | Component::Normal(_)))
}
pub fn identity(
    executable: &Path,
    dependencies: &BTreeMap<String, String>,
    anchors: &[(&Path, &str)],
    forbidden: &[PathBuf],
) -> Result<Deployment, ErrorCode> {
    if dependencies.is_empty() || !absolute(executable) {
        return Err(ErrorCode::DeploymentMismatch);
    }
    for (name, expected) in dependencies {
        let path = Path::new(name);
        if !absolute(path)
            || path == executable
            || forbidden.iter().any(|p| p == path)
            || matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("safetensors" | "gguf" | "pt" | "pth")
            )
            || !p::is_sha256(expected)
            || sha(path)? != *expected
        {
            return Err(ErrorCode::DeploymentMismatch);
        }
    }
    for (path, expected) in anchors {
        if dependencies
            .get(path.to_str().ok_or(ErrorCode::DeploymentMismatch)?)
            .map(String::as_str)
            != Some(*expected)
        {
            return Err(ErrorCode::DeploymentMismatch);
        }
    }
    let map = serde_json::to_vec(dependencies).map_err(|_| ErrorCode::DeploymentMismatch)?;
    Ok(Deployment {
        protocol: p::VERSION,
        profile: Profile::ShortAvNoReferenceV1,
        core_revision: p::CORE_REVISION.into(),
        model_lock_sha256: p::MODEL_LOCK_SHA256.into(),
        worker_sha256: sha(executable)?,
        dependencies_sha256: format!("{:x}", Sha256::digest(map)),
    })
}
impl Config {
    pub fn environment() -> Result<Self, ErrorCode> {
        let path = |name| {
            std::env::var_os(name)
                .map(PathBuf::from)
                .filter(|p| absolute(p))
                .ok_or(ErrorCode::DeploymentMismatch)
        };
        let map_path = path(MAP_ENV)?;
        let model_root = path(MODEL_ENV)?;
        let kitchen = path("IMAGINE_KITCHEN_LIBRARY")?;
        let aimdo = path("IMAGINE_AIMDO_LIBRARY")?;
        let executable = std::env::current_exe().map_err(|_| ErrorCode::DeploymentMismatch)?;
        let file = File::open(&map_path).map_err(|_| ErrorCode::DeploymentMismatch)?;
        if file
            .metadata()
            .map_err(|_| ErrorCode::DeploymentMismatch)?
            .len()
            > 1024 * 1024
        {
            return Err(ErrorCode::DeploymentMismatch);
        }
        let dependencies: BTreeMap<String, String> =
            serde_json::from_reader(file).map_err(|_| ErrorCode::DeploymentMismatch)?;
        let aimdo_digest = dependencies
            .get(aimdo.to_str().ok_or(ErrorCode::DeploymentMismatch)?)
            .ok_or(ErrorCode::DeploymentMismatch)?;
        let deployment = identity(
            &executable,
            &dependencies,
            &[(&kitchen, BRIDGE_SHA256), (&aimdo, aimdo_digest)],
            &[map_path],
        )?;
        Ok(Self {
            model_root,
            kitchen,
            aimdo,
            deployment,
        })
    }
}
