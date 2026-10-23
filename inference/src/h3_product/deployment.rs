//! Actual startup identities. Only filesystem bytes, serde and SHA-256; no CUDA.
use super::protocol::{self as p, Deployment, ErrorCode, Profile};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

/// Historical build-audit anchor only; ordinary startup accepts compatible rebuilds.
pub const BRIDGE_SHA256: &str = "ab7c807cd5f9aac244f58a302d9500a6ab04b0314b7223404881ce407ed0d4a3";
pub const MAP_ENV: &str = "IMAGINE_H3_DEPENDENCY_MAP";
pub const MODEL_ENV: &str = "IMAGINE_H3_MODEL_ROOT";
pub struct Config {
    pub model_root: PathBuf,
    pub kitchen: PathBuf,
    pub aimdo: PathBuf,
    pub pillow: PathBuf,
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
/// Check readable safetensors structure without reading or hashing tensor payloads.
pub fn checkpoint(path: &Path) -> Result<(), ErrorCode> {
    let invalid = || ErrorCode::ModelIdentityMismatch;
    if !absolute(path) {
        return Err(invalid());
    }
    let mut file = File::open(path).map_err(|_| invalid())?;
    let metadata = file.metadata().map_err(|_| invalid())?;
    if !metadata.is_file() {
        return Err(invalid());
    }
    let mut length = [0; 8];
    file.read_exact(&mut length).map_err(|_| invalid())?;
    let length = u64::from_le_bytes(length);
    if length == 0 || length > 64 * 1024 * 1024 || length + 8 > metadata.len() {
        return Err(invalid());
    }
    let mut header = vec![0; length as usize];
    file.read_exact(&mut header).map_err(|_| invalid())?;
    let header: serde_json::Value = serde_json::from_slice(&header).map_err(|_| invalid())?;
    let header = header.as_object().ok_or_else(invalid)?;
    let mut count = 0;
    for (key, tensor) in header {
        if key == "__metadata__" {
            continue;
        }
        let offsets = tensor["data_offsets"]
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or_else(invalid)?;
        let start = offsets[0].as_u64().ok_or_else(invalid)?;
        let end = offsets[1].as_u64().ok_or_else(invalid)?;
        if start > end
            || end > metadata.len() - length - 8
            || tensor["dtype"].as_str().is_none_or(str::is_empty)
            || tensor["shape"]
                .as_array()
                .is_none_or(|v| v.iter().any(|d| d.as_u64().is_none()))
        {
            return Err(invalid());
        }
        count += 1;
    }
    if count == 0 {
        return Err(invalid());
    }
    Ok(())
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
        profile: Profile::AvNoReferenceV2,
        core_revision: p::CORE_REVISION.into(),
        model_lock_sha256: p::MODEL_LOCK_SHA256.into(),
        worker_sha256: sha(executable)?,
        dependencies_sha256: format!("{:x}", Sha256::digest(map)),
    })
}
/// Startup validates declared protocol/dependencies and file accessibility only.
/// Full-file SHA256 remains available through identity() for explicit build audits.
pub fn declaration(
    executable: &Path,
    dependencies: &BTreeMap<String, String>,
    declared: Deployment,
    forbidden: &[PathBuf],
) -> Result<Deployment, ErrorCode> {
    if !declared.valid()
        || !absolute(executable)
        || !executable.is_file()
        || dependencies.is_empty()
    {
        return Err(ErrorCode::DeploymentMismatch);
    }
    for (name, digest) in dependencies {
        let path = Path::new(name);
        if !absolute(path)
            || path == executable
            || forbidden.iter().any(|p| p == path)
            || matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("safetensors" | "gguf" | "pt" | "pth")
            )
            || !p::is_sha256(digest)
            || !path.is_file()
            || File::open(path).is_err()
        {
            return Err(ErrorCode::DeploymentMismatch);
        }
    }
    let map = serde_json::to_vec(dependencies).map_err(|_| ErrorCode::DeploymentMismatch)?;
    if format!("{:x}", Sha256::digest(map)) != declared.dependencies_sha256 {
        return Err(ErrorCode::DeploymentMismatch);
    }
    Ok(declared)
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
        if !dependencies.contains_key(kitchen.to_str().ok_or(ErrorCode::DeploymentMismatch)?)
            || !dependencies.contains_key(aimdo.to_str().ok_or(ErrorCode::DeploymentMismatch)?)
        {
            return Err(ErrorCode::DeploymentMismatch);
        }
        let declared: Deployment = serde_json::from_str(
            &std::env::var("IMAGINE_H3_DEPLOYMENT").map_err(|_| ErrorCode::DeploymentMismatch)?,
        )
        .map_err(|_| ErrorCode::DeploymentMismatch)?;
        let deployment = declaration(&executable, &dependencies, declared, &[map_path])?;
        let pillow = path("IMAGINE_PILLOW_LIBRARY")?;
        {
            let p = &pillow;
            if !absolute(p)
                || !dependencies.contains_key(p.to_str().ok_or(ErrorCode::DeploymentMismatch)?)
            {
                return Err(ErrorCode::DeploymentMismatch);
            }
        }
        Ok(Self {
            pillow,
            model_root,
            kitchen,
            aimdo,
            deployment,
        })
    }
}
