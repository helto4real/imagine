//! Explicit, locally prepared Python-free worker deployment. No CUDA at startup.
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub schema: u32,
    #[serde(default)]
    pub h3: Option<H3>,
    pub reference_revision: String,
    pub binary: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub files: BTreeMap<PathBuf, String>,
}
impl Runtime {
    pub fn load(path: &Path) -> Result<Self> {
        let runtime: Self = serde_json::from_slice(
            &std::fs::read(path).context("Rust runtime missing; run ./prepare-rust-backend.sh")?,
        )?;
        if runtime.schema != 1
            || runtime.reference_revision != "446ef4e005fc7134d53ce6869ebd77d2134cd39c"
            || !runtime.binary.is_absolute()
            || !runtime.files.contains_key(&runtime.binary)
        {
            bail!("Unsupported Rust runtime manifest")
        }
        let allowed = [
            "LD_LIBRARY_PATH",
            "IMAGINE_KITCHEN_LIBRARY",
            "IMAGINE_AIMDO_LIBRARY",
            "IMAGINE_PILLOW_LIBRARY",
            "IMAGINE_TRITON_COMPILER_PROFILE",
            "IMAGINE_TRITON_DRIVER_LIBRARY",
            "IMAGINE_TRITON_FRONTEND_CUBINS",
            "OMP_NUM_THREADS",
        ];
        if runtime
            .environment
            .keys()
            .any(|key| !allowed.contains(&key.as_str()))
        {
            bail!("Unknown Rust runtime environment option")
        }
        runtime.verify()?;
        if let Some(h3) = &runtime.h3 {
            h3.verify()?;
        }
        for key in [
            "IMAGINE_KITCHEN_LIBRARY",
            "IMAGINE_AIMDO_LIBRARY",
            "IMAGINE_PILLOW_LIBRARY",
            "IMAGINE_TRITON_COMPILER_PROFILE",
        ] {
            let value = runtime
                .environment
                .get(key)
                .context("Incomplete Rust runtime environment")?;
            if !runtime.files.contains_key(Path::new(value)) {
                bail!("Unverified Rust runtime dependency")
            }
        }
        Ok(runtime)
    }
    pub fn verify(&self) -> Result<()> {
        // Runtime hashes are build provenance, not a startup/reload gate.
        // Recompilation may replace compatible executables without rewriting the manifest.
        for (path, expected) in &self.files {
            if !path.is_absolute() || !crate::h3_protocol::is_sha256(expected) || !path.is_file() {
                bail!("Rust runtime artifact missing or invalid")
            }
            std::fs::File::open(path).context("Rust runtime artifact is not readable")?;
        }
        Ok(())
    }
}

/// Build-time deployment only; hashes dependencies and writes no runtime data.
pub fn prepare(root: &Path, site: &Path, output: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let site = site.canonicalize()?;
    let binary = root.join("target/debug/imagine-worker");
    let mut environment = BTreeMap::from([
        (
            "LD_LIBRARY_PATH".to_owned(),
            site.join("torch/lib").to_string_lossy().into_owned(),
        ),
        ("OMP_NUM_THREADS".into(), "2".into()),
    ]);
    for (key, relative) in [
        (
            "IMAGINE_KITCHEN_LIBRARY",
            "kitchen-manylinux/libimagine_kitchen.so",
        ),
        ("IMAGINE_AIMDO_LIBRARY", "aimdo/aimdo.so"),
        ("IMAGINE_PILLOW_LIBRARY", "pillow/libimagine_pillow.so"),
        (
            "IMAGINE_TRITON_COMPILER_PROFILE",
            "runtime/triton-compiler-profile.json",
        ),
        (
            "IMAGINE_TRITON_DRIVER_LIBRARY",
            "triton-driver/libimagine_triton_driver.so",
        ),
        (
            "IMAGINE_TRITON_FRONTEND_CUBINS",
            "triton-frontend/standalone",
        ),
    ] {
        environment.insert(
            key.into(),
            root.join(relative).to_string_lossy().into_owned(),
        );
    }
    let mut files = BTreeMap::new();
    for file in std::iter::once(binary.clone()).chain(
        environment
            .values()
            .map(PathBuf::from)
            .filter(|p| p.is_file()),
    ) {
        files.insert(file.clone(), crate::storage::hash(&std::fs::read(file)?));
    }
    let value = serde_json::json!({"schema":1,"reference_revision":"446ef4e005fc7134d53ce6869ebd77d2134cd39c","binary":binary,"environment":environment,"files":files});
    let parent = output.parent().context("Manifest parent")?;
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".rust-runtime-{}.json", uuid::Uuid::new_v4()));
    std::fs::write(&temp, serde_json::to_vec_pretty(&value)?)?;
    std::fs::rename(temp, output)?;
    Ok(())
}

/// Explicit deployment configuration, never accepted from a generation request.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct H3 {
    pub profile_checkpoints: BTreeMap<String, String>,
    pub dependency_map: PathBuf,
    pub identity: crate::h3_protocol::Deployment,
    pub binary: PathBuf,
    pub model_root: PathBuf,
    pub environment: BTreeMap<String, String>,
    pub files: BTreeMap<PathBuf, String>,
}
impl H3 {
    pub fn verify(&self) -> Result<()> {
        let allowed = [
            "LD_LIBRARY_PATH",
            "IMAGINE_KITCHEN_LIBRARY",
            "IMAGINE_AIMDO_LIBRARY",
            "IMAGINE_PILLOW_LIBRARY",
            "IMAGINE_TRITON_COMPILER_PROFILE",
            "IMAGINE_TRITON_DRIVER_LIBRARY",
            "IMAGINE_TRITON_FRONTEND_CUBINS",
            "OMP_NUM_THREADS",
            // Opt-in observer worker used by the isolated Reference acceptance test.
            "IMAGINE_REFERENCE_PROBE_OUTPUT",
        ];
        if self.profile_checkpoints != crate::h3_protocol::profile_checkpoints()
            || !self.identity.valid()
            || !self.binary.is_absolute()
            || !self.model_root.is_absolute()
            || self.files.get(&self.binary) != Some(&self.identity.worker_sha256)
            || self
                .environment
                .keys()
                .any(|k| !allowed.contains(&k.as_str()))
        {
            bail!("h3_deployment_mismatch");
        }
        for (path, expected) in &self.files {
            if !path.is_absolute()
                || path.components().any(|c| {
                    !matches!(
                        c,
                        std::path::Component::RootDir | std::path::Component::Normal(_)
                    )
                })
                || matches!(
                    path.extension().and_then(|s| s.to_str()),
                    Some("safetensors" | "gguf" | "pt" | "pth")
                )
                || !crate::h3_protocol::is_sha256(expected)
                || !path.is_file()
            {
                bail!("h3_deployment_mismatch");
            }
            std::fs::File::open(path).context("H3 dependency is not readable")?;
        }
        let dependencies: BTreeMap<_, _> = self
            .files
            .iter()
            .filter(|(p, _)| *p != &self.binary)
            .collect();
        if crate::storage::hash(&serde_json::to_vec(&dependencies)?)
            != self.identity.dependencies_sha256
        {
            bail!("h3_deployment_mismatch");
        }
        if !self.dependency_map.is_absolute() || self.files.contains_key(&self.dependency_map) {
            bail!("h3_deployment_mismatch");
        }
        let map_file = std::fs::File::open(&self.dependency_map)?;
        if map_file.metadata()?.len() > 1024 * 1024 {
            bail!("h3_deployment_mismatch");
        }
        let supplied: BTreeMap<PathBuf, String> = serde_json::from_reader(map_file)?;
        let expected: BTreeMap<PathBuf, String> = dependencies
            .into_iter()
            .map(|(p, s)| (p.clone(), s.clone()))
            .collect();
        if supplied != expected {
            bail!("h3_deployment_mismatch");
        }
        for key in [
            "IMAGINE_KITCHEN_LIBRARY",
            "IMAGINE_AIMDO_LIBRARY",
            "IMAGINE_PILLOW_LIBRARY",
        ] {
            let path = self
                .environment
                .get(key)
                .context("H3 runtime environment incomplete")?;
            if !self.files.contains_key(Path::new(path)) {
                bail!("h3_deployment_mismatch");
            }
        }
        Ok(())
    }
}

/// Explicit CPU-only assembly; never selects a deployment or changes a running service.
pub fn attach_h3(image_runtime: &Path, h3_deployment: &Path, output: &Path) -> Result<()> {
    Runtime::load(image_runtime)?;
    let h3_value: serde_json::Value = serde_json::from_slice(&std::fs::read(h3_deployment)?)?;
    let h3: H3 = serde_json::from_value(h3_value.clone())?;
    h3.verify()?;
    let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(image_runtime)?)?;
    value["h3"] = h3_value;
    let parent = output.parent().context("Manifest parent")?;
    std::fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".h3-runtime-{}.json", uuid::Uuid::new_v4()));
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&temp)?;
    let result = (|| -> Result<()> {
        file.write_all(&serde_json::to_vec_pretty(&value)?)?;
        file.sync_all()?;
        std::fs::rename(&temp, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}
