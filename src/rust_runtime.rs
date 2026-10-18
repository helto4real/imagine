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
        for (path, expected) in &self.files {
            if !path.is_absolute()
                || crate::storage::hash(
                    &std::fs::read(path).context("Rust runtime artifact missing")?,
                ) != *expected
            {
                bail!("Rust runtime artifact changed; rerun ./prepare-rust-backend.sh")
            }
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
