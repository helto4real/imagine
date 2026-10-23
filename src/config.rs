use anyhow::{Context, Result, bail};
use fs2::FileExt;
use std::{
    fs::{File, OpenOptions},
    net::SocketAddr,
    os::fd::AsRawFd,
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone)]
pub struct Config {
    pub project: PathBuf,
    pub data: PathBuf,
    pub bind: SocketAddr,
    pub model_root: PathBuf,
    pub private_storage: Option<PrivatePaths>,
}
#[derive(Clone)]
pub struct PrivatePaths {
    pub root: crate::private_storage::Root,
    pub project_id: String,
    pub public: crate::storage_layout::PublicPaths,
    pub runtime: PathBuf,
    // Keep the encrypted directory inode alive even if its mount is detached.
    // CLOEXEC is intentional: children open the server's descriptor via /proc.
    temporary: Arc<File>,
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
        let model_root = model_root(&project);
        Ok((
            Self {
                project,
                data,
                bind,
                model_root,
                private_storage: None,
            },
            lock,
        ))
    }

    /// Normal private instance: no legacy data is read, purged or migrated.
    pub fn load_private(
        project: PathBuf,
        root: crate::private_storage::Root,
        project_id: String,
        bind: SocketAddr,
        home: &Path,
        runtime: PathBuf,
    ) -> Result<(Self, File)> {
        if !bind.ip().is_loopback() {
            bail!("Imagine is a local API and requires a loopback bind address");
        }
        let project = project
            .canonicalize()
            .map_err(|_| anyhow::anyhow!("project_directory_unavailable"))?;
        let root = root
            .with_runtime_directory(runtime.clone())
            .map_err(anyhow::Error::msg)?;
        let layout = root.project(&project_id).map_err(anyhow::Error::msg)?;
        let temporary = Arc::new(
            root.directory(
                layout
                    .tmp
                    .strip_prefix(root.layout().root())
                    .map_err(|_| anyhow::anyhow!("video_destination_invalid"))?,
            )
            .map_err(anyhow::Error::msg)?,
        );
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&runtime)
            .map_err(|_| anyhow::anyhow!("runtime_directory_unavailable"))?;
        if std::fs::symlink_metadata(&runtime)
            .map_err(|_| anyhow::anyhow!("runtime_directory_unavailable"))?
            .file_type()
            .is_symlink()
        {
            bail!("runtime_directory_unavailable");
        }
        std::fs::set_permissions(&runtime, std::fs::Permissions::from_mode(0o700))
            .map_err(|_| anyhow::anyhow!("runtime_directory_unavailable"))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(runtime.join(format!("server-{}.lock", bind.port())))
            .map_err(|_| anyhow::anyhow!("server_lock_unavailable"))?;
        lock.try_lock_exclusive()
            .map_err(|_| anyhow::anyhow!("server_instance_already_running"))?;
        let model_root = model_root(&project);
        Ok((
            Self {
                project,
                data: layout.work,
                bind,
                model_root,
                private_storage: Some(PrivatePaths {
                    root,
                    project_id,
                    public: crate::storage_layout::PublicPaths::for_home(home),
                    runtime,
                    temporary,
                }),
            },
            lock,
        ))
    }

    /// Apply after runtime-provided environment. Only the owned child changes.
    /// Cache directories contain compiler artifacts; all temporary payloads stay private.
    pub(crate) fn private_worker_environment(
        &self,
        command: &mut std::process::Command,
        family: &str,
    ) -> Result<()> {
        let Some(paths) = &self.private_storage else {
            return Ok(());
        };
        paths
            .root
            .project(&paths.project_id)
            .map_err(anyhow::Error::msg)?;
        let temporary = PathBuf::from(format!(
            "/proc/{}/fd/{}",
            std::process::id(),
            paths.temporary.as_raw_fd()
        ));
        let cache = paths.public.cache.join(family);
        std::fs::create_dir_all(&cache).map_err(|_| anyhow::anyhow!("worker_cache_unavailable"))?;
        for key in ["TMPDIR", "TMP", "TEMP"] {
            command.env(key, &temporary);
        }
        command.env("IMAGINE_PRIVATE_TMP", &temporary);
        command
            .env("IMAGINE_WORKER_CACHE", &cache)
            .env("CUDA_CACHE_PATH", cache.join("cuda"))
            .env("TRITON_CACHE_DIR", cache.join("triton"))
            .env("TORCH_HOME", cache.join("torch"))
            .env("XDG_CACHE_HOME", &paths.public.cache);
        Ok(())
    }
}
fn model_root(project: &Path) -> PathBuf {
    let local: serde_json::Value =
        std::fs::read(project.join("experiments/qwen-image21-sage/local-settings.json"))
            .ok()
            .and_then(|v| serde_json::from_slice(&v).ok())
            .unwrap_or_default();
    let model_root = std::env::var_os("QWEN_MODEL_ROOT")
        .map(PathBuf::from)
        .or_else(|| local["model_root"].as_str().map(PathBuf::from))
        .unwrap_or_else(|| project.join("models"));
    if model_root.is_absolute() {
        model_root
    } else {
        project.join(model_root)
    }
}
pub fn baseline_dir(project: &Path) -> PathBuf {
    project.join("experiments/qwen-image21-sage/baselines/1600x896-v2")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{private_storage::Root, video_destination as d};
    use std::sync::{Arc, Mutex};

    struct Environment(Mutex<String>);
    impl d::Environment for Environment {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            Ok("own-config-fixture".into())
        }
    }
    fn private_fixture() -> (tempfile::TempDir, Root, Arc<Environment>) {
        let directory =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let mount = directory.path().join("own-volume");
        let root = mount.join("imagine");
        std::fs::create_dir_all(&root).unwrap();
        let config = d::Config {
            root,
            mount: mount.clone(),
            source: "own-volume".into(),
            filesystem: "fixturefs".into(),
            fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
        };
        let environment = Arc::new(Environment(Mutex::new(format!(
            "1 0 0:1 / {} rw - fixturefs own-volume rw\n",
            mount.display()
        ))));
        let root = Root::with_environment(config, environment.clone()).unwrap();
        (directory, root, environment)
    }
    #[test]
    fn private_config_uses_guarded_work_and_port_scoped_runtime_locks() {
        let (fixture, root, environment) = private_fixture();
        let home = fixture.path().join("own-home");
        let runtime = fixture.path().join("own-runtime/imagine");
        let load = |port| {
            Config::load_private(
                fixture.path().into(),
                root.clone(),
                "own-project".into(),
                ([127, 0, 0, 1], port).into(),
                &home,
                runtime.clone(),
            )
        };
        let (config, lock) = load(18221).unwrap();
        let project = root.layout().project("own-project").unwrap();
        assert_eq!(config.data, project.work);
        assert_eq!(
            config
                .private_storage
                .as_ref()
                .unwrap()
                .root
                .runtime_directory()
                .unwrap(),
            runtime
        );
        assert_eq!(
            std::fs::metadata(&runtime).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert!(runtime.join("server-18221.lock").is_file());
        assert!(!project.work.join("server.lock").exists());
        assert!(!project.document.exists());
        assert!(load(18221).is_err());
        let (_, other_lock) = load(18222).unwrap();
        root.write(
            Path::new("projects/own-project/work/own-preserved"),
            b"OWN EXISTING DATA",
        )
        .unwrap();
        drop(lock);
        let (_, lock) = load(18221).unwrap();
        assert_eq!(
            root.read(Path::new("projects/own-project/work/own-preserved"))
                .unwrap()
                .unwrap(),
            b"OWN EXISTING DATA"
        );
        drop((lock, other_lock));
        *environment.0.lock().unwrap() = String::new();
        assert!(load(18221).is_err());
        assert!(!home.exists());
    }
    #[test]
    fn private_worker_environment_is_child_only_and_rechecks_volume() {
        let (fixture, root, environment) = private_fixture();
        let home = fixture.path().join("own-home");
        let (config, _lock) = Config::load_private(
            fixture.path().into(),
            root.clone(),
            "own-project".into(),
            "127.0.0.1:18223".parse().unwrap(),
            &home,
            fixture.path().join("runtime"),
        )
        .unwrap();
        for family in ["rust", "h3"] {
            let mut command = std::process::Command::new("own-unstarted-worker");
            command
                .env("TMPDIR", "own-old-runtime-temp")
                .env("IMAGINE_WORKER_CACHE", "own-old-runtime-cache");
            config
                .private_worker_environment(&mut command, family)
                .unwrap();
            let env: std::collections::BTreeMap<_, _> = command
                .get_envs()
                .map(|(k, v)| (k.to_os_string(), v.unwrap().to_os_string()))
                .collect();
            let project = root.layout().project("own-project").unwrap();
            for name in ["TMPDIR", "TMP", "TEMP"] {
                let temporary = PathBuf::from(&env[std::ffi::OsStr::new(name)]);
                assert!(temporary.starts_with(format!("/proc/{}/fd", std::process::id())));
                assert_eq!(temporary.canonicalize().unwrap(), project.tmp);
            }
            assert_eq!(
                env[std::ffi::OsStr::new("IMAGINE_WORKER_CACHE")],
                home.join(".cache/imagine").join(family).as_os_str()
            );
            assert_eq!(
                env[std::ffi::OsStr::new("CUDA_CACHE_PATH")],
                home.join(".cache/imagine")
                    .join(family)
                    .join("cuda")
                    .as_os_str()
            );
        }
        let mount = environment.0.lock().unwrap().clone();
        for unavailable in [String::new(), mount.replace(" rw", " ro")] {
            *environment.0.lock().unwrap() = unavailable;
            let mut command = std::process::Command::new("own-unstarted-worker");
            assert!(
                config
                    .private_worker_environment(&mut command, "rust")
                    .is_err()
            );
            assert_eq!(command.get_envs().count(), 0);
        }
        *environment.0.lock().unwrap() = mount;
        assert!(
            config
                .private_worker_environment(
                    &mut std::process::Command::new("own-unstarted-worker"),
                    "h3"
                )
                .is_ok()
        );
    }
    #[test]
    fn private_worker_temporary_inode_survives_path_replacement_and_config_clone() {
        let (fixture, root, _) = private_fixture();
        let (config, _lock) = Config::load_private(
            fixture.path().into(),
            root.clone(),
            "own-project".into(),
            "127.0.0.1:18224".parse().unwrap(),
            &fixture.path().join("home"),
            fixture.path().join("runtime"),
        )
        .unwrap();
        let project = root.layout().project("own-project").unwrap();
        let detached = project.root.join("own-detached-tmp");
        std::fs::rename(&project.tmp, &detached).unwrap();
        std::fs::create_dir(&project.tmp).unwrap();
        let cloned = config.clone();
        drop(config);
        let mut command = std::process::Command::new("own-unstarted-worker");
        cloned
            .private_worker_environment(&mut command, "h3")
            .unwrap();
        let temporary = PathBuf::from(
            command
                .get_envs()
                .find(|(key, _)| *key == "TMPDIR")
                .unwrap()
                .1
                .unwrap(),
        );
        assert_eq!(temporary.canonicalize().unwrap(), detached);
        std::fs::write(temporary.join("own-small-fixture"), b"OWN PINNED TEMP").unwrap();
        assert_eq!(
            std::fs::read(detached.join("own-small-fixture")).unwrap(),
            b"OWN PINNED TEMP"
        );
        assert!(!project.tmp.join("own-small-fixture").exists());
    }
    #[test]
    fn actual_python_child_writes_only_through_pinned_private_temporary_directory() {
        let (fixture, root, _) = private_fixture();
        let (config, _lock) = Config::load_private(
            fixture.path().into(),
            root.clone(),
            "own-child".into(),
            "127.0.0.1:18225".parse().unwrap(),
            &fixture.path().join("home"),
            fixture.path().join("runtime"),
        )
        .unwrap();
        let mut command = std::process::Command::new("/usr/bin/python3");
        command.args(["-I", "-c", "import os,runpy,sys,tempfile; runpy.run_path(sys.argv[1])['private_temporary_directory'](); f,p=tempfile.mkstemp(prefix='own-child-'); os.write(f,b'OWN SYNTHETIC CHILD'); os.close(f); print(p)"])
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("workers/comfy/worker.py"));
        config
            .private_worker_environment(&mut command, "rust")
            .unwrap();
        let output = command.output().unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let path = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
        let temporary = root.layout().project("own-child").unwrap().tmp;
        assert_eq!(path.parent().unwrap().canonicalize().unwrap(), temporary);
        assert_eq!(std::fs::read(&path).unwrap(), b"OWN SYNTHETIC CHILD");
    }
    #[test]
    fn explicit_fixture_does_not_override_worker_environment() {
        let fixture =
            tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES"))
                .unwrap();
        let (config, _lock) = Config::load(
            fixture.path().into(),
            fixture.path().join("data"),
            "127.0.0.1:0".parse().unwrap(),
        )
        .unwrap();
        assert!(config.private_storage.is_none());
        let mut command = std::process::Command::new("own-unstarted-worker");
        command
            .env("TMPDIR", "own-fixture-temp")
            .env("IMAGINE_WORKER_CACHE", "own-fixture-cache");
        let before = command
            .get_envs()
            .map(|(k, v)| (k.to_os_string(), v.map(|v| v.to_os_string())))
            .collect::<Vec<_>>();
        config
            .private_worker_environment(&mut command, "rust")
            .unwrap();
        assert_eq!(
            command
                .get_envs()
                .map(|(k, v)| (k.to_os_string(), v.map(|v| v.to_os_string())))
                .collect::<Vec<_>>(),
            before
        );
    }
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
