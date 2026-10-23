use anyhow::{Context, Result};
use imagine_server::{
    api::{self, App},
    comfy_engine::{Comfy, EngineKind},
    config::Config,
    engine::{Backend, EngineState, Native},
    scheduler::{self, Action},
    storage::Store,
};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::{Semaphore, watch};

#[tokio::main]
async fn main() -> Result<()> {
    // Prompts must not be captured by an OS core dump.
    let limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let raw = std::env::args().skip(1).collect::<Vec<_>>();
    if raw.first().map(String::as_str) == Some("--prepare-h3-runtime") {
        if raw.len() != 4 {
            anyhow::bail!(
                "--prepare-h3-runtime needs image manifest, H3 deployment and output manifest"
            )
        }
        return imagine_server::rust_runtime::attach_h3(
            Path::new(&raw[1]),
            Path::new(&raw[2]),
            Path::new(&raw[3]),
        );
    }
    if raw.first().map(String::as_str) == Some("--prepare-rust-runtime") {
        if raw.len() != 4 {
            anyhow::bail!(
                "--prepare-rust-runtime needs dependency root, package site and manifest path"
            )
        }
        return imagine_server::rust_runtime::prepare(
            Path::new(&raw[1]),
            Path::new(&raw[2]),
            Path::new(&raw[3]),
        );
    }
    let mut project = std::env::current_dir()?;
    let mut data = None;
    let mut profiles = None;
    let mut bind = "127.0.0.1:8080".to_owned();
    let mut backend = EngineKind::default();
    let mut comfy_root = None;
    let mut comfy_python = None;
    let mut rust_runtime = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            // Compatibility with launchers that explicitly selected plain storage.
            "--no-encryption" => {}
            "--project" => project = PathBuf::from(args.next().context("--project needs a path")?),
            "--profiles" => {
                profiles = Some(PathBuf::from(
                    args.next().context("--profiles needs a path")?,
                ))
            }
            "--data" => data = Some(PathBuf::from(args.next().context("--data needs a path")?)),
            "--bind" => bind = args.next().context("--bind needs an address")?,
            "--backend" => {
                backend = EngineKind::parse(
                    &args
                        .next()
                        .context("--backend needs rust, native or comfy")?,
                )?
            }
            "--rust-runtime" => {
                rust_runtime = Some(PathBuf::from(
                    args.next().context("--rust-runtime needs a path")?,
                ))
            }
            "--comfy-root" => {
                comfy_root = Some(PathBuf::from(
                    args.next().context("--comfy-root needs a path")?,
                ))
            }
            "--comfy-python" => {
                comfy_python = Some(PathBuf::from(
                    args.next().context("--comfy-python needs a path")?,
                ))
            }
            "--help" => {
                println!(
                    "imagine-server [--project PATH] [--data PATH] [--profiles PATH] [--bind 127.0.0.1:8080] [--backend rust|native|comfy] [--rust-runtime PATH] [--comfy-root PATH --comfy-python PATH]"
                );
                return Ok(());
            }
            _ => anyhow::bail!("Unknown option: {arg}"),
        }
    }
    if backend != EngineKind::Rust && rust_runtime.is_some() {
        anyhow::bail!("--rust-runtime requires --backend rust")
    }
    if backend != EngineKind::Comfy && (comfy_root.is_some() || comfy_python.is_some()) {
        anyhow::bail!("Comfy options require --backend comfy");
    }
    let comfy_paths = if backend == EngineKind::Comfy {
        let home_dir = std::env::var_os("HOME").map(PathBuf::from);
        Some(resolve_comfy_paths(
            &project,
            home_dir.as_deref(),
            comfy_root,
            comfy_python,
        )?)
    } else {
        None
    };
    let data = data.unwrap_or_else(|| project.join(".imagine"));
    let (config, _lock) = Config::load(project, data, bind.parse()?)?;
    // Remove legacy/crash remnants before serving requests.
    imagine_server::session::purge(&config.data)?;
    let profiles = Arc::new(imagine_server::profiles::Catalog::load(
        &profiles.unwrap_or_else(|| config.project.join("profiles")),
    )?);
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    let store = Store::open(&config.data).await?;
    let (engine_tx, engine_rx) = watch::channel(EngineState {
        backend: backend.name().into(),
        phase: "unloaded".into(),
        pid: None,
        ..Default::default()
    });
    let library = Arc::new(std::sync::Mutex::new(
        imagine_server::model_library::Library::load(
            config.model_root.clone(),
            config.data.join("model-library.json"),
        )?,
    ));
    let engine: Box<dyn Backend> = if backend == EngineKind::Rust {
        Box::new(Comfy::new_rust(
            config.clone(),
            engine_tx,
            &rust_runtime.unwrap_or_else(|| config.project.join("comfyui/runtime-current.json")),
        )?)
    } else {
        match comfy_paths {
            Some((python, root)) => Box::new(Comfy::new(config.clone(), engine_tx, python, root)?),
            None => Box::new(Native::new(config.clone(), engine_tx)?),
        }
    };
    let (control, mut scheduler) = scheduler::start_coordinated(
        store.clone(),
        engine,
        std::env::var_os("IMAGINE_COORDINATOR_SOCKET").map(PathBuf::from),
    )
    .await?;
    let app = App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(
            config.data.join("enhancement-profiles.json"),
        )?,
        library,
        profiles,
        store: store.clone(),
        control: control.clone(),
        engine: engine_rx,
        uploads: Arc::new(Semaphore::new(2)),
    };
    eprintln!("Imagine listening on http://{}", listener.local_addr()?);
    let stop = tokio_util::sync::CancellationToken::new();
    let stop_http = stop.clone();
    let mut server = tokio::spawn(async move {
        axum::serve(listener, api::router(app))
            .with_graceful_shutdown(stop_http.cancelled_owned())
            .await
    });
    let outcome: Result<()> = tokio::select! {
        _=shutdown_signal()=> {
            stop.cancel();
            server.abort();
            let _ = (&mut server).await;
            let shutdown = control.call(Action::Shutdown).await;
            let result = scheduler.await;
            shutdown.map_err(|e| anyhow::anyhow!(e.code))
                .and_then(|_| result.context("scheduler task").and_then(|r| r))
        },
        result=&mut scheduler=> {
            stop.cancel(); server.abort(); let _ = server.await;
            result.context("scheduler task").and_then(|r| r)
        },
        result=&mut server=> {
            let _=control.call(Action::Shutdown).await;
            let stopped = scheduler.await.context("scheduler task").and_then(|r| r);
            stopped.and_then(|_| result.context("HTTP task").and_then(|r| r.map_err(Into::into)))
        }
    };
    // All writers are stopped before removing assets and closing the RAM database.
    let cleaned = store.close().await;
    outcome.and(cleaned)
}

fn resolve_comfy_paths(
    project: &Path,
    home_dir: Option<&Path>,
    root: Option<PathBuf>,
    python: Option<PathBuf>,
) -> Result<(PathBuf, PathBuf)> {
    let root = root
        .or_else(|| {
            std::iter::once(project.join("comfyui"))
                .chain(home_dir.map(|home| home.join("git/ComfyUI")))
                .find(|candidate| candidate.join("comfy/sd.py").is_file())
        })
        .context("ComfyUI source not found; specify --comfy-root PATH")?;
    let root = std::path::absolute(root)?;
    if !root.join("comfy/sd.py").is_file() {
        anyhow::bail!(
            "ComfyUI source missing at {}; specify --comfy-root PATH",
            root.display()
        );
    }
    // Keep the venv path rather than resolving its executable symlink.
    let python = std::path::absolute(python.unwrap_or_else(|| root.join(".venv/bin/python")))?;
    if !python.is_file() {
        anyhow::bail!(
            "Comfy Python missing at {}; specify --comfy-python PATH",
            python.display()
        );
    }
    Ok((python, root))
}

async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler");
    tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn installation(root: &Path) {
        std::fs::create_dir_all(root.join("comfy")).unwrap();
        std::fs::create_dir_all(root.join(".venv/bin")).unwrap();
        std::fs::write(root.join("comfy/sd.py"), b"synthetic source marker").unwrap();
        std::fs::write(
            root.join(".venv/bin/python"),
            b"synthetic executable marker",
        )
        .unwrap();
    }

    #[test]
    fn discovers_home_checkout_and_prefers_project_checkout() {
        let directory = tempfile::tempdir().unwrap();
        let project = directory.path().join("project");
        let home_dir = directory.path().join("home");
        let home_root = home_dir.join("git/ComfyUI");
        installation(&home_root);
        assert_eq!(
            resolve_comfy_paths(&project, Some(&home_dir), None, None).unwrap(),
            (home_root.join(".venv/bin/python"), home_root)
        );
        let project_root = project.join("comfyui");
        installation(&project_root);
        assert_eq!(
            resolve_comfy_paths(&project, Some(&home_dir), None, None).unwrap(),
            (project_root.join(".venv/bin/python"), project_root)
        );
    }

    #[test]
    fn explicit_paths_override_discovery_and_preserve_venv_symlink() {
        let directory = tempfile::tempdir().unwrap();
        let project = directory.path().join("project");
        installation(&project.join("comfyui"));
        let root = directory.path().join("chosen");
        installation(&root);
        let python = directory.path().join("python-alternative");
        std::os::unix::fs::symlink(root.join(".venv/bin/python"), &python).unwrap();
        assert_eq!(
            resolve_comfy_paths(&project, None, Some(root.clone()), Some(python.clone())).unwrap(),
            (python, root)
        );
    }

    #[test]
    fn missing_installation_has_actionable_errors_without_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let project = directory.path();
        assert!(
            resolve_comfy_paths(project, None, None, None)
                .unwrap_err()
                .to_string()
                .contains("--comfy-root")
        );
        let root = project.join("comfyui");
        installation(&root);
        std::fs::remove_file(root.join(".venv/bin/python")).unwrap();
        assert!(
            resolve_comfy_paths(project, None, None, None)
                .unwrap_err()
                .to_string()
                .contains("--comfy-python")
        );
        assert!(
            resolve_comfy_paths(project, None, Some(project.join("missing")), None)
                .unwrap_err()
                .to_string()
                .contains("--comfy-root")
        );
    }
}
