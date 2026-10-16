use anyhow::{Context, Result};
use imagine_server::{
    api::{self, App},
    config::Config,
    engine::{EngineState, Native},
    scheduler::{self, Action},
    storage::Store,
};
use std::{path::PathBuf, sync::Arc};
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
    let mut project = std::env::current_dir()?;
    let mut data = None;
    let mut profiles = None;
    let mut bind = "127.0.0.1:8080".to_owned();
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
            "--help" => {
                println!(
                    "imagine-server [--project PATH] [--data PATH] [--profiles PATH] [--bind 127.0.0.1:8080]"
                );
                return Ok(());
            }
            _ => anyhow::bail!("Unknown option: {arg}"),
        }
    }
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
        phase: "unloaded".into(),
        pid: None,
        ..Default::default()
    });
    let native = Native::new(config.clone(), engine_tx)?;
    let (control, mut scheduler) = scheduler::start_coordinated(
        store.clone(),
        Box::new(native),
        std::env::var_os("IMAGINE_COORDINATOR_SOCKET").map(PathBuf::from),
    )
    .await?;
    let app = App {
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

async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler");
    tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
}
