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
    let mut project = std::env::current_dir()?;
    let mut data = None;
    let mut profiles = None;
    let mut bind = "127.0.0.1:8080".to_owned();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
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
        store,
        control: control.clone(),
        engine: engine_rx,
        access: Arc::new(config.access.clone()),
        uploads: Arc::new(Semaphore::new(2)),
    };
    eprintln!(
        "Imagine listening on http://{}; credentials: {}",
        listener.local_addr()?,
        config.data.join("access.json").display()
    );
    let stop = tokio_util::sync::CancellationToken::new();
    let stop_http = stop.clone();
    let mut server = tokio::spawn(async move {
        axum::serve(listener, api::router(app))
            .with_graceful_shutdown(stop_http.cancelled_owned())
            .await
    });
    tokio::select! {
        _=shutdown_signal()=>{},
        result=&mut scheduler=>{stop.cancel();server.abort();return result.context("scheduler task")?;},
        result=&mut server=>{let _=control.call(Action::Shutdown).await; scheduler.await??;return result.context("HTTP task")?.map_err(Into::into);}
    }
    // Close SSE/connections promptly; persisted job reads are available after restart.
    stop.cancel();
    server.abort();
    control
        .call(Action::Shutdown)
        .await
        .map_err(|e| anyhow::anyhow!(e.code))?;
    scheduler.await??;
    Ok(())
}
async fn shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("SIGTERM handler");
    tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
}
