//! Own GPU acceptance only: unchanged PipeBackend/Scheduler, raw AV capture before export.
use imagine_server::{
    api::{self, App},
    comfy_engine::Comfy,
    config::Config,
    engine::{Backend, EngineFuture, EngineState},
    generation::ImagePass,
    h3_protocol,
    scheduler::{self, Action},
    short_av::Av,
    storage::Store,
    video::reference_cache::Control,
};
use std::{io::Write, path::PathBuf, sync::Arc};
use tokio::sync::{Semaphore, watch};
#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;
struct Probe {
    inner: Comfy,
    raw: PathBuf,
}
impl Backend for Probe {
    fn reference_cache(&self) -> Option<Arc<dyn Control>> {
        self.inner.reference_cache()
    }
    fn generate<'a>(&'a mut self, r: &'a ImagePass) -> EngineFuture<'a, Vec<u8>> {
        self.inner.generate(r)
    }
    fn generate_video<'a>(&'a mut self, r: &'a h3_protocol::Request) -> EngineFuture<'a, Av> {
        Box::pin(async move {
            let av = self.inner.generate_video(r).await?;
            let folder = self.raw.join(&r.id);
            std::fs::create_dir(&folder)?;
            let mut rgb = std::fs::File::create_new(folder.join("frames.rgb"))?;
            for frame in &av.frames {
                rgb.write_all(frame)?;
            }
            rgb.flush()?;
            std::fs::write(folder.join("audio.f32le"), &av.pcm)?;
            std::fs::write(
                folder.join("contract.json"),
                serde_json::to_vec_pretty(
                    &serde_json::json!({"id":r.id,"shape":av.shape,"frames":av.frames.len(),"rgb_format":"HWC RGB8 wire before export","pcm_format":"stereo f32le before export","pcm_bytes":av.pcm.len(),"cache":self.inner.reference_cache().unwrap().status().map_err(anyhow::Error::msg)?}),
                )?,
            )?;
            Ok(av)
        })
    }
    fn stage(&mut self, s: Option<&str>) {
        self.inner.stage(s)
    }
    fn mark_idle(&mut self) {
        self.inner.mark_idle()
    }
    fn begin_job(&mut self, id: &str) {
        self.inner.begin_job(id)
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        self.inner.stop()
    }
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let root = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .ok_or_else(|| anyhow::anyhow!("own case root required"))?,
    )
    .canonicalize()?;
    anyhow::ensure!(
        task_fixtures::is_owned(&root),
        "own relocated synthetic task root required"
    );
    let destination: imagine_server::video_destination::Config =
        serde_json::from_slice(&std::fs::read(root.join("state/video-output.json"))?)?;
    let output = imagine_server::video::output::Destination::new(destination);
    output.proof().map_err(anyhow::Error::msg)?;
    std::fs::create_dir_all(root.join("raw"))?;
    let config = Config {
        project: root.join("project"),
        data: root.join("state"),
        bind: "127.0.0.1:0".parse()?,
        model_root: root.join("models"),
        private_storage: None,
    };
    let mut store = Store::open(&config.data).await?;
    store.videos.destination = Some(output);
    let (tx, engine) = watch::channel(EngineState {
        backend: "rust".into(),
        ..Default::default()
    });
    let inner = Comfy::new_rust(config.clone(), tx, &root.join("runtime.json"))?;
    let coordinator = PathBuf::from(
        std::env::var_os("IMAGINE_COORDINATOR_SOCKET")
            .ok_or_else(|| anyhow::anyhow!("normal coordinator required"))?,
    );
    let (control, scheduler) = scheduler::start_coordinated(
        store.clone(),
        Box::new(Probe {
            inner,
            raw: root.join("raw"),
        }),
        Some(coordinator),
    )
    .await?;
    let app = App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(
            config.data.join("enhancement.json"),
        )?,
        library: if std::env::args_os().any(|a| a == "--own-image-catalog") {
            Arc::new(std::sync::Mutex::new(
                imagine_server::model_library::Library::load(
                    root.join("models"),
                    root.join("state/model-library.json"),
                )?,
            ))
        } else {
            Default::default()
        },
        profiles: Arc::new(imagine_server::profiles::Catalog::load(
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/profiles")),
        )?),
        store: store.clone(),
        control: control.clone(),
        engine,
        uploads: Arc::new(Semaphore::new(2)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    std::fs::write(
        root.join("evidence/endpoint.json"),
        serde_json::to_vec(
            &serde_json::json!({"api":format!("http://{}",listener.local_addr()?),"backend":"real H3 with own pre-export wire capture"}),
        )?,
    )?;
    let mut server = tokio::spawn(async move { axum::serve(listener, api::router(app)).await });
    tokio::select! {value=&mut server=>{value??;}, _=tokio::signal::ctrl_c()=>{server.abort();let _=server.await;}}
    control
        .call(Action::Shutdown)
        .await
        .map_err(|e| anyhow::anyhow!(e.code))?;
    scheduler.await??;
    store.close().await?;
    Ok(())
}
