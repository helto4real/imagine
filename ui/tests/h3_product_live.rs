//! Explicit source-owned synthetic GPU test through the actual UI client/API/queue.
//! No existing product state, private media, model discovery, restart or playback.
use anyhow::{Context, Result, ensure};
use imagine_server::{
    api::{self, App},
    comfy_engine::Comfy,
    config::Config,
    engine::EngineState,
    scheduler::{self, Action},
    storage::Store,
};
use imagine_ui::{
    client::{Client, Submission, video_request},
    settings::Connection,
};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::{Semaphore, watch};
#[tokio::test]
#[ignore = "requires explicit GPU authorization and an own prepared H3 runtime"]
async fn own_prompt_uses_product_queue_and_persists_av() -> Result<()> {
    let manifest = PathBuf::from(
        std::env::var_os("IMAGINE_H3_TEST_RUNTIME").context("own H3 test manifest required")?,
    );
    ensure!(manifest.starts_with("/home/thhel/.codex/worktrees/h3-product-worker/imagine/comfyui/h3-product-20261020/manifests"),"test requires own manifest");
    let dir = tempfile::tempdir()?;
    let store = Store::open(dir.path()).await?;
    let (state_tx, state_rx) = watch::channel(EngineState {
        backend: "rust".into(),
        ..Default::default()
    });
    let config = Config {
        project: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .into(),
        data: dir.path().join("own-data"),
        bind: "127.0.0.1:0".parse()?,
        model_root: dir.path().join("unused-image-models"),
    };
    let observed_state = state_rx.clone();
    let engine = Comfy::new_rust(config, state_tx, &manifest)?;
    let profile_dir = dir.path().join("profiles");
    std::fs::create_dir(&profile_dir)?;
    std::fs::write(
        profile_dir.join("synthetic.yaml"),
        serde_json::to_vec(&serde_json::json!({
            "schema_version":1,"id":"synthetic","version":1,"name":"Synthetic only","description":"Own test fixture",
            "defaults":{"model":"qwen-image-2.1-int8","negative_prompt":"SYNTHETIC"}
        }))?,
    )?;
    let profiles = Arc::new(imagine_server::profiles::Catalog::load(&profile_dir)?);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let url = format!("http://{}", listener.local_addr()?);
    let (control, task) = scheduler::start_coordinated(
        store.clone(),
        Box::new(engine),
        Some(PathBuf::from(
            "/home/thhel/ChatGPT/imagine/.imagine/gpu.sock",
        )),
    )
    .await?;
    let router = api::router(App {
        video: control.video.clone(),
        store: store.clone(),
        control: control.clone(),
        engine: state_rx,
        profiles,
        library: Arc::new(std::sync::Mutex::new(
            imagine_server::model_library::Library::default(),
        )),
        enhancement: imagine_server::enhancement::Service::open(
            dir.path().join("own-enhancement.json"),
        )?,
        uploads: Arc::new(Semaphore::new(2)),
    });
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let result=async {
        let client=Client::connect(&Connection{url})?;ensure!(client.video_available().await.map_err(|_|anyhow::anyhow!("H3 unavailable"))?);
        let prompt="A polished red wooden sphere rolls gently across a grey stone surface in a quiet room.";
        let folder=dir.path().join("explicit-output");
        let submission=Submission{request:video_request(prompt),key:uuid::Uuid::new_v4().to_string(),folder:folder.clone(),job:None,upscale_source:None};
        let mut submitted=std::collections::HashSet::new();
        let completed=tokio::time::timeout(Duration::from_secs(600),client.generate_video(submission,|event|{if let imagine_ui::client::Progress::Submitted(id)=event {submitted.insert(id);}})).await?.map_err(|e|anyhow::anyhow!("product generation failed: {}",e.message))?;
        ensure!(submitted.len()==1,"must keep one admitted job identity");
        let job_id=submitted.iter().next().unwrap();
        let job=store.job(job_id,"local").await?.context("own job missing")?;
        ensure!(job["status"]=="succeeded" && job["output"]["format"]=="mp4");
        ensure!(completed.av.valid());ensure!(completed.saved.starts_with(&folder));ensure!(completed.saved.is_file());
        ensure!(completed.av.frames.len()==39 && completed.av.pcm.len()==52000*8);
        println!("Product H3 PASS: UI client -> API -> canonical Ticket/FIFO -> real worker -> persistent AV; 39 frames / 52000 stereo samples; one GPU job.");
        Ok::<_,anyhow::Error>(())
    }.await;
    // Always finish scheduler cleanup before propagating a failed check.
    let stopped = control.call(Action::Shutdown).await;
    let joined = task.await;
    server.abort();
    let _ = server.await;
    ensure!(stopped.is_ok(), "scheduler cleanup failed");
    joined??;
    ensure!(
        observed_state.borrow().pid.is_none(),
        "worker must be reaped at shutdown"
    );
    result
}
