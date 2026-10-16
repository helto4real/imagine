use imagine_server::{
    api::{self, App},
    config::Access,
    engine::{Backend, EngineFuture, EngineState},
    generation::ImagePass,
    scheduler::{self, Action},
    storage::Store,
};
use imagine_ui::{
    client::{
        Client, OutputRole, Progress, Submission, make_request, make_sheet_request,
        make_upscale_request,
    },
    settings::{Connection, GenerationSettings, UpscaleTarget},
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::sync::{Semaphore, watch};

struct Fake {
    state: watch::Sender<EngineState>,
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<ImagePass>>>,
    block_folder: Arc<Mutex<Option<PathBuf>>>,
}
fn png() -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    image::RgbImage::from_pixel(64, 64, image::Rgb([122, 94, 70]))
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();
    buf.into_inner()
}
impl Backend for Fake {
    fn begin_job(&mut self, id: &str) {
        self.state.send_replace(EngineState {
            job_id: Some(id.into()),
            phase: "inference".into(),
            ..Default::default()
        });
    }
    fn stage(&mut self, stage: Option<&str>) {
        self.state.send_modify(|s| {
            s.stage = stage.map(str::to_owned);
            s.progress = None;
        });
    }
    fn generate<'a>(&'a mut self, req: &'a ImagePass) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.requests.lock().unwrap().push(req.clone());
            self.state.send_modify(|state| {
                state.phase = "sampling".into();
                state.progress = Some(imagine_server::progress::SamplingProgress {
                    completed: 6,
                    total: req.sampling.steps,
                    elapsed_seconds: 6.,
                    remaining_seconds: (req.sampling.steps - 6) as f64,
                    updated_at_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                });
            });
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            if let Some(folder) = self.block_folder.lock().unwrap().take() {
                std::fs::remove_dir(&folder)?;
                std::fs::write(folder, b"blocked")?;
            }
            if self.state.borrow().stage.is_some() || req.model == "qwen-image-2.1-turbo" {
                let mut buf = Cursor::new(Vec::new());
                image::RgbImage::from_pixel(req.width, req.height, image::Rgb([122, 94, 70]))
                    .write_to(&mut buf, image::ImageFormat::Png)?;
                Ok(buf.into_inner())
            } else {
                Ok(png())
            }
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
#[tokio::test]
async fn client_uses_proxy_api_defaults_references_retry_and_output_folder() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).await.unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let blocker = Arc::new(Mutex::new(None));
    let (engine_tx, engine) = watch::channel(EngineState::default());
    let socket = dir.path().join("gpu.sock");
    let (control, scheduler) = scheduler::start_coordinated(
        store.clone(),
        Box::new(Fake {
            state: engine_tx,
            calls: calls.clone(),
            requests: requests.clone(),
            block_folder: blocker.clone(),
        }),
        Some(socket.clone()),
    )
    .await
    .unwrap();
    let access = Access {
        admin_token: "unused-admin-token".into(),
        clients: BTreeMap::from([
            ("ui-test".into(), "synthetic-test-token".into()),
            ("other".into(), "other-token".into()),
        ]),
    };
    let access_path = dir.path().join("access.json");
    std::fs::write(&access_path, serde_json::to_vec(&access).unwrap()).unwrap();
    // Keep the shipped upscale settings, with only synthetic plaintext prompts.
    let profiles = dir.path().join("profiles");
    std::fs::create_dir(&profiles).unwrap();
    let profile = include_str!("../../profiles/image_upscale.yaml")
        .lines()
        .map(|line| {
            if line.starts_with("  prompt:") {
                "  prompt: Synthetic upscale fixture"
            } else if line.starts_with("  negative_prompt:") {
                "  negative_prompt: ''"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(profiles.join("image_upscale.yaml"), profile).unwrap();
    let app = App {
        profiles: Arc::new(imagine_server::profiles::Catalog::load(&profiles).unwrap()),
        store,
        control: control.clone(),
        engine,
        access: Arc::new(access),
        uploads: Arc::new(Semaphore::new(2)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server =
        tokio::spawn(async move { axum::serve(listener, api::router(app)).await.unwrap() });
    let backend_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_url = format!("http://{}", backend_listener.local_addr().unwrap());
    let backend_server = tokio::spawn(async move {
        axum::serve(backend_listener, axum::Router::new()
            .route("/health", axum::routing::get(|| async { axum::Json(json!({"service":"strata","status":"ok","loaded":false,"model":"synthetic"})) }))
            .route("/v1/models", axum::routing::get(|| async { axum::Json(json!({"data":[{"id":"synthetic"}]})) }))
            .route("/free", axum::routing::post(|| async { axum::Json(json!({})) })))
            .await.unwrap();
    });
    let proxy = st_proxy::service::Service::start(st_proxy::config::Config {
        chat_port: 0,
        image_port: 0,
        llm_backend: "sc117".into(),
        llm_url: backend_url.clone(),
        comfy_url: backend_url,
        imagine_url: url,
        imagine_access_file: Some(access_path.to_string_lossy().into()),
        coordinator_socket: Some(socket.to_string_lossy().into()),
        poll_interval: 0.01,
        ..Default::default()
    })
    .await
    .unwrap();
    let url = format!("http://127.0.0.1:{}/imagine", proxy.image_port);
    let http = reqwest::Client::new();
    let unauthorized = http.get(format!("{url}/v1/models")).send().await.unwrap();
    assert_eq!(unauthorized.status(), 401);
    assert_eq!(unauthorized.headers()["cache-control"], "no-store");
    let denied = http
        .post(format!("{url}/v1/runtime/unload"))
        .bearer_auth("synthetic-test-token")
        .json(&json!({"active_job":"wait"}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        denied.status(),
        403,
        "proxy must never substitute admin credentials"
    );
    let client = Client::connect(&Connection {
        url: url.clone(),
        access: access_path,
        client: "ui-test".into(),
    })
    .unwrap();
    let meta = client.metadata().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let ref_path = dir.path().join("reference.png");
    std::fs::write(&ref_path, png()).unwrap();
    let a = client.upload(&ref_path).await.unwrap();
    let b = client.upload(&ref_path).await.unwrap();
    let other_asset = http
        .get(format!("{url}/v1/assets/{}", a.id))
        .bearer_auth("other-token")
        .send()
        .await
        .unwrap();
    assert_eq!(other_asset.status(), 404);
    let refs = vec![b.id, a.id];
    let settings = GenerationSettings {
        steps: 24,
        cfg_scale: 3.25,
        random_seed: false,
        seed: u32::MAX,
        aspect_ratio: "3:4".into(),
        max_side: 1024,
        reference_max_pixels_per_image: 1_000_000,
    };
    let request = make_request("  Exact prompt  ", "negative", &refs, &meta, &settings).unwrap();
    assert_eq!(request["operation"], "image_edit");
    assert_eq!(request["inputs"]["reference_images"], json!(refs));
    assert_eq!(request["sampling"]["steps"], 24);
    assert_eq!(
        request["model_options"]["qwen_image_2_1"]["reference_max_pixels_per_image"],
        1_000_000
    );
    let submission = Submission {
        request,
        key: "same-submission".into(),
        folder: dir.path().join("images"),
        job: None,
    };
    let mut id = None;
    let mut observed_progress = Vec::new();
    let first = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(value) = p {
                id = Some(value)
            } else if let Progress::Status {
                sampling: Some(sampling),
                ..
            } = p
            {
                observed_progress.push(sampling);
            }
        })
        .await
        .unwrap();
    assert_eq!(std::fs::read(&first.assets[0].file).unwrap(), png());
    assert_eq!(first.assets[0].file.parent().unwrap(), submission.folder);
    assert_eq!(requests.lock().unwrap()[0].prompt, "  Exact prompt  ");
    assert_eq!(first.seed, Some(u32::MAX as u64));
    assert!(!observed_progress.is_empty());
    assert!(observed_progress.iter().all(|p| p.completed == 6
        && p.total == 24
        && p.remaining_seconds == 18.
        && p.finish_time().is_some()));
    {
        let requests = requests.lock().unwrap();
        let received = &requests[0];
        assert_eq!((received.width, received.height), (768, 1024));
        assert_eq!(received.sampling.steps, 24);
        assert_eq!(received.sampling.cfg_scale, 3.25);
    }
    let hidden = http
        .get(format!("{url}/v1/jobs/{}", id.as_ref().unwrap()))
        .bearer_auth("other-token")
        .send()
        .await
        .unwrap();
    assert_eq!(hidden.status(), 404);
    let events = http
        .get(format!("{url}/v1/jobs/{}/events", id.as_ref().unwrap()))
        .bearer_auth("synthetic-test-token")
        .send()
        .await
        .unwrap();
    assert_eq!(events.headers()["content-type"], "text/event-stream");
    let events = events.text().await.unwrap();
    assert!(events.contains("succeeded"));
    let duplicate = client.generate(submission, |_| {}).await.unwrap();
    assert_ne!(first.assets[0].file, duplicate.assets[0].file);
    assert_eq!(duplicate.seed, first.seed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // Disk failure after generation: changing output folder retries retrieval, not inference.
    let blocked = dir.path().join("blocked-output");
    *blocker.lock().unwrap() = Some(blocked.clone());
    let mut submission = Submission {
        request: make_request("another", "", &[], &meta, &GenerationSettings::default()).unwrap(),
        key: "disk-error".into(),
        folder: blocked,
        job: None,
    };
    let mut job = None;
    let err = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(id) = p {
                job = Some(id)
            }
        })
        .await
        .err()
        .unwrap();
    assert!(err.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    submission.job = job;
    submission.folder = dir.path().join("recovered-output");
    let recovered = client.generate(submission, |_| {}).await.unwrap();
    assert!(recovered.assets[0].file.is_file());
    assert!(recovered.seed.is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let cancelled = client.cancel(id.as_ref().unwrap()).await.err().unwrap();
    assert!(!cancelled.retryable);
    assert!(make_request(" ", "", &[], &meta, &GenerationSettings::default()).is_err());
    assert!(
        make_request(
            "x",
            "",
            &vec!["ref".into(); 11],
            &meta,
            &GenerationSettings::default()
        )
        .is_err()
    );
    // Complete sheet via the real proxy/API, using only a synthetic backend.
    let request = make_sheet_request(
        "Synthetic identity",
        "Green dress",
        512,
        "negative",
        &[],
        &meta,
        &settings,
    )
    .unwrap();
    let submission = Submission {
        request,
        key: "text-sheet".into(),
        folder: dir.path().join("text-sheet"),
        job: None,
    };
    let mut stages = Vec::new();
    let completed = client
        .generate(submission.clone(), |p| {
            if let Progress::Status { label, .. } = p {
                stages.push(label);
            }
        })
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert!(stages.iter().any(|s| s == "Portrait · Sampling"));
    assert!(stages.iter().any(|s| s == "Views · Sampling"));
    assert_eq!(
        completed.assets.iter().map(|a| a.role).collect::<Vec<_>>(),
        [OutputRole::Sheet, OutputRole::Portrait, OutputRole::Views]
    );
    for asset in &completed.assets {
        assert_eq!(asset.file.parent().unwrap(), submission.folder);
        assert!(
            asset
                .file
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with(&format!("-{}.png", asset.role.name()))
        );
        image::load_from_memory(&std::fs::read(&asset.file).unwrap()).unwrap();
    }
    assert_eq!(
        (
            completed.assets[0].preview.width,
            completed.assets[0].preview.height
        ),
        (512, 288)
    );
    assert_eq!(std::fs::read_dir(&submission.folder).unwrap().count(), 3);
    // Reference sheet: failed saving retries the completed job, not either inference pass.
    let blocked = dir.path().join("blocked-sheet");
    *blocker.lock().unwrap() = Some(blocked.clone());
    let mut submission = Submission {
        request: make_sheet_request("", "Blue dress", 512, "", &refs, &meta, &settings).unwrap(),
        key: "reference-sheet".into(),
        folder: blocked,
        job: None,
    };
    let mut job = None;
    let error = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(id) = p {
                job = Some(id)
            }
        })
        .await
        .err()
        .unwrap();
    assert!(error.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    submission.job = job;
    submission.folder = dir.path().join("recovered-sheet");
    let saved = client.generate(submission.clone(), |_| {}).await.unwrap();
    assert_eq!(saved.assets.len(), 3);
    assert_eq!(std::fs::read_dir(&submission.folder).unwrap().count(), 3);
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    {
        let requests = requests.lock().unwrap();
        assert_eq!(requests[4].references.len(), 2);
        assert_eq!(requests[5].references.len(), 4);
    }
    // Cancellation returns a terminal result and saves no partial sheet.
    let submission = Submission {
        request: make_sheet_request("Cancel test", "", 512, "", &[], &meta, &settings).unwrap(),
        key: "cancel-sheet".into(),
        folder: dir.path().join("cancel-sheet"),
        job: None,
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let generation = client.generate(submission.clone(), |p| {
        if let Progress::Submitted(id) = p {
            tx.send(id).unwrap()
        }
    });
    let cancel = async {
        let id = rx.recv().await.unwrap();
        client.cancel(&id).await.unwrap();
    };
    let (result, ()) = tokio::join!(generation, cancel);
    let error = result.err().unwrap();
    assert!(!error.retryable);
    assert_eq!(error.message, "Generation cancelled.");
    assert_eq!(std::fs::read_dir(&submission.folder).unwrap().count(), 0);
    // Both UI workspaces use registered output assets and the same upscale profile.
    for (source, target, ratio, dimensions) in [
        (&first.assets[0], UpscaleTarget::K3, "3:4", (2304, 3072)),
        (
            &completed.assets[0],
            UpscaleTarget::K2,
            "16:9",
            (2048, 1152),
        ),
    ] {
        let source_bytes = std::fs::read(&source.file).unwrap();
        let before = calls.load(Ordering::SeqCst);
        let blocked = dir.path().join(format!("blocked-upscale-{ratio}"));
        *blocker.lock().unwrap() = Some(blocked.clone());
        let mut submission = Submission {
            request: make_upscale_request(&source.asset_id, target, ratio, &settings).unwrap(),
            key: format!("upscale-{ratio}"),
            folder: blocked,
            job: None,
        };
        let mut job = None;
        let error = client
            .generate(submission.clone(), |p| {
                if let Progress::Submitted(id) = p {
                    job = Some(id);
                }
            })
            .await
            .err()
            .unwrap();
        assert!(error.retryable);
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        submission.job = job;
        submission.folder = source.file.parent().unwrap().into();
        let upscaled = client.generate(submission, |_| {}).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        assert_eq!(upscaled.assets.len(), 1);
        let output = &upscaled.assets[0];
        assert_ne!(output.asset_id, source.asset_id);
        assert_ne!(output.file, source.file);
        assert_eq!(std::fs::read(&source.file).unwrap(), source_bytes);
        let saved = image::load_from_memory(&std::fs::read(&output.file).unwrap()).unwrap();
        assert_eq!((saved.width(), saved.height()), dimensions);
        let requests = requests.lock().unwrap();
        let request = requests.last().unwrap();
        assert_eq!((request.width, request.height), dimensions);
        assert_eq!(request.model, "qwen-image-2.1-turbo");
        assert_eq!(request.sampling.steps, 8);
        assert_eq!(request.sampling.cfg_scale, 1.);
        assert_eq!(request.sampling.sampler, "euler");
        assert_eq!(request.sampling.scheduler, "simple");
        assert_eq!(request.references.len(), 1);
        assert_eq!(request.references[0].as_ref(), source_bytes);
    }
    control.call(Action::Shutdown).await.unwrap();
    scheduler.await.unwrap().unwrap();
    proxy.close().await;
    server.abort();
    backend_server.abort();
}

#[tokio::test]
async fn incomplete_sheet_download_saves_nothing_and_retry_reuses_job() {
    use axum::{
        Router,
        extract::{Path, State},
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
    };
    use std::sync::atomic::AtomicBool;
    #[derive(Clone)]
    struct DownloadState {
        fail: Arc<AtomicBool>,
        submitted: Arc<AtomicUsize>,
    }
    async fn submit(State(s): State<DownloadState>) -> axum::Json<serde_json::Value> {
        s.submitted.fetch_add(1, Ordering::SeqCst);
        axum::Json(json!({"id":"job_sheet"}))
    }
    async fn asset(
        State(s): State<DownloadState>,
        Path(id): Path<String>,
    ) -> axum::response::Response {
        if id == "views" && s.fail.load(Ordering::SeqCst) {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        ([("content-type", "image/png")], png()).into_response()
    }
    let root = tempfile::tempdir().unwrap();
    let access = root.path().join("access.json");
    std::fs::write(&access, br#"{"clients":{"test":"synthetic-token"}}"#).unwrap();
    let state = DownloadState {
        fail: Arc::new(AtomicBool::new(true)),
        submitted: Arc::new(AtomicUsize::new(0)),
    };
    let router=Router::new().route("/v1/jobs",post(submit)).route("/v1/jobs/{id}",get(||async {
        axum::Json(json!({"status":"succeeded","effective":{"seed":42},"output":{"assets":[{"id":"views","role":"turnaround"},{"id":"sheet","role":"character_sheet"},{"id":"portrait","role":"portrait"}]}}))
    })).route("/v1/assets/{id}",get(asset)).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let client = Client::connect(&Connection {
        url,
        access,
        client: "test".into(),
    })
    .unwrap();
    let mut submission = Submission {
        request: json!({"operation":"character_sheet"}),
        key: "download-sheet".into(),
        folder: root.path().join("outputs"),
        job: None,
    };
    let mut job = None;
    let failure = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(id) = p {
                job = Some(id)
            }
        })
        .await
        .err()
        .unwrap();
    assert!(failure.retryable);
    assert_eq!(std::fs::read_dir(&submission.folder).unwrap().count(), 0);
    submission.job = job;
    state.fail.store(false, Ordering::SeqCst);
    let completed = client.generate(submission.clone(), |_| {}).await.unwrap();
    assert_eq!(completed.assets.len(), 3);
    assert_eq!(std::fs::read_dir(&submission.folder).unwrap().count(), 3);
    assert_eq!(state.submitted.load(Ordering::SeqCst), 1);
    server.abort();
}
