#[path = "support/video_output.rs"]
mod video_output;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use imagine_server::{
    api::{self, App},
    contract::MODEL,
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action, Control},
    storage::Store,
};
use serde_json::{Value, json};
use std::{
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::{
    sync::{Notify, Semaphore, watch},
    task::JoinHandle,
};
use tower::ServiceExt;

#[derive(Default)]
struct Probe {
    component_failure:
        std::sync::Mutex<Option<imagine_server::component_error::ComponentLoadError>>,
    video_calls: AtomicUsize,
    video_inputs: std::sync::Mutex<Vec<imagine_server::h3_protocol::Request>>,
    video_prompts: std::sync::Mutex<Vec<String>>,
    calls: AtomicUsize,
    stops: AtomicUsize,
    fail: AtomicUsize,
    release: Notify,
    sized_output: AtomicUsize,
    passes: std::sync::Mutex<Vec<imagine_server::generation::ImagePass>>,
    stages: std::sync::Mutex<Vec<String>>,
}
struct Fake(Arc<Probe>);
fn png() -> Vec<u8> {
    let mut b = Cursor::new(Vec::new());
    image::RgbImage::from_pixel(2, 2, image::Rgb([42, 50, 60]))
        .write_to(&mut b, image::ImageFormat::Png)
        .unwrap();
    b.into_inner()
}
impl Backend for Fake {
    fn generate<'a>(
        &'a mut self,
        pass: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            self.0.passes.lock().unwrap().push(pass.clone());
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            self.0.release.notified().await;
            if let Some(error) = self.0.component_failure.lock().unwrap().take() {
                return Err(error.into());
            }
            if self.0.fail.swap(0, Ordering::SeqCst) > 0 {
                anyhow::bail!("simulated crash");
            }
            if self.0.sized_output.load(Ordering::SeqCst) != 0 {
                let mut bytes = Cursor::new(Vec::new());
                image::RgbImage::from_pixel(pass.width, pass.height, image::Rgb([42, 50, 60]))
                    .write_to(&mut bytes, image::ImageFormat::Png)?;
                Ok(bytes.into_inner())
            } else {
                Ok(png())
            }
        })
    }
    fn generate_video<'a>(
        &'a mut self,
        input: &'a imagine_server::h3_protocol::Request,
    ) -> EngineFuture<'a, imagine_server::short_av::Av> {
        Box::pin(async move {
            self.0
                .video_prompts
                .lock()
                .unwrap()
                .push(input.prompt.clone());
            self.0.video_inputs.lock().unwrap().push(input.clone());
            self.0.video_calls.fetch_add(1, Ordering::SeqCst);
            self.0.release.notified().await;
            Ok(imagine_server::short_av::Av {
                continuation: None,
                shape: input.shape,
                frames: (0..input.shape.frames)
                    .map(|i| vec![i as u8; input.shape.frame_bytes()])
                    .collect(),
                pcm: vec![0; input.shape.audio_samples() as usize * 8],
            })
        })
    }
    fn stage(&mut self, stage: Option<&str>) {
        if let Some(stage) = stage {
            self.0.stages.lock().unwrap().push(stage.into());
        }
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            self.0.stops.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}
struct Harness {
    library: Arc<std::sync::Mutex<imagine_server::model_library::Library>>,
    engine_state: watch::Sender<EngineState>,
    dir: tempfile::TempDir,
    store: Store,
    control: Control,
    task: JoinHandle<anyhow::Result<()>>,
    broker: Option<JoinHandle<()>>,
    router: Router,
    probe: Arc<Probe>,
}
async fn setup() -> Harness {
    setup_inner(false).await
}
async fn setup_inner(coordinated: bool) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).await.unwrap();
    store.videos.destination = Some(video_output::destination(dir.path()));
    let probe = Arc::new(Probe::default());
    let socket = dir.path().join("own-broker.sock");
    let broker = if coordinated {
        let listener = tokio::net::UnixListener::bind(&socket).unwrap();
        Some(tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let (read, mut write) = stream.into_split();
                    let mut read = BufReader::new(read);
                    let mut line = String::new();
                    if read.read_line(&mut line).await.unwrap_or(0) == 0 {
                        return;
                    }
                    if write
                        .write_all(b"{\"queued\":true}\n{\"granted\":true}\n")
                        .await
                        .is_err()
                    {
                        return;
                    }
                    line.clear();
                    let _ = read.read_line(&mut line).await;
                });
            }
        }))
    } else {
        None
    };
    let (control, task) = scheduler::start_coordinated(
        store.clone(),
        Box::new(Fake(probe.clone())),
        coordinated.then_some(socket),
    )
    .await
    .unwrap();
    let (engine_state, engine) = watch::channel(EngineState::default());

    let library = Arc::new(std::sync::Mutex::new(
        imagine_server::model_library::Library::default(),
    ));
    // Test only synthetic prompts; never resolve the production encrypted defaults.
    let profile_dir = dir.path().join("profiles");
    std::fs::create_dir(&profile_dir).unwrap();
    for entry in std::fs::read_dir(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/config/image/profiles"
    ))
    .unwrap()
    {
        let path = entry.unwrap().path();
        if path.extension().and_then(|value| value.to_str()) != Some("yaml") {
            continue;
        }
        let mut profile: serde_yaml::Value =
            serde_yaml::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        profile["defaults"]["negative_prompt"] = "synthetic profile negative".into();
        std::fs::write(
            profile_dir.join(path.file_name().unwrap()),
            serde_yaml::to_string(&profile).unwrap(),
        )
        .unwrap();
    }
    let router = api::router(App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(
            dir.path().join("enhancement-profiles.json"),
        )
        .unwrap(),
        library: library.clone(),
        profiles: Arc::new(imagine_server::profiles::Catalog::load(&profile_dir).unwrap()),
        store: store.clone(),
        control: control.clone(),
        engine,

        uploads: Arc::new(Semaphore::new(2)),
    });
    Harness {
        library,
        engine_state,
        dir,
        store,
        control,
        task,
        broker,
        router,
        probe,
    }
}

#[tokio::test]
async fn comfy_selection_is_visible_without_starting_inference() {
    let h = setup().await;
    h.engine_state
        .send_modify(|state| state.backend = "comfy".into());
    let (status, listing) = req(&h.router, "GET", "/v1/models", None, None).await;
    assert_eq!(status, 200);
    assert_eq!(listing["models"][0]["engine"], "ComfyUI");
    assert_eq!(listing["models"][0]["attention"], "comfy_kitchen_int8");
    assert_eq!(listing["models"][2]["available"], false);
    let (_, model) = req(&h.router, "GET", &format!("/v1/models/{MODEL}"), None, None).await;
    assert_eq!(model["live_validated"], false);
    assert_eq!(model["constraints"]["validated_resolutions"], json!([]));
    let (_, runtime) = req(&h.router, "GET", "/v1/runtime", None, None).await;
    assert_eq!(runtime["engine"]["backend"], "comfy");
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}

#[tokio::test]
async fn enhancement_routes_share_session_guard_and_never_start_image_inference() {
    let h = setup().await;
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/enhancement/profiles")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["cache-control"], "no-store");
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/enhancement/profiles")
                .header("X-Imagine-Session", "stale-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 409);
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/enhancement/models")
                .header("X-Imagine-Session", "stale-session")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"profile_id":"proxy","base_url":"http://127.0.0.1:1/v1","api_key":null}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 409);
    assert_eq!(response.headers()["cache-control"], "no-store");
    let profile = imagine_server::enhancement_types::Profile {
        system_prompt: " \n".into(),
        ..Default::default()
    };
    let update = imagine_server::enhancement_types::ProfileUpdate { profile };
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .method("PUT")
                .uri("/v1/enhancement/profiles/proxy")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&update).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}

#[tokio::test]
async fn component_load_details_survive_scheduler_storage_and_the_public_api() {
    use imagine_server::component_error::{Component, ComponentLoadError, LoadDiagnostic, Reason};
    let h = setup().await;
    let failure = ComponentLoadError {
        diagnostic: LoadDiagnostic {
            component: Component::Vae,
            reason: Reason::TensorShapeMismatch,
            tensor: None,
        },
        file: "vae/chosen-decoder.pt".into(),
    };
    *h.probe.component_failure.lock().unwrap() = Some(failure.clone());
    let (status, accepted) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(json!({"profile":"qwen","prompt":"SYNTHETIC private input sentinel"})),
        Some("component-load-error"),
    )
    .await;
    assert_eq!(status, 202, "{accepted}");
    let id = accepted["id"].as_str().unwrap();
    h.probe.release.notify_one();
    wait_status(&h.store, id, "local", "failed").await;
    let (status, value) = req(&h.router, "GET", &format!("/v1/jobs/{id}"), None, None).await;
    assert_eq!(status, 200);
    assert_eq!(value["error"]["code"], "model_component_load_failed");
    assert_eq!(value["error"]["component"], "vae");
    assert_eq!(value["error"]["file"], failure.file);
    assert_eq!(value["error"]["reason"], "tensor_shape_mismatch");
    assert!(
        !value["error"]
            .to_string()
            .contains("private input sentinel")
    );
    stop(h).await;
}
async fn req(
    router: &Router,
    method: &str,
    url: &str,
    value: Option<Value>,
    key: Option<&str>,
) -> (u16, Value) {
    let mut value = value;
    if let Some(v) = &mut value
        && url.starts_with("/v1/jobs")
        && v["client_metadata"]["video_destination"].is_null()
    {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/video-destination")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let proof: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 8192).await.unwrap()).unwrap();
        if let Some(jobs) = v.get_mut("jobs").and_then(Value::as_array_mut) {
            for job in jobs {
                job["client_metadata"]["video_destination"] = proof.clone();
            }
        } else {
            v["client_metadata"]["video_destination"] = proof;
        }
    }
    let mut builder = Request::builder().method(method).uri(url);
    if let Some(key) = key {
        builder = builder.header("Idempotency-Key", key);
    }
    let body = if let Some(value) = value {
        builder = builder.header("Content-Type", "application/json");
        Body::from(value.to_string())
    } else {
        Body::empty()
    };
    let r = router
        .clone()
        .oneshot(builder.body(body).unwrap())
        .await
        .unwrap();
    let status = r.status().as_u16();
    assert_eq!(r.headers()["Cache-Control"], "no-store");
    let bytes = to_bytes(r.into_body(), 32 * 1024 * 1024).await.unwrap();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}
fn request() -> Value {
    json!({"model":MODEL,"operation":"text_to_image","prompt":"synthetic test"})
}
async fn wait_status(store: &Store, id: &str, owner: &str, status: &str) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let v = store.job(id, owner).await.unwrap().unwrap();
            if v["status"] == status {
                return v;
            }
            if v["status"] == "failed" && status != "failed" {
                panic!(
                    "Expected {status}; synthetic job failed: {}",
                    v["error"]["code"]
                );
            }

            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
async fn wait_calls(probe: &Probe, count: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while probe.calls.load(Ordering::SeqCst) < count {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
async fn stop(h: Harness) {
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
    if let Some(broker) = h.broker {
        broker.abort();
        let _ = broker.await;
    }
}

#[tokio::test]
async fn progress_is_scoped_to_active_job_in_polling_and_events() {
    let h = setup().await;
    let (_, job) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    let id = job["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let url = format!("/v1/jobs/{id}");
    let progress = imagine_server::progress::SamplingProgress {
        completed: 10,
        total: 40,
        elapsed_seconds: 8.,
        remaining_seconds: 24.,
        updated_at_ms: 1000,
    };
    h.engine_state.send_replace(EngineState {
        job_id: Some("another-job".into()),
        phase: "sampling".into(),
        progress: Some(progress.clone()),
        ..Default::default()
    });
    assert!(req(&h.router, "GET", &url, None, None).await.1["progress"].is_null());
    h.engine_state
        .send_modify(|state| state.job_id = Some(id.into()));
    let data = req(&h.router, "GET", &url, None, None).await.1;
    assert_eq!(data["progress"]["completed"], 10);
    assert_eq!(data["progress"]["remaining_seconds"], 24.);
    assert_eq!(
        req(&h.router, "GET", "/v1/jobs/job_missing", None, None)
            .await
            .0,
        404
    );
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("{url}/events"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let release = h.probe.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        release.release.notify_one();
    });
    let bytes = tokio::time::timeout(
        Duration::from_secs(5),
        to_bytes(response.into_body(), 65536),
    )
    .await
    .unwrap()
    .unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(text.contains("\"completed\":10"));
    assert!(text.contains("\"status\":\"succeeded\""));
    assert!(req(&h.router, "GET", &url, None, None).await.1["progress"].is_null());
    stop(h).await;
}

#[tokio::test]
async fn queue_idempotency_unload_resume_and_cancel() {
    let h = setup().await;
    assert_eq!(req(&h.router, "GET", "/v1/models", None, None).await.0, 200);
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    let (code, first) = req(&h.router, "POST", "/v1/jobs", Some(request()), Some("same")).await;
    assert_eq!(code, 202);
    let id = first["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let (_, repeat) = req(&h.router, "POST", "/v1/jobs", Some(request()), Some("same")).await;
    assert_eq!(first["id"], repeat["id"]);
    let mut different = request();
    different["prompt"] = json!("different");
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(different), Some("same"))
            .await
            .0,
        409
    );
    let (_, second) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(request()),
        Some("second"),
    )
    .await;
    let second = second["id"].as_str().unwrap();
    assert_ne!(id, second);
    assert_eq!(
        req(&h.router, "GET", &format!("/v1/jobs/{id}"), None, None)
            .await
            .0,
        200
    );
    let (code, op) = req(
        &h.router,
        "POST",
        "/v1/runtime/unload",
        Some(json!({"active_job":"wait"})),
        None,
    )
    .await;
    assert_eq!(code, 202);
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(request()), None)
            .await
            .0,
        409
    );
    assert_eq!(
        req(&h.router, "POST", "/v1/runtime/resume", None, None)
            .await
            .0,
        409
    );
    h.probe.release.notify_one();
    let first = wait_status(&h.store, id, "local", "succeeded").await;
    assert!(first["effective"]["seed"].is_u64());
    assert_eq!(first["output"]["assets"][0]["width"], 2);
    let asset = first["output"]["assets"][0]["id"].as_str().unwrap();
    assert!(h.store.read_asset(asset, "local").await.is_ok());
    let opid = op["id"].as_str().unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if h.store.operation(opid).await.unwrap().unwrap()["status"] == "succeeded" {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 1);
    assert_eq!(h.probe.stops.load(Ordering::SeqCst), 1);
    assert_eq!(
        h.store.job(second, "local").await.unwrap().unwrap()["status"],
        "queued"
    );
    assert_eq!(
        req(&h.router, "POST", "/v1/runtime/resume", None, None)
            .await
            .0,
        200
    );
    wait_calls(&h.probe, 2).await;
    assert_eq!(
        req(
            &h.router,
            "POST",
            &format!("/v1/jobs/{second}/cancel"),
            None,
            None
        )
        .await
        .0,
        202
    );
    let value = wait_status(&h.store, second, "local", "cancelled").await;
    assert!(value["output"].is_null());
    assert!(h.probe.stops.load(Ordering::SeqCst) >= 2);
    stop(h).await;
}

#[tokio::test]
async fn strict_inputs_assets_and_queued_cancel() {
    let h = setup().await;
    let mut wrong = request();
    wrong["loras"] = json!([]);
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(wrong), None)
            .await
            .0,
        422
    );
    let mut wrong = request();
    wrong["model"] = json!("minimax-h3");
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(wrong), None)
            .await
            .1["error"]["code"],
        "h3_runtime_unavailable"
    );
    let mut reference = Cursor::new(Vec::new());
    image::RgbImage::from_pixel(65, 67, image::Rgb([42, 50, 60]))
        .write_to(&mut reference, image::ImageFormat::Png)
        .unwrap();
    let upload = Request::builder()
        .method("POST")
        .uri("/v1/assets")
        .header("Content-Type", "image/png")
        .body(Body::from(reference.into_inner()))
        .unwrap();
    let r = h.router.clone().oneshot(upload).await.unwrap();
    assert_eq!(r.status(), 201);
    let asset: Value =
        serde_json::from_slice(&to_bytes(r.into_body(), 4096).await.unwrap()).unwrap();
    let id = asset["id"].as_str().unwrap();
    assert_eq!(
        req(&h.router, "GET", &format!("/v1/assets/{id}"), None, None)
            .await
            .0,
        200
    );
    let mut edit = request();
    edit["operation"] = json!("image_edit");
    edit["inputs"] = json!({"reference_images":[id, id]});
    let mut missing = edit.clone();
    missing["inputs"] = json!({"reference_images":["asset_missing"]});
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(missing), None)
            .await
            .0,
        404
    );
    let (code, job) = req(&h.router, "POST", "/v1/jobs", Some(edit), None).await;
    assert_eq!(code, 202);
    let job_id = job["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let job = h.store.job(job_id, "local").await.unwrap().unwrap();
    let references = job["effective"]["reference_preprocessing"]
        .as_array()
        .unwrap();
    assert_eq!(references.len(), 2);
    for reference in references {
        assert_eq!(reference["asset_id"], id);
        assert_eq!(reference["vae_width"], 64);
        assert_eq!(reference["vae_height"], 64);
        assert_eq!(reference["source"], "aio_shrink_center_crop_v1");
    }
    let (_, queued) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    let queued = queued["id"].as_str().unwrap();
    req(
        &h.router,
        "POST",
        &format!("/v1/jobs/{queued}/cancel"),
        None,
        None,
    )
    .await;
    wait_status(&h.store, queued, "local", "cancelled").await;
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 1);
    let (_, op) = req(
        &h.router,
        "POST",
        "/v1/runtime/unload",
        Some(json!({"active_job":"cancel"})),
        None,
    )
    .await;
    wait_status(&h.store, job_id, "local", "cancelled").await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if h.store
                .operation(op["id"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap()["status"]
                == "succeeded"
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    stop(h).await;
}

#[tokio::test]
async fn restart_starts_empty_and_unpaused() {
    let h = setup().await;
    let (_, first) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    wait_calls(&h.probe, 1).await;
    let (_, second) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    let (_, operation) = req(
        &h.router,
        "POST",
        "/v1/runtime/unload",
        Some(json!({"active_job":"wait"})),
        None,
    )
    .await;
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
    let reopened = Store::open(h.dir.path()).await.unwrap();
    assert!(!reopened.paused().await.unwrap());
    for job in [first, second] {
        assert!(
            reopened
                .job(job["id"].as_str().unwrap(), "local")
                .await
                .unwrap()
                .is_none()
        );
    }
    assert!(
        reopened
            .operation(operation["id"].as_str().unwrap())
            .await
            .unwrap()
            .is_none()
    );
    assert!(!h.dir.path().join("imagine.sqlite").exists());
}

#[tokio::test]
async fn engine_failure_isolated_next_job_can_run_and_sse_finishes() {
    let h = setup().await;
    h.probe.fail.store(1, Ordering::SeqCst);
    let (_, first) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    wait_calls(&h.probe, 1).await;
    let (_, second) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    h.probe.release.notify_one();
    wait_status(&h.store, first["id"].as_str().unwrap(), "local", "failed").await;
    wait_calls(&h.probe, 2).await;
    h.probe.release.notify_one();
    let id = second["id"].as_str().unwrap();
    wait_status(&h.store, id, "local", "succeeded").await;
    let r = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/jobs/{id}/events"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let bytes = to_bytes(r.into_body(), 65536).await.unwrap();
    let event = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(event.contains("event: job") && event.contains("succeeded"));
    assert_eq!(h.probe.stops.load(Ordering::SeqCst), 1);
    stop(h).await;
}

#[tokio::test]
async fn crash_recovery_does_not_retain_any_jobs() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).await.unwrap();
    for (id, status) in [
        ("active", "running"),
        ("cancelling", "cancelling"),
        ("queued", "queued"),
    ] {
        sqlx::query("INSERT INTO jobs (id,owner,status,request,request_hash,created) VALUES (?,'a',?,'{}','hash',0)")
            .bind(id).bind(status).execute(&store.db).await.unwrap();
    }
    store.close().await.unwrap();
    let reopened = Store::open(dir.path()).await.unwrap();
    for id in ["active", "cancelling", "queued"] {
        assert!(reopened.job(id, "local").await.unwrap().is_none());
    }
    assert!(!dir.path().join("imagine.sqlite").exists());
}

#[tokio::test]
async fn batch_is_atomic_and_idempotent_for_the_entire_request() {
    let h = setup().await;
    let mut invalid = request();
    invalid["seed"] = json!(9007199254740992u64);
    let (status, _) = req(
        &h.router,
        "POST",
        "/v1/jobs/batch",
        Some(json!({"jobs":[request(),invalid]})),
        Some("batch"),
    )
    .await;
    assert_eq!(status, 422);
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        0
    );
    let batch = json!({"jobs":[request(),request()]});
    let (status, first) = req(
        &h.router,
        "POST",
        "/v1/jobs/batch",
        Some(batch.clone()),
        Some("batch"),
    )
    .await;
    assert_eq!(status, 202);
    let (status, retry) = req(
        &h.router,
        "POST",
        "/v1/jobs/batch",
        Some(batch),
        Some("batch"),
    )
    .await;
    assert_eq!(status, 202);
    assert_eq!(first["jobs"][0]["id"], retry["jobs"][0]["id"]);
    assert_eq!(first["jobs"][1]["id"], retry["jobs"][1]["id"]);
    assert_eq!(retry["jobs"][0]["reused"], true);
    let (status, _) = req(
        &h.router,
        "POST",
        "/v1/jobs/batch",
        Some(json!({"jobs":[request()]})),
        Some("batch"),
    )
    .await;
    assert_eq!(status, 409);
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        2
    );
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
}

#[tokio::test]
async fn yaml_profiles_resolve_overrides_and_persist_provenance() {
    let h = setup().await;
    let (status, catalog) = req(&h.router, "GET", "/v1/profiles", None, None).await;
    assert_eq!(status, 200);
    assert!(
        catalog["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["id"] == "qwen")
    );
    let profiles = catalog["profiles"].as_array().unwrap();
    assert_eq!(profiles.len(), 5); // standard, Turbo, MOSS, sheets and upscale
    for (id, name, model) in [
        ("qwen", "Qwen Image 2.1", "qwen-image-2.1-int8"),
        (
            "quick-draft",
            "Qwen Image 2.1 Turbo",
            "qwen-image-2.1-turbo",
        ),
    ] {
        let profile = profiles.iter().find(|profile| profile["id"] == id).unwrap();
        assert_eq!(profile["name"], name);
        assert_eq!(profile["model"], model);
    }
    let (_, description) = req(&h.router, "GET", "/v1/profiles/qwen", None, None).await;
    for (profile, original_guidance) in [
        (
            "qwen",
            include_str!("../proxy/workflows/prompts/prompt-enhance-scene.md"),
        ),
        (
            "quick-draft",
            include_str!("../proxy/workflows/prompts/prompt-enhance-scene.md"),
        ),
    ] {
        let (status, described) = req(
            &h.router,
            "GET",
            &format!("/v1/profiles/{profile}"),
            None,
            None,
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(described["prompt_guidance"], original_guidance, "{profile}");
    }
    assert_eq!(description["defaults"]["sampling"]["steps"], 40);
    assert!(description["defaults"].get("negative_prompt").is_none());
    assert!(
        description["parameters_schema"]["properties"]
            .get("resolved_profile")
            .is_none()
    );
    let body = json!({"profile":"qwen","profile_version":4,"prompt":"synthetic profile test", "width":1024,"sampling":{"steps":12},"model_options":{"qwen_image_2_1":{"generation":{"loras":[{"path":"synthetic.safetensors","strength":0.5}]}}}});
    let (status, validation) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        Some(body.clone()),
        None,
    )
    .await;
    assert_eq!(status, 200, "{validation}");
    assert_eq!(validation["effective"]["width"], 1024);
    assert_eq!(validation["effective"]["height"], 896);
    assert_eq!(validation["effective"]["sampling"]["steps"], 12);
    assert_eq!(validation["effective"]["sampling"]["cfg_scale"], 2.5);
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    assert!(validation["effective"].get("prompt").is_none());
    let (status, submitted) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(body.clone()),
        Some("profile-retry"),
    )
    .await;
    assert_eq!(status, 202, "{submitted}");
    let id = submitted["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let effective = h.store.job(id, "local").await.unwrap().unwrap()["effective"].clone();
    assert_eq!(effective["resolved_profile"]["id"], "qwen");
    assert_eq!(effective["resolved_profile"]["version"], 4);
    assert_eq!(effective["negative_prompt"], "synthetic profile negative");
    assert_eq!(
        effective["resolved_profile"]["sha256"],
        description["sha256"]
    );
    assert!(effective["seed"].is_u64());
    assert_eq!(
        effective["model_options"]["qwen_image_2_1"]["generation"]["loras"][0]["path"],
        "synthetic.safetensors"
    );
    let (_, retry) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(body),
        Some("profile-retry"),
    )
    .await;
    assert_eq!(retry["id"], id);
    assert_eq!(
        h.store.job(id, "local").await.unwrap().unwrap()["effective"]["seed"],
        effective["seed"]
    );
    let (_, cleared) = req(&h.router, "POST", "/v1/jobs/validate", Some(json!({"profile":"quick-draft","prompt":"synthetic","sampling":{"custom_sigmas":[]},"model_options":{"qwen_image_2_1":{"generation":{"loras":[]}}}})), None).await;
    assert!(
        cleared["effective"]["sampling"]
            .get("custom_sigmas")
            .is_none()
    );
    assert_eq!(
        cleared["effective"]["model_options"]["qwen_image_2_1"]["generation"]["loras"],
        json!([])
    );
    h.probe.release.notify_one();
    wait_status(&h.store, id, "local", "succeeded").await;
    stop(h).await;
}

#[tokio::test]
async fn profiles_reject_invalid_overrides_before_batch_admission() {
    let h = setup().await;
    for (patch, code) in [
        (
            json!({"profile":"absent","prompt":"test"}),
            "unknown_profile",
        ),
        (
            json!({"profile":"qwen","profile_version":1,"prompt":"test"}),
            "profile_version_mismatch",
        ),
        (
            json!({"profile":"qwen","prompt":"test","sampling":{"typo":1}}),
            "invalid_request",
        ),
        (
            json!({"profile":"qwen","prompt":"test","sampling":null}),
            "invalid_request",
        ),
        (
            json!({"profile":"qwen","prompt":"test","sampling":{"steps":0}}),
            "unsupported_sampling",
        ),
        (
            json!({"profile":"qwen","prompt":"test","width":1601}),
            "unsupported_resolution",
        ),
        (
            json!({"profile":"qwen","prompt":"test","model":"qwen-image-2.1-turbo"}),
            "model_settings_mismatch",
        ),
        (
            json!({"profile":"qwen","prompt":"test","width":1024,"max_side":1024}),
            "conflicting_dimensions",
        ),
        (
            json!({"profile":"qwen","prompt":"test","nsfw":"true"}),
            "invalid_nsfw",
        ),
        (
            json!({"profile":"qwen","prompt":"test","resolved_profile":{}}),
            "read_only_profile_metadata",
        ),
        (
            json!({"profile":"image_upscale","prompt":"test"}),
            "invalid_reference_count",
        ),
    ] {
        let (status, result) = req(
            &h.router,
            "POST",
            "/v1/jobs/batch",
            Some(json!({"jobs":[{"profile":"qwen","prompt":"valid"},patch]})),
            None,
        )
        .await;
        assert!(status == 422 || status == 409, "{status}: {result}");
        assert_eq!(result["error"]["code"], code);
    }
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    let count: i64 = sqlx::query_scalar("SELECT count(*) FROM jobs")
        .fetch_one(&h.store.db)
        .await
        .unwrap();
    assert_eq!(count, 0);
    // The frozen comparison wrapper is preserved outside the active catalog.
    let archived = imagine_server::profiles::Catalog::load(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/docs/archive/image-profiles"
    )))
    .unwrap();
    let frozen = archived
        .resolve(
            json!({"profile":"qwen-image21-1600x896-v2","prompt":"synthetic frozen control"}),
            &h.store,
            "local",
        )
        .await
        .unwrap();
    let frozen = json!({"effective": frozen.effective()});
    assert_eq!(
        frozen["effective"]["profile_revision"],
        imagine_server::contract::PROFILE
    );
    assert!(
        frozen["effective"]["model_options"]["qwen_image_2_1"]
            .get("generation")
            .is_none()
    );
    assert_eq!(frozen["effective"]["width"], 1600);
    assert_eq!(frozen["effective"]["sampling"]["steps"], 40);
    stop(h).await;
}

#[tokio::test]
async fn profile_reference_sizing_and_missing_assets() {
    let h = setup().await;
    let mut bytes = Cursor::new(Vec::new());
    image::RgbImage::new(256, 128)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let asset = h
        .store
        .save_asset("local", bytes.into_inner())
        .await
        .unwrap();
    let body = json!({"profile":"image_upscale","prompt":"synthetic upscale","negative_prompt":"","max_side":512,"inputs":{"reference_images":[asset["id"]]}});
    let (status, v) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        Some(body.clone()),
        None,
    )
    .await;
    assert_eq!(status, 200, "{v}");
    assert_eq!(v["effective"]["width"], 512);
    assert_eq!(v["effective"]["height"], 256);
    assert_eq!(v["effective"]["operation"], "image_edit");
    assert_eq!(
        v["effective"]["model_options"]["qwen_image_2_1"]["generation"]["loras"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mut missing = body;
    missing["inputs"] = json!({"reference_images":["asset_missing"]});
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs/validate", Some(missing), None)
            .await
            .0,
        404
    );
    let (_, v) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        Some(json!({"profile":"qwen","prompt":"test","max_side":1024,"aspect_ratio":"3:4"})),
        None,
    )
    .await;
    assert_eq!(v["effective"]["width"], 768);
    assert_eq!(v["effective"]["height"], 1024);
    stop(h).await;
}

#[tokio::test]
async fn uploaded_assets_are_ram_only_owned_and_removed_on_close() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = png();
    let store = Store::open(dir.path()).await.unwrap();
    let asset = store.save_asset("local", bytes.clone()).await.unwrap();
    let id = asset["id"].as_str().unwrap();
    let path = store.media.join(id);
    assert!(!path.exists());
    assert_eq!(std::fs::read_dir(&store.media).unwrap().count(), 0);
    assert_eq!(store.read_asset(id, "local").await.unwrap(), bytes);
    assert!(store.read_asset(id, "other").await.is_err());
    store.close().await.unwrap();
    assert!(!path.exists());
}

#[tokio::test]
async fn old_session_requests_are_rejected_before_mutation() {
    let h = setup().await;
    let (status, session) = req(&h.router, "GET", "/v1/session", None, None).await;
    assert_eq!(status, 200);
    assert_eq!(session["id"], h.store.session.id);
    assert_eq!(session["encrypted"], false);
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/jobs")
                .header("Content-Type", "application/json")
                .header("X-Imagine-Session", "expired-session")
                .body(Body::from(request().to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 409);
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}

fn sheet_request() -> Value {
    json!({"profile":"qwen-character-sheet","seed":42,"character_sheet":{
        "character_description":"SYNTHETIC_SHEET_IDENTITY","clothing_description":"SYNTHETIC_SHEET_CLOTHES",
        "layout":{"output_width":512,"portrait_fraction":0.45}}})
}
#[tokio::test]
async fn character_sheet_validation_schema_and_sensitive_fields() {
    let h = setup().await;
    let (_, profile) = req(
        &h.router,
        "GET",
        "/v1/profiles/qwen-character-sheet",
        None,
        None,
    )
    .await;
    assert!(
        profile["parameters_schema"]["properties"]
            .get("prompt")
            .is_none()
    );
    assert!(
        profile["defaults"]["character_sheet"]
            .get("character_description")
            .is_none()
    );
    assert_eq!(profile["character_sheet"]["max_identity_references"], 8);
    let (status, valid) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        Some(sheet_request()),
        None,
    )
    .await;
    assert_eq!(status, 200, "{valid}");
    assert!(!valid.to_string().contains("SYNTHETIC_SHEET"));
    assert_eq!(
        valid["effective"]["execution_plan"]["passes"][1]["seed"],
        43
    );
    assert!(valid["effective"].get("width").is_none());
    let mut cases = Vec::new();
    for (field, value) in [
        ("prompt", json!("ambiguous")),
        ("width", json!(512)),
        ("max_side", json!(512)),
        ("execution_plan", json!({})),
    ] {
        let mut body = sheet_request();
        body[field] = value;
        cases.push(body);
    }
    let mut body = sheet_request();
    body["character_sheet"]["layout"]["output_width"] = json!(513);
    cases.push(body);
    let mut body = sheet_request();
    body["character_sheet"]["layout"]["portrait_fraction"] = json!(0.001);
    cases.push(body);
    let mut body = sheet_request();
    body["character_sheet"]["character_description"] = json!("");
    cases.push(body);
    let mut body = sheet_request();
    body["character_sheet"]["character_description"] =
        json!(r#"{"encrypted":true,"schema":"unknown","ciphertext":"synthetic"}"#);
    cases.push(body);
    let mut body = sheet_request();
    body["inputs"] = json!({"reference_images":vec!["missing";9]});
    cases.push(body);
    let mut body = sheet_request();
    body["model"] = json!(imagine_server::contract::TURBO);
    cases.push(body);
    let mut body = request();
    body["character_sheet"] = json!({});
    cases.push(body);
    let mut body = sheet_request();
    body["model_options"] = json!({"qwen_image_2_1":{"reference_max_pixels_per_image":1}});
    cases.push(body);
    for body in cases {
        let (status, error) = req(&h.router, "POST", "/v1/jobs", Some(body), None).await;
        assert_eq!(status, 422, "{error}");
        assert!(!error.to_string().contains("SYNTHETIC_SHEET"));
    }
    let guide = imagine_server::character_sheet::pose_guide(64, 64).unwrap();
    let asset = h.store.save_asset("local", guide).await.unwrap();
    let mut body = sheet_request();
    body["inputs"] = json!({"pose_guide":"asset_missing"});
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/jobs/validate",
            Some(body.clone()),
            None
        )
        .await
        .0,
        404
    );
    body["inputs"] = json!({"pose_guide":asset["id"]});
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs/validate", Some(body), None)
            .await
            .0,
        200
    );
    let tiny = h.store.save_asset("local", png()).await.unwrap();
    let mut body = sheet_request();
    body["inputs"] = json!({"pose_guide":tiny["id"]});
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs/validate", Some(body), None)
            .await
            .0,
        422
    );
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}

#[tokio::test]
async fn character_sheet_is_one_job_with_ordered_in_memory_passes_and_three_results() {
    let h = setup().await;
    h.probe.sized_output.store(1, Ordering::SeqCst);
    let guide = imagine_server::character_sheet::pose_guide(64, 64).unwrap();
    let reference = h.store.save_asset("local", guide.clone()).await.unwrap();
    let mut body = sheet_request();
    body["seed"] = json!(imagine_server::contract::MAX_SEED);
    body["inputs"] = json!({"reference_images":[reference["id"]],"pose_guide":reference["id"]});
    let (status, job) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(body.clone()),
        Some("sheet-once"),
    )
    .await;
    assert_eq!(status, 202, "{job}");
    let id = job["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let (_, same) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(body),
        Some("sheet-once"),
    )
    .await;
    assert_eq!(same["id"], id);
    h.probe.release.notify_one();
    wait_calls(&h.probe, 2).await;
    // Only the uploaded guide exists: portrait has not been published mid-job.
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM assets")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        1
    );
    {
        let passes = h.probe.passes.lock().unwrap();
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].seed, imagine_server::contract::MAX_SEED);
        assert_eq!(passes[1].seed, 0);
        assert_eq!((passes[0].width, passes[0].height), (224, 288));
        assert_eq!((passes[1].width, passes[1].height), (288, 288));
        assert_eq!(passes[0].references.len(), 1);
        assert_eq!(passes[1].references.len(), 3);
        assert_eq!(&*passes[0].references[0], guide.as_slice());
        assert_eq!(&*passes[1].references[0], guide.as_slice());
        assert_eq!(&*passes[1].references[2], guide.as_slice());
        assert_eq!(
            image::load_from_memory(&passes[1].references[1])
                .unwrap()
                .width(),
            224
        );
        assert!(!passes[0].prompt.contains("SYNTHETIC_SHEET_CLOTHES"));
        assert!(passes[1].prompt.contains("SYNTHETIC_SHEET_CLOTHES"));
    }
    h.probe.release.notify_one();
    let result = wait_status(&h.store, id, "local", "succeeded").await;
    let assets = result["output"]["assets"].as_array().unwrap();
    assert_eq!(assets.len(), 3);
    for (asset, role) in assets
        .iter()
        .zip(["character_sheet", "portrait", "turnaround"])
    {
        assert_eq!(asset["role"], role);
        assert!(
            h.store
                .read_asset(asset["id"].as_str().unwrap(), "other")
                .await
                .is_err()
        );
    }
    assert_eq!(assets[0]["width"], 512);
    assert_eq!(assets[0]["height"], 288);
    assert_eq!(
        result["effective"]["execution_plan"]["passes"][1]["seed"],
        0
    );
    assert_eq!(
        *h.probe.stages.lock().unwrap(),
        ["portrait", "turnaround", "compose"]
    );
    stop(h).await;
}

#[tokio::test]
async fn character_sheet_cancel_or_second_pass_failure_never_publishes_partial_results() {
    for cancel in [true, false] {
        let h = setup().await;
        h.probe.sized_output.store(1, Ordering::SeqCst);
        let (_, job) = req(&h.router, "POST", "/v1/jobs", Some(sheet_request()), None).await;
        let id = job["id"].as_str().unwrap();
        wait_calls(&h.probe, 1).await;
        h.probe.release.notify_one();
        wait_calls(&h.probe, 2).await;
        if cancel {
            h.control
                .call(Action::Cancel {
                    id: id.into(),
                    owner: "local".into(),
                })
                .await
                .unwrap();
        } else {
            h.probe.fail.store(1, Ordering::SeqCst);
            h.probe.release.notify_one();
        }
        wait_status(
            &h.store,
            id,
            "local",
            if cancel { "cancelled" } else { "failed" },
        )
        .await;
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT count(*) FROM assets")
                .fetch_one(&h.store.db)
                .await
                .unwrap(),
            0
        );
        assert_eq!(std::fs::read_dir(&h.store.media).unwrap().count(), 0);
        assert_eq!(h.probe.stops.load(Ordering::SeqCst), 1);
        stop(h).await;
    }
}

#[tokio::test]
async fn output_publication_rolls_back_files_and_rows_if_a_later_role_is_invalid() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = Store::open(dir.path()).await.unwrap();
    store.videos.destination = Some(video_output::destination(dir.path()));
    let destination = store.videos.destination().unwrap();
    let proof = serde_json::to_value(destination.proof().unwrap()).unwrap();
    sqlx::query("INSERT INTO jobs(id,owner,status,request,request_hash,created) VALUES ('job_test','local','running','{}','test',0)").execute(&store.db).await.unwrap();
    let artifacts = vec![
        imagine_server::generation::Artifact {
            role: "character_sheet",
            bytes: png(),
        },
        imagine_server::generation::Artifact {
            role: "portrait",
            bytes: vec![0, 1, 2],
        },
    ];
    assert!(
        store
            .publish_output(
                "job_test",
                "local",
                imagine_server::generation::Generated::Images {
                    artifacts,
                    pending: vec![
                        destination.reserve_image(&proof, 1024, None).unwrap(),
                        destination.reserve_image(&proof, 1024, None).unwrap()
                    ]
                }
            )
            .await
            .is_err()
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM assets")
            .fetch_one(&store.db)
            .await
            .unwrap(),
        0
    );
    assert_eq!(std::fs::read_dir(&store.media).unwrap().count(), 0);
    assert_eq!(
        store.job("job_test", "local").await.unwrap().unwrap()["status"],
        "running"
    );
    // A valid output cannot be published to another owner's job either.
    assert!(
        store
            .publish_output(
                "job_test",
                "other",
                imagine_server::generation::Generated::Images {
                    artifacts: vec![imagine_server::generation::Artifact {
                        role: "image",
                        bytes: png()
                    }],
                    pending: vec![destination.reserve_image(&proof, 1024, None).unwrap()]
                }
            )
            .await
            .is_err()
    );
    assert_eq!(std::fs::read_dir(&store.media).unwrap().count(), 0);
    store.close().await.unwrap();
}

#[path = "../shared/model_catalog_fixture.rs"]
mod model_fixture;
#[tokio::test]
async fn model_library_api_freezes_queued_selection_and_validates_before_admission() {
    let h = setup().await;
    let first_root = h.dir.path().join("models-a");
    let second_root = h.dir.path().join("models-b");
    model_fixture::models(&first_root);
    model_fixture::models(&second_root);
    *h.library.lock().unwrap() = imagine_server::model_library::Library::load(
        first_root.clone(),
        h.dir.path().join("model-library.json"),
    )
    .unwrap();
    let (status, library) = req(&h.router, "GET", "/v1/model-library", None, None).await;
    assert_eq!(status, 200);
    let files = library["files"].as_array().unwrap();
    let diffusion = files.iter().find(|f| f["model"] == MODEL).unwrap()["id"].clone();
    let lora = files
        .iter()
        .find(|f| f["path"] == "loras/nested/style.safetensors")
        .unwrap()["id"]
        .clone();
    let mut value = request();
    value["model_selection"] = json!({"diffusion":diffusion,"loras":[{"id":lora,"strength":0.75}]});
    let (status, validated) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        Some(value.clone()),
        None,
    )
    .await;
    assert_eq!(status, 200, "{validated}");
    assert!(!validated.to_string().contains(first_root.to_str().unwrap()));
    let (status, first) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(value.clone()),
        Some("library-first"),
    )
    .await;
    assert_eq!(status, 202, "{first}");
    wait_calls(&h.probe, 1).await;
    let (status, second) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(value.clone()),
        Some("library-second"),
    )
    .await;
    assert_eq!(status, 202);
    let mut settings = library["settings"].clone();
    settings["root"] = json!(second_root);
    let (status, _) = req(&h.router, "PUT", "/v1/model-library", Some(settings), None).await;
    assert_eq!(status, 200);
    h.probe.release.notify_one();
    wait_calls(&h.probe, 2).await;
    {
        let passes = h.probe.passes.lock().unwrap();
        let second = &passes[1];
        assert_eq!(second.resolved_models.as_ref().unwrap().root, first_root);
        assert_eq!(
            second
                .model_options
                .qwen_image_2_1
                .workflow
                .as_ref()
                .unwrap()
                .loras[0]
                .strength,
            0.75
        );
    }
    h.probe.release.notify_one();
    let completed = wait_status(
        &h.store,
        second["id"].as_str().unwrap(),
        "local",
        "succeeded",
    )
    .await;
    assert!(completed["effective"].get("resolved_models").is_none());
    value["model_selection"]["loras"] = json!([]);
    let (status, next) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(value.clone()),
        Some("library-next"),
    )
    .await;
    assert_eq!(status, 202);
    wait_calls(&h.probe, 3).await;
    {
        let passes = h.probe.passes.lock().unwrap();
        let pass = &passes[2];
        assert_eq!(pass.resolved_models.as_ref().unwrap().root, second_root);
        assert!(
            pass.model_options
                .qwen_image_2_1
                .workflow
                .as_ref()
                .unwrap()
                .loras
                .is_empty()
        );
    }
    h.probe.release.notify_one();
    wait_status(&h.store, next["id"].as_str().unwrap(), "local", "succeeded").await;
    value["resolved_models"] = json!({"root":"/"});
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(value.clone()), None)
            .await
            .0,
        422
    );
    value.as_object_mut().unwrap().remove("resolved_models");
    value["model_selection"]["loras"] = json!([{"id":"missing","strength":1}]);
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/jobs/batch",
            Some(json!({"jobs":[value]})),
            None
        )
        .await
        .0,
        422
    );
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 3);
    stop(h).await;
}

#[tokio::test]
async fn rust_admission_exposes_its_own_capabilities_and_rejects_unsupported_sampler_before_queue()
{
    let h = setup().await;
    h.engine_state.send_modify(|s| s.backend = "rust".into());
    let (_, meta) = req(&h.router, "GET", &format!("/v1/models/{MODEL}"), None, None).await;
    assert_eq!(meta["engine"], "Imagine Rust");
    assert_eq!(meta["constraints"]["samplers"], json!(["er_sde", "euler"]));
    let mut value: Value =
        serde_yaml::from_str::<Value>(include_str!("../config/image/profiles/qwen.yaml")).unwrap()
            ["defaults"]
            .clone();
    value["prompt"] = json!("synthetic");
    value["sampling"]["sampler"] = json!("res_multistep");
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", Some(value), None)
            .await
            .0,
        422
    );
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    let (_, moss) = req(
        &h.router,
        "GET",
        &format!("/v1/models/{}", imagine_server::contract::MOSS),
        None,
        None,
    )
    .await;
    assert_eq!(moss["operations"], json!(["text_to_image"]));
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
}

fn h3_request() -> Value {
    json!({"model":"minimax-h3","operation":"text_to_video","prompt":"  SYNTHETIC h3 åäö\n  ","width":384,"height":256,"seed":42,
    "sampling":{"steps":20,"cfg_scale":1.,"sampler":"res_multistep","scheduler":"simple"},
    "output":{"format":"mp4","count":1},"video":{"profile":"h3-short-av-noref-v1","frames":39,"fps":24,"audio":true}})
}
#[tokio::test]
async fn h3_fixed_requests_require_deployment_and_reject_profile_expansion() {
    let h = setup_inner(true).await;
    let (status, body) = req(&h.router, "POST", "/v1/jobs", Some(h3_request()), None).await;
    assert_eq!(status, 422);
    assert_eq!(body["error"]["code"], "h3_runtime_unavailable");
    h.engine_state.send_modify(|s| {
        s.backend = "rust".into();
        s.h3_available = true;
    });
    let (_, metadata) = req(&h.router, "GET", "/v1/models/minimax-h3", None, None).await;
    assert_eq!(metadata["available"], true);
    assert_eq!(metadata["experimental"], true);
    assert_eq!(metadata["sampling"]["steps"], 20);
    assert_eq!(metadata["sampling"]["sampler"], "res_multistep");
    assert_eq!(metadata["sampling"]["gpu_acceptance"], "pending");
    for (field, value) in [
        ("seed", json!(43)),
        ("width", json!(512)),
        ("negative_prompt", json!("forbidden")),
        ("inputs", json!({"reference_images":["asset_synthetic"]})),
    ] {
        let mut input = h3_request();
        input[field] = value;
        assert_eq!(
            req(&h.router, "POST", "/v1/jobs", Some(input), None)
                .await
                .0,
            422
        );
    }
    assert_eq!(h.probe.video_calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}
#[tokio::test]
async fn h3_and_images_share_fifo_atomic_publication_ranges_and_cancellation() {
    let h = setup_inner(true).await;
    h.engine_state.send_modify(|s| {
        s.backend = "rust".into();
        s.h3_available = true;
    });
    let (status, first) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(h3_request()),
        Some("own-h3-queue"),
    )
    .await;
    assert_eq!(status, 202);
    let first = first["id"].as_str().unwrap();
    wait_status(&h.store, first, "local", "running").await;
    let (status, second) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    assert_eq!(status, 202);
    let second = second["id"].as_str().unwrap();
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    let before = h.store.job(first, "local").await.unwrap().unwrap();
    assert!(before["output"].is_null());
    h.probe.release.notify_one();
    let video = wait_status(&h.store, first, "local", "succeeded").await;
    assert_eq!(
        video["effective"]["profile_revision"],
        "h3-short-av-noref-v1"
    );
    assert_eq!(video["effective"]["prompt"], h3_request()["prompt"]);
    assert_eq!(video["output"]["format"], "mp4");
    assert_eq!(video["output"]["persistent"], true);
    let saved_path = std::path::PathBuf::from(video["output"]["saved_path"].as_str().unwrap());
    assert!(saved_path.exists());
    assert_eq!(std::fs::read_dir(&h.store.media).unwrap().count(), 0);
    assert_eq!(video["output"]["assets"][0]["role"], "video");
    assert_eq!(
        h.probe.video_prompts.lock().unwrap()[0],
        h3_request()["prompt"].as_str().unwrap()
    );
    let asset = video["output"]["assets"][0]["id"].as_str().unwrap();
    let data = h.store.read_asset(asset, "local").await.unwrap();
    let decoded = imagine_server::short_av::decode(&data).unwrap();
    assert!(decoded.frames[38][0].abs_diff(38) <= 3);
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/assets/{asset}"))
                .header("Range", "bytes=2-9")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 206);
    assert_eq!(response.headers()["Cache-Control"], "no-store");
    assert_eq!(
        response.headers()["Content-Type"],
        imagine_server::short_av::MIME
    );
    assert_eq!(
        &to_bytes(response.into_body(), 16).await.unwrap()[..],
        &data[2..10]
    );
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/assets/{asset}"))
                .header("X-Imagine-Session", "wrong-session")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 409);
    wait_calls(&h.probe, 1).await;
    h.probe.release.notify_one();
    wait_status(&h.store, second, "local", "succeeded").await;
    let (_, cancel) = req(&h.router, "POST", "/v1/jobs", Some(h3_request()), None).await;
    let cancel = cancel["id"].as_str().unwrap();
    wait_status(&h.store, cancel, "local", "running").await;
    let (status, _) = req(
        &h.router,
        "POST",
        &format!("/v1/jobs/{cancel}/cancel"),
        None,
        None,
    )
    .await;
    assert_eq!(status, 202);
    let cancelled = wait_status(&h.store, cancel, "local", "cancelled").await;
    assert!(cancelled["output"].is_null());
    assert!(h.probe.stops.load(Ordering::SeqCst) > 0);
    stop(h).await;
}

#[tokio::test]
async fn h3_never_admits_without_coordinator_even_with_a_deployed_worker() {
    let h = setup().await;
    h.engine_state.send_modify(|s| {
        s.backend = "rust".into();
        s.h3_available = true;
    });
    let (_, metadata) = req(&h.router, "GET", "/v1/models/minimax-h3", None, None).await;
    assert_eq!(metadata["available"], false);
    assert_eq!(metadata["reason"], "coordinator_unavailable");
    let (status, error) = req(&h.router, "POST", "/v1/jobs", Some(h3_request()), None).await;
    assert_eq!(status, 503);
    assert_eq!(error["error"]["code"], "coordinator_unavailable");
    let request = serde_json::from_value(h3_request()).unwrap();
    let error = h
        .control
        .call(Action::Submit {
            owner: "local".into(),
            request: Box::new(request),
            key: None,
        })
        .await
        .unwrap_err();
    assert_eq!(error.code, "coordinator_unavailable");
    assert_eq!(h.probe.video_calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}

#[tokio::test]
async fn h3_selected_geometry_reaches_queue_and_output_metadata() {
    let h = setup_inner(true).await;
    h.engine_state.send_modify(|s| {
        s.backend = "rust".into();
        s.h3_available = true;
    });
    let mut value = h3_request();
    value["width"] = 64.into();
    value["height"] = 96.into();
    value["video"]["frames"] = 22.into();
    value["video"]["profile"] = "h3-av-noref-v2".into();
    let (status, job) = req(&h.router, "POST", "/v1/jobs", Some(value.clone()), None).await;
    assert_eq!(status, 202, "{job}");
    let id = job["id"].as_str().unwrap();
    wait_status(&h.store, id, "local", "running").await;
    h.probe.release.notify_one();
    let done = wait_status(&h.store, id, "local", "succeeded").await;
    assert_eq!(done["effective"]["width"], 64);
    assert_eq!(done["output"]["assets"][0]["width"], 64);
    assert_eq!(done["output"]["assets"][0]["height"], 96);
    assert_eq!(done["output"]["video"]["frames"], 22);
    assert_eq!(done["output"]["video"]["samples"], 29333);
    for (key, bad) in [("frames", 23), ("fps", 30)] {
        let mut invalid = value.clone();
        invalid["video"][key] = bad.into();
        assert_eq!(
            req(&h.router, "POST", "/v1/jobs", Some(invalid), None)
                .await
                .0,
            422
        );
    }
    assert_eq!(h.probe.video_calls.load(Ordering::SeqCst), 1);
    stop(h).await;
}

#[tokio::test]
async fn h3_i2v_owns_start_image_preserves_queue_source_and_switches_profiles() {
    let h = setup_inner(true).await;
    h.engine_state.send_modify(|s| {
        s.backend = "rust".into();
        s.h3_available = true;
    });
    // Synthetic image generation uses Fake only; no service/model/GPU.
    let (_, image) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    let image_id = image["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    h.probe.release.notify_one();
    let image = wait_status(&h.store, image_id, "local", "succeeded").await;
    let asset = image["output"]["assets"][0]["id"].as_str().unwrap();
    let mut input = h3_request();
    input["operation"] = "image_to_video".into();
    input["video"]["profile"] = "h3-fl2va-i2va-v1".into();
    input["inputs"] = json!({"start_image":asset});
    input["width"] = 64.into();
    input["height"] = 64.into();
    input["video"]["frames"] = 5.into();
    let foreign = h.store.save_asset("other", png()).await.unwrap();
    for id in ["asset_missing", foreign["id"].as_str().unwrap()] {
        let mut bad = input.clone();
        bad["inputs"]["start_image"] = id.into();
        assert_eq!(
            req(&h.router, "POST", "/v1/jobs", Some(bad), None).await.0,
            404
        );
    }
    let stale = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/jobs")
                .header("Content-Type", "application/json")
                .header("X-Imagine-Session", "stale")
                .body(Body::from(serde_json::to_vec(&input).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(stale.status(), 409);
    let (status, job) = req(&h.router, "POST", "/v1/jobs", Some(input), None).await;
    assert_eq!(status, 202, "{job}");
    let id = job["id"].as_str().unwrap();
    wait_status(&h.store, id, "local", "running").await;
    for _ in 0..100 {
        if h.probe.video_calls.load(Ordering::SeqCst) == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    let seen = h.probe.video_inputs.lock().unwrap()[0].clone();
    assert_eq!(
        seen.profile,
        imagine_server::h3_protocol::Profile::ImageToAvV1
    );
    assert!(seen.valid());
    assert_eq!(seen.image.as_ref().unwrap().width, 2);
    assert_eq!(seen.image_rgb, [42, 50, 60].repeat(4));
    // Subsequent work cannot change the admitted source or overlap worker ownership.
    let mut text = h3_request();
    text["width"] = 64.into();
    text["height"] = 64.into();
    text["video"]["frames"] = 5.into();
    text["video"]["profile"] = "h3-av-noref-v2".into();
    let (_, queued) = req(&h.router, "POST", "/v1/jobs", Some(text), None).await;
    assert_eq!(h.probe.video_calls.load(Ordering::SeqCst), 1);
    h.probe.release.notify_one();
    let done = wait_status(&h.store, id, "local", "succeeded").await;
    assert_eq!(done["effective"]["profile_revision"], "h3-fl2va-i2va-v1");
    let permanent = std::path::PathBuf::from(done["output"]["saved_path"].as_str().unwrap());
    assert_eq!(done["effective"]["inputs"]["start_image"], asset);
    assert_eq!(h.store.read_asset(asset, "local").await.unwrap(), png());
    let text_id = queued["id"].as_str().unwrap();
    wait_status(&h.store, text_id, "local", "running").await;
    h.probe.release.notify_one();
    wait_status(&h.store, text_id, "local", "succeeded").await;
    assert!(h.probe.video_inputs.lock().unwrap()[1].image.is_none());
    let (_, last) = req(&h.router, "POST", "/v1/jobs", Some(request()), None).await;
    wait_calls(&h.probe, 2).await;
    h.probe.release.notify_one();
    wait_status(&h.store, last["id"].as_str().unwrap(), "local", "succeeded").await;
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
    h.store.close().await.unwrap();
    assert!(permanent.exists());
    if let Some(broker) = h.broker {
        broker.abort();
        let _ = broker.await;
    }
}

#[tokio::test]
async fn h3_destination_proof_rejected_before_engine_admission() {
    let h = setup_inner(true).await;
    h.engine_state.send_modify(|s| {
        s.backend = "rust".into();
        s.h3_available = true;
    });
    for proof in [Value::Null, json!({"host":"wrong-host"})] {
        let mut input = h3_request();
        input["client_metadata"] = json!({"video_destination":proof});
        let error = h
            .control
            .call(Action::Submit {
                owner: "local".into(),
                request: Box::new(serde_json::from_value(input).unwrap()),
                key: None,
            })
            .await
            .unwrap_err();
        assert_eq!(error.code, "video_destination_mismatch");
    }
    assert_eq!(h.probe.video_calls.load(Ordering::SeqCst), 0);
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    stop(h).await;
}

#[tokio::test]
async fn images_persist_without_session_copies_keep_owned_identity_and_upscale_naming() {
    let h = setup().await;
    let (status, job) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        Some(json!({"profile":"qwen","prompt":"synthetic persistent fixture"})),
        Some("persistent-image"),
    )
    .await;
    assert_eq!(status, 202, "{job}");
    let id = job["id"].as_str().unwrap();
    h.probe.release.notify_one();
    wait_status(&h.store, id, "local", "succeeded").await;
    let job = h.store.job(id, "local").await.unwrap().unwrap();
    assert_eq!(job["output"]["persistent"], true);
    let asset = &job["output"]["assets"][0];
    let asset_id = asset["id"].as_str().unwrap();
    let path = std::path::PathBuf::from(asset["saved_path"].as_str().unwrap());
    assert_eq!(std::fs::read(&path).unwrap(), png());
    assert_eq!(std::fs::read_dir(&h.store.media).unwrap().count(), 0);
    assert_eq!(h.store.read_asset(asset_id, "local").await.unwrap(), png());
    assert!(h.store.read_asset(asset_id, "other").await.is_err());
    let source = h
        .store
        .videos
        .published
        .lock()
        .unwrap()
        .get(asset_id)
        .cloned()
        .unwrap();
    let destination = h.store.videos.destination().unwrap();
    let proof = serde_json::to_value(destination.proof().unwrap()).unwrap();
    let mut pending = destination
        .reserve_image(&proof, 1024, Some(&source))
        .unwrap();
    let variant = pending
        .publish_image(&png(), "image", "image/png", None)
        .unwrap();
    assert_eq!(variant.path, path.with_file_name("image_0001_upscaled.png"));
    pending.commit();
    std::fs::write(&path, b"own externally changed fixture").unwrap();
    assert!(h.store.read_asset(asset_id, "local").await.is_err());
    h.control.call(Action::Shutdown).await.unwrap();
    h.task.await.unwrap().unwrap();
    h.store.close().await.unwrap();
    assert!(path.exists());
    assert!(variant.path.exists());
}
#[tokio::test]
async fn images_fail_closed_before_engine_when_destination_missing_or_proof_wrong() {
    let h = setup().await;
    let proof =
        serde_json::to_value(h.store.videos.destination().unwrap().proof().unwrap()).unwrap();
    let mut request: imagine_server::contract::JobRequest = serde_json::from_value(json!({"model":"qwen-image-2.1-int8","operation":"text_to_image","prompt":"synthetic","width":1600,"height":896,"seed":42,"client_metadata":{"video_destination":proof}})).unwrap();
    request.client_metadata["video_destination"]["host"] = "wrong-host".into();
    let error = h
        .control
        .call(Action::Submit {
            owner: "local".into(),
            request: Box::new(request.clone()),
            key: Some("invalid-destination".into()),
        })
        .await
        .unwrap_err();
    assert_eq!(error.code, "video_destination_mismatch");
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    request
        .client_metadata
        .insert("video_destination".into(), proof.clone());
    let root = std::path::PathBuf::from(proof["config"]["root"].as_str().unwrap());
    let hidden = root.with_file_name("own-temporarily-unavailable");
    std::fs::rename(&root, &hidden).unwrap();
    let error = h
        .control
        .call(Action::Submit {
            owner: "local".into(),
            request: Box::new(request),
            key: Some("missing-destination".into()),
        })
        .await
        .unwrap_err();
    assert!(
        error.code.starts_with("video_destination_"),
        "{}",
        error.code
    );
    assert!(!root.exists());
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    std::fs::rename(&hidden, &root).unwrap();
    stop(h).await;
}
