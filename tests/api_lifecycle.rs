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
    engine_state: watch::Sender<EngineState>,
    dir: tempfile::TempDir,
    store: Store,
    control: Control,
    task: JoinHandle<anyhow::Result<()>>,
    router: Router,
    probe: Arc<Probe>,
}
async fn setup() -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).await.unwrap();
    let probe = Arc::new(Probe::default());
    let (control, task) = scheduler::start(store.clone(), Box::new(Fake(probe.clone())))
        .await
        .unwrap();
    let (engine_state, engine) = watch::channel(EngineState::default());

    let router = api::router(App {
        profiles: Arc::new(
            imagine_server::profiles::Catalog::load(std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/profiles"
            )))
            .unwrap(),
        ),
        store: store.clone(),
        control: control.clone(),
        engine,

        uploads: Arc::new(Semaphore::new(2)),
    });
    Harness {
        engine_state,
        dir,
        store,
        control,
        task,
        router,
        probe,
    }
}
async fn req(
    router: &Router,
    method: &str,
    url: &str,
    value: Option<Value>,
    key: Option<&str>,
) -> (u16, Value) {
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
        "model_not_validated"
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
        (
            "image_generation",
            include_str!("../proxy/workflows/prompts/prompt-enhance-scene.md"),
        ),
        (
            "qwen_scene",
            include_str!("../proxy/workflows/prompts/reference-aware-scene.md"),
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
    let body = json!({"profile":"qwen","profile_version":2,"prompt":"synthetic profile test", "width":1024,"sampling":{"steps":12},"model_options":{"qwen_image_2_1":{"generation":{"loras":[{"path":"synthetic.safetensors","strength":0.5}]}}}});
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
    assert_eq!(effective["resolved_profile"]["version"], 2);
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
    // Frozen v2 still uses the original settings and native reference preprocessing.
    let (_, frozen) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        Some(json!({"profile":"qwen-image21-1600x896-v2","prompt":"test"})),
        None,
    )
    .await;
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
async fn session_assets_are_plain_owned_and_removed_on_close() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = png();
    let store = Store::open(dir.path()).await.unwrap();
    let asset = store.save_asset("local", bytes.clone()).await.unwrap();
    let id = asset["id"].as_str().unwrap();
    let path = store.media.join(id);
    let disk = std::fs::read(&path).unwrap();
    assert_eq!(disk, bytes);
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
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
    let store = Store::open(dir.path()).await.unwrap();
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
            .publish_output("job_test", "local", artifacts)
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
                vec![imagine_server::generation::Artifact {
                    role: "image",
                    bytes: png()
                }]
            )
            .await
            .is_err()
    );
    assert_eq!(std::fs::read_dir(&store.media).unwrap().count(), 0);
    store.close().await.unwrap();
}
