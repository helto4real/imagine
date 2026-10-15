use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use imagine_server::{
    api::{self, App},
    config::Access,
    contract::{JobRequest, MODEL},
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action, Control},
    storage::Store,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
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
        _: &'a JobRequest,
        _: &'a str,
        _: &'a Store,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            self.0.release.notified().await;
            if self.0.fail.swap(0, Ordering::SeqCst) > 0 {
                anyhow::bail!("simulated crash");
            }
            Ok(png())
        })
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
    let access = Access {
        admin_token: "root".into(),
        clients: BTreeMap::from([("a".into(), "alice".into()), ("b".into(), "bob".into())]),
    };
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
        access: Arc::new(access),
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
    token: &str,
    value: Option<Value>,
    key: Option<&str>,
) -> (u16, Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(url)
        .header("Authorization", format!("Bearer {token}"));
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
async fn progress_is_scoped_to_active_owned_job_in_polling_and_events() {
    let h = setup().await;
    let (_, job) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        None,
    )
    .await;
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
    assert!(req(&h.router, "GET", &url, "alice", None, None).await.1["progress"].is_null());
    h.engine_state
        .send_modify(|state| state.job_id = Some(id.into()));
    let data = req(&h.router, "GET", &url, "alice", None, None).await.1;
    assert_eq!(data["progress"]["completed"], 10);
    assert_eq!(data["progress"]["remaining_seconds"], 24.);
    assert_eq!(req(&h.router, "GET", &url, "bob", None, None).await.0, 404);
    let response = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("{url}/events"))
                .header("Authorization", "Bearer alice")
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
    assert!(req(&h.router, "GET", &url, "alice", None, None).await.1["progress"].is_null());
    stop(h).await;
}

#[tokio::test]
async fn queue_idempotency_unload_resume_and_cancel() {
    let h = setup().await;
    assert_eq!(
        req(&h.router, "GET", "/v1/models", "invalid", None, None)
            .await
            .0,
        401
    );
    assert_eq!(
        req(&h.router, "GET", "/v1/models", "alice", None, None)
            .await
            .0,
        200
    );
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 0);
    let (code, first) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        Some("same"),
    )
    .await;
    assert_eq!(code, 202);
    let id = first["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let (_, repeat) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        Some("same"),
    )
    .await;
    assert_eq!(first["id"], repeat["id"]);
    let mut different = request();
    different["prompt"] = json!("different");
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/jobs",
            "alice",
            Some(different),
            Some("same")
        )
        .await
        .0,
        409
    );
    let (_, second) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "bob",
        Some(request()),
        Some("same"),
    )
    .await;
    let second = second["id"].as_str().unwrap();
    assert_ne!(id, second);
    assert_eq!(
        req(
            &h.router,
            "GET",
            &format!("/v1/jobs/{id}"),
            "bob",
            None,
            None
        )
        .await
        .0,
        404
    );
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/runtime/unload",
            "alice",
            Some(json!({"active_job":"wait"})),
            None
        )
        .await
        .0,
        403
    );
    let (code, op) = req(
        &h.router,
        "POST",
        "/v1/runtime/unload",
        "root",
        Some(json!({"active_job":"wait"})),
        None,
    )
    .await;
    assert_eq!(code, 202);
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/jobs",
            "alice",
            Some(request()),
            None
        )
        .await
        .0,
        409
    );
    assert_eq!(
        req(&h.router, "POST", "/v1/runtime/resume", "root", None, None)
            .await
            .0,
        409
    );
    h.probe.release.notify_one();
    let first = wait_status(&h.store, id, "a", "succeeded").await;
    assert!(first["effective"]["seed"].is_u64());
    assert_eq!(first["output"]["assets"][0]["width"], 2);
    let asset = first["output"]["assets"][0]["id"].as_str().unwrap();
    assert!(h.store.read_asset(asset, "a").await.is_ok());
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
        h.store.job(second, "b").await.unwrap().unwrap()["status"],
        "queued"
    );
    assert_eq!(
        req(&h.router, "POST", "/v1/runtime/resume", "root", None, None)
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
            "bob",
            None,
            None
        )
        .await
        .0,
        202
    );
    let value = wait_status(&h.store, second, "b", "cancelled").await;
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
        req(&h.router, "POST", "/v1/jobs", "alice", Some(wrong), None)
            .await
            .0,
        422
    );
    let mut wrong = request();
    wrong["model"] = json!("minimax-h3");
    assert_eq!(
        req(&h.router, "POST", "/v1/jobs", "alice", Some(wrong), None)
            .await
            .1["error"]["code"],
        "model_not_validated"
    );
    let upload = Request::builder()
        .method("POST")
        .uri("/v1/assets")
        .header("Authorization", "Bearer alice")
        .header("Content-Type", "image/png")
        .body(Body::from(png()))
        .unwrap();
    let r = h.router.clone().oneshot(upload).await.unwrap();
    assert_eq!(r.status(), 201);
    let asset: Value =
        serde_json::from_slice(&to_bytes(r.into_body(), 4096).await.unwrap()).unwrap();
    let id = asset["id"].as_str().unwrap();
    assert_eq!(
        req(
            &h.router,
            "GET",
            &format!("/v1/assets/{id}"),
            "bob",
            None,
            None
        )
        .await
        .0,
        404
    );
    let mut edit = request();
    edit["operation"] = json!("image_edit");
    edit["inputs"] = json!({"reference_images":[id]});
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/jobs",
            "bob",
            Some(edit.clone()),
            None
        )
        .await
        .0,
        404
    );
    let (code, job) = req(&h.router, "POST", "/v1/jobs", "alice", Some(edit), None).await;
    assert_eq!(code, 202);
    let job_id = job["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let job = h.store.job(job_id, "a").await.unwrap().unwrap();
    assert_eq!(
        job["effective"]["reference_preprocessing"][0]["vae_width"],
        992
    );
    let (_, queued) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        None,
    )
    .await;
    let queued = queued["id"].as_str().unwrap();
    req(
        &h.router,
        "POST",
        &format!("/v1/jobs/{queued}/cancel"),
        "alice",
        None,
        None,
    )
    .await;
    wait_status(&h.store, queued, "a", "cancelled").await;
    assert_eq!(h.probe.calls.load(Ordering::SeqCst), 1);
    let (_, op) = req(
        &h.router,
        "POST",
        "/v1/runtime/unload",
        "root",
        Some(json!({"active_job":"cancel"})),
        None,
    )
    .await;
    wait_status(&h.store, job_id, "a", "cancelled").await;
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
    let (_, first) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        None,
    )
    .await;
    wait_calls(&h.probe, 1).await;
    let (_, second) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        None,
    )
    .await;
    let (_, operation) = req(
        &h.router,
        "POST",
        "/v1/runtime/unload",
        "root",
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
                .job(job["id"].as_str().unwrap(), "a")
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
    let (_, first) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        None,
    )
    .await;
    wait_calls(&h.probe, 1).await;
    let (_, second) = req(
        &h.router,
        "POST",
        "/v1/jobs",
        "alice",
        Some(request()),
        None,
    )
    .await;
    h.probe.release.notify_one();
    wait_status(&h.store, first["id"].as_str().unwrap(), "a", "failed").await;
    wait_calls(&h.probe, 2).await;
    h.probe.release.notify_one();
    let id = second["id"].as_str().unwrap();
    wait_status(&h.store, id, "a", "succeeded").await;
    let r = h
        .router
        .clone()
        .oneshot(
            Request::builder()
                .uri(format!("/v1/jobs/{id}/events"))
                .header("Authorization", "Bearer alice")
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
        assert!(reopened.job(id, "a").await.unwrap().is_none());
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
        "alice",
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
        "alice",
        Some(batch.clone()),
        Some("batch"),
    )
    .await;
    assert_eq!(status, 202);
    let (status, retry) = req(
        &h.router,
        "POST",
        "/v1/jobs/batch",
        "alice",
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
        "alice",
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
    assert_eq!(
        req(&h.router, "GET", "/v1/profiles", "invalid", None, None)
            .await
            .0,
        401
    );
    let (status, catalog) = req(&h.router, "GET", "/v1/profiles", "alice", None, None).await;
    assert_eq!(status, 200);
    assert!(
        catalog["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["id"] == "qwen")
    );
    let (_, description) = req(&h.router, "GET", "/v1/profiles/qwen", "alice", None, None).await;
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
            "alice",
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
        "alice",
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
        "alice",
        Some(body.clone()),
        Some("profile-retry"),
    )
    .await;
    assert_eq!(status, 202, "{submitted}");
    let id = submitted["id"].as_str().unwrap();
    wait_calls(&h.probe, 1).await;
    let effective = h.store.job(id, "a").await.unwrap().unwrap()["effective"].clone();
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
        "alice",
        Some(body),
        Some("profile-retry"),
    )
    .await;
    assert_eq!(retry["id"], id);
    assert_eq!(
        h.store.job(id, "a").await.unwrap().unwrap()["effective"]["seed"],
        effective["seed"]
    );
    let (_, cleared) = req(&h.router, "POST", "/v1/jobs/validate", "alice", Some(json!({"profile":"quick-draft","prompt":"synthetic","sampling":{"custom_sigmas":[]},"model_options":{"qwen_image_2_1":{"generation":{"loras":[]}}}})), None).await;
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
    wait_status(&h.store, id, "a", "succeeded").await;
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
            "alice",
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
        "alice",
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
async fn profile_reference_sizing_and_asset_ownership() {
    let h = setup().await;
    let mut bytes = Cursor::new(Vec::new());
    image::RgbImage::new(256, 128)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let asset = h.store.save_asset("a", bytes.into_inner()).await.unwrap();
    let body = json!({"profile":"image_upscale","prompt":"synthetic upscale","negative_prompt":"","max_side":512,"inputs":{"reference_images":[asset["id"]]}});
    let (status, v) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        "alice",
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
    assert_eq!(
        req(
            &h.router,
            "POST",
            "/v1/jobs/validate",
            "bob",
            Some(body),
            None
        )
        .await
        .0,
        404
    );
    let (_, v) = req(
        &h.router,
        "POST",
        "/v1/jobs/validate",
        "alice",
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
    let asset = store.save_asset("a", bytes.clone()).await.unwrap();
    let id = asset["id"].as_str().unwrap();
    let path = store.media.join(id);
    let disk = std::fs::read(&path).unwrap();
    assert_eq!(disk, bytes);
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(store.read_asset(id, "a").await.unwrap(), bytes);
    assert!(store.read_asset(id, "other").await.is_err());
    store.close().await.unwrap();
    assert!(!path.exists());
}

#[tokio::test]
async fn old_session_requests_are_rejected_before_mutation() {
    let h = setup().await;
    let (status, session) = req(&h.router, "GET", "/v1/session", "alice", None, None).await;
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
                .header("Authorization", "Bearer alice")
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
