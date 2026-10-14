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
async fn restart_preserves_pause_queue_and_marks_interrupted() {
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
    assert!(reopened.paused().await.unwrap());
    assert_eq!(
        reopened
            .job(first["id"].as_str().unwrap(), "a")
            .await
            .unwrap()
            .unwrap()["status"],
        "interrupted"
    );
    assert_eq!(
        reopened
            .job(second["id"].as_str().unwrap(), "a")
            .await
            .unwrap()
            .unwrap()["status"],
        "queued"
    );
    assert_eq!(
        reopened
            .operation(operation["id"].as_str().unwrap())
            .await
            .unwrap()
            .unwrap()["status"],
        "interrupted"
    );
    let probe = Arc::new(Probe::default());
    let (control, task) = scheduler::start(reopened, Box::new(Fake(probe.clone())))
        .await
        .unwrap();
    control.call(Action::Resume).await.unwrap();
    wait_calls(&probe, 1).await;
    control.call(Action::Shutdown).await.unwrap();
    task.await.unwrap().unwrap();
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
async fn crash_recovery_does_not_replay_active_rows() {
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
    store.db.close().await;
    let reopened = Store::open(dir.path()).await.unwrap();
    for id in ["active", "cancelling"] {
        let job = reopened.job(id, "a").await.unwrap().unwrap();
        assert_eq!(job["status"], "interrupted");
        assert_eq!(job["error"]["code"], "server_restarted");
        assert!(job["output"].is_null());
    }
    assert_eq!(
        reopened.job("queued", "a").await.unwrap().unwrap()["status"],
        "queued"
    );
}
