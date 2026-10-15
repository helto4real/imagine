use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use serde_json::{Value, json};
use st_proxy::{config::Config, mcp::Mcp};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
#[derive(Default)]
struct Probe {
    keys: Mutex<Vec<String>>,
    polls: AtomicUsize,
    downloads: AtomicUsize,
    available: AtomicBool,
}
async fn submit(State(p): State<Arc<Probe>>, headers: HeaderMap) -> (StatusCode, Json<Value>) {
    let mut keys = p.keys.lock().unwrap();
    keys.push(headers["Idempotency-Key"].to_str().unwrap().into());
    // Simulate an admitted job whose first submission response was lost.
    if keys.len() == 1 {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":{"code":"unavailable"}})),
        )
    } else {
        (StatusCode::OK, Json(json!({"id":"one-job"})))
    }
}
async fn status(State(p): State<Arc<Probe>>) -> (StatusCode, Json<Value>) {
    p.polls.fetch_add(1, Ordering::SeqCst);
    if !p.available.load(Ordering::SeqCst) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error":{"code":"unavailable"}})),
        )
    } else {
        (
            StatusCode::OK,
            Json(
                json!({"status":"succeeded","effective":{"seed":42},"output":{"assets":[{"id":"image"}]}}),
            ),
        )
    }
}
async fn download(State(p): State<Arc<Probe>>) -> (StatusCode, Vec<u8>) {
    if p.downloads.fetch_add(1, Ordering::SeqCst) == 0 {
        (StatusCode::SERVICE_UNAVAILABLE, Vec::new())
    } else {
        (StatusCode::OK, vec![137u8, 80, 78, 71])
    }
}
#[tokio::test]
async fn ambiguous_submission_and_poll_failures_recover_same_job() {
    let p = Arc::new(Probe::default());
    let router = Router::new()
        .route(
            "/v1/session",
            get(|| async { Json(json!({"id":"synthetic-session"})) }),
        )
        .route(
            "/v1/profiles/qwen",
            get(|| async { Json(json!({"version":1,"references":{"min":0,"max":10}})) }),
        )
        .route(
            "/v1/jobs/validate",
            post(|| async { Json(json!({"effective":{"seed":42}})) }),
        )
        .route("/v1/jobs", post(submit))
        .route("/v1/jobs/one-job", get(status))
        .route("/v1/assets/image", get(download))
        .with_state(p.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let access = dir.path().join("access.json");
    std::fs::write(&access, r#"{"admin_token":"synthetic"}"#).unwrap();
    let mcp = Mcp::new(Config {
        imagine_url: url,
        imagine_access_file: Some(access.to_string_lossy().into()),
        coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
        mcp_output_dir: Some(dir.path().join("output").to_string_lossy().into()),
        poll_interval: 0.01,
        ..Default::default()
    })
    .await
    .unwrap();
    let args = json!({"job_id":"recover","profile":"qwen","parameters":{"prompt":"synthetic"}});
    let mc = mcp.clone();
    let a = args.clone();
    let generation = tokio::spawn(async move { mc.call("generate", &a).await });
    tokio::time::timeout(Duration::from_secs(5), async {
        while p.polls.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let state = mcp
        .call("get_job", &json!({"job_id":"recover","wait":false}))
        .await;
    assert_eq!(state["structuredContent"]["status"], "recovering");
    let mc = mcp.clone();
    let a = args.clone();
    let retry = tokio::spawn(async move { mc.call("generate", &a).await });
    p.available.store(true, Ordering::SeqCst);
    let result = tokio::time::timeout(Duration::from_secs(5), generation)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(result["structuredContent"]["status"], "succeeded");
    assert_eq!(result["structuredContent"]["error"], Value::Null);
    assert_eq!(result["structuredContent"]["parameters"]["seed"], 42);
    assert_eq!(retry.await.unwrap(), result);
    assert_eq!(p.downloads.load(Ordering::SeqCst), 2);
    {
        let keys = p.keys.lock().unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], keys[1]);
    }
    let temporary = std::path::PathBuf::from(
        result["structuredContent"]["images"][0]["path"]
            .as_str()
            .unwrap(),
    );
    let saved = dir.path().join("saved.png");
    std::fs::copy(&temporary, &saved).unwrap();
    assert!(temporary.exists());
    // Also interrupt an active recovery. Its HTTP caller must be released.
    p.available.store(false, Ordering::SeqCst);
    let previous_polls = p.polls.load(Ordering::SeqCst);
    let mc = mcp.clone();
    let pending = tokio::spawn(async move {
        mc.call(
            "generate",
            &json!({"job_id":"pending","profile":"qwen","parameters":{"prompt":"synthetic"}}),
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while p.polls.load(Ordering::SeqCst) == previous_polls {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    // Stopping only Imagine must also purge the still-running proxy's exports.
    server.abort();
    let _ = server.await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while temporary.exists() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(saved.exists());
    let interrupted = tokio::time::timeout(Duration::from_secs(5), pending)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        interrupted["structuredContent"]["error"]["category"],
        "privacy"
    );
    let stale = mcp
        .call("get_job", &json!({"job_id":"recover","wait":false}))
        .await;
    assert_eq!(stale["structuredContent"]["error"]["category"], "privacy");
    mcp.close().await;
}
