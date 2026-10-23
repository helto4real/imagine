#[path = "support/destination.rs"]
mod destination;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::{get, post},
};
use serde_json::{Value, json};
use st_proxy::{config::Config, mcp::Mcp};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
#[derive(Default)]
struct Probe {
    keys: Mutex<Vec<String>>,
    polls: AtomicUsize,
    downloads: AtomicUsize,
    available: AtomicBool,
    output: Value,
}
async fn submit(State(p): State<Arc<Probe>>, headers: HeaderMap) -> (StatusCode, Json<Value>) {
    assert_eq!(headers["X-Imagine-Session"], "synthetic-session");
    let mut keys = p.keys.lock().unwrap();
    keys.push(headers["Idempotency-Key"].to_str().unwrap().into());
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
                json!({"status":"succeeded","effective":{"seed":42},"output":{"assets":[p.output]}}),
            ),
        )
    }
}
async fn download(State(p): State<Arc<Probe>>) -> StatusCode {
    p.downloads.fetch_add(1, Ordering::SeqCst);
    StatusCode::INTERNAL_SERVER_ERROR
}
#[tokio::test]
async fn ambiguous_submission_and_poll_failures_recover_same_persistent_job_without_download() {
    let dir = tempfile::tempdir().unwrap();
    let published = dir.path().join("projects/own-workspace/outputs/images");
    std::fs::create_dir_all(&published).unwrap();
    let path = published.join("own-published.png");
    let bytes = b"own procedural image payload";
    std::fs::write(&path, bytes).unwrap();
    let proof = destination::proof(&published);
    let p = Arc::new(Probe {
        output: json!({"id":"image","persistent":true,"saved_path":path,"mime_type":"image/png","bytes":bytes.len(),"sha256":st_proxy::references::hash(bytes)}),
        ..Default::default()
    });
    let router = Router::new()
        .route(
            "/v1/media-destination",
            get(move || {
                let p = proof.clone();
                async move { Json(p) }
            }),
        )
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
        .with_state(p.clone())
        .layer(axum::middleware::from_fn(
            |request: axum::extract::Request, next: axum::middleware::Next| async move {
                assert!(!request.headers().contains_key("authorization"));
                next.run(request).await
            },
        ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let mcp = Mcp::new(Config {
        imagine_url: url,
        coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
        mcp_output_dir: Some(dir.path().join("old-cache").to_string_lossy().into()),
        poll_interval: 0.01,
        ..Default::default()
    })
    .await
    .unwrap();
    let args = json!({"job_id":"recover","profile":"qwen","parameters":{"prompt":"own synthetic fixture"}});
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
    assert_eq!(
        mcp.call("get_job", &json!({"job_id":"recover","wait":false}))
            .await["structuredContent"]["status"],
        "recovering"
    );
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
    assert_eq!(
        result["structuredContent"]["images"][0]["path"],
        json!(path)
    );
    assert_eq!(result["structuredContent"]["images"][0]["storage"], "saved");
    assert_eq!(p.downloads.load(Ordering::SeqCst), 0);
    {
        let keys = p.keys.lock().unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0], keys[1]);
    }
    assert!(!dir.path().join("old-cache").exists());
    assert_eq!(std::fs::read_dir(&published).unwrap().count(), 1);
    p.available.store(false, Ordering::SeqCst);
    let previous = p.polls.load(Ordering::SeqCst);
    let mc = mcp.clone();
    let pending = tokio::spawn(async move {
        mc.call("generate",&json!({"job_id":"pending","profile":"qwen","parameters":{"prompt":"own pending fixture"}})).await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while p.polls.load(Ordering::SeqCst) == previous {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    server.abort();
    let _ = server.await;
    assert_eq!(
        mcp.call("get_job", &json!({"job_id":"recover","wait":false}))
            .await["structuredContent"]["error"]["category"],
        "privacy"
    );
    let interrupted = tokio::time::timeout(Duration::from_secs(5), pending)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        interrupted["structuredContent"]["error"]["category"],
        "privacy"
    );
    mcp.close().await;
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
}
#[tokio::test]
async fn register_links_confined_original_with_ram_preview_and_rejects_changed_original() {
    let dir = tempfile::tempdir().unwrap();
    let published = dir.path().join("outputs/images");
    std::fs::create_dir_all(&published).unwrap();
    let proof = destination::proof(&published);
    let source = dir.path().join("own-source.png");
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(8, 8)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    std::fs::write(&source, png.get_ref()).unwrap();
    let linked = Arc::new(AtomicUsize::new(0));
    let flag = linked.clone();
    let original = source.clone();
    let digest = st_proxy::references::hash(png.get_ref());
    let length = png.get_ref().len();
    let router=Router::new()
        .route("/v1/media-destination",get(move || {let p=proof.clone();async move {Json(p)}}))
        .route("/v1/session",get(||async {Json(json!({"id":"synthetic-session"}))}))
        .route("/v1/profiles/i2v",get(||async {Json(json!({"version":1,"references":{"min":1,"max":1}}))}))
        .route("/v1/assets/link",post(move |headers:HeaderMap,Json(body):Json<Value>| {let flag=flag.clone();let original=original.clone();let digest=digest.clone();async move {
            assert_eq!(headers["X-Imagine-Session"],"synthetic-session");assert_eq!(body,json!({"path":original,"normalize_image":true}));
            flag.fetch_add(1,Ordering::SeqCst);Json(json!({"id":"own-linked","mime_type":"image/png","sha256":digest,"bytes":length}))
        }}));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let mcp = Mcp::new(Config {
        imagine_url: url,
        coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
        mcp_input_dir: dir.path().to_string_lossy().into(),
        mcp_output_dir: Some(dir.path().join("old-cache").to_string_lossy().into()),
        ..Default::default()
    })
    .await
    .unwrap();
    let result = mcp
        .call("register_image", &json!({"profile":"i2v","path":source}))
        .await;
    assert_eq!(result["isError"], false, "{result}");
    assert_eq!(linked.load(Ordering::SeqCst), 1);
    assert_eq!(result["structuredContent"]["preview_path"], json!(source));
    assert_eq!(result["structuredContent"]["preview_storage"], "memory");
    assert_eq!(std::fs::read(&source).unwrap(), *png.get_ref());
    assert!(!dir.path().join("old-cache").exists());
    let outside = tempfile::tempdir().unwrap();
    let outside_source = outside.path().join("own-outside.png");
    std::fs::write(&outside_source, png.get_ref()).unwrap();
    assert_eq!(
        mcp.call(
            "register_image",
            &json!({"profile":"i2v","path":outside_source})
        )
        .await["isError"],
        true
    );
    assert_eq!(linked.load(Ordering::SeqCst), 1);
    std::fs::write(&source, b"own changed original").unwrap();
    let changed=mcp.call("generate",&json!({"profile":"i2v","job_id":"own-changed","parameters":{},"references":[result["structuredContent"]["image_id"]]})).await;
    assert_eq!(changed["isError"], true, "{changed}");
    assert!(
        changed["structuredContent"]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("registered reference has changed")
    );
    mcp.close().await;
    assert_eq!(std::fs::read(&source).unwrap(), b"own changed original");
    server.abort();
    let _ = server.await;
}
