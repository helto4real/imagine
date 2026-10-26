#[path = "../ui/tests/support/destination.rs"]
#[allow(dead_code)]
mod destination;
use axum::{
    Json, Router,
    routing::{get, post},
};
use destination::task_fixtures;
use imagine_server::{
    api::{self, App},
    contract::MODEL,
    engine::{Backend, EngineFuture, EngineState},
    private_storage::Root,
    scheduler::{self, Action},
    storage::Store,
    video_destination as d,
};
use serde_json::{Value, json};
use std::{
    io::Cursor,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::sync::{Notify, Semaphore, watch};
#[derive(Default)]
struct Probe {
    llm: AtomicBool,
    image: AtomicBool,
    calls: AtomicUsize,
    stops: AtomicUsize,
    unloads: AtomicUsize,
    fail_next: AtomicBool,
    release: Notify,
}
struct Engine(Arc<Probe>);
impl Backend for Engine {
    fn generate<'a>(
        &'a mut self,
        _: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            assert!(!self.0.llm.load(Ordering::SeqCst), "LLM overlapped Imagine");
            self.0.image.store(true, Ordering::SeqCst);
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            if self.0.fail_next.swap(false, Ordering::SeqCst) {
                return Err(imagine_server::engine::EngineError("native_generation_failed").into());
            }
            self.0.release.notified().await;
            let mut bytes = Cursor::new(Vec::new());
            image::RgbImage::new(2, 2).write_to(&mut bytes, image::ImageFormat::Png)?;
            Ok(bytes.into_inner())
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            self.0.image.store(false, Ordering::SeqCst);
            self.0.stops.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }
}
async fn until(mut f: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(12), async {
        while !f() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}
async fn serve(router: Router) -> (String, tokio::task::JoinHandle<()>) {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let u = format!("http://{}", l.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(l, router).await.unwrap();
    });
    (u, task)
}
/// A private root on the owned task volume with an MCP root for workspaces.
fn private_root(dir: &std::path::Path) -> (Root, std::path::PathBuf) {
    let root = dir.join("own-imagine");
    let mcp = dir.join("own-mcp-root");
    let runtime = dir.join("own-runtime");
    for path in [&root, &mcp, &runtime] {
        std::fs::create_dir(path).unwrap();
    }
    let config = d::Config {
        fsid: d::fsid(&d::directory(&root).unwrap()).unwrap(),
        root,
        mount: task_fixtures::MOUNT.into(),
        source: task_fixtures::SOURCE.into(),
        filesystem: task_fixtures::FILESYSTEM.into(),
    };
    let (root, ignored) = Root::with_environment(config, Arc::new(d::System))
        .unwrap()
        .with_runtime_directory(runtime)
        .unwrap()
        .with_project_roots(std::slice::from_ref(&mcp));
    assert!(ignored.is_empty());
    (root, mcp)
}
#[tokio::test]
async fn direct_jobs_share_fifo_and_coordinator_loss_stops_gpu() {
    let dir = tempfile::tempdir().unwrap();
    assert!(task_fixtures::is_owned(dir.path()));
    let (root, mcp_root) = private_root(dir.path());
    let mut store = Store::open_private(root.clone(), "own-image-workspace".into())
        .await
        .unwrap();
    store.workspaces = Some(Arc::new(imagine_server::workspaces::Workspaces::open(
        root.rooted_at(&mcp_root).unwrap(),
    )));
    let probe = Arc::new(Probe::default());
    let socket = dir.path().join("gpu.sock");
    let (control, scheduler) = scheduler::start_coordinated(
        store.clone(),
        Box::new(Engine(probe.clone())),
        Some(socket.clone()),
    )
    .await
    .unwrap();

    let (_, engine) = watch::channel(EngineState::default());
    let (imagine, imagine_task) = serve(api::router(App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open_private(
            root.clone(),
            "library/enhancement-profiles.json".into(),
        )
        .unwrap(),
        library: Default::default(),
        profiles: Arc::new(
            imagine_server::profiles::Catalog::load(std::path::Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/config/image/profiles"
            )))
            .unwrap(),
        ),
        store: store.clone(),
        control: control.clone(),
        engine,

        uploads: Arc::new(Semaphore::new(2)),
    }))
    .await;
    let p = probe.clone();
    let health = move || {
        let p = p.clone();
        async move {
            Json(
                json!({"service":"strata","status":"ok","loaded":p.llm.load(Ordering::SeqCst),"model":"synthetic"}),
            )
        }
    };
    let p = probe.clone();
    let load = move || {
        let p = p.clone();
        async move {
            assert!(!p.image.load(Ordering::SeqCst));
            p.llm.store(true, Ordering::SeqCst);
            Json(json!({"status":"loaded"}))
        }
    };
    let p = probe.clone();
    let unload = move || {
        let p = p.clone();
        async move {
            p.unloads.fetch_add(1, Ordering::SeqCst);
            p.llm.store(false, Ordering::SeqCst);
            Json(json!({"status":"unloaded"}))
        }
    };
    let (llm, llm_task) = serve(
        Router::new()
            .route("/health", get(health))
            .route(
                "/v1/models",
                get(|| async { Json(json!({"data":[{"id":"synthetic"}]})) }),
            )
            .route("/v1/load", post(load))
            .route("/v1/unload", post(unload))
            .route(
                "/v1/chat/completions",
                post(|| async { Json(json!({"choices":[]})) }),
            ),
    )
    .await;
    let config = st_proxy::config::Config {
        mcp_port: Some(0),
        mcp_workflows_dir: "/nonexistent/no-comfy-graphs-needed".into(),
        mcp_output_dir: Some(dir.path().join("output").to_string_lossy().into()),
        chat_port: 0,
        image_port: 0,
        llm_backend: "sc117".into(),
        llm_url: llm,
        comfy_enabled: false,
        comfy_url: "http://127.0.0.1:1".into(),
        coordinator_socket: Some(socket.to_str().unwrap().into()),
        imagine_url: imagine.clone(),

        poll_interval: 0.01,
        ..Default::default()
    };
    let running = st_proxy::service::Service::start(config).await.unwrap();
    let client = reqwest::Client::new();
    let blocked = client
        .post(format!("http://127.0.0.1:{}/prompt", running.image_port))
        .json(&json!({"prompt":{}}))
        .send()
        .await
        .unwrap();
    assert_eq!(blocked.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        blocked.json::<Value>().await.unwrap()["error"]["code"],
        "comfy_disabled"
    );

    // Invalid MCP calls must never unload the agent's already-loaded LLM.
    let agent_url = format!("http://127.0.0.1:{}/v1/chat/completions", running.chat_port);
    let mcp_url = format!("http://127.0.0.1:{}/mcp", running.mcp_port.unwrap());
    assert!(
        client
            .post(&agent_url)
            .json(&json!({"model":"synthetic","messages":[]}))
            .send()
            .await
            .unwrap()
            .status()
            .is_success()
    );
    assert!(probe.llm.load(Ordering::SeqCst));
    // The agent's workspace; every image call carries its key.
    let key = mcp(
        &client,
        &mcp_url,
        "workspace_create",
        json!({"title":"Own Images"}),
    )
    .await["workspace_key"]
        .as_str()
        .unwrap()
        .to_owned();
    let unloads = probe.unloads.load(Ordering::SeqCst);
    for (tool, mut args) in [
        (
            "generate",
            json!({"job_id":"bad-size","profile":"qwen","parameters":{"negative_prompt":"synthetic negative","prompt":"synthetic","max_side":4128}}),
        ),
        (
            "generate",
            json!({"job_id":"bad-sampling","profile":"qwen","parameters":{"negative_prompt":"synthetic negative","prompt":"synthetic","sampling":{"steps":0}}}),
        ),
        (
            "generate",
            json!({"job_id":"missing-reference","profile":"image_upscale","parameters":{}}),
        ),
        (
            "generate",
            json!({"job_id":"unknown-profile","profile":"unknown","parameters":{"prompt":"synthetic"}}),
        ),
        (
            "generate_batch",
            json!({"job_id":"bad-batch","profile":"qwen","items":[{"prompt":"valid","negative_prompt":"synthetic negative"},{"prompt":"invalid","negative_prompt":"synthetic negative","max_side":4128}]}),
        ),
    ] {
        args["workspace_key"] = json!(key);
        let rejected = mcp(&client, &mcp_url, tool, args).await;
        assert_eq!(rejected["error"]["category"], "validation", "{rejected}");
        assert!(probe.llm.load(Ordering::SeqCst));
        assert_eq!(probe.unloads.load(Ordering::SeqCst), unloads);
        assert_eq!(probe.calls.load(Ordering::SeqCst), 0);
        assert_eq!(running.service.coordinator.status()["waiting_images"], 0);
    }
    // The next troubleshooting response needs no unload/reload cycle either.
    assert!(
        client
            .post(&agent_url)
            .json(&json!({"model":"synthetic","messages":[]}))
            .send()
            .await
            .unwrap()
            .status()
            .is_success()
    );
    assert_eq!(probe.unloads.load(Ordering::SeqCst), unloads);

    let submit = || {
        client
            .post(format!("{imagine}/v1/jobs"))
            .json(&json!({"model":MODEL,"operation":"text_to_image","prompt":"synthetic"}))
    };
    let first: Value = submit().send().await.unwrap().json().await.unwrap();
    until(|| probe.calls.load(Ordering::SeqCst) == 1).await;
    let chat_url = format!("http://127.0.0.1:{}/v1/chat/completions", running.chat_port);
    let c = client.clone();
    let chat = tokio::spawn(async move {
        c.post(chat_url)
            .json(&json!({"model":"synthetic","messages":[]}))
            .send()
            .await
            .unwrap()
    });
    until(|| {
        running.service.coordinator.status()["waiting_chats"].as_u64() == Some(1)
            && running.service.coordinator.status()["state"] == "image_active"
    })
    .await;
    let cancelled: Value = submit().send().await.unwrap().json().await.unwrap();
    until(|| running.service.coordinator.status()["waiting_images"] == 1).await;
    let response = client
        .post(format!(
            "{imagine}/v1/jobs/{}/cancel",
            cancelled["id"].as_str().unwrap()
        ))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_success());
    until(|| running.service.coordinator.status()["waiting_images"] == 0).await;
    let second: Value = submit().send().await.unwrap().json().await.unwrap();
    assert_ne!(first["id"], second["id"]);
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!chat.is_finished());
    assert_eq!(probe.calls.load(Ordering::SeqCst), 1);
    probe.release.notify_one();
    assert!(chat.await.unwrap().status().is_success());
    until(|| probe.calls.load(Ordering::SeqCst) == 2).await;
    assert!(!probe.llm.load(Ordering::SeqCst));
    // MCP goes through the public Imagine API. Batches occupy consecutive
    // positions, retry attaches to the original jobs, and 53-bit seeds survive.
    let mcp_url = format!("http://127.0.0.1:{}/mcp", running.mcp_port.unwrap());
    let list = mcp(&client, &mcp_url, "list_profiles", json!({})).await;
    assert!(
        list["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["id"] == "qwen")
    );
    let described = mcp(
        &client,
        &mcp_url,
        "describe_profile",
        json!({"profile":"quick-draft"}),
    )
    .await;
    assert_eq!(described["defaults"]["sampling"]["steps"], 8);
    assert!(described["defaults"].get("prompt").is_none());
    let invalid = mcp(&client, &mcp_url, "generate_batch", json!({"workspace_key":key,"job_id":"invalid-profile-batch","profile":"qwen","items":[{"prompt":"valid","negative_prompt":"synthetic negative"},{"prompt":"bad","negative_prompt":"synthetic negative","sampling":{"steps":0}}]})).await;
    assert_eq!(invalid["error"]["category"], "validation");
    let missing = mcp(
        &client,
        &mcp_url,
        "get_job",
        json!({"workspace_key":key,"job_id":"invalid-profile-batch","wait":false}),
    )
    .await;
    assert_eq!(missing["error"]["category"], "not_found");
    assert_eq!(probe.calls.load(Ordering::SeqCst), 2);
    let mut synthetic = Cursor::new(Vec::new());
    image::RgbImage::new(64, 64)
        .write_to(&mut synthetic, image::ImageFormat::Png)
        .unwrap();
    use base64::Engine as _;
    let uploaded = mcp(
        &client,
        &mcp_url,
        "workspace_upload",
        json!({"workspace_key":key,"name":"synthetic.png",
            "content_base64":base64::engine::general_purpose::STANDARD.encode(synthetic.get_ref())}),
    )
    .await;
    let registered = mcp(
        &client,
        &mcp_url,
        "register_image",
        json!({"workspace_key":key,"profile":"quick-draft","file_id":uploaded["file_id"]}),
    )
    .await;
    assert!(registered["image_id"].is_string(), "{registered}");
    let args = json!({"workspace_key":key,"job_id":"batch-synthetic","profile":"quick-draft","items":[{"parameters":{"prompt":"synthetic one","seed":9007199254740991u64},"references":[registered["image_id"]]},{"prompt":"synthetic two","nsfw":true}]});
    let c = client.clone();
    let u = mcp_url.clone();
    let a = args.clone();
    let batch = tokio::spawn(async move { mcp(&c, &u, "generate_batch", a).await });
    until(|| running.service.coordinator.status()["waiting_images"] == 2).await;
    let chat_url = format!("http://127.0.0.1:{}/v1/chat/completions", running.chat_port);
    let c = client.clone();
    let after_batch = tokio::spawn(async move {
        c.post(chat_url)
            .json(&json!({"model":"synthetic","messages":[]}))
            .send()
            .await
            .unwrap()
    });
    probe.release.notify_one();
    until(|| probe.calls.load(Ordering::SeqCst) == 3).await;
    assert!(!after_batch.is_finished());
    probe.release.notify_one();
    until(|| probe.calls.load(Ordering::SeqCst) == 4).await;
    assert!(!after_batch.is_finished());
    probe.release.notify_one();
    let result = batch.await.unwrap();
    assert_eq!(result["status"], "succeeded", "{result}");
    assert_eq!(
        result["items"][0]["parameters"]["seed"],
        9007199254740991u64
    );
    assert!(result["items"][0]["parameters"].get("prompt").is_none());
    assert_eq!(
        mcp(&client, &mcp_url, "generate_batch", args.clone()).await,
        result
    );
    assert_eq!(probe.calls.load(Ordering::SeqCst), 4);
    assert_eq!(
        result["items"][1]["parameters"]["model"],
        "qwen-image-2.1-turbo"
    );
    assert_eq!(result["items"][1]["parameters"]["sampling"]["steps"], 8);
    assert_eq!(
        result["items"][1]["parameters"]["model_options"]["qwen_image_2_1"]["generation"]["loras"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        result["items"][1]["parameters"]["resolved_profile"]["id"],
        "quick-draft"
    );
    assert!(result["items"][1]["parameters"]["seed"].is_u64());
    // Outputs are published in the workspace's image folder and named by file ID.
    let workspace = store.workspaces.as_ref().unwrap().resolve(&key).unwrap();
    let mut ids = Vec::new();
    for item in result["items"].as_array().unwrap() {
        let image = &item["images"][0];
        assert!(image.get("path").is_none() && image.get("saved_path").is_none());
        let entry = store
            .workspaces
            .as_ref()
            .unwrap()
            .file(&workspace, image["file_id"].as_str().unwrap())
            .unwrap();
        assert!(entry.path.starts_with(workspace.images()));
        let digest = st_proxy::references::hash(&std::fs::read(&entry.path).unwrap());
        assert_eq!(image["bytes"], entry.bytes);
        assert_eq!(image["sha256"], digest);
        ids.push(image["file_id"].clone());
    }
    assert_ne!(ids[0], ids[1]);
    let recovered = mcp(&client, &mcp_url, "generate_batch", args).await;
    assert_eq!(recovered["items"][0]["images"][0]["file_id"], ids[0]);
    assert_eq!(probe.calls.load(Ordering::SeqCst), 4);
    assert!(after_batch.await.unwrap().status().is_success());
    // An actual inference failure is distinct from validation. Its stable code
    // reaches MCP, and the agent can resume through the normal next chat request.
    probe.fail_next.store(true, Ordering::SeqCst);
    let failed = mcp(
        &client,
        &mcp_url,
        "generate",
        json!({"workspace_key":key,"job_id":"engine-failure","profile":"qwen","parameters":{"negative_prompt":"synthetic negative","prompt":"synthetic"}}),
    )
    .await;
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["error"]["category"], "generation");
    assert_eq!(failed["error"]["code"], "native_generation_failed");
    assert!(
        client
            .post(&agent_url)
            .json(&json!({"model":"synthetic","messages":[]}))
            .send()
            .await
            .unwrap()
            .status()
            .is_success()
    );
    assert!(probe.llm.load(Ordering::SeqCst));
    let third: Value = submit().send().await.unwrap().json().await.unwrap();
    until(|| probe.calls.load(Ordering::SeqCst) == 6).await;
    // Abrupt loss of the ticket socket while GPU work is active must stop the engine.
    running.close().await;
    until(|| !probe.image.load(Ordering::SeqCst)).await;
    let job = store
        .job(third["id"].as_str().unwrap(), "local")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(job["status"], "failed");
    assert_eq!(job["error"]["code"], "coordinator_unavailable");
    let rejected = submit().send().await.unwrap();
    assert_eq!(rejected.status(), 503);
    assert!(
        !socket.exists(),
        "coordinator socket must be removed on shutdown"
    );
    control.call(Action::Shutdown).await.unwrap();
    scheduler.await.unwrap().unwrap();
    imagine_task.abort();
    llm_task.abort();
}

async fn mcp(client: &reqwest::Client, url: &str, name: &str, args: Value) -> Value {
    let response=client.post(url).header("Accept","application/json, text/event-stream").json(&json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":args}})).send().await.unwrap();
    assert!(response.status().is_success());
    let body = response.text().await.unwrap();
    let data = body
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .unwrap();
    serde_json::from_str::<Value>(data).unwrap()["result"]["structuredContent"].clone()
}
