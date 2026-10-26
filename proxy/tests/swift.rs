use axum::{
    Json, Router,
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;
use st_proxy::{
    backend::{Backend, Backends},
    config::Config,
    profiles::Profiles,
};
use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

const MODEL: &str = "sc117-swift-1.5-qwen3.8-flash-next-gsq-rco-abliterated-iq3_xxs";

#[derive(Clone)]
struct Fake {
    kind: &'static str,
    loaded: Arc<AtomicBool>,
    reject_unload: Arc<AtomicBool>,
    events: Arc<Mutex<Vec<String>>>,
}
async fn respond(State(fake): State<Fake>, request: Request) -> Response {
    match (request.method().as_str(), request.uri().path()) {
        ("GET", "/health") => Json(json!({"service":"strata","status":"ok","model":MODEL,"loaded":fake.loaded.load(Ordering::SeqCst)})).into_response(),
        ("GET", "/v1/models") => Json(json!({"data":[{"id":MODEL,"meta":{"n_ctx":131072}}]})).into_response(),
        ("POST", "/v1/load") => {
            fake.events.lock().unwrap().push(format!("{}:load",fake.kind));
            fake.loaded.store(true,Ordering::SeqCst);
            Json(json!({"status":"loaded"})).into_response()
        }
        ("POST", "/v1/unload") => {
            fake.events.lock().unwrap().push(format!("{}:unload",fake.kind));
            if fake.reject_unload.load(Ordering::SeqCst) {
                return StatusCode::INTERNAL_SERVER_ERROR.into_response();
            }
            fake.loaded.store(false,Ordering::SeqCst);
            Json(json!({"status":"unloaded"})).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}
async fn server(
    kind: &'static str,
    events: Arc<Mutex<Vec<String>>>,
) -> (String, Fake, tokio::task::JoinHandle<()>) {
    let fake = Fake {
        kind,
        loaded: Arc::new(AtomicBool::new(false)),
        reject_unload: Arc::new(AtomicBool::new(false)),
        events,
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let origin = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new().fallback(respond).with_state(fake.clone());
    let task = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    (origin, fake, task)
}

#[test]
fn swift_origins_and_cli_preserve_distinct_backends() {
    for (kind, port) in [("swift", 48648), ("swift_128k", 48650)] {
        let (cfg, _) =
            Config::parse(["--llm-backend".into(), kind.into()], &HashMap::new()).unwrap();
        assert_eq!(cfg.llm_url, format!("http://127.0.0.1:{port}"));
        assert!(Backend::new(kind, &cfg.llm_url, &cfg, reqwest::Client::new()).strata());
    }
    let (mut cfg, _) = Config::parse(
        [
            "--llm-backend".into(),
            "all".into(),
            "--swift-url".into(),
            "http://127.0.0.1:48748".into(),
        ],
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(cfg.swift_url, "http://127.0.0.1:48748");
    let backends = Backends::new(&cfg, reqwest::Client::new());
    assert!(backends.entries.iter().any(|b| b.kind == "swift_128k"));
    cfg.swift_url = cfg.sc117_url.clone();
    assert!(cfg.validate().is_err());
}

#[test]
fn swift_profile_uses_recommended_sampling_and_keeps_caller_controls() {
    let profiles = Profiles::parse(
        serde_json::from_str(include_str!("../../config/proxy/model-profiles.json")).unwrap(),
    )
    .unwrap();
    for kind in ["swift", "swift_128k"] {
        let payload = json!({"model":MODEL,"messages":[],"reasoning_effort":"low","max_tokens":1234,"temperature":0.4});
        let effective = profiles.apply(&payload, kind, Some(MODEL));
        assert_eq!(effective["top_k"], 20);
        assert_eq!(effective["top_p"], 0.95);
        assert_eq!(effective["temperature"], 0.4);
        assert_eq!(effective["reasoning_effort"], "low");
        assert_eq!(effective["max_tokens"], 1234);
        let non_thinking = profiles.apply(&json!({"reasoning_effort":"none"}), kind, Some(MODEL));
        assert_eq!(non_thinking["temperature"], 0.7);
        assert_eq!(non_thinking["top_p"], 0.8);
        assert_eq!(non_thinking["presence_penalty"], 1.5);
    }
}

#[tokio::test]
async fn swift_catalog_is_passive_and_switching_confirms_unload_before_load() {
    let events = Arc::new(Mutex::new(vec![]));
    let (base_url, base, base_task) = server("sc117", events.clone()).await;
    let (swift_url, swift, swift_task) = server("swift_128k", events.clone()).await;
    let cfg = Config {
        llm_backend: "all".into(),
        sc117_url: base_url,
        swift_128k_url: swift_url,
        ..Config::default()
    };
    let mut backends = Backends::new(&cfg, reqwest::Client::new());
    backends
        .entries
        .retain(|b| ["sc117", "swift_128k"].contains(&b.kind.as_str()));
    let catalog = backends.catalog(None).await.unwrap();
    assert!(
        catalog["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == format!("swift_128k/{MODEL}"))
    );
    assert!(events.lock().unwrap().is_empty());
    let target = backends
        .target(Some(&format!("sc117/{MODEL}")))
        .await
        .unwrap();
    backends.acquire(Some(target)).await.unwrap();
    assert!(base.loaded.load(Ordering::SeqCst));
    let target = backends
        .target(Some(&format!("swift_128k/{MODEL}")))
        .await
        .unwrap();
    backends.acquire(Some(target)).await.unwrap();
    assert_eq!(
        *events.lock().unwrap(),
        ["sc117:load", "sc117:unload", "swift_128k:load"]
    );
    assert!(!base.loaded.load(Ordering::SeqCst));
    assert!(swift.loaded.load(Ordering::SeqCst));
    base_task.abort();
    swift_task.abort();
}

#[tokio::test]
async fn failed_swift_unload_blocks_gpu_handoff() {
    let events = Arc::new(Mutex::new(vec![]));
    let (base_url, _, base_task) = server("sc117", events.clone()).await;
    let (swift_url, swift, swift_task) = server("swift", events.clone()).await;
    let cfg = Config {
        llm_backend: "all".into(),
        sc117_url: base_url,
        swift_url,
        ..Config::default()
    };
    let mut backends = Backends::new(&cfg, reqwest::Client::new());
    backends
        .entries
        .retain(|b| ["sc117", "swift"].contains(&b.kind.as_str()));
    let target = backends
        .target(Some(&format!("swift/{MODEL}")))
        .await
        .unwrap();
    backends.acquire(Some(target)).await.unwrap();
    swift.reject_unload.store(true, Ordering::SeqCst);
    let target = backends
        .target(Some(&format!("sc117/{MODEL}")))
        .await
        .unwrap();
    assert!(backends.acquire(Some(target)).await.is_err());
    assert_eq!(*events.lock().unwrap(), ["swift:load", "swift:unload"]);
    base_task.abort();
    swift_task.abort();
}
