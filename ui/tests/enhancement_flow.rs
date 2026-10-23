//! Owned local HTTP fixtures only; no real proxy, inference engine, or user data.
use axum::{
    Json, Router,
    routing::{get, post},
};
use imagine_ui::{
    client::Client,
    enhancement_types::{ModelsRequest, Profile, ProfileUpdate, Request},
    settings::Connection,
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

#[tokio::test]
async fn saved_profiles_round_trip_through_client_and_enhance_with_edited_system_prompt() {
    let root = tempfile::tempdir().unwrap();
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let captured_request = captured.clone();
    let provider = Router::new().route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
        let captured = captured_request.clone();
        async move {
            captured.lock().unwrap().push(body);
            Json(json!({"choices":[{"finish_reason":"stop","message":{"content":"SYNTHETIC improved text"}}]}))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let provider_task = tokio::spawn(async move {
        axum::serve(listener, provider).await.unwrap();
    });
    let path = root.path().join("enhancement-profiles.json");
    let service = imagine_server::enhancement::Service::open(path.clone()).unwrap();
    let router = imagine_server::enhancement::routes(service.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::connect(&Connection {
        url: format!("http://{}", listener.local_addr().unwrap()),
    })
    .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let profile = Profile {
        base_url: provider_url,
        model: "synthetic-model".into(),
        parameters: json!({"top_p":0.9,"top_k":32,"min_p":0.05,"typical_p":0.95,
            "repetition_penalty":1.1,"frequency_penalty":0.2,"presence_penalty":-0.1,
            "seed":42,"stop":["<end>"],"repeat_last_n":64})
        .as_object()
        .unwrap()
        .clone(),
        system_prompt: "SYNTHETIC edited system prompt\nPreserve åäö and reference order.\n"
            .repeat(4096),
        ..Default::default()
    };
    assert!(profile.system_prompt.len() > 65536);
    let profiles = client
        .save_enhancement_profile(ProfileUpdate {
            profile: profile.clone(),
        })
        .await
        .unwrap();
    assert!(
        serde_json::to_value(&profiles).unwrap()[0]
            .get("has_api_key")
            .is_none()
    );
    assert_eq!(client.enhancement_profiles().await.unwrap(), profiles);
    let id = uuid::Uuid::new_v4().to_string();
    let prompt = "SYNTHETIC user text\nReference 1";
    client
        .start_enhancement(&Request {
            id: id.clone(),
            profile_id: profile.id.clone(),
            prompt: prompt.into(),
        })
        .await
        .unwrap();
    let text = client.wait_enhancement(&id).await.unwrap();
    assert_eq!(text, "SYNTHETIC improved text");
    for (name, value) in &profile.parameters {
        assert_eq!(&captured.lock().unwrap()[0][name], value);
    }
    assert_eq!(
        captured.lock().unwrap()[0]["messages"],
        json!([{"role":"system","content":profile.system_prompt},{"role":"user","content":prompt}])
    );
    client.remove_enhancement(&id).await;
    assert_eq!(
        service.status(&id).unwrap_err().code,
        "enhancement_not_found"
    );
    let disk = std::fs::read_to_string(&path).unwrap();
    assert!(!disk.contains(prompt));
    assert!(!disk.contains("SYNTHETIC improved text"));
    let restored = imagine_server::enhancement::Service::open(path).unwrap();
    assert_eq!(
        serde_json::to_value(restored.profiles()).unwrap(),
        serde_json::to_value(profiles).unwrap()
    );
    assert!(
        client
            .delete_enhancement_profile(&profile.id)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(client.enhancement_profiles().await.unwrap().is_empty());
    server.abort();
    provider_task.abort();
}

#[tokio::test]
async fn model_lookup_uses_draft_connection_without_provider_authorization() {
    use axum::http::HeaderMap;
    let root = tempfile::tempdir().unwrap();
    let keys = Arc::new(Mutex::new(Vec::new()));
    let received = keys.clone();
    let provider = Router::new()
        .route("/v1/models", get(move |headers: HeaderMap| {
            let received = received.clone();
            async move {
                received.lock().unwrap().push(headers.get("authorization").map(|h| h.to_str().unwrap().to_owned()));
                Json(json!({"data":[{"id":"synthetic-z"},{"id":"synthetic-a"},{"id":"synthetic-a"}]}))
            }
        }))
        .route("/empty/models", get(|| async { Json(json!({"data":[]})) }))
        .route("/invalid/models", get(|| async { Json(json!({"data":[{"bad":"id"}]})) }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let provider_task = tokio::spawn(async move {
        axum::serve(listener, provider).await.unwrap();
    });
    let service =
        imagine_server::enhancement::Service::open(root.path().join("profiles.json")).unwrap();
    service
        .save(imagine_server::enhancement_types::ProfileUpdate {
            profile: imagine_server::enhancement_types::Profile {
                base_url: format!("{base}/v1"),
                ..Default::default()
            },
        })
        .unwrap();
    let before = std::fs::read(root.path().join("profiles.json")).unwrap();
    let router = imagine_server::enhancement::routes(service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::connect(&Connection {
        url: format!("http://{}", listener.local_addr().unwrap()),
    })
    .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mut lookup = ModelsRequest {
        profile_id: "proxy".into(),
        base_url: format!("{base}/v1/"),
    };
    assert_eq!(
        client.enhancement_models(&lookup).await.unwrap(),
        vec!["synthetic-a", "synthetic-z"]
    );
    // Old wire clients may still send credentials; the new wire discards them.
    let legacy: ModelsRequest = serde_json::from_value(json!({"profile_id":"proxy",
        "base_url":lookup.base_url,"api_key":"synthetic-draft-key"}))
    .unwrap();
    client.enhancement_models(&legacy).await.unwrap();
    lookup.base_url = lookup.base_url.replace("127.0.0.1", "localhost");
    client.enhancement_models(&lookup).await.unwrap();
    assert_eq!(*keys.lock().unwrap(), vec![None, None, None]);
    lookup.base_url = format!("{base}/empty");
    assert!(client.enhancement_models(&lookup).await.unwrap().is_empty());
    lookup.base_url = format!("{base}/missing");
    assert!(
        client
            .enhancement_models(&lookup)
            .await
            .unwrap_err()
            .message
            .contains("no /models endpoint")
    );
    lookup.base_url = format!("{base}/invalid");
    assert!(
        client
            .enhancement_models(&lookup)
            .await
            .unwrap_err()
            .message
            .contains("invalid model list")
    );
    lookup.base_url = format!("{base}/v1?bad=query");
    assert!(
        client
            .enhancement_models(&lookup)
            .await
            .unwrap_err()
            .message
            .contains("base URL")
    );
    assert_eq!(
        std::fs::read(root.path().join("profiles.json")).unwrap(),
        before
    );
    server.abort();
    provider_task.abort();
}

#[tokio::test]
async fn client_reports_invalid_system_prompt_and_cancels_an_owned_operation() {
    let root = tempfile::tempdir().unwrap();
    let entered = Arc::new(Notify::new());
    let signal = entered.clone();
    let provider = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let signal = signal.clone();
            async move {
                signal.notify_one();
                std::future::pending::<Json<Value>>().await
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let provider_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let provider_task = tokio::spawn(async move {
        axum::serve(listener, provider).await.unwrap();
    });
    let service =
        imagine_server::enhancement::Service::open(root.path().join("profiles.json")).unwrap();
    let router = imagine_server::enhancement::routes(service.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let client = Client::connect(&Connection {
        url: format!("http://{}", listener.local_addr().unwrap()),
    })
    .unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mut profile = Profile {
        base_url: provider_url,
        ..Default::default()
    };
    profile.system_prompt.clear();
    let error = client
        .save_enhancement_profile(ProfileUpdate {
            profile: profile.clone(),
        })
        .await
        .unwrap_err();
    assert!(error.message.contains("system prompt"));
    profile.system_prompt = "SYNTHETIC system instructions".into();
    client
        .save_enhancement_profile(ProfileUpdate { profile })
        .await
        .unwrap();
    let id = uuid::Uuid::new_v4().to_string();
    client
        .start_enhancement(&Request {
            id: id.clone(),
            profile_id: "proxy".into(),
            prompt: "SYNTHETIC text".into(),
        })
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), entered.notified())
        .await
        .unwrap();
    client.remove_enhancement(&id).await;
    assert_eq!(
        service.status(&id).unwrap_err().code,
        "enhancement_not_found"
    );
    let error = client.wait_enhancement(&id).await.unwrap_err();
    assert!(error.message.contains("expired"));
    server.abort();
    provider_task.abort();
}
