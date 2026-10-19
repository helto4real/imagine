use super::*;
use axum::{
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use tokio::sync::{Notify, mpsc};

#[test]
fn profiles_require_system_prompts_and_keep_keys_out_of_public_data() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("enhancement.json");
    let service = Service::open(path.clone()).unwrap();
    let mut profile = Profile::default();
    assert_eq!(profile.base_url, "http://127.0.0.1:5002/v1");
    profile.system_prompt = "  \n ".into();
    assert_eq!(
        service
            .save(ProfileUpdate {
                profile: profile.clone(),
                api_key: None
            })
            .unwrap_err()
            .code,
        "invalid_enhancement_profile"
    );
    profile.system_prompt = "SYNTHETIC editable system instructions\nKeep åäö.".into();
    let saved = service
        .save(ProfileUpdate {
            profile: profile.clone(),
            api_key: Some("synthetic-secret".into()),
        })
        .unwrap();
    assert!(saved[0].has_api_key);
    assert!(
        !serde_json::to_string(&saved)
            .unwrap()
            .contains("synthetic-secret")
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(Service::open(path.clone()).unwrap().profiles(), saved);
    profile.system_prompt = "SYNTHETIC revised system instructions".into();
    service
        .save(ProfileUpdate {
            profile: profile.clone(),
            api_key: None,
        })
        .unwrap();
    assert_eq!(
        service.state.lock().unwrap().saved.secrets[&profile.id],
        "synthetic-secret"
    );
    service
        .save(ProfileUpdate {
            profile: profile.clone(),
            api_key: Some(String::new()),
        })
        .unwrap();
    assert!(!service.profiles()[0].has_api_key);
    service.delete(&profile.id).unwrap();
    assert!(Service::open(path).unwrap().profiles().is_empty());
}

#[test]
fn failed_saves_preserve_the_last_profile_and_key() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("blocked");
    let service = Service::open(path.clone()).unwrap();
    std::fs::create_dir(&path).unwrap();
    let before = service.profiles();
    let profile = Profile {
        name: "Changed".into(),
        ..Default::default()
    };
    assert_eq!(
        service
            .save(ProfileUpdate {
                profile,
                api_key: Some("synthetic-key".into())
            })
            .unwrap_err()
            .code,
        "enhancement_save_failed"
    );
    assert_eq!(service.profiles(), before);
    assert!(service.state.lock().unwrap().saved.secrets.is_empty());
}

#[test]
fn sampling_parameters_migrate_and_validate_without_overriding_operation_fields() {
    let mut old = serde_json::to_value(Profile::default()).unwrap();
    old.as_object_mut().unwrap().remove("parameters");
    let mut profile: Profile = serde_json::from_value(old).unwrap();
    assert!(profile.parameters.is_empty());
    for parameters in [
        json!({"model":"wrong"}),
        json!({"messages":[]}),
        json!({"stream":true}),
        json!({"n":2}),
        json!({"max_tokens":3}),
        json!({"max_completion_tokens":4}),
        json!({"temperature":1}),
        json!({"top_p":1.1}),
        json!({"min_p":-0.1}),
        json!({"top_k":1.5}),
        json!({"seed":"bad"}),
        json!({"stop":[4]}),
    ] {
        profile.parameters = parameters.as_object().unwrap().clone();
        assert!(validate(&profile).is_err(), "{parameters}");
    }
    profile.parameters = json!({"top_p":0.8,"top_k":0,"seed":-1,"stop":["a","b"],
        "chat_template_kwargs":{"enable_thinking":false}})
    .as_object()
    .unwrap()
    .clone();
    validate(&profile).unwrap();
}

struct Provider {
    base: String,
    requests: mpsc::UnboundedReceiver<(HeaderMap, Value)>,
    release: Arc<Notify>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Provider {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn provider() -> Provider {
    let (sender, requests) = mpsc::unbounded_channel();
    let release = Arc::new(Notify::new());
    let signal = release.clone();
    let app = Router::new().route("/v1/chat/completions", post(move |headers: HeaderMap, Json(body): Json<Value>| {
        let sender = sender.clone(); let release = signal.clone();
        async move {
            let _ = sender.send((headers, body.clone()));
            let model = body["model"].as_str().unwrap();
            if model == "slow" { release.notified().await; }
            if model == "unauthorized" { return (StatusCode::UNAUTHORIZED, "SYNTHETIC private provider diagnostics").into_response(); }
            let (finish_reason, content) = match model {
                "truncated" => ("length", "Partial prompt"),
                "filtered" => ("content_filter", ""),
                "empty" => ("stop", "  \n"),
                "tools" => ("tool_calls", ""),
                _ => ("stop", "SYNTHETIC enhanced prompt åäö"),
            };
            Json(json!({"choices":[{"finish_reason":finish_reason,"message":{"role":"assistant","content":content,"reasoning_content":"SYNTHETIC hidden reasoning"}}]})).into_response()
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Provider {
        base,
        requests,
        release,
        task,
    }
}
fn submit(service: &Arc<Service>, prompt: &str) -> String {
    let id = uuid::Uuid::new_v4().to_string();
    service
        .start(Request {
            id: id.clone(),
            profile_id: "proxy".into(),
            prompt: prompt.into(),
        })
        .unwrap();
    id
}
async fn result(service: &Service, id: &str) -> Status {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            let status = service.status(id).unwrap();
            if !matches!(status, Status::Running) {
                return status;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}

#[tokio::test]
async fn chat_completion_freezes_profile_and_separates_system_prompt_from_user_text() {
    let mut provider = provider().await;
    let root = tempfile::tempdir().unwrap();
    let service = Service::open(root.path().join("profiles.json")).unwrap();
    let mut profile = Profile {
        base_url: provider.base.clone(),
        model: "slow".into(),
        system_prompt: "SYNTHETIC system before edit".into(),
        token_parameter: TokenParameter::MaxCompletionTokens,
        temperature: Some(0.4),
        ..Default::default()
    };
    service
        .save(ProfileUpdate {
            profile: profile.clone(),
            api_key: Some("synthetic-key".into()),
        })
        .unwrap();
    let prompt = "SYNTHETIC original prompt\nReference 1: åäö";
    let id = submit(&service, prompt);
    let (headers, body) = provider.requests.recv().await.unwrap();
    profile.system_prompt = "SYNTHETIC system after edit".into();
    service
        .save(ProfileUpdate {
            profile: profile.clone(),
            api_key: None,
        })
        .unwrap();
    assert_eq!(headers["authorization"], "Bearer synthetic-key");
    assert_eq!(
        body["messages"],
        json!([{"role":"system","content":"SYNTHETIC system before edit"},{"role":"user","content":prompt}])
    );
    assert_eq!(body["max_completion_tokens"], 1024);
    assert!(body.get("max_tokens").is_none());
    assert_eq!(body["temperature"], json!(0.4_f32));
    assert_eq!(body["stream"], false);
    provider.release.notify_one();
    assert!(
        matches!(result(&service, &id).await, Status::Succeeded { text } if text == "SYNTHETIC enhanced prompt åäö")
    );
    service.remove(&id);
    assert_eq!(
        service.status(&id).unwrap_err().code,
        "enhancement_not_found"
    );
    profile.model = "normal".into();
    profile.temperature = None;
    profile.token_parameter = TokenParameter::MaxTokens;
    service
        .save(ProfileUpdate {
            profile,
            api_key: Some(String::new()),
        })
        .unwrap();
    let id = submit(&service, prompt);
    let (headers, body) = provider.requests.recv().await.unwrap();
    assert!(headers.get("authorization").is_none());
    assert_eq!(
        body["messages"][0]["content"],
        "SYNTHETIC system after edit"
    );
    assert_eq!(body["max_tokens"], 1024);
    assert!(body.get("temperature").is_none());
    assert!(matches!(
        result(&service, &id).await,
        Status::Succeeded { .. }
    ));
}

#[tokio::test]
async fn provider_errors_and_partial_or_reasoning_only_responses_never_become_prompt_text() {
    let provider = provider().await;
    let root = tempfile::tempdir().unwrap();
    let service = Service::open(root.path().join("profiles.json")).unwrap();
    for (model, expected) in [
        ("truncated", "enhancement_truncated"),
        ("filtered", "enhancement_refused"),
        ("empty", "enhancement_empty_response"),
        ("tools", "enhancement_invalid_response"),
        ("unauthorized", "enhancement_auth_failed"),
    ] {
        service
            .save(ProfileUpdate {
                profile: Profile {
                    base_url: provider.base.clone(),
                    model: model.into(),
                    ..Default::default()
                },
                api_key: None,
            })
            .unwrap();
        let id = submit(&service, "SYNTHETIC user prompt");
        let status = result(&service, &id).await;
        assert!(matches!(&status, Status::Failed { code } if code == expected));
        assert!(
            !serde_json::to_string(&status)
                .unwrap()
                .contains("private provider")
        );
        service.remove(&id);
    }
}

#[tokio::test]
async fn cancelled_requests_release_slots_and_ignore_late_results() {
    let mut provider = provider().await;
    let root = tempfile::tempdir().unwrap();
    let service = Service::open(root.path().join("profiles.json")).unwrap();
    service
        .save(ProfileUpdate {
            profile: Profile {
                base_url: provider.base.clone(),
                model: "slow".into(),
                ..Default::default()
            },
            api_key: None,
        })
        .unwrap();
    let id = submit(&service, "SYNTHETIC prompt");
    provider.requests.recv().await.unwrap();
    service.remove(&id);
    tokio::time::timeout(Duration::from_secs(2), async {
        while service.slots.available_permits() != 4 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    provider.release.notify_one();
    assert_eq!(
        service.status(&id).unwrap_err().code,
        "enhancement_not_found"
    );
}
