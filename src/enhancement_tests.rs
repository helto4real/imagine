use super::*;
use axum::{
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use tokio::sync::{Notify, mpsc};

#[test]
fn legacy_credentials_are_discarded_and_never_persisted_or_returned() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("enhancement.json");
    let mut legacy = serde_json::to_value(Profile::default()).unwrap();
    legacy["has_api_key"] = json!(true);
    legacy["api_key"] = json!("synthetic-profile-key");
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "profiles":[legacy], "secrets":{"proxy":"synthetic-saved-key"}
        }))
        .unwrap(),
    )
    .unwrap();
    let service = Service::open(path.clone()).unwrap();
    let public = serde_json::to_value(service.profiles()).unwrap();
    assert!(public[0].get("has_api_key").is_none());
    assert!(public[0].get("api_key").is_none());
    let mut profile = service.profiles()[0].clone();
    profile.system_prompt = "  \n ".into();
    assert_eq!(
        service
            .save(ProfileUpdate {
                profile: profile.clone()
            })
            .unwrap_err()
            .code,
        "invalid_enhancement_profile"
    );
    profile.system_prompt = "SYNTHETIC editable system instructions\nKeep åäö.".into();
    profile.story_transport = crate::enhancement_types::StoryTransport::Ollama;
    let update: ProfileUpdate = serde_json::from_value(json!({"profile":profile,
        "api_key":"synthetic-update-key"}))
    .unwrap();
    assert_eq!(
        serde_json::to_value(&update)
            .unwrap()
            .as_object()
            .unwrap()
            .len(),
        1
    );
    let saved = service.save(update).unwrap();
    let disk: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert!(disk.get("secrets").is_none());
    assert!(!disk.to_string().contains("synthetic-"));
    assert!(disk["profiles"][0].get("has_api_key").is_none());
    assert!(disk["profiles"][0].get("api_key").is_none());
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(Service::open(path.clone()).unwrap().profiles(), saved);
    // Compatibility is narrow: unrelated misspellings still use the same wire validation.
    let mut invalid = serde_json::to_value(&profile).unwrap();
    invalid["model_typo"] = json!("unexpected");
    assert!(serde_json::from_value::<Profile>(invalid).is_err());
    let lookup: ModelsRequest = serde_json::from_value(json!({"profile_id":"proxy",
        "base_url":"http://127.0.0.1:5002/v1","api_key":{"ignored":"any old value"}}))
    .unwrap();
    assert!(
        serde_json::to_value(lookup)
            .unwrap()
            .get("api_key")
            .is_none()
    );
    service.delete(&profile.id).unwrap();
    assert!(Service::open(path).unwrap().profiles().is_empty());
}

#[test]
fn failed_saves_preserve_the_last_profile() {
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
        service.save(ProfileUpdate { profile }).unwrap_err().code,
        "enhancement_save_failed"
    );
    assert_eq!(service.profiles(), before);
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
async fn legacy_keys_cannot_reach_enhancement_or_story_provider_requests() {
    let mut provider = provider().await;
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("profiles.json");
    let profile = Profile {
        base_url: provider.base.clone(),
        model: "normal".into(),
        ..Default::default()
    };
    let mut legacy = serde_json::to_value(&profile).unwrap();
    legacy["has_api_key"] = json!(true);
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "profiles":[legacy],"secrets":{"proxy":"own-synthetic-never-transmitted-key"}
        }))
        .unwrap(),
    )
    .unwrap();
    let service = Service::open(path).unwrap();
    let snapshot = service.snapshot("proxy").unwrap();
    let parameters = json!({"temperature":0.5,"top_p":0.9,"top_k":40,
        "min_p":0.05,"repeat_penalty":1.1,"max_tokens":128,"reasoning":{}})
    .as_object()
    .unwrap()
    .clone();
    assert_eq!(
        service
            .authoring_complete(
                &snapshot,
                "koboldcpp",
                "SYNTHETIC own system template",
                "SYNTHETIC own user template",
                &parameters,
                &[]
            )
            .await
            .unwrap(),
        "SYNTHETIC enhanced prompt åäö"
    );
    let (headers, _) = provider.requests.recv().await.unwrap();
    assert!(headers.get("authorization").is_none());
    service
        .complete(snapshot.profile, "SYNTHETIC own direct template".into())
        .await
        .unwrap();
    let (headers, _) = provider.requests.recv().await.unwrap();
    assert!(headers.get("authorization").is_none());
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
        })
        .unwrap();
    let prompt = "SYNTHETIC original prompt\nReference 1: åäö";
    let id = submit(&service, prompt);
    let (headers, body) = provider.requests.recv().await.unwrap();
    profile.system_prompt = "SYNTHETIC system after edit".into();
    service
        .save(ProfileUpdate {
            profile: profile.clone(),
        })
        .unwrap();
    assert!(headers.get("authorization").is_none());
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
    service.save(ProfileUpdate { profile }).unwrap();
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

mod private_store {
    use super::*;
    use crate::{private_storage::Root, video_destination as d};
    use std::path::Path;

    struct Volume(Mutex<String>);
    impl d::Environment for Volume {
        fn mounts(&self) -> d::Result<String> {
            Ok(self.0.lock().unwrap().clone())
        }
        fn host(&self) -> d::Result<String> {
            Ok("own-provider-store-fixture".into())
        }
    }
    fn fixture() -> (tempfile::TempDir, Root, Arc<Volume>, String) {
        let own = crate::task_fixtures::env_root("TMPDIR");
        let temp = tempfile::Builder::new()
            .prefix("cp42provider-")
            .tempdir_in(own)
            .unwrap();
        let mount = temp.path().join("volume");
        let root = mount.join("imagine");
        std::fs::create_dir_all(&root).unwrap();
        let mounts = format!(
            "1 0 0:1 / {} rw - fixturefs own-provider-store rw\n",
            mount.display()
        );
        let env = Arc::new(Volume(Mutex::new(mounts.clone())));
        let config = d::Config {
            root,
            mount: mount.clone(),
            source: "own-provider-store".into(),
            filesystem: "fixturefs".into(),
            fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
        };
        (
            temp,
            Root::with_environment(config, env.clone()).unwrap(),
            env,
            mounts,
        )
    }
    const RELATIVE: &str = "library/enhancement-profiles.json";
    #[test]
    fn private_profiles_roundtrip_and_strip_legacy_credentials_without_fixture_fallback() {
        let (_temp, root, _env, _mounts) = fixture();
        let mut profile = serde_json::to_value(Profile::default()).unwrap();
        profile["has_api_key"] = json!(true);
        profile["api_key"] = json!("own-old-profile-key");
        root.write(
            Path::new(RELATIVE),
            &serde_json::to_vec(&json!({"profiles":[profile],
            "secrets":{"proxy":"own-old-saved-key"}}))
            .unwrap(),
        )
        .unwrap();
        let service = Service::open_private(root.clone(), RELATIVE.into()).unwrap();
        let mut selected = service.profiles()[0].clone();
        selected.system_prompt = "OWN PRIVATE PROVIDER INSTRUCTIONS".into();
        let profiles = service.save(ProfileUpdate { profile: selected }).unwrap();
        let bytes = root.read(Path::new(RELATIVE)).unwrap().unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(value.get("secrets").is_none());
        assert!(!String::from_utf8(bytes).unwrap().contains("own-old-"));
        assert_eq!(
            Service::open_private(root.clone(), RELATIVE.into())
                .unwrap()
                .profiles(),
            profiles
        );
        service.delete("proxy").unwrap();
        assert!(
            Service::open_private(root.clone(), RELATIVE.into())
                .unwrap()
                .profiles()
                .is_empty()
        );
        assert!(Service::open_private(root, "../outside.json".into()).is_err());
    }
    #[test]
    fn private_profile_save_and_delete_revalidate_volume_preserve_state_and_recover() {
        let (temp, root, env, mounts) = fixture();
        let service = Service::open_private(root.clone(), RELATIVE.into()).unwrap();
        let initial = Profile {
            name: "Own original profile".into(),
            ..Default::default()
        };
        service
            .save(ProfileUpdate {
                profile: initial.clone(),
            })
            .unwrap();
        let path = root.layout().root().join(RELATIVE);
        let disk = std::fs::read(&path).unwrap();
        let previous = service.profiles();
        let changed = Profile {
            name: "Own changed profile".into(),
            ..initial
        };
        for inaccessible in [String::new(), mounts.replace(" rw", " ro")] {
            *env.0.lock().unwrap() = inaccessible;
            assert_eq!(
                service
                    .save(ProfileUpdate {
                        profile: changed.clone()
                    })
                    .unwrap_err()
                    .code,
                "enhancement_save_failed"
            );
            assert_eq!(
                service.delete("proxy").unwrap_err().code,
                "enhancement_save_failed"
            );
            assert_eq!(service.profiles(), previous);
            // This known synthetic file is inspected directly only to establish
            // that the failed guarded save did not touch its previous bytes.
            assert_eq!(std::fs::read(&path).unwrap(), disk);
            assert!(Service::open_private(root.clone(), RELATIVE.into()).is_err());
            assert!(
                !temp
                    .path()
                    .join("library/enhancement-profiles.json")
                    .exists()
            );
        }
        *env.0.lock().unwrap() = mounts;
        let saved = service.save(ProfileUpdate { profile: changed }).unwrap();
        assert_eq!(
            Service::open_private(root, RELATIVE.into())
                .unwrap()
                .profiles(),
            saved
        );
        assert_ne!(std::fs::read(path).unwrap(), disk);
    }
}

#[test]
fn profiles_saved_without_a_timeout_use_the_default_and_out_of_range_values_are_rejected() {
    let mut saved = serde_json::to_value(Profile::default()).unwrap();
    saved.as_object_mut().unwrap().remove("timeout_seconds");
    let profile: Profile = serde_json::from_value(saved).unwrap();
    assert_eq!(profile.timeout_seconds, DEFAULT_TIMEOUT_SECONDS);
    assert_eq!(DEFAULT_TIMEOUT_SECONDS, 300);
    for timeout_seconds in [0, 3601] {
        assert!(
            validate(&Profile {
                timeout_seconds,
                ..Default::default()
            })
            .is_err()
        );
    }
}

#[tokio::test]
async fn a_request_fails_with_enhancement_timeout_after_the_profile_timeout() {
    let provider = provider().await;
    let root = tempfile::tempdir().unwrap();
    let service = Service::open(root.path().join("profiles.json")).unwrap();
    let profile = Profile {
        base_url: provider.base.clone(),
        model: "slow".into(),
        timeout_seconds: 1,
        ..Default::default()
    };
    service.save(ProfileUpdate { profile }).unwrap();
    let id = submit(&service, "SYNTHETIC prompt");
    assert!(
        matches!(result(&service, &id).await, Status::Failed { code } if code == "enhancement_timeout")
    );
}
