//! Real child/pipe lifecycle, with stdlib synthetic inference only.
//! A separate test process keeps forks from briefly inheriting unit-test file locks.
use imagine_server::{
    comfy_engine::Comfy,
    config::Config,
    contract::{self, ModelOptions, Sampling},
    engine::{Backend, EngineError, EngineState},
    generation::ImagePass,
};
use std::{path::Path, time::Duration};
use tokio::sync::watch;

fn pass(prompt: &str) -> ImagePass {
    ImagePass {
        resolved_models: None,
        model: contract::MODEL.into(),
        model_options: ModelOptions::default(),
        sampling: Sampling::default(),
        prompt: prompt.into(),
        negative_prompt: String::new(),
        width: 64,
        height: 32,
        seed: 42,
        references: vec![],
    }
}
fn fake() -> (tempfile::TempDir, Comfy, watch::Receiver<EngineState>) {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("source");
    std::fs::create_dir_all(root.join("comfy")).unwrap();
    std::fs::write(root.join("comfy/sd.py"), b"synthetic source marker").unwrap();
    let config = Config {
        project: directory.path().into(),
        data: directory.path().join("data"),
        bind: "127.0.0.1:0".parse().unwrap(),
        model_root: directory.path().join("models"),
    };
    let (sender, receiver) = watch::channel(EngineState {
        backend: "comfy".into(),
        ..Default::default()
    });
    let script = directory.path().join("workers/comfy/worker.py");
    std::fs::create_dir_all(script.parent().unwrap()).unwrap();
    let fake_script = Path::new(env!("CARGO_MANIFEST_DIR")).join("workers/comfy/fake_worker.py");
    std::fs::write(
        &script,
        format!(
            "import runpy\nrunpy.run_path({}, run_name=\"__main__\")\n",
            serde_json::to_string(&fake_script).unwrap()
        ),
    )
    .unwrap();
    let engine = Comfy::new(config, sender, "/usr/bin/python3".into(), root).unwrap();
    (directory, engine, receiver)
}
#[tokio::test]
async fn default_rust_sampling_runs_through_python_adapter_without_manual_sigmas() {
    let (_directory, mut engine, _) = fake();
    let result = engine.generate(&pass("adapter")).await;
    engine.stop().await.unwrap();
    assert!(
        result.is_ok(),
        "default sampling must reach the adapter's output: {result:?}"
    );
}
#[tokio::test]
async fn turbo_rust_profile_accepts_comfy_conditioning_without_pooled_tensor() {
    let (_directory, mut engine, _) = fake();
    let mut request = pass("adapter");
    let profile: serde_json::Value =
        serde_yaml::from_str(include_str!("../profiles/quick-draft.yaml")).unwrap();
    request.model = contract::TURBO.into();
    request.sampling = serde_json::from_value(profile["defaults"]["sampling"].clone()).unwrap();
    request.model_options =
        serde_json::from_value(profile["defaults"]["model_options"].clone()).unwrap();
    assert!(
        request
            .model_options
            .qwen_image_2_1
            .workflow
            .as_ref()
            .unwrap()
            .zero_negative_conditioning
    );
    let result = engine.generate(&request).await;
    engine.stop().await.unwrap();
    assert!(
        result.is_ok(),
        "Turbo profile must reach output with pooled_output=None: {result:?}"
    );
}
#[tokio::test]
async fn persistent_child_reuses_pid_and_restarts_after_confirmed_stop() {
    let (_directory, mut engine, state) = fake();
    assert!(state.borrow().pid.is_none()); // Startup and construction have no GPU/process side effects.
    engine.begin_job("first");
    let output = engine.generate(&pass("success")).await.unwrap();
    assert_eq!(image::load_from_memory(&output).unwrap().width(), 64);
    let first = state.borrow().pid.unwrap();
    assert_eq!(state.borrow().progress.as_ref().unwrap().completed, 40);
    engine.mark_idle();
    assert_eq!(state.borrow().phase, "ready");
    engine.begin_job("second");
    engine.generate(&pass("success")).await.unwrap();
    assert_eq!(state.borrow().pid, Some(first));
    engine.stop().await.unwrap();
    assert!(!Path::new(&format!("/proc/{first}")).exists());
    assert_eq!(state.borrow().phase, "unloaded");
    engine.generate(&pass("success")).await.unwrap();
    assert_ne!(state.borrow().pid, Some(first));
    engine.stop().await.unwrap();
}
#[tokio::test]
async fn cancellation_drops_generation_then_stop_reaps_before_returning() {
    let (_directory, mut engine, mut state) = fake();
    let request = pass("hang");
    {
        let generation = engine.generate(&request);
        tokio::pin!(generation);
        let started = async {
            loop {
                state.changed().await.unwrap();
                if state.borrow().phase == "inference" {
                    break;
                }
            }
        };
        tokio::select! {
            result = &mut generation => panic!("unexpected completion: {result:?}"),
            result = tokio::time::timeout(Duration::from_secs(5),started) => result.unwrap(),
        }
    }
    let pid = state.borrow().pid.unwrap();
    tokio::time::timeout(Duration::from_secs(5), engine.stop())
        .await
        .unwrap()
        .unwrap();
    assert!(!Path::new(&format!("/proc/{pid}")).exists());
    engine.generate(&pass("success")).await.unwrap();
    engine.stop().await.unwrap();
}
#[tokio::test]
async fn failures_are_safe_and_next_job_can_restart() {
    for (mode, code) in [
        ("failure", "comfy_generation_failed"),
        ("wrong_id", "comfy_protocol_error"),
        ("invalid_output", "comfy_invalid_output"),
        ("disconnect", "comfy_protocol_error"),
    ] {
        let (_directory, mut engine, _) = fake();
        let error = engine.generate(&pass(mode)).await.unwrap_err();
        assert_eq!(error.downcast_ref::<EngineError>().unwrap().0, code);
        assert!(!format!("{error:?}").contains("synthetic-private-sentinel"));
        engine.stop().await.unwrap();
        engine.generate(&pass("success")).await.unwrap();
        engine.stop().await.unwrap();
    }
}

#[tokio::test]
#[ignore = "requires explicit Comfy paths, model files and a GPU-coordinator ticket"]
async fn live_lora_through_rust_backend() {
    // A fixed source fixture only. No API job store, private inputs or saved outputs.
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    let directory = tempfile::tempdir().unwrap();
    let config = Config {
        project: env!("CARGO_MANIFEST_DIR").into(),
        data: directory.path().into(),
        bind: "127.0.0.1:0".parse().unwrap(),
        model_root: path("IMAGINE_TEST_MODEL_ROOT"),
    };
    let (sender, _) = watch::channel(EngineState::default());
    let mut engine = Comfy::new(
        config,
        sender,
        path("IMAGINE_TEST_COMFY_PYTHON"),
        path("IMAGINE_TEST_COMFY_ROOT"),
    )
    .unwrap();
    let mut request = pass("A blue cube on a plain gray background, studio photograph.");
    request.width = 1600;
    request.height = 896;
    let profile: serde_json::Value =
        serde_yaml::from_str(include_str!("../profiles/qwen.yaml")).unwrap();
    let mut workflow: contract::WorkflowOptions = serde_json::from_value(
        profile["defaults"]["model_options"]["qwen_image_2_1"]["generation"].clone(),
    )
    .unwrap();
    workflow.loras.push(contract::Lora {
        path: std::env::var("IMAGINE_TEST_LORA").expect("explicit LoRA relative to loras/"),
        strength: 1.0,
    });
    request.model_options.qwen_image_2_1.workflow = Some(workflow);
    let mut ticket = imagine_server::coordination::Ticket::register(
        &path("IMAGINE_TEST_COORDINATOR_SOCKET"),
        &format!("job_synthetic_{}", uuid::Uuid::new_v4().simple()),
        1,
    )
    .await
    .unwrap()
    .acquire()
    .await
    .unwrap();
    let result = tokio::select! {
        result = engine.generate(&request) => result,
        _ = ticket.lost() => Err(anyhow::anyhow!("GPU ticket lost")),
    };
    let stopped = engine.stop().await;
    if stopped.is_ok() {
        ticket.finish().await.unwrap();
    }
    stopped.unwrap();
    match result {
        Ok(bytes) => drop(bytes), // Do not inspect, hash or save the generated image.
        Err(error) => panic!("isolated Comfy LoRA generation failed: {error}"),
    }
}
