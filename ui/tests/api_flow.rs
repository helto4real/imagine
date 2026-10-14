use imagine_server::{
    api::{self, App},
    config::Access,
    contract::JobRequest,
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action},
    storage::Store,
};
use imagine_ui::{
    client::{Client, Progress, Submission, make_request},
    settings::{Connection, GenerationSettings},
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Cursor,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::sync::{Semaphore, watch};

struct Fake {
    state: watch::Sender<EngineState>,
    calls: Arc<AtomicUsize>,
    requests: Arc<Mutex<Vec<JobRequest>>>,
    block_folder: Arc<Mutex<Option<PathBuf>>>,
}
fn png() -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    image::RgbImage::from_pixel(12, 8, image::Rgb([122, 94, 70]))
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();
    buf.into_inner()
}
impl Backend for Fake {
    fn begin_job(&mut self, id: &str) {
        self.state.send_replace(EngineState {
            job_id: Some(id.into()),
            phase: "inference".into(),
            ..Default::default()
        });
    }
    fn generate<'a>(
        &'a mut self,
        req: &'a JobRequest,
        _: &'a str,
        _: &'a Store,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.requests.lock().unwrap().push(req.clone());
            self.state.send_modify(|state| {
                state.phase = "sampling".into();
                state.progress = Some(imagine_server::progress::SamplingProgress {
                    completed: 6,
                    total: req.sampling.steps,
                    elapsed_seconds: 6.,
                    remaining_seconds: (req.sampling.steps - 6) as f64,
                    updated_at_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                });
            });
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
            if let Some(folder) = self.block_folder.lock().unwrap().take() {
                std::fs::remove_dir(&folder)?;
                std::fs::write(folder, b"blocked")?;
            }
            Ok(png())
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
#[tokio::test]
async fn client_uses_real_api_defaults_references_retry_and_output_folder() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path()).await.unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let blocker = Arc::new(Mutex::new(None));
    let (engine_tx, engine) = watch::channel(EngineState::default());
    let (control, scheduler) = scheduler::start(
        store.clone(),
        Box::new(Fake {
            state: engine_tx,
            calls: calls.clone(),
            requests: requests.clone(),
            block_folder: blocker.clone(),
        }),
    )
    .await
    .unwrap();
    let access = Access {
        admin_token: "unused-admin-token".into(),
        clients: BTreeMap::from([("ui-test".into(), "synthetic-test-token".into())]),
    };
    let access_path = dir.path().join("access.json");
    std::fs::write(&access_path, serde_json::to_vec(&access).unwrap()).unwrap();
    let app = App {
        store,
        control: control.clone(),
        engine,
        access: Arc::new(access),
        uploads: Arc::new(Semaphore::new(2)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server =
        tokio::spawn(async move { axum::serve(listener, api::router(app)).await.unwrap() });
    let client = Client::connect(&Connection {
        url,
        access: access_path,
        client: "ui-test".into(),
    })
    .unwrap();
    let meta = client.metadata().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let ref_path = dir.path().join("reference.png");
    std::fs::write(&ref_path, png()).unwrap();
    let a = client.upload(&ref_path).await.unwrap();
    let b = client.upload(&ref_path).await.unwrap();
    let refs = vec![b.id, a.id];
    let settings = GenerationSettings {
        steps: 24,
        cfg_scale: 3.25,
        random_seed: false,
        seed: u32::MAX,
        aspect_ratio: "3:4".into(),
        max_side: 1024,
    };
    let request = make_request("  Exact prompt  ", "negative", &refs, &meta, &settings).unwrap();
    assert_eq!(request["operation"], "image_edit");
    assert_eq!(request["inputs"]["reference_images"], json!(refs));
    assert_eq!(request["sampling"]["steps"], 24);
    assert_eq!(
        request["model_options"]["qwen_image_2_1"]["reference_max_pixels_per_image"],
        1_000_000
    );
    let submission = Submission {
        request,
        key: "same-submission".into(),
        folder: dir.path().join("images"),
        job: None,
    };
    let mut id = None;
    let mut observed_progress = Vec::new();
    let first = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(value) = p {
                id = Some(value)
            } else if let Progress::Status {
                sampling: Some(sampling),
                ..
            } = p
            {
                observed_progress.push(sampling);
            }
        })
        .await
        .unwrap();
    assert_eq!(std::fs::read(&first.file).unwrap(), png());
    assert_eq!(first.file.parent().unwrap(), submission.folder);
    assert_eq!(requests.lock().unwrap()[0].prompt, "  Exact prompt  ");
    assert_eq!(first.seed, Some(u32::MAX as u64));
    assert!(!observed_progress.is_empty());
    assert!(observed_progress.iter().all(|p| p.completed == 6
        && p.total == 24
        && p.remaining_seconds == 18.
        && p.finish_time().is_some()));
    {
        let requests = requests.lock().unwrap();
        let received = &requests[0];
        assert_eq!((received.width, received.height), (768, 1024));
        assert_eq!(received.sampling.steps, 24);
        assert_eq!(received.sampling.cfg_scale, 3.25);
    }
    let duplicate = client.generate(submission, |_| {}).await.unwrap();
    assert_ne!(first.file, duplicate.file);
    assert_eq!(duplicate.seed, first.seed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // Disk failure after generation: changing output folder retries retrieval, not inference.
    let blocked = dir.path().join("blocked-output");
    *blocker.lock().unwrap() = Some(blocked.clone());
    let mut submission = Submission {
        request: make_request("another", "", &[], &meta, &GenerationSettings::default()).unwrap(),
        key: "disk-error".into(),
        folder: blocked,
        job: None,
    };
    let mut job = None;
    let err = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(id) = p {
                job = Some(id)
            }
        })
        .await
        .err()
        .unwrap();
    assert!(err.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    submission.job = job;
    submission.folder = dir.path().join("recovered-output");
    let recovered = client.generate(submission, |_| {}).await.unwrap();
    assert!(recovered.file.is_file());
    assert!(recovered.seed.is_some());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let cancelled = client.cancel(id.as_ref().unwrap()).await.err().unwrap();
    assert!(!cancelled.retryable);
    assert!(make_request(" ", "", &[], &meta, &GenerationSettings::default()).is_err());
    assert!(
        make_request(
            "x",
            "",
            &vec!["ref".into(); 11],
            &meta,
            &GenerationSettings::default()
        )
        .is_err()
    );
    control.call(Action::Shutdown).await.unwrap();
    scheduler.await.unwrap().unwrap();
    server.abort();
}
