#[path = "support/destination.rs"]
mod destination;
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    generation::ImagePass,
    scheduler::{self, Action},
    storage::Store,
};
use imagine_ui::{
    client::{
        Client, EditSource, OutputRole, Progress, Submission, make_edit_request, make_request,
        make_sheet_request, make_upscale_request,
    },
    settings::{Connection, GenerationSettings, UpscaleTarget},
};
use serde_json::json;
use std::{
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
    requests: Arc<Mutex<Vec<ImagePass>>>,
    block_folder: Arc<Mutex<Option<PathBuf>>>,
}
fn png() -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    image::RgbImage::from_pixel(64, 64, image::Rgb([122, 94, 70]))
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
    fn stage(&mut self, stage: Option<&str>) {
        self.state.send_modify(|s| {
            s.stage = stage.map(str::to_owned);
            s.progress = None;
        });
    }
    fn generate<'a>(&'a mut self, req: &'a ImagePass) -> EngineFuture<'a, Vec<u8>> {
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
            if self.state.borrow().stage.is_some() || req.model == "qwen-image-2.1-turbo" {
                let mut buf = Cursor::new(Vec::new());
                image::RgbImage::from_pixel(req.width, req.height, image::Rgb([122, 94, 70]))
                    .write_to(&mut buf, image::ImageFormat::Png)?;
                Ok(buf.into_inner())
            } else {
                Ok(png())
            }
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
#[tokio::test]
async fn client_uses_proxy_api_defaults_references_retry_and_output_folder() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("images");
    let mut store = Store::open(dir.path()).await.unwrap();
    store.videos.destination = Some(destination::destination(&output));
    let calls = Arc::new(AtomicUsize::new(0));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let blocker = Arc::new(Mutex::new(None));
    let (engine_tx, engine) = watch::channel(EngineState::default());
    let socket = dir.path().join("gpu.sock");
    let (control, scheduler) = scheduler::start_coordinated(
        store.clone(),
        Box::new(Fake {
            state: engine_tx,
            calls: calls.clone(),
            requests: requests.clone(),
            block_folder: blocker.clone(),
        }),
        Some(socket.clone()),
    )
    .await
    .unwrap();

    // Keep the shipped upscale settings, with only synthetic plaintext prompts.
    let profiles = dir.path().join("profiles");
    std::fs::create_dir(&profiles).unwrap();
    let profile = include_str!("../../config/image/profiles/image_upscale.yaml")
        .lines()
        .map(|line| {
            if line.starts_with("  prompt:") {
                "  prompt: Synthetic upscale fixture"
            } else if line.starts_with("  negative_prompt:") {
                "  negative_prompt: ''"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(profiles.join("image_upscale.yaml"), profile).unwrap();
    let library = Arc::new(std::sync::Mutex::new(
        imagine_server::model_library::Library::default(),
    ));
    let app = App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(
            dir.path().join("enhancement-profiles.json"),
        )
        .unwrap(),
        library: library.clone(),
        profiles: Arc::new(imagine_server::profiles::Catalog::load(&profiles).unwrap()),
        store,
        control: control.clone(),
        engine,

        uploads: Arc::new(Semaphore::new(2)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server =
        tokio::spawn(async move { axum::serve(listener, api::router(app)).await.unwrap() });
    let backend_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let backend_url = format!("http://{}", backend_listener.local_addr().unwrap());
    let backend_server = tokio::spawn(async move {
        axum::serve(backend_listener, axum::Router::new()
            .route("/health", axum::routing::get(|| async { axum::Json(json!({"service":"strata","status":"ok","loaded":false,"model":"synthetic"})) }))
            .route("/v1/models", axum::routing::get(|| async { axum::Json(json!({"data":[{"id":"synthetic"}]})) }))
            .route("/free", axum::routing::post(|| async { axum::Json(json!({})) })))
            .await.unwrap();
    });
    let proxy = st_proxy::service::Service::start(st_proxy::config::Config {
        chat_port: 0,
        image_port: 0,
        llm_backend: "sc117".into(),
        llm_url: backend_url.clone(),
        comfy_url: backend_url,
        imagine_url: url,

        coordinator_socket: Some(socket.to_string_lossy().into()),
        poll_interval: 0.01,
        ..Default::default()
    })
    .await
    .unwrap();
    let url = format!("http://127.0.0.1:{}/imagine", proxy.image_port);
    let http = reqwest::Client::new();
    let catalog_response = http.get(format!("{url}/v1/models")).send().await.unwrap();
    assert_eq!(catalog_response.status(), 200);
    assert_eq!(catalog_response.headers()["cache-control"], "no-store");
    let client = Client::connect(&Connection { url: url.clone() }).unwrap();
    assert!(
        !client.enhancement_profiles().await.unwrap()[0]
            .system_prompt
            .is_empty()
    );
    let meta = client.metadata().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let ref_path = dir.path().join("reference.png");
    std::fs::write(&ref_path, png()).unwrap();
    let a = client.upload(&ref_path).await.unwrap();
    let b = client.upload(&ref_path).await.unwrap();
    let shared_asset = http
        .get(format!("{url}/v1/assets/{}", a.id))
        .send()
        .await
        .unwrap();
    assert_eq!(shared_asset.status(), 200);
    let refs = vec![b.id, a.id];
    let settings = GenerationSettings {
        sampler: None,
        scheduler: None,
        steps: 24,
        cfg_scale: 3.25,
        random_seed: false,
        seed: u32::MAX,
        aspect_ratio: "3:4".into(),
        max_side: 1024,
        reference_max_pixels_per_image: 1_000_000,
        ..Default::default()
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
        upscale_source: None,
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
    assert_eq!(std::fs::read(&first.assets[0].file).unwrap(), png());
    assert_eq!(
        first.assets[0].file.parent().unwrap().parent().unwrap(),
        submission.folder
    );
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
    let shared_job = http
        .get(format!("{url}/v1/jobs/{}", id.as_ref().unwrap()))
        .send()
        .await
        .unwrap();
    assert_eq!(shared_job.status(), 200);
    let events = http
        .get(format!("{url}/v1/jobs/{}/events", id.as_ref().unwrap()))
        .send()
        .await
        .unwrap();
    assert_eq!(events.headers()["content-type"], "text/event-stream");
    let events = events.text().await.unwrap();
    assert!(events.contains("succeeded"));
    let duplicate = client.generate(submission, |_| {}).await.unwrap();
    assert_eq!(first.assets[0].file, duplicate.assets[0].file);
    assert_eq!(duplicate.seed, first.seed);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    // A different UI destination fails before submission; restoring the configured root permits it.
    let blocked = dir.path().join("blocked-output");

    let mut submission = Submission {
        upscale_source: None,
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
    assert!(!err.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(job.is_none());
    submission.folder = output.clone();
    let recovered = client.generate(submission, |_| {}).await.unwrap();
    assert!(recovered.assets[0].file.is_file());
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
    // Complete sheet via the real proxy/API, using only a synthetic backend.
    let request = make_sheet_request(
        "Synthetic identity",
        "Green dress",
        512,
        "negative",
        &[],
        &meta,
        &settings,
    )
    .unwrap();
    let submission = Submission {
        upscale_source: None,
        request,
        key: "text-sheet".into(),
        folder: output.clone(),
        job: None,
    };
    let mut stages = Vec::new();
    let completed = client
        .generate(submission.clone(), |p| {
            if let Progress::Status { label, .. } = p {
                stages.push(label);
            }
        })
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert!(stages.iter().any(|s| s == "Portrait · Sampling"));
    assert!(stages.iter().any(|s| s == "Views · Sampling"));
    assert_eq!(
        completed.assets.iter().map(|a| a.role).collect::<Vec<_>>(),
        [OutputRole::Sheet, OutputRole::Portrait, OutputRole::Views]
    );
    for asset in &completed.assets {
        assert_eq!(
            asset.file.parent().unwrap().parent().unwrap(),
            submission.folder
        );
        assert!(
            asset
                .file
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with(&format!("_{}.png", asset.role.name()))
        );
        image::load_from_memory(&std::fs::read(&asset.file).unwrap()).unwrap();
    }
    assert_eq!(
        (
            completed.assets[0].preview.width,
            completed.assets[0].preview.height
        ),
        (512, 288)
    );
    assert!(completed.assets.iter().all(|a| a.file.exists()));
    // Reference sheet: destination mismatch is rejected before either inference pass.
    let blocked = dir.path().join("blocked-sheet");

    let mut submission = Submission {
        upscale_source: None,
        request: make_sheet_request("", "Blue dress", 512, "", &refs, &meta, &settings).unwrap(),
        key: "reference-sheet".into(),
        folder: blocked,
        job: None,
    };
    let mut job = None;
    let error = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(id) = p {
                job = Some(id)
            }
        })
        .await
        .err()
        .unwrap();
    assert!(!error.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    assert!(job.is_none());
    submission.folder = output.clone();
    let saved = client.generate(submission.clone(), |_| {}).await.unwrap();
    assert_eq!(saved.assets.len(), 3);
    assert!(saved.assets.iter().all(|a| a.file.exists()));
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    {
        let requests = requests.lock().unwrap();
        assert_eq!(requests[4].references.len(), 2);
        assert_eq!(requests[5].references.len(), 4);
    }
    // Cancellation returns a terminal result and saves no partial sheet.
    let submission = Submission {
        upscale_source: None,
        request: make_sheet_request("Cancel test", "", 512, "", &[], &meta, &settings).unwrap(),
        key: "cancel-sheet".into(),
        folder: output.clone(),
        job: None,
    };
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let generation = client.generate(submission.clone(), |p| {
        if let Progress::Submitted(id) = p {
            tx.send(id).unwrap()
        }
    });
    let cancel = async {
        let id = rx.recv().await.unwrap();
        client.cancel(&id).await.unwrap();
    };
    let (result, ()) = tokio::join!(generation, cancel);
    let error = result.err().unwrap();
    assert!(!error.retryable);
    assert_eq!(error.message, "Generation cancelled.");
    assert!(
        std::fs::read_dir(completed.assets[0].file.parent().unwrap())
            .unwrap()
            .all(|p| !p
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".partial"))
    );
    // Both UI workspaces use registered output assets and the same upscale profile.
    for (source, target, ratio, dimensions) in [
        (&first.assets[0], UpscaleTarget::K3, "3:4", (2304, 3072)),
        (
            &completed.assets[0],
            UpscaleTarget::K2,
            "16:9",
            (2048, 1152),
        ),
    ] {
        let source_bytes = std::fs::read(&source.file).unwrap();
        let before = calls.load(Ordering::SeqCst);
        let mut submission = Submission {
            upscale_source: Some(source.file.clone()),
            request: make_upscale_request(&source.asset_id, target, ratio, &settings).unwrap(),
            key: format!("upscale-{ratio}"),
            folder: output.clone(),
            job: None,
        };
        let mut job = None;
        let upscaled = client
            .generate(submission.clone(), |p| {
                if let Progress::Submitted(id) = p {
                    job = Some(id);
                }
            })
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        assert_eq!(submission.folder, output);
        submission.job = job;
        // A UI-only naming hint cannot redirect the server's already published result.
        submission.upscale_source = Some(dir.path().join("untrusted-hint/image.png"));
        let recovered = client.generate(submission, |_| {}).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), before + 1);
        let stem = source.file.file_stem().unwrap().to_str().unwrap();
        assert_eq!(
            upscaled.assets[0].file,
            source.file.with_file_name(format!("{stem}_upscaled.png"))
        );
        assert_eq!(
            recovered.assets[0].file,
            source.file.with_file_name(format!("{stem}_upscaled.png"))
        );
        assert_eq!(upscaled.assets.len(), 1);
        let output = &upscaled.assets[0];
        assert_ne!(output.asset_id, source.asset_id);
        assert_ne!(output.file, source.file);
        assert_eq!(std::fs::read(&source.file).unwrap(), source_bytes);
        let saved = image::load_from_memory(&std::fs::read(&output.file).unwrap()).unwrap();
        assert_eq!((saved.width(), saved.height()), dimensions);
        let requests = requests.lock().unwrap();
        let request = requests.last().unwrap();
        assert_eq!((request.width, request.height), dimensions);
        assert_eq!(request.model, "qwen-image-2.1-turbo");
        assert_eq!(request.sampling.steps, 8);
        assert_eq!(request.sampling.cfg_scale, 1.);
        assert_eq!(request.sampling.sampler, "euler");
        assert_eq!(request.sampling.scheduler, "simple");
        assert_eq!(request.references.len(), 1);
        assert_eq!(request.references[0].as_ref(), source_bytes);
    }
    // Editing a registered output with generic, sparsely numbered references.
    // Save recovery must reuse the completed job and never overwrite the source.
    let source = &completed.assets[0];
    let original_bytes = std::fs::read(&source.file).unwrap();
    let bindings = vec![(1, refs[0].clone()), (3, refs[1].clone())];
    let edit_prompt = "  Use reference 3 with referens 1.\nKeep the composition.  ";
    let request = make_edit_request(
        edit_prompt,
        "negative",
        EditSource {
            asset_id: &source.asset_id,
            width: 512,
            height: 288,
        },
        &bindings,
        &meta,
        &settings,
    )
    .unwrap();
    let before = calls.load(Ordering::SeqCst);
    let blocked = dir.path().join("blocked-edit");

    let mut edit = Submission {
        upscale_source: None,
        request: request.clone(),
        key: "edit-version".into(),
        folder: blocked,
        job: None,
    };
    let mut job = None;
    let error = client
        .generate(edit.clone(), |progress| {
            if let Progress::Submitted(id) = progress {
                job = Some(id);
            }
        })
        .await
        .err()
        .unwrap();
    assert!(!error.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), before);
    assert!(job.is_none());
    edit.folder = output.clone();
    let saved_edit = client.generate(edit, |_| {}).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), before + 1);
    assert_ne!(saved_edit.assets[0].file, source.file);
    assert_eq!(std::fs::read(&source.file).unwrap(), original_bytes);
    {
        let requests = requests.lock().unwrap();
        let received = requests.last().unwrap();
        assert_eq!((received.width, received.height), (512, 288));
        assert!(received.prompt.ends_with(edit_prompt));
        assert!(
            received
                .prompt
                .contains("Reference 3 / referens 3 = <image3>")
        );
        assert_eq!(received.references.len(), 3);
        assert_eq!(received.references[0].as_ref(), original_bytes);
    }
    let mut missing = request;
    missing["inputs"]["reference_images"][0] = json!("asset_missing_source");
    let error = client
        .generate(
            Submission {
                upscale_source: None,
                request: missing,
                key: "missing-edit-source".into(),
                folder: output.clone(),
                job: None,
            },
            |_| {},
        )
        .await
        .err()
        .unwrap();
    assert!(!error.retryable);
    assert_eq!(calls.load(Ordering::SeqCst), before + 1);
    // Catalog choices travel through the actual native client's HTTP transport and proxy.
    let model_root = dir.path().join("synthetic-models");
    model_fixture::models(&model_root);
    for relative in ["text_encoders/other/custom.gguf", "vae/other/custom.pt"] {
        let path = model_root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"synthetic uninspected component").unwrap();
    }
    *library.lock().unwrap() = imagine_server::model_library::Library::load(
        model_root,
        dir.path().join("model-library.json"),
    )
    .unwrap();
    let mut catalog = client.library(None, true).await.unwrap();
    let selected = catalog
        .models
        .iter()
        .find(|m| m["model"] == imagine_server::contract::MODEL)
        .unwrap();
    let lora = catalog
        .files
        .iter()
        .find(|f| f.path == "loras/nested/style.safetensors")
        .unwrap();
    let encoder = catalog
        .files
        .iter()
        .find(|f| f.path == "text_encoders/other/custom.gguf")
        .unwrap();
    let vae = catalog
        .files
        .iter()
        .find(|f| f.path == "vae/other/custom.pt")
        .unwrap();
    catalog.settings.components.insert(
        selected["id"].as_str().unwrap().into(),
        imagine_ui::model_catalog::Components {
            text_encoder: encoder.id.clone(),
            vae: vae.id.clone(),
        },
    );
    client
        .library(Some(&catalog.settings), false)
        .await
        .unwrap();
    let mut catalog_request = make_request(
        "Synthetic catalog selection",
        "",
        &[],
        selected,
        &GenerationSettings::default(),
    )
    .unwrap();
    catalog_request["model_selection"] =
        json!({"diffusion":selected["id"],"loras":[{"id":lora.id,"strength":0.4}]});
    let before = calls.load(Ordering::SeqCst);
    client
        .generate(
            Submission {
                request: catalog_request,
                key: "catalog-ui".into(),
                folder: output.clone(),
                job: None,
                upscale_source: None,
            },
            |_| {},
        )
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), before + 1);
    {
        let received = requests.lock().unwrap();
        let workflow = received
            .last()
            .unwrap()
            .model_options
            .generation(imagine_server::contract::MODEL)
            .unwrap();
        assert_eq!(workflow.text_encoder, "text_encoders/other/custom.gguf");
        assert_eq!(workflow.vae, "vae/other/custom.pt");
    }
    assert_eq!(
        requests
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .model_options
            .qwen_image_2_1
            .workflow
            .as_ref()
            .unwrap()
            .loras[0]
            .strength,
        0.4
    );
    let refreshed = client
        .library(Some(&catalog.settings), false)
        .await
        .unwrap();
    assert_eq!(refreshed.models.len(), 2);

    // UI series use the same proxy route, admission key and server FIFO as other clients.
    let before = calls.load(Ordering::SeqCst);
    let series = [
        "SYNTHETIC batch first",
        "SYNTHETIC batch second",
        "SYNTHETIC batch cancelled",
    ]
    .into_iter()
    .enumerate()
    .map(|(index, prompt)| {
        let mut request = make_request(prompt, "", &[], selected, &settings).unwrap();
        request["model_selection"] = json!({"diffusion":selected["id"],"loras":[]});
        Submission {
            request,
            key: format!("series-{index}"),
            folder: output.clone(),
            job: None,
            upscale_source: None,
        }
    })
    .collect::<Vec<_>>();
    let mut invalid = series.clone();
    invalid[1].request["sampling"]["steps"] = json!(0);
    assert!(
        client
            .submit_batch(&invalid, "invalid-series")
            .await
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), before);
    let ids = client.submit_batch(&series, "same-series").await.unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(
        client.submit_batch(&series, "same-series").await.unwrap(),
        ids
    );
    client.cancel(&ids[2]).await.unwrap();
    for (mut submission, id) in series.into_iter().zip(&ids).take(2) {
        submission.job = Some(id.clone());
        assert_eq!(
            client
                .generate(submission, |_| {})
                .await
                .unwrap()
                .assets
                .len(),
            1
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), before + 2);
    {
        let received = requests.lock().unwrap();
        assert_eq!(received[received.len() - 2].prompt, "SYNTHETIC batch first");
        assert_eq!(
            received[received.len() - 1].prompt,
            "SYNTHETIC batch second"
        );
    }
    control.call(Action::Shutdown).await.unwrap();
    scheduler.await.unwrap().unwrap();
    proxy.close().await;
    server.abort();
    backend_server.abort();
}

#[tokio::test]
async fn incomplete_preview_download_preserves_saved_set_and_retry_reuses_job() {
    use axum::{
        Router,
        extract::{Path, State},
        http::StatusCode,
        response::IntoResponse,
        routing::{get, post},
    };
    use std::sync::atomic::AtomicBool;
    #[derive(Clone)]
    struct DownloadState {
        fail: Arc<AtomicBool>,
        submitted: Arc<AtomicUsize>,
        output: PathBuf,
        proof: serde_json::Value,
    }
    async fn submit(State(s): State<DownloadState>) -> axum::Json<serde_json::Value> {
        s.submitted.fetch_add(1, Ordering::SeqCst);
        axum::Json(json!({"id":"job_sheet"}))
    }
    async fn asset(
        State(s): State<DownloadState>,
        Path(id): Path<String>,
    ) -> axum::response::Response {
        if id == "views" && s.fail.load(Ordering::SeqCst) {
            return StatusCode::SERVICE_UNAVAILABLE.into_response();
        }
        ([("content-type", "image/png")], png()).into_response()
    }
    let root = tempfile::tempdir().unwrap();
    let output = root.path().join("outputs");
    let proof = destination::proof(&output);
    for role in ["turnaround", "character_sheet", "portrait"] {
        std::fs::write(output.join(format!("image_0001_{role}.png")), png()).unwrap();
    }
    let state = DownloadState {
        fail: Arc::new(AtomicBool::new(true)),
        submitted: Arc::new(AtomicUsize::new(0)),
        output,
        proof,
    };
    let router=Router::new().route("/v1/media-destination",get(|State(s):State<DownloadState>|async move { axum::Json(s.proof) }))
      .route("/v1/jobs",post(submit)).route("/v1/jobs/{id}",get(|State(s):State<DownloadState>|async move {
        let assets = [("views","turnaround"),("sheet","character_sheet"),("portrait","portrait")].map(|(id,role)|json!({"id":id,"role":role,"persistent":true,"saved_path":s.output.join(format!("image_0001_{role}.png"))}));
        axum::Json(json!({"status":"succeeded","effective":{"seed":42},"output":{"persistent":true,"assets":assets}}))
      })).route("/v1/assets/{id}",get(asset)).with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let client = Client::connect(&Connection { url }).unwrap();
    let mut submission = Submission {
        upscale_source: None,
        request: json!({"operation":"character_sheet"}),
        key: "download-sheet".into(),
        folder: root.path().join("outputs"),
        job: None,
    };
    let mut job = None;
    let failure = client
        .generate(submission.clone(), |p| {
            if let Progress::Submitted(id) = p {
                job = Some(id)
            }
        })
        .await
        .err()
        .unwrap();
    assert!(failure.retryable);
    assert_eq!(std::fs::read_dir(&submission.folder).unwrap().count(), 3);
    submission.job = job;
    state.fail.store(false, Ordering::SeqCst);
    let completed = client.generate(submission.clone(), |_| {}).await.unwrap();
    assert_eq!(completed.assets.len(), 3);
    assert_eq!(
        std::fs::read_dir(completed.assets[0].file.parent().unwrap())
            .unwrap()
            .count(),
        3
    );
    assert_eq!(state.submitted.load(Ordering::SeqCst), 1);
    server.abort();
}

#[path = "../../shared/model_catalog_fixture.rs"]
mod model_fixture;
