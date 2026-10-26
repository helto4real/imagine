//! Own generated fixtures only. The only listener binds loopback port 0.
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action, Control},
    storage::Store,
    video::{contract::*, persistence::*},
};
use imagine_ui::{client::Client, settings::Connection};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{Cursor, Write},
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};
use tokio::sync::{Semaphore, oneshot, watch};
#[path = "support/destination.rs"]
#[allow(dead_code)]
mod destination;
fn root() -> PathBuf {
    let p = destination::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES");
    let p = p.join(format!("streaming-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&p).unwrap();
    p
}
struct NoInference;
impl Backend for NoInference {
    fn generate<'a>(
        &'a mut self,
        _: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async { anyhow::bail!("unexpected inference") })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
struct Harness {
    client: Client,
    session: String,
    store: Store,
    control: Control,
    scheduler: tokio::task::JoinHandle<anyhow::Result<()>>,
    server: tokio::task::JoinHandle<std::io::Result<()>>,
    stop: oneshot::Sender<()>,
}
async fn start(root: &Path) -> Harness {
    let state = root.join(format!("state-{}", uuid::Uuid::new_v4()));
    fs::create_dir(&state).unwrap();
    let mut store = Store::open(&state).await.unwrap();
    store.videos.destination = Some(destination::destination(&root.join("output")));
    let (control, scheduler) = scheduler::start(store.clone(), Box::new(NoInference))
        .await
        .unwrap();
    let (_, engine) = watch::channel(EngineState::default());
    let app = App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(state.join("enhancement.json"))
            .unwrap(),
        library: Default::default(),
        profiles: Arc::new(
            imagine_server::profiles::Catalog::load(Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../config/image/profiles"
            )))
            .unwrap(),
        ),
        store: store.clone(),
        control: control.clone(),
        engine,
        uploads: Arc::new(Semaphore::new(2)),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, api::router(app))
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let client = Client::connect(&Connection {
        url: format!("http://{address}"),
    })
    .unwrap();
    Harness {
        client,
        session: store.session.id.clone(),
        store,
        control,
        scheduler,
        server,
        stop,
    }
}
async fn finish(h: Harness) {
    let _ = h.stop.send(());
    h.server.await.unwrap().unwrap();
    h.control.call(Action::Shutdown).await.unwrap();
    h.scheduler.await.unwrap().unwrap();
    h.store.close().await.unwrap();
}
fn wave(path: &Path, data: u64) {
    assert!(data.is_multiple_of(4) && data + 36 <= u32::MAX as u64);
    let mut f = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    f.write_all(b"RIFF").unwrap();
    f.write_all(&((data + 36) as u32).to_le_bytes()).unwrap();
    f.write_all(b"WAVEfmt ").unwrap();
    for value in [16u32, 0x00020001, 44100, 176400, 0x00100004] {
        f.write_all(&value.to_le_bytes()).unwrap();
    }
    f.write_all(b"data").unwrap();
    f.write_all(&(data as u32).to_le_bytes()).unwrap();
    f.set_len(data + 44).unwrap();
    f.sync_all().unwrap();
}
async fn call(h: &Harness, post: bool, path: &str, body: Option<Value>) -> Value {
    h.client
        .timeline_request(&h.session, post, path, body.as_ref())
        .await
        .unwrap()
}
async fn location(h: &Harness, path: &Path) -> String {
    call(
        h,
        true,
        "/v1/video/locations",
        Some(json!({"schema_version":1,"path":path})),
    )
    .await["location_id"]
        .as_str()
        .unwrap()
        .into()
}
fn peak_rss() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    unsafe { usage.assume_init() }.ru_maxrss as u64 * 1024
}
#[tokio::test]
async fn streaming_import_small_files_cancel_wrong_media_and_relink_errors() {
    let root = root();
    let h = start(&root).await;
    let missing = root.join("missing.wav");
    let cancelled = h
        .client
        .upload_media_cancellable(&h.session, &missing, Arc::new(AtomicBool::new(true)))
        .await
        .unwrap_err();
    assert_eq!(cancelled.code, "media_import_cancelled");
    let invalid = root.join("invalid.wav");
    fs::write(&invalid, b"OWN invalid media").unwrap();
    assert!(h.client.upload_media(&h.session, &invalid).await.is_err());
    let (_, descriptor) = {
        let path = root.join("small.wav");
        wave(&path, 17640);
        let value = h.client.upload_media(&h.session, &path).await.unwrap();
        (path, value)
    };
    assert_eq!(descriptor["bytes"], 17684);
    let create: Value = serde_json::from_str(include_str!(
        "../../design/video-contract/create-project.json"
    ))
    .unwrap();
    let document = call(&h, true, "/v1/video/projects", Some(create)).await;
    let id = document["project"]["id"].as_str().unwrap();
    let route = format!("/v1/video/projects/{id}");
    let attached = call(
        &h,
        true,
        &format!("{route}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":descriptor["id"]})),
    )
    .await;
    let target = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap();
    let mut bytes = Cursor::new(vec![]);
    image::RgbImage::from_pixel(8, 8, image::Rgb([0, 20, 40]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let image = root.join("wrong-kind.png");
    fs::write(&image, bytes.into_inner()).unwrap();
    let image = h.client.upload_media(&h.session, &image).await.unwrap();
    let request = json!({"schema_version":1,"expected_project_revision":1,"asset_id":target,"expected_asset_revision":1,"replacement_asset_id":image["id"]});
    assert!(
        h.client
            .timeline_request(
                &h.session,
                true,
                &format!("{route}/media/relink-preview"),
                Some(&request)
            )
            .await
            .is_err()
    );
    let after = call(&h, false, &route, None).await;
    assert_eq!(after["project"], attached["project"]);
    assert_eq!(after["runtime_assets"], attached["runtime_assets"]);
    finish(h).await;
}
#[tokio::test]
#[ignore = "explicit >2 GiB own encrypted upload; run with IMAGINE_WORKER_A_FIXTURES and --ignored"]
async fn streaming_import_over_two_gib_and_owned_bundle_reopen_relink() {
    let root = root();
    let source = root.join("large-source.wav");
    let data = 2u64 * 1024 * 1024 * 1024 + 65536;
    wave(&source, data);
    let bytes = data + 44;
    let before = peak_rss();
    let h = start(&root).await;
    let uploaded = tokio::time::timeout(
        std::time::Duration::from_secs(240),
        h.client.upload_media(&h.session, &source),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(uploaded["bytes"].as_u64(), Some(bytes));
    let inspected = call(
        &h,
        false,
        &format!("/v1/video/assets/{}", uploaded["id"].as_str().unwrap()),
        None,
    )
    .await;
    assert_eq!(inspected["bytes"].as_u64(), Some(bytes));
    assert_eq!(inspected["media"]["kind"], "audio");
    let owned = h
        .store
        .open_media(uploaded["id"].as_str().unwrap())
        .unwrap()
        .unwrap();
    imagine_server::video::media_health::validate(
        &owned,
        &MediaKind::Audio,
        &AtomicBool::new(false),
    )
    .unwrap();
    drop(owned);
    // A separately generated byte-identical sparse owned bundle avoids a second dense 2 GiB
    // allocation on this own fixture volume. It is validated by the real importer digest.
    let saved = root.join("owned-project");
    fs::create_dir(&saved).unwrap();
    let media = saved.join("media");
    fs::create_dir(&media).unwrap();
    let digest = uploaded["sha256"].as_str().unwrap();
    let blob = media.join(digest);
    wave(&blob, data);
    let mut project = VideoProject::create(
        serde_json::from_str(include_str!(
            "../../design/video-contract/create-project.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let stable = AssetId("stable-large-audio".into());
    project.project_revision = 1;
    project.assets.insert(
        stable.clone(),
        Asset {
            library_source: None,
            name: "Own large audio".into(),
            id: stable.clone(),
            revision: 1,
            kind: MediaKind::Audio,
        },
    );
    project.scenes.push(Scene {
        id: SceneId("scene".into()),
        start: Time { num: 0, den: 1 },
        duration: Time { num: 1, den: 1 },
        description: String::new(),
        model: ModelChoice::ProjectDefault,
        settings: vec![],
        sections: vec![],
        bindings: vec![Binding {
            id: BindingId("stable-binding".into()),
            asset: stable.clone(),
            role: InputRole::AudioReference,
            enabled: true,
            video: None,
        }],
        continuity: None,
        accepted_take: None,
    });
    let mut document = ProjectDocument::new(project);
    let resource = Resource::inspect(
        "large-reference".into(),
        ResourceKind::Reference {
            asset_id: stable.clone(),
            revision: 1,
        },
        &mut File::open(&blob).unwrap(),
    )
    .unwrap();
    assert_eq!(resource.sha256, digest);
    assert_eq!(resource.bytes, bytes);
    document.resources.push(resource);
    fs::write(
        saved.join("project.json"),
        serde_json::to_vec(&document).unwrap(),
    )
    .unwrap();
    let selected = location(&h, &saved).await;
    let opened = call(
        &h,
        true,
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":selected})),
    )
    .await;
    let id = opened["project"]["id"].as_str().unwrap().to_owned();
    assert_eq!(opened["project"]["assets"][&stable.0]["revision"], 1);
    let small = root.join("replacement.wav");
    wave(&small, 17640);
    let replacement = h.client.upload_media(&h.session, &small).await.unwrap();
    let request = json!({"schema_version":1,"expected_project_revision":1,"asset_id":stable,"expected_asset_revision":1,"replacement_asset_id":replacement["id"]});
    let preview = call(
        &h,
        true,
        &format!("/v1/video/projects/{id}/media/relink-preview"),
        Some(request.clone()),
    )
    .await;
    assert_eq!(preview["impacts"][0]["binding"], "stable-binding");
    let applied = call(
        &h,
        true,
        &format!("/v1/video/projects/{id}/media/relink"),
        Some(request),
    )
    .await;
    assert_eq!(
        applied["snapshot"]["project"]["assets"][&stable.0]["revision"],
        2
    );
    assert_eq!(
        applied["snapshot"]["project"]["scenes"],
        opened["project"]["scenes"]
    );
    assert_eq!(
        applied["snapshot"]["resources"].as_array().unwrap().len(),
        2
    );
    let increase = peak_rss().saturating_sub(before);
    assert!(
        increase < 128 * 1024 * 1024,
        "own-process peak RSS grew {increase} bytes"
    );
    finish(h).await;
    let again = start(&root).await;
    let selected = location(&again, &saved).await;
    let reopened = call(
        &again,
        true,
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":selected})),
    )
    .await;
    assert_eq!(reopened["project"]["assets"][&stable.0]["revision"], 2);
    assert_eq!(
        reopened["project"]["scenes"][0]["bindings"][0]["asset"],
        stable.0
    );
    assert!(
        reopened["resources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["bytes"].as_u64() == Some(bytes))
    );
    let evidence = destination::task_fixtures::env_root("IMAGINE_WORKER_A_EVIDENCE");
    fs::write(evidence.join("large-import-proof.json"),serde_json::to_vec_pretty(&json!({"uploaded_bytes":bytes,"sha256":digest,"rss_increase_bytes":increase,"peak_rss_bytes":peak_rss(),"buffer_bytes":imagine_server::video::streaming_import::CHUNK_BYTES,"retained_old_revision":true,"reopened_current_revision":2,"fixture":"Valid PCM WAV with sparse zero samples. Actual HTTP upload is fully read/written/hashed; owned saved bundle is separately generated byte-identical and digest-verified, not copied through initial Save."})).unwrap()).unwrap();
    finish(again).await;
}

#[tokio::test]
#[ignore = "explicit >2 GiB own encrypted video upload; run with own fixture/evidence roots and --ignored"]
async fn streaming_import_over_two_gib_video_with_bounded_reference_interval() {
    use std::io::{Seek, SeekFrom};
    let root = root();
    let source = root.join("large-source.mp4");
    assert!(
        std::process::Command::new("/usr/bin/ffmpeg")
            .args([
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=64x48:rate=10:duration=1",
                "-c:v",
                "mpeg4",
                "-movflags",
                "+faststart"
            ])
            .arg(&source)
            .status()
            .unwrap()
            .success()
    );
    let mut file = File::options()
        .read(true)
        .write(true)
        .open(&source)
        .unwrap();
    let offset = file.seek(SeekFrom::End(0)).unwrap();
    let padding = 2u64 * 1024 * 1024 * 1024 + 65536;
    file.write_all(&(padding as u32).to_be_bytes()).unwrap();
    file.write_all(b"free").unwrap();
    file.set_len(offset + padding).unwrap();
    file.sync_all().unwrap();
    drop(file);
    let bytes = fs::metadata(&source).unwrap().len();
    let before = peak_rss();
    let h = start(&root).await;
    let uploaded = tokio::time::timeout(
        std::time::Duration::from_secs(240),
        h.client.upload_media(&h.session, &source),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(uploaded["bytes"].as_u64(), Some(bytes));
    assert_eq!(uploaded["media"]["kind"], "video");
    let descriptor = h
        .store
        .open_media(uploaded["id"].as_str().unwrap())
        .unwrap()
        .unwrap();
    let decoded = imagine_server::video::reference_video::decode(
        &descriptor,
        &json!({"source_in":0.2,"source_out":0.4}),
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(decoded.frames, 2);
    assert_eq!(decoded.rgb.len(), 2 * 64 * 48 * 3);
    drop(descriptor);
    let increase = peak_rss().saturating_sub(before);
    assert!(
        increase < 128 * 1024 * 1024,
        "own-process RSS grew {increase}"
    );
    let evidence = destination::task_fixtures::env_root("IMAGINE_WORKER_A_EVIDENCE");
    fs::write(evidence.join("large-video-import-proof.json"),serde_json::to_vec_pretty(&json!({"uploaded_bytes":bytes,"sha256":uploaded["sha256"],"rss_increase_bytes":increase,"peak_rss_bytes":peak_rss(),"selected_frames":decoded.frames,"selected_rgb_bytes":decoded.rgb.len(),"fixture":"Valid generated MP4 followed by a valid sparse ISO BMFF free box; complete HTTP byte stream is read, written and hashed."})).unwrap()).unwrap();
    finish(h).await;
}

#[tokio::test]
async fn streaming_import_client_cancel_aborts_an_inflight_request() {
    let root = root();
    let source = root.join("cancel-request.wav");
    wave(&source, 1024 * 1024);
    let (started, received) = oneshot::channel();
    let started = Arc::new(std::sync::Mutex::new(Some(started)));
    let expected = source.clone();
    let route = axum::Router::new().route(
        "/v1/video/assets/link",
        axum::routing::post(move |request: axum::extract::Request| {
            let started = started.clone();
            let expected = expected.clone();
            async move {
                let bytes = axum::body::to_bytes(request.into_body(), 64 * 1024)
                    .await
                    .unwrap();
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(body["path"], json!(expected));
                if let Some(signal) = started.lock().unwrap().take() {
                    let _ = signal.send(());
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                axum::Json(json!({"id":"own-unused-upload"}))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, route)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
    });
    let client = Client::connect(&Connection {
        url: format!("http://{address}"),
    })
    .unwrap();
    let flag = Arc::new(AtomicBool::new(false));
    let cancel = flag.clone();
    let task = tokio::spawn(async move {
        client
            .upload_media_cancellable("own-session", &source, cancel)
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(3), received)
        .await
        .unwrap()
        .unwrap();
    flag.store(true, std::sync::atomic::Ordering::Release);
    let result = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(result.code, "media_import_cancelled");
    let _ = stop.send(());
    server.await.unwrap().unwrap();
}

/// The source AND the first saved resource are produced through real writes and
/// normal API paths. No sparse source, hand-authored project.json or saved bundle.
#[tokio::test]
#[ignore = "dense >2 GiB first HTTP Attach+Save; own fixture/evidence roots and --ignored required"]
async fn streaming_import_dense_over_two_gib_first_attach_save_and_fresh_harness_reopen() {
    use std::{os::unix::fs::MetadataExt, time::Instant};

    fn fingerprint(path: &Path) -> Resource {
        Resource::inspect(
            "own-dense-content-probe".into(),
            ResourceKind::Reference {
                asset_id: AssetId("own-dense-content-probe".into()),
                revision: 1,
            },
            &mut File::open(path).unwrap(),
        )
        .unwrap()
    }
    fn allocation(path: &Path) -> Value {
        let metadata = fs::metadata(path).unwrap();
        assert!(metadata.is_file());
        let allocated_bytes = metadata.blocks() * 512;
        // The high-entropy payload is deliberately not compressible zero data.
        // st_blocks is per-inode allocated space; it does NOT prove physically
        // unique extents (including on CoW/deduplicating filesystems).
        assert!(
            allocated_bytes >= metadata.len(),
            "dense fixture allocation below logical bytes"
        );
        json!({"path":path,"logical_bytes":metadata.len(),"st_blocks":metadata.blocks(),
            "allocated_bytes":allocated_bytes,"physical_unique_extents_measured":false})
    }
    fn write_dense_wave(path: &Path, data: u64) -> u64 {
        assert!(data.is_multiple_of(4) && data + 36 <= u64::from(u32::MAX));
        let mut pool = vec![0u8; 1024 * 1024];
        let mut state = 0x6d50_0a9b_7c31_42e7u64;
        for byte in &mut pool {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = (state >> 24) as u8;
        }
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        let mut header = Vec::with_capacity(44);
        header.extend_from_slice(b"RIFF");
        header.extend_from_slice(&((data + 36) as u32).to_le_bytes());
        header.extend_from_slice(b"WAVEfmt ");
        for value in [16u32, 0x00020001, 44100, 176400, 0x00100004] {
            header.extend_from_slice(&value.to_le_bytes());
        }
        header.extend_from_slice(b"data");
        header.extend_from_slice(&(data as u32).to_le_bytes());
        assert_eq!(header.len(), 44);
        file.write_all(&header).unwrap();
        let mut written = 0;
        while written < data {
            let count = (data - written).min(pool.len() as u64) as usize;
            file.write_all(&pool[..count]).unwrap();
            written += count as u64;
        }
        file.sync_all().unwrap();
        assert_eq!(file.metadata().unwrap().len(), data + 44);
        written + header.len() as u64
    }

    let root = root();
    let evidence = destination::task_fixtures::env_root("IMAGINE_WORKER_A_EVIDENCE");
    let source = root.join("dense-first-attach-source.wav");
    let saved_root = root.join("first-api-save");
    fs::create_dir(&saved_root).unwrap();
    assert_eq!(fs::read_dir(&saved_root).unwrap().count(), 0);
    let data = 2u64 * 1024 * 1024 * 1024 + 65536;
    let bytes = data + 44;
    let total_started = Instant::now();
    let rss_before = peak_rss();
    let started = Instant::now();
    let written = write_dense_wave(&source, data);
    let source_write_seconds = started.elapsed().as_secs_f64();
    assert_eq!(written, bytes);
    let source_allocation = allocation(&source);
    let started = Instant::now();
    let original = fingerprint(&source);
    let original_hash_seconds = started.elapsed().as_secs_f64();
    let rss_after_fixture = peak_rss();
    assert_eq!(original.bytes, bytes);

    let started = Instant::now();
    let first = Arc::new(start(&root).await);
    let first_start_seconds = started.elapsed().as_secs_f64();
    // The spawned stage captures assertion panics as JoinError. The outer test
    // always joins HTTP + scheduler and closes Store before surfacing that error.
    let stage = {
        let h = first.clone();
        let source = source.clone();
        let saved_root = saved_root.clone();
        let expected_hash = original.sha256.clone();
        tokio::spawn(async move {
            let mut timings = serde_json::Map::new();
            let mut rss = serde_json::Map::new();
            let started = Instant::now();
            let uploaded = h.client.upload_media(&h.session, &source).await.unwrap();
            timings.insert(
                "upload_seconds".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            rss.insert("after_upload".into(), json!(peak_rss()));
            assert_eq!(uploaded["bytes"].as_u64(), Some(bytes));
            assert_eq!(uploaded["sha256"].as_str(), Some(expected_hash.as_str()));
            let upload_id = uploaded["id"].as_str().unwrap();
            let create: Value = serde_json::from_str(include_str!(
                "../../design/video-contract/create-project.json"
            ))
            .unwrap();
            let created = call(&h, true, "/v1/video/projects", Some(create)).await;
            assert_eq!(created["persistence"]["state"], "session_only");
            assert!(created["project"]["assets"].as_object().unwrap().is_empty());
            let project_id = created["project"]["id"].as_str().unwrap().to_owned();
            let route = format!("/v1/video/projects/{project_id}");
            let revision = created["project"]["project_revision"].as_u64().unwrap();
            let started = Instant::now();
            let attached = call(
                &h,
                true,
                &format!("{route}/assets"),
                Some(json!({
                    "schema_version":1,"expected_project_revision":revision,
                    "asset_id":upload_id,"name":"Own dense PCM16 stereo source"
                })),
            )
            .await;
            timings.insert(
                "first_attach_seconds".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            rss.insert("after_first_attach".into(), json!(peak_rss()));
            assert_eq!(attached["persistence"]["state"], "session_only");
            let assets = attached["project"]["assets"].as_object().unwrap();
            assert_eq!(assets.len(), 1);
            let stable_id = assets.keys().next().unwrap().to_owned();
            assert_eq!(assets[&stable_id]["kind"], "audio");
            assert_eq!(assets[&stable_id]["revision"], 1);
            assert_eq!(attached["runtime_assets"][&stable_id], upload_id);
            let resources = attached["resources"].as_array().unwrap();
            assert_eq!(resources.len(), 1);
            let reference: Resource = serde_json::from_value(resources[0].clone()).unwrap();
            assert!(
                matches!(&reference.kind, ResourceKind::Reference {asset_id,revision} if asset_id.0 == stable_id && *revision == 1)
            );
            assert_eq!(reference.bytes, bytes);
            assert_eq!(reference.sha256, expected_hash);
            assert_eq!(fs::read_dir(&saved_root).unwrap().count(), 0);
            let location = location(&h, &saved_root).await;
            let revision = attached["project"]["project_revision"].as_u64().unwrap();
            let started = Instant::now();
            let saved = call(
                &h,
                true,
                &format!("{route}/save"),
                Some(json!({
                    "schema_version":1,"expected_project_revision":revision,"location_id":location
                })),
            )
            .await;
            timings.insert(
                "first_save_seconds".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            rss.insert("after_first_save".into(), json!(peak_rss()));
            assert_eq!(saved["persistence"]["state"], "saved");
            assert_eq!(
                saved["persistence"]["saved_revision"].as_u64(),
                Some(revision)
            );
            assert_eq!(saved["project"], attached["project"]);
            assert_eq!(saved["resources"], attached["resources"]);
            assert!(saved_root.join("project.json").is_file());
            let blob = saved_root.join("media").join(&reference.sha256);
            let saved_allocation = allocation(&blob);
            let started = Instant::now();
            let stored = fingerprint(&blob);
            timings.insert(
                "saved_blob_hash_seconds".into(),
                json!(started.elapsed().as_secs_f64()),
            );
            assert_eq!(stored.bytes, bytes);
            assert_eq!(stored.sha256, expected_hash);
            (
                project_id,
                stable_id,
                saved,
                saved_allocation,
                timings,
                rss,
                h.session.clone(),
            )
        })
    };
    let result = stage.await;
    let started = Instant::now();
    finish(Arc::try_unwrap(first).unwrap_or_else(|_| panic!("first stage retained its harness")))
        .await;
    let first_shutdown_seconds = started.elapsed().as_secs_f64();
    let (project_id, stable_id, saved, saved_allocation, mut timings, mut rss, first_session) =
        result.expect("first Attach+Save failed; first harness was closed");
    let started = Instant::now();
    let fresh = Arc::new(start(&root).await);
    let fresh_start_seconds = started.elapsed().as_secs_f64();
    let stage = {
        let h = fresh.clone();
        let saved_root = saved_root.clone();
        let stable_id = stable_id.clone();
        let project_id = project_id.clone();
        let expected = saved.clone();
        let first_session = first_session.clone();
        tokio::spawn(async move {
            assert_ne!(h.session, first_session);
            let location = location(&h, &saved_root).await;
            let started = Instant::now();
            let reopened = call(
                &h,
                true,
                "/v1/video/projects/open",
                Some(json!({
                    "schema_version":1,"location_id":location
                })),
            )
            .await;
            let open_seconds = started.elapsed().as_secs_f64();
            assert_eq!(
                reopened["project"]["id"].as_str(),
                Some(project_id.as_str())
            );
            assert_eq!(reopened["persistence"]["state"], "saved");
            assert_eq!(reopened["project"], expected["project"]);
            assert_eq!(reopened["resources"], expected["resources"]);
            let runtime_id = reopened["runtime_assets"][&stable_id].as_str().unwrap();
            let runtime = call(&h, false, &format!("/v1/video/assets/{runtime_id}"), None).await;
            assert_eq!(runtime["bytes"].as_u64(), Some(bytes));
            assert_eq!(runtime["sha256"], expected["resources"][0]["sha256"]);
            assert_eq!(runtime["media"]["kind"], "audio");
            (open_seconds, peak_rss(), runtime, h.session.clone())
        })
    };
    let result = stage.await;
    let started = Instant::now();
    finish(Arc::try_unwrap(fresh).unwrap_or_else(|_| panic!("reopen stage retained its harness")))
        .await;
    let fresh_shutdown_seconds = started.elapsed().as_secs_f64();
    let (reopen_seconds, rss_after_reopen, runtime, fresh_session) =
        result.expect("reopen failed; fresh harness was closed");
    let started = Instant::now();
    let original_after = fingerprint(&source);
    let original_recheck_seconds = started.elapsed().as_secs_f64();
    assert_eq!(original_after.bytes, original.bytes);
    assert_eq!(original_after.sha256, original.sha256);
    timings.extend([
        ("source_write_seconds".into(), json!(source_write_seconds)),
        ("original_hash_seconds".into(), json!(original_hash_seconds)),
        (
            "first_harness_start_seconds".into(),
            json!(first_start_seconds),
        ),
        (
            "first_harness_shutdown_seconds".into(),
            json!(first_shutdown_seconds),
        ),
        (
            "fresh_harness_start_seconds".into(),
            json!(fresh_start_seconds),
        ),
        ("fresh_reopen_seconds".into(), json!(reopen_seconds)),
        (
            "fresh_harness_shutdown_seconds".into(),
            json!(fresh_shutdown_seconds),
        ),
        (
            "original_recheck_seconds".into(),
            json!(original_recheck_seconds),
        ),
        (
            "total_seconds".into(),
            json!(total_started.elapsed().as_secs_f64()),
        ),
    ]);
    rss.extend([
        ("before_fixture".into(), json!(rss_before)),
        ("after_fixture".into(), json!(rss_after_fixture)),
        ("after_fresh_reopen".into(), json!(rss_after_reopen)),
        ("peak_bytes".into(), json!(peak_rss())),
        (
            "peak_growth_bytes".into(),
            json!(peak_rss().saturating_sub(rss_before)),
        ),
    ]);
    let proof = json!({
        "schema_version":1,"test":"dense_first_http_attach_save_reopen","case_root":root,
        "content":{"bytes":bytes,"sha256":original.sha256,"data_bytes":data,
            "generator":"deterministic xorshift64 high-entropy 1MiB pool, repeated write_all",
            "write_all_bytes":written,"sparse_source":false,"saved_bundle_mock":false},
        "source_allocation":source_allocation,"saved_blob_allocation":saved_allocation,
        "allocation_note":"st_blocks is per-inode allocation, not unique physical extents; no physical uniqueness claim",
        "stable_project_id":project_id,"stable_asset_id":stable_id,
        "asset_revision":1,"resource":saved["resources"][0],"reopened_runtime_asset":runtime,
        "first_session_id":first_session,"fresh_session_id":fresh_session,
        "closed_harnesses":2,"timings":timings,"process_peak_rss_bytes":rss,
        "rss_note":"getrusage ru_maxrss is cumulative for this one test process; run this ignored test alone",
        "transport_buffer_bytes":imagine_server::video::streaming_import::CHUNK_BYTES,
        "product_settings_changed":false
    });
    let proof_path = evidence.join(format!(
        "dense-first-attach-save-{}.json",
        uuid::Uuid::new_v4()
    ));
    let mut proof_file = File::options()
        .write(true)
        .create_new(true)
        .open(proof_path)
        .unwrap();
    proof_file
        .write_all(&serde_json::to_vec_pretty(&proof).unwrap())
        .unwrap();
    proof_file.sync_all().unwrap();
}
