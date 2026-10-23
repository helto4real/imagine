//! Synthetic HTTP/media/persistence cases, with all file fixtures on the owned encrypted volume.
#[path = "support/video_authoring.rs"]
mod authoring_tests;
#[path = "../ui/tests/support/destination.rs"]
#[allow(dead_code)]
mod destination;
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action, Control},
    storage::Store,
};
use serde_json::{Value, json};
use std::{
    io::{Cursor, Write},
    path::Path,
    sync::Arc,
};
use tokio::sync::{Semaphore, watch};
use tower::ServiceExt;
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
    router: Router,
    store: Store,
    control: Control,
    scheduler: tokio::task::JoinHandle<anyhow::Result<()>>,
}
async fn setup(root: &Path, name: &str) -> Harness {
    let data = root.join(name);
    std::fs::create_dir_all(&data).unwrap();
    let mut store = Store::open(&data).await.unwrap();
    store.videos.destination = Some(destination::destination(&root.join("output")));
    let (control, scheduler) = scheduler::start(store.clone(), Box::new(NoInference))
        .await
        .unwrap();
    let (_, engine) = watch::channel(EngineState::default());
    let router = api::router(App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(data.join("enhancement.json"))
            .unwrap(),
        library: Default::default(),
        profiles: Arc::new(
            imagine_server::profiles::Catalog::load(Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/profiles"
            )))
            .unwrap(),
        ),
        store: store.clone(),
        control: control.clone(),
        engine,
        uploads: Arc::new(Semaphore::new(2)),
    });
    Harness {
        router,
        store,
        control,
        scheduler,
    }
}
async fn stop(h: Harness) {
    h.control.call(Action::Shutdown).await.unwrap();
    h.scheduler.await.unwrap().unwrap();
    h.store.close().await.unwrap();
}
async fn call(h: &Harness, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
    raw(
        h,
        method,
        path,
        body.map(|v| Body::from(v.to_string()))
            .unwrap_or_else(Body::empty),
        "application/json",
        true,
    )
    .await
}
async fn raw(
    h: &Harness,
    method: &str,
    path: &str,
    body: Body,
    mime: &str,
    session: bool,
) -> (u16, Value) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("Content-Type", mime);
    if session {
        request = request.header("X-Imagine-Session", &h.store.session.id);
    }
    let response = h
        .router
        .clone()
        .oneshot(request.body(body).unwrap())
        .await
        .unwrap();
    assert_eq!(response.headers()["Cache-Control"], "no-store");
    let status = response.status().as_u16();
    let bytes = to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}
fn create() -> Value {
    serde_json::from_str(include_str!("../design/video-contract/create-project.json")).unwrap()
}
fn png() -> Vec<u8> {
    let mut out = Cursor::new(vec![]);
    image::RgbImage::from_pixel(16, 16, image::Rgb([50, 80, 120]))
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}
async fn select(h: &Harness, path: &Path) -> String {
    std::fs::create_dir_all(path).unwrap();
    let (status, result) = call(
        h,
        "POST",
        "/v1/video/locations",
        Some(json!({"schema_version":1,"path":path})),
    )
    .await;
    assert_eq!(status, 201, "{result}");
    result["location_id"].as_str().unwrap().into()
}
async fn download(h: &Harness, id: &str, range: Option<&str>) -> (u16, Vec<u8>) {
    let mut request = Request::builder()
        .uri(format!("/v1/assets/{id}"))
        .header("X-Imagine-Session", &h.store.session.id);
    if let Some(range) = range {
        request = request.header("Range", range);
    }
    let response = h
        .router
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    (
        status,
        to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap()
            .to_vec(),
    )
}
#[tokio::test]
async fn opening_the_same_saved_project_reattaches_its_live_document() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "reattach-project").await;
    let (_, created) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let id = created["project"]["id"].as_str().unwrap();
    let folder = root.path().join("saved-project");
    let location = select(&h, &folder).await;
    let (status, saved) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{id}/save"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"location_id":location})),
    )
    .await;
    assert_eq!(status, 200, "{saved}");
    let (status,edited)=call(&h,"POST",&format!("/v1/video/projects/{id}/edits"),Some(json!({"schema_version":1,"expected_project_revision":0,"commands":[{"operation":"set_global_prompt","prompt":"own live source"}]}))).await;
    assert_eq!(status, 200, "{edited}");
    for _ in 0..2 {
        let (status, opened) = call(
            &h,
            "POST",
            "/v1/video/projects/open",
            Some(json!({"schema_version":1,"location_id":location})),
        )
        .await;
        assert_eq!(status, 200, "{opened}");
        assert_eq!(opened["project"], edited["project"]);
        assert_eq!(opened["runtime_assets"], edited["runtime_assets"]);
    }
    let duplicate = root.path().join("independent-folder");
    let other = select(&h, &duplicate).await;
    std::fs::copy(folder.join("project.json"), duplicate.join("project.json")).unwrap();
    let (status, error) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":other})),
    )
    .await;
    assert_eq!(status, 409, "{error}");
    assert_eq!(error["error"]["code"], "project_already_open");
    assert_eq!(
        call(&h, "GET", &format!("/v1/video/projects/{id}"), None)
            .await
            .1["project"],
        edited["project"]
    );
    stop(h).await;
}

#[tokio::test]
async fn projects_save_autosave_reopen_media_and_save_as_survive_session_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state1").await;
    assert_eq!(
        raw(
            &h,
            "POST",
            "/v1/video/projects",
            Body::from(create().to_string()),
            "application/json",
            false
        )
        .await
        .0,
        401
    );
    let (_, created) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let id = created["project"]["id"].as_str().unwrap().to_owned();
    let url = format!("/v1/video/projects/{id}");
    assert_eq!(created["persistence"]["state"], "session_only");
    let image = png();
    let (status, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(image.clone()),
        "image/png",
        true,
    )
    .await;
    assert_eq!(status, 201, "{uploaded}");
    let old_asset = uploaded["id"].as_str().unwrap().to_owned();
    let (status, attached) = call(
        &h,
        "POST",
        &format!("{url}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":old_asset})),
    )
    .await;
    assert_eq!(status, 200, "{attached}");
    assert_eq!(attached["project"]["project_revision"], 1);
    assert_eq!(
        std::fs::read_dir(root.path().join("output"))
            .unwrap()
            .count(),
        0
    );
    let (status,conflict)=call(&h,"POST",&format!("{url}/edits"),Some(json!({"schema_version":1,"expected_project_revision":0,"commands":[{"operation":"set_global_prompt","prompt":"stale"}]}))).await;
    assert_eq!(status, 409, "{conflict}");
    assert_eq!(
        call(&h, "GET", &url, None).await.1["project"]["global_prompt"],
        ""
    );
    let project = root.path().join("saved-project");
    let location = select(&h, &project).await;
    let (status, saved) = call(
        &h,
        "POST",
        &format!("{url}/save"),
        Some(json!({"schema_version":1,"expected_project_revision":1,"location_id":location})),
    )
    .await;
    assert_eq!(status, 200, "{saved}");
    assert_eq!(saved["persistence"]["state"], "saved");
    let (status,edited)=call(&h,"POST",&format!("{url}/edits"),Some(json!({"schema_version":1,"expected_project_revision":1,"commands":[{"operation":"set_global_prompt","prompt":"own saved text"}]}))).await;
    assert_eq!(status, 200, "{edited}");
    assert_eq!(edited["persistence"]["saved_revision"], 2);
    let text = std::fs::read_to_string(project.join("project.json")).unwrap();
    assert!(!text.contains(&old_asset));
    assert!(!text.contains(&h.store.session.id));
    let media = h.store.media.clone();
    stop(h).await;
    assert!(!media.exists());
    let h = setup(root.path(), "state2").await;
    assert_eq!(download(&h, &old_asset, None).await.0, 404);
    let location = select(&h, &project).await;
    let (status, opened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":location})),
    )
    .await;
    assert_eq!(status, 200, "{opened}");
    assert_eq!(opened["project"]["id"], id);
    assert_eq!(opened["project"]["global_prompt"], "own saved text");
    let new_asset = opened["runtime_assets"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned();
    assert_ne!(new_asset, old_asset);
    assert_eq!(download(&h, &new_asset, None).await, (200, image.clone()));
    assert_eq!(
        download(&h, &new_asset, Some("bytes=2-9")).await,
        (206, image[2..10].to_vec())
    );
    let copied = root.path().join("copied-project");
    let location = select(&h, &copied).await;
    let (status, copy) = call(
        &h,
        "POST",
        &format!("{url}/save-as"),
        Some(json!({"schema_version":1,"expected_project_revision":2,"location_id":location})),
    )
    .await;
    assert_eq!(status, 201, "{copy}");
    assert_ne!(copy["project"]["id"], id);
    std::fs::remove_dir_all(&project).unwrap();
    stop(h).await;
    let h = setup(root.path(), "state3").await;
    let location = select(&h, &copied).await;
    let (status, opened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":location})),
    )
    .await;
    assert_eq!(status, 200, "{opened}");
    let asset = opened["runtime_assets"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(download(&h, asset, None).await, (200, image));
    stop(h).await;
}
#[tokio::test]
async fn video_and_audio_uploads_stream_inspect_range_and_reject_interrupted_bodies() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let shape = imagine_server::h3_protocol::Shape {
        width: 64,
        height: 64,
        frames: 5,
        fps: 24,
    };
    let bytes = imagine_server::short_av::encode(imagine_server::short_av::Av {
        continuation: None,
        shape,
        frames: vec![vec![91; shape.frame_bytes()]; 5],
        pcm: vec![0; shape.audio_samples() as usize * 8],
    })
    .unwrap();
    let chunks = bytes
        .chunks(37)
        .map(|c| Ok::<_, std::io::Error>(axum::body::Bytes::copy_from_slice(c)))
        .collect::<Vec<_>>();
    let (status, video) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from_stream(futures_util::stream::iter(chunks)),
        "video/mp4",
        true,
    )
    .await;
    assert_eq!(status, 201, "{video}");
    assert_eq!(video["media"]["audio"], true);
    assert_eq!(video["media"]["fps"], json!({"num":24,"den":1}));
    let duration = &video["media"]["duration"];
    let seconds = duration["num"].as_f64().unwrap() / duration["den"].as_f64().unwrap();
    assert!((seconds - 5.0 / 24.0).abs() <= 1.0 / 32000.0);
    assert_eq!(
        download(&h, video["id"].as_str().unwrap(), Some("bytes=-8")).await,
        (206, bytes[bytes.len() - 8..].to_vec())
    );
    let mut wav = Cursor::new(vec![]);
    wav.write_all(b"RIFF").unwrap();
    wav.write_all(&(36u32 + 6400).to_le_bytes()).unwrap();
    wav.write_all(b"WAVEfmt ").unwrap();
    for v in [16u32, 0x00020001, 32000, 128000, 0x00100004] {
        wav.write_all(&v.to_le_bytes()).unwrap();
    }
    wav.write_all(b"data").unwrap();
    wav.write_all(&6400u32.to_le_bytes()).unwrap();
    wav.write_all(&vec![0; 6400]).unwrap();
    let (status, audio) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(wav.into_inner()),
        "audio/wav",
        true,
    )
    .await;
    assert_eq!(status, 201, "{audio}");
    assert_eq!(audio["media"]["kind"], "audio");
    assert_eq!(audio["media"]["sample_rate"], 32000);
    assert_eq!(audio["media"]["duration"], json!({"num":1,"den":20}));
    let (status, metadata) = call(
        &h,
        "GET",
        &format!("/v1/video/assets/{}", audio["id"].as_str().unwrap()),
        None,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(metadata, audio);
    let before: i64 = sqlx::query_scalar("SELECT count(*) FROM assets")
        .fetch_one(&h.store.db)
        .await
        .unwrap();
    let chunks = [
        Ok(axum::body::Bytes::from_static(b"own partial")),
        Err(std::io::Error::other("synthetic interruption")),
    ];
    let (status, error) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from_stream(futures_util::stream::iter(chunks)),
        "video/mp4",
        true,
    )
    .await;
    assert_eq!(status, 422);
    assert_eq!(error["error"]["code"], "upload_interrupted");
    assert_eq!(
        raw(
            &h,
            "POST",
            "/v1/video/assets",
            Body::from(png()),
            "video/mp4",
            true
        )
        .await
        .0,
        422
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM assets")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        before
    );
    assert_eq!(
        std::fs::read_dir(root.path().join("output"))
            .unwrap()
            .count(),
        0
    );
    stop(h).await;
}
#[test]
fn imported_piped_mkv_uses_packet_span_instead_of_bitrate_estimate() {
    use std::process::Stdio;
    // Unknown-length rawvideo, like the former exporter. An output -t would
    // publish a known duration and fail to reproduce its missing Duration field.
    let mut child = std::process::Command::new("/usr/bin/ffmpeg")
        .args([
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "416x224",
            "-framerate",
            "24",
            "-i",
            "pipe:0",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=1.625",
            "-ac",
            "2",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            "-c:a",
            "pcm_f32le",
            "-f",
            "matroska",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let write = std::thread::spawn(move || {
        let mut seed = 0x12345678u32;
        let bytes: Vec<u8> = (0..416 * 224 * 3 * 39)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as u8
            })
            .collect();
        input.write_all(&bytes).unwrap();
    });
    let out = child.wait_with_output().unwrap();
    write.join().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut source = tempfile::NamedTempFile::new().unwrap();
    source.write_all(&out.stdout).unwrap();
    let probe = std::process::Command::new("/usr/bin/ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
        ])
        .arg(source.path())
        .output()
        .unwrap();
    assert!(probe.status.success());
    let value: Value = serde_json::from_slice(&probe.stdout).unwrap();
    let estimate: f64 = value["format"]["duration"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        (estimate - 1.625).abs() > 0.01,
        "fixture must exercise absent Duration and the wrong bitrate estimate: {estimate}"
    );
    let media =
        imagine_server::video::media::inspect(source.as_file(), "video/x-matroska").unwrap();
    let time = media.duration.unwrap();
    assert!((time.num as f64 / time.den as f64 - 1.625).abs() <= 0.001);
    assert_eq!(media.fps.unwrap().num, 24);
    assert!(media.audio);
}
#[tokio::test]
async fn reopen_is_atomic_when_a_later_resource_has_invalid_media_and_ranges_are_independent() {
    use imagine_server::video::{contract::*, persistence::*};
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let pixels = png();
    let (_, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(pixels.clone()),
        "image/png",
        true,
    )
    .await;
    let asset = uploaded["id"].as_str().unwrap();
    let (a, b) = tokio::join!(
        download(&h, asset, Some("bytes=0-7")),
        download(&h, asset, Some("bytes=9-20"))
    );
    assert_eq!(a, (206, pixels[..8].to_vec()));
    assert_eq!(b, (206, pixels[9..21].to_vec()));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/video/assets")
        .header("X-Imagine-Session", &h.store.session.id)
        .header("Content-Type", "video/mp4")
        .header(
            "Content-Length",
            (imagine_server::video::media::MAX_UPLOAD + 1).to_string(),
        )
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        h.router.clone().oneshot(request).await.unwrap().status(),
        413
    );
    let mut project = ProjectSession::new(
        VideoProject::create(serde_json::from_value(create()).unwrap()).unwrap(),
    )
    .unwrap();
    let mut document = project.document().clone();
    let id = document.project.id.0.clone();
    let mut inputs = std::collections::BTreeMap::new();
    for (name, bytes) in [
        ("first", pixels.as_slice()),
        ("second", b"own invalid image".as_slice()),
    ] {
        let asset_id = AssetId(name.into());
        document.project.assets.insert(
            asset_id.clone(),
            Asset {
                id: asset_id.clone(),
                revision: 1,
                kind: MediaKind::Image,
            },
        );
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        let resource = Resource::inspect(
            name.into(),
            ResourceKind::Reference {
                asset_id,
                revision: 1,
            },
            &mut file,
        )
        .unwrap();
        inputs.insert(resource.key(), file);
        document.resources.push(resource);
    }
    let staging = SessionStaging::select(&root.path().join("output")).unwrap();
    project.update(0, document, inputs, &staging).unwrap();
    let folder = root.path().join("bad-project");
    std::fs::create_dir(&folder).unwrap();
    project.save_to(&folder).unwrap();
    drop(project);
    let location = select(&h, &folder).await;
    let (status, error) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":location})),
    )
    .await;
    assert_eq!(status, 422, "{error}");
    assert_eq!(
        call(&h, "GET", &format!("/v1/video/projects/{id}"), None)
            .await
            .0,
        404
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT count(*) FROM assets")
            .fetch_one(&h.store.db)
            .await
            .unwrap(),
        1
    );
    stop(h).await;
}
#[tokio::test]
async fn changed_location_and_failed_autosave_keep_unsaved_draft_and_old_manifest() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state").await;
    let (_, created) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let id = created["project"]["id"].as_str().unwrap();
    let url = format!("/v1/video/projects/{id}");
    let path = root.path().join("project");
    let location = select(&h, &path).await;
    let moved = root.path().join("moved");
    std::fs::rename(&path, &moved).unwrap();
    std::fs::create_dir(&path).unwrap();
    let (status, error) = call(
        &h,
        "POST",
        &format!("{url}/save"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"location_id":location})),
    )
    .await;
    assert_eq!(status, 409, "{error}");
    assert_eq!(std::fs::read_dir(&path).unwrap().count(), 0);
    std::fs::remove_dir(&path).unwrap();
    std::fs::rename(&moved, &path).unwrap();
    let location = select(&h, &path).await;
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("{url}/save"),
            Some(json!({"schema_version":1,"expected_project_revision":0,"location_id":location}))
        )
        .await
        .0,
        200
    );
    let old = std::fs::read(path.join("project.json")).unwrap();
    std::fs::rename(&path, &moved).unwrap();
    std::fs::create_dir(&path).unwrap();
    let (status,error)=call(&h,"POST",&format!("{url}/edits"),Some(json!({"schema_version":1,"expected_project_revision":0,"commands":[{"operation":"set_global_prompt","prompt":"own retained draft"}]}))).await;
    assert_eq!(status, 409, "{error}");
    assert_eq!(
        error["snapshot"]["project"]["global_prompt"],
        "own retained draft"
    );
    assert_eq!(error["snapshot"]["persistence"]["state"], "not_saved");
    assert_eq!(error["snapshot"]["persistence"]["saved_revision"], 0);
    assert_eq!(std::fs::read(moved.join("project.json")).unwrap(), old);
    std::fs::remove_dir(&path).unwrap();
    std::fs::rename(&moved, &path).unwrap();
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("{url}/save"),
            Some(json!({"schema_version":1,"expected_project_revision":1}))
        )
        .await
        .0,
        200
    );
    stop(h).await;
}
#[tokio::test]
async fn explicit_library_project_and_character_copies_are_independent_and_revision_bound() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "state1").await;
    let (_, created) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let project = created["project"]["id"].as_str().unwrap();
    let (_, uploaded) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(png()),
        "image/png",
        true,
    )
    .await;
    let asset = uploaded["id"].as_str().unwrap().to_owned();
    let (s, _) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":asset})),
    )
    .await;
    assert_eq!(s, 200);
    let folder = root.path().join("library");
    let location = select(&h, &folder).await;
    let (s, library) = call(
        &h,
        "POST",
        "/v1/video/libraries",
        Some(json!({"schema_version":1,"location_id":location,"mode":"create"})),
    )
    .await;
    assert_eq!(s, 200, "{library}");
    let id = library["id"].as_str().unwrap().to_owned();
    let(s,saved)=call(&h,"POST",&format!("/v1/video/libraries/{id}/projects"),Some(json!({"schema_version":1,"expected_library_revision":0,"project_id":project,"expected_project_revision":1,"title":"Own reusable project"}))).await;
    assert_eq!(s, 200, "{saved}");
    let entry = saved["entry_id"].as_str().unwrap().to_owned();
    let (s, copied) = call(
        &h,
        "POST",
        &format!("/v1/video/libraries/{id}/entries/{entry}/project-copy"),
        Some(json!({"schema_version":1,"expected_entry_revision":1})),
    )
    .await;
    assert_eq!(s, 200, "{copied}");
    assert_ne!(copied["project"]["id"], project);
    assert_eq!(copied["origin"]["entry_id"], entry);
    assert_eq!(copied["persistence"]["state"], "session_only");
    let copy_asset = copied["runtime_assets"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned();
    assert_ne!(copy_asset, asset);
    assert_eq!(download(&h, &copy_asset, None).await, (200, png()));
    call(&h,"POST",&format!("/v1/video/projects/{project}/edits"),Some(json!({"schema_version":1,"expected_project_revision":1,"commands":[{"operation":"set_global_prompt","prompt":"Own revised library source"}]}))).await;
    let(s,updated)=call(&h,"POST",&format!("/v1/video/libraries/{id}/projects"),Some(json!({"schema_version":1,"expected_library_revision":1,"project_id":project,"expected_project_revision":2,"title":"Updated","replace":{"entry_id":entry,"expected_entry_revision":1}}))).await;
    assert_eq!(s, 200, "{updated}");
    assert_eq!(updated["library"]["entries"][0]["revision"], 2);
    assert_eq!(
        call(
            &h,
            "POST",
            &format!("/v1/video/libraries/{id}/entries/{entry}/project-copy"),
            Some(json!({"schema_version":1,"expected_entry_revision":1}))
        )
        .await
        .0,
        409
    );
    let copy_id = copied["project"]["id"].as_str().unwrap();
    assert_eq!(
        call(&h, "GET", &format!("/v1/video/projects/{copy_id}"), None)
            .await
            .1["project"]["global_prompt"],
        ""
    );
    let character = json!({"name":"Own synthetic character","alias":"own","description":"Fixture","tags":["test"],"visual_asset_ids":[asset],"voice":null});
    let(s,saved)=call(&h,"POST",&format!("/v1/video/libraries/{id}/characters"),Some(json!({"schema_version":1,"expected_library_revision":2,"title":"Own character","character":character}))).await;
    assert_eq!(s, 200, "{saved}");
    let char_entry = saved["entry_id"].as_str().unwrap().to_owned();
    let (s, char_copy) = call(
        &h,
        "POST",
        &format!("/v1/video/libraries/{id}/entries/{char_entry}/character-copy"),
        Some(json!({"schema_version":1,"expected_entry_revision":1})),
    )
    .await;
    assert_eq!(s, 200, "{char_copy}");
    let char_id = char_copy["character"]["id"].as_str().unwrap().to_owned();
    let resource = char_copy["character"]["visual_ids"][0].as_str().unwrap();
    let runtime = char_copy["runtime_assets"][resource]["id"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(download(&h, &runtime, None).await, (200, png()));
    let(s,saved)=call(&h,"POST",&format!("/v1/video/libraries/{id}/characters"),Some(json!({"schema_version":1,"expected_library_revision":3,"title":"Updated character","replace":{"entry_id":char_entry,"expected_entry_revision":1},"character":character}))).await;
    assert_eq!(s, 200, "{saved}");
    assert_eq!(
        saved["library"]["entries"][1]["content"]["document"]["id"],
        char_id
    );
    assert_eq!(call(&h,"POST",&format!("/v1/video/libraries/{id}/characters"),Some(json!({"schema_version":1,"expected_library_revision":3,"title":"Stale","character":character}))).await.0,409);
    let copy_folder = root.path().join("copied-project");
    let copy_location = select(&h, &copy_folder).await;
    assert_eq!(call(&h,"POST",&format!("/v1/video/projects/{copy_id}/save"),Some(json!({"schema_version":1,"expected_project_revision":0,"location_id":copy_location}))).await.0,200);
    stop(h).await;
    let h = setup(root.path(), "state2").await;
    let location = select(&h, &folder).await;
    let (s, opened) = call(
        &h,
        "POST",
        "/v1/video/libraries",
        Some(json!({"schema_version":1,"location_id":location,"mode":"open"})),
    )
    .await;
    assert_eq!(s, 200, "{opened}");
    assert_eq!(opened["id"], id);
    assert_eq!(opened["revision"], 4);
    stop(h).await;
    std::fs::remove_dir_all(folder).unwrap();
    let h = setup(root.path(), "state3").await;
    let location = select(&h, &copy_folder).await;
    let (s, opened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":location})),
    )
    .await;
    assert_eq!(s, 200, "{opened}");
    let asset = opened["runtime_assets"]
        .as_object()
        .unwrap()
        .values()
        .next()
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(download(&h, asset, None).await, (200, png()));
    stop(h).await;
}
