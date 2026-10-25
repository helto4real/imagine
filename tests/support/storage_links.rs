//! New procedural originals only; no inference, submitted prompts or user files.
use super::*;

#[tokio::test]
async fn linked_original_http_save_reopen_and_relink_never_copies_reference_bytes() {
    let root = tempfile::tempdir().unwrap();
    let original = root.path().join("own original Å.png");
    let replacement = root.path().join("own replacement.png");
    std::fs::write(&original, png()).unwrap();
    std::fs::write(&replacement, png()).unwrap();
    let h = setup(root.path(), "link-session").await;
    let (status, _) = raw(
        &h,
        "POST",
        "/v1/video/assets/link",
        Body::from(json!({"path":original}).to_string()),
        "application/json",
        false,
    )
    .await;
    assert_eq!(status, 401, "Local video session guard remains required");
    let (status, linked) = call(
        &h,
        "POST",
        "/v1/video/assets/link",
        Some(json!({"path":original})),
    )
    .await;
    assert_eq!(status, 201, "{linked}");
    assert_eq!(
        h.store
            .original_path(linked["id"].as_str().unwrap())
            .unwrap(),
        Some(original.clone())
    );
    let (_, created) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let id = created["project"]["id"].as_str().unwrap();
    let base = format!("/v1/video/projects/{id}");
    let (status, attached) = call(
        &h,
        "POST",
        &format!("{base}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":linked["id"]})),
    )
    .await;
    assert_eq!(status, 200, "{attached}");
    assert_eq!(attached["resources"][0]["source"], json!(original));
    let asset = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let folder = root.path().join("saved-project");
    let location = select(&h, &folder).await;
    let (status, saved) = call(
        &h,
        "POST",
        &format!("{base}/save"),
        Some(json!({"schema_version":1,"expected_project_revision":1,"location_id":location})),
    )
    .await;
    assert_eq!(status, 200, "{saved}");
    let digest = attached["resources"][0]["sha256"].as_str().unwrap();
    assert!(!folder.join("media").join(digest).exists());
    assert_eq!(std::fs::read(&original).unwrap(), png());
    stop(h).await;

    let moved = root.path().join("temporarily-away.png");
    std::fs::rename(&original, &moved).unwrap();
    let h = setup(root.path(), "reopen-session").await;
    let location = select(&h, &folder).await;
    let (status, opened) = call(
        &h,
        "POST",
        "/v1/video/projects/open",
        Some(json!({"schema_version":1,"location_id":location})),
    )
    .await;
    assert_eq!(
        status, 200,
        "Missing linked original does not prevent metadata repair: {opened}"
    );
    let (status, health) = call(
        &h,
        "POST",
        &format!("{base}/media/health"),
        Some(json!({"schema_version":1,"expected_project_revision":1})),
    )
    .await;
    assert_eq!(status, 200, "{health}");
    assert_eq!(health["entries"][0]["code"], "source_missing");
    let (_, next) = call(
        &h,
        "POST",
        "/v1/video/assets/link",
        Some(json!({"path":replacement})),
    )
    .await;
    let request = json!({"schema_version":1,"expected_project_revision":1,"asset_id":asset,"expected_asset_revision":1,"replacement_asset_id":next["id"]});
    let (status, updated) = call(&h, "POST", &format!("{base}/media/relink"), Some(request)).await;
    assert_eq!(status, 200, "{updated}");
    let resources = updated["snapshot"]["resources"].as_array().unwrap();
    assert_eq!(resources.last().unwrap()["source"], json!(replacement));
    assert!(!folder.join("media").join(digest).exists());
    assert_eq!(std::fs::read(&replacement).unwrap(), png());
    std::fs::rename(moved, original).unwrap();
    stop(h).await;
}

pub(super) struct OwnVolume(pub(super) std::sync::Mutex<String>);
impl imagine_server::video_destination::Environment for OwnVolume {
    fn mounts(&self) -> imagine_server::video_destination::Result<String> {
        Ok(self.0.lock().unwrap().clone())
    }
    fn host(&self) -> imagine_server::video_destination::Result<String> {
        Ok("own-storage-http".into())
    }
}
pub(super) fn private_volume(
    path: &Path,
) -> (imagine_server::private_storage::Root, Arc<OwnVolume>) {
    use imagine_server::video_destination as d;
    let mount = path.join("own-volume");
    let root = mount.join("imagine");
    std::fs::create_dir_all(&root).unwrap();
    let config = d::Config {
        root,
        mount: mount.clone(),
        source: "own-volume".into(),
        filesystem: "fixturefs".into(),
        fsid: d::fsid(&d::directory(&mount).unwrap()).unwrap(),
    };
    let environment = Arc::new(OwnVolume(std::sync::Mutex::new(format!(
        "1 0 0:1 / {} rw - fixturefs own-volume rw\n",
        mount.display()
    ))));
    std::fs::create_dir(path.join("own-runtime")).unwrap();
    let root = imagine_server::private_storage::Root::with_environment(config, environment.clone())
        .unwrap()
        .with_runtime_directory(path.join("own-runtime"))
        .unwrap();
    (root, environment)
}
pub(super) async fn setup_private(root: &imagine_server::private_storage::Root) -> Harness {
    setup_private_mcp(root, None).await
}
/// A private harness whose MCP root holds agent workspaces.
pub(super) async fn setup_private_mcp(
    root: &imagine_server::private_storage::Root,
    mcp: Option<&Path>,
) -> Harness {
    let mut store = Store::open_private(root.clone(), "own-image-workspace".into())
        .await
        .unwrap();
    store.workspaces = mcp.map(|path| {
        Arc::new(imagine_server::workspaces::Workspaces::open(
            root.rooted_at(path).unwrap(),
        ))
    });
    let (control, scheduler) = scheduler::start(store.clone(), Box::new(NoInference))
        .await
        .unwrap();
    let (_, engine) = watch::channel(EngineState::default());
    let router = api::router(App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open_private(
            root.clone(),
            "library/enhancement-profiles.json".into(),
        )
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
#[tokio::test]
async fn private_http_project_library_originals_and_output_classes_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let (private, environment) = private_volume(directory.path());
    let h = setup_private(&private).await;
    let (status, session) = call(&h, "GET", "/v1/session", None).await;
    assert_eq!(status, 200);
    assert_eq!(session["encrypted"], true);
    let (_, image_proof) = call(&h, "GET", "/v1/media-destination", None).await;
    let (_, video_proof) = call(&h, "GET", "/v1/video-destination", None).await;
    assert_eq!(
        image_proof["config"]["root"],
        json!(
            private
                .layout()
                .project("own-image-workspace")
                .unwrap()
                .images
        )
    );
    assert_eq!(
        video_proof["config"]["root"],
        json!(
            private
                .layout()
                .project("own-image-workspace")
                .unwrap()
                .videos
        )
    );
    let original = directory.path().join("own original.png");
    std::fs::write(&original, png()).unwrap();
    let (status, linked) = call(
        &h,
        "POST",
        "/v1/video/assets/link",
        Some(json!({"path":original})),
    )
    .await;
    assert_eq!(status, 201, "{linked}");
    let (status, created) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    assert_eq!(status, 201, "{created}");
    assert_eq!(created["storage_layout"], 1);
    let id = created["project"]["id"].as_str().unwrap();
    let layout = private.layout().project(id).unwrap();
    assert!(layout.document.is_file());
    let (status, attached) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{id}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":linked["id"]})),
    )
    .await;
    assert_eq!(status, 200, "{attached}");
    assert_eq!(attached["resources"][0]["source"], json!(original));
    assert_eq!(std::fs::read_dir(&layout.images).unwrap().count(), 0);
    let asset = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap();
    let (status, library) = call(
        &h,
        "POST",
        "/v1/video/libraries",
        Some(json!({"schema_version":1,"mode":"create"})),
    )
    .await;
    assert_eq!(status, 200, "{library}");
    let library_id = library["id"].as_str().unwrap();
    let (status,saved)=call(&h,"POST",&format!("/v1/video/libraries/{library_id}/assets"),Some(json!({"schema_version":1,"expected_library_revision":0,"project_id":id,"expected_project_revision":1,"asset_id":asset,"title":"Own linked fixture"}))).await;
    assert_eq!(status, 200, "{saved}");
    assert_eq!(std::fs::read(&original).unwrap(), png());
    assert!(!private.layout().library().join("media").exists());
    let (status, copied) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{id}/save-as"),
        Some(json!({"schema_version":1,"expected_project_revision":1})),
    )
    .await;
    assert_eq!(status, 201, "{copied}");
    assert_ne!(copied["project"]["id"], id);
    assert_eq!(copied["resources"][0]["source"], json!(original));
    let destination = h.store.output_destination(Some(id), false).unwrap();
    let mut image = destination
        .reserve(&json!(destination.proof().unwrap()), 0)
        .unwrap();
    let published = image
        .publish_image(&png(), "image", "image/png", None)
        .unwrap();
    assert_eq!(published.path.parent(), Some(layout.images.as_path()));
    image.commit();
    let video_destination = h.store.output_destination(Some(id), true).unwrap();
    let mut video = video_destination
        .reserve(&json!(video_destination.proof().unwrap()), 0)
        .unwrap();
    // Storage mechanics only: codec correctness is covered by unchanged export tests.
    let published = video.publish(b"OWN STORAGE BYTES").unwrap();
    assert_eq!(published.path.parent(), Some(layout.videos.as_path()));
    video.commit();
    assert!(
        h.store
            .media
            .starts_with(private.layout().project("own-image-workspace").unwrap().tmp)
    );
    let valid = json!(destination.proof().unwrap());
    let mut wrong = valid.clone();
    wrong["config"]["root"] = json!(directory.path().join("outside"));
    assert!(h.store.destination_for_proof(&wrong).is_err());
    let writable = environment.0.lock().unwrap().clone();
    *environment.0.lock().unwrap() = writable.replace(" rw", " ro");
    assert!(destination.reserve(&valid, 0).is_err());
    let (status, failed) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    assert_eq!(status, 422, "{failed}");
    assert!(!directory.path().join("outside").exists());
    // Restore only the own fake mount witness to permit own-session cleanup.
    *environment.0.lock().unwrap() = writable;
    stop(h).await;
}

#[tokio::test]
async fn mcp_link_normalization_preserves_old_exif_pixels_in_ram_without_changing_raw_origin() {
    let directory = tempfile::tempdir().unwrap();
    let (private, _) = private_volume(directory.path());
    let h = setup_private(&private).await;
    let pixels = image::RgbImage::from_fn(9, 5, |x, y| {
        image::Rgb([(x * 20) as u8, (y * 40) as u8, 80])
    });
    let mut jpeg = vec![];
    image::codecs::jpeg::JpegEncoder::new(&mut jpeg)
        .encode(pixels.as_raw(), 9, 5, image::ExtendedColorType::Rgb8)
        .unwrap();
    // Own minimal EXIF TIFF IFD: orientation 6 (90 degrees clockwise).
    let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
    let mut raw = jpeg[..2].to_vec();
    raw.extend_from_slice(b"\xff\xe1");
    raw.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
    raw.extend_from_slice(exif);
    raw.extend_from_slice(&jpeg[2..]);
    let original = directory.path().join("own exif original.jpg");
    std::fs::write(&original, &raw).unwrap();
    let (old, width, height) = st_proxy::references::normalize(&raw).unwrap();
    assert_eq!((width, height), (5, 9));
    let (status, linked) = call(
        &h,
        "POST",
        "/v1/assets/link",
        Some(json!({"path":original,"normalize_image":true})),
    )
    .await;
    assert_eq!(status, 201, "{linked}");
    assert_eq!(linked["width"], width);
    assert_eq!(linked["height"], height);
    assert_eq!(linked["sha256"], imagine_server::storage::hash(&raw));
    assert_eq!(linked["bytes"], raw.len());
    assert_eq!(linked["mime_type"], "image/jpeg");
    let id = linked["id"].as_str().unwrap();
    assert_eq!(h.store.read_asset(id, "local").await.unwrap(), raw);
    assert_eq!(h.store.read_image_asset(id, "local").await.unwrap(), old);
    assert_eq!(std::fs::read(&original).unwrap(), raw);
    let (status, plain) = call(
        &h,
        "POST",
        "/v1/assets/link",
        Some(json!({"path":original})),
    )
    .await;
    assert_eq!(status, 201, "{plain}");
    assert_eq!(plain["width"], 9);
    assert_eq!(
        h.store
            .read_image_asset(plain["id"].as_str().unwrap(), "local")
            .await
            .unwrap(),
        raw
    );
    assert_eq!(std::fs::read_dir(&h.store.media).unwrap().count(), 0);
    raw.push(0);
    std::fs::write(&original, &raw).unwrap();
    assert_eq!(
        h.store
            .read_image_asset(id, "local")
            .await
            .unwrap_err()
            .to_string(),
        "source_changed"
    );
    stop(h).await;
}
