//! Owned synthetic media only. Set IMAGINE_WORKER_A_FIXTURES inside the approved task root.
#[path = "../shared/task_fixtures.rs"]
mod task_fixtures;
use imagine_server::video::{contract::*, media_health::*, persistence::*};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Cursor, Write},
    path::PathBuf,
    sync::atomic::AtomicBool,
};
fn root() -> PathBuf {
    let root = task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES");
    let path = root.join(format!("health-{}", uuid::Uuid::new_v4().simple()));
    fs::create_dir(&path).unwrap();
    path
}
fn png(root: &std::path::Path, value: u8) -> File {
    let mut bytes = Cursor::new(vec![]);
    image::RgbImage::from_pixel(8, 8, image::Rgb([value, 0, 0]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let path = root.join(format!("image-{}.png", uuid::Uuid::new_v4().simple()));
    let mut file = File::create(&path).unwrap();
    file.write_all(bytes.get_ref()).unwrap();
    drop(file);
    File::open(path).unwrap()
}
fn session(root: &std::path::Path) -> (ProjectSession, SessionStaging) {
    let staging = SessionStaging::select(root).unwrap();
    let project = VideoProject::create(
        serde_json::from_str(include_str!("../design/video-contract/create-project.json")).unwrap(),
    )
    .unwrap();
    let mut session = ProjectSession::new(project).unwrap();
    let mut document = session.document().clone();
    let asset = AssetId("stable".into());
    document.project.assets.insert(
        asset.clone(),
        Asset {
            library_source: None,
            name: "Synthetic".into(),
            id: asset.clone(),
            revision: 1,
            kind: MediaKind::Image,
        },
    );
    document.project.scenes.push(Scene {
        id: SceneId("scene".into()),
        start: Time { num: 0, den: 1 },
        duration: Time { num: 1, den: 1 },
        description: String::new(),
        model: ModelChoice::ProjectDefault,
        settings: vec![],
        sections: vec![],
        bindings: vec![Binding {
            id: BindingId("binding".into()),
            asset: asset.clone(),
            role: InputRole::VisualReference,
            enabled: false,
            video: None,
        }],
        continuity: None,
        accepted_take: None,
    });
    let mut file = png(root, 30);
    let resource = Resource::inspect(
        "original".into(),
        ResourceKind::Reference {
            asset_id: asset,
            revision: 1,
        },
        &mut file,
    )
    .unwrap();
    document.resources.push(resource.clone());
    session
        .update(
            0,
            document,
            BTreeMap::from([(resource.key(), file)]),
            &staging,
        )
        .unwrap();
    (session, staging)
}
fn request(session: &ProjectSession) -> Relink {
    Relink {
        schema_version: 1,
        expected_project_revision: session.document().project.project_revision,
        asset_id: AssetId("stable".into()),
        expected_asset_revision: session.document().project.assets[&AssetId("stable".into())]
            .revision,
        replacement_asset_id: "synthetic-upload".into(),
    }
}
#[test]
fn valid_stereo_float_audio_health_uses_pcm_sample_budget() {
    let root = root();
    let frames = 4410_u32;
    let bytes = frames * 8;
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + bytes).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&3_u16.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&44100_u32.to_le_bytes());
    wav.extend_from_slice(&352800_u32.to_le_bytes());
    wav.extend_from_slice(&8_u16.to_le_bytes());
    wav.extend_from_slice(&32_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&bytes.to_le_bytes());
    for _ in 0..frames {
        wav.extend_from_slice(&0.125_f32.to_le_bytes());
        wav.extend_from_slice(&(-0.25_f32).to_le_bytes());
    }
    let path = root.join("own-audio.wav");
    fs::write(&path, wav).unwrap();
    let info = validate(
        &File::open(path).unwrap(),
        &MediaKind::Audio,
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(info.sample_rate, Some(44100));
    assert_eq!(info.channels, Some(2));
}
#[test]
fn media_health_check_relink_preserves_identity_bindings_and_reopen() {
    let root = root();
    let (mut session, staging) = session(&root);
    let report = Check::capture(&session, 1)
        .unwrap()
        .run(&AtomicBool::new(false))
        .unwrap();
    assert_eq!(report.entries[0].status, "ok");
    let before = session.document().project.scenes.clone();
    let request = request(&session);
    let mut replacement = png(&root, 70);
    validate(&replacement, &MediaKind::Image, &AtomicBool::new(false)).unwrap();
    let preview = prepare_relink(&session, &request, &mut replacement, &MediaKind::Image).unwrap();
    assert_eq!(session.document().project.project_revision, 1);
    assert_eq!(preview.2[0].binding, "binding");
    apply_relink(
        &mut session,
        &request,
        replacement,
        &MediaKind::Image,
        &staging,
    )
    .unwrap();
    assert_eq!(session.document().project.scenes, before);
    assert_eq!(
        session.document().project.assets[&AssetId("stable".into())].revision,
        2
    );
    assert_eq!(session.document().resources.len(), 2);
    assert!(
        apply_relink(
            &mut session,
            &request,
            png(&root, 90),
            &MediaKind::Image,
            &staging
        )
        .is_err()
    );
    let saved = root.join("project");
    fs::create_dir(&saved).unwrap();
    session.save_to(&saved).unwrap();
    let expected = session.document().clone();
    drop(session);
    let reopened = ProjectSession::open(&saved).unwrap();
    assert_eq!(reopened.document(), &expected);
    assert_eq!(
        Check::capture(&reopened, 2)
            .unwrap()
            .run(&AtomicBool::new(false))
            .unwrap()
            .entries[0]
            .status,
        "ok"
    );
}
#[test]
fn media_health_wrong_type_cancel_and_revision_failure_do_not_mutate() {
    let root = root();
    let (mut session, staging) = session(&root);
    let before = session.document().clone();
    let mut request = request(&session);
    assert!(validate(&png(&root, 1), &MediaKind::Video, &AtomicBool::new(false)).is_err());
    assert!(validate(&png(&root, 1), &MediaKind::Image, &AtomicBool::new(true)).is_err());
    assert!(Check::capture(&session, 0).is_err());
    assert!(
        Check::capture(&session, 1)
            .unwrap()
            .run(&AtomicBool::new(true))
            .is_err()
    );
    request.expected_asset_revision = 0;
    assert!(
        apply_relink(
            &mut session,
            &request,
            png(&root, 90),
            &MediaKind::Image,
            &staging
        )
        .is_err()
    );
    assert_eq!(session.document(), &before);
}
