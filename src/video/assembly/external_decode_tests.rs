//! Procedural CPU sources only: exercise the real freeze and AV decode paths.
use super::*;
use crate::video::{contract::*, persistence::*, service::Service};
use std::{
    collections::BTreeMap,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
    sync::Arc,
};

fn fixture() -> (tempfile::TempDir, Frozen, PathBuf, PathBuf) {
    let root =
        tempfile::tempdir_in(crate::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES")).unwrap();
    let video = root.path().join("own-procedural.mp4");
    let audio = root.path().join("own-procedural.wav");
    let frames = 5;
    let pcm = (0..9188)
        .flat_map(|i| {
            let sample = 0.08 * (std::f32::consts::TAU * 440. * i as f32 / 44100.).sin();
            [sample, -sample]
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    let bytes = crate::short_av::encode_stream(
        crate::short_av::EncodeSpec {
            width: 16,
            height: 16,
            frames,
            fps_num: 24,
            fps_den: 1,
            sample_rate: 44100,
            bit_depth: 8,
        },
        (0..frames).map(|i| Ok(vec![40 + i as u8 * 20; 16 * 16 * 3])),
        pcm,
        &AtomicBool::new(false),
    )
    .unwrap();
    std::fs::write(&video, bytes).unwrap();
    let samples = (0..9188)
        .map(|i| ((i % 100) as i16 - 50) * 100)
        .flat_map(i16::to_le_bytes)
        .collect::<Vec<_>>();
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + samples.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&44100_u32.to_le_bytes());
    wav.extend_from_slice(&88200_u32.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    wav.extend_from_slice(&samples);
    std::fs::write(&audio, wav).unwrap();
    let mut project = VideoProject::create(
        serde_json::from_str(include_str!(
            "../../../design/video-contract/create-project.json"
        ))
        .unwrap(),
    )
    .unwrap();
    project.delivery = DeliverySettings {
        width: 16,
        height: 16,
        fps: Time { num: 24, den: 1 },
    };
    let duration = Time { num: 5, den: 24 };
    let scene = SceneId("own-scene".into());
    project.scenes.push(Scene {
        id: scene.clone(),
        start: time(0.),
        duration,
        description: String::new(),
        model: ModelChoice::ProjectDefault,
        settings: vec![],
        sections: vec![],
        bindings: vec![],
        continuity: None,
        accepted_take: None,
    });
    let mut resources = vec![];
    for (id, path, kind) in [
        ("video", &video, MediaKind::Video),
        ("audio", &audio, MediaKind::Audio),
    ] {
        let asset_id = AssetId(id.into());
        project.assets.insert(
            asset_id.clone(),
            Asset {
                id: asset_id.clone(),
                revision: 1,
                kind,
                name: String::new(),
                library_source: None,
            },
        );
        resources.push(
            Resource::external(
                id.into(),
                ResourceKind::Reference {
                    asset_id,
                    revision: 1,
                },
                path,
                &mut File::open(path).unwrap(),
            )
            .unwrap(),
        );
    }
    project.assembly.clips.insert(
        scene,
        imagine_video_contract::assembly::ClipInstance {
            asset: AssetId("video".into()),
            source_in: time(0.),
            source_out: None,
            speed: 1.,
            include_audio: true,
        },
    );
    let mut clip = imagine_video_contract::audio::AudioClip::new(
        AudioClipId("own-audio".into()),
        time(0.),
        duration,
        &project.audio,
    );
    clip.asset = Some(AssetId("audio".into()));
    project
        .audio_tracks
        .push(imagine_video_contract::audio::AudioTrack {
            id: AudioTrackId("own-track".into()),
            clips: vec![clip],
        });
    let mut session = ProjectSession::new(project).unwrap();
    let mut document = session.document().clone();
    document.resources = resources;
    let staging = SessionStaging::select(root.path()).unwrap();
    session
        .update(0, document, BTreeMap::new(), &staging)
        .unwrap();
    let request = Request {
        schema_version: 1,
        project_id: session.document().project.id.clone(),
        expected_project_revision: session.document().project.project_revision,
        options: Options {
            normalize_mix: false,
            ..Default::default()
        },
    };
    let mut service = Service::default();
    service.projects.insert(session).unwrap();
    let frozen = freeze(&service, &request, &staging).unwrap();
    (root, frozen, video, audio)
}

fn mutate(path: &Path) {
    use std::os::unix::fs::FileExt;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let offset = file.metadata().unwrap().len() - 1;
    let mut byte = [0];
    file.read_at(&mut byte, offset).unwrap();
    file.write_all_at(&[byte[0] ^ 1], offset).unwrap();
}

#[test]
fn external_final_decode_preserves_original_inodes_and_raw_video_audio() {
    let (_root, frozen, video, audio) = fixture();
    assert_eq!(frozen.external_sources.len(), 2);
    for (id, path) in [("video", video), ("audio", audio)] {
        let held = frozen.files[id].metadata().unwrap();
        let original = std::fs::metadata(path).unwrap();
        assert_eq!((held.dev(), held.ino()), (original.dev(), original.ino()));
    }
    let mut same_descriptors_without_origin_guards = frozen.try_clone().unwrap();
    assert_eq!(
        same_descriptors_without_origin_guards.external_sources,
        frozen.external_sources
    );
    same_descriptors_without_origin_guards
        .external_sources
        .clear();
    let guarded = prepare_media(frozen, Arc::default(), Progress::default()).unwrap();
    let unchanged = prepare_media(
        same_descriptors_without_origin_guards,
        Arc::default(),
        Progress::default(),
    )
    .unwrap();
    assert_eq!(guarded.pcm, unchanged.pcm);
    let a = guarded.frames.collect::<Result<Vec<_>, _>>().unwrap();
    let b = unchanged.frames.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 5);
}

#[test]
fn external_final_video_and_audio_reject_changed_source_after_freeze() {
    for id in ["video", "audio"] {
        let (_root, mut frozen, video, audio) = fixture();
        mutate(if id == "video" { &video } else { &audio });
        if id == "video" {
            // Isolate picture decoding from soundtrack decoding in prepare_media.
            frozen.plan.options.audio_mode = MixMode::Timeline;
            let mut media = prepare_media(frozen, Arc::default(), Progress::default()).unwrap();
            assert_eq!(
                media.frames.next().unwrap().unwrap_err().to_string(),
                "source_changed"
            );
        } else {
            assert!(matches!(
                prepare_media(frozen, Arc::default(), Progress::default()),
                Err("source_changed")
            ));
        }
    }
}

#[test]
fn external_final_video_and_audio_reject_mutation_during_decode() {
    for id in ["video", "audio"] {
        let (_root, frozen, video, audio) = fixture();
        let source = if id == "video" {
            &frozen.plan.clips[0].source
        } else {
            &frozen.plan.audio[0].source
        };
        let result = decode_source(&frozen, source, |file| {
            // Deterministically mutate after a successful real AV decode and
            // before its witness check, without timing-dependent writer races.
            if id == "video" {
                assert!(
                    !decode_video(file, &frozen.plan.clips[0], &AtomicBool::new(false))?.is_empty()
                );
            } else {
                assert!(!media::audio(file, None, &AtomicBool::new(false))?.is_empty());
            }
            mutate(if id == "video" { &video } else { &audio });
            Ok(())
        });
        assert_eq!(result, Err("source_changed"));
    }
}
