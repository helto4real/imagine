use super::*;
use crate::{
    editing::{self, Edit, EditRequest},
    planning::input_revision,
};
use serde_json::{Value, json};
fn t(n: u64) -> Time {
    Time { num: n, den: 1 }
}
fn project() -> VideoProject {
    let mut p = VideoProject::create(CreateProject {
        generation: None,
        schema_version: 1,
        default_model: ModelSelection {
            adapter: "synthetic".into(),
            model: "synthetic".into(),
            variant: "synthetic".into(),
            profile: "synthetic".into(),
            profile_revision: "1".into(),
        },
        delivery: DeliverySettings {
            fps: t(24),
            width: 64,
            height: 48,
        },
    })
    .unwrap();
    p.scenes.push(Scene {
        id: SceneId("s".into()),
        start: t(0),
        duration: t(6),
        description: String::new(),
        model: ModelChoice::ProjectDefault,
        settings: vec![],
        sections: vec![],
        bindings: vec![],
        continuity: None,
        accepted_take: None,
    });
    p
}
fn apply(p: &mut VideoProject, commands: Vec<Edit>) -> Result<(), Issue> {
    editing::apply(
        p,
        EditRequest {
            schema_version: 1,
            expected_project_revision: p.project_revision,
            commands,
        },
    )
}
fn oracle() -> Value {
    serde_json::from_str(include_str!(
        "../../../assets/video/director-audio-defaults.json"
    ))
    .unwrap()
}
fn seconds(t: Time) -> f64 {
    t.num as f64 / t.den as f64
}
#[test]
fn reference_copy_wording_invalidates_when_the_final_mix_changes() {
    let mut p = project();
    p.assets.insert(
        AssetId("tone".into()),
        Asset {
            library_source: None,
            name: String::new(),
            id: AssetId("tone".into()),
            revision: 1,
            kind: MediaKind::Audio,
        },
    );
    let mut clip = AudioClip::new(
        AudioClipId("tone-clip".into()),
        t(0),
        t(6),
        &Default::default(),
    );
    clip.asset = Some(AssetId("tone".into()));
    clip.usage.generation_reference = true;
    p.audio_tracks.push(AudioTrack {
        id: AudioTrackId("track".into()),
        clips: vec![clip],
    });
    let initial = input_revision(&p, &SceneId("s".into())).unwrap();
    assert_eq!(
        references::entries(&p, &p.scenes[0])[0]["retention_marker"],
        "fully_copy"
    );
    p.audio_tracks[0].clips[0].volume = 80.;
    assert_ne!(input_revision(&p, &SceneId("s".into())).unwrap(), initial);
    assert_eq!(
        references::entries(&p, &p.scenes[0])[0]["retention_marker"],
        "partially_copy"
    );
    p.audio_tracks[0].clips[0].usage.final_mix = false;
    assert_eq!(
        references::entries(&p, &p.scenes[0])[0]["retention_marker"],
        "reference"
    );
}
#[test]
fn legacy_documents_and_exact_director_defaults() {
    let p = project();
    let mut old = serde_json::to_value(&p).unwrap();
    old.as_object_mut().unwrap().remove("audio");
    old.as_object_mut().unwrap().remove("audio_tracks");
    assert_eq!(serde_json::from_value::<VideoProject>(old).unwrap(), p);
    let oracle = oracle();
    let expected: AudioSettings = serde_json::from_value(oracle["project"].clone()).unwrap();
    assert_eq!(AudioSettings::default(), expected);
    let clip = AudioClip::new(AudioClipId("a".into()), t(0), t(1), &expected);
    let raw = &oracle["clip"];
    assert!(clip.asset.is_none() && raw["audio"].is_null());
    for (field, value) in [
        ("start_time", seconds(clip.start)),
        (
            "end_time",
            seconds(clip.start.checked_add(clip.duration).unwrap()),
        ),
        ("source_in", seconds(clip.source_in)),
        ("volume", clip.volume),
        ("fade_in", clip.fade_in),
        ("fade_out", clip.fade_out),
    ] {
        assert_eq!(raw[field].as_f64(), Some(value), "{field}");
    }
    assert!(clip.source_out.is_none() && raw["source_out"].is_null());
    assert_eq!(raw["enabled"].as_bool(), Some(clip.enabled));
    assert_eq!(raw["locked"].as_bool(), Some(clip.locked));
    assert_eq!(raw["name"].as_str(), Some(clip.name.as_str()));
    assert_eq!(raw["lane"].as_u64(), Some(u64::from(clip.lane)));
    assert_eq!(
        clip.normalization,
        serde_json::from_value(raw["normalization"].clone()).unwrap()
    );
    assert_eq!(
        clip.usage,
        serde_json::from_value(raw["usage"].clone()).unwrap()
    );
    assert_eq!(
        clip.reference,
        serde_json::from_value(raw["reference"].clone()).unwrap()
    );
}
#[test]
fn first_fit_lanes_match_director_and_exact_half_open_ranges() {
    let mut clips = vec![];
    for (index, (start, duration)) in [(0, 1), (0, 2), (1, 2), (2, 1)].into_iter().enumerate() {
        clips.push(AudioClip::new(
            AudioClipId(index.to_string()),
            t(start),
            t(duration),
            &Default::default(),
        ));
    }
    clips[1].locked = true;
    let mut tracks = vec![AudioTrack {
        id: AudioTrackId("audio_track_001".into()),
        clips,
    }];
    stack(&mut tracks);
    for (clip, expected) in tracks[0]
        .clips
        .iter()
        .zip(oracle()["stacking"].as_array().unwrap())
    {
        assert_eq!(seconds(clip.start), expected["start"].as_f64().unwrap());
        assert_eq!(
            seconds(clip.start.checked_add(clip.duration).unwrap()),
            expected["end"].as_f64().unwrap()
        );
        assert_eq!(u64::from(clip.lane), expected["lane"].as_u64().unwrap());
    }
    let c = &tracks[0].clips[0];
    assert!(!c.overlaps(t(1), t(1)));
    assert!(c.overlaps(Time { num: 1, den: 2 }, t(1)));
    assert_eq!(t(MAX_INTEGER).checked_add(t(1)), None);
    assert_eq!(
        Time { num: 1, den: 3 }.checked_add(Time { num: 1, den: 6 }),
        Some(Time { num: 1, den: 2 })
    );
}
#[test]
fn invalid_audio_batch_is_atomic_and_final_mix_is_not_a_model_input() {
    let mut p = project();
    let original = p.clone();
    let mut clip = AudioClip::new(AudioClipId("c".into()), t(0), t(2), &p.audio);
    clip.asset = Some(AssetId("missing".into()));
    assert!(
        apply(
            &mut p,
            vec![Edit::AddAudioClip {
                track_id: AudioTrackId("audio_track_001".into()),
                clip: Box::new(clip.clone())
            }]
        )
        .is_err()
    );
    assert_eq!(p, original);
    p.assets.insert(
        AssetId("a".into()),
        Asset {
            library_source: None,
            name: String::new(),
            id: AssetId("a".into()),
            kind: MediaKind::Audio,
            revision: 1,
        },
    );
    clip.asset = Some(AssetId("a".into()));
    let before = input_revision(&p, &SceneId("s".into())).unwrap();
    apply(
        &mut p,
        vec![Edit::AddAudioClip {
            track_id: AudioTrackId("audio_track_001".into()),
            clip: Box::new(clip),
        }],
    )
    .unwrap();
    assert_eq!(before, input_revision(&p, &SceneId("s".into())).unwrap());
    p.audio_tracks[0].clips[0].volume = 15.;
    p.audio_tracks[0].clips[0].usage.final_mix = false;
    assert_eq!(before, input_revision(&p, &SceneId("s".into())).unwrap());
    p.audio_tracks[0].clips[0].usage.generation_reference = true;
    let referenced = input_revision(&p, &SceneId("s".into())).unwrap();
    assert_ne!(before, referenced);
    p.audio_tracks[0].clips[0].reference.strength = 0.5;
    assert_ne!(
        referenced,
        input_revision(&p, &SceneId("s".into())).unwrap()
    );
    let current = input_revision(&p, &SceneId("s".into())).unwrap();
    p.delivery.width = 128;
    assert_ne!(current, input_revision(&p, &SceneId("s".into())).unwrap());
    let mut bad = p.audio_tracks[0].clips[0].clone();
    bad.volume = 401.;
    let original = p.clone();
    assert!(
        apply(
            &mut p,
            vec![Edit::UpdateAudioClip {
                clip: Box::new(bad)
            }]
        )
        .is_err()
    );
    assert_eq!(p, original);
    let text = serde_json::to_string(&p).unwrap();
    assert!(!text.contains("file://"));
    assert_eq!(json!(p.audio_tracks[0].clips[0].asset), json!("a"));
}
