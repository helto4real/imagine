use imagine_server::h3_protocol::*;
fn request() -> Request {
    Request {
        capture_continuation: false,
        continuation: None,
        continuation_bytes: vec![],
        references: Default::default(),
        sampling: None,
        protocol: VERSION,
        kind: RequestKind::GenerateVideo,
        id: "synthetic-42".into(),
        profile: Profile::ShortAvNoReferenceV1,
        prompt: "Synthetic rotating cube".into(),
        shape: Default::default(),
        image: None,
        image_rgb: Vec::new(),
        seed: 42,
    }
}
#[test]
fn request_has_no_model_paths_overrides_or_silent_versions() {
    let r = request();
    assert!(r.valid());
    let mut v = serde_json::to_value(&r).unwrap();
    v["model_path"] = serde_json::json!("/private/model");
    assert!(serde_json::from_value::<Request>(v).is_err());
    for (version, seed) in [(VERSION - 1, 42), (VERSION + 1, 42), (VERSION, 43)] {
        let mut r = request();
        r.protocol = version;
        r.seed = seed;
        assert!(!r.valid());
    }
    let mut r = request();
    r.prompt = "x".repeat(MAX_PROMPT_BYTES + 1);
    assert!(!r.valid());
    r.prompt = " ".into();
    assert!(!r.valid());
    r.prompt = "synthetic".into();
    r.id = "private/path".into();
    assert!(!r.valid());
}
#[test]
fn deployment_pins_core_models_and_binary_identity() {
    let mut d = Deployment {
        protocol: VERSION,
        profile: Profile::ShortAvNoReferenceV1,
        core_revision: CORE_REVISION.into(),
        model_lock_sha256: MODEL_LOCK_SHA256.into(),
        worker_sha256: "a".repeat(64),
        dependencies_sha256: "b".repeat(64),
    };
    assert!(d.valid());
    d.core_revision = "unreviewed".into();
    assert!(!d.valid());
    const { assert!(MAX_HEADER < MAX_PAYLOAD) };
    assert_eq!(FPS, 24);
    assert_eq!(SAMPLE_RATE, 32000);
}
#[test]
fn stream_rejects_partial_misordered_nonfinite_and_late_results() {
    let id = "fixture".to_owned();
    let mut s = Stream::new(id.clone()).unwrap();
    let complete = Event::Complete {
        id: id.clone(),
        frames: FRAMES,
        audio_samples: AUDIO_SAMPLES,
    };
    assert!(s.accept(&complete, &[]).is_err());
    assert!(
        s.accept(
            &Event::VideoFrame {
                id: id.clone(),
                index: 1
            },
            &vec![0; FRAME_BYTES]
        )
        .is_err()
    );
    assert!(
        s.accept(
            &Event::AudioChunk {
                id: id.clone(),
                start: 0,
                samples: 1
            },
            &[0; 7]
        )
        .is_err()
    );
    let mut nan = vec![0; 8];
    nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        s.accept(
            &Event::AudioChunk {
                id: id.clone(),
                start: 0,
                samples: 1
            },
            &nan
        )
        .is_err()
    );
    for step in 1..=STEPS {
        s.accept(
            &Event::Progress {
                id: id.clone(),
                completed: step,
                total: STEPS,
                elapsed_seconds: 0.,
            },
            &[],
        )
        .unwrap();
    }
    for index in 0..FRAMES {
        s.accept(
            &Event::VideoFrame {
                id: id.clone(),
                index,
            },
            &vec![0; FRAME_BYTES],
        )
        .unwrap();
    }
    s.accept(
        &Event::AudioChunk {
            id: id.clone(),
            start: 0,
            samples: AUDIO_SAMPLES,
        },
        &vec![0; AUDIO_SAMPLES as usize * 8],
    )
    .unwrap();
    s.accept(&complete, &[]).unwrap();
    assert!(s.accept(&complete, &[]).is_err());
}

#[test]
fn escaped_prompt_fits_the_explicit_stage_header_budget_but_raw_prompt_limit_remains() {
    let mut r = request();
    r.prompt = "\u{0001}".repeat(MAX_PROMPT_BYTES);
    assert!(r.prompt.len() <= MAX_PROMPT_BYTES);
    assert!(serde_json::to_vec(&r).unwrap().len() > 64 * 1024);
    assert!(serde_json::to_vec(&r).unwrap().len() <= MAX_HEADER);
    assert!(r.valid());
    r.prompt.push('x');
    assert!(!r.valid());
    r.prompt = "\u{0001}".repeat(10_000);
    assert!(r.valid());
    r.prompt = format!("a{}", "\n".repeat(MAX_PROMPT_BYTES - 1));
    assert!(r.valid());
}

#[test]
fn viewing_export_preserves_h3_clamp_and_ties_to_even() {
    let values = [
        -1.,
        0.5 / 255.,
        1.5 / 255.,
        2.5 / 255.,
        127.5 / 255.,
        128.5 / 255.,
        254.5 / 255.,
        2.,
    ];
    assert_eq!(
        export_rgb8(&values).unwrap(),
        [0, 0, 2, 2, 128, 128, 254, 255]
    );
    assert!(export_rgb8(&[f32::NAN]).is_err());
    assert!(export_rgb8(&[f32::INFINITY]).is_err());
}
