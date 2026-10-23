use imagine_server::h3_protocol::{self as h3, Selection, Shape};
#[test]
fn matches_actual_director_projection_for_all_presets_ratios_orientations_and_durations() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/h3-director-geometry.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let selected: Selection = serde_json::from_value(case["selection"].clone()).unwrap();
        let expected: Shape = serde_json::from_value(case["shape"].clone()).unwrap();
        assert_eq!(selected.resolve().unwrap(), expected, "{selected:?}");
    }
    assert_eq!(Selection::default().resolve().unwrap(), Shape::default());
}
#[test]
fn rejects_invalid_shapes_and_memory_requirements_before_worker_admission() {
    for shape in [
        Shape {
            width: u32::MAX,
            ..Default::default()
        },
        Shape {
            height: 257,
            ..Default::default()
        },
        Shape {
            frames: 40,
            ..Default::default()
        },
        Shape {
            fps: 30,
            ..Default::default()
        },
        Shape {
            frames: 86405,
            ..Default::default()
        },
    ] {
        assert!(!shape.valid());
    }
    for seconds in [f64::NAN, f64::INFINITY, -1., 0., 3601.] {
        assert!(
            Selection {
                seconds,
                ..Default::default()
            }
            .resolve()
            .is_err()
        );
    }
    assert!(
        Selection {
            ratio: "unknown".into(),
            ..Default::default()
        }
        .resolve()
        .is_err()
    );
    assert!(
        Selection {
            quality: "unknown".into(),
            ..Default::default()
        }
        .resolve()
        .is_err()
    );
}
#[test]
fn temporal_grid_and_audio_export_match_native_latent_rate_without_changing_default() {
    for k in 0..22 {
        let shape = Shape {
            frames: 17 * k + 5,
            ..Default::default()
        };
        assert!(shape.valid());
        assert_eq!(shape.video_latent_t(), i64::from(k * 5 + 2));
        let raw = vec![0.25; shape.native_audio_samples() as usize * 2];
        let output = shape.fit_audio(raw.clone()).unwrap();
        assert_eq!(output.len(), shape.audio_samples() as usize * 2);
        assert_eq!(
            &output[..output.len().min(raw.len())],
            &raw[..output.len().min(raw.len())]
        );
        assert!((shape.audio_samples() as f64 / 32000. - shape.seconds()).abs() <= 0.5 / 32000.);
        assert!(shape.native_audio_samples().abs_diff(shape.audio_samples()) <= 267);
    }
    let shape = Shape::default();
    assert_eq!(shape.native_audio_samples(), h3::AUDIO_SAMPLES);
}
#[test]
fn nondefault_geometry_survives_mp4_and_rejects_wrong_size() {
    let shape = Shape {
        width: 64,
        height: 96,
        frames: 22,
        fps: 24,
    };
    let av = imagine_server::short_av::Av {
        continuation: None,
        shape,
        frames: vec![vec![91; shape.frame_bytes()]; shape.frames as usize],
        pcm: vec![0; shape.audio_samples() as usize * 8],
    };
    let bytes = imagine_server::short_av::encode(av).unwrap();
    let decoded = imagine_server::short_av::decode(&bytes).unwrap();
    assert_eq!(decoded.shape, shape);
    assert!(decoded.valid());
    assert_eq!(decoded.frames.len(), 22);
    assert_eq!(decoded.pcm.len(), 29333 * 8);
}
