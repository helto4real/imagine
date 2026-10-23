use imagine_server::h3_protocol::{self as p, continuation as c};
#[test]
fn continuation_audio_fits_the_sampled_grid_before_visible_trim() {
    let visible = p::Shape {
        width: 416,
        height: 224,
        frames: 124,
        fps: 24,
    };
    for (method, start) in [
        (c::Method::MotionContext, 45333),
        (c::Method::MotionContextExp, 28000),
    ] {
        let (sampled, head, _) = c::runtime(visible, method).unwrap();
        let input: Vec<f32> = (0..sampled.native_audio_samples() * 2)
            .map(|i| (i as f32 - 200_000.) / 1_048_576.)
            .collect();
        assert_eq!(sampled.native_audio_samples(), 210400);
        assert!(
            visible.fit_audio(input.clone()).is_err(),
            "old visible-grid check must reject this valid sampled buffer"
        );
        let mut expected = input.clone();
        expected.resize(sampled.audio_samples() as usize * 2, 0.);
        let end = (start + visible.audio_samples() as usize) * 2;
        let actual = c::trim_audio(sampled, visible, head, input.clone()).unwrap();
        assert_eq!(actual, expected[start * 2..end]);
        assert_eq!(actual.len(), 165333 * 2);
        assert_eq!(actual[0], input[start * 2]);
        assert_eq!(actual[1], input[start * 2 + 1]);
        if method == c::Method::MotionContext {
            assert!(actual[actual.len() - 532..].iter().all(|v| *v == 0.));
        }
        assert!(c::trim_audio(sampled, visible, 35, input.clone()).is_err());
        assert!(c::trim_audio(sampled, visible, head, input[..input.len() - 2].to_vec()).is_err());
        let mut invalid = input;
        invalid[0] = f32::NAN;
        assert!(c::trim_audio(sampled, visible, head, invalid).is_err());
    }
    let input = vec![0.125; visible.native_audio_samples() as usize * 2];
    assert_eq!(
        c::trim_audio(visible, visible, 0, input.clone()).unwrap(),
        visible.fit_audio(input).unwrap()
    );
    let wrong_geometry = p::Shape {
        width: 448,
        ..visible
    };
    assert!(c::trim_audio(visible, wrong_geometry, 0, vec![]).is_err());
}
#[test]
fn native_tail_descriptor_and_runtime_match_director_grid_and_trim() {
    let shape = p::Shape {
        width: 416,
        height: 224,
        frames: 124,
        fps: 24,
    };
    assert_eq!(c::endpoint(37, 124), (37, 124));
    assert_eq!(c::endpoint(47, 145), (43, 145));
    for (method, head, guide) in [
        (c::Method::MotionContext, 34, 14),
        (c::Method::MotionContextExp, 21, 0),
    ] {
        let (sampled, h, g) = c::runtime(shape, method).unwrap();
        assert_eq!((sampled.frames, h, g), (158, head, guide));
        let d = c::Descriptor {
            schema_version: 1,
            shape,
            sampled_frames: sampled.frames,
            visible_end_frame: head + 124,
            selected_end_frame: head + 124,
        };
        assert!(d.valid());
        let mut bytes = vec![0; d.bytes()];
        assert!(d.valid_bytes(&bytes));
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(!d.valid_bytes(&bytes));
        let mut bad = d.clone();
        bad.selected_end_frame -= 1;
        assert!(!bad.valid());
        bad = d.clone();
        bad.schema_version = 2;
        assert!(!bad.valid());
        assert!(!d.valid_bytes(&[]));
    }
    let weights = c::step_weights(47, 21);
    assert_eq!(weights[0], 1.);
    assert_eq!(weights[46], 0.);
    assert!(weights.iter().any(|w| *w > 0. && *w < 1.));
    assert!(weights.windows(2).all(|v| v[0] >= v[1]));
}
#[test]
fn continuation_event_is_bounded_finite_single_and_at_end_of_sampling() {
    let shape = p::Shape {
        width: 32,
        height: 32,
        frames: 39,
        fps: 24,
    };
    let descriptor = c::Descriptor {
        schema_version: 1,
        shape,
        sampled_frames: 39,
        visible_end_frame: 39,
        selected_end_frame: 39,
    };
    let bytes = vec![0; descriptor.bytes()];
    let event = p::Event::Continuation {
        id: "own_stream".into(),
        descriptor,
    };
    let mut stream = p::Stream::with_shape("own_stream".into(), shape).unwrap();
    assert!(stream.accept(&event, &bytes).is_err());
    stream
        .accept(
            &p::Event::Progress {
                id: "own_stream".into(),
                completed: 20,
                total: 20,
                elapsed_seconds: 1.,
            },
            &[],
        )
        .unwrap();
    let mut bad = bytes.clone();
    bad[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(stream.accept(&event, &bad).is_err());
    stream.accept(&event, &bytes).unwrap();
    assert!(stream.captured_continuation());
    assert!(stream.accept(&event, &bytes).is_err());
}
