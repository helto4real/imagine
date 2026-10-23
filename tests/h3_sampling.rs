use imagine_server::{
    h3_protocol::{self as p, sampling::*},
    video::{adapters::h3, contract::ModelSelection, planning::H3Settings},
};
use serde_json::json;
fn model() -> ModelSelection {
    ModelSelection {
        adapter: "h3".into(),
        model: "minimax-h3".into(),
        variant: "reference".into(),
        profile: "h3-av-noref-v2".into(),
        profile_revision: "2".into(),
    }
}
fn settings() -> H3Settings {
    serde_json::from_value(json!({"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":{"sampling":{"steps":8,"sampler":"euler","scheduler":"simple"},"model":model(),"handoff":"continue_latent","trained_latent_upscale_model":null}})).unwrap()
}
#[test]
fn explicit_model_stage_defaults_and_progress_count_are_preserved() {
    let settings = settings();
    let sampling = h3::sampling(&settings).unwrap();
    assert_eq!(sampling.base_steps_to_run, 4);
    assert_eq!(sampling.total_steps(), 12);
    assert_eq!(sampling.first.denoise, 1.);
    assert_eq!(
        sampling.second.unwrap().profile,
        p::Profile::AvNoReferenceV2
    );
    assert!(h3::profile(&model()).is_ok());
    let mut invalid = model();
    invalid.variant = "i2v".into();
    assert_eq!(
        h3::profile(&invalid).unwrap_err().code,
        "scene_profile_mismatch"
    );
}
#[test]
fn inactive_stage_two_editor_state_does_not_change_worker_sampling() {
    let mut settings = settings();
    settings.stage_two = None;
    let before = h3::sampling(&settings).unwrap();
    settings.stage_two_draft = Some(imagine_server::video::planning::H3StageTwoDraft {
        settings: superfluous_second(),
        raw_sigmas: "0.4, unfinished".into(),
    });
    let after = h3::sampling(&settings).unwrap();
    assert_eq!(
        serde_json::to_value(before).unwrap(),
        serde_json::to_value(after).unwrap()
    );
}
fn superfluous_second() -> imagine_server::video::planning::H3SecondStage {
    settings().stage_two.unwrap()
}
#[test]
fn manual_sigmas_override_base_count_but_both_handoff_endpoints_must_match() {
    let mut settings = settings();
    settings.stage_one.sigmas = vec![1., 0.8, 0.6];
    settings.stage_two.as_mut().unwrap().sampling.sigmas = vec![0.6, 0.3, 0.];
    settings.base_steps_to_run = 0;
    let sampling = h3::sampling(&settings).unwrap();
    assert_eq!(sampling.total_steps(), 4);
    settings.stage_two.as_mut().unwrap().sampling.sigmas[0] = 0.5;
    assert!(h3::sampling(&settings).is_err());
    settings.stage_two.as_mut().unwrap().sampling.sigmas[0] = 0.6;
    settings.stage_one.sigmas[2] = 0.;
    assert!(h3::sampling(&settings).is_err());
}
#[test]
fn explicit_sigmas_follow_director_instead_of_unused_generated_schedule_limits() {
    let mut settings = settings();
    settings.stage_one.sigmas = vec![1.2, 0.6];
    settings.stage_one.steps = 0;
    settings.stage_one.denoise = -1.;
    settings.stage_one.scheduler = "unused manual curve".into();
    let second = &mut settings.stage_two.as_mut().unwrap().sampling;
    second.sigmas = vec![0.6, 0.2, 0.];
    second.steps = 0;
    second.denoise = -1.;
    second.scheduler = "also unused".into();
    settings.base_steps_to_run = 0;
    settings.validate().unwrap();
    let sampling = h3::sampling(&settings).unwrap();
    assert_eq!(sampling.total_steps(), 3);
    assert_eq!(sampling.first.sigmas, [1.2, 0.6]);
    settings.stage_one.sigmas[0] = f32::NAN;
    assert!(h3::sampling(&settings).is_err());
    settings.stage_one.sigmas = vec![0.2, 0.6];
    assert!(h3::sampling(&settings).is_err());
}
#[test]
fn sampler_choice_reaches_runtime_without_preflight_name_gates_or_model_reduction() {
    let mut settings = settings();
    settings
        .stage_two
        .as_mut()
        .unwrap()
        .trained_latent_upscale_model = Some("explicit-model".into());
    assert!(h3::sampling(&settings).is_ok()); // inactive file remains a user choice
    settings
        .stage_two
        .as_mut()
        .unwrap()
        .trained_latent_upscale_model = None;
    settings.stage_one.sampler = "unbound-stochastic-sampler".into();
    let sampling = h3::sampling(&settings).unwrap();
    assert_eq!(sampling.first.sampler, "unbound-stochastic-sampler");
    // The executable registry reports an unknown method; admission must neither
    // silently substitute res_multistep nor block additional compatible methods.
    settings.stage_one.sampler.clear();
    assert!(h3::sampling(&settings).is_err());
    settings.stage_one.sampler = "res_multistep".into();
    settings.stage_two.as_mut().unwrap().model.profile_revision = "unknown".into();
    assert!(h3::sampling(&settings).is_err());
}
#[test]
fn trained_upscale_transport_preserves_base_and_final_canvas_without_model_name_gates() {
    let mut sampling = h3::sampling(&settings()).unwrap();
    let second = sampling.second.as_mut().unwrap();
    second.handoff = Handoff::ReNoiseVideoPrediction;
    second.upscale = Some(Upscale {
        factor: 1.25,
        base_width: 416,
        base_height: 224,
        model: Some("/own/compatible alternative.weights".into()),
    });
    let output = p::Shape {
        width: 512,
        height: 288,
        ..p::Shape::default()
    };
    assert!(sampling.valid_shape(output));
    assert!(sampling.valid());
    sampling.second.as_mut().unwrap().handoff = Handoff::ContinueLatent;
    assert!(!sampling.valid()); // An executable upscale must not be silently ignored.
    sampling.second.as_mut().unwrap().handoff = Handoff::ReNoiseVideoPrediction;
    assert_eq!(
        sampling.input_shape(output),
        p::Shape {
            width: 416,
            height: 224,
            ..p::Shape::default()
        }
    );
    let copy: Sampling = serde_json::from_value(serde_json::to_value(&sampling).unwrap()).unwrap();
    assert_eq!(
        copy.second.unwrap().upscale.unwrap().model.as_deref(),
        Some("/own/compatible alternative.weights")
    );
    sampling
        .second
        .as_mut()
        .unwrap()
        .upscale
        .as_mut()
        .unwrap()
        .factor = f64::NAN;
    assert!(!sampling.valid());
}
#[test]
fn nonfinite_unbounded_or_nonmonotone_schedules_are_rejected_before_native_admission() {
    let mut sampling = Sampling::default();
    for values in [
        vec![1., 0.5, 0.7, 0.],
        vec![1., f32::NAN, 0.],
        vec![1., 1.],
        vec![1., -0.1],
        vec![f32::INFINITY, 0.],
    ] {
        sampling.first.sigmas = values;
        assert!(!sampling.valid());
    }
    sampling.first.sigmas = vec![];
    sampling.first.denoise = 0.00001;
    assert!(!sampling.valid());
}
#[test]
fn progress_is_one_monotone_sequence_across_both_native_stages() {
    let mut stream =
        p::Stream::with_steps("stage-fixture".into(), p::Shape::default(), 12).unwrap();
    let event = |completed, total| p::Event::Progress {
        id: "stage-fixture".into(),
        completed,
        total,
        elapsed_seconds: 1.,
    };
    assert!(stream.accept(&event(1, 20), &[]).is_ok());
    assert!(stream.accept(&event(4, 12), &[]).is_ok());
    assert!(stream.accept(&event(1, 12), &[]).is_err());
    assert!(stream.accept(&event(5, 12), &[]).is_ok());
    assert!(stream.accept(&event(12, 12), &[]).is_ok());
    assert!(stream.accept(&event(13, 12), &[]).is_err());
}
#[test]
fn adaptive_progress_can_expand_and_correct_totals_without_extra_model_callbacks() {
    let mut stream =
        p::Stream::with_steps("adaptive-fixture".into(), p::Shape::default(), 4).unwrap();
    let event = |completed, total| p::Event::Progress {
        id: "adaptive-fixture".into(),
        completed,
        total,
        elapsed_seconds: 1.,
    };
    for step in 1..=6 {
        stream.accept(&event(step, step.max(4)), &[]).unwrap();
    }
    assert!(stream.accept(&event(5, 6), &[]).is_err());
    assert!(stream.accept(&event(6, 6), &[]).is_err());
    stream.accept(&event(7, 10), &[]).unwrap();
    stream.accept(&event(8, 10), &[]).unwrap();
    stream.accept(&event(8, 8), &[]).unwrap();
    assert!(stream.accept(&event(8, 7), &[]).is_err());
}
