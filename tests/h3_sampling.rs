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
fn unsupported_methods_and_model_resources_fail_closed_without_silent_reduction() {
    let mut settings = settings();
    settings
        .stage_two
        .as_mut()
        .unwrap()
        .trained_latent_upscale_model = Some("explicit-model".into());
    assert_eq!(
        h3::sampling(&settings).unwrap_err().code,
        "h3_trained_upscaler_unavailable"
    );
    settings
        .stage_two
        .as_mut()
        .unwrap()
        .trained_latent_upscale_model = None;
    settings.stage_one.sampler = "unbound-stochastic-sampler".into();
    assert_eq!(
        h3::sampling(&settings).unwrap_err().code,
        "unsupported_h3_sampling"
    );
    settings.stage_one.sampler = "res_multistep".into();
    settings.stage_two.as_mut().unwrap().model.profile_revision = "unknown".into();
    assert!(h3::sampling(&settings).is_err());
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
    assert!(stream.accept(&event(1, 20), &[]).is_err());
    assert!(stream.accept(&event(4, 12), &[]).is_ok());
    assert!(stream.accept(&event(1, 12), &[]).is_err());
    assert!(stream.accept(&event(5, 12), &[]).is_ok());
    assert!(stream.accept(&event(12, 12), &[]).is_ok());
    assert!(stream.accept(&event(13, 12), &[]).is_err());
}
