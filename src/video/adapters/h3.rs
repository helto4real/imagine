use crate::{
    h3_protocol as wire,
    video::{contract::*, persistence::StorageError, planning},
};
pub mod bindings;
pub mod prompt;
fn unsupported(code: &str) -> StorageError {
    StorageError::new(code)
}
pub fn profile(model: &ModelSelection) -> Result<wire::Profile, StorageError> {
    if model.adapter != "h3" || model.model != "minimax-h3" || model.profile_revision != "2" {
        return Err(unsupported("scene_runtime_unsupported"));
    }
    match (model.variant.as_str(), model.profile.as_str()) {
        ("reference", super::super::job::PROFILE) => Ok(wire::Profile::AvNoReferenceV2),
        ("i2v", super::super::job::I2V_PROFILE) => Ok(wire::Profile::ImageToAvV1),
        _ => Err(unsupported("scene_profile_mismatch")),
    }
}
fn stage(stage: &planning::H3Stage) -> wire::sampling::Stage {
    let manual = !stage.sigmas.is_empty();
    wire::sampling::Stage {
        steps: if manual {
            stage.sigmas.len().saturating_sub(1) as u32
        } else {
            stage.steps
        },
        sampler: stage.sampler.clone(),
        scheduler: if manual {
            "simple".into()
        } else {
            stage.scheduler.clone()
        },
        denoise: if manual { 1. } else { stage.denoise },
        sigmas: stage.sigmas.clone(),
    }
}
pub fn sampling(settings: &planning::H3Settings) -> Result<wire::sampling::Sampling, StorageError> {
    settings.validate()?;
    let second = settings
        .stage_two
        .as_ref()
        .map(|second| -> Result<_, StorageError> {
            Ok(wire::sampling::SecondStage {
                upscale: None,
                model_file: (if second.model.variant == "i2v" {
                    second.i2v_model_file.clone()
                } else {
                    second.reference_model_file.clone()
                })
                .filter(|s| !s.trim().is_empty()),
                sampling: stage(&second.sampling),
                profile: profile(&second.model)?,
                handoff: match second.handoff {
                    planning::H3Handoff::ContinueLatent => wire::sampling::Handoff::ContinueLatent,
                    planning::H3Handoff::ReNoiseVideoPrediction => {
                        wire::sampling::Handoff::ReNoiseVideoPrediction
                    }
                },
            })
        })
        .transpose()?;
    let sampling = wire::sampling::Sampling {
        first: stage(&settings.stage_one),
        base_steps_to_run: settings.base_steps_to_run,
        second,
    };
    if !sampling.valid() {
        return Err(unsupported("unsupported_h3_sampling"));
    }
    Ok(sampling)
}
pub fn options(settings: &planning::H3Settings) -> wire::options::Options {
    wire::options::Options {
        loras: vec![],
        components: wire::options::Components {
            model: settings.components.model.clone(),
            text_encoder: settings.components.text_encoder.clone(),
            video_vae: settings.components.video_vae.clone(),
            audio_vae: settings.components.audio_vae.clone(),
        },
        shift_video: settings.shift_video,
        shift_audio: settings.shift_audio,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manual_sigmas_ignore_steps_scheduler_and_denoise_without_changing_the_curve() {
        let mut settings = imagine_video_contract::generation::h3_defaults();
        settings.stage_one.sigmas = vec![1., 0.5, 0.];
        let expected = sampling(&settings).unwrap();
        settings.stage_one.steps = 0;
        settings.stage_one.scheduler = "not-used-with-sigmas".into();
        settings.stage_one.denoise = 0.;
        let actual = sampling(&settings).unwrap();
        assert_eq!(
            serde_json::to_value(actual).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
        settings.stage_one.sigmas.clear();
        assert!(sampling(&settings).is_err());
    }
}
