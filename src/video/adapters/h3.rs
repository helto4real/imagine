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
    wire::sampling::Stage {
        steps: stage.steps,
        sampler: stage.sampler.clone(),
        scheduler: stage.scheduler.clone(),
        denoise: stage.denoise,
        sigmas: stage.sigmas.clone(),
    }
}
pub fn sampling(settings: &planning::H3Settings) -> Result<wire::sampling::Sampling, StorageError> {
    settings.validate()?;
    let second = settings
        .stage_two
        .as_ref()
        .map(|second| {
            if second.trained_latent_upscale_model.is_some() {
                return Err(unsupported("h3_trained_upscaler_unavailable"));
            }
            Ok(wire::sampling::SecondStage {
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
