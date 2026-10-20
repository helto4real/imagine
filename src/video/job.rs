//! H3 geometry with frozen sampling settings; no image defaults reach inference.
use crate::{
    contract::{JobRequest, ModelOptions, Operation},
    h3_protocol as h3,
};
pub const PROFILE: &str = "h3-av-noref-v2";
pub fn request(job: &JobRequest) -> Result<h3::Request, &'static str> {
    let v = job.video.as_ref().ok_or("h3_profile_required")?;
    let shape = h3::Shape {
        width: job.width,
        height: job.height,
        frames: v.frames,
        fps: v.fps,
    };
    let profile = match v.profile.as_deref() {
        Some(PROFILE) => h3::Profile::AvNoReferenceV2,
        Some("h3-short-av-noref-v1") if shape == h3::Shape::default() => {
            h3::Profile::ShortAvNoReferenceV1
        }
        _ => return Err("unsupported_h3_profile"),
    };
    if job.model != "minimax-h3"
        || job.operation != Operation::TextToVideo
        || !shape.valid()
        || !v.audio
        || job.seed != Some(42)
        || job.sampling.steps != 4
        || job.sampling.cfg_scale != 1.
        || job.sampling.sampler != "euler"
        || job.sampling.scheduler != "simple"
        || !job.sampling.custom_sigmas.is_empty()
        || job.output.format != "mkv"
        || job.output.count != 1
    {
        return Err("unsupported_h3_profile");
    }
    if !job.negative_prompt.is_empty()
        || job.character_sheet.is_some()
        || job.asset_ids().next().is_some()
        || job.inputs.start_image.is_some()
        || job.inputs.end_image.is_some()
        || job.inputs.mask.is_some()
        || job.resolved_models.is_some()
        || job.resolved_profile.is_some()
        || serde_json::to_value(&job.model_options).ok()
            != serde_json::to_value(ModelOptions::default()).ok()
    {
        return Err("unsupported_h3_input");
    }
    if serde_json::to_vec(&job.client_metadata).map_or(true, |v| v.len() > 4096) {
        return Err("metadata_too_large");
    }
    let mut r = h3::Request {
        protocol: h3::VERSION,
        kind: h3::RequestKind::GenerateVideo,
        id: "x".repeat(128),
        profile,
        shape,
        prompt: job.prompt.clone(),
        seed: 42,
    };
    if !r.valid() {
        return Err("invalid_h3_request");
    }
    r.id = uuid::Uuid::new_v4().to_string();
    Ok(r)
}
