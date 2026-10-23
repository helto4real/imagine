//! H3 geometry with frozen sampling settings; no image defaults reach inference.
use crate::{
    contract::{JobRequest, ModelOptions, Operation},
    h3_protocol as h3,
};
pub const PROFILE: &str = "h3-av-noref-v2";
pub const I2V_PROFILE: &str = "h3-fl2va-i2va-v1";
pub fn request(job: &JobRequest) -> Result<h3::Request, &'static str> {
    let v = job.video.as_ref().ok_or("h3_profile_required")?;
    let shape = h3::Shape {
        width: job.width,
        height: job.height,
        frames: v.frames,
        fps: v.fps,
    };
    let profile = match v.profile.as_deref() {
        Some(I2V_PROFILE) => h3::Profile::ImageToAvV1,
        Some(PROFILE) => h3::Profile::AvNoReferenceV2,
        Some("h3-short-av-noref-v1") if shape == h3::Shape::default() => {
            h3::Profile::ShortAvNoReferenceV1
        }
        _ => return Err("unsupported_h3_profile"),
    };
    if job.model != "minimax-h3"
        || job.operation
            != if profile == h3::Profile::ImageToAvV1 {
                Operation::ImageToVideo
            } else {
                Operation::TextToVideo
            }
        || !shape.valid()
        || !v.audio
        || job.seed != Some(42)
        || job.sampling.steps != h3::STEPS
        || job.sampling.cfg_scale != 1.
        || job.sampling.sampler != h3::SAMPLER
        || job.sampling.scheduler != h3::SCHEDULER
        || !job.sampling.custom_sigmas.is_empty()
        || job.output.format != "mp4"
        || job.output.count != 1
    {
        return Err("unsupported_h3_profile");
    }
    if !job.negative_prompt.is_empty()
        || job.character_sheet.is_some()
        || !job.inputs.reference_images.is_empty()
        || job.inputs.pose_guide.is_some()
        || (profile == h3::Profile::ImageToAvV1) != job.inputs.start_image.is_some()
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
        capture_continuation: false,
        continuation: None,
        continuation_bytes: vec![],
        references: Default::default(),
        protocol: h3::VERSION,
        kind: h3::RequestKind::GenerateVideo,
        id: "x".repeat(128),
        profile,
        shape,
        image: None,
        image_rgb: Vec::new(),
        prompt: job.prompt.clone(),
        seed: 42,
        sampling: None,
    };
    if !r.valid_settings() {
        return Err("invalid_h3_request");
    }
    r.id = uuid::Uuid::new_v4().to_string();
    Ok(r)
}
