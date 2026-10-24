use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub use crate::model_spec::{MOSS_BF16, MOSS_INT8 as MOSS};
pub const MODEL: &str = "qwen-image-2.1-int8";
pub const BF16: &str = "qwen-image-2.1-bf16";
pub const TURBO: &str = "qwen-image-2.1-turbo";
pub const MCP_PROFILE: &str = "qwen-image21-workflow-v1";
pub const MAX_SEED: u64 = 9_007_199_254_740_991;
pub const PROFILE: &str = "qwen-image21-1600x896-v2";
pub const API_REVISION: &str = "imagine-api-v1";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub resolved_models: Option<crate::model_library::ResolvedModels>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_profile: Option<ResolvedProfile>,
    pub model: String,
    pub operation: Operation,
    #[serde(default)]
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_sheet: Option<crate::character_sheet::CharacterSheet>,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "width")]
    pub width: u32,
    #[serde(default = "height")]
    pub height: u32,
    pub seed: Option<u64>,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub inputs: Inputs,
    #[serde(default)]
    pub model_options: ModelOptions,
    #[serde(default)]
    pub output: Output,
    pub video: Option<Video>,
    #[serde(default)]
    pub client_metadata: serde_json::Map<String, Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ResolvedProfile {
    pub id: String,
    pub version: u32,
    pub sha256: String,
}
fn width() -> u32 {
    1600
}
fn height() -> u32 {
    896
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    TextToImage,
    ImageEdit,
    CharacterSheet,
    TextToVideo,
    ImageToVideo,
    ReferenceToVideo,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Sampling {
    pub steps: u32,
    pub cfg_scale: f32,
    pub sampler: String,
    pub scheduler: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub custom_sigmas: Vec<f32>,
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            steps: 40,
            cfg_scale: 2.5,
            sampler: "er_sde".into(),
            scheduler: "beta".into(),
            custom_sigmas: Vec::new(),
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Inputs {
    pub reference_images: Vec<String>,
    pub start_image: Option<String>,
    pub end_image: Option<String>,
    pub mask: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pose_guide: Option<String>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ModelOptions {
    pub qwen_image_2_1: QwenOptions,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub moss_image: Option<WorkflowOptions>,
}
impl ModelOptions {
    pub fn generation(&self, model: &str) -> Option<&WorkflowOptions> {
        if crate::model_spec::is_moss(model) {
            self.moss_image.as_ref()
        } else {
            self.qwen_image_2_1.workflow.as_ref()
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct QwenOptions {
    pub reference_max_pixels_per_image: u32,
    #[serde(
        rename = "generation",
        alias = "workflow",
        skip_serializing_if = "Option::is_none"
    )]
    pub workflow: Option<WorkflowOptions>,
}
impl Default for QwenOptions {
    fn default() -> Self {
        Self {
            reference_max_pixels_per_image: 1_000_000,
            workflow: None,
        }
    }
}
/// Explicit workflow settings are separate from the frozen default profile.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct WorkflowOptions {
    pub diffusion_model: String,
    pub text_encoder: String,
    pub vae: String,
    #[serde(default)]
    pub loras: Vec<Lora>,
    #[serde(default)]
    pub zero_negative_conditioning: bool,
    #[serde(default)]
    pub color_match_strength: f32,
    #[serde(default)]
    pub brightness_strength: f32,
    #[serde(default)]
    pub contrast_strength: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Lora {
    pub path: String,
    pub strength: f32,
}
fn relative_model_path(path: &str) -> bool {
    !path.is_empty()
        && std::path::Path::new(path)
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Output {
    pub format: String,
    pub count: u32,
}
impl Default for Output {
    fn default() -> Self {
        Self {
            format: "png".into(),
            count: 1,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Video {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    pub frames: u32,
    pub fps: u32,
    pub audio: bool,
}

impl JobRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.model == "minimax-h3" {
            return crate::video::job::request(self).map(|_| ());
        }
        if crate::model_spec::model(&self.model).is_none() {
            return Err("unknown_model");
        }
        if !matches!(
            self.operation,
            Operation::TextToImage | Operation::ImageEdit | Operation::CharacterSheet
        ) || self.video.is_some()
        {
            return Err("unsupported_operation");
        }
        if self.operation == Operation::CharacterSheet {
            crate::model_adapter::sheet_strategy(&self.model)?;
            let sheet = self
                .character_sheet
                .as_ref()
                .ok_or("character_sheet_required")?;
            sheet.validate(self.inputs.reference_images.len())?;
            if !self.prompt.is_empty() {
                return Err("unexpected_prompt");
            }
            if self.inputs.reference_images.len() > 8 {
                return Err("invalid_reference_count");
            }
            // This strategy has its own canvas and no scene colour transfer.
            if self
                .model_options
                .qwen_image_2_1
                .workflow
                .as_ref()
                .is_some_and(|w| {
                    w.color_match_strength > 0.
                        || w.brightness_strength > 0.
                        || w.contrast_strength > 0.
                })
            {
                return Err("unsupported_character_sheet_color_matching");
            }
            let cap = self
                .model_options
                .qwen_image_2_1
                .reference_max_pixels_per_image;
            for (w, h) in sheet.layout.pass_sizes()? {
                let (_, _, w, h) = crate::workflow_image::dimensions(w, h, cap);
                if w == 0 || h == 0 {
                    return Err("unsupported_reference_geometry");
                }
            }
        } else if self.character_sheet.is_some() || self.inputs.pose_guide.is_some() {
            return Err("unexpected_character_sheet_input");
        }
        if (crate::model_spec::is_moss(&self.model)
            && self.model_options.qwen_image_2_1.workflow.is_some())
            || (!crate::model_spec::is_moss(&self.model) && self.model_options.moss_image.is_some())
        {
            return Err("model_settings_mismatch");
        }
        let workflow = self.model_options.generation(&self.model);
        let minimum = if workflow.is_some() { 32 } else { 128 };
        if self.operation != Operation::CharacterSheet
            && (!(minimum..=4096).contains(&self.width)
                || !(minimum..=4096).contains(&self.height)
                || self.width.max(self.height) < 256
                || !self.width.is_multiple_of(32)
                || !self.height.is_multiple_of(32))
        {
            return Err("unsupported_resolution");
        }
        if (self.operation != Operation::CharacterSheet && self.prompt.trim().is_empty())
            || self.prompt.len() > 32768
            || self.negative_prompt.len() > 32768
        {
            return Err("invalid_prompt_length");
        }
        if self.sampling.steps == 0
            || !self.sampling.cfg_scale.is_finite()
            || (!matches!(
                self.sampling.sampler.as_str(),
                "er_sde" | "euler" | "res_multistep"
            ))
            || (!matches!(
                self.sampling.scheduler.as_str(),
                "beta" | "simple" | "normal" | "sgm_uniform"
            ))
            || (workflow.is_none()
                && (self.sampling.sampler != "er_sde"
                    || self.sampling.scheduler != "beta"
                    || !self.sampling.custom_sigmas.is_empty()))
            || self
                .sampling
                .custom_sigmas
                .iter()
                .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err("unsupported_sampling");
        }
        if !self.sampling.custom_sigmas.is_empty()
            && (usize::try_from(self.sampling.steps)
                .ok()
                .and_then(|n| n.checked_add(1))
                != Some(self.sampling.custom_sigmas.len())
                || self.sampling.custom_sigmas.last() != Some(&0.)
                || self.sampling.custom_sigmas[0] <= 0.
                || self.sampling.custom_sigmas.windows(2).any(|s| s[1] > s[0]))
        {
            return Err("invalid_sigma_schedule");
        }
        if self.seed.is_some_and(|seed| seed > MAX_SEED) {
            return Err("invalid_seed");
        }
        if crate::model_spec::is_moss(&self.model)
            && (self.operation != Operation::TextToImage
                || !self.inputs.reference_images.is_empty()
                || workflow.is_none()
                || self.sampling.sampler != "euler"
                || !matches!(self.sampling.scheduler.as_str(), "normal" | "sgm_uniform")
                || workflow.is_some_and(|w| {
                    w.zero_negative_conditioning
                        || w.color_match_strength != 0.
                        || w.brightness_strength != 0.
                        || w.contrast_strength != 0.
                }))
        {
            return Err("unsupported_moss_configuration");
        }
        if !crate::model_spec::is_moss(&self.model)
            && matches!(self.sampling.scheduler.as_str(), "normal" | "sgm_uniform")
        {
            return Err("unsupported_sampling");
        }
        if matches!(self.model.as_str(), BF16 | TURBO) && workflow.is_none() {
            return Err("workflow_settings_required");
        }
        if let Some(w) = workflow {
            if ![&w.diffusion_model, &w.text_encoder, &w.vae]
                .iter()
                .all(|p| relative_model_path(p))
                || w.loras
                    .iter()
                    .any(|l| !relative_model_path(&l.path) || !l.strength.is_finite())
            {
                return Err("invalid_model_path");
            }
            if [
                w.color_match_strength,
                w.brightness_strength,
                w.contrast_strength,
            ]
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            {
                return Err("invalid_color_matching");
            }
            if (w.color_match_strength > 0.
                || w.brightness_strength > 0.
                || w.contrast_strength > 0.)
                && self.inputs.reference_images.is_empty()
            {
                return Err("invalid_reference_count");
            }
            // Zero negative conditioning is immaterial at CFG=1. The pinned native
            // API does not implement Comfy's zero-tensor conditioning at other CFGs.
            if w.zero_negative_conditioning && self.sampling.cfg_scale != 1. {
                return Err("unsupported_zero_negative_conditioning");
            }
        }
        if self.output.format != "png" || self.output.count != 1 {
            return Err("unsupported_output");
        }
        if self.inputs.start_image.is_some()
            || self.inputs.end_image.is_some()
            || self.inputs.mask.is_some()
        {
            return Err("unsupported_input");
        }
        match self.operation {
            Operation::TextToImage if !self.inputs.reference_images.is_empty() => {
                return Err("unexpected_reference_images");
            }
            Operation::ImageEdit
                if self.inputs.reference_images.is_empty()
                    || self.inputs.reference_images.len() > 10 =>
            {
                return Err("invalid_reference_count");
            }
            _ => {}
        }
        if workflow.is_none()
            && !(262_144..=4_000_000).contains(
                &self
                    .model_options
                    .qwen_image_2_1
                    .reference_max_pixels_per_image,
            )
        {
            return Err("invalid_reference_pixel_limit");
        }
        if serde_json::to_vec(&self.client_metadata).map_or(true, |v| v.len() > 4096) {
            return Err("metadata_too_large");
        }
        Ok(())
    }
    pub fn asset_ids(&self) -> impl Iterator<Item = &String> {
        self.inputs
            .reference_images
            .iter()
            .chain(self.inputs.pose_guide.iter())
            .chain(self.inputs.start_image.iter())
    }
    pub fn effective(&self) -> Value {
        let mut v = serde_json::to_value(self).expect("serializable request");
        if self.model == "minimax-h3" {
            v.as_object_mut().unwrap().remove("model_options");
            v["profile_revision"] = json!(
                self.video
                    .as_ref()
                    .and_then(|v| v.profile.as_deref())
                    .unwrap_or(crate::video::job::PROFILE)
            );
            v["api_revision"] = json!(API_REVISION);
            return v;
        }
        v["profile_revision"] = json!(if self.model_options.qwen_image_2_1.workflow.is_some() {
            MCP_PROFILE
        } else {
            PROFILE
        });
        v["api_revision"] = json!(API_REVISION);
        if let Some(sheet) = &self.character_sheet {
            v.as_object_mut().unwrap().remove("width");
            v.as_object_mut().unwrap().remove("height");
            v["execution_plan"] = crate::model_adapter::sheet_plan(self, sheet);
        }
        v
    }
}

pub fn models() -> Value {
    json!({"models":[model(MODEL).unwrap(), model(TURBO).unwrap(), model(MOSS).unwrap(), model(MOSS_BF16).unwrap(), model(BF16).unwrap(), model("minimax-h3").unwrap()]})
}
pub fn model(id: &str) -> Option<Value> {
    match id {
        BF16 => {
            let mut v = model(MODEL)?;
            v["id"] = json!(BF16);
            v["engine"] = json!("Imagine Rust");
            v["availability"] = json!("requires_rust_backend_and_weights");
            v["profile_revision"] = json!(MCP_PROFILE);
            v["constraints"]["loras"] = json!(true);
            v["constraints"]["samplers"] = json!(["er_sde", "euler"]);
            v["constraints"]["schedulers"] = json!(["beta", "simple"]);
            v["constraints"]["validated_resolutions"] = json!([]);
            Some(v)
        }
        MOSS | MOSS_BF16 => {
            let spec = crate::model_spec::model(id)?;
            let mut v = json!({
                "id": id, "available": true, "engine":"Imagine Rust",
                "availability":"requires_rust_backend_and_weights", "profile_revision":PROFILE,
                "operations":spec.architecture.operations(),
                "constraints": {
                    "dimensions":{"min":128,"max":4096,"multiple":32,"min_max_side":256},
                    "validated_resolutions":[], "steps":{"min":1}, "cfg_scale":{"finite":true},
                    "samplers":spec.architecture.samplers(), "schedulers":spec.architecture.schedulers(),
                    "output_formats":["png"],"max_output_count":1,"loras":true,
                    "reference_images":{"max_count":0}
                }, "request_schema":schemars::schema_for!(JobRequest)
            });
            v["defaults"] = serde_yaml::from_str::<Value>(include_str!("../profiles/moss.yaml"))
                .ok()?["defaults"]
                .clone();
            v["defaults"]["model"] = json!(id);
            if id == MOSS_BF16 {
                v["defaults"]["model_options"]["moss_image"]["diffusion_model"] =
                    json!("diffusion_models/moss/moss_image_sft_bf16.safetensors");
            }
            Some(v)
        }
        TURBO => Some(
            json!({"id":TURBO,"available":true,"availability":"requires_local_workflow_binary_and_weights","engine":"stable-diffusion.cpp","profile_revision":MCP_PROFILE,"operations":["text_to_image","image_edit"],"request_schema":schemars::schema_for!(JobRequest)}),
        ),
        MODEL => Some(json!({
            "id": MODEL, "available": true, "availability": "requires_local_frozen_binary_and_weights",
            "engine":"stable-diffusion.cpp", "profile_revision":PROFILE,
            "operations":["text_to_image","image_edit","character_sheet"],
            "character_sheet":crate::model_adapter::sheet_capabilities(),
            "constraints": {
                "dimensions":{"min":128,"max":4096,"multiple":32,"min_max_side":256},
                "validated_resolutions":[{"width":1600,"height":896}],
                "resolution_note":"Other dimensions use the same runtime configuration but are not locally validated; memory requirements vary",
                "steps":{"min":1},
                "cfg_scale":{"finite":true}, "samplers":["er_sde"], "schedulers":["beta"],
                "output_formats":["png"],"max_output_count":1,"loras":false,
                "reference_images":{"max_count":10,"locally_validated_count":1,"ordered":true,
                    "pixel_limit_min":262144,"pixel_limit_max":4000000,"pixel_limit_scope":"per_image",
                    "resize":"CPU Lanczos shrink only; center crop to multiples of 32; shared VAE/vision input",
                    "resource_note":"Multiple references and larger caps can exceed available memory"}
            },
            "defaults":{"width":1600,"height":896,"sampling":Sampling::default(),"output":Output::default(),"model_options":ModelOptions::default()},
            "request_schema":schemars::schema_for!(JobRequest)
        })),
        "minimax-h3" => Some(
            json!({"id":id,"available":false,"reason":"model_not_validated",
            "planned_operations":["text_to_video","image_to_video","reference_to_video"],
            "planned_video":{"fps":24,"frames_rule":"17*k+5","audio":true},
            "execution_enabled":false}),
        ),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dimensions_accept_aligned_sizes_and_reject_out_of_bounds() {
        let mut r: JobRequest = serde_json::from_value(
            json!({"model":MODEL,"operation":"text_to_image","prompt":"test"}),
        )
        .unwrap();
        for (width, height) in [
            (1600, 896),
            (896, 1600),
            (768, 1024),
            (256, 128),
            (4096, 4096),
        ] {
            r.width = width;
            r.height = height;
            assert!(r.validate().is_ok());
        }
        for (width, height) in [
            (0, 896),
            (96, 256),
            (128, 128),
            (1025, 1024),
            (4096, 4128),
            (u32::MAX, 896),
        ] {
            r.width = width;
            r.height = height;
            assert_eq!(r.validate(), Err("unsupported_resolution"));
        }
    }
    #[test]
    fn strict_contract_and_order() {
        let mut r: JobRequest = serde_json::from_value(json!({"model":MODEL,"operation":"image_edit","prompt":"test","inputs":{"reference_images":["b","a"]}})).unwrap();
        assert!(r.validate().is_ok());
        assert_eq!(r.inputs.reference_images, ["b", "a"]);
        r.inputs.reference_images = vec!["a".into(); 11];
        assert_eq!(r.validate(), Err("invalid_reference_count"));
        assert!(
            serde_json::from_value::<JobRequest>(
                json!({"model":MODEL,"operation":"text_to_image","prompt":"x","loras":[]})
            )
            .is_err()
        );
        r.inputs.reference_images.truncate(1);
        r.width = 1025;
        assert_eq!(r.validate(), Err("unsupported_resolution"));
    }
}
