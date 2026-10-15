use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const MODEL: &str = "qwen-image-2.1-int8";
pub const PROFILE: &str = "qwen-image21-1600x896-v2";
pub const API_REVISION: &str = "imagine-api-v1";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobRequest {
    pub model: String,
    pub operation: Operation,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "width")]
    pub width: u32,
    #[serde(default = "height")]
    pub height: u32,
    pub seed: Option<u32>,
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
}
impl Default for Sampling {
    fn default() -> Self {
        Self {
            steps: 40,
            cfg_scale: 2.5,
            sampler: "er_sde".into(),
            scheduler: "beta".into(),
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
}
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct ModelOptions {
    pub qwen_image_2_1: QwenOptions,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct QwenOptions {
    pub reference_max_pixels_per_image: u32,
}
impl Default for QwenOptions {
    fn default() -> Self {
        Self {
            reference_max_pixels_per_image: 1_000_000,
        }
    }
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
    pub frames: u32,
    pub fps: u32,
    pub audio: bool,
}

impl JobRequest {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.model == "minimax-h3" {
            return Err("model_not_validated");
        }
        if self.model != MODEL {
            return Err("unknown_model");
        }
        if !matches!(
            self.operation,
            Operation::TextToImage | Operation::ImageEdit
        ) || self.video.is_some()
        {
            return Err("unsupported_operation");
        }
        if !(128..=4096).contains(&self.width)
            || !(128..=4096).contains(&self.height)
            || self.width.max(self.height) < 256
            || !self.width.is_multiple_of(32)
            || !self.height.is_multiple_of(32)
        {
            return Err("unsupported_resolution");
        }
        if self.prompt.trim().is_empty()
            || self.prompt.len() > 32768
            || self.negative_prompt.len() > 32768
        {
            return Err("invalid_prompt_length");
        }
        if !(1..=100).contains(&self.sampling.steps)
            || !self.sampling.cfg_scale.is_finite()
            || !(0.0..=20.0).contains(&self.sampling.cfg_scale)
            || self.sampling.sampler != "er_sde"
            || self.sampling.scheduler != "beta"
        {
            return Err("unsupported_sampling");
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
        if !(262_144..=4_000_000).contains(
            &self
                .model_options
                .qwen_image_2_1
                .reference_max_pixels_per_image,
        ) {
            return Err("invalid_reference_pixel_limit");
        }
        if serde_json::to_vec(&self.client_metadata).map_or(true, |v| v.len() > 4096) {
            return Err("metadata_too_large");
        }
        Ok(())
    }
    pub fn effective(&self) -> Value {
        let mut v = serde_json::to_value(self).expect("serializable request");
        v["profile_revision"] = json!(PROFILE);
        v["api_revision"] = json!(API_REVISION);
        v
    }
}

pub fn models() -> Value {
    json!({"models":[model(MODEL).unwrap(), model("minimax-h3").unwrap()]})
}
pub fn model(id: &str) -> Option<Value> {
    match id {
        MODEL => Some(json!({
            "id": MODEL, "available": true, "availability": "requires_local_frozen_binary_and_weights",
            "engine":"stable-diffusion.cpp", "profile_revision":PROFILE,
            "operations":["text_to_image","image_edit"],
            "constraints": {
                "dimensions":{"min":128,"max":4096,"multiple":32,"min_max_side":256},
                "validated_resolutions":[{"width":1600,"height":896}],
                "resolution_note":"Other dimensions use the same runtime configuration but are not locally validated; memory requirements vary",
                "steps":{"min":1,"max":100},
                "cfg_scale":{"min":0,"max":20}, "samplers":["er_sde"], "schedulers":["beta"],
                "output_formats":["png"],"max_output_count":1,"loras":false,
                "reference_images":{"max_count":10,"locally_validated_count":1,"ordered":true,
                    "pixel_limit_min":262144,"pixel_limit_max":4000000,"pixel_limit_scope":"per_image",
                    "resize":"native VAE cap; dimensions rounded to multiples of 32",
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
