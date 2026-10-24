//! Product model identity and checkpoint admission. No tensors, filesystem or inference.
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const QWEN_INT8: &str = "qwen-image-2.1-int8";
pub const QWEN_BF16: &str = "qwen-image-2.1-bf16";
pub const QWEN_TURBO: &str = "qwen-image-2.1-turbo";
pub const MOSS_INT8: &str = "moss-image-sft-int8";
pub const MOSS_BF16: &str = "moss-image-sft-bf16";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    QwenImage21,
    MossImage,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightFormat {
    Bf16,
    Int8ConvRot,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QkvLayout {
    Separate,
    Fused,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProfile {
    Standard,
    Turbo,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointSpec {
    pub architecture: Architecture,
    pub format: WeightFormat,
    pub qkv: QkvLayout,
}
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    pub architecture: Architecture,
    pub format: WeightFormat,
    pub profile: ExecutionProfile,
}
#[derive(Clone, Copy, Debug)]
pub struct ComponentRequirements {
    pub text_encoder: &'static str,
    pub vae: &'static str,
    pub default_text_file: &'static str,
    pub default_vae_file: &'static str,
}
impl Architecture {
    pub fn components(self) -> ComponentRequirements {
        match self {
            Self::QwenImage21 => ComponentRequirements {
                text_encoder: "qwen3vl_8b",
                vae: "qwen_image21",
                default_text_file: "qwen3vl_8b_int8_convrot.safetensors",
                default_vae_file: "qwen_image_2.1_vae_bf16.safetensors",
            },
            Self::MossImage => ComponentRequirements {
                text_encoder: "qwen35_9b",
                vae: "flux2",
                default_text_file: "qwen3.5_9b_int8_convrot.safetensors",
                default_vae_file: "flux2-vae.safetensors",
            },
        }
    }
    pub fn operations(self) -> &'static [&'static str] {
        match self {
            Self::QwenImage21 => &["text_to_image", "image_edit", "character_sheet"],
            Self::MossImage => &["text_to_image"],
        }
    }
    pub fn lora_components(self) -> &'static [&'static str] {
        match self {
            Self::QwenImage21 => &["dit", "text"],
            Self::MossImage => &["dit"],
        }
    }
    pub fn group(self) -> &'static str {
        match self {
            Self::QwenImage21 => "qwen-image-2.1",
            Self::MossImage => "moss-image",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::QwenImage21 => "Qwen Image 2.1",
            Self::MossImage => "MOSS Image",
        }
    }
    pub fn profiles(self) -> &'static [ExecutionProfile] {
        match self {
            Self::QwenImage21 => &[ExecutionProfile::Standard, ExecutionProfile::Turbo],
            Self::MossImage => &[ExecutionProfile::Standard],
        }
    }
    pub fn samplers(self) -> &'static [&'static str] {
        match self {
            Self::QwenImage21 => &["er_sde", "euler"],
            Self::MossImage => &["euler"],
        }
    }
    pub fn schedulers(self) -> &'static [&'static str] {
        match self {
            Self::QwenImage21 => &["beta", "simple"],
            Self::MossImage => &["normal", "sgm_uniform"],
        }
    }
}
impl CheckpointSpec {
    pub fn model(self, profile: ExecutionProfile) -> Result<&'static str, &'static str> {
        match (self.architecture, self.format, profile) {
            (Architecture::QwenImage21, WeightFormat::Int8ConvRot, ExecutionProfile::Standard) => {
                Ok(QWEN_INT8)
            }
            (Architecture::QwenImage21, WeightFormat::Bf16, ExecutionProfile::Standard) => {
                Ok(QWEN_BF16)
            }
            (Architecture::QwenImage21, WeightFormat::Bf16, ExecutionProfile::Turbo) => {
                Ok(QWEN_TURBO)
            }
            (Architecture::MossImage, WeightFormat::Int8ConvRot, ExecutionProfile::Standard) => {
                Ok(MOSS_INT8)
            }
            (Architecture::MossImage, WeightFormat::Bf16, ExecutionProfile::Standard) => {
                Ok(MOSS_BF16)
            }
            _ => Err("unsupported_execution_profile"),
        }
    }
}
pub fn model(id: &str) -> Option<ModelSpec> {
    let (architecture, format, profile) = match id {
        QWEN_INT8 => (
            Architecture::QwenImage21,
            WeightFormat::Int8ConvRot,
            ExecutionProfile::Standard,
        ),
        QWEN_BF16 => (
            Architecture::QwenImage21,
            WeightFormat::Bf16,
            ExecutionProfile::Standard,
        ),
        QWEN_TURBO => (
            Architecture::QwenImage21,
            WeightFormat::Bf16,
            ExecutionProfile::Turbo,
        ),
        MOSS_INT8 => (
            Architecture::MossImage,
            WeightFormat::Int8ConvRot,
            ExecutionProfile::Standard,
        ),
        MOSS_BF16 => (
            Architecture::MossImage,
            WeightFormat::Bf16,
            ExecutionProfile::Standard,
        ),
        _ => return None,
    };
    Some(ModelSpec {
        architecture,
        format,
        profile,
    })
}
pub fn is_moss(id: &str) -> bool {
    model(id).is_some_and(|s| s.architecture == Architecture::MossImage)
}
/// Header admission only. The component loader still validates every parameter and quantization metadata.
pub fn detect(header: &Value) -> Result<CheckpointSpec, &'static str> {
    let h = header.as_object().ok_or("invalid_checkpoint")?;
    let shape = |name: &str, expected: &[i64]| {
        h.get(name)
            .and_then(|v| v["shape"].as_array())
            .is_some_and(|s| {
                s.iter()
                    .map(Value::as_i64)
                    .eq(expected.iter().copied().map(Some))
            })
    };
    let architecture =
        if h.contains_key("txt_in.text_norm.weight") && h.contains_key("img_in.weight") {
            if !shape("txt_in.text_norm.weight", &[4096]) || !shape("img_in.weight", &[4096, 64]) {
                return Err("unsupported_model_configuration");
            }
            Architecture::QwenImage21
        } else if h.contains_key("video_embedder.weight")
            || h.contains_key("shared_block_modulation.weight")
        {
            if !shape("video_embedder.weight", &[3072, 128])
                || !shape("shared_block_modulation.weight", &[18432, 3072])
            {
                return Err("unsupported_model_configuration");
            }
            Architecture::MossImage
        } else {
            return Err("unsupported_model_architecture");
        };
    let (prefix, blocks) = match architecture {
        Architecture::QwenImage21 => ("transformer_blocks.", 32),
        Architecture::MossImage => ("layers.", 42),
    };
    if h.keys()
        .filter_map(|k| {
            k.strip_prefix(prefix)?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .any(|i| i >= blocks)
    {
        return Err("unsupported_model_configuration");
    }
    let qkv =
        if architecture == Architecture::MossImage && h.contains_key("layers.0.proj.qkv.weight") {
            QkvLayout::Fused
        } else {
            QkvLayout::Separate
        };
    for i in 0..blocks {
        if architecture == Architecture::QwenImage21 {
            for p in ["q", "k", "v"] {
                if !shape(
                    &format!("transformer_blocks.{i}.attn.to_{p}.weight"),
                    &[4096, 4096],
                ) {
                    return Err("unsupported_model_configuration");
                }
            }
            if !shape(
                &format!("transformer_blocks.{i}.img_mlp.gate_up.weight"),
                &[24576, 4096],
            ) {
                return Err("unsupported_model_configuration");
            }
        } else {
            let fused = format!("layers.{i}.proj.qkv.weight");
            match qkv {
                QkvLayout::Fused => {
                    if !shape(&fused, &[9216, 3072])
                        || ["q", "k", "v"]
                            .iter()
                            .any(|p| h.contains_key(&format!("layers.{i}.proj.{p}.weight")))
                    {
                        return Err("unsupported_qkv_layout");
                    }
                }
                QkvLayout::Separate => {
                    if h.contains_key(&fused)
                        || ["q", "k", "v"]
                            .iter()
                            .any(|p| !shape(&format!("layers.{i}.proj.{p}.weight"), &[3072, 3072]))
                    {
                        return Err("unsupported_qkv_layout");
                    }
                }
            }
            if !shape(
                &format!("layers.{i}.proj.ffn_up_gate.weight"),
                &[16384, 3072],
            ) {
                return Err("unsupported_model_configuration");
            }
        }
    }
    // Float32 auxiliaries are allowed; primary linear weights determine the representation.
    let mut quantized = false;
    for (name, tensor) in h {
        if name == "__metadata__" {
            continue;
        }
        let dtype = tensor["dtype"].as_str().ok_or("invalid_checkpoint")?;
        if name.ends_with(".weight") {
            match dtype {
                "BF16" | "F32" => {}
                "I8" if h
                    .contains_key(&format!("{}.comfy_quant", name.trim_end_matches(".weight"))) =>
                {
                    quantized = true
                }
                _ => return Err("unsupported_weight_format"),
            }
        }
    }
    let primary = if architecture == Architecture::QwenImage21 {
        "img_in.weight"
    } else {
        "video_embedder.weight"
    };
    if !quantized && h[primary]["dtype"] != "BF16" {
        return Err("unsupported_weight_format");
    }
    Ok(CheckpointSpec {
        architecture,
        format: if quantized {
            WeightFormat::Int8ConvRot
        } else {
            WeightFormat::Bf16
        },
        qkv,
    })
}
