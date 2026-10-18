//! `_load_quantized_module` parameter preparation from frozen comfy/ops.py.
//! State-key consumption and module policy are explicit; no checkpoint-wide cast.
use super::{exception, gap};
use crate::{
    execution_value::Value,
    layouts,
    native::{Error, Kitchen},
    storage::{LayoutParams, QuantizedStorage, ScalarField},
};
use serde_json::Value as Json;
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

/// Comfy's FP8 subclasses preserve their original public layout identities and
/// inherit the unchanged Kitchen dispatch/decoder implementations.
pub fn bind_comfy_layouts(rt: &crate::execution::Runtime) {
    for name in ["TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout"] {
        rt.register_layout(
            name,
            [
                name,
                "_TensorCoreFP8LayoutBase",
                "TensorCoreFP8Layout",
                "QuantizedLayout",
                "ABC",
                "object",
            ]
            .map(str::to_owned)
            .to_vec(),
            None,
        );
    }
}
fn tuple(shape: &[i64]) -> String {
    format!(
        "({}{})",
        shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if shape.len() == 1 { "," } else { "" }
    )
}

pub struct QuantizedModule {
    pub weight: Option<Value>,
    pub quant_format: Option<String>,
    pub full_precision: bool,
    pub full_precision_config: bool,
    pub extras: BTreeMap<String, Tensor>,
    pub consumed: Vec<String>,
}
pub struct LoadOptions<'a> {
    pub device: Device,
    pub compute_kind: Kind,
    pub original_shape: Vec<i64>,
    pub full_precision: bool,
    pub disabled_formats: &'a [&'a str],
    pub load_extra_params: bool,
}
fn pop(
    state: &mut BTreeMap<String, Tensor>,
    prefix: &str,
    name: &str,
    device: Device,
    consumed: &mut Vec<String>,
) -> Result<Option<Tensor>, Error> {
    let key = format!("{prefix}{name}");
    let value = state
        .remove(&key)
        .map(|t| t.f_to_device(device))
        .transpose()?;
    if value.is_some() {
        consumed.push(key);
    }
    Ok(value)
}
fn required_scale(value: Option<Tensor>, message: impl Into<String>) -> Result<Tensor, Error> {
    value.ok_or_else(|| exception("ValueError", message))
}
fn truth(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(x) => *x,
        Json::Number(n) => n.as_f64() != Some(0.0),
        Json::String(s) => !s.is_empty(),
        Json::Array(v) => !v.is_empty(),
        Json::Object(v) => !v.is_empty(),
    }
}
fn int(value: Option<&Json>, default: i64) -> Result<i64, Error> {
    match value {
        None => Ok(default),
        Some(Json::Bool(v)) => Ok(i64::from(*v)),
        Some(Json::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|v| v.trunc() as i64))
            .ok_or_else(|| gap("Unbounded quantization integer")),
        Some(Json::String(s)) => s.trim().parse().map_err(|_| {
            exception(
                "ValueError",
                format!("invalid literal for int() with base 10: '{s}'"),
            )
        }),
        _ => Err(gap("Python int conversion of quantization metadata")),
    }
}
/// Prepare the original module's weight and per-format state. Bias loading,
/// missing-key aggregation and shape checks belong to the original super-loader
/// boundary, which the caller must execute separately.
pub fn load_quantized_module(
    state: &mut BTreeMap<String, Tensor>,
    prefix: &str,
    options: LoadOptions<'_>,
    native: &Kitchen,
) -> Result<QuantizedModule, Error> {
    let mut out = QuantizedModule {
        weight: None,
        quant_format: None,
        full_precision: options.full_precision,
        full_precision_config: false,
        extras: BTreeMap::new(),
        consumed: vec![],
    };
    let weight_key = format!("{prefix}weight");
    let Some(weight) = state.remove(&weight_key) else {
        return Ok(out);
    };
    out.consumed.push(weight_key);
    let conf = state.remove(&format!("{prefix}comfy_quant"));
    let Some(conf) = conf else {
        out.weight = Some(Value::tensor(
            weight
                .f_to_device_(options.device, options.compute_kind, false, false)?
                .f_detach()?,
        ));
        return Ok(out);
    };
    // `.numpy().tobytes()` requires CPU storage in the source. Do not silently
    // transfer an invalid device-side metadata tensor to CPU.
    if conf.device() != Device::Cpu {
        return Err(exception(
            "TypeError",
            "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.",
        ));
    }
    let bytes = conf
        .f_contiguous()?
        .f_reshape([-1])?
        .f_view_dtype(Kind::Uint8)?;
    let mut data = vec![0_u8; bytes.numel()];
    let count = data.len();
    bytes.f_copy_data_u8(&mut data, count)?;
    let conf: Json = serde_json::from_slice(&data)
        .map_err(|_| gap("Python JSON quantization metadata decoding"))?;
    let name = prefix.trim_end_matches('.');
    let format = conf
        .get("format")
        .filter(|v| !v.is_null())
        .map(|v| {
            v.as_str()
                .ok_or_else(|| gap("Non-string quantization format"))
        })
        .transpose()?
        .ok_or_else(|| {
            exception(
                "ValueError",
                format!("Unknown quantization format for layer {name}"),
            )
        })?;
    out.quant_format = Some(format.into());
    out.full_precision_config = conf.get("full_precision_matrix_mult").is_some_and(truth);
    if !out.full_precision {
        out.full_precision = out.full_precision_config;
    }
    if options.disabled_formats.contains(&format) {
        out.full_precision = true;
    }
    let params = conf.get("params").filter(|v| v.is_object());
    let option = |name: &str| conf.get(name).or_else(|| params.and_then(|v| v.get(name)));
    let group = || int(option("convrot_groupsize"), 256);
    let mut scale = |key: &str| pop(state, prefix, key, options.device, &mut out.consumed);
    let original_shape = options.original_shape;
    let original_kind = options.compute_kind;
    let q = match format {
        "int8_tensorwise" => {
            let scale = required_scale(
                scale("weight_scale")?,
                format!("Missing INT8 weight scale for layer {name}"),
            )?;
            let rotated = option("convrot").is_some_and(truth);
            layouts::int8_parts(
                weight.f_to_device_(options.device, Kind::Int8, false, false)?,
                scale,
                original_kind,
                original_shape,
                true,
                rotated,
                if rotated { group()? } else { 256 },
            )
        }
        "float8_e4m3fn" | "float8_e5m2" => {
            let scale = scale("weight_scale")?.ok_or_else(|| {
                exception(
                    "AttributeError",
                    "'NoneType' object has no attribute 'clone'",
                )
            })?;
            let storage_kind = if format == "float8_e4m3fn" {
                Kind::Float8e4m3fn
            } else {
                Kind::Float8e5m2
            };
            // Both Comfy-specific FP8 classes inherit identical Params/storage
            // handlers from Kitchen; the payload dtype selects E4M3/E5M2.
            let mut q = layouts::fp8_parts(
                weight.f_to_device_(options.device, storage_kind, false, false)?,
                scale,
                original_kind,
                original_shape,
            )?;
            q.layout = if format == "float8_e4m3fn" {
                "TensorCoreFP8E4M3Layout"
            } else {
                "TensorCoreFP8E5M2Layout"
            }
            .into();
            q
        }
        "mxfp8" => {
            let scale = required_scale(
                scale("weight_scale")?,
                format!("Missing MXFP8 block scales for layer {name}"),
            )?;
            let scale = native.view_shell_dtype(&scale, crate::native::ShellDType::E8M0)?;
            layouts::mxfp8_parts(
                weight.f_to_device_(options.device, Kind::Float8e4m3fn, false, false)?,
                scale,
                original_kind,
                original_shape,
            )
        }
        "nvfp4" => {
            let total = scale("weight_scale_2")?;
            let block = scale("weight_scale")?
                .map(|t| t.f_view_dtype(Kind::Float8e4m3fn))
                .transpose()?;
            if total.is_none() || block.is_none() {
                return Err(exception(
                    "ValueError",
                    format!("Missing NVFP4 scales for layer {name}"),
                ));
            }
            layouts::nvfp4_parts(
                weight.f_to_device_(options.device, Kind::Uint8, false, false)?,
                total.unwrap(),
                block.unwrap(),
                original_kind,
                original_shape,
            )?
        }
        "convrot_w4a4" => {
            let scale = required_scale(
                scale("weight_scale")?,
                format!("Missing ConvRot W4A4 weight scale for layer {name}"),
            )?;
            let linear_dtype = option("linear_dtype")
                .and_then(Json::as_str)
                .unwrap_or("int4");
            if !["int4", "int8"].contains(&linear_dtype) {
                return Err(exception(
                    "ValueError",
                    format!(
                        "ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got '{linear_dtype}'"
                    ),
                ));
            }
            QuantizedStorage::from_validated_parts(
                weight.f_to_device_(options.device, Kind::Int8, false, false)?,
                "TensorCoreConvRotW4A4Layout".into(),
                LayoutParams {
                    original_kind,
                    original_shape,
                    tensors: BTreeMap::from([("scale".into(), scale)]),
                    fields: BTreeMap::from([
                        ("convrot_groupsize".into(), ScalarField::Integer(group()?)),
                        ("quant_group_size".into(), ScalarField::Integer(64)),
                        (
                            "linear_dtype".into(),
                            ScalarField::Text(linear_dtype.into()),
                        ),
                        ("transposed".into(), ScalarField::Bool(false)),
                    ]),
                    field_order: [
                        "scale",
                        "orig_dtype",
                        "orig_shape",
                        "convrot_groupsize",
                        "quant_group_size",
                        "linear_dtype",
                        "transposed",
                    ]
                    .map(str::to_owned)
                    .to_vec(),
                },
            )
        }
        "asym_w4a8_int8" | "w6a8_int8" => {
            let bits = if format == "asym_w4a8_int8" { 4 } else { 6 };
            if weight.size().len() < 2 || original_shape.len() < 2 {
                return Err(exception("IndexError", "tuple index out of range"));
            }
            if weight.size()[1] * 8 != original_shape[1] * bits {
                return Err(exception(
                    "ValueError",
                    format!(
                        "{format} layer {name}: packed weight width {} does not match K={} at {bits} bits",
                        weight.size()[1],
                        original_shape[1]
                    ),
                ));
            }
            let rel = required_scale(
                scale("weight_s_rel")?,
                format!("Missing {format} group scale (weight_s_rel) for layer {name}"),
            )?;
            let rel = if rel.kind() == Kind::Uint8 {
                rel.f_view_dtype(Kind::Float8e4m3fn)?
            } else {
                rel
            };
            let channel = scale("weight_s_channel")?;
            let codebook = scale("weight_codebook")?;
            let rotation = group()?;
            let group = int(option("group_size"), 16)?;
            if original_shape.len() != 2 {
                return Err(exception(
                    "ValueError",
                    format!("orig_shape must be 2D, got {}", tuple(&original_shape)),
                ));
            }
            let (n, k) = (original_shape[0], original_shape[1]);
            let invalid = if group < 4 || k % 16 != 0 || k % group != 0 {
                true
            } else {
                if rotation == 0 {
                    return Err(exception("ZeroDivisionError", "integer modulo by zero"));
                }
                k % rotation != 0 || (16 % group != 0 && group % 16 != 0)
            };
            if invalid {
                return Err(exception(
                    "ValueError",
                    format!(
                        "K={k} must be divisible by 16, group_size={group}, and convrot_groupsize={rotation}; group_size must be >=4 and divide 16 or be a multiple of 16"
                    ),
                ));
            }
            let groups = k / group;
            if rel.size() != [n, groups] {
                return Err(exception(
                    "ValueError",
                    format!(
                        "scale must have shape {}, got {}",
                        tuple(&[n, groups]),
                        tuple(&rel.size())
                    ),
                ));
            }
            if channel.as_ref().is_none_or(|t| t.size() != [n]) {
                return Err(exception(
                    "ValueError",
                    format!(
                        "s_channel must have shape {}, got {}",
                        tuple(&[n]),
                        channel
                            .as_ref()
                            .map(|t| tuple(&t.size()))
                            .unwrap_or_else(|| "None".into())
                    ),
                ));
            }
            if let Some(t) = &codebook
                && t.size() != [16]
            {
                return Err(exception(
                    "ValueError",
                    format!("codebook must have shape (16,), got {}", tuple(&t.size())),
                ));
            }
            let mut tensors = BTreeMap::from([
                ("scale".into(), rel),
                ("s_channel".into(), channel.unwrap()),
            ]);
            let mut fields = BTreeMap::from([
                ("correction".into(), ScalarField::None),
                ("group_size".into(), ScalarField::Integer(group)),
                ("convrot_groupsize".into(), ScalarField::Integer(rotation)),
                ("transposed".into(), ScalarField::Bool(false)),
            ]);
            if let Some(t) = codebook {
                tensors.insert("codebook".into(), t);
            } else {
                fields.insert("codebook".into(), ScalarField::None);
            }
            QuantizedStorage::from_validated_parts(
                weight.f_to_device_(options.device, Kind::Int8, false, false)?,
                "AsymW4A8Int8Layout".into(),
                LayoutParams {
                    original_kind,
                    original_shape,
                    tensors,
                    fields,
                    field_order: [
                        "scale",
                        "orig_dtype",
                        "orig_shape",
                        "s_channel",
                        "correction",
                        "codebook",
                        "group_size",
                        "convrot_groupsize",
                        "transposed",
                    ]
                    .map(str::to_owned)
                    .to_vec(),
                },
            )
        }
        _ => return Err(exception("KeyError", format!("'{format}'"))),
    };
    // Parameter(QuantizedTensor, requires_grad=False) invokes the original
    // detach handler: payload storage aliases, while Params tensor fields clone.
    out.weight = Some(Value::quantized(q.detach(native)?));
    if options.load_extra_params {
        let extra: &[&str] = match format {
            "nvfp4" => &["input_scale", "pre_quant_scale"],
            "float8_e4m3fn" | "float8_e5m2" | "mxfp8" => &["input_scale"],
            _ => &[],
        };
        for key in extra {
            if let Some(value) = pop(state, prefix, key, options.device, &mut out.consumed)? {
                out.extras.insert((*key).into(), value);
            }
        }
    }
    Ok(out)
}
