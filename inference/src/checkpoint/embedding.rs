//! Distinct MixedPrecisionOps.Embedding checkpoint path: its scale and ConvRot
//! rules differ from Linear and must not be replaced by the Linear loader.
use super::{exception, gap};
use crate::{
    execution_value::Value,
    layouts,
    native::{Error, Kitchen},
};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
pub fn load_quantized_embedding(
    state: &mut BTreeMap<String, Tensor>,
    prefix: &str,
    shape: Vec<i64>,
    compute: Kind,
    assign: bool,
    native: &Kitchen,
) -> Result<Option<Value>, Error> {
    let weight_key = format!("{prefix}weight");
    let conf_key = format!("{prefix}comfy_quant");
    let conf=state.get(&conf_key).map(|t|->Result<serde_json::Value,Error>{
        if t.device()!=Device::Cpu{return Err(exception("TypeError","can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."));}
        let t=t.f_contiguous()?.f_reshape([-1])?.f_view_dtype(Kind::Uint8)?;
        let mut bytes=vec![0;t.numel()];let len=bytes.len();t.f_copy_data_u8(&mut bytes,len)?;
        serde_json::from_slice(&bytes).map_err(|_|gap("Python JSON embedding metadata decoding"))
    }).transpose()?;
    if let Some(conf) = conf {
        let format = conf.get("format").and_then(serde_json::Value::as_str);
        if !matches!(
            format,
            Some("float8_e4m3fn" | "float8_e5m2" | "int8_tensorwise")
        ) {
            return Err(gap(
                "Embedding unsupported-format marker round-trip and super-loader",
            ));
        }
        if !state.contains_key(&weight_key) {
            return Err(gap("Embedding missing quantized weight super-loader"));
        }
        state.remove(&conf_key);
        let weight = state.remove(&weight_key).unwrap();
        let scale = match state.remove(&format!("{prefix}weight_scale")) {
            Some(t) => t.f_to_kind(Kind::Float)?,
            None => Tensor::f_ones([], (Kind::Float, Device::Cpu))?,
        };
        let q = if format == Some("int8_tensorwise") {
            let rotated = conf.get("convrot").is_some_and(|v| match v {
                serde_json::Value::Bool(b) => *b,
                serde_json::Value::Null => false,
                serde_json::Value::Number(n) => n.as_f64() != Some(0.),
                serde_json::Value::String(s) => !s.is_empty(),
                serde_json::Value::Array(v) => !v.is_empty(),
                serde_json::Value::Object(v) => !v.is_empty(),
            });
            let group = if rotated {
                conf.get("convrot_groupsize")
                    .map(|v| {
                        v.as_i64()
                            .ok_or_else(|| gap("Embedding Python int conversion"))
                    })
                    .transpose()?
                    .unwrap_or(256)
            } else {
                256
            };
            layouts::int8_parts(
                weight.f_to_kind(Kind::Int8)?,
                scale,
                compute,
                shape,
                true,
                rotated,
                group,
            )
        } else {
            let kind = if format == Some("float8_e4m3fn") {
                Kind::Float8e4m3fn
            } else {
                Kind::Float8e5m2
            };
            let mut q = layouts::fp8_parts(weight.f_to_kind(kind)?, scale, compute, shape)?;
            q.layout = if format == Some("float8_e4m3fn") {
                "TensorCoreFP8E4M3Layout"
            } else {
                "TensorCoreFP8E5M2Layout"
            }
            .into();
            q
        };
        return Ok(Some(Value::quantized(q.detach(native)?)));
    }
    state
        .get(&weight_key)
        .map(|w| {
            Ok(Value::tensor(if assign {
                w.f_detach()?
            } else {
                native.clone_tensor(&w.f_to_kind(compute)?)?.f_detach()?
            }))
        })
        .transpose()
}
