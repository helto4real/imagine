//! Concrete layout preparation and the remaining source-defined decode routes.
use super::*;
use crate::storage::{LayoutParams, QuantizedStorage, ScalarField};
use std::collections::BTreeMap;
fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn option<'a>(k: &'a Kwargs, n: &str) -> Option<&'a Value> {
    k.iter().find(|(s, _)| s == n).map(|(_, v)| v)
}
fn flag(k: &Kwargs, n: &str, default: bool) -> Result<bool, Error> {
    match option(k, n) {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(type_error("Expected bool")),
    }
}
fn field_int(q: &QuantizedStorage, n: &str, default: i64) -> i64 {
    match q.params.fields.get(n) {
        Some(ScalarField::Integer(n)) => *n,
        _ => default,
    }
}
fn field_bool(q: &QuantizedStorage, n: &str) -> bool {
    q.params.fields.get(n) == Some(&ScalarField::Bool(true))
}
fn tuple_values(v: Value) -> Result<Vec<Value>, Error> {
    if let Value::Tuple(v) = v {
        Ok(v)
    } else {
        Err(Error::Native("Invalid internal quantizer result".into()))
    }
}
fn invoke(rt: &Runtime, op: &str, k: Kwargs) -> Result<Value, Error> {
    unsafe { rt.call_kitchen(op, &k) }
}
fn tensor_field(q: &QuantizedStorage, n: &str) -> Value {
    q.params
        .tensors
        .get(n)
        .map_or(Value::None, |t| Value::tensor(t.shallow_clone()))
}
fn fields_params(
    w: &Tensor,
    tensors: BTreeMap<String, Tensor>,
    fields: BTreeMap<String, ScalarField>,
    order: &[&str],
) -> LayoutParams {
    LayoutParams {
        original_kind: w.kind(),
        original_shape: w.size(),
        tensors,
        fields,
        field_order: order.iter().map(|s| s.to_string()).collect(),
    }
}
fn quant_scale(w: &Tensor, k: &Kwargs, max: f64) -> Result<Tensor, Error> {
    let value = option(k, "scale");
    let t = match value {
        None | Some(Value::None) => w.f_abs()?.f_amax([], false)?.f_div_scalar(max)?,
        Some(Value::Text(s)) if s == "recalculate" => {
            w.f_abs()?.f_amax([], false)?.f_div_scalar(max)?
        }
        Some(Value::Tensor(_)) => value.unwrap().as_tensor()?,
        Some(Value::Float(x)) => Tensor::from(*x).f_to_kind(Kind::Float)?,
        Some(Value::Int(x)) => Tensor::from(*x),
        Some(Value::Bool(x)) => Tensor::from(i64::from(*x)),
        _ => return Err(type_error("new(): invalid data type 'str'")),
    };
    Ok(t.f_to_device_(w.device(), Kind::Float, false, false)?)
}
impl Runtime {
    /// # Safety
    /// The selected native quantizer must receive its original supported operands.
    pub unsafe fn from_float(
        &self,
        w: &Tensor,
        layout: &str,
        k: &Kwargs,
    ) -> Result<QuantizedStorage, Error> {
        let weight = || Value::tensor(w.shallow_clone());
        let seed = option(k, "stochastic_rounding")
            .cloned()
            .unwrap_or(Value::Int(0));
        match layout {
            "TensorCoreAWQW4A16Layout" => crate::awq_layout::quantize(w),
            "TensorCoreSVDQuantW4A4Layout" => Err(exception(
                "NotImplementedError",
                "SVDQuant W4A4 requires offline calibration (DeepCompressor). Load pre-quantized tensors via `from_state_dict` instead.",
            )),
            "TensorCoreFP8Layout" => {
                let kind = dtype(option(k, "dtype"), Kind::Float8e4m3fn)?;
                let max = match kind {
                    Kind::Float8e4m3fn => 448.,
                    Kind::Float8e5m2 => 57344.,
                    Kind::Half => 65504.,
                    Kind::BFloat16 => f32::from_bits(0x7f7f0000) as f64,
                    Kind::Float => f32::MAX as f64,
                    Kind::Double => f64::MAX,
                    _ => {
                        return Err(type_error(
                            "torch.finfo() requires a floating point input type. Use torch.iinfo to handle 'torch.finfo'",
                        ));
                    }
                };
                let scale = quant_scale(w, k, max)?;
                let q = invoke(
                    self,
                    "quantize_per_tensor_fp8",
                    vec![
                        ("x".into(), weight()),
                        ("scale".into(), Value::tensor(scale.shallow_clone())),
                        ("output_type".into(), Value::DType(kind)),
                    ],
                )?
                .as_tensor()?;
                layouts::fp8_parts(q, scale, w.kind(), w.size())
            }
            "TensorWiseINT8Layout" => {
                let is_weight = flag(k, "is_weight", true)?;
                let per_channel = flag(k, "per_channel", false)?;
                let convrot = flag(k, "convrot", false)?;
                let group = integer(option(k, "convrot_groupsize"), 256)?;
                if convrot && !is_weight {
                    return Err(exception(
                        "ValueError",
                        "convrot is only supported when is_weight is True",
                    ));
                }
                if convrot && !per_channel {
                    return Err(exception(
                        "ValueError",
                        "convrot is only supported when per_channel is True",
                    ));
                }
                let mut kwargs = vec![(if convrot { "weight" } else { "x" }.into(), weight())];
                let op = if convrot {
                    kwargs.push(("group_size".into(), Value::Int(group)));
                    "quantize_int8_convrot_weight"
                } else if !is_weight || per_channel {
                    "quantize_int8_rowwise"
                } else {
                    kwargs.push((
                        "scale".into(),
                        option(k, "scale").cloned().unwrap_or(Value::None),
                    ));
                    "quantize_int8_tensorwise"
                };
                kwargs.push(("stochastic_rounding".into(), seed));
                let v = tuple_values(invoke(self, op, kwargs)?)?;
                Ok(layouts::int8_parts(
                    v[0].as_tensor()?,
                    v[1].as_tensor()?,
                    w.kind(),
                    w.size(),
                    is_weight,
                    convrot,
                    group,
                ))
            }
            "TensorCoreNVFP4Layout" | "TensorCoreMXFP8Layout" => {
                let fp4 = layout == "TensorCoreNVFP4Layout";
                let name = if fp4 { "NVFP4" } else { "MXFP8" };
                if w.dim() != 2 {
                    return Err(exception(
                        "ValueError",
                        format!("{name} requires 2D tensor, got {}D", w.dim()),
                    ));
                }
                let pad = if fp4 { 16 } else { 32 };
                let padding = w.size().iter().any(|n| n % pad != 0);
                let mut kwargs = vec![("x".into(), weight())];
                let scale = if fp4 {
                    Some(quant_scale(w, k, 448. * 6.)?)
                } else {
                    None
                };
                if let Some(scale) = &scale {
                    kwargs.push((
                        "per_tensor_scale".into(),
                        Value::tensor(scale.shallow_clone()),
                    ));
                }
                kwargs.push((
                    if fp4 { "pad_16x" } else { "pad_32x" }.into(),
                    Value::Bool(padding),
                ));
                let v = tuple_values(invoke(
                    self,
                    if fp4 {
                        "quantize_nvfp4"
                    } else {
                        "quantize_mxfp8"
                    },
                    kwargs,
                )?)?;
                if fp4 {
                    layouts::nvfp4_parts(
                        v[0].as_tensor()?,
                        scale.unwrap(),
                        v[1].as_tensor()?,
                        w.kind(),
                        w.size(),
                    )
                } else {
                    Ok(layouts::mxfp8_parts(
                        v[0].as_tensor()?,
                        v[1].as_tensor()?,
                        w.kind(),
                        w.size(),
                    ))
                }
            }
            "TensorCoreConvRotW4A4Layout" => {
                let group = integer(option(k, "convrot_groupsize"), 256)?;
                let qgroup = integer(option(k, "quant_group_size"), 64)?;
                let linear = option(k, "linear_dtype")
                    .cloned()
                    .unwrap_or(Value::Text("int4".into()));
                let linear = if let Value::Text(s) = linear {
                    if s != "int4" && s != "int8" {
                        return Err(exception(
                            "ValueError",
                            format!(
                                "ConvRot W4A4 linear_dtype must be 'int4' or 'int8', got '{s}'"
                            ),
                        ));
                    }
                    s
                } else {
                    return Err(type_error("Expected linear_dtype string"));
                };
                let v = tuple_values(invoke(
                    self,
                    "quantize_convrot_w4a4_weight",
                    vec![
                        ("weight".into(), weight()),
                        ("convrot_groupsize".into(), Value::Int(group)),
                        ("quant_group_size".into(), Value::Int(qgroup)),
                        ("stochastic_rounding".into(), seed),
                    ],
                )?)?;
                let p = fields_params(
                    w,
                    BTreeMap::from([("scale".into(), v[1].as_tensor()?)]),
                    BTreeMap::from([
                        ("convrot_groupsize".into(), ScalarField::Integer(group)),
                        ("quant_group_size".into(), ScalarField::Integer(qgroup)),
                        ("linear_dtype".into(), ScalarField::Text(linear)),
                        ("transposed".into(), ScalarField::Bool(false)),
                    ]),
                    &[
                        "scale",
                        "orig_dtype",
                        "orig_shape",
                        "convrot_groupsize",
                        "quant_group_size",
                        "linear_dtype",
                        "transposed",
                    ],
                );
                Ok(QuantizedStorage::from_validated_parts(
                    v[0].as_tensor()?,
                    layout.into(),
                    p,
                ))
            }
            "AsymW4A8Int8Layout" => {
                let group = integer(option(k, "group_size"), 16)?;
                let rotation = integer(option(k, "convrot_groupsize"), 256)?;
                let scale = dtype(option(k, "scale_dtype"), Kind::Float8e4m3fn)?;
                if !matches!(scale, Kind::Float | Kind::Float8e4m3fn) {
                    return Err(exception(
                        "ValueError",
                        format!(
                            "scale_dtype must be float32 or float8_e4m3fn, got {}",
                            crate::registry::AllowedType::DType(scale).python_name()
                        ),
                    ));
                }
                let mut kwargs = vec![("weight".into(), weight())];
                for (n, default) in [
                    ("group_size", Value::Int(group)),
                    ("convrot_groupsize", Value::Int(rotation)),
                    ("symmetric", Value::Bool(true)),
                    ("scale_dtype", Value::DType(scale)),
                    ("codebook", Value::Bool(true)),
                    ("codebook_tensor", Value::None),
                    ("stochastic_rounding", Value::Int(0)),
                    ("bits", Value::Int(4)),
                    ("scale_search", Value::Bool(true)),
                ] {
                    kwargs.push((n.into(), option(k, n).cloned().unwrap_or(default)));
                }
                let v = tuple_values(invoke(self, "quantize_w4a8_int8_weight", kwargs)?)?;
                let mut tensors = BTreeMap::from([
                    ("scale".into(), v[1].as_tensor()?),
                    ("s_channel".into(), v[2].as_tensor()?),
                ]);
                let mut fields = BTreeMap::from([
                    ("group_size".into(), ScalarField::Integer(group)),
                    ("convrot_groupsize".into(), ScalarField::Integer(rotation)),
                    ("transposed".into(), ScalarField::Bool(false)),
                ]);
                for (i, n) in [(3, "correction"), (4, "codebook")] {
                    if matches!(v[i], Value::None) {
                        fields.insert(n.into(), ScalarField::None);
                    } else {
                        tensors.insert(n.into(), v[i].as_tensor()?);
                    }
                }
                let p = fields_params(
                    w,
                    tensors,
                    fields,
                    &[
                        "scale",
                        "orig_dtype",
                        "orig_shape",
                        "s_channel",
                        "correction",
                        "codebook",
                        "group_size",
                        "convrot_groupsize",
                        "transposed",
                    ],
                );
                Ok(QuantizedStorage::from_validated_parts(
                    v[0].as_tensor()?,
                    layout.into(),
                    p,
                ))
            }
            _ => Err(exception("KeyError", format!("'{layout}'"))),
        }
    }
    pub fn requantize_kwargs(&self, q: &QuantizedStorage) -> Result<Kwargs, Error> {
        Ok(match q.layout.as_str() {
            "AsymW4A8Int8Layout" => vec![
                (
                    "group_size".into(),
                    Value::Int(field_int(q, "group_size", 16)),
                ),
                (
                    "convrot_groupsize".into(),
                    Value::Int(field_int(q, "convrot_groupsize", 256)),
                ),
                (
                    "symmetric".into(),
                    Value::Bool(!q.params.tensors.contains_key("correction")),
                ),
                (
                    "codebook".into(),
                    Value::Bool(q.params.tensors.contains_key("codebook")),
                ),
                ("codebook_tensor".into(), tensor_field(q, "codebook")),
                (
                    "scale_dtype".into(),
                    Value::DType(q.params.tensors["scale"].kind()),
                ),
                (
                    "bits".into(),
                    Value::Int(
                        crate::packing::geometry(
                            &q.payload,
                            &q.params.tensors["scale"],
                            field_int(q, "group_size", 16),
                        )?
                        .2,
                    ),
                ),
                ("scale_search".into(), Value::Bool(false)),
            ],
            "TensorCoreConvRotW4A4Layout" => vec![
                (
                    "convrot_groupsize".into(),
                    Value::Int(field_int(q, "convrot_groupsize", 256)),
                ),
                (
                    "quant_group_size".into(),
                    Value::Int(field_int(q, "quant_group_size", 64)),
                ),
                (
                    "linear_dtype".into(),
                    Value::Text(match q.params.fields.get("linear_dtype") {
                        Some(ScalarField::Text(s)) => s.clone(),
                        _ => "int4".into(),
                    }),
                ),
            ],
            "TensorWiseINT8Layout" => {
                let values = layouts::int8_requantize_kwargs(q)?;
                ["is_weight", "per_channel", "convrot", "convrot_groupsize"]
                    .into_iter()
                    .map(|n| {
                        let v = values[n].clone();
                        (
                            n.into(),
                            match v {
                                ScalarField::Bool(b) => Value::Bool(b),
                                ScalarField::Integer(i) => Value::Int(i),
                                _ => unreachable!(),
                            },
                        )
                    })
                    .collect()
            }
            _ => vec![],
        })
    }
    /// # Safety
    /// Same original quantizer contracts as from_float; options override preserved fields.
    pub unsafe fn requantize_from_float(
        &self,
        q: &QuantizedStorage,
        w: &Tensor,
        k: &Kwargs,
    ) -> Result<QuantizedStorage, Error> {
        let mut options = self.requantize_kwargs(q)?;
        for (n, v) in k {
            if let Some((_, old)) = options.iter_mut().find(|(key, _)| key == n) {
                *old = v.clone();
            } else {
                options.push((n.clone(), v.clone()));
            }
        }
        unsafe { self.from_float(w, &q.layout, &options) }
    }
    pub fn state_dict(
        &self,
        q: &QuantizedStorage,
        prefix: &str,
    ) -> Result<Vec<(String, Tensor)>, Error> {
        let names: &[(&str, &str)] = match q.layout.as_str() {
            "TensorCoreAWQW4A16Layout" => &[("_scale", "scale"), ("_zeros", "zeros")],
            "TensorCoreNVFP4Layout" => &[("_scale", "block_scale"), ("_scale_2", "scale")],
            "AsymW4A8Int8Layout" => &[
                ("_s_rel", "scale"),
                ("_s_channel", "s_channel"),
                ("_correction", "correction"),
                ("_codebook", "codebook"),
            ],
            "TensorCoreSVDQuantW4A4Layout" => &[
                ("_scale", "scale"),
                ("_proj_down", "proj_down"),
                ("_proj_up", "proj_up"),
                ("_smooth_factor", "smooth_factor"),
            ],
            _ => &[("_scale", "scale")],
        };
        let mut result = vec![(prefix.into(), q.payload.shallow_clone())];
        for (suffix, n) in names {
            if let Some(t) = q.params.tensors.get(*n) {
                result.push((format!("{prefix}{suffix}"), t.shallow_clone()));
            }
        }
        Ok(result)
    }
}
fn decode(rt: &Runtime, q: &QuantizedStorage) -> Result<Tensor, Error> {
    layouts::dequantize_with(q, |payload, p| {
        let v = Value::tensor(payload.shallow_clone());
        let tensor = |n: &str| {
            p.tensors
                .get(n)
                .map_or(Value::None, |t| Value::tensor(t.shallow_clone()))
        };
        let int = |n: &str, default| Value::Int(field_int(q, n, default));
        let (op, k) = match q.layout.as_str() {
            "TensorCoreNVFP4Layout" => (
                "dequantize_nvfp4",
                vec![
                    ("qx".into(), v),
                    ("per_tensor_scale".into(), tensor("scale")),
                    ("block_scales".into(), tensor("block_scale")),
                    ("output_type".into(), Value::DType(p.original_kind)),
                ],
            ),
            "TensorCoreMXFP8Layout" => (
                "dequantize_mxfp8",
                vec![
                    ("qx".into(), v),
                    ("block_scales".into(), tensor("scale")),
                    ("output_type".into(), Value::DType(p.original_kind)),
                ],
            ),
            "TensorCoreConvRotW4A4Layout" => (
                "dequantize_convrot_w4a4_weight",
                vec![
                    ("qdata".into(), v),
                    ("scales".into(), tensor("scale")),
                    ("convrot_groupsize".into(), int("convrot_groupsize", 256)),
                    ("quant_group_size".into(), int("quant_group_size", 64)),
                    ("output_dtype".into(), Value::DType(p.original_kind)),
                ],
            ),
            "AsymW4A8Int8Layout" => (
                "dequantize_w4a8_int8_weight",
                vec![
                    ("qdata".into(), v),
                    ("s_rel".into(), tensor("scale")),
                    ("s_channel".into(), tensor("s_channel")),
                    ("codebook".into(), tensor("codebook")),
                    ("correction".into(), tensor("correction")),
                    ("group_size".into(), int("group_size", 16)),
                    ("convrot_groupsize".into(), int("convrot_groupsize", 256)),
                    ("output_dtype".into(), Value::DType(p.original_kind)),
                ],
            ),
            "TensorWiseINT8Layout" => {
                let code = quantization::dtype_code(p.original_kind).unwrap_or(0);
                let mut k = vec![("q".into(), v), ("scale".into(), tensor("scale"))];
                if field_bool(q, "convrot") {
                    k.push(("group_size".into(), int("convrot_groupsize", 256)));
                }
                k.push(("output_dtype_code".into(), Value::Int(code)));
                return Ok(invoke(
                    rt,
                    if field_bool(q, "convrot") {
                        "dequantize_int8_convrot_weight_dtype"
                    } else {
                        "dequantize_int8_simple_dtype"
                    },
                    k,
                )?
                .as_tensor()?
                .f_to_kind(p.original_kind)?);
            }
            "TensorCoreSVDQuantW4A4Layout" => {
                let width = p.original_shape[1];
                let eye = Tensor::f_eye(width, (p.original_kind, payload.device()))?;
                let mut x = tuple_values(invoke(
                    rt,
                    "quantize_svdquant_w4a4",
                    vec![
                        ("x".into(), Value::tensor(eye)),
                        ("smooth".into(), tensor("smooth_factor")),
                        ("lora_down".into(), tensor("proj_down")),
                    ],
                )?)?;
                let la = x.pop().unwrap();
                let xs = x.pop().unwrap();
                let xq = x.pop().unwrap();
                let out = invoke(
                    rt,
                    "scaled_mm_svdquant_w4a4",
                    vec![
                        ("act".into(), xq),
                        ("wgt".into(), v),
                        ("ascales".into(), xs),
                        ("wscales".into(), tensor("scale")),
                        ("lora_act_in".into(), la),
                        ("lora_up".into(), tensor("proj_up")),
                        ("bias".into(), Value::None),
                    ],
                )?
                .as_tensor()?;
                return Ok(out.f_slice(0, 0, width, 1)?.f_tr()?.f_contiguous()?);
            }
            _ => unreachable!(),
        };
        invoke(rt, op, k)?.as_tensor()
    })
}
pub(super) fn install(rt: &Runtime) {
    for name in [
        "TensorCoreNVFP4Layout",
        "TensorCoreMXFP8Layout",
        "TensorCoreConvRotW4A4Layout",
        "AsymW4A8Int8Layout",
        "TensorCoreSVDQuantW4A4Layout",
        "TensorWiseINT8Layout",
    ] {
        rt.register_layout(
            name,
            vec![
                name.into(),
                "QuantizedLayout".into(),
                "ABC".into(),
                "object".into(),
            ],
            Some(Arc::new(decode)),
        );
    }
    for name in [
        "TensorCoreNVFP4Layout",
        "TensorCoreMXFP8Layout",
        "TensorCoreConvRotW4A4Layout",
        "AsymW4A8Int8Layout",
        "TensorCoreSVDQuantW4A4Layout",
    ] {
        rt.register_handler(
            "aten::t.default",
            name,
            body(|_, a, _| {
                let q = a[0].as_quantized()?;
                if matches!(
                    q.layout.as_str(),
                    "TensorCoreNVFP4Layout" | "TensorCoreMXFP8Layout"
                ) {
                    return Ok(Value::quantized(layouts::packed_transpose(&q)?));
                }
                if q.params.original_shape.len() < 2 {
                    return Err(exception("IndexError", "tuple index out of range"));
                }
                let mut p = q.params.shallow_copy();
                p.original_shape = vec![p.original_shape[1], p.original_shape[0]];
                p.fields.insert(
                    "transposed".into(),
                    ScalarField::Bool(!field_bool(&q, "transposed")),
                );
                Ok(Value::quantized(QuantizedStorage::from_validated_parts(
                    q.payload.shallow_clone(),
                    q.layout,
                    p,
                )))
            }),
        );
    }
    for name in [
        "TensorWiseINT8Layout",
        "TensorCoreNVFP4Layout",
        "TensorCoreMXFP8Layout",
        "TensorCoreConvRotW4A4Layout",
        "AsymW4A8Int8Layout",
        "TensorCoreSVDQuantW4A4Layout",
    ] {
        for op in [
            "aten::linear.default",
            "aten::mm.default",
            "aten::addmm.default",
        ] {
            if name == "TensorCoreNVFP4Layout" && op == "aten::addmm.default" {
                continue;
            }
            rt.register_handler(op, name, body(move |rt, a, k| math(rt, name, op, a, k)));
        }
    }
}

fn fallback(rt: &Runtime, op: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let a = a
        .iter()
        .map(|v| unsafe { rt.dequantize_args(v) })
        .collect::<Result<Vec<_>, _>>()?;
    let k = k
        .iter()
        .map(|(n, v)| Ok((n.clone(), unsafe { rt.dequantize_args(v)? })))
        .collect::<Result<Kwargs, Error>>()?;
    unsafe { rt.ordinary(op, &a, &k) }
}
fn scalar_one(v: Option<&Value>) -> bool {
    match v {
        None => true,
        Some(Value::Int(n)) => *n == 1,
        Some(Value::Float(n)) => *n == 1.,
        Some(Value::Bool(n)) => *n,
        _ => false,
    }
}
fn source_catches(e: &Error) -> bool {
    match e {
        Error::ReferenceException { class, .. } => matches!(
            *class,
            "RuntimeError"
                | "TypeError"
                | "NotImplementedError"
                | "OutOfMemoryError"
                | "AcceleratorError"
        ),
        Error::Selection(_) | Error::Torch(_) => true,
        _ => false,
    }
}
fn required_tensor(q: &QuantizedStorage, n: &str) -> Result<Value, Error> {
    q.params
        .tensors
        .get(n)
        .map(|t| Value::tensor(t.shallow_clone()))
        .ok_or_else(|| {
            exception(
                "AttributeError",
                format!("'Params' object has no attribute '{n}'"),
            )
        })
}
fn required_int(q: &QuantizedStorage, n: &str) -> Result<i64, Error> {
    match q.params.fields.get(n) {
        Some(ScalarField::Integer(n)) => Ok(*n),
        _ => Err(exception(
            "AttributeError",
            format!("'Params' object has no attribute '{n}'"),
        )),
    }
}
fn math(rt: &Runtime, layout: &str, op: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let linear = op == "aten::linear.default";
    let add = op == "aten::addmm.default";
    let empty = vec![];
    let wi = if add { 2 } else { 1 };
    let xi = usize::from(add);
    let bias = if add {
        a[0].clone()
    } else if linear {
        a.get(2).cloned().unwrap_or(Value::None)
    } else {
        Value::None
    };
    let Some(q) = a.get(wi).and_then(|v| v.as_quantized().ok()) else {
        return fallback(
            rt,
            op,
            a,
            if layout == "TensorWiseINT8Layout" {
                k
            } else {
                &empty
            },
        );
    };
    if matches!(layout, "TensorWiseINT8Layout" | "AsymW4A8Int8Layout") && q.layout != layout {
        return fallback(rt, op, a, k);
    }
    if layout == "AsymW4A8Int8Layout" && add {
        let matrix_addend = match &bias {
            Value::Tensor(_) => bias.as_tensor()?.dim() != 1,
            Value::Quantized(_) => bias.as_quantized()?.params.original_shape.len() != 1,
            _ => false,
        };
        if !scalar_one(option(k, "alpha")) || !scalar_one(option(k, "beta")) || matrix_addend {
            return fallback(rt, op, a, k);
        }
    }
    if layout == "TensorWiseINT8Layout" && linear && field_bool(&q, "transposed") {
        return fallback(rt, op, a, k);
    }
    if matches!(layout, "TensorCoreNVFP4Layout" | "TensorCoreMXFP8Layout") {
        return packed_math(rt, layout, op, a, k, &q, &bias);
    }
    let x = unsafe { rt.dequantize_args(&a[xi])? }.as_tensor()?;
    if linear && field_bool(&q, "transposed") {
        return unsafe {
            rt.op(
                op,
                &[Value::tensor(x), Value::tensor(rt.decode(&q)?), bias],
                &vec![],
            )
        };
    }
    let mut kwargs = vec![
        ("x".into(), Value::tensor(x.f_contiguous()?)),
        ("bias".into(), bias),
    ];
    let kernel = match layout {
        "TensorWiseINT8Layout" => {
            let kind = option(k, "out_dtype")
                .cloned()
                .unwrap_or(Value::DType(x.kind()));
            if !matches!(
                kind,
                Value::DType(Kind::Float | Kind::Half | Kind::BFloat16)
            ) {
                let repr: String = if let Value::DType(d) = kind {
                    crate::registry::AllowedType::DType(d).python_name().into()
                } else if matches!(kind, Value::None) {
                    "None".into()
                } else {
                    return Err(type_error("Expected dtype"));
                };
                return Err(exception(
                    "ValueError",
                    format!("Unsupported INT8 output dtype: {repr}"),
                ));
            }
            let mut w = q.payload.shallow_clone();
            let convrot = field_bool(&q, "convrot");
            if !linear && !field_bool(&q, "transposed") {
                if q.params.tensors["scale"].numel() == 1 && !convrot {
                    w = w.f_tr()?.f_contiguous()?;
                } else {
                    return fallback(rt, op, a, k);
                }
            }
            kwargs.extend([
                ("weight".into(), Value::tensor(w.f_contiguous()?)),
                ("weight_scale".into(), tensor_field(&q, "scale")),
                ("out_dtype".into(), kind),
                ("convrot".into(), Value::Bool(convrot)),
                (
                    "convrot_groupsize".into(),
                    Value::Int(field_int(&q, "convrot_groupsize", 256)),
                ),
            ]);
            "int8_linear"
        }
        "AsymW4A8Int8Layout" => {
            if !linear && !field_bool(&q, "transposed") {
                return Err(exception(
                    "RuntimeError",
                    "W4A8 GEMM expects RHS W.T. Use F.linear(x, W) or mm(x, W.t()).",
                ));
            }
            kwargs[0].1 = Value::tensor(x);
            kwargs.extend([
                ("qdata".into(), Value::tensor(q.payload.shallow_clone())),
                ("s_rel".into(), tensor_field(&q, "scale")),
                ("s_channel".into(), tensor_field(&q, "s_channel")),
                ("codebook".into(), tensor_field(&q, "codebook")),
                ("correction".into(), tensor_field(&q, "correction")),
                (
                    "group_size".into(),
                    Value::Int(field_int(&q, "group_size", 16)),
                ),
                (
                    "convrot_groupsize".into(),
                    Value::Int(field_int(&q, "convrot_groupsize", 256)),
                ),
                (
                    "out_dtype".into(),
                    if linear {
                        option(k, "out_dtype")
                            .cloned()
                            .unwrap_or(Value::DType(q.params.original_kind))
                    } else {
                        Value::DType(q.params.original_kind)
                    },
                ),
            ]);
            "w4a8_int8_linear"
        }
        "TensorCoreConvRotW4A4Layout" => {
            if !linear && !field_bool(&q, "transposed") {
                return Err(exception(
                    "RuntimeError",
                    "ConvRot W4A4 GEMM expects RHS W.T. Use F.linear(x, W) or mm(x, W.t()).",
                ));
            }
            kwargs[0].1 = Value::tensor(x);
            let linear = match q.params.fields.get("linear_dtype") {
                Some(ScalarField::Text(s)) => Value::Text(s.clone()),
                _ => {
                    return Err(exception(
                        "AttributeError",
                        "'Params' object has no attribute 'linear_dtype'",
                    ));
                }
            };
            kwargs.extend([
                ("qweight".into(), Value::tensor(q.payload.shallow_clone())),
                ("wscales".into(), tensor_field(&q, "scale")),
                (
                    "convrot_groupsize".into(),
                    Value::Int(required_int(&q, "convrot_groupsize")?),
                ),
                (
                    "quant_group_size".into(),
                    Value::Int(required_int(&q, "quant_group_size")?),
                ),
                ("linear_dtype".into(), linear),
            ]);
            "convrot_w4a4_linear"
        }
        "TensorCoreSVDQuantW4A4Layout" => {
            if !linear && !field_bool(&q, "transposed") {
                return Err(exception(
                    "RuntimeError",
                    "SVDQuant W4A4 GEMM expects the RHS to be W.T (stored W). Use F.linear(x, W) or mm(x, W.t()).",
                ));
            }
            return svd_forward(rt, &x, &q, &kwargs[1].1);
        }
        _ => unreachable!(),
    };
    invoke(rt, kernel, kwargs)
}
fn packed_math(
    rt: &Runtime,
    layout: &str,
    op: &str,
    a: &[Value],
    k: &Kwargs,
    q: &QuantizedStorage,
    bias: &Value,
) -> Result<Value, Error> {
    let linear = op == "aten::linear.default";
    let add = op == "aten::addmm.default";
    let xi = usize::from(add);
    let Some(x) = a[xi].as_quantized().ok() else {
        return fallback(rt, op, a, &vec![]);
    };
    if x.payload.dim() != 2 || field_bool(&x, "transposed") || field_bool(q, "transposed") == linear
    {
        return fallback(rt, op, a, &vec![]);
    }
    let fp4 = layout == "TensorCoreNVFP4Layout";
    let mut kwargs = vec![
        ("a".into(), Value::tensor(x.payload.shallow_clone())),
        ("b".into(), Value::tensor(q.payload.shallow_clone())),
    ];
    if fp4 {
        kwargs.extend([
            ("tensor_scale_a".into(), required_tensor(&x, "scale")?),
            ("tensor_scale_b".into(), required_tensor(q, "scale")?),
            ("block_scale_a".into(), required_tensor(&x, "block_scale")?),
            ("block_scale_b".into(), required_tensor(q, "block_scale")?),
        ]);
    } else {
        kwargs.extend([
            ("block_scale_a".into(), required_tensor(&x, "scale")?),
            ("block_scale_b".into(), required_tensor(q, "scale")?),
        ]);
    }
    kwargs.push((
        "out_dtype".into(),
        if add {
            Value::DType(x.params.original_kind)
        } else {
            option(k, "out_dtype")
                .cloned()
                .unwrap_or(Value::DType(x.params.original_kind))
        },
    ));
    if linear || add {
        kwargs.push(("bias".into(), bias.clone()));
    }
    let result = invoke(
        rt,
        if fp4 {
            "scaled_mm_nvfp4"
        } else {
            "scaled_mm_mxfp8"
        },
        kwargs,
    );
    match result {
        Ok(v) => {
            let t = v.as_tensor()?;
            let m = x.params.original_shape[0];
            let n = q.params.original_shape[usize::from(!linear)];
            Ok(Value::tensor(if t.size()[0] != m || t.size()[1] != n {
                t.f_slice(0, 0, m, 1)?.f_slice(1, 0, n, 1)?
            } else {
                t
            }))
        }
        Err(e) if source_catches(&e) => fallback(rt, op, a, &vec![]),
        Err(e) => Err(e),
    }
}
fn svd_forward(
    rt: &Runtime,
    x: &Tensor,
    q: &QuantizedStorage,
    bias: &Value,
) -> Result<Value, Error> {
    let smooth = required_tensor(q, "smooth_factor")?;
    let down = required_tensor(q, "proj_down")?;
    let up = required_tensor(q, "proj_up")?;
    let shape = x.size();
    let width = *shape
        .last()
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    let x2 = x.f_reshape([-1, width])?;
    let m = x2.size()[0];
    let unsigned = field_bool(q, "act_unsigned");
    let main = if unsigned {
        x2.f_add_scalar(0.171875)?
    } else {
        x2.shallow_clone()
    };
    let enabled = std::env::var("COMFY_KITCHEN_SVDQUANT_DIRECT_CUDA")
        .map(|v| {
            !matches!(
                v.trim().to_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true);
    let direct = enabled
        && rt
            .backends
            .policy
            .current_override()
            .is_none_or(|s| s == "cuda")
        && matches!(main.device(), tch::Device::Cuda(_))
        && main.device() == q.payload.device();
    let call = |op: &str, kw: Kwargs| -> Result<Value, Error> {
        if direct {
            unsafe { rt.backends.call_bound(rt, "cuda", op, &kw) }
        } else {
            invoke(rt, op, kw)
        }
    };
    let mut parts = tuple_values(call(
        "quantize_svdquant_w4a4",
        vec![
            ("x".into(), Value::tensor(main)),
            ("smooth".into(), smooth),
            ("lora_down".into(), down),
            ("act_unsigned".into(), Value::Bool(unsigned)),
            (
                "lora_x".into(),
                if unsigned {
                    Value::tensor(x2)
                } else {
                    Value::None
                },
            ),
        ],
    )?)?;
    let la = parts.pop().unwrap();
    let xs = parts.pop().unwrap();
    let act = parts.pop().unwrap();
    let out = call(
        "scaled_mm_svdquant_w4a4",
        vec![
            ("act".into(), act),
            ("wgt".into(), Value::tensor(q.payload.shallow_clone())),
            ("ascales".into(), xs),
            ("wscales".into(), required_tensor(q, "scale")?),
            ("lora_act_in".into(), la),
            ("lora_up".into(), up),
            ("bias".into(), bias.clone()),
            ("act_unsigned".into(), Value::Bool(unsigned)),
        ],
    )?
    .as_tensor()?;
    let n = if q.payload.dim() == 4 {
        q.payload.size()[0] * 128
    } else {
        q.payload.size()[0]
    };
    let mut result = shape[..shape.len() - 1].to_vec();
    result.push(n);
    Ok(Value::tensor(
        out.f_slice(0, 0, m, 1)?.f_reshape(result.as_slice())?,
    ))
}
