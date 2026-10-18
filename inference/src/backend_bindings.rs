//! Remaining executable bindings, kept separate from layout dispatch.
use super::*;
use crate::{cuda_linear, int8_eager_ops, int8_linear, native::KernelArg};
#[path = "backend_conv.rs"]
mod conv;
#[path = "backend_na.rs"]
mod na;
#[path = "backend_nvfp4.rs"]
mod nvfp4;
#[path = "backend_eager.rs"]
mod portable;
#[path = "backend_quant.rs"]
mod quant;
#[path = "backend_rope.rs"]
pub(crate) mod rope;
#[path = "backend_seed.rs"]
mod seeds;
#[path = "backend_sol.rs"]
mod sol;
#[path = "backend_sol_cuda.rs"]
pub(crate) mod sol_cuda;
#[path = "backend_w4a4_linear.rs"]
mod w4a4_linear;
#[path = "backend_w4a8.rs"]
pub(crate) mod w4a8;
#[path = "backend_w4a8_quant.rs"]
mod w4a8_quant;

fn optional_tensor(a: &[Value], k: &Kwargs, i: usize, name: &str) -> Result<Option<Tensor>, Error> {
    match get(a, k, i, name) {
        None | Some(Value::None) => Ok(None),
        Some(v) => Ok(Some(v.as_tensor()?)),
    }
}
fn boolean(v: Option<&Value>, default: bool) -> Result<bool, Error> {
    match v {
        None => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        _ => Err(type_error("Expected bool")),
    }
}
fn real(v: Option<&Value>, default: f64) -> Result<f64, Error> {
    match v {
        None => Ok(default),
        Some(Value::Float(n)) => Ok(*n),
        Some(Value::Int(n)) => Ok(*n as f64),
        _ => Err(type_error("Expected float")),
    }
}
#[path = "native_argument_signatures.rs"]
mod argument_signatures;
fn native_seed_error(name: &str, tensor: bool) -> Error {
    use argument_signatures::*;
    let (signature, invoked) = match name {
        "quantize_int8_rowwise" => (
            QUANTIZE_INT8_ROWWISE_SIGNATURE,
            QUANTIZE_INT8_ROWWISE_INVOKED,
        ),
        "quantize_int8_rowwise_convrot" => (
            QUANTIZE_INT8_ROWWISE_CONVROT_SIGNATURE,
            QUANTIZE_INT8_ROWWISE_CONVROT_INVOKED,
        ),
        "quantize_int8_rowwise_convrot64" => (
            QUANTIZE_INT8_ROWWISE_CONVROT64_SIGNATURE,
            QUANTIZE_INT8_ROWWISE_CONVROT64_INVOKED,
        ),
        "quantize_int8_convrot_staged" => (
            QUANTIZE_INT8_CONVROT_STAGED_SIGNATURE,
            QUANTIZE_INT8_CONVROT_STAGED_INVOKED,
        ),
        "quantize_int4_rowwise" => (
            QUANTIZE_INT4_ROWWISE_SIGNATURE,
            QUANTIZE_INT4_ROWWISE_INVOKED,
        ),
        "quantize_int4_rowwise_convrot64" => (
            QUANTIZE_INT4_ROWWISE_CONVROT64_SIGNATURE,
            QUANTIZE_INT4_ROWWISE_CONVROT64_INVOKED,
        ),
        "quantize_w4a8_convrot" => (
            QUANTIZE_W4A8_CONVROT_SIGNATURE,
            QUANTIZE_W4A8_CONVROT_INVOKED,
        ),
        "quantize_wxa8_convrot_fused" => (
            QUANTIZE_WXA8_CONVROT_FUSED_SIGNATURE,
            QUANTIZE_WXA8_CONVROT_FUSED_INVOKED,
        ),
        _ => unreachable!("Frozen source seed native boundary"),
    };
    let mut types = invoked.split(", ").collect::<Vec<_>>();
    if tensor {
        let index = types.iter().position(|kind| *kind == "bool").unwrap();
        types[index] = "torch.Tensor";
    }
    Error::ReferenceException {
        class: "TypeError",
        message: format!(
            "{name}(): incompatible function arguments. The following argument types are supported:\n    1. {signature}\n\nInvoked with types: {}",
            types.join(", ")
        ),
    }
}
fn native_nv_bool(value: Option<&Value>, operation: &str) -> Result<bool, Error> {
    let value = value.unwrap_or(&Value::Bool(true));
    if let Value::Bool(b) = value {
        return Ok(*b);
    }
    use argument_signatures::*;
    let (signature, invoked, index) = match operation {
        "quantize_nvfp4" => (QUANTIZE_NVFP4_SIGNATURE, QUANTIZE_NVFP4_INVOKED, 6),
        "dequantize_nvfp4" => (DEQUANTIZE_NVFP4_SIGNATURE, DEQUANTIZE_NVFP4_INVOKED, 5),
        _ => unreachable!(),
    };
    let mut types = invoked.split(", ").collect::<Vec<_>>();
    types[index] = if matches!(value, Value::Tensor(_)) {
        "torch.Tensor"
    } else {
        crate::backend_triton_fp8::typename(value)
    };
    Err(Error::ReferenceException {
        class: "TypeError",
        message: format!(
            "{operation}(): incompatible function arguments. The following argument types are supported:\n    1. {signature}\n\nInvoked with types: {}",
            types.join(", ")
        ),
    })
}
fn cuda_seed(rt: &Runtime, value: Option<&Value>, operation: &str) -> Result<(bool, i64), Error> {
    let value = value.cloned().unwrap_or(Value::Int(0));
    let tensor = matches!(&value, Value::Tensor(_) | Value::Quantized(_));
    let stochastic = if tensor {
        unsafe { rt.op("aten::gt.Scalar", &[value.clone(), Value::Int(0)], &vec![])? };
        false
    } else {
        crate::backend_triton_rowwise::positive(rt, &value)?
    };
    let raw = if crate::backend_triton_mxfp8::truthy(rt, &value)? {
        value
    } else {
        Value::Int(0)
    };
    let seed = seeds::Seed::Raw(raw).integer(rt)?;
    let parsed = u64::try_from(seed);
    if tensor || parsed.is_err() {
        return Err(native_seed_error(operation, tensor));
    }
    Ok((stochastic, parsed.unwrap() as i64))
}
fn cuda_rowwise_seed(rt: &Runtime, value: Option<&Value>) -> Result<(bool, i64), Error> {
    cuda_seed(rt, value, "quantize_int8_rowwise")
}
fn eager_seed(rt: &Runtime, value: Option<&Value>) -> Result<Option<u64>, Error> {
    let value = value.unwrap_or(&Value::Int(0));
    if !crate::backend_triton_rowwise::positive(rt, value)? {
        return Ok(None);
    }
    match value {
        Value::Int(seed) => Ok(Some(*seed as u64)),
        Value::UInt(seed) => Ok(Some(*seed)),
        _ => Err(Error::ReferenceException {
            class: "RuntimeError",
            message: format!(
                "manual_seed expected a long, but got {}",
                crate::backend_triton_fp8::typename(value)
            ),
        }),
    }
}
fn pair((a, b): (Tensor, Tensor)) -> Value {
    Value::Tuple(vec![Value::tensor(a), Value::tensor(b)])
}
fn code_dtype(v: Option<&Value>) -> Result<Kind, Error> {
    match integer(v, 0)? {
        0 => Ok(Kind::Float),
        1 => Ok(Kind::Half),
        2 => Ok(Kind::BFloat16),
        5 => Ok(Kind::Float8e4m3fn),
        6 => Ok(Kind::Float8e5m2),
        n => Err(Error::ReferenceException {
            class: "KeyError",
            message: n.to_string(),
        }),
    }
}
fn stream(rt: &Runtime, t: &Tensor) -> Result<i64, Error> {
    match t.device() {
        tch::Device::Cuda(i) => Ok(rt.native.current_stream(i)? as i64),
        _ => Err(Error::BindingGap {
            backend: "cuda".into(),
            operation: "non-CUDA direct call".into(),
        }),
    }
}
fn fp8_code(k: Kind) -> Result<i64, Error> {
    match k {
        Kind::Float8e4m3fn => Ok(5),
        Kind::Float8e5m2 => Ok(6),
        _ => quantization::dtype_code(k),
    }
}
fn assertion(condition: bool, message: impl Into<String>) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error::ReferenceException {
            class: "AssertionError",
            message: message.into(),
        })
    }
}

fn input_fuses(
    rt: &Runtime,
    cuda: bool,
    x: &Tensor,
    a: &[Value],
    k: &Kwargs,
) -> Result<bool, Error> {
    if cuda && let tch::Device::Cuda(device) = x.device() {
        let raw_width = x.size().last().copied().unwrap_or(0);
        let rows = if raw_width == 0 {
            0
        } else {
            x.numel() as i64 / raw_width
        };
        let width = raw_width
            / if matches!(get(a,k,7,"input_act"),Some(Value::Text(s)) if s == "swiglu" || s == "situ")
            {
                2
            } else {
                1
            };
        cuda_linear::input_activation_fuses(
            &rt.native,
            device,
            rows,
            width,
            integer(get(a, k, 6, "convrot_groupsize"), 256)?,
            boolean(get(a, k, 5, "convrot"), false)?,
        )
    } else {
        Ok(false)
    }
}
fn input_epsilon(rt: &Runtime, fused: bool, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let value = get(a, k, 9, "input_act_eps").unwrap_or(&Value::Float(0.));
    if fused {
        Ok(Value::Float(crate::python_numbers::float_value(rt, value)?))
    } else {
        crate::backend_triton_int8::eps(rt, value)
    }
}
pub(super) fn install(rt: &Runtime) {
    na::install(rt);
    rope::install(rt);
    conv::install(rt);
    quant::install(rt);
    sol::install(rt);
    w4a4_linear::install(rt);
    w4a8::install(rt);
    w4a8_quant::install(rt);
    nvfp4::install(rt);
    sol_cuda::install(rt);
    for backend in ["eager", "cuda"] {
        let cuda = backend == "cuda";
        for op in [
            "quantize_per_tensor_fp8",
            "quantize_int8_tensorwise",
            "quantize_int8_rowwise",
            "quantize_int8_convrot_weight",
            "rotate_int8_convrot_weight",
            "quantize_and_rotate_rowwise",
            "dequantize_int8_convrot_weight",
            "dequantize_int8_convrot_weight_dtype",
            "quantize_mxfp8",
            "dequantize_nvfp4",
            "fp16_linear",
            "int8_linear",
            "adaln",
            "rms_adaln",
        ] {
            rt.backends
                .bind_reference(backend, op, body(move |rt, a, k| run(rt, cuda, op, a, k)));
        }
    }
    for op in [
        "quantize_nvfp4",
        "stochastic_rounding_fp8",
        "quantize_convrot_w4a4_weight",
        "dequantize_convrot_w4a4_weight",
        "prepare_int4_weight_for_int8_linear",
        "quantize_svdquant_w4a4",
        "scaled_mm_svdquant_w4a4",
    ] {
        rt.backends
            .bind_reference("cuda", op, body(move |rt, a, k| run(rt, true, op, a, k)));
    }
    for op in [
        "dequantize_mxfp8",
        "dequantize_int8_embedding",
        "stochastic_rounding_fp8",
        "quantize_convrot_w4a4_weight",
        "dequantize_convrot_w4a4_weight",
        "prepare_int4_weight_for_int8_linear",
    ] {
        rt.backends
            .bind_reference("eager", op, body(move |rt, a, k| run(rt, false, op, a, k)));
    }
}

fn run(rt: &Runtime, cuda: bool, op: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    use KernelArg::{Int, Tensor as T};
    let first_name = match op {
        "quantize_int8_convrot_weight"
        | "rotate_int8_convrot_weight"
        | "quantize_convrot_w4a4_weight"
        | "prepare_int4_weight_for_int8_linear" => "weight",
        "dequantize_int8_convrot_weight"
        | "dequantize_int8_convrot_weight_dtype"
        | "dequantize_int8_embedding" => "q",
        "dequantize_nvfp4" | "dequantize_mxfp8" => "qx",
        "dequantize_convrot_w4a4_weight" => "qdata",
        "scaled_mm_svdquant_w4a4" => "act",
        _ => "x",
    };
    let x = tensor(a, k, 0, first_name)?;
    let out = match op {
        "quantize_svdquant_w4a4" => {
            let smooth = tensor(a, k, 1, "smooth")?;
            let down = tensor(a, k, 2, "lora_down")?;
            let source = optional_tensor(a, k, 5, "lora_x")?;
            let req = crate::svdquant::Quantize {
                x: &x,
                smooth: &smooth,
                lora_down: &down,
                pad_size: integer(get(a, k, 3, "pad_size"), 256)?,
                act_unsigned: boolean(get(a, k, 4, "act_unsigned"), false)?,
                lora_x: source.as_ref(),
            };
            let settings = crate::svdquant::QuantizeSettings {
                lora_act_fp32: std::env::var("COMFY_KITCHEN_SVDQUANT_LORA_ACT_FP32").ok(),
                reuse_workspace: std::env::var("COMFY_KITCHEN_SVDQUANT_REUSE_WORKSPACE").ok(),
            };
            let (q, s, l) = unsafe { crate::svdquant::quantize(&rt.native, &req, &settings)? };
            return Ok(Value::Tuple(vec![
                Value::tensor(q),
                Value::tensor(s),
                Value::tensor(l),
            ]));
        }
        "scaled_mm_svdquant_w4a4" => {
            let w = tensor(a, k, 1, "wgt")?;
            let asc = tensor(a, k, 2, "ascales")?;
            let wsc = tensor(a, k, 3, "wscales")?;
            let la = tensor(a, k, 4, "lora_act_in")?;
            let b = optional_tensor(a, k, 6, "bias")?;
            let Value::Tensor(up) = required(a, k, 5, "lora_up")? else {
                return Err(type_error("Expected lora_up Tensor"));
            };
            let mut object = up.lock().unwrap();
            let current = object.tensor.shallow_clone();
            let up = object
                .lora_up
                .get_or_insert_with(|| crate::svdquant::LoraUp::new(current.shallow_clone()));
            up.tensor = current;
            let settings = crate::svdquant::Settings {
                fast_accum: std::env::var("COMFY_KITCHEN_SVDQUANT_FAST_ACCUM").ok(),
                shared_scale: std::env::var("COMFY_KITCHEN_SVDQUANT_SHARED_SCALE").ok(),
                fuse_lora_up: std::env::var("COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP").ok(),
            };
            let mut req = crate::svdquant::ScaledMm {
                act: &x,
                weight: &w,
                act_scales: &asc,
                weight_scales: &wsc,
                lora_act: &la,
                lora_up: up,
                bias: b.as_ref(),
                act_unsigned: boolean(get(a, k, 7, "act_unsigned"), false)?,
            };
            unsafe { crate::svdquant::scaled_mm(&rt.native, &mut req, &settings)? }
        }
        "quantize_per_tensor_fp8" => {
            let scale = tensor(a, k, 1, "scale")?;
            if matches!(get(a, k, 2, "output_type"), Some(Value::None)) {
                return Err(Error::ReferenceException {
                    class: if cuda { "KeyError" } else { "ValueError" },
                    message: if cuda {
                        "None".into()
                    } else {
                        "Unsupported output_type: None. Expected torch.float8_e4m3fn or torch.float8_e5m2".into()
                    },
                });
            }
            let kind = dtype(get(a, k, 2, "output_type"), Kind::Float8e4m3fn)?;
            if cuda {
                let input_code = quantization::dtype_code(x.kind())?;
                let output_code = fp8_code(kind)?;
                let x = x.f_contiguous()?;
                let out = Tensor::f_empty(x.size(), (Kind::Uint8, x.device()))?;
                unsafe {
                    rt.native.call_unchecked(
                        "quantize_per_tensor_fp8",
                        &[
                            T(&x),
                            T(&scale),
                            T(&out),
                            Int(input_code),
                            Int(output_code),
                            Int(x.numel() as i64),
                            Int(stream(rt, &x)?),
                        ],
                    )?;
                }
                out.f_view_dtype(kind)?
            } else {
                eager::quantize_fp8(&x, &scale, kind)?
            }
        }
        "stochastic_rounding_fp8" => {
            let rng = tensor(a, k, 1, "rng")?;
            if matches!(get(a, k, 2, "output_type"), Some(Value::None)) {
                return Err(Error::ReferenceException {
                    class: if cuda { "KeyError" } else { "ValueError" },
                    message: if cuda {
                        "None".into()
                    } else {
                        "Unsupported output_type: None. Expected torch.float8_e4m3fn or torch.float8_e5m2".into()
                    },
                });
            }
            let kind = dtype(get(a, k, 2, "output_type"), Kind::Float8e4m3fn)?;
            if !cuda {
                return Ok(Value::tensor(portable::stochastic_fp8(&x, &rng, kind)?));
            }
            let code = fp8_code(kind)?;
            let x = x.f_contiguous()?;
            let rng = rng.f_contiguous()?;
            unsafe {
                rt.native.call_unchecked(
                    "stochastic_round_fp8",
                    &[
                        T(&rng),
                        T(&x),
                        Int(code),
                        Int(x.numel() as i64),
                        Int(stream(rt, &x)?),
                    ],
                )?;
            }
            rng.f_view_dtype(kind)?
        }
        "quantize_int8_tensorwise" => {
            let scale = match get(a, k, 1, "scale") {
                None | Some(Value::None) => None,
                Some(Value::Text(s)) if s == "recalculate" => None,
                Some(Value::Tensor(_)) => Some(tensor(a, k, 1, "scale")?),
                Some(Value::Float(n)) => {
                    Some(Tensor::from(*n).f_to_device_(x.device(), Kind::Float, false, false)?)
                }
                Some(Value::Int(n)) => {
                    Some(Tensor::from(*n).f_to_device_(x.device(), Kind::Float, false, false)?)
                }
                Some(Value::Text(_)) => return Err(type_error("new(): invalid data type 'str'")),
                _ => {
                    return Err(Error::BindingGap {
                        backend: "eager".into(),
                        operation: "tensorwise scale argument".into(),
                    });
                }
            };
            // CUDA delegates to this exact eager body in the reference too.
            let (q, s) =
                eager::quantize_int8_tensorwise_with_seed(&rt.native, &x, scale.as_ref(), || {
                    eager_seed(rt, get(a, k, 2, "stochastic_rounding"))
                })?;
            // Tensor.to with unchanged device/dtype returns the original Python
            // object, not merely another tensor sharing the scale's storage.
            let scale_value = match get(a, k, 1, "scale") {
                Some(v @ Value::Tensor(_))
                    if scale
                        .as_ref()
                        .is_some_and(|s| s.kind() == Kind::Float && s.device() == x.device()) =>
                {
                    v.clone()
                }
                _ => Value::tensor(s),
            };
            return Ok(Value::Tuple(vec![Value::tensor(q), scale_value]));
        }
        "quantize_int8_rowwise" => {
            return Ok(pair(if cuda {
                quantization::int8_rowwise_with_seed(&rt.native, &x, || {
                    cuda_rowwise_seed(rt, get(a, k, 1, "stochastic_rounding"))
                })?
            } else {
                eager::quantize_int8_rowwise_with_seed(&rt.native, &x, || {
                    eager_seed(rt, get(a, k, 1, "stochastic_rounding"))
                })?
            }));
        }
        "quantize_int8_convrot_weight" => {
            let group = integer(get(a, k, 1, "group_size"), 0)?;
            return Ok(pair(if cuda {
                unsafe {
                    convrot::quantize_weight_with_seed(&rt.native, &x, group, |name| {
                        cuda_seed(rt, get(a, k, 2, "stochastic_rounding"), name).map(
                            |(enabled, seed)| crate::rng_seed::NativeSeed {
                                enabled,
                                bits: seed as u64,
                            },
                        )
                    })?
                }
            } else {
                eager::quantize_int8_rowwise_with_seed(
                    &rt.native,
                    &int8_eager_ops::rotate_weight(&x, group)?,
                    || eager_seed(rt, get(a, k, 2, "stochastic_rounding")),
                )?
            }));
        }
        "rotate_int8_convrot_weight" => {
            let group = integer(get(a, k, 1, "group_size"), 0)?;
            if cuda {
                unsafe { convrot::rotate_weight(&rt.native, &x, group)? }
            } else {
                int8_eager_ops::rotate_weight(&x, group)?
            }
        }
        "quantize_and_rotate_rowwise" => {
            let h = tensor(a, k, 1, "h")?;
            let group = integer(get(a, k, 2, "group_size"), 0)?;
            return Ok(pair(if cuda {
                quantization::int8_rowwise_with_seed(
                    &rt.native,
                    &crate::int8_utils::rotate_activation(&x, &h, group)?,
                    || cuda_rowwise_seed(rt, get(a, k, 3, "stochastic_rounding")),
                )?
            } else {
                eager::quantize_int8_rowwise_with_seed(
                    &rt.native,
                    &crate::int8_utils::rotate_activation(&x, &h, group)?,
                    || eager_seed(rt, get(a, k, 3, "stochastic_rounding")),
                )?
            }));
        }
        "dequantize_int8_convrot_weight" | "dequantize_int8_convrot_weight_dtype" => {
            let scale = tensor(a, k, 1, "scale")?;
            let group = integer(get(a, k, 2, "group_size"), 0)?;
            let kind = if op.ends_with("_dtype") {
                code_dtype(get(a, k, 3, "output_dtype_code"))?
            } else {
                Kind::Float
            };
            if cuda {
                unsafe { convrot::dequantize_weight(&rt.native, &x, &scale, group, kind)? }
            } else {
                int8_eager_ops::dequantize_weight(&x, &scale, group, kind)?
            }
        }
        "quantize_mxfp8" => {
            if cuda {
                assertion(x.is_contiguous(), "Input tensor must be contiguous")?;
                if !boolean(get(a, k, 1, "pad_32x"), false)? {
                    assertion(
                        x.size()[0] % 32 == 0,
                        format!("num_rows must be divisible by 32, got {}", x.size()[0]),
                    )?;
                    assertion(
                        x.size()[1] % 32 == 0,
                        format!("num_cols must be divisible by 32, got {}", x.size()[1]),
                    )?;
                }
            }
            return Ok(pair(if cuda {
                quantization::mxfp8(&rt.native, &x, boolean(get(a, k, 1, "pad_32x"), false)?)?
            } else {
                eager::quantize_mxfp8(&rt.native, &x, boolean(get(a, k, 1, "pad_32x"), false)?)?
            }));
        }
        "dequantize_mxfp8" => eager::dequantize_mxfp8(
            &x,
            &tensor(a, k, 1, "block_scales")?,
            if matches!(get(a, k, 2, "output_type"), Some(Value::None)) {
                Kind::Float
            } else {
                dtype(get(a, k, 2, "output_type"), Kind::BFloat16)?
            },
        )?,
        "quantize_nvfp4" => {
            let s = tensor(a, k, 1, "per_tensor_scale")?;
            let pad = boolean(get(a, k, 3, "pad_16x"), false)?;
            assertion(x.is_contiguous(), "Input tensor must be contiguous")?;
            if !pad {
                assertion(
                    x.size()[0] % 16 == 0,
                    format!("num_rows must be divisible by 16, got {}", x.size()[0]),
                )?;
                assertion(
                    x.size()[1] % 16 == 0,
                    format!("num_cols must be divisible by 16, got {}", x.size()[1]),
                )?;
            }
            return Ok(pair(unsafe {
                quantization::nvfp4_with_hi(
                    &rt.native,
                    &x,
                    &s,
                    real(get(a, k, 2, "epsilon"), 0.)?,
                    pad,
                    || native_nv_bool(get(a, k, 4, "hi_first"), "quantize_nvfp4"),
                )?
            }));
        }
        "dequantize_nvfp4" => {
            let s = tensor(a, k, 1, "per_tensor_scale")?;
            let blocks = tensor(a, k, 2, "block_scales")?;
            if matches!(get(a, k, 3, "output_type"), Some(Value::None)) {
                if cuda {
                    assertion(x.is_contiguous(), "Input tensor must be contiguous")?;
                    let [m, n]: [i64; 2] = x
                        .size()
                        .try_into()
                        .map_err(|_| type_error("Expected two tensor dimensions"))?;
                    let _output =
                        Tensor::f_empty([m, n * 2], (rt.native.default_dtype()?, x.device()))?;
                    let _scale = if s.dim() == 0 {
                        s.f_reshape([1])?
                    } else {
                        s.shallow_clone()
                    };
                    let _blocks = blocks.f_view_dtype(Kind::Uint8)?;
                    return Err(Error::ReferenceException {
                        class: "KeyError",
                        message: "None".into(),
                    });
                }
                return Ok(Value::tensor(eager::dequantize_nvfp4_optional(
                    &x,
                    &s,
                    &blocks,
                    None,
                    || {
                        crate::backend_triton_mxfp8::truthy(
                            rt,
                            get(a, k, 4, "hi_first").unwrap_or(&Value::Bool(true)),
                        )
                    },
                    |a, b| {
                        unsafe {
                            rt.ordinary(
                                "aten::mul.Tensor",
                                &[
                                    Value::tensor(a.shallow_clone()),
                                    Value::tensor(b.shallow_clone()),
                                ],
                                &vec![],
                            )
                        }?
                        .as_tensor()
                    },
                )?));
            }
            let kind = dtype(get(a, k, 3, "output_type"), Kind::BFloat16)?;
            let hi = get(a, k, 4, "hi_first");
            if cuda {
                assertion(x.is_contiguous(), "Input tensor must be contiguous")?;
                unsafe {
                    quantization::dequantize_nvfp4_with_hi(
                        &rt.native,
                        &x,
                        &s,
                        &blocks,
                        kind,
                        || native_nv_bool(hi, "dequantize_nvfp4"),
                    )?
                }
            } else {
                eager::dequantize_nvfp4_optional(
                    &x,
                    &s,
                    &blocks,
                    Some(kind),
                    || crate::backend_triton_mxfp8::truthy(rt, hi.unwrap_or(&Value::Bool(true))),
                    |a, b| Ok::<_, Error>(a.f_mul(b)?),
                )?
            }
        }
        "dequantize_int8_embedding" => int8_eager_ops::embedding(
            &rt.native,
            &x,
            &tensor(a, k, 1, "scale")?,
            &tensor(a, k, 2, "indices")?,
            integer(get(a, k, 3, "group_size"), 0)?,
            code_dtype(get(a, k, 4, "output_dtype_code"))?,
        )?,
        "quantize_convrot_w4a4_weight" if !cuda => {
            return Ok(pair(portable::quantize_int4_weight(
                rt,
                &x,
                integer(get(a, k, 1, "convrot_groupsize"), 256)?,
                integer(get(a, k, 2, "quant_group_size"), 64)?,
                eager_seed(rt, get(a, k, 3, "stochastic_rounding"))?,
            )?));
        }
        "quantize_convrot_w4a4_weight" => {
            return Ok(pair(unsafe {
                crate::convrot_w4a4::quantize_weight_with_seed(
                    &rt.native,
                    &x,
                    integer(get(a, k, 1, "convrot_groupsize"), 256)?,
                    integer(get(a, k, 2, "quant_group_size"), 64)?,
                    |name| {
                        cuda_seed(rt, get(a, k, 3, "stochastic_rounding"), name).map(
                            |(enabled, seed)| crate::rng_seed::NativeSeed {
                                enabled,
                                bits: seed as u64,
                            },
                        )
                    },
                )?
            }));
        }
        "dequantize_convrot_w4a4_weight" if !cuda => portable::dequantize_int4_weight(
            &x,
            &tensor(a, k, 1, "scales")?,
            integer(get(a, k, 2, "convrot_groupsize"), 256)?,
            integer(get(a, k, 3, "quant_group_size"), 64)?,
            dtype(get(a, k, 4, "output_dtype"), Kind::Float)?,
        )?,
        "dequantize_convrot_w4a4_weight" => unsafe {
            crate::convrot_w4a4::dequantize_weight(
                &rt.native,
                &x,
                &tensor(a, k, 1, "scales")?,
                integer(get(a, k, 2, "convrot_groupsize"), 256)?,
                integer(get(a, k, 3, "quant_group_size"), 64)?,
                dtype(get(a, k, 4, "output_dtype"), Kind::Float)?,
            )?
        },
        "prepare_int4_weight_for_int8_linear" if !cuda => {
            if x.dim() != 2 {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "prepared INT4 fallback weight expects a 2D tensor".into(),
                });
            }
            crate::packing::unpack_int4_row_major(&x, true)?.f_to_kind(Kind::Int8)?
        }
        "prepare_int4_weight_for_int8_linear" => unsafe {
            crate::convrot_w4a4::prepare_int8_weight(&rt.native, &x)?
        },
        "adaln" | "rms_adaln" => portable::adaln(
            rt,
            &x,
            &tensor(a, k, 1, "scale")?,
            &tensor(a, k, 2, "shift")?,
            real(get(a, k, 3, "eps"), 1e-6)?,
            op == "rms_adaln",
            cuda,
        )?,
        "fp16_linear" => {
            let w = tensor(a, k, 1, "weight")?;
            let b = optional_tensor(a, k, 2, "bias")?;
            let r = optional_tensor(a, k, 3, "residual")?;
            let rs = optional_tensor(a, k, 4, "residual_scale")?;
            if cuda {
                unsafe {
                    cuda_linear::fp16_linear(
                        &rt.native,
                        &x,
                        &w,
                        b.as_ref(),
                        r.as_ref(),
                        rs.as_ref(),
                    )?
                }
            } else {
                int8_linear::fp16_linear(&x, &w, b.as_ref(), r.as_ref(), rs.as_ref())?
            }
        }
        "int8_linear" => {
            let w = tensor(a, k, 1, "weight")?;
            let s = tensor(a, k, 2, "weight_scale")?;
            let b = optional_tensor(a, k, 3, "bias")?;
            let fused = input_fuses(rt, cuda, &x, a, k)?;
            let mode = get(a, k, 7, "input_act").unwrap_or(&Value::None);
            let rms = matches!(mode,Value::Text(s) if s == "rms_norm");
            let actw = if rms {
                match get(a, k, 8, "input_act_weight").unwrap_or(&Value::None) {
                    Value::None if !fused => {
                        return Err(Error::ReferenceException {
                            class: "ValueError",
                            message: "input_act 'rms_norm' requires act_weight".into(),
                        });
                    }
                    Value::None => None,
                    weight => {
                        crate::backend_triton_fp8::info(weight, "to")?;
                        let converted = unsafe {
                            rt.op(
                                "aten::to.dtype",
                                &[weight.clone(), Value::DType(x.kind())],
                                &vec![],
                            )?
                        };
                        Some(if fused {
                            let moved = unsafe {
                                rt.op(
                                    "aten::to.dtype_layout",
                                    &[converted],
                                    &vec![
                                        ("device".into(), Value::Device(x.device())),
                                        ("dtype".into(), Value::DType(x.kind())),
                                    ],
                                )?
                            };
                            unsafe { rt.op("aten::contiguous.default", &[moved], &vec![])? }
                        } else {
                            converted
                        })
                    }
                }
            } else {
                None
            };
            // Raw fused export precedes float(eps); ordinary RMS argument
            // parsing precedes dispatch/dequantization of a quantized weight.
            let fused_weight = if fused {
                actw.as_ref().map(Value::as_cuda_export).transpose()?
            } else {
                None
            };
            let rms_epsilon = if rms {
                Some(input_epsilon(rt, fused, a, k)?)
            } else {
                None
            };
            let actw = if fused {
                fused_weight
            } else {
                actw.as_ref()
                    .map(|v| unsafe { rt.dequantize_args(v)?.as_tensor() })
                    .transpose()?
            };
            let activation = match mode {
                Value::None => convrot::InputActivation::None,
                Value::Text(s) if s == "none" => convrot::InputActivation::None,
                Value::Text(s) if s == "gelu_tanh" => convrot::InputActivation::GeluTanh,
                Value::Text(s) if s == "swiglu" => convrot::InputActivation::SwiGlu,
                Value::Text(s) if s == "situ" => convrot::InputActivation::Situ,
                Value::Text(s) if s == "rms_norm" => match rms_epsilon.unwrap() {
                    Value::None => convrot::InputActivation::RmsNormDefault {
                        weight: actw.as_ref(),
                    },
                    Value::Float(epsilon) => convrot::InputActivation::RmsNorm {
                        weight: actw.as_ref(),
                        epsilon,
                    },
                    _ => unreachable!("Original optional float epsilon parser"),
                },
                mode => {
                    if fused {
                        crate::python_value::ensure_hashable(mode)?;
                    }
                    return Err(Error::ReferenceException {
                        class: "ValueError",
                        message: format!(
                            "unsupported input_act: {} (expected one of ['gelu_tanh', 'none', 'rms_norm', 'situ', 'swiglu'])",
                            crate::python_value::repr_with_native(mode, &rt.native)?
                        ),
                    });
                }
            };
            // Both fused quantizers, including the ConvRot M=1 GEMV, convert
            // eps even when the selected activation ignores it.
            if fused && !rms {
                input_epsilon(rt, true, a, k)?;
            }
            let residual = optional_tensor(a, k, 10, "residual")?;
            let rs = optional_tensor(a, k, 11, "residual_scale")?;
            let kind = dtype(
                get(a, k, 4, "out_dtype"),
                if cuda {
                    x.kind()
                } else if matches!(get(a, k, 4, "out_dtype"), Some(Value::None)) {
                    Kind::Float
                } else {
                    Kind::BFloat16
                },
            )?;
            let request = int8_linear::Linear {
                x: &x,
                weight: &w,
                weight_scale: &s,
                bias: b.as_ref(),
                output_kind: kind,
                convrot: boolean(get(a, k, 5, "convrot"), false)?,
                convrot_group: integer(get(a, k, 6, "convrot_groupsize"), 256)?,
                input_activation: activation,
                residual: residual.as_ref(),
                residual_scale: rs.as_ref(),
            };
            if cuda {
                unsafe { cuda_linear::int8_linear(&rt.native, &request, rt.disable_cutlass_int8)? }
            } else {
                int8_linear::linear(&rt.native, &request)?
            }
        }
        _ => unreachable!(),
    };
    Ok(Value::tensor(out))
}
