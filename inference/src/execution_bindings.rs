//! Bind the existing audited implementations; preserve full reference selection.
use crate::{
    awq, awq_layout, convrot, eager,
    execution::{Callable, Decoder, Runtime, required},
    execution_value::{Kwargs, Value, get, type_error},
    layouts,
    native::Error,
    quantization,
};
use std::sync::Arc;
use tch::{Kind, Tensor};
#[path = "backend_bindings.rs"]
pub(crate) mod backends;
#[path = "layout_complete.rs"]
mod complete;
fn dtype(v: Option<&Value>, default: Kind) -> Result<Kind, Error> {
    match v {
        None | Some(Value::None) => Ok(default),
        Some(Value::DType(k)) => Ok(*k),
        _ => Err(type_error("Expected dtype")),
    }
}
fn integer(v: Option<&Value>, default: i64) -> Result<i64, Error> {
    match v {
        None => Ok(default),
        Some(Value::Int(n)) => Ok(*n),
        Some(Value::Bool(b)) => Ok(i64::from(*b)),
        _ => Err(type_error("Expected int")),
    }
}
fn tensor(a: &[Value], k: &Kwargs, i: usize, name: &str) -> Result<Tensor, Error> {
    required(a, k, i, name)?.as_tensor()
}
fn body(
    f: impl Fn(&Runtime, &[Value], &Kwargs) -> Result<Value, Error> + Send + Sync + 'static,
) -> Callable {
    Arc::new(f)
}
fn call(rt: &Runtime, op: &str, kwargs: Kwargs) -> Result<Tensor, Error> {
    unsafe { rt.call_kitchen(op, &kwargs)? }.as_tensor()
}
fn decode_backend(
    rt: &Runtime,
    q: &crate::storage::QuantizedStorage,
    op: &str,
) -> Result<Tensor, Error> {
    layouts::dequantize_with(q, |payload, p| {
        let mut kwargs = vec![
            (
                if op == "dequantize_per_tensor_fp8" {
                    "x"
                } else {
                    "q"
                }
                .into(),
                Value::tensor(payload.shallow_clone()),
            ),
            (
                "scale".into(),
                Value::tensor(p.tensors["scale"].shallow_clone()),
            ),
        ];
        if op == "dequantize_per_tensor_fp8" {
            kwargs.push(("output_type".into(), Value::DType(p.original_kind)));
        } else {
            kwargs.push((
                "output_dtype_code".into(),
                Value::Int(quantization::dtype_code(p.original_kind)?),
            ));
        }
        call(rt, op, kwargs)
    })
}

pub fn install(rt: &Runtime) {
    // Preserve every frozen handler slot. An unbound handler must not disappear
    // and silently choose the generic floating fallback instead.
    for name in [
        "AsymW4A8Int8Layout",
        "TensorCoreAWQW4A16Layout",
        "TensorCoreConvRotW4A4Layout",
        "TensorCoreFP8Layout",
        "TensorCoreMXFP8Layout",
        "TensorCoreNVFP4Layout",
        "TensorWiseINT8Layout",
        "TensorCoreSVDQuantW4A4Layout",
    ] {
        rt.register_layout(
            name,
            vec![
                name.into(),
                "QuantizedLayout".into(),
                "ABC".into(),
                "object".into(),
            ],
            None,
        );
        let ops: &[&str] = if name == "TensorCoreNVFP4Layout" {
            &["t", "mm", "linear"]
        } else if name == "TensorCoreFP8Layout" {
            &["t", "view", "reshape", "linear", "mm", "addmm"]
        } else {
            &["t", "linear", "mm", "addmm"]
        };
        for op in ops {
            let op = format!("aten::{op}.default");
            let missing = op.clone();
            rt.register_handler(
                &op,
                name,
                body(move |_, _, _| {
                    Err(Error::BindingGap {
                        backend: format!("layout:{name}"),
                        operation: missing.clone(),
                    })
                }),
            );
        }
    }
    for backend in ["eager", "cuda"] {
        let cuda = backend == "cuda";
        rt.backends.bind_reference(
            backend,
            "gemv_awq_w4a16",
            body(move |rt, a, k| {
                let x = tensor(a, k, 0, "x")?;
                let q = tensor(a, k, 1, "qweight")?;
                let s = tensor(a, k, 2, "wscales")?;
                let z = tensor(a, k, 3, "wzeros")?;
                let bias = match get(a, k, 4, "bias") {
                    None | Some(Value::None) => None,
                    Some(v) => Some(v.as_tensor()?),
                };
                let request = awq::Linear {
                    x: &x,
                    qweight: &q,
                    wscales: &s,
                    wzeros: &z,
                    bias: bias.as_ref(),
                    group_size: integer(get(a, k, 5, "group_size"), 64)?,
                };
                Ok(Value::tensor(if cuda {
                    unsafe { awq::cuda(&rt.native, &request)? }
                } else {
                    awq::eager(&request)?
                }))
            }),
        );
        rt.backends.bind_reference(
            backend,
            "dequantize_per_tensor_fp8",
            body(move |rt, a, k| {
                let x = tensor(a, k, 0, "x")?;
                let s = tensor(a, k, 1, "scale")?;
                if cuda && s.numel() != 1 {
                    return Err(Error::ReferenceException {
                        class: "AssertionError",
                        message: "Scale must be a scalar tensor".into(),
                    });
                }
                if matches!(get(a, k, 2, "output_type"), Some(Value::None)) {
                    return if cuda {
                        Err(Error::ReferenceException {
                            class: "KeyError",
                            message: "None".into(),
                        })
                    } else {
                        // torch-sys erases the C++ exception subtype. The
                        // audited boxed ABI preserves NotImplementedError too.
                        unsafe {
                            rt.ordinary(
                                "aten::mul.Tensor",
                                &[Value::tensor(x), Value::tensor(s)],
                                &Vec::new(),
                            )
                        }
                    };
                }
                let kind = dtype(get(a, k, 2, "output_type"), Kind::BFloat16)?;
                let out = if cuda {
                    unsafe { quantization::dequantize_fp8(&rt.native, &x, &s, kind)? }
                } else {
                    eager::dequantize_fp8(&x, &s, kind)?
                };
                Ok(Value::tensor(out))
            }),
        );
        for name in ["dequantize_int8_simple", "dequantize_int8_simple_dtype"] {
            rt.backends.bind_reference(
                backend,
                name,
                body(move |rt, a, k| {
                    let x = tensor(a, k, 0, "q")?;
                    let s = tensor(a, k, 1, "scale")?;
                    let code = if name.ends_with("_dtype") {
                        integer(get(a, k, 2, "output_dtype_code"), 0)?
                    } else {
                        0
                    };
                    let kind = match code {
                        0 => Kind::Float,
                        1 => Kind::Half,
                        2 => Kind::BFloat16,
                        5 => Kind::Float8e4m3fn,
                        6 => Kind::Float8e5m2,
                        _ => {
                            return Err(Error::ReferenceException {
                                class: "KeyError",
                                message: code.to_string(),
                            });
                        }
                    };
                    let out = if cuda {
                        convrot::dequantize_simple(&rt.native, &x, &s, kind)?
                    } else {
                        eager::dequantize_int8(&x, &s)?.f_to_kind(kind)?
                    };
                    Ok(Value::tensor(out))
                }),
            );
        }
    }
    backends::install(rt);
    let root = vec!["QuantizedLayout".into(), "ABC".into(), "object".into()];
    rt.register_layout("QuantizedLayout", root.clone(), None);
    let definitions: [(&str, Decoder); 3] = [
        (
            "TensorCoreAWQW4A16Layout",
            Arc::new(|_, q| awq_layout::dequantize(q)),
        ),
        (
            "TensorCoreFP8Layout",
            Arc::new(|rt, q| decode_backend(rt, q, "dequantize_per_tensor_fp8")),
        ),
        (
            "TensorWiseINT8Layout",
            Arc::new(|rt, q| {
                if q.params.fields.get("convrot") == Some(&crate::storage::ScalarField::Bool(true))
                {
                    return Err(Error::BindingGap {
                        backend: "layout:TensorWiseINT8Layout".into(),
                        operation: "dequantize_int8_convrot_weight_dtype".into(),
                    });
                }
                decode_backend(rt, q, "dequantize_int8_simple_dtype")
            }),
        ),
    ];
    for (name, decoder) in definitions {
        let mut mro = vec![name.into()];
        mro.extend(root.clone());
        rt.register_layout(name, mro, Some(decoder));
    }
    for op in [
        "aten::detach.default",
        "aten::clone.default",
        "aten::contiguous.default",
        "aten::is_contiguous.default",
        "aten::_to_copy.default",
        "aten::to.dtype",
        "aten::to.dtype_layout",
        "aten::copy_.default",
        "aten::empty_like.default",
        "aten::_has_compatible_shallow_copy_type.default",
    ] {
        rt.register_generic(op, body(move |rt, a, k| generic(rt, op, a, k)));
    }
    rt.register_handler(
        "aten::t.default",
        "TensorCoreAWQW4A16Layout",
        body(|_, a, _| {
            Ok(Value::quantized(awq_layout::transpose(
                &a[0].as_quantized()?,
            )?))
        }),
    );
    for op in [
        "aten::linear.default",
        "aten::mm.default",
        "aten::addmm.default",
    ] {
        rt.register_handler(
            op,
            "TensorCoreAWQW4A16Layout",
            body(move |rt, a, k| awq_math(rt, op, a, k)),
        );
    }
    for (op, transpose) in [
        ("aten::t.default", true),
        ("aten::view.default", false),
        ("aten::reshape.default", false),
    ] {
        rt.register_handler(
            op,
            "TensorCoreFP8Layout",
            body(move |_, a, _| {
                let q = a[0].as_quantized()?;
                let out = if transpose {
                    layouts::fp8_shape(&q, layouts::Fp8ShapeOp::Transpose)?
                } else {
                    let dims = shape(&a[1])?;
                    layouts::fp8_shape(
                        &q,
                        if op == "aten::view.default" {
                            layouts::Fp8ShapeOp::View(&dims)
                        } else {
                            layouts::Fp8ShapeOp::Reshape(&dims)
                        },
                    )?
                };
                Ok(Value::quantized(out))
            }),
        );
    }
    rt.register_handler(
        "aten::t.default",
        "TensorWiseINT8Layout",
        body(|_, a, _| {
            Ok(Value::quantized(layouts::int8_transpose(
                &a[0].as_quantized()?,
            )?))
        }),
    );
    for op in [
        "aten::linear.default",
        "aten::mm.default",
        "aten::addmm.default",
    ] {
        rt.register_handler(
            op,
            "TensorCoreFP8Layout",
            body(move |rt, a, k| fp8_math(rt, op, a, k)),
        );
        rt.register_handler(
            op,
            "TensorWiseINT8Layout",
            body(move |rt, a, k| {
                let index = if op == "aten::addmm.default" { 2 } else { 1 };
                let right = a[index].as_quantized().ok();
                if right
                    .as_ref()
                    .is_some_and(|q| q.layout == "TensorWiseINT8Layout")
                {
                    return Err(Error::BindingGap {
                        backend: "layout:TensorWiseINT8Layout".into(),
                        operation: op.into(),
                    });
                }
                let args = a
                    .iter()
                    .map(|v| unsafe { rt.dequantize_args(v) })
                    .collect::<Result<Vec<_>, _>>()?;
                let kwargs = k
                    .iter()
                    .map(|(key, v)| Ok((key.clone(), unsafe { rt.dequantize_args(v)? })))
                    .collect::<Result<Kwargs, Error>>()?;
                unsafe { rt.ordinary(op, &args, &kwargs) }
            }),
        );
    }
    complete::install(rt);
}

enum FpOperand {
    Plain(Tensor),
    Quantized(crate::storage::QuantizedStorage),
}
impl FpOperand {
    fn from(v: &Value) -> Result<Self, Error> {
        Ok(match v {
            Value::Quantized(_) => Self::Quantized(v.as_quantized()?),
            _ => Self::Plain(v.as_tensor()?),
        })
    }
    fn awq(&self) -> awq_layout::Operand<'_> {
        match self {
            Self::Plain(t) => awq_layout::Operand::Plain(t),
            Self::Quantized(q) => awq_layout::Operand::Quantized(q),
        }
    }
    fn get(&self) -> crate::fp8_ops::Operand<'_> {
        match self {
            Self::Plain(t) => crate::fp8_ops::Operand::Plain(t),
            Self::Quantized(q) => crate::fp8_ops::Operand::Quantized(q),
        }
    }
}
fn fp8_math(rt: &Runtime, op: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    use crate::fp8_ops as ops;
    let offset = usize::from(op == "aten::addmm.default");
    let left = FpOperand::from(&a[offset])?;
    let right = FpOperand::from(&a[offset + 1])?;
    let bias = if offset == 1 { Some(&a[0]) } else { a.get(2) };
    let bias = match bias {
        None | Some(Value::None) => None,
        Some(v) => Some(FpOperand::from(v)?),
    };
    let out_dtype = match k.iter().find(|(key, _)| key == "out_dtype").map(|(_, v)| v) {
        None => None,
        Some(Value::None) => Some(None),
        Some(Value::DType(d)) => Some(Some(*d)),
        _ => return Err(type_error("Expected out_dtype")),
    };
    let decoder = |q: &crate::storage::QuantizedStorage| unsafe { rt.decode(q) };
    let output = match op {
        "aten::linear.default" => ops::linear_with_bias(
            &rt.native,
            left.get(),
            right.get(),
            bias.as_ref().map(FpOperand::get),
            out_dtype,
            decoder,
        )?,
        "aten::mm.default" => ops::mm(&rt.native, left.get(), right.get(), out_dtype, decoder)?,
        _ => ops::addmm_with_bias(
            &rt.native,
            bias.as_ref().unwrap().get(),
            left.get(),
            right.get(),
            out_dtype,
            decoder,
        )?,
    };
    Ok(match output.value {
        ops::Value::Plain(t) => Value::tensor(t),
        ops::Value::Quantized(q) => Value::quantized(q),
    })
}
fn shape(v: &Value) -> Result<Vec<i64>, Error> {
    match v {
        Value::Tuple(v) | Value::List(v) => v.iter().map(|x| integer(Some(x), 0)).collect(),
        _ => Err(type_error("Expected shape sequence")),
    }
}
fn options(
    rt: &Runtime,
    a: &[Value],
    k: &Kwargs,
) -> Result<(Option<crate::native::DeviceSpec>, Option<Kind>), Error> {
    // Positional replacements precede final keyword-string conversion in base.py.
    let mut device = match get(&[], k, 0, "device") {
        Some(Value::Device(d)) => rt.native.device_spec(*d).map(Some),
        Some(Value::Text(s)) => rt.native.parse_device(s).map(Some),
        None | Some(Value::None) => Ok(None),
        _ => Err(option_gap("device argument")),
    };
    let mut kind = match get(&[], k, 0, "dtype") {
        Some(Value::DType(d)) => Ok(Some(*d)),
        None | Some(Value::None) => Ok(None),
        _ => Err(option_gap("dtype argument")),
    };
    for v in a.iter().skip(1) {
        match v {
            Value::Device(d) => device = rt.native.device_spec(*d).map(Some),
            Value::DType(d) => kind = Ok(Some(*d)),
            Value::Text(s) => {
                if let Ok(parsed) = rt.native.parse_device(s) {
                    device = Ok(Some(parsed));
                }
            }
            _ => (),
        }
    }
    Ok((device?, kind?))
}

fn option_gap(operation: &str) -> Error {
    Error::BindingGap {
        backend: "layout:generic".into(),
        operation: operation.into(),
    }
}
fn generic(rt: &Runtime, op: &str, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    if op == "aten::_has_compatible_shallow_copy_type.default" {
        return Ok(Value::Bool(true));
    }
    let first = a.first().ok_or_else(|| type_error("Missing self"))?;
    let q = first.as_quantized()?;
    Ok(match op {
        "aten::detach.default" => Value::quantized(q.detach(&rt.native)?),
        "aten::clone.default" => Value::quantized(q.clone_storage(&rt.native)?),
        "aten::is_contiguous.default" => Value::Bool(q.payload.is_contiguous()),
        "aten::contiguous.default" => match q.contiguous(&rt.native)? {
            crate::storage::StorageResult::Borrowed(_) => first.clone(),
            crate::storage::StorageResult::Owned(q) => Value::quantized(q),
        },
        "aten::to.dtype" | "aten::to.dtype_layout" | "aten::_to_copy.default" => {
            let (d, kind) = options(rt, a, k)?;
            match q.to_device_spec(&rt.native, d, kind, op == "aten::_to_copy.default")? {
                crate::storage::StorageResult::Borrowed(_) => first.clone(),
                crate::storage::StorageResult::Owned(q) => Value::quantized(q),
            }
        }
        "aten::empty_like.default" => {
            // Unlike _handle_to, the original reads only kwargs here.
            let (d, kind) = options(rt, &[], k)?;
            Value::quantized(q.empty_like_device_spec(&rt.native, d, kind)?)
        }
        "aten::copy_.default" => {
            let Some(Value::Quantized(src)) = a.get(1) else {
                return Err(type_error("Cannot copy Tensor to QuantizedTensor"));
            };
            let src = src.lock().unwrap().shallow_parts();
            if q.layout != src.layout {
                return Err(type_error(format!(
                    "Layout mismatch: {} vs {}",
                    q.layout, src.layout
                )));
            }
            let nonblocking = match get(&[], k, 0, "non_blocking") {
                Some(Value::Bool(b)) => *b,
                None => a.len() >= 3,
                _ => return Err(option_gap("copy_ non_blocking argument")),
            };
            let Value::Quantized(dst) = first else {
                unreachable!()
            };
            dst.lock()
                .unwrap()
                .copy_from(&rt.native, &src, nonblocking)?;
            first.clone()
        }
        _ => unreachable!(),
    })
}

fn awq_math(rt: &Runtime, op: &str, a: &[Value], _k: &Kwargs) -> Result<Value, Error> {
    let offset = usize::from(op == "aten::addmm.default");
    let left = FpOperand::from(&a[offset])?;
    let right = FpOperand::from(&a[offset + 1])?;
    let bias = if offset == 1 {
        Some(FpOperand::from(&a[0])?)
    } else {
        match a.get(2) {
            None | Some(Value::None) => None,
            Some(v) => Some(FpOperand::from(v)?),
        }
    };
    let mut implementation = |r: &awq::Linear<'_>| {
        let kwargs = vec![
            ("x".into(), Value::tensor(r.x.shallow_clone())),
            ("qweight".into(), Value::tensor(r.qweight.shallow_clone())),
            ("wscales".into(), Value::tensor(r.wscales.shallow_clone())),
            ("wzeros".into(), Value::tensor(r.wzeros.shallow_clone())),
            (
                "bias".into(),
                r.bias
                    .map(|t| Value::tensor(t.shallow_clone()))
                    .unwrap_or(Value::None),
            ),
            ("group_size".into(), Value::Int(r.group_size)),
        ];
        unsafe { rt.call_kitchen("gemv_awq_w4a16", &kwargs)?.as_tensor() }
    };
    let mut dispatch = awq_layout::Dispatch {
        native: &rt.native,
        backend: awq_layout::Backend::Eager,
        implementation: Some(&mut implementation),
        observer: None,
        dequantize_other: |q: &crate::storage::QuantizedStorage| unsafe { rt.decode(q) },
    };
    let output = unsafe {
        match op {
            "aten::linear.default" => {
                dispatch.linear(left.awq(), right.awq(), bias.as_ref().map(FpOperand::awq))
            }
            "aten::mm.default" => dispatch.mm(left.awq(), right.awq()),
            _ => dispatch.addmm(bias.as_ref().unwrap().awq(), left.awq(), right.awq()),
        }
    }?;
    Ok(Value::tensor(output))
}
