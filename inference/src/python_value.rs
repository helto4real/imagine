//! Python scalar/container formatting at frozen argument and cache boundaries.
use crate::{execution_value::Value, native::Error, triton_specialization::python_str_repr};
pub(crate) fn display(v: &Value) -> Result<String, Error> {
    match v {
        Value::Text(text) => Ok(text.clone()),
        Value::Device(device) => Ok(match device {
            tch::Device::Cuda(i) => format!("cuda:{i}"),
            device => crate::execution_value::device_name(*device),
        }),
        _ => repr(v),
    }
}
pub(crate) fn repr(v: &Value) -> Result<String, Error> {
    repr_inner(v, None)
}
pub(crate) fn repr_with_native(
    v: &Value,
    native: &crate::native::Kitchen,
) -> Result<String, Error> {
    repr_inner(v, Some(native))
}
fn repr_inner(v: &Value, native: Option<&crate::native::Kitchen>) -> Result<String, Error> {
    Ok(match v {
        Value::None => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Int(n) => n.to_string(),
        Value::UInt(n) => n.to_string(),
        Value::Float(n) => crate::registry::python_float(*n),
        Value::Text(s) => python_str_repr(s),
        Value::DType(k) => crate::registry::AllowedType::DType(*k).python_name().into(),
        Value::Device(d) => format!(
            "device(type='{}'{})",
            match d {
                tch::Device::Cpu => "cpu",
                tch::Device::Cuda(_) => "cuda",
                tch::Device::Mps => "mps",
                tch::Device::Vulkan => "vulkan",
            },
            if let tch::Device::Cuda(i) = d {
                format!(", index={i}")
            } else {
                String::new()
            }
        ),
        Value::List(v) => format!(
            "[{}]",
            v.iter()
                .map(|v| repr_inner(v, native))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        Value::Tuple(v) => format!(
            "({}{})",
            v.iter()
                .map(|v| repr_inner(v, native))
                .collect::<Result<Vec<_>, _>>()?
                .join(", "),
            if v.len() == 1 { "," } else { "" }
        ),
        Value::NamedTuple(object)
            if std::ptr::eq(object.class, &crate::execution_value::TORCH_SIZE_CLASS) =>
        {
            format!(
                "torch.Size([{}])",
                object
                    .fields
                    .iter()
                    .map(|(_, v)| repr_inner(v, native))
                    .collect::<Result<Vec<_>, Error>>()?
                    .join(", ")
            )
        }
        Value::NamedTuple(object) => format!(
            "{}({})",
            object.class.name,
            object
                .fields
                .iter()
                .map(|(name, v)| Ok(format!("{name}={}", repr_inner(v, native)?)))
                .collect::<Result<Vec<_>, Error>>()?
                .join(", ")
        ),
        Value::Dict(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(k, v)| Ok(format!(
                    "{}: {}",
                    python_str_repr(k),
                    repr_inner(v, native)?
                )))
                .collect::<Result<Vec<_>, Error>>()?
                .join(", ")
        ),
        Value::Quantized(q) => {
            let q = q.lock().unwrap();
            let tuple = |s: &[i64]| {
                format!(
                    "({}{})",
                    s.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
                    if s.len() == 1 { "," } else { "" }
                )
            };
            format!(
                "QuantizedTensor(shape={}, storage_shape={}, layout={}, dtype={})",
                tuple(&q.params.original_shape),
                tuple(&q.payload.size()),
                q.layout,
                crate::registry::AllowedType::DType(q.params.original_kind).python_name()
            )
        }
        Value::Tensor(_) if native.is_some() => return scalar_tensor_display(v, native.unwrap()),
        Value::Tensor(_) => {
            return Err(Error::BindingGap {
                backend: "python argument formatting".into(),
                operation: "tensor representation".into(),
            });
        }
    })
}

/// Tensor constexpr iteration yields zero-dimensional Tensor objects. Preserve
/// the original default dense scalar representation in ASTSource cache keys.
pub(crate) fn scalar_tensor_display(
    value: &Value,
    native: &crate::native::Kitchen,
) -> Result<String, Error> {
    crate::python_tensor_repr::display(&value.as_tensor()?, native)
}

pub(crate) fn constexpr_key(
    value: &Value,
    native: &crate::native::Kitchen,
) -> Result<crate::triton_arguments::ConstexprKey, Error> {
    fn memory(value: &Value) -> Result<String, Error> {
        Ok(match value {
            Value::Bool(b) => format!("number:{}", u8::from(*b)),
            Value::Int(n) => format!("number:{n}"),
            Value::UInt(n) => format!("number:{n}"),
            Value::Float(n) if *n == 0. => "number:0".into(),
            Value::Float(n) if n.is_finite() && n.fract() == 0. => format!("number:{n:.0}"),
            Value::Float(n) => format!("float:{:016x}", n.to_bits()),
            Value::None => "none".into(),
            Value::Text(s) => format!("str:{s}"),
            Value::Tensor(t) => format!("tensor:{:p}", std::sync::Arc::as_ptr(t)),
            Value::Tuple(items) => format!(
                "tuple:{}",
                serde_json::to_string(&items.iter().map(memory).collect::<Result<Vec<_>, _>>()?)
                    .unwrap()
            ),
            Value::NamedTuple(object) => memory(&Value::Tuple(
                object.fields.iter().map(|(_, v)| v.clone()).collect(),
            ))?,
            Value::List(_) | Value::Dict(_) => {
                return Err(crate::execution_value::type_error(format!(
                    "unhashable type: '{}'",
                    crate::backend_triton_fp8::typename(value)
                )));
            }
            _ => format!("object:{}", repr(value)?),
        })
    }
    // Python dictionary lookup hashes the complete key before AST source repr.
    let memory = serde_json::to_string(&memory(value)?).unwrap();
    Ok(crate::triton_arguments::ConstexprKey {
        source: if matches!(value, Value::Text(_) | Value::Device(_)) {
            display(value)?
        } else {
            repr_with_native(value, native)?
        },
        memory,
    })
}

pub(crate) fn ensure_hashable(value: &Value) -> Result<(), Error> {
    match value {
        Value::List(_) | Value::Dict(_) => Err(crate::execution_value::type_error(format!(
            "unhashable type: '{}'",
            crate::backend_triton_fp8::typename(value)
        ))),
        Value::NamedTuple(v) => {
            for (_, v) in &v.fields {
                ensure_hashable(v)?;
            }
            Ok(())
        }
        Value::Tuple(v) => {
            for value in v {
                ensure_hashable(value)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
