//! Original native JIT numeric specialization and typed argument packing.
use crate::{
    backend_triton_fp8::typename,
    execution_value::{Value, type_error},
    native::Error,
    triton_driver::Scalar,
    triton_frontend::scalar::ScalarType,
};
pub(crate) fn numeric(value: &Value) -> Result<(ScalarType, Option<Scalar>), Error> {
    Ok(match value {
        Value::UInt(x) if *x <= i64::MAX as u64 => return numeric(&Value::Int(*x as i64)),
        Value::UInt(x) => (
            ScalarType::U64 {
                divisible_by_16: x % 16 == 0,
            },
            Some(Scalar::U64(*x)),
        ),
        Value::Float(x) => (ScalarType::F32, Some(Scalar::F32(*x as f32))),
        Value::Bool(x) => (ScalarType::Bool, Some(Scalar::Bool(*x))),
        Value::Int(1) => (ScalarType::One, None),
        Value::Int(x) => {
            if let Ok(x32) = i32::try_from(*x) {
                (
                    ScalarType::I32 {
                        divisible_by_16: x % 16 == 0,
                    },
                    Some(Scalar::I32(x32)),
                )
            } else {
                (
                    ScalarType::I64 {
                        divisible_by_16: x % 16 == 0,
                    },
                    Some(Scalar::I64(*x)),
                )
            }
        }
        Value::Text(_) | Value::List(_) | Value::Dict(_) | Value::DType(_) | Value::Device(_) => {
            return Err(type_error(format!(
                "failed to specialize argument of type: {}",
                typename(value)
            )));
        }
        _ => {
            return Err(Error::BindingGap {
                backend: "triton".into(),
                operation: "non-numeric scalar specialization".into(),
            });
        }
    })
}

pub(crate) fn argument(
    value: &Value,
) -> Result<(crate::triton_arguments::ArgumentType, Vec<Scalar>), Error> {
    use crate::triton_arguments::ArgumentType;
    Ok(match value {
        Value::None => (ArgumentType::None, vec![]),
        Value::Tuple(values) => {
            let mut types = vec![];
            let mut args = vec![];
            for value in values {
                let (kind, values) = argument(value)?;
                types.push(kind);
                args.extend(values);
            }
            (ArgumentType::Tuple(types), args)
        }
        Value::Tensor(_) | Value::Quantized(_) => {
            let (_, kind, pointer) = crate::backend_triton_fp8::info(value, "dtype")?;
            let pointee = match kind {
                tch::Kind::Float => "fp32",
                tch::Kind::Double => "fp64",
                tch::Kind::Half => "fp16",
                tch::Kind::BFloat16 => "bf16",
                tch::Kind::Bool => "u1",
                tch::Kind::Uint8 => "u8",
                tch::Kind::Int8 => "i8",
                tch::Kind::Int16 => "i16",
                tch::Kind::Int => "i32",
                tch::Kind::Int64 => "i64",
                tch::Kind::Float8e4m3fn => "fp8e4nv",
                tch::Kind::Float8e5m2 => "fp8e5",
                tch::Kind::Float8e4m3fnuz => "fp8e4b8",
                tch::Kind::Float8e5m2fnuz => "fp8e5b16",
                _ => {
                    return Err(Error::ReferenceException {
                        class: "KeyError",
                        message: format!(
                            "'{}'",
                            crate::registry::AllowedType::DType(kind)
                                .python_name()
                                .trim_start_matches("torch.")
                        ),
                    });
                }
            };
            (
                ArgumentType::Pointer {
                    pointee,
                    divisible_by_16: pointer.is_multiple_of(16),
                },
                vec![Scalar::TensorObject(value.clone())],
            )
        }
        _ => {
            let (kind, argument) = numeric(value)?;
            (ArgumentType::Numeric(kind), argument.into_iter().collect())
        }
    })
}
