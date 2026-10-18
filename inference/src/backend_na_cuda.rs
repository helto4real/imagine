//! Original CUDA NA wrapper indexing, Python conversions and native boundary.
use super::*;
fn indexed(rt: &Runtime, value: &Value, index: usize) -> Result<Value, Error> {
    match value {
        Value::List(v) | Value::Tuple(v) => v.get(index).cloned().ok_or_else(|| {
            error(
                "IndexError",
                if matches!(value, Value::List(_)) {
                    "list index out of range"
                } else {
                    "tuple index out of range"
                },
            )
        }),
        Value::Dict(_) => Err(error("KeyError", index.to_string())),
        Value::Text(s) => s
            .chars()
            .nth(index)
            .map(|c| Value::Text(c.to_string()))
            .ok_or_else(|| error("IndexError", "string index out of range")),
        Value::Tensor(_) | Value::Quantized(_) => {
            let info = crate::backend_triton_fp8::info(value, "dim")?;
            if info.0.is_empty() {
                return Err(error(
                    "IndexError",
                    "invalid index of a 0-dim tensor. Use `tensor.item()` in Python or `tensor.item<T>()` in C++ to convert a 0-dim tensor to a number",
                ));
            }
            unsafe {
                rt.op(
                    "aten::select.int",
                    &[value.clone(), Value::Int(0), Value::Int(index as i64)],
                    &vec![],
                )
            }
        }
        _ => Err(type_error(format!(
            "'{}' object is not subscriptable",
            crate::backend_triton_fp8::typename(value)
        ))),
    }
}
fn native_error() -> Error {
    use super::super::argument_signatures::{NA3D_INVOKED, NA3D_SIGNATURE};
    error(
        "TypeError",
        format!(
            "na3d(): incompatible function arguments. The following argument types are supported:\n    1. {NA3D_SIGNATURE}\n\nInvoked with types: {NA3D_INVOKED}"
        ),
    )
}
pub(super) fn run(rt: &Runtime, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let q = tensor(a, kw, 0, "q")?;
    let key = tensor(a, kw, 1, "k")?;
    let value = tensor(a, kw, 2, "v")?;
    let causal = match get(a, kw, 4, "is_causal") {
        None | Some(Value::None) => vec![Value::Bool(false); 3],
        Some(v) => crate::backend_triton_na::sequence(rt, v)?,
    };
    let dims = shape(&q)?;
    let default_scale = if get(a, kw, 5, "scale").is_none_or(|v| matches!(v, Value::None)) {
        if dims[5] == 0 {
            return Err(error(
                "ZeroDivisionError",
                "0.0 cannot be raised to a negative power",
            ));
        }
        Some(Value::Float((dims[5] as f64).powf(-0.5)))
    } else {
        None
    };
    let q = q.f_contiguous()?;
    let key = key.f_contiguous()?;
    let value = value.f_contiguous()?;
    let out = q.f_empty_like()?;
    let stream = stream(rt, &q)?;
    let kernels = required(a, kw, 3, "kernel_size")?;
    let mut sizes = vec![];
    for index in 0..3 {
        sizes.push(crate::python_numbers::integer_value(
            rt,
            &indexed(rt, kernels, index)?,
        )?);
    }
    let mut flags = vec![];
    for index in 0..3 {
        let v = causal
            .get(index)
            .ok_or_else(|| error("IndexError", "list index out of range"))?;
        flags.push(crate::python_numbers::integer_value(rt, v)?);
    }
    let scale = crate::python_numbers::float_value(
        rt,
        default_scale
            .as_ref()
            .or_else(|| get(a, kw, 5, "scale"))
            .unwrap(),
    )?;
    let code = match q.kind() {
        Kind::Float => 0,
        Kind::Half => 1,
        Kind::BFloat16 => 2,
        other => {
            return Err(error(
                "KeyError",
                crate::registry::AllowedType::DType(other).python_name(),
            ));
        }
    };
    // All Python arguments are evaluated before nanobind checks scalar ranges.
    if sizes.iter().any(|n| i64::try_from(*n).is_err())
        || flags.iter().any(|n| i32::try_from(*n).is_err())
    {
        return Err(native_error());
    }
    use KernelArg::{Float, Int, Tensor as T};
    let mut args = vec![T(&q), T(&key), T(&value), T(&out)];
    args.extend(dims.into_iter().map(Int));
    args.extend(sizes.into_iter().map(|n| Int(n as i64)));
    args.extend(flags.into_iter().map(|n| Int(n as i64)));
    args.extend([Float(scale), Int(code), Int(stream)]);
    unsafe {
        rt.native.call_unchecked("na3d", &args)?;
    }
    Ok(Value::tensor(out))
}
