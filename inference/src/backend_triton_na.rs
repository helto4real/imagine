//! Frozen NA3D wrapper and dynamic source-positioned online softmax kernel.
use crate::{
    backend_triton_adaln::{count, strides},
    backend_triton_fp8::{bind_arguments, element, info, typename},
    backend_triton_mxfp8::truthy,
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, na},
};
use std::sync::Arc;
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(crate) fn sequence(rt: &Runtime, value: &Value) -> Result<Vec<Value>, Error> {
    match value {
        Value::List(v) | Value::Tuple(v) => Ok(v.clone()),
        Value::Text(s) => Ok(s.chars().map(|c| Value::Text(c.to_string())).collect()),
        Value::Dict(v) => Ok(v.iter().map(|(k, _)| Value::Text(k.clone())).collect()),
        Value::Tensor(_) | Value::Quantized(_) => {
            if info(value, "dim")?.0.is_empty() {
                return Err(type_error("iteration over a 0-d tensor"));
            }
            let views =
                unsafe { rt.op("aten::unbind.int", &[value.clone(), Value::Int(0)], &vec![])? };
            let (Value::List(views) | Value::Tuple(views)) = views else {
                unreachable!("Original unbind produces tensor views")
            };
            Ok(views)
        }
        _ => Err(type_error(format!(
            "'{}' object is not iterable",
            typename(value)
        ))),
    }
}
fn kernel_value(v: &Value) -> Result<na::KernelValue, Error> {
    Ok(match v {
        Value::Int(n) => na::KernelValue::Int(*n),
        Value::UInt(n) => na::KernelValue::UInt(*n),
        Value::Bool(b) => na::KernelValue::Bool(*b),
        Value::Float(n) => na::KernelValue::Float(*n),
        // Non-numeric constexprs retain their actual cache text and the cold
        // source exception above; this value is never lowered for those types.
        _ => na::KernelValue::Int(0),
    })
}
fn kernel_placeholder(v: &Value) -> i64 {
    match v {
        Value::Int(n) => *n,
        Value::Bool(b) => i64::from(*b),
        Value::Float(n) => *n as i64,
        _ => 0,
    }
}
fn int_less_float(integer: i64, float: f64) -> bool {
    if float.is_nan() || float <= i64::MIN as f64 {
        return false;
    }
    if float >= 9223372036854775808. {
        return true;
    }
    let truncated = float as i64;
    integer < truncated || (integer == truncated && float > truncated as f64)
}
fn float_less_int(float: f64, integer: i64) -> bool {
    if float.is_nan() || float >= 9223372036854775808. {
        return false;
    }
    if float < i64::MIN as f64 {
        return true;
    }
    let truncated = float as i64;
    truncated < integer || (truncated == integer && float < (truncated as f64))
}
fn integer_block_width(mut n: i128) -> u32 {
    n -= 1;
    for shift in [1, 2, 4, 8, 16, 32] {
        n |= n >> shift;
    }
    n += 1;
    n.clamp(16, 32) as u32
}
fn dimension_less(rt: &Runtime, dimension: i64, value: &Value) -> Result<bool, Error> {
    Ok(match value {
        Value::Int(n) => dimension < *n,
        Value::UInt(n) => i128::from(dimension) < i128::from(*n),
        Value::Bool(b) => dimension < i64::from(*b),
        Value::Float(n) => int_less_float(dimension, *n),
        Value::Tensor(_) | Value::Quantized(_) => {
            let greater = unsafe {
                rt.op(
                    "aten::gt.Scalar",
                    &[value.clone(), Value::Int(dimension)],
                    &vec![],
                )?
            };
            truthy(rt, &greater)?
        }
        _ => {
            return Err(type_error(format!(
                "'<' not supported between instances of 'int' and '{}'",
                typename(value)
            )));
        }
    })
}
fn block_width(rt: &Runtime, width: i64, kernel: &Value) -> Result<u32, Error> {
    let rhs = match kernel {
        Value::Int(n) => {
            let n = i128::from(*n) + 16;
            if n > i128::from(i64::MAX) {
                return Ok(integer_block_width(i128::from(width)));
            }
            Value::Int(n as i64)
        }
        Value::UInt(_) => return Ok(integer_block_width(i128::from(width))),
        Value::Bool(b) => Value::Int(16 + i64::from(*b)),
        Value::Float(n) => Value::Float(16. + n),
        Value::Tensor(_) | Value::Quantized(_) => unsafe {
            rt.op(
                "aten::add.Scalar",
                &[kernel.clone(), Value::Int(16)],
                &vec![],
            )?
        },
        _ => {
            return Err(type_error(format!(
                "unsupported operand type(s) for +: 'int' and '{}'",
                typename(kernel)
            )));
        }
    };
    let use_rhs = match &rhs {
        Value::Int(n) => *n < width,
        Value::Float(n) => float_less_int(*n, width),
        Value::Tensor(_) | Value::Quantized(_) => {
            let less = unsafe {
                rt.op(
                    "aten::lt.Scalar",
                    &[rhs.clone(), Value::Int(width)],
                    &vec![],
                )?
            };
            truthy(rt, &less)?
        }
        _ => unreachable!(),
    };
    let sweep = if use_rhs { rhs } else { Value::Int(width) };
    match sweep {
        Value::Int(n) => Ok(integer_block_width(i128::from(n))),
        Value::Float(_) => Err(type_error(
            "unsupported operand type(s) for >>: 'float' and 'int'",
        )),
        Value::Tensor(_) | Value::Quantized(_) => {
            let n = unsafe { rt.op("aten::sub_.Scalar", &[sweep, Value::Int(1)], &vec![])? };
            for shift in [1, 2, 4, 8, 16, 32] {
                let shifted = unsafe {
                    rt.op(
                        "aten::bitwise_right_shift.Tensor_Scalar",
                        &[n.clone(), Value::Int(shift)],
                        &vec![],
                    )?
                };
                unsafe { rt.op("aten::bitwise_or_.Tensor", &[n.clone(), shifted], &vec![])? };
            }
            unsafe { rt.op("aten::add_.Scalar", &[n.clone(), Value::Int(1)], &vec![])? };
            let less = unsafe { rt.op("aten::lt.Scalar", &[n.clone(), Value::Int(32)], &vec![])? };
            if !truthy(rt, &less)? {
                return Ok(32);
            }
            let greater = unsafe { rt.op("aten::gt.Scalar", &[n, Value::Int(16)], &vec![])? };
            assert!(
                !truthy(rt, &greater)?,
                "The original integer bit-twiddle cannot produce a power between 16 and 32"
            );
            Ok(16)
        }
        _ => unreachable!(),
    }
}
fn next_power(n: i64) -> Result<u32, Error> {
    if n <= 0 {
        return Ok(0);
    }
    let bits = { 128 - (i128::from(n) - 1).unsigned_abs().leading_zeros() };
    1u32.checked_shl(bits).ok_or_else(|| Error::BindingGap {
        backend: "triton".into(),
        operation: "NA3D oversized compile-time block".into(),
    })
}
pub(crate) fn zip_end(first: usize, second: usize) -> Result<(), Error> {
    let n = first.min(second).min(3);
    let message = if n == first {
        if second > n {
            Some("zip() argument 2 is longer than argument 1")
        } else if 3 > n {
            Some("zip() argument 3 is longer than arguments 1-2")
        } else {
            None
        }
    } else if n == second {
        Some("zip() argument 2 is shorter than argument 1")
    } else {
        Some("zip() argument 3 is shorter than arguments 1-2")
    };
    if let Some(message) = message {
        Err(error("ValueError", message))
    } else {
        Ok(())
    }
}
fn contiguous(rt: &Runtime, v: Value) -> Result<Value, Error> {
    unsafe { rt.op("aten::contiguous.default", &[v], &vec![]) }
}
fn run(rt: &Runtime, compiler: &Compiler, args: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    bind_arguments(
        "na3d",
        &["q", "k", "v", "kernel_size", "is_causal", "scale"],
        4,
        args,
        kw,
    )?;
    let q = get(args, kw, 0, "q").unwrap().clone();
    let dims = info(&q, "shape")?.0;
    let [batch, t, h, w, nh, hd]: [i64; 6] = dims.clone().try_into().map_err(|_| {
        error(
            "ValueError",
            if dims.len() < 6 {
                format!(
                    "not enough values to unpack (expected 6, got {})",
                    dims.len()
                )
            } else {
                "too many values to unpack (expected 6)".into()
            },
        )
    })?;
    let none = Value::None;
    let causal = match get(args, kw, 4, "is_causal").unwrap_or(&none) {
        Value::None => vec![Value::Bool(false); 3],
        v => sequence(rt, v)?,
    };
    let kernels = sequence(rt, get(args, kw, 3, "kernel_size").unwrap())?;
    let mut normalized = vec![];
    for (i, (kernel, c)) in kernels.iter().zip(&causal).take(3).enumerate() {
        normalized.push(if truthy(rt, c)? {
            kernel.clone()
        } else {
            let dimension = [t, h, w][i];
            if dimension_less(rt, dimension, kernel)? {
                Value::Int(dimension)
            } else {
                kernel.clone()
            }
        });
    }
    zip_end(kernels.len(), causal.len())?;
    let scale = match get(args, kw, 5, "scale").unwrap_or(&none) {
        Value::None if hd == 0 => {
            return Err(error(
                "ZeroDivisionError",
                "0.0 cannot be raised to a negative power",
            ));
        }
        Value::None => Value::Float((hd as f64).powf(-0.5)),
        v => v.clone(),
    };
    let q = contiguous(rt, q)?;
    let k = contiguous(rt, get(args, kw, 1, "k").unwrap().clone())?;
    let v = contiguous(rt, get(args, kw, 2, "v").unwrap().clone())?;
    let out = unsafe {
        rt.op(
            "aten::empty_like.default",
            std::slice::from_ref(&q),
            &vec![],
        )?
    };
    let hd_pad = 16.max(next_power(hd)?);
    let block_q = 16;
    let block_k = block_width(rt, w, &normalized[2])?;
    let strides = strides(&q)?;
    let values = [
        t, h, w, nh, strides[0], strides[1], strides[2], strides[3], strides[4],
    ];
    let mut launch = vec![q, k, v, out.clone()];
    launch.extend(values.map(Value::Int));
    launch.push(scale.clone());
    let mut launch_kw = vec![
        ("kt".into(), normalized[0].clone()),
        ("kh".into(), normalized[1].clone()),
        ("kw".into(), normalized[2].clone()),
        ("causal_t".into(), causal[0].clone()),
        ("causal_h".into(), causal[1].clone()),
        ("causal_w".into(), causal[2].clone()),
    ];
    launch_kw.extend(
        [
            ("hd", hd),
            ("hd_pad", i64::from(hd_pad)),
            ("block_q", i64::from(block_q)),
            ("block_k", i64::from(block_k)),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Int(v))),
    );
    launch_kw.push((
        "is_fp32".into(),
        Value::Bool(info(&launch[0], "dtype")?.1 == tch::Kind::Float),
    ));
    launch_kw.push(("num_warps".into(), Value::Int(4)));
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run("na3d", &launch, &launch_kw)?;
    let pointers = launch[..4]
        .iter()
        .map(|v| info(v, "dtype").map(|v| v.2))
        .collect::<Result<Vec<_>, _>>()?;
    let scalars = values.map(count);
    let (scale_argument, scale_payload) = crate::backend_triton_scalar::argument(&scale)?;
    let scale_kind = if let crate::triton_arguments::ArgumentType::Numeric(kind) = &scale_argument {
        *kind
    } else {
        na::Scale::F32
    };
    let mut literals = vec![];
    let mut failure = None;
    for (i, value) in normalized.iter().enumerate() {
        literals.push((
            14 + i,
            crate::python_value::constexpr_key(value, &rt.native)?,
        ));
    }
    for (i, value) in causal.iter().enumerate() {
        literals.push((
            17 + i,
            crate::python_value::constexpr_key(value, &rt.native)?,
        ));
    }
    for i in 0..3 {
        let value = &causal[i];
        if !matches!(value, Value::Bool(_) | Value::Int(_) | Value::None) {
            let mut e = crate::triton_frontend::errors::CompilationFailure::at(
                na::FUNCTION_SOURCE,
                na::FUNCTION_DEFINITION,
                [(49, 4), (55, 4), (64, 4)][i],
                &format!(
                    "`if` conditionals can only accept values of type {{int, bool, NoneType}}, not objects of type {}",
                    typename(value)
                ),
            );
            e.class = "UnsupportedLanguageConstruct";
            failure = Some(e);
            break;
        }
        if !matches!(
            &normalized[i],
            Value::Int(_)
                | Value::Float(_)
                | Value::Bool(_)
                | Value::Tensor(_)
                | Value::Quantized(_)
        ) {
            failure = Some(crate::triton_frontend::errors::CompilationFailure::at(
                na::FUNCTION_SOURCE,
                na::FUNCTION_DEFINITION,
                [(50, 26), (56, 26), (65, 29)][i],
                &format!(
                    "AttributeError(\"'{}' object has no attribute 'type'\")",
                    typename(&normalized[i])
                ),
            ));
            break;
        }
        if matches!(&normalized[i], Value::Tensor(_)) {
            let point = if truthy(rt, value)? {
                [(50, 26), (56, 26), (65, 29)][i]
            } else {
                [(53, 37), (59, 37), (72, 40)][i]
            };
            failure = Some(crate::triton_frontend::errors::CompilationFailure::at(
                na::FUNCTION_SOURCE,
                na::FUNCTION_DEFINITION,
                point,
                "AttributeError(\"'builtin_function_or_method' object has no attribute 'scalar'\")",
            ));
            break;
        }
    }
    let kernels = normalized
        .iter()
        .map(|v| {
            if matches!(v, Value::Tensor(_)) {
                Ok(na::KernelValue::Int(kernel_placeholder(v)))
            } else {
                kernel_value(v)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let causal = causal
        .iter()
        .map(|v| truthy(rt, v))
        .collect::<Result<Vec<_>, _>>()?;
    let specialization = na::Specialization {
        input: element(info(&launch[0], "dtype")?.1)?,
        key: element(info(&launch[1], "dtype")?.1)?,
        value: element(info(&launch[2], "dtype")?.1)?,
        scale: scale_kind,
        kt: kernel_placeholder(&normalized[0]),
        kh: kernel_placeholder(&normalized[1]),
        kw: kernel_placeholder(&normalized[2]),
        causal_t: causal[0],
        causal_h: causal[1],
        causal_w: causal[2],
        hd: hd as u32,
        hd_pad,
        block_q,
        block_k,
        scalars,
        pointer_divisibility_16: std::array::from_fn(|i| pointers[i] % 16 == 0),
        source_path: "",
        standard_source_path: "",
    };
    let kernel = unsafe {
        compiler.jit_na3d_arguments(
            device,
            &specialization,
            kernels.try_into().ok().unwrap(),
            &literals,
            &scale_argument,
            failure,
        )?
    };
    let mut argv = launch[..4]
        .iter()
        .cloned()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    for (v, c) in values.into_iter().zip(scalars) {
        match c {
            Count::One => {}
            Count::I32 { .. } => argv.push(Scalar::I32(v as i32)),
            Count::I64 { .. } => argv.push(Scalar::I64(v)),
        }
    }
    argv.extend(scale_payload);
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    let grid = [
        (i128::from(w) + 15) / 16,
        i128::from(t) * i128::from(h),
        i128::from(batch) * i128::from(nh),
    ]
    .map(|n| i32::try_from(n).map_err(|_| type_error("signed integer is greater than maximum")))
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    unsafe { kernel.run([grid[0], grid[1], grid[2]], stream, &mut argv)? };
    Ok(out)
}
impl Runtime {
    pub fn bind_triton_na(&self, compiler: Arc<Compiler>) {
        self.backends.bind_reference(
            "triton",
            "na3d",
            Arc::new(move |rt, args, kw| run(rt, &compiler, args, kw)),
        );
    }
}
