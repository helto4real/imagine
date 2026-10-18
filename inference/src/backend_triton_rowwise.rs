//! Original registered INT8 wrappers, eager seeded branch and rowwise JIT.
use crate::{
    backend_triton_fp8::{bind_arguments, element, info, typename},
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, rowwise::Specialization},
};
use std::sync::Arc;
use tch::{Device, Kind, Tensor};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(crate) fn device(v: &Value) -> Result<Device, Error> {
    match v {
        Value::Tensor(t) => Ok(t.lock().unwrap().device()),
        Value::Quantized(q) => Ok(q.lock().unwrap().payload.device()),
        _ => Err(error(
            "AttributeError",
            format!("'{}' object has no attribute 'device'", typename(v)),
        )),
    }
}
pub(crate) fn positive(rt: &Runtime, v: &Value) -> Result<bool, Error> {
    match v {
        Value::None => Ok(false),
        Value::Bool(b) => Ok(*b),
        Value::Int(n) => Ok(*n > 0),
        Value::UInt(n) => Ok(*n > 0),
        Value::Float(n) => Ok(*n > 0.),
        Value::Tensor(_) | Value::Quantized(_) => {
            let greater =
                unsafe { rt.op("aten::gt.Scalar", &[v.clone(), Value::Int(0)], &vec![])? };
            crate::backend_triton_mxfp8::truthy(rt, &greater)
        }
        _ => Err(type_error(format!(
            "'>' not supported between instances of '{}' and 'int'",
            typename(v)
        ))),
    }
}
fn shape(v: Vec<i64>) -> Value {
    Value::List(v.into_iter().map(Value::Int).collect())
}
pub(crate) fn rotate(rt: &Runtime, x: &Value, h: &Value, group: &Value) -> Result<Value, Error> {
    let (original, kind, _) = info(x, "shape")?;
    let features = *original
        .last()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let group = match group {
        Value::Int(n) => *n,
        Value::Bool(b) => i64::from(*b),
        _ => {
            return Err(Error::BindingGap {
                backend: "triton".into(),
                operation: "non-integer rotation group arithmetic".into(),
            });
        }
    };
    if group == 0 {
        return Err(error("ZeroDivisionError", "integer modulo by zero"));
    }
    if features % group != 0 {
        return Err(error(
            "ValueError",
            format!("features {features} not divisible by group_size {group}"),
        ));
    }
    let grouped = unsafe {
        rt.op(
            "aten::reshape.default",
            &[x.clone(), shape(vec![-1, features / group, group])],
            &vec![],
        )?
    };
    info(h, "to")?;
    let h = unsafe {
        rt.op(
            "aten::to.dtype_layout",
            std::slice::from_ref(h),
            &vec![
                ("dtype".into(), Value::DType(kind)),
                ("device".into(), Value::Device(device(x)?)),
            ],
        )?
    };
    let rotated = unsafe { rt.op("aten::matmul.default", &[grouped, h], &vec![])? };
    unsafe {
        rt.op(
            "aten::reshape.default",
            &[rotated, shape(original)],
            &vec![],
        )
    }
}
pub(crate) fn rowwise(rt: &Runtime, compiler: &Compiler, x: Value) -> Result<Value, Error> {
    let (dims, _, _) = info(&x, "shape")?;
    if dims.len() != 2 {
        return Err(error(
            "ValueError",
            if dims.len() > 2 {
                "too many values to unpack (expected 2)".into()
            } else {
                format!(
                    "not enough values to unpack (expected 2, got {})",
                    dims.len()
                )
            },
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let y = unsafe {
        rt.op(
            "aten::empty_like.default",
            std::slice::from_ref(&x),
            &vec![("dtype".into(), Value::DType(Kind::Int8))],
        )?
    };
    let scales = Value::tensor(Tensor::f_empty([rows, 1], (Kind::Float, device(&x)?))?);
    let (_, kind, input) = info(&x, "dtype")?;
    let code = match kind {
        Kind::Half => 1,
        Kind::BFloat16 => 2,
        _ => 0,
    };
    let block = (cols as u64)
        .checked_next_power_of_two()
        .unwrap_or(0)
        .max(128);
    let block = u32::try_from(block).map_err(|_| Error::BindingGap {
        backend: "triton".into(),
        operation: "unrepresentable rowwise block size".into(),
    })?;
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        "rowwise",
        &[x.clone(), y.clone(), scales.clone(), Value::Int(cols)],
        &vec![
            ("block_size".into(), Value::Int(i64::from(block))),
            ("input_dtype_code".into(), Value::Int(code)),
        ],
    )?;
    let (_, _, output) = info(&y, "data_ptr")?;
    let (_, _, scale) = info(&scales, "data_ptr")?;
    let count = if cols == 1 {
        Count::One
    } else if cols <= i32::MAX as i64 {
        Count::I32 {
            divisible_by_16: cols % 16 == 0,
        }
    } else {
        Count::I64 {
            divisible_by_16: cols % 16 == 0,
        }
    };
    let spec = Specialization {
        input: element(kind)?,
        input_dtype_code: code as u8,
        block_size: block,
        pointer_divisibility_16: [input, output, scale].map(|p| p % 16 == 0),
        count,
        source_path: compiler.source_path(),
        standard_source_path: "",
    };
    let kernel = unsafe { compiler.jit_rowwise(device, &spec)? };
    let mut args = [x.clone(), y.clone(), scales.clone()]
        .into_iter()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    match count {
        Count::One => {}
        Count::I32 { .. } => args.push(Scalar::I32(cols as i32)),
        Count::I64 { .. } => args.push(Scalar::I64(cols)),
    }
    args.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    unsafe {
        kernel.run(
            [
                rows.try_into()
                    .map_err(|_| type_error("signed integer is greater than maximum"))?,
                1,
                1,
            ],
            stream,
            &mut args,
        )?;
    }
    Ok(Value::Tuple(vec![y, scales]))
}
fn run(
    rt: &Runtime,
    compiler: &Compiler,
    rotated: bool,
    args: &[Value],
    kwargs: &Kwargs,
) -> Result<Value, Error> {
    let name = if rotated {
        "quantize_and_rotate_rowwise"
    } else {
        "quantize_int8_rowwise"
    };
    let names: &[&str] = if rotated {
        &["x", "h", "group_size", "stochastic_rounding"]
    } else {
        &["x", "stochastic_rounding"]
    };
    let required = if rotated { 3 } else { 1 };
    bind_arguments(name, names, required, args, kwargs)?;
    let seed = get(args, kwargs, required, "stochastic_rounding")
        .cloned()
        .unwrap_or(Value::Int(0));
    if positive(rt, &seed)? {
        let mut forwarded = names[..required]
            .iter()
            .enumerate()
            .map(|(i, n)| (n.to_string(), get(args, kwargs, i, n).unwrap().clone()))
            .collect::<Kwargs>();
        forwarded.push(("stochastic_rounding".into(), seed));
        return unsafe { rt.backends.call_bound(rt, "eager", name, &forwarded) };
    }
    let x = get(args, kwargs, 0, "x").unwrap();
    let x = if rotated {
        rotate(
            rt,
            x,
            get(args, kwargs, 1, "h").unwrap(),
            get(args, kwargs, 2, "group_size").unwrap(),
        )?
    } else {
        x.clone()
    };
    rowwise(rt, compiler, x)
}
impl Runtime {
    /// Connect the two original registered rowwise bodies. Positive seeds retain
    /// the source's eager delegation; ordinary calls use the original Triton JIT.
    pub fn bind_triton_rowwise(&self, compiler: Arc<Compiler>) {
        for (name, rotated) in [
            ("quantize_int8_rowwise", false),
            ("quantize_and_rotate_rowwise", true),
        ] {
            let compiler = compiler.clone();
            self.backends.bind_reference(
                "triton",
                name,
                Arc::new(move |rt, args, kwargs| run(rt, &compiler, rotated, args, kwargs)),
            );
        }
    }
}
