//! Original AdaLN wrapper, shared modulation geometry and two-pass Triton body.
use crate::{
    backend_triton_fp8::{bind_arguments, element, info},
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, adaln::Specialization},
};
use std::sync::Arc;
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn shape(v: Vec<i64>) -> Value {
    Value::List(v.into_iter().map(Value::Int).collect())
}
fn numel(dims: &[i64]) -> Result<i64, Error> {
    dims.iter()
        .try_fold(1i64, |n, d| n.checked_mul(*d))
        .ok_or_else(|| error("RuntimeError", "numel: integer multiplication overflow"))
}
fn contiguous(rt: &Runtime, x: Value) -> Result<Value, Error> {
    if matches!(
        unsafe {
            rt.op(
                "aten::is_contiguous.default",
                std::slice::from_ref(&x),
                &vec![],
            )?
        },
        Value::Bool(true)
    ) {
        Ok(x)
    } else {
        unsafe { rt.op("aten::contiguous.default", &[x], &vec![]) }
    }
}
fn reshape(rt: &Runtime, x: Value, dims: Vec<i64>) -> Result<Value, Error> {
    unsafe { rt.op("aten::reshape.default", &[x, shape(dims)], &vec![]) }
}
pub(crate) fn strides(v: &Value) -> Result<Vec<i64>, Error> {
    match v {
        Value::Tensor(t) => Ok(t.lock().unwrap().stride()),
        Value::Quantized(_) => {
            // _make_wrapper_subclass receives no strides in the frozen source.
            let (dims, _, _) = info(v, "stride")?;
            let mut stride = 1i64;
            let mut result = vec![0; dims.len()];
            for i in (0..dims.len()).rev() {
                result[i] = stride;
                stride = stride
                    .checked_mul(dims[i].max(1))
                    .ok_or_else(|| error("RuntimeError", "Stride calculation overflowed"))?;
            }
            Ok(result)
        }
        _ => {
            info(v, "stride")?;
            unreachable!()
        }
    }
}
fn modulation(rt: &Runtime, t: &Value, x: &Value, n: i64, d: i64) -> Result<(Value, i64), Error> {
    let (ts, _, _) = info(t, "shape")?;
    let (xs, _, _) = info(x, "shape")?;
    let last = ts
        .last()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    if *last == d && ts.len() <= xs.len() {
        let mut lead = vec![1; xs.len() - ts.len()];
        lead.extend_from_slice(&ts[..ts.len() - 1]);
        let mut seen = false;
        let mut groupable = true;
        for (s, x) in lead.iter().zip(&xs[..xs.len() - 1]) {
            if s == x {
                if *x != 1 && seen {
                    groupable = false;
                    break;
                }
            } else if *s == 1 {
                if *x != 1 {
                    seen = true;
                }
            } else {
                groupable = false;
                break;
            }
        }
        if groupable {
            let rows = numel(&ts)? / d;
            if rows > 0 && n % rows == 0 {
                return Ok((
                    contiguous(rt, reshape(rt, t.clone(), vec![rows, d])?)?,
                    n / rows,
                ));
            }
        }
    }
    let expanded = unsafe { rt.op("aten::expand_as.default", &[t.clone(), x.clone()], &vec![])? };
    Ok((contiguous(rt, reshape(rt, expanded, vec![n, d])?)?, 1))
}
pub(crate) fn count(n: i64) -> Count {
    if n == 1 {
        Count::One
    } else if i32::try_from(n).is_ok() {
        Count::I32 {
            divisible_by_16: n % 16 == 0,
        }
    } else {
        Count::I64 {
            divisible_by_16: n % 16 == 0,
        }
    }
}
fn run(
    rt: &Runtime,
    compiler: &Compiler,
    subtract_mean: bool,
    args: &[Value],
    kwargs: &Kwargs,
) -> Result<Value, Error> {
    let name = if subtract_mean { "adaln" } else { "rms_adaln" };
    bind_arguments(name, &["x", "scale", "shift", "eps"], 3, args, kwargs)?;
    let x = get(args, kwargs, 0, "x").unwrap();
    let scale = get(args, kwargs, 1, "scale").unwrap();
    let shift = get(args, kwargs, 2, "shift").unwrap();
    let eps = get(args, kwargs, 3, "eps")
        .cloned()
        .unwrap_or(Value::Float(1e-6));
    let (original, kind, _) = info(x, "shape")?;
    let d = *original
        .last()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    if d == 0 {
        return Err(error(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let n = numel(&original)? / d;
    let input = contiguous(rt, reshape(rt, x.clone(), vec![n, d])?)?;
    let (scale, sg) = modulation(rt, scale, x, n, d)?;
    let (shift, hg) = modulation(rt, shift, x, n, d)?;
    let output = unsafe {
        rt.op(
            "aten::empty_like.default",
            std::slice::from_ref(&input),
            &vec![],
        )?
    };
    let block = (d as u64).next_power_of_two().min(4096) as u32;
    let dimensions = [
        d,
        sg,
        hg,
        strides(&input)?[0],
        strides(&scale)?[0],
        strides(&shift)?[0],
        strides(&output)?[0],
    ];
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    let mut launch = vec![input.clone(), scale.clone(), shift.clone(), output.clone()];
    launch.extend(dimensions.into_iter().map(Value::Int));
    launch.push(eps.clone());
    compiler.pre_run(
        "adaln",
        &launch,
        &vec![
            ("block_d".into(), Value::Int(i64::from(block))),
            ("dtype".into(), Value::DType(kind)),
            ("subtract_mean".into(), Value::Bool(subtract_mean)),
        ],
    )?;
    let (epsilon_type, epsilon_value) = crate::backend_triton_scalar::argument(&eps)?;
    let (_, input_kind, input_ptr) = info(&input, "dtype")?;
    let (_, scale_kind, scale_ptr) = info(&scale, "dtype")?;
    let (_, shift_kind, shift_ptr) = info(&shift, "dtype")?;
    let (_, _, output_ptr) = info(&output, "dtype")?;
    if input_kind != kind {
        return Err(Error::BindingGap {
            backend: "triton".into(),
            operation: "AdaLN input reshape changes dtype".into(),
        });
    }
    let pointers = [input_ptr, scale_ptr, shift_ptr, output_ptr];
    let counts = dimensions.map(count);
    let spec = Specialization {
        input: element(kind)?,
        scale: element(scale_kind)?,
        shift: element(shift_kind)?,
        block_size: block,
        subtract_mean,
        scalars: counts,
        pointer_divisibility_16: pointers.map(|p| p % 16 == 0),
        source_path: "",
        standard_source_path: "",
    };
    let kernel = unsafe { compiler.jit_adaln_argument(device, &spec, &epsilon_type)? };
    let mut argv = launch[..4]
        .iter()
        .cloned()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    for (n, c) in dimensions.into_iter().zip(counts) {
        match c {
            Count::One => {}
            Count::I32 { .. } => argv.push(Scalar::I32(n as i32)),
            Count::I64 { .. } => argv.push(Scalar::I64(n)),
        }
    }
    argv.extend(epsilon_value);
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    unsafe {
        kernel.run(
            [
                n.try_into()
                    .map_err(|_| type_error("signed integer is greater than maximum"))?,
                1,
                1,
            ],
            stream,
            &mut argv,
        )?;
    }
    reshape(rt, output, original)
}
impl Runtime {
    /// Connect the original AdaLN and RMS-AdaLN wrappers to the same JIT function.
    pub fn bind_triton_adaln(&self, compiler: Arc<Compiler>) {
        for (name, subtract_mean) in [("adaln", true), ("rms_adaln", false)] {
            let compiler = compiler.clone();
            self.backends.bind_reference(
                "triton",
                name,
                Arc::new(move |rt, args, kwargs| run(rt, &compiler, subtract_mean, args, kwargs)),
            );
        }
    }
}
