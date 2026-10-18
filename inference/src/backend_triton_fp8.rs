//! Original FP8 Python wrapper order and dynamic typed runtime specialization.
use crate::{
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, Element, Fp8Body, Fp8Specialization},
};
use std::sync::Arc;
use tch::Kind;
pub(crate) fn typename(v: &Value) -> &'static str {
    match v {
        Value::None => "NoneType",
        Value::Bool(_) => "bool",
        Value::Int(_) | Value::UInt(_) => "int",
        Value::Float(_) => "float",
        Value::Text(_) => "str",
        Value::DType(_) => "torch.dtype",
        Value::Device(_) => "torch.device",
        Value::Tensor(_) => "Tensor",
        Value::Quantized(_) => "QuantizedTensor",
        Value::List(_) => "list",
        Value::Tuple(_) => "tuple",
        Value::Dict(_) => "dict",
    }
}
fn display(v: &Value) -> Result<String, Error> {
    crate::python_value::display(v)
}
pub(crate) use crate::execution_value::bind_arguments;
pub(crate) fn info(v: &Value, attribute: &str) -> Result<(Vec<i64>, Kind, u64), Error> {
    match v {
        Value::Tensor(t) => {
            let t = t.lock().unwrap();
            Ok((t.size(), t.kind(), t.data_ptr() as u64))
        }
        Value::Quantized(q) => {
            let q = q.lock().unwrap();
            Ok((q.params.original_shape.clone(), q.params.original_kind, 0))
        }
        _ => Err(Error::ReferenceException {
            class: "AttributeError",
            message: format!("'{}' object has no attribute '{attribute}'", typename(v)),
        }),
    }
}
pub(crate) fn element(kind: Kind) -> Result<Element, Error> {
    Ok(match kind {
        Kind::Float => Element::F32,
        Kind::Half => Element::F16,
        Kind::BFloat16 => Element::BF16,
        Kind::Float8e4m3fn => Element::E4M3,
        Kind::Float8e5m2 => Element::E5M2,
        _ => {
            return Err(Error::BindingGap {
                backend: "triton".into(),
                operation: format!("direct FP8 kernel element {kind:?}"),
            });
        }
    })
}
fn run(
    rt: &Runtime,
    compiler: &Compiler,
    quantize: bool,
    args: &[Value],
    kwargs: &Kwargs,
) -> Result<Value, Error> {
    let name = if quantize {
        "quantize_per_tensor_fp8"
    } else {
        "dequantize_per_tensor_fp8"
    };
    bind_arguments(name, &["x", "scale", "output_type"], 2, args, kwargs)?;
    let input = get(args, kwargs, 0, "x").unwrap();
    let scale = get(args, kwargs, 1, "scale").unwrap();
    let default = Value::DType(if quantize {
        Kind::Float8e4m3fn
    } else {
        Kind::BFloat16
    });
    let output = get(args, kwargs, 2, "output_type").unwrap_or(&default);
    let limit = if quantize {
        match output {
            Value::DType(Kind::Float8e4m3fn) => 448.,
            Value::DType(Kind::Float8e5m2) => 57344.,
            _ => {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: format!(
                        "Unsupported output_type: {}. Expected torch.float8_e4m3fn or torch.float8_e5m2",
                        display(output)?
                    ),
                });
            }
        }
    } else {
        0.
    };
    info(input, "is_contiguous")?;
    let empty = vec![];
    let contiguous = unsafe {
        rt.op(
            "aten::is_contiguous.default",
            std::slice::from_ref(input),
            &empty,
        )?
    };
    let input = if matches!(contiguous, Value::Bool(true)) {
        input.clone()
    } else {
        unsafe {
            rt.op(
                "aten::contiguous.default",
                std::slice::from_ref(input),
                &empty,
            )?
        }
    };
    let (shape, _, _) = info(&input, "shape")?;
    let n = shape
        .iter()
        .try_fold(1i64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| Error::ReferenceException {
            class: "RuntimeError",
            message: "numel: integer multiplication overflow".into(),
        })?;
    // ATen flatten returns the original tensor object for an existing vector.
    // This also preserves QuantizedTensor's original null pointer in that case.
    let flat = if shape.len() == 1 {
        input
    } else {
        unsafe {
            rt.op(
                "aten::view.default",
                &[input, Value::List(vec![Value::Int(n)])],
                &empty,
            )?
        }
    };
    if !matches!(output, Value::None | Value::DType(_)) {
        return Err(type_error(format!(
            "empty_like(): argument 'dtype' must be torch.dtype, not {}",
            typename(output)
        )));
    }
    let out = unsafe {
        rt.op(
            "aten::empty_like.default",
            std::slice::from_ref(&flat),
            &vec![("dtype".into(), output.clone())],
        )?
    };
    let block = if n < 32768 {
        128
    } else if n < 131072 {
        256
    } else if n < 524288 {
        512
    } else {
        1024
    };
    let op = if quantize { "quantize" } else { "dequantize" };
    let mut launch = vec![flat.clone(), out.clone(), scale.clone()];
    if quantize {
        launch.push(Value::Float(limit));
    }
    launch.push(Value::Int(n));
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(op, &launch, &vec![("block_size".into(), Value::Int(block))])?;
    let (_, input_kind, input_ptr) = info(&flat, "dtype")?;
    let (_, output_kind, output_ptr) = info(&out, "dtype")?;
    let (_, scale_kind, scale_ptr) = info(scale, "dtype")?;
    let count = if n == 1 {
        Count::One
    } else if n <= i32::MAX as i64 {
        Count::I32 {
            divisible_by_16: n % 16 == 0,
        }
    } else {
        Count::I64 {
            divisible_by_16: n % 16 == 0,
        }
    };
    let spec = Fp8Specialization {
        body: if quantize {
            Fp8Body::Quantize
        } else {
            Fp8Body::Dequantize
        },
        input: element(input_kind)?,
        output: element(output_kind)?,
        scale: element(scale_kind)?,
        block_size: block as u32,
        pointer_divisibility_16: [input_ptr, output_ptr, scale_ptr].map(|p| p % 16 == 0),
        count,
        source_path: compiler.source_path(),
    };
    let kernel = unsafe { compiler.jit_fp8(device, &spec)? };
    let mut argv = launch[..3]
        .iter()
        .cloned()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    if quantize {
        argv.push(Scalar::F32(limit as f32));
    }
    match count {
        Count::One => {}
        Count::I32 { .. } => argv.push(Scalar::I32(n as i32)),
        Count::I64 { .. } => argv.push(Scalar::I64(n)),
    }
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    unsafe {
        kernel.run(
            [
                ((n + block - 1) / block)
                    .try_into()
                    .map_err(|_| type_error("signed integer is greater than maximum"))?,
                1,
                1,
            ],
            stream,
            &mut argv,
        )?;
    }
    unsafe {
        rt.op(
            "aten::view.default",
            &[
                out,
                Value::List(shape.into_iter().map(Value::Int).collect()),
            ],
            &empty,
        )
    }
}
impl Runtime {
    /// Connect only these two original registered bodies to the verified runtime
    /// compiler. The remaining general Triton bindings are explicit work.
    pub fn bind_triton_fp8(&self, compiler: Arc<Compiler>) {
        for (name, quantize) in [
            ("quantize_per_tensor_fp8", true),
            ("dequantize_per_tensor_fp8", false),
        ] {
            let compiler = compiler.clone();
            self.backends.bind_reference(
                "triton",
                name,
                Arc::new(move |rt, args, kwargs| run(rt, &compiler, quantize, args, kwargs)),
            );
        }
    }
}
