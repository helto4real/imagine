//! Original MXFP8 wrapper, fused quantizer and blocked E8M0 scale view.
use crate::{
    backend_triton_adaln::count,
    backend_triton_fp8::{bind_arguments, element, info},
    backend_triton_rowwise::device,
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    native::{Error, ShellDType},
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, mxfp8::Specialization},
};
use std::sync::Arc;
use tch::{Kind, Tensor};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub(crate) fn truthy(rt: &Runtime, value: &Value) -> Result<bool, Error> {
    Ok(match value {
        Value::None => false,
        Value::Bool(b) => *b,
        Value::Int(n) => *n != 0,
        Value::UInt(n) => *n != 0,
        Value::Float(n) => *n != 0.,
        Value::Text(s) => !s.is_empty(),
        Value::List(v) | Value::Tuple(v) => !v.is_empty(),
        Value::Dict(v) => !v.is_empty(),
        Value::DType(_) | Value::Device(_) => true,
        Value::Tensor(_) | Value::Quantized(_) => match unsafe {
            rt.op(
                "aten::is_nonzero.default",
                std::slice::from_ref(value),
                &vec![],
            )?
        } {
            Value::Bool(b) => b,
            _ => unreachable!(),
        },
    })
}
fn shape2(value: &Value) -> Result<[i64; 2], Error> {
    let (shape, _, _) = info(value, "shape")?;
    if shape.len() != 2 {
        return Err(error(
            "ValueError",
            if shape.len() > 2 {
                "too many values to unpack (expected 2)".into()
            } else {
                format!(
                    "not enough values to unpack (expected 2, got {})",
                    shape.len()
                )
            },
        ));
    }
    Ok([shape[0], shape[1]])
}
fn run(rt: &Runtime, compiler: &Compiler, args: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    bind_arguments("quantize_mxfp8", &["x", "pad_32x"], 1, args, kw)?;
    let mut x = get(args, kw, 0, "x").unwrap().clone();
    if truthy(
        rt,
        get(args, kw, 1, "pad_32x").unwrap_or(&Value::Bool(false)),
    )? {
        let [rows, cols] = shape2(&x)?;
        let pr = ((i128::from(rows) + 31) / 32 * 32 - i128::from(rows)) as i64;
        let pc = ((i128::from(cols) + 31) / 32 * 32 - i128::from(cols)) as i64;
        if pr > 0 || pc > 0 {
            x = unsafe {
                rt.op(
                    "aten::constant_pad_nd.default",
                    &[
                        x,
                        Value::List(vec![
                            Value::Int(0),
                            Value::Int(pc),
                            Value::Int(0),
                            Value::Int(pr),
                        ]),
                        Value::Float(0.),
                    ],
                    &vec![],
                )?
            };
        }
    }
    let [m, n] = shape2(&x)?;
    let blocks = n / 32;
    let contiguous = match &x {
        Value::Tensor(t) => t.lock().unwrap().is_contiguous(),
        Value::Quantized(_) => true,
        _ => unreachable!(),
    };
    if !contiguous {
        x = unsafe { rt.op("aten::contiguous.default", &[x], &vec![])? };
    }
    let padded_rows = (i128::from(m) + 127) / 128 * 128;
    let padded_rows = i64::try_from(padded_rows).map_err(|_| Error::BindingGap {
        backend: "triton".into(),
        operation: "MXFP8 oversized scale allocation".into(),
    })?;
    let padded_cols = (blocks + 3) / 4 * 4;
    let dev = device(&x)?;
    let output = Value::tensor(Tensor::f_empty([m, n], (Kind::Float8e4m3fn, dev))?);
    let scales = Value::tensor(Tensor::f_zeros(
        [padded_rows, padded_cols],
        (Kind::Uint8, dev),
    )?);
    let total = i128::from(m) * i128::from(blocks);
    let bpp = if total < 1024 {
        1
    } else if total < 4096 {
        2
    } else {
        4
    };
    let dims = [m, n, blocks, m, blocks, padded_cols];
    let mut launch = vec![x.clone(), output.clone(), scales.clone()];
    launch.extend(dims.into_iter().map(Value::Int));
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        "mxfp8",
        &launch,
        &vec![
            ("block_size".into(), Value::Int(32)),
            ("blocks_per_program".into(), Value::Int(bpp)),
        ],
    )?;
    let (_, kind, xptr) = info(&x, "dtype")?;
    let optr = info(&output, "dtype")?.2;
    let sptr = info(&scales, "dtype")?.2;
    let pointers = [xptr, optr, sptr];
    let counts = dims.map(count);
    let spec = Specialization {
        input: element(kind)?,
        block_size: 32,
        blocks_per_program: bpp as u32,
        scalars: counts,
        pointer_divisibility_16: pointers.map(|p| p % 16 == 0),
        source_path: "",
        standard_source_path: "",
    };
    let kernel = unsafe { compiler.jit_mxfp8(device, &spec)? };
    let mut argv = launch[..3]
        .iter()
        .cloned()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    for (n, c) in dims.into_iter().zip(counts) {
        match c {
            Count::One => {}
            Count::I32 { .. } => argv.push(Scalar::I32(n as i32)),
            Count::I64 { .. } => argv.push(Scalar::I64(n)),
        }
    }
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    let gx = m
        .try_into()
        .map_err(|_| type_error("signed integer is greater than maximum"))?;
    let gy = ((blocks + bpp - 1) / bpp)
        .try_into()
        .map_err(|_| type_error("signed integer is greater than maximum"))?;
    unsafe {
        kernel.run([gx, gy, 1], stream, &mut argv)?;
    }
    let view = rt
        .native
        .view_shell_dtype(&scales.as_tensor()?, ShellDType::E8M0)?;
    Ok(Value::Tuple(vec![output, Value::tensor(view)]))
}
impl Runtime {
    /// Connect the original fused MXFP8 quantizer without an eager substitution.
    pub fn bind_triton_mxfp8(&self, compiler: Arc<Compiler>) {
        self.backends.bind_reference(
            "triton",
            "quantize_mxfp8",
            Arc::new(move |rt, args, kw| run(rt, &compiler, args, kw)),
        );
    }
}
