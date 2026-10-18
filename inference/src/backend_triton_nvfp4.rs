//! Frozen NVFP4 wrappers; native SM89 assembler failures are retained.
use crate::{
    backend_triton_adaln::count,
    backend_triton_fp8::{bind_arguments, element, info},
    backend_triton_mxfp8::truthy,
    backend_triton_rowwise::device,
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, nvfp4},
};
use std::sync::Arc;
use tch::{Kind, Tensor};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn shape(dims: Vec<i64>) -> Value {
    Value::List(dims.into_iter().map(Value::Int).collect())
}
fn reshape(rt: &Runtime, x: Value, dims: Vec<i64>) -> Result<Value, Error> {
    unsafe { rt.op("aten::reshape.default", &[x, shape(dims)], &vec![]) }
}
fn conditional(v: &Value) -> (bool, Option<&'static str>) {
    match v {
        Value::Bool(b) => (*b, None),
        Value::Int(n) => (*n != 0, None),
        Value::UInt(n) => (*n != 0, None),
        Value::None => (false, None),
        _ => (false, Some(crate::backend_triton_fp8::typename(v))),
    }
}
fn pad(rt: &Runtime, mut x: Value) -> Result<Value, Error> {
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
    let pr = ((i128::from(dims[0]) + 15) / 16 * 16 - i128::from(dims[0])) as i64;
    let pc = ((i128::from(dims[1]) + 15) / 16 * 16 - i128::from(dims[1])) as i64;
    if pr > 0 || pc > 0 {
        x = unsafe {
            rt.op(
                "aten::constant_pad_nd.default",
                &[x, shape(vec![0, pc, 0, pr]), Value::Float(0.)],
                &vec![],
            )?
        };
    }
    Ok(x)
}
fn quantize(
    rt: &Runtime,
    compiler: &Compiler,
    args: &[Value],
    kw: &Kwargs,
) -> Result<Value, Error> {
    bind_arguments(
        "quantize_nvfp4",
        &["x", "per_tensor_scale", "epsilon", "pad_16x", "hi_first"],
        2,
        args,
        kw,
    )?;
    let mut x = get(args, kw, 0, "x").unwrap().clone();
    let scale = get(args, kw, 1, "per_tensor_scale").unwrap();
    let mut orig = info(&x, "shape")?.0;
    if truthy(
        rt,
        get(args, kw, 3, "pad_16x").unwrap_or(&Value::Bool(false)),
    )? {
        x = pad(rt, x)?;
        orig = info(&x, "shape")?.0;
    }
    let first = *orig
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let x = reshape(rt, x, vec![first, -1, 16])?;
    let dims = info(&x, "shape")?.0;
    let (m, blocks) = (dims[0], dims[1]);
    let n = blocks * 16;
    let x = reshape(rt, x, vec![m, n])?;
    let x = unsafe { rt.op("aten::contiguous.default", &[x], &vec![])? };
    let rows = i64::try_from((i128::from(m) + 127) / 128 * 128).map_err(|_| Error::BindingGap {
        backend: "triton".into(),
        operation: "NVFP4 oversized scale allocation".into(),
    })?;
    let cols = (blocks + 3) / 4 * 4;
    let dev = device(&x)?;
    let output = Value::tensor(Tensor::f_empty([m, n / 2], (Kind::Uint8, dev))?);
    let scales = Value::tensor(Tensor::f_zeros([rows, cols], (Kind::Float8e4m3fn, dev))?);
    let total = i128::from(m) * i128::from(blocks);
    let bpp = if total < 1024 {
        1
    } else if total < 4096 {
        2
    } else {
        4
    };
    let hi = get(args, kw, 4, "hi_first")
        .cloned()
        .unwrap_or(Value::Bool(true));
    let values = [m, n, blocks, m, blocks, rows, cols];
    let mut launch = vec![x.clone(), output.clone(), scales.clone(), scale.clone()];
    launch.extend(values.into_iter().map(Value::Int));
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        "nvfp4",
        &launch,
        &vec![
            ("block_size".into(), Value::Int(16)),
            ("blocks_per_program".into(), Value::Int(bpp)),
            ("hi_first".into(), hi.clone()),
        ],
    )?;
    let (kind, pointers) = (
        info(&x, "dtype")?.1,
        [
            info(&x, "dtype")?.2,
            info(&output, "dtype")?.2,
            info(&scales, "dtype")?.2,
            info(scale, "dtype")?.2,
        ],
    );
    let counts = values.map(count);
    let spec = nvfp4::Specialization {
        input: element(kind)?,
        global_scale: element(info(scale, "dtype")?.1)?,
        block_size: 16,
        blocks_per_program: bpp as u32,
        hi_first: conditional(&hi).0,
        scalars: counts,
        pointer_divisibility_16: pointers.map(|p| p % 16 == 0),
        source_path: "",
        standard_source_path: "",
    };
    let kernel = unsafe {
        compiler.jit_nvfp4_literal(
            device,
            &spec,
            &crate::python_value::constexpr_key(&hi, &rt.native)?,
            conditional(&hi).1,
        )?
    };
    let mut argv = arguments(&launch[..4], values, counts);
    unsafe {
        kernel.run(
            [grid(m)?, grid((blocks + bpp - 1) / bpp)?, 1],
            stream,
            &mut argv,
        )?;
    }
    let last = orig.last_mut().unwrap();
    *last /= 2;
    let output = reshape(rt, output, orig)?;
    Ok(Value::Tuple(vec![output, scales]))
}
fn grid(n: i64) -> Result<i32, Error> {
    n.try_into()
        .map_err(|_| type_error("signed integer is greater than maximum"))
}
fn arguments<const N: usize>(
    pointers: &[Value],
    values: [i64; N],
    counts: [Count; N],
) -> Vec<Scalar> {
    let mut out = pointers
        .iter()
        .cloned()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    for (n, c) in values.into_iter().zip(counts) {
        match c {
            Count::One => {}
            Count::I32 { .. } => out.push(Scalar::I32(n as i32)),
            Count::I64 { .. } => out.push(Scalar::I64(n)),
        }
    }
    out.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    out
}
fn dequantize(
    rt: &Runtime,
    compiler: &Compiler,
    args: &[Value],
    kw: &Kwargs,
) -> Result<Value, Error> {
    bind_arguments(
        "dequantize_nvfp4",
        &[
            "qx",
            "per_tensor_scale",
            "block_scales",
            "output_type",
            "hi_first",
        ],
        3,
        args,
        kw,
    )?;
    let x = get(args, kw, 0, "qx").unwrap();
    let scale = get(args, kw, 1, "per_tensor_scale").unwrap();
    let scales = get(args, kw, 2, "block_scales").unwrap();
    let mut dims = info(x, "shape")?.0;
    let n = dims
        .last()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?
        .checked_mul(2)
        .ok_or_else(|| Error::BindingGap {
            backend: "triton".into(),
            operation: "NVFP4 oversized output allocation".into(),
        })?;
    let blocks = n / 16;
    let cb = (blocks + 3) / 4;
    let pc = cb * 4;
    *dims.last_mut().unwrap() = n;
    let output_type = get(args, kw, 3, "output_type")
        .cloned()
        .unwrap_or(Value::DType(Kind::BFloat16));
    let output = unsafe {
        rt.op(
            "aten::empty.memory_format",
            &[shape(dims)],
            &vec![
                ("dtype".into(), output_type),
                ("device".into(), Value::Device(device(x)?)),
            ],
        )?
    };
    let hi = get(args, kw, 4, "hi_first")
        .cloned()
        .unwrap_or(Value::Bool(true));
    let values = [n, blocks, cb, pc];
    let mut launch = vec![x.clone(), scales.clone(), scale.clone(), output.clone()];
    launch.extend(values.into_iter().map(Value::Int));
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        "dequant_nvfp4",
        &launch,
        &vec![
            ("block_size".into(), Value::Int(16)),
            ("tile_size".into(), Value::Int(128)),
            ("hi_first".into(), hi.clone()),
        ],
    )?;
    let pointers = [
        info(x, "dtype")?.2,
        info(scales, "dtype")?.2,
        info(scale, "dtype")?.2,
        info(&output, "dtype")?.2,
    ];
    let counts = values.map(count);
    let spec = nvfp4::dequantize::Specialization {
        output: match info(&output, "dtype")?.1 {
            tch::Kind::Double => crate::triton_frontend::Element::F64,
            kind => element(kind)?,
        },
        block_scale: element(info(scales, "dtype")?.1)?,
        global_scale: element(info(scale, "dtype")?.1)?,
        block_size: 16,
        tile_size: 128,
        hi_first: conditional(&hi).0,
        scalars: counts,
        pointer_divisibility_16: pointers.map(|p| p % 16 == 0),
        source_path: "",
    };
    let kernel = unsafe {
        compiler.jit_dequant_nvfp4_literal(
            device,
            &spec,
            &crate::python_value::constexpr_key(&hi, &rt.native)?,
            conditional(&hi).1,
        )?
    };
    let mut argv = arguments(&launch[..4], values, counts);
    let numel = info(x, "numel")?.0.into_iter().product::<i64>();
    unsafe {
        kernel.run([grid((numel + 127) / 128)?, 1, 1], stream, &mut argv)?;
    }
    Ok(output)
}
impl Runtime {
    pub fn bind_triton_nvfp4(&self, compiler: Arc<Compiler>) {
        for (name, quant) in [("quantize_nvfp4", true), ("dequantize_nvfp4", false)] {
            let compiler = compiler.clone();
            self.backends.bind_reference(
                "triton",
                name,
                Arc::new(move |rt, args, kw| {
                    if quant {
                        quantize(rt, &compiler, args, kw)
                    } else {
                        dequantize(rt, &compiler, args, kw)
                    }
                }),
            );
        }
    }
}
