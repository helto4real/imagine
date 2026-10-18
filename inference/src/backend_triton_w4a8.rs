//! Frozen grouped INT4/INT6 unpacking followed by the shared Triton INT8 body.
use crate::{
    backend_triton_adaln::{count, strides},
    backend_triton_fp8::{bind_arguments, info},
    backend_triton_int8,
    execution::Runtime,
    execution_bindings::backends::w4a8::Weight,
    execution_value::{Kwargs, Value, get},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, w4a8},
};
use std::sync::Arc;
use tch::{Kind, Tensor};
fn contiguous(rt: &Runtime, v: Value) -> Result<Value, Error> {
    unsafe { rt.op("aten::contiguous.default", &[v], &vec![]) }
}
fn float_contiguous(rt: &Runtime, v: Value) -> Result<Value, Error> {
    let v = unsafe { rt.op("aten::to.dtype", &[v, Value::DType(Kind::Float)], &vec![])? };
    contiguous(rt, v)
}
fn decode(
    rt: &Runtime,
    compiler: &Compiler,
    args: &[Value],
    kw: &Kwargs,
    weight: &Weight,
) -> Result<Value, Error> {
    let (n, k, bits) = weight.geometry;
    let qi = contiguous(rt, get(args, kw, 1, "qdata").unwrap().clone())?;
    let rel = float_contiguous(rt, get(args, kw, 2, "s_rel").unwrap().clone())?;
    let cb = match get(args, kw, 4, "codebook") {
        None | Some(Value::None) => {
            Value::tensor(Tensor::f_empty([16], (Kind::Float, weight.q.device()))?)
        }
        Some(v) => float_contiguous(rt, v.clone())?,
    };
    let out = Value::tensor(Tensor::f_empty([n, k], (Kind::Int8, weight.q.device()))?);
    let qstride = strides(&qi)?;
    let sstride = strides(&rel)?;
    let ostride = strides(&out)?;
    let values = [
        n,
        k,
        weight.group,
        qstride[0],
        qstride[1],
        sstride[0],
        sstride[1],
        ostride[0],
        ostride[1],
    ];
    let mut launch = vec![qi, rel, cb, out.clone()];
    launch.extend(values.map(Value::Int));
    let has_cb = weight.codebook.is_some();
    let launch_kw = vec![
        ("has_cb".into(), Value::Bool(has_cb)),
        ("bits".into(), Value::Int(bits)),
        ("block_n".into(), Value::Int(32)),
        ("block_kh".into(), Value::Int(128)),
    ];
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run("w4a8_unpack", &launch, &launch_kw)?;
    let pointers: Vec<_> = launch[..4]
        .iter()
        .map(|v| info(v, "dtype").map(|v| v.2))
        .collect::<Result<_, _>>()?;
    let scalars = values.map(count);
    let spec = w4a8::Specialization {
        has_cb,
        bits: bits as u32,
        block_n: 32,
        block_kh: 128,
        scalars,
        pointer_divisibility_16: std::array::from_fn(|i| pointers[i] % 16 == 0),
        source_path: "",
    };
    let kernel = unsafe { compiler.jit_w4a8_unpack(device, &spec)? };
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
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    let grid = [((n + 31) / 32) as i32, ((k / 2 + 127) / 128) as i32, 1];
    unsafe { kernel.run(grid, stream, &mut argv)? };
    Ok(out)
}
fn linear(rt: &Runtime, compiler: &Compiler, args: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    bind_arguments(
        "w4a8_int8_linear",
        &[
            "x",
            "qdata",
            "s_rel",
            "s_channel",
            "codebook",
            "correction",
            "bias",
            "group_size",
            "convrot_groupsize",
            "out_dtype",
        ],
        4,
        args,
        kw,
    )?;
    let w = Weight::parse(args, kw, true)?;
    let x = get(args, kw, 0, "x").unwrap().clone();
    let dims = info(&x, "shape")?.0;
    let width = dims
        .last()
        .copied()
        .ok_or_else(|| Error::ReferenceException {
            class: "IndexError",
            message: "tuple index out of range".into(),
        })?;
    if width != w.geometry.1 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: format!("Input K={width} does not match qdata K={}", w.geometry.1),
        });
    }
    if w.correction.is_some() {
        return crate::execution_bindings::backends::w4a8::run(rt, false, true, args, kw);
    }
    let decoded = decode(rt, compiler, args, kw, &w)?;
    backend_triton_int8::linear(
        rt,
        compiler,
        &[x, decoded, get(args, kw, 3, "s_channel").unwrap().clone()],
        &vec![
            (
                "bias".into(),
                get(args, kw, 6, "bias").cloned().unwrap_or(Value::None),
            ),
            (
                "out_dtype".into(),
                get(args, kw, 9, "out_dtype")
                    .cloned()
                    .unwrap_or(Value::DType(Kind::BFloat16)),
            ),
            ("convrot".into(), Value::Bool(true)),
            ("convrot_groupsize".into(), Value::Int(w.rotation)),
        ],
    )
}
impl Runtime {
    pub fn bind_triton_w4a8(&self, compiler: Arc<Compiler>) {
        self.backends.bind_reference(
            "triton",
            "w4a8_int8_linear",
            Arc::new(move |rt, args, kw| linear(rt, &compiler, args, kw)),
        );
    }
}
