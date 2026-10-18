//! Original RMS-RoPE wrappers, layout detection and partial-rotary eager calls.
use crate::{
    backend_triton_adaln::{count, strides},
    backend_triton_fp8::{bind_arguments, element, info},
    execution::Runtime,
    execution_bindings::backends::rope::inplace_check,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, rope::rms::Specialization},
};
use std::sync::Arc;
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn contiguous(rt: &Runtime, x: &Value) -> Result<Value, Error> {
    let yes = match x {
        Value::Tensor(t) => t.lock().unwrap().is_contiguous(),
        Value::Quantized(_) => true,
        _ => {
            info(x, "is_contiguous")?;
            unreachable!()
        }
    };
    if yes {
        Ok(x.clone())
    } else {
        unsafe { rt.op("aten::contiguous.default", std::slice::from_ref(x), &vec![]) }
    }
}
#[allow(clippy::too_many_arguments)]
fn apply(
    rt: &Runtime,
    compiler: &Compiler,
    x: &Value,
    f: &Value,
    scale: &Value,
    epsilon: &Value,
    split: bool,
    inplace: bool,
) -> Result<Value, Error> {
    let scale = contiguous(rt, scale)?;
    let (xs, kind, xptr) = info(x, "shape")?;
    if xs.len() != 4 {
        return Err(error(
            "ValueError",
            if xs.len() > 4 {
                "too many values to unpack (expected 4)".into()
            } else {
                format!("not enough values to unpack (expected 4, got {})", xs.len())
            },
        ));
    }
    let (fs, fkind, fptr) = info(f, "shape")?;
    let fb = *fs
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let d = xs[3];
    let valid = fs.len() == 6
        && d != 0
        && d % 2 == 0
        && fs[3..] == [d / 2, 2, 2]
        && (fb == 1 || fb == xs[0]);
    let bnhd = if valid && fs[1] == 1 && (fs[2] == 1 || fs[2] == xs[2]) {
        false
    } else if valid && (fs[1] == 1 || fs[1] == xs[1]) && fs[2] == 1 {
        true
    } else {
        return Err(error(
            "ValueError",
            "RMS-RoPE frequencies are not broadcastable to input",
        ));
    };
    let head = if bnhd { 2 } else { 1 };
    let seq = if bnhd { 1 } else { 2 };
    let out = if inplace {
        x.clone()
    } else {
        unsafe { rt.op("aten::empty_like.default", std::slice::from_ref(x), &vec![])? }
    };
    let block = (d as u64 / 2).next_power_of_two() as u32;
    let sx = strides(x)?;
    let so = strides(&out)?;
    let sf = strides(f)?;
    let dims = [
        d, fb, fs[seq], sx[0], sx[head], sx[seq], sx[3], so[0], so[head], so[seq], so[3], sf[0],
        sf[seq], sf[3], sf[4], sf[5],
    ];
    let mut launch = vec![x.clone(), f.clone(), scale.clone(), out.clone()];
    launch.extend(dims.into_iter().map(Value::Int));
    launch.push(epsilon.clone());
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        "rms_rope",
        &launch,
        &vec![
            ("compute_dtype".into(), Value::DType(fkind)),
            ("norm_dtype".into(), Value::DType(kind)),
            ("block_size".into(), Value::Int(i64::from(block))),
            ("split_half".into(), Value::Bool(split)),
        ],
    )?;
    let (epsilon_type, epsilon_value) = crate::backend_triton_scalar::argument(epsilon)?;
    let (_, skind, sptr) = info(&scale, "dtype")?;
    let (_, _, optr) = info(&out, "dtype")?;
    let pointers = [xptr, fptr, sptr, optr];
    let counts = dims.map(count);
    let spec = Specialization {
        input: element(kind)?,
        scale: element(skind)?,
        frequencies: element(fkind)?,
        block_size: block,
        split_half: split,
        scalars: counts,
        pointer_divisibility_16: pointers.map(|p| p % 16 == 0),
        source_path: "",
        standard_source_path: "",
    };
    let kernel = unsafe { compiler.jit_rms_rope_argument(device, &spec, &epsilon_type)? };
    let mut argv = launch[..4]
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
    argv.extend(epsilon_value);
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    let grid = [xs[0], xs[head], xs[seq]].map(|n| {
        n.try_into()
            .map_err(|_| type_error("signed integer is greater than maximum"))
    });
    let [a, b, c] = grid;
    unsafe {
        kernel.run([a?, b?, c?], stream, &mut argv)?;
    }
    Ok(out)
}
fn run(
    rt: &Runtime,
    compiler: &Compiler,
    op: &str,
    args: &[Value],
    kw: &Kwargs,
) -> Result<Value, Error> {
    let single = op.ends_with('1') || op.ends_with("1_");
    let inplace = op.ends_with('_');
    let split = op.contains("split_half");
    let names: &[&str] = if single {
        &["x", "freqs_cis", "scale", "epsilon"]
    } else if split {
        &[
            "q",
            "k",
            "freqs_cis",
            "q_scale",
            "k_scale",
            "epsilon",
            "rot_dim",
        ]
    } else {
        &["q", "k", "freqs_cis", "q_scale", "k_scale", "epsilon"]
    };
    bind_arguments(op, names, if single { 3 } else { 4 }, args, kw)?;
    let x = get(args, kw, 0, names[0]).unwrap();
    let k = if single {
        None
    } else {
        Some(get(args, kw, 1, "k").unwrap())
    };
    let f = get(args, kw, if single { 1 } else { 2 }, "freqs_cis").unwrap();
    let scale = get(
        args,
        kw,
        if single { 2 } else { 3 },
        if single { "scale" } else { "q_scale" },
    )
    .unwrap();
    let kscale = get(args, kw, 4, "k_scale")
        .filter(|v| !matches!(v, Value::None))
        .unwrap_or(scale);
    let epsilon = get(args, kw, if single { 3 } else { 5 }, "epsilon")
        .cloned()
        .unwrap_or(Value::Float(1e-6));
    if !single && split {
        let rot = get(args, kw, 6, "rot_dim")
            .cloned()
            .unwrap_or(Value::Int(0));
        let truthy = crate::backend_triton_mxfp8::truthy(rt, &rot)?;
        if truthy {
            let (shape, _, _) = info(x, "shape")?;
            let d = *shape
                .last()
                .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
            let equal = match rot {
                Value::Int(n) => n == d,
                Value::Bool(b) => i64::from(b) == d,
                Value::Float(n) => n == d as f64,
                Value::Tensor(_) | Value::Quantized(_) => {
                    crate::backend_triton_mxfp8::truthy(rt, &unsafe {
                        rt.op("aten::eq.Scalar", &[rot.clone(), Value::Int(d)], &vec![])?
                    })?
                }
                _ => false,
            };
            if !equal {
                return unsafe {
                    rt.backends.call_bound(
                        rt,
                        "eager",
                        op,
                        &vec![
                            ("q".into(), x.clone()),
                            ("k".into(), k.unwrap().clone()),
                            ("freqs_cis".into(), f.clone()),
                            ("q_scale".into(), scale.clone()),
                            ("k_scale".into(), kscale.clone()),
                            ("epsilon".into(), epsilon),
                            ("rot_dim".into(), rot),
                        ],
                    )
                };
            }
        }
    }
    if inplace {
        let x = x.as_tensor()?;
        let k = k.map(Value::as_tensor).transpose()?;
        let f = f.as_tensor()?;
        let s = scale.as_tensor()?;
        let ks = kscale.as_tensor()?;
        let mut xs = vec![&x];
        if let Some(k) = &k {
            xs.push(k);
        }
        let mut readonly = vec![&f, &s];
        if !single {
            readonly.push(&ks);
        }
        inplace_check(rt, &xs, &readonly)?;
    }
    let first = apply(rt, compiler, x, f, scale, &epsilon, split, inplace)?;
    if let Some(k) = k {
        Ok(Value::Tuple(vec![
            first,
            apply(rt, compiler, k, f, kscale, &epsilon, split, inplace)?,
        ]))
    } else {
        Ok(first)
    }
}
impl Runtime {
    /// Connect the original eight RMS-RoPE wrappers and partial-rotary fallback.
    pub fn bind_triton_rms_rope(&self, compiler: Arc<Compiler>) {
        for op in [
            "rms_rope1",
            "rms_rope1_",
            "rms_rope",
            "rms_rope_",
            "rms_rope_split_half1",
            "rms_rope_split_half1_",
            "rms_rope_split_half",
            "rms_rope_split_half_",
        ] {
            let compiler = compiler.clone();
            self.backends.bind_reference(
                "triton",
                op,
                Arc::new(move |rt, args, kw| run(rt, &compiler, op, args, kw)),
            );
        }
    }
}
