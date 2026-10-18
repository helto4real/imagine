//! Original flexible-layout RoPE wrappers and their paired-launch split.
use crate::{
    backend_triton_adaln::{count, strides},
    backend_triton_fp8::{bind_arguments, element, info},
    execution::Runtime,
    execution_bindings::backends::rope::inplace_check,
    execution_value::{Kwargs, Value, get, type_error},
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, rope::Specialization},
};
use std::sync::Arc;

fn value_error(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
fn padded(mut values: Vec<i64>, rank: usize, fill: i64) -> Vec<i64> {
    let mut output = vec![fill; rank.saturating_sub(values.len())];
    output.append(&mut values);
    output
}
fn apply(
    rt: &Runtime,
    compiler: &Compiler,
    x: &Value,
    k: Option<&Value>,
    f: &Value,
    split: bool,
    inplace: bool,
) -> Result<(Value, Value), Error> {
    let (xs, xkind, xptr) = info(x, "shape")?;
    if let Some(k) = k {
        let (ks, _, _) = info(k, "shape")?;
        if xs != ks || strides(x)? != strides(k)? {
            return Ok((
                apply(rt, compiler, x, None, f, split, inplace)?.0,
                apply(rt, compiler, k, None, f, split, inplace)?.0,
            ));
        }
    }
    if xs.len() > 4 {
        return Err(value_error("too many values to unpack (expected 4)"));
    }
    let shape = padded(xs, 4, 1);
    let (fs, fkind, fptr) = info(f, "shape")?;
    let fs = padded(fs, 6, 1);
    let out = if inplace {
        x.clone()
    } else {
        unsafe { rt.op("aten::empty_like.default", std::slice::from_ref(x), &vec![])? }
    };
    let kout = match k {
        None => Value::None,
        Some(k) if inplace => k.clone(),
        Some(k) => unsafe { rt.op("aten::empty_like.default", std::slice::from_ref(k), &vec![])? },
    };
    let pairs = shape[3] / 2;
    let total = shape[..3]
        .iter()
        .try_fold(pairs, |n, d| n.checked_mul(*d))
        .ok_or_else(|| value_error("numel: integer multiplication overflow"))?;
    let block = if total < 4096 {
        256
    } else if total < 32768 {
        512
    } else {
        1024
    };
    let sx = padded(strides(x)?, 4, 0);
    let so = padded(strides(&out)?, 4, 0);
    let sf = padded(strides(f)?, 6, 0);
    let dims = [
        shape[0], shape[1], shape[2], shape[3], fs[0], fs[1], fs[2], sx[0], sx[1], sx[2], sx[3],
        so[0], so[1], so[2], so[3], sf[0], sf[1], sf[2], sf[3], sf[4], sf[5],
    ];
    let mut launch = vec![
        x.clone(),
        k.cloned().unwrap_or(Value::None),
        f.clone(),
        out.clone(),
        kout.clone(),
    ];
    launch.extend(dims.into_iter().map(Value::Int));
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        "rope",
        &launch,
        &vec![
            ("compute_dtype".into(), Value::DType(fkind)),
            ("block_size".into(), Value::Int(block)),
            ("split_half".into(), Value::Bool(split)),
        ],
    )?;
    let (kkind, kptr) = if let Some(k) = k {
        let (_, kind, ptr) = info(k, "dtype")?;
        (Some(element(kind)?), ptr)
    } else {
        (None, 0)
    };
    let (_, _, outptr) = info(&out, "dtype")?;
    let koutptr = if k.is_some() {
        info(&kout, "dtype")?.2
    } else {
        0
    };
    let ptrs = [xptr, kptr, fptr, outptr, koutptr];
    let counts = dims.map(count);
    let spec = Specialization {
        input: element(xkind)?,
        key: kkind,
        frequencies: element(fkind)?,
        block_size: block as u32,
        split_half: split,
        scalars: counts,
        pointer_divisibility_16: ptrs.map(|p| p % 16 == 0),
        source_path: "",
    };
    let kernel = unsafe { compiler.jit_rope(device, &spec)? };
    let mut argv = launch[..5]
        .iter()
        .enumerate()
        .filter_map(|(i, value)| {
            (!(k.is_none() && matches!(i, 1 | 4))).then_some(Scalar::TensorObject(value.clone()))
        })
        .collect::<Vec<_>>();
    for (n, c) in dims.into_iter().zip(counts) {
        match c {
            Count::One => {}
            Count::I32 { .. } => argv.push(Scalar::I32(n as i32)),
            Count::I64 { .. } => argv.push(Scalar::I64(n)),
        }
    }
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    let grid = (total + block - 1) / block;
    unsafe {
        kernel.run(
            [
                grid.try_into()
                    .map_err(|_| type_error("signed integer is greater than maximum"))?,
                1,
                1,
            ],
            stream,
            &mut argv,
        )?;
    }
    Ok((out, kout))
}
fn run(
    rt: &Runtime,
    compiler: &Compiler,
    op: &str,
    args: &[Value],
    kwargs: &Kwargs,
) -> Result<Value, Error> {
    let single = op.ends_with('1') || op.ends_with("1_");
    let inplace = op.ends_with('_');
    let split = op.contains("split_half");
    let names: &[&str] = if single {
        &["x", "freqs_cis"]
    } else {
        &["xq", "xk", "freqs_cis"]
    };
    bind_arguments(op, names, names.len(), args, kwargs)?;
    let x = get(args, kwargs, 0, names[0]).unwrap();
    let k = if single {
        None
    } else {
        Some(get(args, kwargs, 1, "xk").unwrap())
    };
    let f = get(args, kwargs, names.len() - 1, "freqs_cis").unwrap();
    if inplace {
        let x = x.as_tensor()?;
        let k = k.map(Value::as_tensor).transpose()?;
        let f = f.as_tensor()?;
        let mut xs = vec![&x];
        if let Some(k) = &k {
            xs.push(k);
        }
        inplace_check(rt, &xs, &[&f])?;
    }
    let (x, k) = apply(rt, compiler, x, k, f, split, inplace)?;
    Ok(if single { x } else { Value::Tuple(vec![x, k]) })
}
impl Runtime {
    /// Connect all eight original flexible-layout RoPE wrappers.
    pub fn bind_triton_rope(&self, compiler: Arc<Compiler>) {
        for op in [
            "apply_rope1",
            "apply_rope1_",
            "apply_rope",
            "apply_rope_",
            "apply_rope_split_half1",
            "apply_rope_split_half1_",
            "apply_rope_split_half",
            "apply_rope_split_half_",
        ] {
            let compiler = compiler.clone();
            self.backends.bind_reference(
                "triton",
                op,
                Arc::new(move |rt, args, kwargs| run(rt, &compiler, op, args, kwargs)),
            );
        }
    }
}
