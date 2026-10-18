//! Frozen INT8 wrapper, two independent autotuners and original candidate calls.
use crate::{
    autotuning::{
        self, BenchFailure, CudaBenchmark, INT8_CONFIGS, Int8ArgumentsState, Int8Autotuner, Int8Key,
    },
    backend_triton_adaln::{count, strides},
    backend_triton_fp8::{bind_arguments, element, info, typename},
    backend_triton_mxfp8::truthy,
    backend_triton_rowwise::{device, rotate, rowwise},
    convrot::InputActivation,
    execution::Runtime,
    execution_value::{Kwargs, Value, get, type_error},
    int8_linear, int8_utils,
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::int8,
};
use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tch::{Kind, Tensor};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn shape(v: Vec<i64>) -> Value {
    Value::List(v.into_iter().map(Value::Int).collect())
}
fn repr(v: &Value) -> Result<String, Error> {
    crate::python_value::repr(v)
}
fn floating_element(kind: Kind) -> Result<crate::triton_frontend::Element, Error> {
    match kind {
        Kind::Double => Ok(crate::triton_frontend::Element::F64),
        kind => element(kind),
    }
}
fn last(v: &[i64]) -> Result<i64, Error> {
    v.last()
        .copied()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))
}
pub(crate) fn eps(rt: &Runtime, v: &Value) -> Result<Value, Error> {
    let invalid = || {
        type_error(format!(
            "rms_norm(): argument 'eps' (position 4) must be float, not {}",
            typename(v)
        ))
    };
    Ok(match v {
        Value::None => Value::None,
        Value::Float(n) => Value::Float(*n),
        Value::Int(n) => Value::Float(*n as f64),
        Value::UInt(n) => Value::Float(*n as f64),
        Value::Bool(b) => Value::Float(f64::from(*b)),
        Value::Tensor(t) => {
            let t = t.lock().unwrap();
            if t.dim() != 0 || t.requires_grad() {
                return Err(invalid());
            }
            drop(t);
            return Ok(Value::Float(
                rt.native
                    .item_double(&unsafe { rt.dequantize_args(v)? }.as_tensor()?)?,
            ));
        }
        Value::Quantized(q) => {
            if !q.lock().unwrap().params.original_shape.is_empty() {
                return Err(invalid());
            }
            return Ok(Value::Float(
                rt.native
                    .item_double(&unsafe { rt.dequantize_args(v)? }.as_tensor()?)?,
            ));
        }
        _ => return Err(invalid()),
    })
}
fn apply_input(
    rt: &Runtime,
    x: &Value,
    mode: &Value,
    weight: &Value,
    epsilon: &Value,
) -> Result<Value, Error> {
    if matches!(mode, Value::None) || matches!(mode,Value::Text(s)if s=="none") {
        return Ok(x.clone());
    }
    let name = if let Value::Text(s) = mode {
        s.as_str()
    } else {
        ""
    };
    match name {
        "gelu_tanh" | "swiglu" | "situ" => {
            let x = unsafe { rt.dequantize_args(x)? }.as_tensor()?;
            Ok(Value::tensor(int8_linear::apply_input(
                &x,
                &if name == "gelu_tanh" {
                    InputActivation::GeluTanh
                } else if name == "situ" {
                    InputActivation::Situ
                } else {
                    InputActivation::SwiGlu
                },
            )?))
        }
        "rms_norm" => {
            if matches!(weight, Value::None) {
                return Err(error(
                    "ValueError",
                    "input_act 'rms_norm' requires act_weight",
                ));
            }
            let (dims, kind, _) = info(x, "shape")?;
            let width = last(&dims)?;
            info(weight, "to")?;
            let w = unsafe {
                rt.op(
                    "aten::to.dtype",
                    &[weight.clone(), Value::DType(kind)],
                    &vec![],
                )?
            };
            unsafe {
                rt.op(
                    "aten::rms_norm.default",
                    &[x.clone(), shape(vec![width]), w, eps(rt, epsilon)?],
                    &vec![],
                )
            }
        }
        _ => Err(error(
            "ValueError",
            format!(
                "unsupported input_act: {} (expected one of ['gelu_tanh', 'none', 'rms_norm', 'situ', 'swiglu'])",
                crate::python_value::repr_with_native(mode, &rt.native)?
            ),
        )),
    }
}
fn hadamard(x: &Value, size: &Value) -> Result<Value, Error> {
    let n = match size {
        Value::Int(n) => *n,
        Value::Bool(_) => {
            return Err(error(
                "ValueError",
                format!(
                    "Regular Hadamard size must be a power of 4, got {}",
                    repr(size)?
                ),
            ));
        }
        Value::Float(n) => {
            return Err(if *n < 4. {
                error(
                    "ValueError",
                    format!(
                        "Regular Hadamard size must be a power of 4, got {}",
                        repr(size)?
                    ),
                )
            } else {
                type_error("unsupported operand type(s) for &: 'float' and 'float'")
            });
        }
        Value::Tensor(_) | Value::Quantized(_) => {
            return Err(Error::BindingGap {
                backend: "triton".into(),
                operation: "INT8 tensor-valued Hadamard size".into(),
            });
        }
        _ => {
            return Err(type_error(format!(
                "'<' not supported between instances of '{}' and 'int'",
                typename(size)
            )));
        }
    };
    let (_, kind, _) = info(x, "dtype")?;
    Ok(Value::tensor(int8_utils::hadamard(n, device(x)?, kind)?))
}
fn stride(v: &Value, index: usize) -> Result<i64, Error> {
    let strides = strides(v)?;
    strides.get(index).copied().ok_or_else(|| {
        error(
            "IndexError",
            format!(
                "Dimension out of range (expected to be in range of [-{}, {}], but got {index})",
                strides.len(),
                strides.len() as i64 - 1
            ),
        )
    })
}
fn scalar_scale(value: &Value, dev: tch::Device) -> Result<Value, Error> {
    let n = match value {
        Value::Float(n) => *n,
        Value::Int(n) => *n as f64,
        Value::Bool(b) => f64::from(*b),
        Value::None | Value::Text(_) | Value::DType(_) | Value::Device(_) => {
            return Err(type_error(format!(
                "must be real number, not {}",
                typename(value)
            )));
        }
        _ => {
            return Err(Error::BindingGap {
                backend: "triton".into(),
                operation: "INT8 sequence-valued non-tensor scale".into(),
            });
        }
    };
    Ok(Value::tensor(
        Tensor::f_from_slice(&[n as f32])?.f_to_device(dev)?,
    ))
}
fn apply_residual(
    rt: &Runtime,
    out: Value,
    residual: &Value,
    scale: &Value,
) -> Result<Value, Error> {
    if matches!(residual, Value::None) {
        return Ok(out);
    }
    if matches!(scale, Value::None) {
        return Err(error("ValueError", "residual requires residual_scale"));
    }
    let kind = info(&out, "dtype")?.1;
    info(residual, "to")?;
    let residual = unsafe {
        rt.op(
            "aten::to.dtype",
            &[residual.clone(), Value::DType(kind)],
            &vec![],
        )?
    };
    info(scale, "to")?;
    let scale = unsafe {
        rt.op(
            "aten::to.dtype",
            &[scale.clone(), Value::DType(kind)],
            &vec![],
        )?
    };
    unsafe { rt.op("aten::addcmul.default", &[residual, out, scale], &vec![]) }
}
fn stamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
fn mapping_present(compiler: &Compiler, which: usize) -> Result<(), Error> {
    if compiler.int8_tuners[which].lock().unwrap().arguments == Int8ArgumentsState::EmptyMapping {
        Ok(())
    } else {
        Err(type_error("'NoneType' object is not a mapping"))
    }
}
fn run_candidate(
    rt: &Runtime,
    compiler: &Compiler,
    per_row: bool,
    index: usize,
    operands: &[Value; 6],
    values: [i64; 9],
    has_bias: bool,
) -> Result<(), Error> {
    let config = INT8_CONFIGS[index];
    let mut kw = operands
        .iter()
        .cloned()
        .zip([
            "a_ptr",
            "b_ptr",
            "c_ptr",
            "a_scale_ptr",
            "b_scale_ptr",
            "bias_ptr",
        ])
        .map(|(v, k)| (k.into(), v))
        .collect::<Kwargs>();
    kw.extend(
        int8::SCALAR_NAMES
            .into_iter()
            .zip(values)
            .map(|(k, v)| (k.into(), Value::Int(v))),
    );
    kw.push(("has_bias".into(), Value::Bool(has_bias)));
    kw.extend(
        [
            ("block_m", i64::from(config.block_m)),
            ("block_n", i64::from(config.block_n)),
            ("block_k", i64::from(config.block_k)),
            ("group_size_m", i64::from(config.group_size_m)),
            ("num_warps", i64::from(config.warps)),
            ("num_ctas", 1),
            ("num_stages", i64::from(config.stages)),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), Value::Int(v))),
    );
    let device = rt.native.current_device()?;
    let stream = rt.native.current_stream(device)?;
    compiler.pre_run(
        if per_row {
            "int8_per_row"
        } else {
            "int8_tensorwise"
        },
        &[],
        &kw,
    )?;
    let pointers: Vec<_> = operands
        .iter()
        .map(|v| info(v, "dtype").map(|i| i.2))
        .collect::<Result<_, _>>()?;
    if info(&operands[1], "dtype")?.1 != Kind::Int8 {
        return Err(Error::BindingGap {
            backend: "triton".into(),
            operation: "INT8 non-int8 weight specialization".into(),
        });
    }
    let counts = values.map(count);
    let spec = int8::Specialization {
        per_row,
        output: floating_element(info(&operands[2], "dtype")?.1)?,
        weight_scale: element(info(&operands[4], "dtype")?.1)?,
        bias: floating_element(info(&operands[5], "dtype")?.1)?,
        has_bias,
        block_m: config.block_m,
        block_n: config.block_n,
        block_k: config.block_k,
        group_size_m: config.group_size_m,
        scalars: counts,
        pointer_divisibility_16: std::array::from_fn(|i| pointers[i] % 16 == 0),
        source_path: "",
        standard_source_path: "",
    };
    let kernel = unsafe { compiler.jit_int8(device, &spec, index)? };
    let mut argv = operands
        .iter()
        .cloned()
        .map(Scalar::TensorObject)
        .collect::<Vec<_>>();
    for (n, c) in values.into_iter().zip(counts) {
        match c {
            crate::triton_frontend::Count::One => {}
            crate::triton_frontend::Count::I32 { .. } => argv.push(Scalar::I32(n as i32)),
            crate::triton_frontend::Count::I64 { .. } => argv.push(Scalar::I64(n)),
        }
    }
    argv.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
    let ceil = |n: i64, b: u32| (i128::from(n) + i128::from(b) - 1) / i128::from(b);
    let grid = i32::try_from(ceil(values[0], config.block_m) * ceil(values[1], config.block_n))
        .map_err(|_| type_error("signed integer is greater than maximum"))?;
    unsafe { kernel.run([grid, 1, 1], stream, &mut argv) }
}
fn matmul(
    rt: &Runtime,
    compiler: &Compiler,
    per_row: bool,
    operands: [Value; 6],
    values: [i64; 9],
    has_bias: bool,
) -> Result<(), Error> {
    let which = usize::from(per_row);
    compiler.int8_tuners[which].lock().unwrap().arguments = Int8ArgumentsState::EmptyMapping;
    let dtypes = operands
        .iter()
        .map(|v| {
            info(v, "dtype").map(|i| {
                crate::registry::AllowedType::DType(i.1)
                    .python_name()
                    .to_string()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let key = Int8Key {
        m: values[0],
        n: values[1],
        k: values[2],
        pointer_dtypes: dtypes.try_into().unwrap(),
    };
    let cached = compiler.int8_tuners[which]
        .lock()
        .unwrap()
        .policy
        .cached_index(&key);
    let selected = if let Some(index) = cached {
        compiler.int8_tuners[which]
            .lock()
            .unwrap()
            .policy
            .best_config = Some(index);
        index
    } else {
        let start = stamp();
        let mut local = Int8Autotuner::default();
        let index = local.select(key.clone(), |index, _| {
            mapping_present(compiler, which).map_err(BenchFailure::Unexpected)?;
            let device = rt
                .native
                .current_device()
                .map_err(BenchFailure::Unexpected)?;
            autotuning::do_bench(
                &CudaBenchmark {
                    native: &rt.native,
                    device,
                },
                || run_candidate(rt, compiler, per_row, index, &operands, values, has_bias),
            )
            .map_err(|e| match &e {
                Error::ReferenceException {
                    class: "OutOfResources",
                    ..
                } => BenchFailure::OutOfResources,
                Error::ReferenceException {
                    class: "CompileTimeAssertionFailure",
                    ..
                } => BenchFailure::CompileTimeAssertionFailure,
                Error::ReferenceException {
                    class: "PTXASError",
                    ..
                } => BenchFailure::PtxasError,
                _ => BenchFailure::Unexpected(e),
            })
        })?;
        let mut state = compiler.int8_tuners[which].lock().unwrap();
        state.bench_time = Some(stamp() - start);
        state.policy.publish_index(key.clone(), index);
        if state.arguments != Int8ArgumentsState::EmptyMapping {
            return Err(type_error("'NoneType' object is not a mapping"));
        }
        state.policy.last_timings = local.last_timings;
        let index = state.policy.cached_index(&key).unwrap();
        state.policy.best_config = Some(index);
        index
    };
    run_candidate(rt, compiler, per_row, selected, &operands, values, has_bias)?;
    compiler.int8_tuners[which].lock().unwrap().arguments = Int8ArgumentsState::None;
    Ok(())
}
pub(crate) fn linear(
    rt: &Runtime,
    compiler: &Compiler,
    args: &[Value],
    kw: &Kwargs,
) -> Result<Value, Error> {
    let names = [
        "x",
        "weight",
        "weight_scale",
        "bias",
        "out_dtype",
        "convrot",
        "convrot_groupsize",
        "input_act",
        "input_act_weight",
        "input_act_eps",
        "residual",
        "residual_scale",
    ];
    bind_arguments("int8_linear", &names, 3, args, kw)?;
    let none = Value::None;
    let x = apply_input(
        rt,
        get(args, kw, 0, "x").unwrap(),
        get(args, kw, 7, "input_act").unwrap_or(&none),
        get(args, kw, 8, "input_act_weight").unwrap_or(&none),
        get(args, kw, 9, "input_act_eps").unwrap_or(&Value::Float(0.)),
    )?;
    let orig = info(&x, "shape")?.0;
    let width = last(&orig)?;
    let x2 = unsafe {
        rt.op(
            "aten::reshape.default",
            &[x.clone(), shape(vec![-1, width])],
            &vec![],
        )?
    };
    let dims = info(&x2, "shape")?.0;
    let (m, k) = (dims[0], dims[1]);
    let weight = get(args, kw, 1, "weight").unwrap().clone();
    let n = *info(&weight, "shape")?
        .0
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let q = if truthy(
        rt,
        get(args, kw, 5, "convrot").unwrap_or(&Value::Bool(false)),
    )? {
        let group = get(args, kw, 6, "convrot_groupsize").unwrap_or(&Value::Int(256));
        let h = hadamard(&x, group)?;
        let rotated = rotate(rt, &x2, &h, group)?;
        rowwise(rt, compiler, rotated)?
    } else {
        rowwise(rt, compiler, x2)?
    };
    let Value::Tuple(q) = q else { unreachable!() };
    let output_type = get(args, kw, 4, "out_dtype")
        .cloned()
        .unwrap_or(Value::DType(Kind::BFloat16));
    let output = unsafe {
        rt.op(
            "aten::empty.memory_format",
            &[shape(vec![m, n])],
            &vec![
                ("device".into(), Value::Device(device(&x)?)),
                ("dtype".into(), output_type),
            ],
        )?
    };
    let mut weight_scale = get(args, kw, 2, "weight_scale").unwrap().clone();
    let mut per_row = false;
    if matches!(weight_scale, Value::Tensor(_) | Value::Quantized(_)) {
        let shape0 = info(&weight_scale, "numel")?.0;
        let numel = shape0.iter().fold(1i128, |a, n| a * i128::from(*n));
        if numel == 1 {
            weight_scale = unsafe {
                rt.op(
                    "aten::reshape.default",
                    &[weight_scale, shape(vec![1])],
                    &vec![],
                )?
            };
        } else {
            per_row = true;
            weight_scale = unsafe {
                rt.op(
                    "aten::reshape.default",
                    &[weight_scale, shape(vec![n])],
                    &vec![],
                )?
            };
            weight_scale = unsafe { rt.op("aten::contiguous.default", &[weight_scale], &vec![])? };
        }
    } else {
        weight_scale = scalar_scale(&weight_scale, device(&x)?)?;
    }
    let bias = get(args, kw, 3, "bias").unwrap_or(&none);
    let has_bias = !matches!(bias, Value::None);
    let bias = if has_bias { bias.clone() } else { x };
    let values = [
        m,
        n,
        k,
        stride(&q[0], 0)?,
        stride(&q[0], 1)?,
        stride(&weight, 1)?,
        stride(&weight, 0)?,
        stride(&output, 0)?,
        stride(&output, 1)?,
    ];
    matmul(
        rt,
        compiler,
        per_row,
        [
            q[0].clone(),
            weight,
            output.clone(),
            q[1].clone(),
            weight_scale,
            bias,
        ],
        values,
        has_bias,
    )?;
    let mut result_shape = orig[..orig.len() - 1].to_vec();
    result_shape.push(n);
    let out = unsafe {
        rt.op(
            "aten::reshape.default",
            &[output, shape(result_shape)],
            &vec![],
        )?
    };
    apply_residual(
        rt,
        out,
        get(args, kw, 10, "residual").unwrap_or(&none),
        get(args, kw, 11, "residual_scale").unwrap_or(&none),
    )
}
impl Runtime {
    pub fn bind_triton_int8(&self, compiler: Arc<Compiler>) {
        self.backends.bind_reference(
            "triton",
            "int8_linear",
            Arc::new(move |rt, args, kw| linear(rt, &compiler, args, kw)),
        );
    }
}
