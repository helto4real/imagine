//! Kitchen 0.2.38 resident MMA-packed W4A8/W6A8, original eager/CUDA dispatch.
use super::*;
use crate::packing::mma;
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
};
use tch::Device;
fn fail(message: &str) -> Error {
    Error::ReferenceException {
        class: "RuntimeError",
        message: message.into(),
    }
}
fn fixed_decode_lut(device: Device) -> Result<Tensor, Error> {
    static LUTS: OnceLock<Mutex<HashMap<Device, Tensor>>> = OnceLock::new();
    let mut cache = LUTS.get_or_init(Default::default).lock().unwrap();
    if let Some(value) = cache.get(&device) {
        return Ok(value.shallow_clone());
    }
    let cb = Tensor::f_from_slice(&w4a8_quant::LUT)?;
    let scales = Tensor::f_arange(256, (Kind::Int64, Device::Cpu))?
        .f_to_kind(Kind::Uint8)?
        .f_view_dtype(Kind::Float8e4m3fn)?
        .f_to_kind(Kind::Float)?;
    let table = scales
        .f_unsqueeze(1)?
        .f_mul(&cb.f_unsqueeze(0)?)?
        .f_round_()?
        .f_clamp_(-127, 127)?
        .f_to_kind(Kind::Int8)?
        .f_contiguous()?;
    let ctx = crate::kitchen_sol::allocation_context();
    if let Some(ctx) = &ctx {
        ctx.enter()?;
    }
    let allocated = table.f_to_device(device).map_err(Error::from);
    let suppressed = if let Some(ctx) = ctx {
        ctx.exit(allocated.as_ref().err())?
    } else {
        false
    };
    if allocated.is_err() && suppressed {
        return Err(Error::ReferenceException {
            class: "UnboundLocalError",
            message:
                "cannot access local variable 'decode_lut' where it is not associated with a value"
                    .into(),
        });
    }
    let value = allocated?;
    cache.insert(device, value.shallow_clone());
    Ok(value)
}
fn cluster_ok(rt: &Runtime, k: i64, mode: &Value, device: Device) -> Result<bool, Error> {
    let supported =
        matches!(mode, Value::None) || matches!(mode,Value::Text(s)if s=="swiglu"||s=="rms_norm");
    if let Device::Cuda(i) = device {
        Ok(supported
            && k % 256 == 0
            && k <= 256 * 8 * 8 * 4
            && rt.native.device_properties(i)?.major >= 9)
    } else {
        Ok(false)
    }
}
fn run(rt: &Runtime, cuda: bool, a: &[Value], kw: &Kwargs) -> Result<Value, Error> {
    let original_x = get(a, kw, 0, "x").unwrap();
    let mode = get(a, kw, 10, "input_act").unwrap_or(&Value::None);
    let weight = get(a, kw, 11, "input_act_weight").unwrap_or(&Value::None);
    let eps = get(a, kw, 12, "input_act_eps").unwrap_or(&Value::Float(0.));
    if !cuda {
        let x = crate::backend_triton_int8::apply_input(rt, original_x, mode, weight, eps)?
            .as_tensor()?;
        let channel = tensor(a, kw, 3, "s_channel")?;
        let k = x
            .size()
            .last()
            .copied()
            .ok_or_else(|| super::w4a8::err("IndexError", "tuple index out of range"))?;
        let (q, scales) = mma::unpack(
            &tensor(a, kw, 1, "qdata")?,
            channel.numel() as i64,
            k,
            Kind::Float8e4m3fn,
        )?;
        let mut args = vec![
            Value::tensor(x),
            Value::tensor(q),
            Value::tensor(scales),
            Value::tensor(channel),
        ];
        for (i, name, default) in [
            (4, "codebook", Value::None),
            (5, "correction", Value::None),
            (6, "bias", Value::None),
            (7, "group_size", Value::Int(16)),
            (8, "convrot_groupsize", Value::Int(256)),
            (9, "out_dtype", Value::DType(Kind::BFloat16)),
        ] {
            args.push(get(a, kw, i, name).unwrap_or(&default).clone());
        }
        return w4a8::run(
            rt,
            false,
            true,
            &args,
            &vec![
                (
                    "residual".into(),
                    get(a, kw, 13, "residual").unwrap_or(&Value::None).clone(),
                ),
                (
                    "residual_scale".into(),
                    get(a, kw, 14, "residual_scale")
                        .unwrap_or(&Value::None)
                        .clone(),
                ),
            ],
        );
    }
    let mode = if matches!(mode,Value::Text(s)if s=="none") {
        &Value::None
    } else {
        mode
    };
    let residual = get(a, kw, 13, "residual").unwrap_or(&Value::None);
    let residual_scale = get(a, kw, 14, "residual_scale").unwrap_or(&Value::None);
    if !matches!(residual, Value::None) && matches!(residual_scale, Value::None) {
        return Err(super::w4a8::err(
            "ValueError",
            "w4a8_int8_linear: residual requires residual_scale",
        ));
    }
    let channel = tensor(a, kw, 3, "s_channel")?;
    let n = channel.numel() as i64;
    let raw = original_x.as_tensor()?;
    let raw_k = raw
        .size()
        .last()
        .copied()
        .ok_or_else(|| super::w4a8::err("IndexError", "tuple index out of range"))?;
    let width = if matches!(mode,Value::Text(s)if s=="swiglu"||s=="situ") {
        2
    } else {
        1
    };
    let k = raw_k / width;
    let fuse = !matches!(mode, Value::None)
        && raw.numel() as i64 <= 8 * raw_k
        && cluster_ok(rt, k, mode, raw.device())?;
    let (x, mode) = if !matches!(mode, Value::None) && !fuse {
        (
            crate::backend_triton_int8::apply_input(rt, original_x, mode, weight, eps)?
                .as_tensor()?,
            &Value::None,
        )
    } else {
        (raw, mode)
    };
    let q = tensor(a, kw, 1, "qdata")?;
    let rel = tensor(a, kw, 2, "s_rel")?;
    let group = integer(get(a, kw, 7, "group_size"), 16)?;
    let rotation = integer(get(a, kw, 8, "convrot_groupsize"), 256)?;
    if q.dim() != 1
        || q.kind() != Kind::Int8
        || n % 16 != 0
        || ![4, 6]
            .iter()
            .any(|bits| q.numel() as i64 == n * k * mma::record_bytes(*bits) / 512)
        || rel.numel() != 0
        || group != 16
        || !matches!(get(a, kw, 5, "correction"), None | Some(Value::None))
    {
        return Err(super::w4a8::err(
            "ValueError",
            "invalid MMA-packed W4A8 operands",
        ));
    }
    let x2 = x
        .f_reshape([-1, *x.size().last().unwrap()])?
        .f_contiguous()?;
    let m = x2.size()[0];
    let residual_fits = matches!(residual, Value::None)
        || (m <= 8
            && residual.as_tensor()?.numel() as i64 == m * n
            && residual_scale.as_tensor()?.numel() as i64 == n);
    if !residual_fits {
        let mut args = vec![
            Value::tensor(x.shallow_clone()),
            Value::tensor(q),
            Value::tensor(rel),
            Value::tensor(channel),
        ];
        for (i, name, default) in [
            (4, "codebook", Value::None),
            (5, "correction", Value::None),
            (6, "bias", Value::None),
            (7, "group_size", Value::Int(16)),
            (8, "convrot_groupsize", Value::Int(256)),
            (9, "out_dtype", Value::DType(Kind::BFloat16)),
            (10, "input_act", mode.clone()),
            (11, "input_act_weight", Value::None),
            (12, "input_act_eps", Value::Float(0.)),
        ] {
            args.push(get(a, kw, i, name).unwrap_or(&default).clone());
        }
        // Source recursion receives the already applied activation and its
        // current mode, not the caller's pre-activation tensor/mode.
        args[10] = mode.clone();
        let out = run(rt, true, &args, &vec![])?;
        return crate::backend_triton_int8::apply_residual(rt, out, residual, residual_scale);
    }
    let output_dtype = get(a, kw, 9, "out_dtype");
    let kind = if matches!(output_dtype, Some(Value::None)) {
        rt.native.default_dtype()?
    } else {
        dtype(output_dtype, Kind::BFloat16)?
    };
    if m > 8 {
        let w = w4a8::Weight {
            q,
            rel,
            channel,
            codebook: optional_tensor(a, kw, 4, "codebook")?,
            correction: None,
            group,
            rotation,
            geometry: (
                n,
                k,
                mma::bit_depth(tensor(a, kw, 1, "qdata")?.numel() as i64, n, k)?,
            ),
        };
        return w4a8::cuda_linear(
            rt,
            &w,
            &x,
            optional_tensor(a, kw, 6, "bias")?.as_ref(),
            kind,
            true,
        );
    }
    let xq = Tensor::f_empty([m, k], (Kind::Int8, x.device()))?;
    let xs = Tensor::f_empty([m, 1], (Kind::Float, x.device()))?;
    let out = Tensor::f_empty([m, n], (kind, x.device()))?;
    let staged = matches!(mode, Value::None) && !cluster_ok(rt, k, &Value::None, x.device())?;
    let rotated = if staged {
        Some(x2.f_empty_like()?)
    } else {
        None
    };
    let partial = if staged {
        if rotation == 0 {
            return Err(super::w4a8::err(
                "ZeroDivisionError",
                "integer division or modulo by zero",
            ));
        }
        Some(Tensor::f_empty(
            [m, k / rotation],
            (Kind::Float, x.device()),
        )?)
    } else {
        None
    };
    let act_weight =
        if matches!(mode,Value::Text(s)if s=="rms_norm") && !matches!(weight, Value::None) {
            Some(
                weight
                    .as_tensor()?
                    .f_to_device_(x.device(), x2.kind(), false, false)?
                    .f_contiguous()?,
            )
        } else {
            None
        };
    let resid = if matches!(residual, Value::None) {
        None
    } else {
        Some(
            residual
                .as_tensor()?
                .f_reshape([m, n])?
                .f_to_kind(kind)?
                .f_contiguous()?,
        )
    };
    let scale = if resid.is_none() {
        None
    } else {
        Some(crate::cuda_linear::vector(
            &rt.native,
            &residual_scale.as_tensor()?,
            x.device(),
            kind,
        )?)
    };
    let bias = optional_tensor(a, kw, 6, "bias")?
        .map(|b| Ok::<_, Error>(b.f_to_kind(Kind::Float)?.f_contiguous()?))
        .transpose()?;
    // The caller resolves DTYPE_TO_CODE before entering _w4a8_mma_linear.
    if matches!(output_dtype, Some(Value::None)) {
        return Err(super::w4a8::err("KeyError", "None"));
    }
    let output_code = quantization::dtype_code(kind)?;
    let scratch = Tensor::f_zeros([m * n + n / 16], (Kind::Int, x.device()))?;
    let workspace = scratch.f_narrow(0, 0, m * n)?.f_view([m, n])?;
    let counters = scratch.f_narrow(0, m * n, n / 16)?;
    let bits = mma::bit_depth(q.numel() as i64, n, k)?;
    let lut = if bits == 4 {
        Some(fixed_decode_lut(q.device())?)
    } else {
        None
    };
    let epsilon = crate::python_numbers::float_value(rt, eps)?;
    let act = match mode {
        Value::None => 0,
        Value::Text(s) if s == "swiglu" => 2,
        Value::Text(s) if s == "rms_norm" => 3,
        _ => {
            return Err(super::w4a8::err(
                "ValueError",
                "unsupported packed input activation",
            ));
        }
    };
    use KernelArg::{Float, Int, Tensor as T};
    let used = unsafe {
        rt.native.call_unchecked(
            "w4a8_codebook_mma_linear",
            &[
                T(&x2),
                rotated.as_ref().map_or(KernelArg::None, T),
                partial.as_ref().map_or(KernelArg::None, T),
                act_weight.as_ref().map_or(KernelArg::None, T),
                Float(epsilon),
                Int(act),
                T(&xq),
                T(&q),
                lut.as_ref().map_or(KernelArg::None, T),
                T(&channel),
                T(&xs),
                bias.as_ref().map_or(KernelArg::None, T),
                resid.as_ref().map_or(KernelArg::None, T),
                scale.as_ref().map_or(KernelArg::None, T),
                T(&workspace),
                T(&counters),
                T(&out),
                Int(rotation),
                Int(group),
                Int(output_code),
                Int(stream(rt, &x)?),
            ],
        )?
    }
    .value
        != 0;
    if !used {
        return Err(fail("MMA-packed W4A8 weight is unsupported on this device"));
    }
    let mut shape = x.size();
    *shape.last_mut().unwrap() = n;
    Ok(Value::tensor(out.f_reshape(shape.as_slice())?))
}
pub(super) fn install(rt: &Runtime) {
    for backend in ["eager", "cuda"] {
        rt.backends.bind_reference(
            backend,
            "w4a8_packed_linear",
            body(move |rt, a, k| run(rt, backend == "cuda", a, k)),
        );
    }
}
