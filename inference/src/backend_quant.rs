//! Remaining original eager FP4/SVDQuant bodies and scaled-mm-v2 frontend.
use super::*;
use crate::{block_scales, packing};

fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn tuple(s: &[i64]) -> String {
    format!(
        "({}{})",
        s.iter().map(i64::to_string).collect::<Vec<_>>().join(", "),
        if s.len() == 1 { "," } else { "" }
    )
}
fn dimensions<const N: usize>(s: &[i64]) -> Result<[i64; N], Error> {
    s.try_into().map_err(|_| {
        error(
            "ValueError",
            if s.len() < N {
                format!(
                    "not enough values to unpack (expected {N}, got {})",
                    s.len()
                )
            } else {
                format!("too many values to unpack (expected {N})")
            },
        )
    })
}
fn floatx_e2m1(x: &Tensor) -> Result<Tensor, Error> {
    assertion(x.kind() == Kind::Float, "")?;
    let raw = x.f_view_dtype(Kind::Int)?;
    let sign = raw.f_bitwise_and(0x80000000_i64)?;
    let x = raw.f_bitwise_xor_tensor(&sign)?.f_view_dtype(Kind::Float)?;
    let saturate = x.f_ge(6.)?;
    let denormal = saturate.f_logical_not()?.f_logical_and(&x.f_lt(1.)?)?;
    let normal = saturate.f_logical_or(&denormal)?.f_logical_not()?;
    let denorm = x
        .f_add_scalar(4194304.)?
        .f_view_dtype(Kind::Int)?
        .f_sub_scalar(149_i64 << 23)?
        .f_to_kind(Kind::Uint8)?;
    let mut norm = x.f_view_dtype(Kind::Int)?;
    let odd = norm
        .f_bitwise_right_shift_tensor_scalar(22)?
        .f_bitwise_and(1)?;
    let _ = norm.f_add_scalar_((-126_i64 << 23) + ((1_i64 << 21) - 1))?;
    let _ = norm.f_add_(&odd)?;
    let norm = norm
        .f_bitwise_right_shift_tensor_scalar(22)?
        .f_to_kind(Kind::Uint8)?;
    let all = x.f_full_like(7.)?.f_to_kind(Kind::Uint8)?;
    let codes = denorm.f_where_self(&denormal, &all)?;
    let codes = norm.f_where_self(&normal, &codes)?;
    let sign = sign
        .f_bitwise_right_shift_tensor_scalar(28)?
        .f_to_kind(Kind::Uint8)?
        .f_bitwise_and(8)?;
    Ok(codes.f_bitwise_or_tensor(&sign)?.f_to_kind(Kind::Uint8)?)
}
fn nvfp4(
    x: &Tensor,
    scale: &Tensor,
    pad: bool,
    hi: impl FnOnce() -> Result<bool, Error>,
) -> Result<(Tensor, Tensor), Error> {
    let mut shape = x.size();
    let mut x = x.shallow_clone();
    if pad {
        let [r, c] = dimensions::<2>(&shape)?;
        let (pr, pc) = (((r + 15) / 16) * 16, ((c + 15) / 16) * 16);
        if (pr, pc) != (r, c) {
            x = x.f_pad([0, pc - c, 0, pr - r], "constant", None)?;
            shape = x.size();
        }
    }
    let rows = *shape
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let blocked = x.f_reshape([rows, -1, 16])?;
    let absmax = blocked.f_abs()?.f_amax([-1], false)?;
    let scales = absmax
        .f_to_kind(Kind::Float)?
        .f_div_scalar(6.)?
        .f_div(scale)?
        .f_clamp_max(448.)?;
    let rounded = scales
        .f_to_kind(Kind::Float8e4m3fn)?
        .f_to_kind(Kind::Float)?;
    let total = scale.f_mul(&rounded)?;
    let zeros = total.f_eq(0)?;
    let safe = total.f_ones_like()?.f_where_self(&zeros, &total)?;
    let data = blocked
        .f_to_kind(Kind::Float)?
        .f_div(&safe.f_unsqueeze(-1)?)?;
    let data = data
        .f_zeros_like()?
        .f_where_self(&zeros.f_unsqueeze(-1)?, &data)?
        .f_clamp(-6., 6.)?
        .f_view_(shape.as_slice())?;
    let codes = floatx_e2m1(&data)?;
    let width = *shape.last().unwrap();
    assertion(width % 2 == 0, "")?;
    let flat = codes.f_contiguous()?.f_view([-1])?;
    let even = flat.f_slice(0, 0, flat.size()[0], 2)?;
    let odd = flat.f_slice(0, 1, flat.size()[0], 2)?;
    let packed = if hi()? {
        even.f_bitwise_left_shift_tensor_scalar(4)?
            .f_bitwise_or_tensor(&odd)?
    } else {
        odd.f_bitwise_left_shift_tensor_scalar(4)?
            .f_bitwise_or_tensor(&even)?
    };
    *shape.last_mut().unwrap() /= 2;
    Ok((
        packed.f_view_(shape.as_slice())?,
        block_scales::to_blocked(&scales.f_to_kind(Kind::Float8e4m3fn)?, false)?,
    ))
}
fn quantize_svd(a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let x = tensor(a, k, 0, "x")?;
    let smooth = tensor(a, k, 1, "smooth")?;
    let down = tensor(a, k, 2, "lora_down")?;
    if x.dim() != 2 {
        return Err(error(
            "ValueError",
            format!("expected 2D input, got shape {}", tuple(&x.size())),
        ));
    }
    let [m, width] = dimensions::<2>(&x.size())?;
    if down.dim() < 2 {
        return Err(error("IndexError", "tuple index out of range"));
    }
    if width % 64 != 0 {
        return Err(error(
            "ValueError",
            format!("K={width} not divisible by group_size=64"),
        ));
    }
    let pad = integer(get(a, k, 3, "pad_size"), 256)?;
    if pad == 0 {
        return Err(error(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let numerator = -m;
    let floor = numerator.div_euclid(pad) - i64::from(pad < 0 && numerator % pad != 0);
    let mp = -floor * pad;
    let unsigned = boolean(get(a, k, 4, "act_unsigned"), false)?;
    let src = optional_tensor(a, k, 5, "lora_x")?.unwrap_or_else(|| x.shallow_clone());
    let mut lora = src
        .f_to_kind(Kind::Float)?
        .f_matmul(&down.f_to_kind(Kind::Float)?)?;
    let groups = x.f_div(&smooth)?.f_view([m, width / 64, 64])?;
    let scales = groups
        .f_abs()?
        .f_amax([-1], false)?
        .f_clamp_min(1e-10)?
        .f_div_scalar(if unsigned { 15. } else { 7. })?;
    let values = groups
        .f_div(&scales.f_unsqueeze(-1)?)?
        .f_round()?
        .f_clamp(
            if unsigned { 0. } else { -7. },
            if unsigned { 15. } else { 7. },
        )?
        .f_to_kind(Kind::Int8)?
        .f_reshape([m, width])?;
    let mut q = packing::pack_int4_row_major(&values)?;
    let mut scales = scales;
    if mp > m {
        q = q.f_pad([0, 0, 0, mp - m], "constant", None)?;
        scales = scales.f_pad([0, 0, 0, mp - m], "constant", None)?;
        lora = lora.f_pad([0, 0, 0, mp - m], "constant", None)?;
    }
    Ok(Value::Tuple(vec![
        Value::tensor(q),
        Value::tensor(scales.f_tr()?.f_contiguous()?.f_to_kind(x.kind())?),
        Value::tensor(lora),
    ]))
}
fn scaled_svd(a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let act = tensor(a, k, 0, "act")?;
    let mut w = tensor(a, k, 1, "wgt")?;
    let asc = tensor(a, k, 2, "ascales")?;
    let mut wsc = tensor(a, k, 3, "wscales")?;
    let la = tensor(a, k, 4, "lora_act_in")?;
    let mut up = tensor(a, k, 5, "lora_up")?;
    if w.dim() == 4 {
        let [nt, kg, nq, by] = dimensions::<4>(&w.size())?;
        if nq != 32 {
            return Err(error(
                "ValueError",
                format!("unexpected tile-packed N quads: {nq}"),
            ));
        }
        if by != 128 {
            return Err(error(
                "ValueError",
                format!("unexpected tile-packed byte axis: {by}"),
            ));
        }
        w = w
            .f_view_([nt, kg, nq, 4, 32].as_slice())?
            .f_permute([0, 2, 3, 1, 4])?
            .f_contiguous()?
            .f_view([nt * 128, kg * 32])?;
    }
    if wsc.dim() == 3 {
        let s = wsc.size();
        wsc = wsc
            .f_permute([1, 0, 2])?
            .f_contiguous()?
            .f_view([s[1], s[0] * 128])?;
    }
    if up.dim() == 3 {
        let s = up.size();
        up = up
            .f_permute([0, 2, 1])?
            .f_contiguous()?
            .f_view([s[0] * 128, s[1]])?;
    }
    let [m, kh] = dimensions::<2>(&act.size())?;
    let n = *w
        .size()
        .first()
        .ok_or_else(|| error("IndexError", "tuple index out of range"))?;
    let width = kh * 2;
    let kind = wsc.kind();
    let wf = packing::unpack_int4_row_major(&w, true)?
        .f_to_kind(kind)?
        .f_view([n, width / 64, 64])?
        .f_mul(&wsc.f_tr()?.f_unsqueeze(-1)?)?
        .f_view([n, width])?;
    let unsigned = boolean(get(a, k, 7, "act_unsigned"), false)?;
    let xf = packing::unpack_int4_row_major(&act, !unsigned)?
        .f_to_kind(kind)?
        .f_view([m, width / 64, 64])?
        .f_mul(&asc.f_tr()?.f_unsqueeze(-1)?)?
        .f_view([m, width])?;
    let out = xf.f_matmul(&wf.f_tr()?)?;
    let lora = la
        .f_to_kind(Kind::Float)?
        .f_matmul(&up.f_to_kind(Kind::Float)?.f_tr()?)?;
    let mut out = out.f_add(&lora.f_to_kind(out.kind())?)?;
    if let Some(b) = optional_tensor(a, k, 6, "bias")? {
        out = out.f_add(&b)?;
    }
    Ok(Value::tensor(out))
}
fn list_tensors(xs: &[Tensor]) -> Value {
    Value::List(
        xs.iter()
            .map(|x| Value::tensor(x.shallow_clone()))
            .collect(),
    )
}
fn ints(xs: &[i64]) -> Value {
    Value::List(xs.iter().copied().map(Value::Int).collect())
}
fn scaled_mm(rt: &Runtime, fp4: bool, a: &[Value], k: &Kwargs) -> Result<Value, Error> {
    let x = tensor(a, k, 0, "a")?;
    let w = tensor(a, k, 1, "b")?;
    let (x, w) = if fp4 {
        (
            rt.native
                .view_shell_dtype(&x, crate::native::ShellDType::PackedE2M1)?,
            rt.native
                .view_shell_dtype(&w, crate::native::ShellDType::PackedE2M1)?,
        )
    } else {
        (x, w)
    };
    let w = w.f_tr()?;
    let (sc_a, sc_b, recipe, swizzle, bias_i, dtype_i) = if fp4 {
        let gsa = tensor(a, k, 2, "tensor_scale_a")?;
        let gsb = tensor(a, k, 3, "tensor_scale_b")?;
        let sa = tensor(a, k, 4, "block_scale_a")?.f_view([-1])?;
        let sb = tensor(a, k, 5, "block_scale_b")?.f_view([-1])?;
        (vec![sa, gsa], vec![sb, gsb], vec![2, 0], vec![1, 0], 6, 7)
    } else {
        (
            vec![tensor(a, k, 2, "block_scale_a")?],
            vec![tensor(a, k, 3, "block_scale_b")?],
            vec![3],
            vec![1],
            4,
            5,
        )
    };
    let bias = get(a, k, bias_i, "bias").cloned().unwrap_or(Value::None);
    let dtype = get(a, k, dtype_i, "out_dtype")
        .cloned()
        .unwrap_or(Value::None);
    if !matches!(dtype, Value::None | Value::DType(_)) {
        let name = match dtype {
            Value::Tensor(_) => "Tensor",
            Value::Quantized(_) => "QuantizedTensor",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Text(_) => "str",
            Value::Device(_) => "torch.device",
            Value::Tuple(_) => "tuple",
            Value::List(_) => "list",
            Value::Dict(_) => "dict",
            _ => unreachable!(),
        };
        return Err(error(
            "TypeError",
            format!(
                "_scaled_mm_v2(): argument 'out_dtype' (position 10) must be torch.dtype, not {name}"
            ),
        ));
    }
    // Original functional.scaled_mm expands the frozen enum recipes to lists,
    // passes None through, contracts default dimensions, and disables fast accumulation.
    unsafe {
        crate::boxed_aten::call(
            &rt.native,
            "aten::_scaled_mm_v2",
            "",
            &[
                Value::tensor(x),
                Value::tensor(w),
                list_tensors(&sc_a),
                ints(&recipe),
                ints(&swizzle),
                list_tensors(&sc_b),
                ints(&recipe),
                ints(&swizzle),
                bias,
                dtype,
                ints(&[]),
                Value::Bool(false),
            ],
            &vec![],
        )
    }
}
pub(super) fn install(rt: &Runtime) {
    for op in [
        "quantize_nvfp4",
        "quantize_svdquant_w4a4",
        "scaled_mm_svdquant_w4a4",
        "scaled_mm_mxfp8",
        "scaled_mm_nvfp4",
    ] {
        rt.backends.bind_reference(
            "eager",
            op,
            body(move |rt, a, k| match op {
                "quantize_nvfp4" => {
                    let x = tensor(a, k, 0, "x")?;
                    let s = tensor(a, k, 1, "per_tensor_scale")?;
                    Ok(pair(nvfp4(
                        &x,
                        &s,
                        boolean(get(a, k, 3, "pad_16x"), false)?,
                        || {
                            crate::backend_triton_mxfp8::truthy(
                                rt,
                                get(a, k, 4, "hi_first").unwrap_or(&Value::Bool(true)),
                            )
                        },
                    )?))
                }
                "quantize_svdquant_w4a4" => quantize_svd(a, k),
                "scaled_mm_svdquant_w4a4" => scaled_svd(a, k),
                "scaled_mm_mxfp8" | "scaled_mm_nvfp4" => {
                    scaled_mm(rt, op == "scaled_mm_nvfp4", a, k)
                }
                _ => unreachable!(),
            }),
        );
    }
}
