//! Remaining original eager bodies and the shared AdaLN broadcast preparation.
use super::*;
use crate::{int8_utils, packing};

pub(super) fn stochastic_fp8(x: &Tensor, rng: &Tensor, kind: Kind) -> Result<Tensor, Error> {
    let (eb, mb, bias) = match kind {
        Kind::Float8e4m3fn => (4, 3, 7),
        Kind::Float8e5m2 => (5, 2, 15),
        _ => {
            return Err(Error::BindingGap {
                backend: "eager".into(),
                operation: "unsupported stochastic FP8 output dtype".into(),
            });
        }
    };
    let x = x.f_to_kind(Kind::Half)?;
    let mut sign = x.f_sign()?;
    let mut abs = x.f_abs()?;
    sign = sign.f_zeros_like()?.f_where_self(&abs.f_eq(0)?, &sign)?;
    let exponent = abs
        .f_log2()?
        .f_floor()?
        .f_add_scalar(bias)?
        .f_clamp(0, (1 << eb) - 1)?;
    let normal = exponent.f_eq(0)?.f_logical_not()?;
    let power = Tensor::f_pow_scalar(2., &exponent.f_sub_scalar(bias)?)?;
    let normal_mant = abs
        .f_div(&power)?
        .f_sub_scalar(1.)?
        .f_mul_scalar((1 << mb) as f64)?;
    let subnormal = abs.f_div_scalar(2_f64.powi((-bias + 1 - mb) as i32))?;
    let mut scaled = normal_mant.f_where_self(&normal, &subnormal)?;
    let _ = scaled.f_add_(&rng.f_to_kind(scaled.kind())?.f_mul_scalar(1. / 256.)?)?;
    let mantissa = scaled.f_floor()?.f_div_scalar((1 << mb) as f64)?;
    abs.f_copy_(&mantissa)?;
    let factor = power
        .f_mul(&abs.f_add_scalar(1.)?)?
        .f_where_self(&normal, &abs.f_mul_scalar(2_f64.powi((-bias + 1) as i32))?)?;
    let _ = sign.f_mul_(&factor)?;
    let limit = if kind == Kind::Float8e4m3fn {
        448.
    } else {
        57344.
    };
    let _ = sign.f_clamp_(-limit, limit)?;
    Ok(sign.f_to_kind(kind)?)
}
fn value_error(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
fn tuple(shape: &[i64]) -> String {
    let s = shape
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("({s}{})", if shape.len() == 1 { "," } else { "" })
}
fn quant_group(group: i64) -> Result<(), Error> {
    if group == 64 {
        Ok(())
    } else {
        Err(value_error("int4 MMA kernel requires quant_group_size 64"))
    }
}
pub(super) fn quantize_int4_weight(
    rt: &Runtime,
    x: &Tensor,
    group: i64,
    qgroup: i64,
    seed: Option<u64>,
) -> Result<(Tensor, Tensor), Error> {
    quant_group(qgroup)?;
    if x.dim() != 2 {
        return Err(value_error(format!(
            "ConvRot W4A4 expects a 2D tensor, got shape {}",
            tuple(&x.size())
        )));
    }
    let k = x.size()[1];
    if group == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer modulo by zero".into(),
        });
    }
    if k % group != 0 {
        return Err(value_error(format!(
            "in_features {k} not divisible by convrot_groupsize {group}"
        )));
    }
    if k % qgroup != 0 {
        return Err(value_error(format!(
            "in_features {k} not divisible by quant_group_size {qgroup}"
        )));
    }
    let h = int8_utils::eager_w4_hadamard(group, x.device(), x.kind())?;
    let rotated = int8_utils::rotate_weight(x, &h, group)?;
    let absmax = rotated.f_abs()?.f_amax([-1], true)?.f_clamp_min(1e-10)?;
    let scales = absmax.f_div_scalar(7.)?;
    let mut q = rotated.f_div(&scales)?;
    if let Some(seed) = seed {
        let noise = rt
            .native
            .generator(q.device(), seed)?
            .rand(&q.size(), q.kind())?;
        let _ = q.f_add_(&noise)?;
        let _ = q.f_floor_()?;
    } else {
        let _ = q.f_round_()?;
    }
    let q = q.f_clamp_(-7., 7.)?.f_to_kind(Kind::Int8)?;
    Ok((
        packing::pack_int4_row_major(&q)?,
        scales.f_reshape([x.size()[0]])?.f_to_kind(Kind::Float)?,
    ))
}
pub(super) fn dequantize_int4_weight(
    q: &Tensor,
    scales: &Tensor,
    group: i64,
    qgroup: i64,
    kind: Kind,
) -> Result<Tensor, Error> {
    quant_group(qgroup)?;
    let w = packing::unpack_int4_row_major(q, true)?.f_to_kind(Kind::Float)?;
    let w = w.f_mul(
        &scales
            .f_to_device_(q.device(), Kind::Float, false, false)?
            .f_reshape([-1, 1])?,
    )?;
    let h = int8_utils::eager_w4_hadamard(group, q.device(), Kind::Float)?;
    Ok(int8_utils::rotate_weight(&w.f_to_kind(Kind::Float)?, &h, group)?.f_to_kind(kind)?)
}
fn modulation(t: &Tensor, x: &Tensor, n: i64, d: i64) -> Result<(Tensor, i64), Error> {
    let ts = t.size();
    let xs = x.size();
    if ts.is_empty() {
        return Err(Error::ReferenceException {
            class: "IndexError",
            message: "tuple index out of range".into(),
        });
    }
    if ts.last() == Some(&d) && ts.len() <= xs.len() {
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
            let rows = t.numel() as i64 / d;
            if rows > 0 && n % rows == 0 {
                return Ok((t.f_reshape([rows, d])?.f_contiguous()?, n / rows));
            }
        }
    }
    Ok((t.f_expand_as(x)?.f_reshape([n, d])?.f_contiguous()?, 1))
}
pub(super) fn adaln(
    rt: &Runtime,
    x: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f64,
    rms: bool,
    cuda: bool,
) -> Result<Tensor, Error> {
    let shape = x.size();
    if !cuda {
        let normalized_shape = &shape[shape.len().saturating_sub(1)..];
        let normalized = if rms {
            x.f_rms_norm(normalized_shape, None::<&Tensor>, eps)?
        } else {
            x.f_layer_norm(
                normalized_shape,
                None::<&Tensor>,
                None::<&Tensor>,
                eps,
                true,
            )?
        };
        return Ok(normalized.f_mul(&scale.f_add_scalar(1)?)?.f_add(shift)?);
    }
    let d = *shape.last().ok_or_else(|| Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })?;
    if d == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer division or modulo by zero".into(),
        });
    }
    let n = x.numel() as i64 / d;
    let input = x.f_reshape([n, d])?.f_contiguous()?;
    let (scale, sg) = modulation(scale, x, n, d)?;
    let (shift, tg) = modulation(shift, x, n, d)?;
    let out = rt.native.empty_like(&input, input.device())?;
    use KernelArg::{Float, Int, Tensor as T};
    unsafe {
        rt.native.call_unchecked(
            if rms { "rms_adaln" } else { "adaln" },
            &[
                T(&input),
                T(&scale),
                T(&shift),
                T(&out),
                Int(n),
                Int(d),
                Int(sg),
                Int(tg),
                Float(eps),
                Int(quantization::dtype_code(x.kind())?),
                Int(stream(rt, x)?),
            ],
        )?;
    }
    Ok(out.f_reshape(shape.as_slice())?)
}
