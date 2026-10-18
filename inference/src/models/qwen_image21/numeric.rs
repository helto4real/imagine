use crate::{execution::Runtime, execution_value::Value, native::Error};
use tch::{Kind, Tensor};
#[path = "latent_constants.rs"]
mod constants;

pub fn zero_centered_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor, Error> {
    let w = weight
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_add_scalar(1.0)?;
    Ok(x.f_to_kind(Kind::Float)?
        .f_rms_norm(w.size().as_slice(), Some(&w), Some(eps))?
        .f_to_kind(x.kind())?)
}

pub fn round_timesteps(t: &Tensor, kind: Kind) -> Result<Tensor, Error> {
    Ok(t.f_mul_scalar(1000_i64)?
        .f_to_kind(kind)?
        .f_div_scalar(1000_i64)?
        .f_to_kind(kind)?)
}

pub fn timestep_embedding(
    t: &Tensor,
    dim: i64,
    max_period: f64,
    time_factor: f64,
) -> Result<Tensor, Error> {
    let t = t.f_mul_scalar(time_factor)?;
    let half = dim / 2;
    let freq = Tensor::f_arange(half, (Kind::Float, t.device()))?
        .f_mul_scalar(-max_period.ln())?
        .f_div_scalar(half)?
        .f_exp()?;
    let args = t
        .f_unsqueeze(1)?
        .f_to_kind(Kind::Float)?
        .f_mul(&freq.f_unsqueeze(0)?)?;
    let mut out = Tensor::f_cat(&[args.f_cos()?, args.f_sin()?], -1)?;
    if dim % 2 != 0 {
        out = Tensor::f_cat(
            &[
                out.shallow_clone(),
                out.f_slice(1, 0, 1, 1)?.f_zeros_like()?,
            ],
            -1,
        )?;
    }
    if t.f_is_floating_point()? {
        out = out.f_to_kind(t.kind())?;
    }
    Ok(out)
}

pub fn split_rows(p: &Tensor) -> Result<(Tensor, Tensor), Error> {
    Ok((
        p.f_slice(0, -1, i64::MAX, 1)?.f_unsqueeze(1)?,
        p.f_slice(0, 0, -1, 1)?.f_unsqueeze(1)?,
    ))
}

/// # Safety
/// Inputs must satisfy the frozen Kitchen AdaLN dtype/layout/device contract.
pub unsafe fn modulated_norm(
    rt: &Runtime,
    x: &Tensor,
    scale: (&Tensor, &Tensor),
    prefix: i64,
    zero: &Tensor,
    eps: f64,
    training: bool,
) -> Result<Tensor, Error> {
    if training {
        let norm = x.f_layer_norm(
            [*x.size().last().unwrap()].as_slice(),
            None::<&Tensor>,
            None::<&Tensor>,
            eps,
            true,
        )?;
        return Ok(Tensor::f_cat(
            &[
                norm.f_slice(1, 0, prefix, 1)?
                    .f_mul(&scale.0.f_add_scalar(1)?)?,
                norm.f_slice(1, prefix, i64::MAX, 1)?
                    .f_mul(&scale.1.f_add_scalar(1)?)?,
            ],
            1,
        )?);
    }
    let apply = |input: &Tensor, s: &Tensor| unsafe {
        rt.call_kitchen(
            "adaln",
            &vec![
                ("x".into(), Value::tensor(input.shallow_clone())),
                ("scale".into(), Value::tensor(s.shallow_clone())),
                ("shift".into(), Value::tensor(zero.shallow_clone())),
                ("eps".into(), Value::Float(eps)),
            ],
        )?
        .as_tensor()
    };
    let out = apply(x, scale.1)?;
    if prefix != 0 {
        out.f_slice(1, 0, prefix, 1)?
            .f_copy_(&apply(&x.f_slice(1, 0, prefix, 1)?, scale.0)?)?;
    }
    Ok(out)
}

pub fn gated_residual(
    x: &Tensor,
    y: &Tensor,
    gate: (&Tensor, &Tensor),
    prefix: i64,
) -> Result<Tensor, Error> {
    let _ = x
        .f_slice(1, prefix, i64::MAX, 1)?
        .f_addcmul_(&y.f_slice(1, prefix, i64::MAX, 1)?, gate.1)?;
    if prefix != 0 {
        let _ = x
            .f_slice(1, 0, prefix, 1)?
            .f_addcmul_(&y.f_slice(1, 0, prefix, 1)?, gate.0)?;
    }
    Ok(x.shallow_clone())
}

pub fn prefix_cache_key(
    x: &Tensor,
    context: &Tensor,
    refs: &[Tensor],
    slots: &[i64],
) -> Result<Tensor, Error> {
    let shape = x.size();
    let mut header = vec![
        context.size()[1],
        refs.len() as i64,
        shape[shape.len() - 2],
        shape[shape.len() - 1],
    ];
    header.extend_from_slice(slots);
    for r in refs {
        let s = r.size();
        header.extend_from_slice(&s[s.len() - 2..]);
    }
    let header = Tensor::f_from_slice(&header)?
        .f_to_device_(context.device(), Kind::Float, false, false)?
        .f_expand([context.size()[0], -1], false)?;
    let mut values = vec![header, context.f_to_kind(Kind::Float)?.f_flatten(1, -1)?];
    for r in refs {
        values.push(r.f_to_kind(Kind::Float)?.f_flatten(1, -1)?);
    }
    Ok(Tensor::f_cat(&values, 1)?)
}

pub struct LatentNormalization {
    mean: Tensor,
    std: Tensor,
}
impl LatentNormalization {
    pub fn new(rt: &Runtime) -> Result<Self, Error> {
        let kind = rt.native.default_dtype()?;
        Ok(Self {
            mean: Tensor::f_from_slice(&constants::LATENTS_MEAN)?
                .f_to_kind(kind)?
                .f_view([1, 64, 1, 1])?,
            std: Tensor::f_from_slice(&constants::LATENTS_STD)?
                .f_to_kind(kind)?
                .f_view([1, 64, 1, 1])?,
        })
    }
    pub fn process_in(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(
            x.f_sub(&self.mean.f_to_device_(x.device(), x.kind(), false, false)?)?
                .f_div(&self.std.f_to_device_(x.device(), x.kind(), false, false)?)?,
        )
    }
    pub fn process_out(&self, x: &Tensor) -> Result<Tensor, Error> {
        Ok(
            x.f_mul(&self.std.f_to_device_(x.device(), x.kind(), false, false)?)?
                .f_add(&self.mean.f_to_device_(x.device(), x.kind(), false, false)?)?,
        )
    }
}
