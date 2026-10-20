//! Built-in inference guidance callbacks, separate from graph/UI construction.
use super::schedules::ScheduleModel;
use crate::{
    execution_value::Value as V,
    native::{Error, Kitchen},
};
use tch::{Device, Kind, Tensor};

pub fn optimized_scale(positive: &Tensor, negative: &Tensor) -> Result<Tensor, Error> {
    let positive_flat = positive.f_reshape([positive.size()[0], -1])?;
    let negative_flat = negative.f_reshape([negative.size()[0], -1])?;
    let dot = positive_flat
        .f_mul(&negative_flat)?
        .f_sum_dim_intlist([1].as_slice(), true, None)?;
    let norm = negative_flat
        .f_pow_tensor_scalar(2)?
        .f_sum_dim_intlist([1].as_slice(), true, None)?
        .f_add_scalar(1e-8)?;
    let mut shape = vec![1; positive.dim()];
    shape[0] = positive.size()[0];
    Ok(dot.f_div(&norm)?.f_reshape(&shape)?)
}
pub fn cfg_zero_star(
    x: &Tensor,
    cond: &Tensor,
    uncond: &Tensor,
    out: &Tensor,
    scale: f64,
) -> Result<Tensor, Error> {
    let alpha = optimized_scale(&x.f_sub(cond)?, &x.f_sub(uncond)?)?;
    Ok(out.f_add(&uncond.f_mul(&alpha.f_sub_scalar(1.)?)?)?.f_add(
        &uncond
            .f_mul_scalar(scale)?
            .f_mul(&alpha.f_rsub_scalar(1.)?)?,
    )?)
}
pub fn cfg_norm_post(cond: &Tensor, out: &Tensor, strength: f64) -> Result<Tensor, Error> {
    let a = cond.f_norm_scalaropt_dim(2., [1], true)?;
    let b = out.f_norm_scalaropt_dim(2., [1], true)?;
    let scale = a.f_div(&b.f_add_scalar(1e-8)?)?.f_clamp(0., 1.)?;
    Ok(out.f_mul(&scale)?.f_mul_scalar(strength)?)
}
pub fn cfg_norm_pre(
    cond: &Tensor,
    uncond: &Tensor,
    scale: f64,
    strength: f64,
) -> Result<Tensor, Error> {
    let combined = uncond.f_add(&cond.f_sub(uncond)?.f_mul_scalar(scale)?)?;
    let cond_norm = cond.f_norm_scalaropt_dim(2., [1], true)?;
    let combined_norm = combined.f_norm_scalaropt_dim(2., [1], true)?;
    let rescale = cond_norm
        .f_div(&combined_norm.f_clamp_min(1e-12)?)?
        .f_where_self(&combined_norm.f_gt(0)?, &combined_norm.f_ones_like()?)?;
    let result = combined.f_mul(&rescale)?;
    if strength != 1. {
        Ok(result
            .f_mul_scalar(strength)?
            .f_add(&combined.f_mul_scalar(1. - strength)?)?)
    } else {
        Ok(result)
    }
}
pub fn perp_neg(
    positive: &Tensor,
    negative: &Tensor,
    empty: &Tensor,
    negative_scale: f64,
    scale: f64,
) -> Result<Tensor, Error> {
    let pos = positive.f_sub(empty)?;
    let neg = negative.f_sub(empty)?;
    let projection = neg
        .f_mul(&pos)?
        .f_sum(None)?
        .f_div(&pos.f_norm()?.f_pow_tensor_scalar(2)?)?;
    let perpendicular = neg.f_sub(&projection.f_mul(&pos)?)?;
    Ok(empty.f_add(
        &pos.f_sub(&perpendicular.f_mul_scalar(negative_scale)?)?
            .f_mul_scalar(scale)?,
    )?)
}
fn vh(native: &Kitchen, input: &Tensor) -> Result<Tensor, Error> {
    // driver=None must remain absent; passing even an empty driver string
    // changes CPU behavior in ATen's optional driver contract.
    let out = unsafe {
        crate::boxed_aten::call(
            native,
            "aten::linalg_svd",
            "",
            &[V::tensor(input.shallow_clone())],
            &vec![("full_matrices".into(), V::Bool(false))],
        )
    }?;
    let V::Tuple(values) = out else {
        return Err(Error::Native("ATen SVD tuple contract".into()));
    };
    values[2].as_tensor()
}
fn runtime_error(error: &Error) -> bool {
    matches!(
        error.root(),
        Error::Native(_)
            | Error::Torch(tch::TchError::Torch(_))
            | Error::AcceleratorException { .. }
            | Error::ReferenceException {
                class: "RuntimeError" | "_LinAlgError" | "OutOfMemoryError",
                ..
            }
    )
}
pub fn tangential_damping(
    native: &Kitchen,
    cond: &Tensor,
    uncond: &Tensor,
) -> Result<Tensor, Error> {
    tangential_damping_with(cond, uncond, &mut |input| vh(native, input))
}
/// The injected operation is the same SVD call; it also permits bounded tests
/// of the source RuntimeError-only CPU fallback without exhausting memory.
pub fn tangential_damping_with(
    cond: &Tensor,
    uncond: &Tensor,
    svd: &mut dyn FnMut(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let batch = cond.size()[0];
    let cond_flat = cond.f_reshape([batch, 1, -1])?.f_to_kind(Kind::Float)?;
    let uncond_flat = uncond.f_reshape([batch, 1, -1])?.f_to_kind(Kind::Float)?;
    let matrix = Tensor::f_cat(&[&uncond_flat, &cond_flat], 1)?;
    let vectors = match svd(&matrix) {
        Ok(v) => v,
        Err(e) if runtime_error(&e) => svd(&matrix.f_to(Device::Cpu)?)?,
        Err(e) => return Err(e),
    };
    let v1 = vectors
        .f_slice(0, 0, batch, 1)?
        .f_slice(1, 0, 1, 1)?
        .f_to(uncond_flat.device())?;
    Ok(uncond_flat
        .f_matmul(&v1.f_transpose(-2, -1)?)?
        .f_mul(&v1)?
        .f_reshape_as(uncond)?
        .f_to_kind(uncond.kind())?)
}
pub fn tcfg(
    native: &Kitchen,
    x: &Tensor,
    predictions: Vec<V>,
    conditions_present: [bool; 2],
) -> Result<Vec<V>, Error> {
    if predictions.len() <= 1 || !conditions_present[0] || !conditions_present[1] {
        return Ok(predictions);
    }
    let cond = predictions[0].as_tensor()?;
    let uncond = predictions[1].as_tensor()?;
    let transformed = tangential_damping(native, &x.f_sub(&cond)?, &x.f_sub(&uncond)?)?;
    let mut result = vec![predictions[0].clone(), V::tensor(x.f_sub(&transformed)?)];
    result.extend(predictions[2..].iter().cloned());
    Ok(result)
}
pub fn differential_mask(
    model: &dyn ScheduleModel,
    sigma: &Tensor,
    mask: &Tensor,
    sigmas: &Tensor,
    strength: f64,
) -> Result<Tensor, Error> {
    let mut to = model.sigma_min()?;
    let last = sigmas.f_get(-1)?;
    if last.f_gt_tensor(&to)?.f_is_nonzero()? {
        to = last;
    }
    let from = model.timestep(&sigmas.f_get(0)?)?;
    let to = model.timestep(&to)?;
    let now = model.timestep(&sigma.f_get(0)?)?;
    let threshold = now.f_sub(&to)?.f_div(&from.f_sub(&to)?)?;
    let binary = mask.f_ge_tensor(&threshold)?.f_to_kind(mask.kind())?;
    if strength != 0. && strength < 1. {
        Ok(binary
            .f_mul_scalar(strength)?
            .f_add(&mask.f_mul_scalar(1. - strength)?)?)
    } else {
        Ok(binary)
    }
}
