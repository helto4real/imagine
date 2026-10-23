//! Director two-stage schedule and AV handoff semantics, independent of model ownership.
use super::protocol::sampling::{Sampling, Stage};
use crate::native::Error;
use tch::{Device, Kind, Tensor};
fn invalid() -> Error {
    Error::Native("Invalid H3 stage schedule".into())
}
fn validate(sigmas: &Tensor) -> Result<(), Error> {
    if sigmas.dim() != 1 || sigmas.size()[0] < 2 {
        return Err(invalid());
    }
    let values =
        Vec::<f32>::try_from(sigmas.f_to_device_(Device::Cpu, Kind::Float, false, false)?)?;
    if !super::protocol::sampling::valid_sigmas(&values) {
        return Err(invalid());
    }
    Ok(())
}
pub fn remaining_curve(schedule: &Tensor, boundary: f64, steps: u32) -> Result<Tensor, Error> {
    validate(schedule)?;
    if steps == 0
        || schedule.double_value(&[-1]) != 0.
        || !(0. ..1.).contains(&boundary)
        || boundary == 0.
    {
        return Err(invalid());
    }
    let mut values =
        Vec::<f32>::try_from(schedule.f_to_device_(Device::Cpu, Kind::Float, false, false)?)?;
    if boundary > f64::from(values[0]) {
        values.insert(0, boundary as f32);
    }
    values.dedup();
    let upper = values
        .iter()
        .position(|value| f64::from(*value) <= boundary)
        .ok_or_else(invalid)?;
    let start = if f64::from(values[upper]) == boundary {
        upper as f64
    } else {
        let lower = upper.checked_sub(1).ok_or_else(invalid)?;
        lower as f64
            + (f64::from(values[lower]) - boundary) / f64::from(values[lower] - values[upper])
    };
    let curve = Tensor::f_from_slice(&values)?;
    let positions = Tensor::f_linspace(
        start,
        (values.len() - 1) as f64,
        i64::from(steps) + 1,
        (Kind::Double, Device::Cpu),
    )?;
    let left = positions.f_floor()?.f_to_kind(Kind::Int64)?;
    let right = left
        .f_add_scalar(1)?
        .f_clamp_max((values.len() - 1) as i64)?;
    let weight = positions.f_sub(&left)?.f_to_kind(curve.kind())?;
    let result = curve
        .f_index_select(0, &left)?
        .f_lerp_tensor(&curve.f_index_select(0, &right)?, &weight)?;
    let _ = result.f_get(0)?.f_fill_(boundary)?;
    let _ = result.f_get(-1)?.f_fill_(0.)?;
    validate(&result)?;
    let out = Vec::<f32>::try_from(&result)?;
    if !out.windows(2).all(|pair| pair[1] < pair[0]) {
        return Err(invalid());
    }
    Ok(result)
}
pub fn resolve(
    spec: &Sampling,
    mut generated: impl FnMut(&Stage) -> Result<Tensor, Error>,
) -> Result<Vec<Tensor>, Error> {
    if !spec.valid() {
        return Err(invalid());
    }
    let mut first = if spec.first.sigmas.is_empty() {
        generated(&spec.first)?
    } else {
        Tensor::f_from_slice(&spec.first.sigmas)?
    };
    if spec.second.is_some() && spec.first.sigmas.is_empty() {
        first = first.f_slice(0, 0, i64::from(spec.base_steps_to_run) + 1, 1)?;
    }
    validate(&first)?;
    let mut result = vec![first];
    if let Some(second) = &spec.second {
        let boundary = result[0].double_value(&[-1]);
        if !(0. ..1.).contains(&boundary) || boundary == 0. {
            return Err(invalid());
        }
        let sigmas = if !second.sampling.sigmas.is_empty() {
            Tensor::f_from_slice(&second.sampling.sigmas)?
        } else if second.sampling.steps == 1 {
            Tensor::f_from_slice(&[boundary as f32, 0.])?
        } else {
            remaining_curve(
                &generated(&Stage {
                    denoise: 1.,
                    ..second.sampling.clone()
                })?,
                boundary,
                second.sampling.steps,
            )?
        };
        validate(&sigmas)?;
        if (sigmas.double_value(&[0]) - boundary).abs() > 1e-6
            || sigmas.double_value(&[-1]).abs() > 1e-6
        {
            return Err(invalid());
        }
        result.push(sigmas);
    }
    if result
        .iter()
        .map(|sigmas| sigmas.size()[0] as u32 - 1)
        .sum::<u32>()
        != spec.total_steps()
    {
        return Err(invalid());
    }
    Ok(result)
}
pub fn renoise_video(prediction: &Tensor, noise: &Tensor, boundary: f64) -> Result<Tensor, Error> {
    if prediction.size() != noise.size() || !(0. ..1.).contains(&boundary) || boundary == 0. {
        return Err(invalid());
    }
    // Native zero-interval CONST sampler: noise_scaling followed by inverse_noise_scaling.
    let sigma = Tensor::f_from_slice(&[boundary as f32])?.f_to_device(prediction.device())?;
    let remaining = sigma.f_rsub_scalar(1.)?;
    Ok(sigma
        .f_mul(&noise.f_to_device(prediction.device())?)?
        .f_add(&remaining.f_mul(prediction)?)?
        .f_div(&remaining)?)
}
