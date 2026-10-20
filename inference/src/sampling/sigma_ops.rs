//! Inference operations behind the built-in sigma-editing nodes; no graph/UI.
use super::schedules::{self, ScheduleModel};
use crate::native::{Error, Kitchen};
use tch::{Device, Kind, Tensor};

fn slice(t: &Tensor, start: i64, end: i64) -> Result<Tensor, Error> {
    let len = t.size()[0];
    let index = |n: i64| if n < 0 { (len + n).max(0) } else { n.min(len) };
    let start = index(start);
    let end = index(end);
    Ok(t.f_slice(0, start, end.max(start), 1)?)
}
fn copy(t: &Tensor) -> Result<Tensor, Error> {
    crate::native::clone_tensor(t)
}
pub fn basic(
    native: &Kitchen,
    model: &dyn ScheduleModel,
    name: &str,
    steps: i64,
    denoise: f64,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    let total = if denoise < 1. {
        if denoise <= 0. {
            return Ok(Tensor::f_empty([0], (Kind::Float, Device::Cpu))?);
        }
        (steps as f64 / denoise) as i64
    } else {
        steps
    };
    let sigmas =
        schedules::calculate(model, name, total, native, warning)?.f_to_device(Device::Cpu)?;
    slice(&sigmas, -(steps + 1), sigmas.size()[0])
}
pub fn sd_turbo(model: &dyn ScheduleModel, steps: i64, denoise: f64) -> Result<Tensor, Error> {
    let start = 10 - (10. * denoise) as i64;
    let times = Tensor::f_arange_start(1, 11, (Kind::Int64, Device::Cpu))?
        .f_mul_scalar(100)?
        .f_sub_scalar(1)?
        .f_flip([0])?;
    let times = slice(&times, start, start + steps)?;
    schedules::append_zero(&model.sigma(&times)?)
}
pub fn split(sigmas: &Tensor, step: i64) -> Result<(Tensor, Tensor), Error> {
    Ok((
        slice(sigmas, 0, step + 1)?,
        slice(sigmas, step, sigmas.size()[0])?,
    ))
}
pub fn split_denoise(sigmas: &Tensor, denoise: f64) -> Result<(Tensor, Tensor), Error> {
    let steps = (sigmas.size()[sigmas.dim() - 1] - 1).max(0);
    let total = (steps as f64 * denoise).round_ties_even() as i64;
    // Python [:-0] is empty, whereas [-(0+1):] retains the final element.
    Ok((
        slice(sigmas, 0, -total)?,
        slice(sigmas, -(total + 1), sigmas.size()[0])?,
    ))
}
pub fn flip(sigmas: &Tensor) -> Result<Tensor, Error> {
    if sigmas.size()[0] == 0 {
        return Ok(sigmas.shallow_clone());
    }
    let result = sigmas.f_flip([0])?;
    if result.f_get(0)?.double_value(&[]) == 0. {
        let _ = result.f_get(0)?.f_fill_(0.0001)?;
    }
    Ok(result)
}
pub fn set_first(sigmas: &Tensor, value: f64) -> Result<Tensor, Error> {
    let result = copy(sigmas)?;
    if result.size()[0] == 0 {
        return Err(Error::ReferenceException {
            class: "IndexError",
            message: "index 0 is out of bounds for dimension 0 with size 0".into(),
        });
    }
    let _ = result.f_get(0)?.f_fill_(value)?;
    Ok(result)
}
pub fn extend(
    native: &Kitchen,
    sigmas: &Tensor,
    steps: i64,
    start: f64,
    end: f64,
    spacing: &str,
) -> Result<Tensor, Error> {
    if !["linear", "cosine", "sine"].contains(&spacing) {
        return Err(Error::ReferenceException {
            class: "KeyError",
            message: format!("{spacing:?}").replace('"', "'"),
        });
    }
    if steps + 1 < 0 {
        return Err(Error::ReferenceException {
            class: "RuntimeError",
            message: "number of steps must be non-negative".into(),
        });
    }
    let start = if start < 0. { f64::INFINITY } else { start };
    let ramp = Tensor::f_linspace(
        0.,
        1.,
        steps + 1,
        (native.default_dtype()?, sigmas.device()),
    )?;
    let ramp = slice(&ramp, 1, -1)?;
    let spacing = match spacing {
        "cosine" => ramp
            .f_mul_scalar(std::f64::consts::PI)?
            .f_div_scalar(2)?
            .f_sin()?,
        "sine" => ramp
            .f_mul_scalar(std::f64::consts::PI)?
            .f_div_scalar(2)?
            .f_cos()?
            .f_rsub_scalar(1)?,
        _ => ramp,
    };
    let mut values = Vec::new();
    for i in 0..sigmas.size()[0] - 1 {
        let current = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let value = current.double_value(&[]);
        values.push(value as f32);
        if end <= value && value <= start {
            let intermediate = spacing.f_mul(&next.f_sub(&current)?)?.f_add(&current)?;
            for j in 0..intermediate.size()[0] {
                values.push(intermediate.f_get(j)?.double_value(&[]) as f32);
            }
        }
    }
    if sigmas.size()[0] > 0 {
        values.push(sigmas.f_get(-1)?.double_value(&[]) as f32);
    }
    Ok(Tensor::f_from_slice(&values)?)
}
fn digit(c: char) -> Option<u32> {
    crate::python_numbers::decimal(c).map(u32::from)
}
/// Literal source pattern [-+]?(?:\d*\.*\d+), including its unusual handling
/// of exponent text and malformed multi-dot tokens. Decimal data is pinned.
pub fn manual(text: &str) -> Result<Tensor, Error> {
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    let mut values = Vec::new();
    while index < chars.len() {
        let start = index;
        let mut first = start;
        if matches!(chars[first], '+' | '-') {
            first += 1;
        }
        let mut digits = first;
        while digits < chars.len() && digit(chars[digits]).is_some() {
            digits += 1;
        }
        let mut dots = digits;
        while dots < chars.len() && chars[dots] == '.' {
            dots += 1;
        }
        let mut last = dots;
        while last < chars.len() && digit(chars[last]).is_some() {
            last += 1;
        }
        let end = if last > dots {
            last
        } else if digits > first {
            digits
        } else {
            index += 1;
            continue;
        };
        let token: String = chars[start..end].iter().collect();
        let ascii: String = token
            .chars()
            .map(|c| digit(c).map_or(c, |n| char::from_digit(n, 10).unwrap()))
            .collect();
        let value = ascii
            .parse::<f64>()
            .map_err(|_| Error::ReferenceException {
                class: "ValueError",
                message: format!("could not convert string to float: '{token}'"),
            })?;
        values.push(value as f32);
        index = end;
    }
    Ok(Tensor::f_from_slice(&values)?)
}
