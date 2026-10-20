//! comfy.samplers schedulers and k_diffusion schedule primitives.
use crate::native::{Error, Kitchen};
use tch::{Device, Kind, Tensor};

pub const NAMES: [&str; 9] = [
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
];

/// Model parameterization belongs to the model, independently of the schedule.
pub trait ScheduleModel {
    fn sigmas(&self) -> &Tensor;
    fn sigma(&self, timestep: &Tensor) -> Result<Tensor, Error>;
    fn timestep(&self, sigma: &Tensor) -> Result<Tensor, Error>;
    fn sigma_min(&self) -> Result<Tensor, Error> {
        Ok(self.sigmas().f_get(0)?)
    }
    fn sigma_max(&self) -> Result<Tensor, Error> {
        Ok(self.sigmas().f_get(-1)?)
    }
}

impl ScheduleModel for crate::models::qwen_image21::FluxSampling {
    fn sigmas(&self) -> &Tensor {
        &self.sigmas
    }
    fn sigma(&self, t: &Tensor) -> Result<Tensor, Error> {
        self.sigma(t)
    }
    fn timestep(&self, sigma: &Tensor) -> Result<Tensor, Error> {
        Ok(sigma.shallow_clone())
    }
}
impl ScheduleModel for crate::models::moss_image::MossSampling {
    fn sigmas(&self) -> &Tensor {
        &self.sigmas
    }
    fn sigma(&self, t: &Tensor) -> Result<Tensor, Error> {
        self.sigma(t)
    }
    fn timestep(&self, sigma: &Tensor) -> Result<Tensor, Error> {
        self.timestep(sigma)
    }
}

fn exception(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn divide(a: f64, b: f64) -> Result<f64, Error> {
    if b == 0. {
        return Err(exception("ZeroDivisionError", "division by zero"));
    }
    Ok(a / b)
}
pub(crate) fn linspace(
    start: f64,
    end: f64,
    n: i64,
    options: (Kind, Device),
) -> Result<Tensor, Error> {
    // Keep ATen's rejection before tch's temporary Scalar destructor can consume
    // its pending C error and panic. This is the original factory precondition.
    if n < 0 {
        return Err(exception(
            "RuntimeError",
            "number of steps must be non-negative",
        ));
    }
    Ok(Tensor::f_linspace(start, end, n, options)?)
}
pub fn append_zero(x: &Tensor) -> Result<Tensor, Error> {
    Ok(Tensor::f_cat(
        &[x, &x.f_new_zeros([1], (x.kind(), x.device()))?],
        0,
    )?)
}
pub fn simple(model: &dyn ScheduleModel, steps: i64) -> Result<Tensor, Error> {
    let len = model.sigmas().size()[0];
    let stride = divide(len as f64, steps as f64)?;
    let mut sigmas = Vec::new();
    for i in 0..steps {
        sigmas.push(
            model
                .sigmas()
                .f_get(-(1 + (i as f64 * stride) as i64))?
                .double_value(&[]) as f32,
        );
    }
    sigmas.push(0.);
    Ok(Tensor::f_from_slice(&sigmas)?)
}
pub fn ddim(model: &dyn ScheduleModel, mut steps: i64) -> Result<Tensor, Error> {
    let source = model.sigmas();
    let mut values = if source.f_get(1)?.double_value(&[]).abs() <= 1e-5 {
        steps += 1;
        Vec::new()
    } else {
        vec![0.]
    };
    if steps == 0 {
        return Err(exception(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let len = source.size()[0];
    let stride = len.div_euclid(steps).max(1);
    let mut i = 1;
    while i < len {
        values.push(source.f_get(i)?.double_value(&[]) as f32);
        i += stride;
    }
    values.reverse();
    Ok(Tensor::f_from_slice(&values)?)
}
pub fn normal(
    model: &dyn ScheduleModel,
    mut steps: i64,
    sgm: bool,
    kind: Kind,
) -> Result<Tensor, Error> {
    let start = model.timestep(&model.sigma_max()?)?;
    let end = model.timestep(&model.sigma_min()?)?;
    let append = sgm || model.sigma(&end)?.double_value(&[]).abs() > 1e-5;
    if !append {
        steps += 1;
    }
    let time = linspace(
        start.double_value(&[]),
        end.double_value(&[]),
        steps + i64::from(sgm),
        (kind, Device::Cpu),
    )?;
    // Python [:-1] also handles an empty linspace.
    let count = if sgm {
        (time.size()[0] - 1).max(0)
    } else {
        time.size()[0]
    };
    let mut values = Vec::new();
    for i in 0..count {
        values.push(model.sigma(&time.f_get(i)?)?.double_value(&[]) as f32);
    }
    if append {
        values.push(0.);
    }
    Ok(Tensor::f_from_slice(&values)?)
}
pub fn beta(
    model: &dyn ScheduleModel,
    steps: i64,
    alpha: f64,
    beta: f64,
    native: &Kitchen,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    if steps < 0 {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: format!("Number of samples, {steps}, must be non-negative."),
        });
    }
    let total = model.sigmas().size()[0] - 1;
    let step = if steps > 0 { 1. / steps as f64 } else { 0. };
    let mut last = -1;
    let mut values = Vec::new();
    for i in 0..steps {
        let p = 1. - i as f64 * step;
        let q = if !(alpha > 0. && beta > 0.) {
            f64::NAN
        } else if p == 1. {
            1.
        } else {
            let result = native.beta_quantile(p, alpha, beta)?;
            if let Some(message) = result.warning {
                warning(&message);
            }
            result.value
        };
        let t = (q * total as f64).round_ties_even();
        if !t.is_finite() {
            return Err(exception(
                if t.is_nan() {
                    "ValueError"
                } else {
                    "OverflowError"
                },
                if t.is_nan() {
                    "cannot convert float NaN to integer"
                } else {
                    "cannot convert float infinity to integer"
                },
            ));
        }
        let t = t as i64;
        if t != last {
            values.push(model.sigmas().f_get(t)?.double_value(&[]) as f32);
        }
        last = t;
    }
    values.push(0.);
    Ok(Tensor::f_from_slice(&values)?)
}
pub fn linear_quadratic(
    model: &dyn ScheduleModel,
    steps: i64,
    threshold: f64,
    linear_steps: Option<i64>,
) -> Result<Tensor, Error> {
    fn divide(a: f64, b: f64) -> Result<f64, Error> {
        if b == 0. {
            return Err(exception("ZeroDivisionError", "float division by zero"));
        }
        Ok(a / b)
    }
    let values = if steps == 1 {
        vec![1., 0.]
    } else {
        let linear = linear_steps.unwrap_or(steps.div_euclid(2));
        let mut values = Vec::new();
        for i in 0..linear {
            values.push(divide(i as f64 * threshold, linear as f64)?);
        }
        let diff = linear as f64 - threshold * steps as f64;
        let quadratic = steps - linear;
        let q2 = (quadratic as f64).powi(2);
        let qc = divide(diff, linear as f64 * q2)?;
        let lc = divide(threshold, linear as f64)? - divide(2. * diff, q2)?;
        let constant = qc * (linear as f64).powi(2);
        for i in linear..steps {
            values.push(qc * (i as f64).powi(2) + lc * i as f64 + constant);
        }
        values.push(1.);
        values.into_iter().map(|x| (1. - x) as f32).collect()
    };
    Ok(Tensor::f_from_slice(&values)?.f_mul(&model.sigma_max()?.f_to_device(Device::Cpu)?)?)
}
pub fn kl_optimal(n: i64, min: f64, max: f64) -> Result<Tensor, Error> {
    if n < 0 {
        return Err(exception(
            "RuntimeError",
            "upper bound and lower bound inconsistent with step sign",
        ));
    }
    let mut adj = Tensor::f_arange(n, (Kind::Float, Device::Cpu))?;
    let _ = adj.f_div_scalar_(n - 1)?;
    let mut values = adj
        .f_mul_scalar(min.atan())?
        .f_add(&adj.f_rsub_scalar(1)?.f_mul_scalar(max.atan())?)?;
    let _ = values.f_tan_()?;
    append_zero(&values)
}
pub fn karras(
    n: i64,
    min: f64,
    max: f64,
    rho: f64,
    options: (Kind, Device),
) -> Result<Tensor, Error> {
    let inverse = divide(1., rho)?;
    let ramp = linspace(0., 1., n, options)?;
    append_zero(
        &ramp
            .f_mul_scalar(min.powf(inverse) - max.powf(inverse))?
            .f_add_scalar(max.powf(inverse))?
            .f_pow_tensor_scalar(rho)?,
    )
}
fn log(x: f64) -> Result<f64, Error> {
    if x <= 0. {
        return Err(exception("ValueError", "math domain error"));
    }
    Ok(x.ln())
}
pub fn exponential(n: i64, min: f64, max: f64, options: (Kind, Device)) -> Result<Tensor, Error> {
    append_zero(&linspace(log(max)?, log(min)?, n, options)?.f_exp()?)
}
pub fn polyexponential(
    n: i64,
    min: f64,
    max: f64,
    rho: f64,
    options: (Kind, Device),
) -> Result<Tensor, Error> {
    let ramp = linspace(1., 0., n, options)?.f_pow_tensor_scalar(rho)?;
    append_zero(
        &ramp
            .f_mul_scalar(log(max)? - log(min)?)?
            .f_add_scalar(log(min)?)?
            .f_exp()?,
    )
}
pub fn vp(
    n: i64,
    beta_d: f64,
    beta_min: f64,
    eps: f64,
    options: (Kind, Device),
) -> Result<Tensor, Error> {
    let t = linspace(1., eps, n, options)?;
    append_zero(
        &t.f_pow_tensor_scalar(2)?
            .f_mul_scalar(beta_d)?
            .f_div_scalar(2)?
            .f_add(&t.f_mul_scalar(beta_min)?)?
            .f_expm1()?
            .f_sqrt()?,
    )
}
pub fn laplace(
    n: i64,
    min: f64,
    max: f64,
    mu: f64,
    beta: f64,
    options: (Kind, Device),
) -> Result<Tensor, Error> {
    let x = linspace(0., 1., n, options)?;
    let delta = x.f_rsub_scalar(0.5)?;
    let log = delta
        .f_abs()?
        .f_mul_scalar(2)?
        .f_rsub_scalar(1)?
        .f_add_scalar(1e-5)?
        .f_log()?;
    Ok(delta
        .f_sign()?
        .f_mul_scalar(beta)?
        .f_mul(&log)?
        .f_rsub_scalar(mu)?
        .f_exp()?
        .f_clamp(min, max)?)
}
pub fn calculate(
    model: &dyn ScheduleModel,
    name: &str,
    steps: i64,
    native: &Kitchen,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    match name {
        "simple" => simple(model, steps),
        "sgm_uniform" => normal(model, steps, true, native.default_dtype()?),
        "normal" => normal(model, steps, false, native.default_dtype()?),
        "ddim_uniform" => ddim(model, steps),
        "beta" => beta(model, steps, 0.6, 0.6, native, warning),
        "linear_quadratic" => linear_quadratic(model, steps, 0.025, None),
        "karras" | "exponential" | "kl_optimal" => {
            let min = model.sigma_min()?.double_value(&[]);
            let max = model.sigma_max()?.double_value(&[]);
            match name {
                "karras" => karras(steps, min, max, 7., (native.default_dtype()?, Device::Cpu)),
                "exponential" => {
                    exponential(steps, min, max, (native.default_dtype()?, Device::Cpu))
                }
                _ => kl_optimal(steps, min, max),
            }
        }
        _ => {
            let message = format!("error invalid scheduler {name}");
            warning(&message);
            Err(Error::ReferenceException {
                class: "ValueError",
                message,
            })
        }
    }
}
