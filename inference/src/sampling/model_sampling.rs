//! Shared model-owned noise schedules, ported from the locked model_sampling.py.
//! Tensor operation order (including in-place zero-SNR rescaling) is observable.
use super::schedules::ScheduleModel;
use crate::native::{Error, Kitchen};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Flavor {
    Discrete,
    DiscreteEdm,
    Distilled,
    ContinuousEdm,
    ContinuousV,
    DiscreteFlow,
    AudioVideo,
    Cascade,
    Flux,
    Cosmos,
}
#[derive(Clone, Debug)]
pub struct Config {
    pub beta_schedule: String,
    pub linear_start: f64,
    pub linear_end: f64,
    pub timesteps: i64,
    pub zsnr: bool,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub sigma_data: f64,
    pub shift: f64,
    pub multiplier: f64,
    pub noise_scale: f64,
    pub audio_shift: Option<f64>,
    pub cosine_s: f64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            beta_schedule: "linear".into(),
            linear_start: 0.00085,
            linear_end: 0.012,
            timesteps: 1000,
            zsnr: false,
            sigma_min: 0.002,
            sigma_max: 120.,
            sigma_data: 1.,
            shift: 1.,
            multiplier: 1000.,
            noise_scale: 1.,
            audio_shift: None,
            cosine_s: 0.008,
        }
    }
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn linspace(a: f64, b: f64, n: i64, kind: Kind) -> Result<Tensor, Error> {
    if n < 0 {
        return Err(error(
            "RuntimeError",
            "number of steps must be non-negative",
        ));
    }
    Ok(Tensor::f_linspace(a, b, n, (kind, Device::Cpu))?)
}
pub fn beta_schedule(name: &str, n: i64, start: f64, end: f64, s: f64) -> Result<Tensor, Error> {
    Ok(match name {
        "linear" => {
            linspace(start.powf(0.5), end.powf(0.5), n, Kind::Double)?.f_pow_tensor_scalar(2)?
        }
        "sqrt_linear" => linspace(start, end, n, Kind::Double)?,
        "sqrt" => linspace(start, end, n, Kind::Double)?.f_pow_tensor_scalar(0.5)?,
        "cosine" => {
            let time = Tensor::f_arange(n + 1, (Kind::Double, Device::Cpu))?
                .f_div_scalar(n)?
                .f_add_scalar(s)?;
            let alpha = time
                .f_div_scalar(1. + s)?
                .f_mul_scalar(std::f64::consts::PI)?
                .f_div_scalar(2)?
                .f_cos()?
                .f_pow_tensor_scalar(2)?;
            let alpha = alpha.f_div(&alpha.f_get(0)?)?;
            alpha
                .f_slice(0, 1, n + 1, 1)?
                .f_div(&alpha.f_slice(0, 0, n, 1)?)?
                .f_rsub_scalar(1)?
                .f_clamp(0., 0.999)?
        }
        "squaredcos_cap_v2" => {
            let f = |t: f64| {
                ((t + 0.008) / 1.008 * std::f64::consts::PI / 2.)
                    .cos()
                    .powi(2)
            };
            let values: Vec<f64> = (0..n)
                .map(|i| (1. - f((i + 1) as f64 / n as f64) / f(i as f64 / n as f64)).min(0.999))
                .collect();
            Tensor::f_from_slice(&values)?
        }
        _ => return Err(error("ValueError", format!("schedule '{name}' unknown."))),
    })
}
pub fn rescale_zero_snr(sigmas: &Tensor) -> Result<Tensor, Error> {
    let alpha = sigmas.f_mul(sigmas)?.f_add_scalar(1)?.f_reciprocal()?;
    let mut root = alpha.f_sqrt()?;
    let first = crate::native::clone_tensor(&root.f_get(0)?)?;
    let last = crate::native::clone_tensor(&root.f_get(-1)?)?;
    let _ = root.f_sub_(&last)?;
    let _ = root.f_mul_(&first.f_div(&first.f_sub(&last)?)?)?;
    let alpha = root.f_pow_tensor_scalar(2)?;
    let _ = alpha.f_get(-1)?.f_fill_(4.8973451890853435e-08)?;
    Ok(alpha
        .f_rsub_scalar(1)?
        .f_div(&alpha)?
        .f_pow_tensor_scalar(0.5)?)
}
pub struct ModelSampling {
    pub flavor: Flavor,
    pub config: Config,
    pub sigmas: Tensor,
    pub log_sigmas: Option<Tensor>,
    pub num_timesteps: i64,
    pub cosine_s: Option<Tensor>,
    pub initial_alpha: Option<Tensor>,
    pub skip_steps: i64,
}
impl ModelSampling {
    /// Callers supply source defaults for their class (Flux uses shift 1.15).
    pub fn new(
        native: &Kitchen,
        flavor: Flavor,
        config: Config,
        given: Option<&Tensor>,
    ) -> Result<Self, Error> {
        let dtype = native.default_dtype()?;
        let mut model = Self {
            flavor,
            config,
            sigmas: Tensor::new(),
            log_sigmas: None,
            num_timesteps: 0,
            cosine_s: None,
            initial_alpha: None,
            skip_steps: 1,
        };
        match flavor {
            Flavor::Discrete | Flavor::DiscreteEdm | Flavor::Distilled | Flavor::Cascade => {
                // StableCascade's base constructor deliberately uses default settings.
                let base = if flavor == Flavor::Cascade {
                    Config::default()
                } else {
                    model.config.clone()
                };
                if given.is_none() && base.beta_schedule == "squaredcos_cap_v2" {
                    return Err(error(
                        "TypeError",
                        "cumprod(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
                    ));
                }
                let betas = if let Some(betas) = given {
                    betas.shallow_clone()
                } else {
                    beta_schedule(
                        &base.beta_schedule,
                        base.timesteps,
                        base.linear_start,
                        base.linear_end,
                        0.008,
                    )?
                };
                model.num_timesteps = betas.size()[0];
                let alphas = betas.f_rsub_scalar(1.)?.f_cumprod(0, None)?;
                let mut sigmas = alphas
                    .f_rsub_scalar(1)?
                    .f_div(&alphas)?
                    .f_pow_tensor_scalar(0.5)?;
                if base.zsnr {
                    sigmas = rescale_zero_snr(&sigmas)?;
                }
                model.set_sigmas(&sigmas)?;
                if flavor == Flavor::Distilled {
                    model.skip_steps = model.num_timesteps / 50;
                    let output = Tensor::f_zeros([50], (Kind::Float, Device::Cpu))?;
                    if model.num_timesteps == 0 {
                        return Err(error(
                            "IndexError",
                            "index -1 is out of bounds for dimension 0 with size 0",
                        ));
                    }
                    for i in 0..50 {
                        output.f_get(49 - i)?.f_copy_(
                            &model
                                .sigmas
                                .f_get(model.num_timesteps - 1 - i * model.skip_steps)?,
                        )?;
                    }
                    model.set_sigmas(&output)?;
                }
                if flavor == Flavor::Cascade {
                    let s = Tensor::from(model.config.cosine_s).f_to_kind(dtype)?;
                    let initial = s
                        .f_div(&s.f_add_scalar(1)?)?
                        .f_mul_scalar(std::f64::consts::PI)?
                        .f_mul_scalar(0.5)?
                        .f_cos()?
                        .f_pow_tensor_scalar(2)?;
                    model.cosine_s = Some(s);
                    model.initial_alpha = Some(initial);
                    model.num_timesteps = 10000;
                    let sigmas = Tensor::f_empty([10000], (Kind::Float, Device::Cpu))?;
                    for i in 0..10000 {
                        let t = Tensor::from((i + 1) as f64 / 10000.).f_to_kind(dtype)?;
                        sigmas.f_get(i)?.f_copy_(&model.cascade_sigma(&t)?)?;
                    }
                    model.set_sigmas(&sigmas)?;
                }
            }
            Flavor::ContinuousEdm | Flavor::ContinuousV | Flavor::Cosmos => {
                let c = &model.config;
                if c.sigma_min <= 0. || c.sigma_max <= 0. {
                    return Err(error("ValueError", "math domain error"));
                }
                model.sigmas =
                    linspace(c.sigma_min.ln(), c.sigma_max.ln(), 1000, dtype)?.f_exp()?;
                model.log_sigmas = Some(model.sigmas.f_log()?);
            }
            Flavor::DiscreteFlow | Flavor::AudioVideo | Flavor::Flux => {
                let c = &model.config;
                let mut time =
                    Tensor::f_arange_start(1, c.timesteps + 1, (Kind::Int64, Device::Cpu))?
                        .f_div_scalar(c.timesteps)?;
                if flavor != Flavor::Flux {
                    time = time.f_mul_scalar(c.multiplier)?;
                }
                model.sigmas = model.sigma(&time)?;
            }
        }
        Ok(model)
    }
    pub fn set_sigmas(&mut self, sigmas: &Tensor) -> Result<(), Error> {
        self.sigmas = sigmas.f_to_kind(Kind::Float)?;
        self.log_sigmas = Some(sigmas.f_log()?.f_to_kind(Kind::Float)?);
        Ok(())
    }
    pub fn audio_scale(&self) -> Result<f64, Error> {
        if let Some(shift) = self.config.audio_shift {
            if shift == 0. {
                return Err(error("ZeroDivisionError", "float division by zero"));
            }
            Ok(self.config.shift / shift)
        } else {
            Ok(1.)
        }
    }
    fn cascade_sigma(&self, t: &Tensor) -> Result<Tensor, Error> {
        let s = self.cosine_s.as_ref().unwrap();
        let mut alpha = t
            .f_add(s)?
            .f_div(&s.f_add_scalar(1)?)?
            .f_mul_scalar(std::f64::consts::PI)?
            .f_mul_scalar(0.5)?
            .f_cos()?
            .f_pow_tensor_scalar(2)?
            .f_div(self.initial_alpha.as_ref().unwrap())?;
        if self.config.shift != 1. {
            let mut logsnr = alpha.f_div(&alpha.f_rsub_scalar(1)?)?.f_log()?;
            // torch.tensor(shift) uses the current global default type, the
            // source constants remain CPU values even when timestep is CUDA.
            let shift = Tensor::from(self.config.shift).f_to_kind(s.kind())?;
            let _ = logsnr.f_add_(
                &shift
                    .f_reciprocal()?
                    .f_mul_scalar(1.)?
                    .f_log()?
                    .f_mul_scalar(2)?,
            )?;
            alpha = logsnr.f_sigmoid()?;
        }
        alpha = alpha.f_clamp(0.0001, 0.9999)?;
        Ok(alpha
            .f_rsub_scalar(1)?
            .f_div(&alpha)?
            .f_pow_tensor_scalar(0.5)?)
    }
    pub fn percent_to_sigma(
        &self,
        native: &Kitchen,
        percent: f64,
    ) -> Result<crate::execution_value::Value, Error> {
        use crate::execution_value::Value as V;
        let flow = matches!(
            self.flavor,
            Flavor::Flux | Flavor::DiscreteFlow | Flavor::AudioVideo
        );
        if percent <= 0. {
            return Ok(V::Float(if flow { 1. } else { 999999999.9 }));
        }
        if percent >= 1. {
            return Ok(V::Float(0.));
        }
        let p = 1. - percent;
        Ok(match self.flavor {
            Flavor::Flux => {
                V::Float(self.config.shift.exp() / (self.config.shift.exp() + (1. / p - 1.)))
            }
            Flavor::DiscreteFlow | Flavor::AudioVideo => V::Float(if self.config.shift == 1. {
                p
            } else {
                self.config.shift * p / (1. + (self.config.shift - 1.) * p)
            }),
            Flavor::ContinuousEdm | Flavor::ContinuousV | Flavor::Cosmos => {
                let lo = self.sigmas.f_get(0)?.double_value(&[]).ln();
                V::Float(((self.sigmas.f_get(-1)?.double_value(&[]).ln() - lo) * p + lo).exp())
            }
            Flavor::Cascade => {
                V::tensor(self.sigma(&Tensor::from(p).f_to_kind(native.default_dtype()?)?)?)
            }
            _ => V::Float(
                self.sigma(&Tensor::from(p * 999.).f_to_kind(native.default_dtype()?)?)?
                    .double_value(&[]),
            ),
        })
    }
}
impl ScheduleModel for ModelSampling {
    fn sigmas(&self) -> &Tensor {
        &self.sigmas
    }
    fn timestep(&self, sigma: &Tensor) -> Result<Tensor, Error> {
        Ok(match self.flavor {
            Flavor::Discrete | Flavor::Distilled => {
                let logs = self.log_sigmas.as_ref().unwrap();
                let distance = sigma
                    .f_log()?
                    .f_to_device(logs.device())?
                    .f_sub(&logs.f_unsqueeze(1)?)?;
                let t = distance
                    .f_abs()?
                    .f_argmin(0, false)?
                    .f_view(sigma.size().as_slice())?
                    .f_to_device(sigma.device())?;
                if self.flavor == Flavor::Distilled {
                    t.f_mul_scalar(self.skip_steps)?
                        .f_add_scalar(self.skip_steps - 1)?
                } else {
                    t
                }
            }
            Flavor::DiscreteEdm | Flavor::ContinuousEdm => sigma.f_log()?.f_mul_scalar(0.25)?,
            Flavor::ContinuousV => sigma
                .f_atan()?
                .f_div_scalar(std::f64::consts::PI)?
                .f_mul_scalar(2)?,
            Flavor::DiscreteFlow | Flavor::AudioVideo => {
                sigma.f_mul_scalar(self.config.multiplier)?
            }
            Flavor::Flux => sigma.shallow_clone(),
            Flavor::Cosmos => sigma.f_div(&sigma.f_add_scalar(1)?)?,
            Flavor::Cascade => {
                let var = sigma
                    .f_mul(sigma)?
                    .f_add_scalar(1)?
                    .f_reciprocal()?
                    .f_mul_scalar(1)?
                    .f_clamp(0., 1.)?;
                let s = self.cosine_s.as_ref().unwrap().f_to_device(var.device())?;
                let initial = self
                    .initial_alpha
                    .as_ref()
                    .unwrap()
                    .f_to_device(var.device())?;
                var.f_mul(&initial)?
                    .f_pow_tensor_scalar(0.5)?
                    .f_acos()?
                    .f_div_scalar(std::f64::consts::PI * 0.5)?
                    .f_mul(&s.f_add_scalar(1)?)?
                    .f_sub(&s)?
            }
        })
    }
    fn sigma(&self, time: &Tensor) -> Result<Tensor, Error> {
        Ok(match self.flavor {
            Flavor::Discrete | Flavor::Distilled => {
                let logs = self.log_sigmas.as_ref().unwrap();
                let mut t = time.f_to_kind(Kind::Float)?.f_to_device(logs.device())?;
                if self.flavor == Flavor::Distilled {
                    t = t
                        .f_sub_scalar(self.skip_steps - 1)?
                        .f_div_scalar(self.skip_steps)?;
                }
                let t = t.f_clamp(0., (self.sigmas.size()[0] - 1) as f64)?;
                if self.sigmas.numel() == 0 && t.dim() == 0 {
                    return Err(error(
                        "IndexError",
                        "index -1 is out of bounds for dimension 0 with size 0",
                    ));
                }
                let low = logs.f_index(&[Some(t.f_floor()?.f_to_kind(Kind::Int64)?)])?;
                let high = logs.f_index(&[Some(t.f_ceil()?.f_to_kind(Kind::Int64)?)])?;
                let w = t.f_frac()?;
                w.f_rsub_scalar(1)?
                    .f_mul(&low)?
                    .f_add(&w.f_mul(&high)?)?
                    .f_exp()?
                    .f_to_device(time.device())?
            }
            Flavor::DiscreteEdm | Flavor::ContinuousEdm => time.f_div_scalar(0.25)?.f_exp()?,
            Flavor::ContinuousV => time
                .f_mul_scalar(std::f64::consts::PI)?
                .f_div_scalar(2)?
                .f_tan()?,
            Flavor::DiscreteFlow | Flavor::AudioVideo => {
                let t = time.f_div_scalar(self.config.multiplier)?;
                if self.config.shift == 1. {
                    t
                } else {
                    t.f_mul_scalar(self.config.shift)?
                        .f_div(&t.f_mul_scalar(self.config.shift - 1.)?.f_add_scalar(1)?)?
                }
            }
            Flavor::Flux => time
                .f_reciprocal()?
                .f_mul_scalar(1)?
                .f_sub_scalar(1)?
                .f_pow_tensor_scalar(1.)?
                .f_add_scalar(self.config.shift.exp())?
                .f_reciprocal()?
                .f_mul_scalar(self.config.shift.exp())?,
            Flavor::Cascade => self.cascade_sigma(time)?,
            Flavor::Cosmos => {
                let max = self.sigmas.f_get(-1)?;
                let test = time.f_ge_tensor(&max.f_div(&max.f_add_scalar(1)?)?)?;
                if test.numel() != 1 {
                    return Err(error(
                        "RuntimeError",
                        "Boolean value of Tensor with more than one value is ambiguous",
                    ));
                }
                if test.int64_value(&[]) != 0 {
                    max
                } else {
                    time.f_div(&time.f_rsub_scalar(1)?)?
                }
            }
        })
    }
}
