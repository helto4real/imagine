//! Frozen FLUX/CONST parameterization used by Qwen21. Explicit initial sampler
//! bindings; this module does not narrow the complete sampler registry target.
use crate::native::Error;
use tch::{Device, Kind, Tensor};
pub struct FluxSampling {
    pub shift: f64,
    pub sigmas: Tensor,
}
impl FluxSampling {
    pub fn new(shift: f64, timesteps: i64) -> Result<Self, Error> {
        let t = Tensor::f_arange_start(1, timesteps + 1, (Kind::Int64, Device::Cpu))?
            .f_div_scalar(timesteps)?;
        let sigmas = Self::shift_time(shift, &t)?;
        Ok(Self { shift, sigmas })
    }
    pub fn qwen21() -> Result<Self, Error> {
        Self::new(0.69, 10000)
    }
    fn shift_time(mu: f64, t: &Tensor) -> Result<Tensor, Error> {
        let e = mu.exp();
        Ok(t.f_reciprocal()?
            .f_mul_scalar(1)?
            .f_sub_scalar(1)?
            .f_pow_tensor_scalar(1.)?
            .f_add_scalar(e)?
            .f_reciprocal()?
            .f_mul_scalar(e)?)
    }
    pub fn sigma(&self, t: &Tensor) -> Result<Tensor, Error> {
        Self::shift_time(self.shift, t)
    }
    pub fn simple(&self, steps: usize) -> Result<Tensor, Error> {
        let size = self.sigmas.size()[0];
        let ss = size as f64 / steps as f64;
        let mut values = (0..steps)
            .map(|i| {
                self.sigmas
                    .double_value(&[size - (1 + (i as f64 * ss) as i64)]) as f32
            })
            .collect::<Vec<_>>();
        values.push(0.);
        Ok(Tensor::f_from_slice(&values)?)
    }
    pub fn normal(&self, mut steps: i64, sgm: bool) -> Result<Tensor, Error> {
        let size = self.sigmas.size()[0];
        let start = self.sigmas.double_value(&[size - 1]);
        let end = self.sigmas.double_value(&[0]);
        let append = sgm
            || self
                .sigma(&Tensor::f_from_slice(&[end as f32])?)?
                .double_value(&[0])
                .abs()
                > 0.00001;
        if !append {
            steps += 1;
        }
        let t = Tensor::f_linspace(
            start,
            end,
            steps + i64::from(sgm),
            (Kind::Float, Device::Cpu),
        )?;
        let t = if sgm { t.f_narrow(0, 0, steps)? } else { t };
        let mut values = Vec::new();
        for i in 0..t.size()[0] {
            values.push(self.sigma(&t.f_get(i)?)?.double_value(&[]) as f32);
        }
        if append {
            values.push(0.);
        }
        Ok(Tensor::f_from_slice(&values)?)
    }
}
fn append_sigma(sigma: &Tensor, rank: usize) -> Result<Tensor, Error> {
    let mut shape = sigma.size();
    shape.resize(rank, 1);
    Ok(sigma.f_reshape(&shape)?)
}
pub fn flow_denoised(sigma: &Tensor, output: &Tensor, input: &Tensor) -> Result<Tensor, Error> {
    Ok(input.f_sub(&output.f_mul(&append_sigma(sigma, output.dim())?)?)?)
}
pub fn flow_noise_scaling(
    sigma: &Tensor,
    noise: &Tensor,
    latent: &Tensor,
    noise_scale: f64,
) -> Result<Tensor, Error> {
    let s = append_sigma(sigma, noise.dim())?;
    Ok(s.f_mul(&noise.f_mul_scalar(noise_scale)?)?
        .f_add(&s.f_neg()?.f_add_scalar(1)?.f_mul(latent)?)?)
}
pub fn flow_inverse_noise_scaling(sigma: &Tensor, latent: &Tensor) -> Result<Tensor, Error> {
    Ok(latent.f_div(
        &append_sigma(sigma, latent.dim())?
            .f_neg()?
            .f_add_scalar(1)?,
    )?)
}
pub fn cfg_combine(cond: &Tensor, uncond: &Tensor, scale: f64) -> Result<Tensor, Error> {
    Ok(uncond.f_add(&cond.f_sub(uncond)?.f_mul_scalar(scale)?)?)
}
/// Source _calc_cond_batch for one full-area, unit-strength positive condition
/// and no negative branch (the CFG=1 shortcut). Preserve its actual aggregation
/// operations, including signed-zero behavior and the 1e-37 count initializer.
pub fn single_full_condition_cfg1(x: &Tensor, model_denoised: &Tensor) -> Result<Tensor, Error> {
    let mult = x.f_ones_like()?.f_mul_scalar(1.)?;
    let mut cond = x.f_zeros_like()?;
    let mut count = x.f_ones_like()?.f_mul_scalar(1e-37)?;
    let _ = cond.f_add_(&model_denoised.f_mul(&mult)?)?;
    let _ = count.f_add_(&mult)?;
    let _ = cond.f_div_(&count)?;
    let mut uncond = x.f_zeros_like()?;
    let _ = uncond.f_div_(&x.f_ones_like()?.f_mul_scalar(1e-37)?)?;
    cfg_combine(&cond, &uncond, 1.)
}
/// The source uses math.isclose(rel_tol=1e-9, abs_tol=0) for this shortcut.
pub fn cfg1_shortcut(scale: f64, disabled: bool) -> bool {
    !disabled
        && scale.is_finite()
        && (scale == 1. || (scale - 1.).abs() <= 1e-9 * scale.abs().max(1.))
}
#[derive(Clone, Copy)]
pub struct EulerSettings {
    pub churn: f64,
    pub tmin: f64,
    pub tmax: f64,
    pub noise: f64,
}
impl Default for EulerSettings {
    fn default() -> Self {
        Self {
            churn: 0.,
            tmin: 0.,
            tmax: f64::INFINITY,
            noise: 1.,
        }
    }
}
pub struct EulerStep<'a> {
    pub x: &'a Tensor,
    pub i: i64,
    pub sigma: &'a Tensor,
    pub sigma_hat: &'a Tensor,
    pub denoised: &'a Tensor,
}
pub trait EulerExecution {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor) -> Result<Tensor, Error>;
    /// Exact source torch.randn_like stream; no alternative Rust RNG.
    fn noise_like(&mut self, x: &Tensor) -> Result<Tensor, Error>;
    fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error>;
}
pub fn sample_euler<E: EulerExecution>(
    mut x: Tensor,
    sigmas: &Tensor,
    settings: EulerSettings,
    e: &mut E,
) -> Result<Tensor, Error> {
    let s_in = Tensor::f_ones([x.size()[0]], (x.kind(), x.device()))?;
    let steps = sigmas.size()[0] - 1;
    for i in 0..steps {
        let s = sigmas.f_get(i)?;
        let gamma = if settings.churn > 0.
            && s.double_value(&[]) >= settings.tmin
            && s.double_value(&[]) <= settings.tmax
        {
            (settings.churn / steps as f64).min(2f64.sqrt() - 1.)
        } else {
            0.
        };
        let hat = if settings.churn > 0. {
            s.f_mul_scalar(gamma + 1.)?
        } else {
            s.shallow_clone()
        };
        if gamma > 0. {
            let eps = e.noise_like(&x)?.f_mul_scalar(settings.noise)?;
            x = x.f_add(
                &eps.f_mul(
                    &hat.f_pow_tensor_scalar(2)?
                        .f_sub(&s.f_pow_tensor_scalar(2)?)?
                        .f_pow_tensor_scalar(0.5)?,
                )?,
            )?;
        }
        let denoised = e.denoise(&x, &hat.f_mul(&s_in)?)?;
        let d = x.f_sub(&denoised)?.f_div(&append_sigma(&hat, x.dim())?)?;
        e.callback(EulerStep {
            x: &x,
            i,
            sigma: &s,
            sigma_hat: &hat,
            denoised: &denoised,
        })?;
        let dt = sigmas.f_get(i + 1)?.f_sub(&hat)?;
        x = x.f_add(&d.f_mul(&dt)?)?;
    }
    Ok(x)
}
