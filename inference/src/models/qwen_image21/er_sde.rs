//! Original `sample_er_sde` CONST/FLUX path, preserving ATen expression order.
use super::{EulerExecution, EulerStep, FluxSampling};
use crate::native::Error;
use tch::{Kind, Tensor};

#[derive(Clone, Copy)]
pub struct ErSdeSettings {
    pub noise: f64,
    pub max_stage: i64,
}
impl Default for ErSdeSettings {
    fn default() -> Self {
        Self {
            noise: 1.,
            max_stage: 3,
        }
    }
}
fn noise_scaler(x: &Tensor) -> Result<Tensor, Error> {
    Ok(x.f_mul(&x.f_pow_tensor_scalar(0.3)?.f_exp()?.f_add_scalar(10.)?)?)
}
fn previous(value: &Option<Tensor>) -> Result<&Tensor, Error> {
    value.as_ref().ok_or_else(|| Error::ReferenceException {
        class: "TypeError",
        message: "unsupported operand type(s) for -: 'Tensor' and 'NoneType'".into(),
    })
}
/// The execution's noise sampler must use the original seeded generator:
/// CUDA seed, CPU seed+1. It is separate from initial `prepare_noise` state.
pub fn sample_er_sde<E: EulerExecution>(
    mut x: Tensor,
    sigmas: &Tensor,
    sampling: &FluxSampling,
    settings: ErSdeSettings,
    execution: &mut E,
) -> Result<Tensor, Error> {
    let _grad = tch::no_grad_guard();
    // default_noise_sampler closes over the original input, even if a custom
    // denoiser later changes the solver state's dtype. Retain that ownership.
    let noise_template = x.shallow_clone();
    let s_in = Tensor::f_ones([x.size()[0]], (x.kind(), x.device()))?;
    let points = Tensor::f_arange_start(0., 200., (Kind::Float, x.device()))?;
    let sigmas = if sigmas.size()[0] > 1 && sigmas.double_value(&[0]) >= 1. {
        let copy = sigmas.copy();
        let _ = copy.f_get(0)?.f_fill_(sampling.percent_to_sigma(1e-4))?;
        copy
    } else {
        sigmas.shallow_clone()
    };
    let half_log_snrs = sigmas.f_logit(None)?.f_neg()?;
    let lambdas = half_log_snrs.f_neg()?.f_exp()?;
    let mut old_denoised: Option<Tensor> = None;
    let mut old_derivative: Option<Tensor> = None;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next_sigma = sigmas.f_get(i + 1)?;
        let denoised = execution.denoise(&x, &sigma.f_mul(&s_in)?)?;
        execution.callback(EulerStep {
            x: &x,
            i,
            sigma: &sigma,
            sigma_hat: &sigma,
            denoised: &denoised,
        })?;
        let stage = settings.max_stage.min(i + 1);
        if next_sigma.double_value(&[]) == 0. {
            x = denoised.shallow_clone();
        } else {
            let s = lambdas.f_get(i)?;
            let t = lambdas.f_get(i + 1)?;
            let alpha_s = sigma.f_div(&s)?;
            let alpha_t = next_sigma.f_div(&t)?;
            let r_alpha = alpha_t.f_div(&alpha_s)?;
            let r = noise_scaler(&t)?.f_div(&noise_scaler(&s)?)?;
            x = r_alpha
                .f_mul(&r)?
                .f_mul(&x)?
                .f_add(&alpha_t.f_mul(&r.f_rsub_scalar(1.)?)?.f_mul(&denoised)?)?;
            if stage >= 2 {
                let dt = t.f_sub(&s)?;
                let step = dt.f_neg()?.f_div_scalar(200.)?;
                let positions = t.f_add(&points.f_mul(&step)?)?;
                let scaled = noise_scaler(&positions)?;
                let integral = scaled
                    .f_reciprocal()?
                    .f_mul_scalar(1.)?
                    .f_sum(None)?
                    .f_mul(&step)?;
                let derivative = denoised
                    .f_sub(previous(&old_denoised)?)?
                    .f_div(&s.f_sub(&lambdas.f_get(i - 1)?)?)?;
                x = x.f_add(
                    &alpha_t
                        .f_mul(&dt.f_add(&integral.f_mul(&noise_scaler(&t)?)?)?)?
                        .f_mul(&derivative)?,
                )?;
                if stage >= 3 {
                    let integral_u = positions
                        .f_sub(&s)?
                        .f_div(&scaled)?
                        .f_sum(None)?
                        .f_mul(&step)?;
                    let second = derivative
                        .f_sub(previous(&old_derivative)?)?
                        .f_div(&s.f_sub(&lambdas.f_get(i - 2)?)?.f_div_scalar(2)?)?;
                    x = x.f_add(
                        &alpha_t
                            .f_mul(
                                &dt.f_pow_tensor_scalar(2)?
                                    .f_div_scalar(2)?
                                    .f_add(&integral_u.f_mul(&noise_scaler(&t)?)?)?,
                            )?
                            .f_mul(&second)?,
                    )?;
                }
                old_derivative = Some(derivative);
            }
            if settings.noise > 0. {
                let deviation = t
                    .f_pow_tensor_scalar(2)?
                    .f_sub(
                        &s.f_pow_tensor_scalar(2)?
                            .f_mul(&r.f_pow_tensor_scalar(2)?)?,
                    )?
                    .f_sqrt()?
                    .f_nan_to_num(Some(0.), None, None)?;
                x = x.f_add(
                    &alpha_t
                        .f_mul(&execution.noise_like(&noise_template)?)?
                        .f_mul_scalar(settings.noise)?
                        .f_mul(&deviation)?,
                )?;
            }
        }
        old_denoised = Some(denoised);
    }
    Ok(x)
}
