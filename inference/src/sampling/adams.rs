//! Stochastic Adams predictor/corrector and coefficient systems from sa_solver.
use super::{
    locals::SourceLocals,
    parameterization::Parameterization,
    sde::offset_first,
    solvers::{Execution, NoiseKind, half_log_snr},
};
use crate::native::Error;
use tch::Tensor;

pub fn exponential_coefficients(
    s: &Tensor,
    t: &Tensor,
    order: i64,
    tau: f64,
) -> Result<Tensor, Error> {
    let multiplier = 1. + tau.powi(2);
    let h = t.f_sub(s)?;
    let p = Tensor::f_arange(order, (s.kind(), s.device()))?;
    let products = t
        .f_pow(&p)?
        .f_sub(&s.f_pow(&p)?.f_mul(&h.f_mul_scalar(-multiplier)?.f_exp()?)?)?;
    let depth = p.f_unsqueeze(1)?.f_sub(&p.f_unsqueeze(0)?)?;
    let factorial = p.f_add_scalar(1)?.f_lgamma()?;
    let mut coefficients = factorial
        .f_unsqueeze(1)?
        .f_sub(&factorial.f_unsqueeze(0)?)?;
    if tau > 0. {
        coefficients = coefficients.f_sub(&depth.f_mul_scalar(multiplier.ln())?)?;
    }
    let signs = Tensor::f_where_scalar(&depth.f_remainder(2)?.f_eq(0)?, 1., -1.)?;
    coefficients = coefficients.f_exp()?.f_mul(&signs)?.f_tril(0)?;
    Ok(coefficients.f_matmul(&products)?)
}
pub fn coefficients(
    sigma_next: &Tensor,
    lambdas: &Tensor,
    s: &Tensor,
    t: &Tensor,
    tau: f64,
    simple: bool,
    corrector: bool,
) -> Result<Tensor, Error> {
    let n = lambdas.size()[0];
    let alpha = sigma_next.f_mul(&t.f_exp()?)?;
    if simple && n == 2 {
        let multiplier = 1. + tau.powi(2);
        let h = t.f_sub(s)?;
        let (a, b) = if corrector {
            let b1 = alpha.f_mul(&h.f_mul_scalar(0.5 * multiplier)?)?;
            let b2 = alpha
                .f_mul(&h.f_neg()?.f_mul_scalar(multiplier)?.f_expm1()?.f_neg()?)?
                .f_sub(&b1)?;
            (b1, b2)
        } else {
            let b2 = alpha
                .f_mul(&h.f_pow_tensor_scalar(2)?.f_mul_scalar(0.5 * multiplier)?)?
                .f_div(&lambdas.f_get(-2)?.f_sub(s)?)?;
            let b1 = alpha
                .f_mul(&h.f_neg()?.f_mul_scalar(multiplier)?.f_expm1()?.f_neg()?)?
                .f_sub(&b2)?;
            (b1, b2)
        };
        return Ok(Tensor::f_stack(&[b, a], 0)?);
    }
    let exp = exponential_coefficients(s, t, n, tau)?;
    let vander = Tensor::f_vander(lambdas, Some(n), true)?.f_transpose(0, 1)?;
    let integrals = Tensor::f_linalg_solve(&vander, &exp, true)?;
    Ok(alpha.f_mul(&integrals)?)
}
pub type Tau<'a> = &'a dyn Fn(&Tensor) -> Result<f64, Error>;
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub noise: f64,
    pub predictor: i64,
    pub corrector: i64,
    pub pece: bool,
    pub simple_order_2: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            noise: 1.,
            predictor: 3,
            corrector: 4,
            pece: false,
            simple_order_2: false,
        }
    }
}
fn tail(values: &[Tensor], count: i64) -> Vec<Tensor> {
    let n = values.len();
    let count = if count == 0 { n } else { count.max(0) as usize };
    values[n.saturating_sub(count)..]
        .iter()
        .map(Tensor::shallow_clone)
        .collect()
}
fn contraction(
    frame: &mut SourceLocals,
    predictions: &[Tensor],
    count: i64,
    coefficients: &Tensor,
) -> Result<Tensor, Error> {
    let matrix = frame.keep("pred_mat", Tensor::f_stack(&tail(predictions, count), 1)?);
    Ok(matrix.f_tensordot(coefficients, [1], [0])?)
}
pub fn sample(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: Settings,
    tau: Option<Tau<'_>>,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    if sigmas.size()[0] <= 1 {
        return frame.finish(exec, x);
    }
    let template = x.shallow_clone();
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise_scale = settings.noise * model.noise_scale;
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    let lambdas = frame.keep("lambdas", half_log_snr(&sigmas, model)?);
    let default_interval = if tau.is_none() {
        Some((exec.percent_to_sigma(0.2)?, exec.percent_to_sigma(0.8)?))
    } else {
        None
    };
    let max_order = settings.predictor.max(settings.corrector);
    let mut predicted = frame.keep("x_pred", x.shallow_clone());
    let mut h: Option<Tensor> = None;
    let mut tau_t = 0_f64;
    let mut noise: Option<Tensor> = None;
    let mut history: Vec<Tensor> = Vec::new();
    let lower = sigmas.double_value(&[sigmas.size()[0] - 1]) == 0.;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let mut p = frame
            .denoise(exec, &predicted, &sigma.f_mul(&ones)?, false)?
            .denoised;
        let _ = frame.keep("denoised", p.shallow_clone());
        frame.callback(
            exec,
            crate::models::qwen_image21::EulerStep {
                x: &predicted,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p,
            },
        )?;
        history.push(p.shallow_clone());
        history = tail(&history, max_order);
        frame.keep_list("pred_list", &history);
        let mut predictor = settings.predictor.min(history.len() as i64);
        let mut corrector = if i == 0 || (next.double_value(&[]) == 0. && !settings.pece) {
            0
        } else {
            settings.corrector.min(history.len() as i64)
        };
        if lower {
            predictor = predictor.min(sigmas.size()[0] - 2 - i);
            corrector = corrector.min(sigmas.size()[0] - 1 - i);
        }
        if corrector == 0 {
            x = frame.keep("x", predicted.shallow_clone());
        } else {
            let times = frame.keep(
                "curr_lambdas",
                lambdas.f_narrow(0, i - corrector + 1, corrector)?,
            );
            let b = frame.keep(
                "b_coeffs",
                coefficients(
                    &sigma,
                    &times,
                    &lambdas.f_get(i - 1)?,
                    &lambdas.f_get(i)?,
                    tau_t,
                    settings.simple_order_2,
                    true,
                )?,
            );
            let residual = contraction(&mut frame, &history, corrector, &b)?;
            let residual = frame.keep("corr_res", residual);
            x = frame.keep(
                "x",
                sigma
                    .f_div(&sigmas.f_get(i - 1)?)?
                    .f_mul(&h.as_ref().unwrap().f_mul_scalar(-tau_t.powi(2))?.f_exp()?)?
                    .f_mul(&x)?
                    .f_add(&residual)?,
            );
            if tau_t > 0. && noise_scale > 0. {
                x = frame.keep("x", x.f_add(noise.as_ref().unwrap())?);
            }
            if settings.pece {
                p = frame
                    .denoise(exec, &x, &sigma.f_mul(&ones)?, false)?
                    .denoised;
                let _ = frame.keep("denoised", p.shallow_clone());
                let last = history.len() - 1;
                history[last] = p.shallow_clone();
                frame.keep_list("pred_list", &history);
            }
        }
        if next.double_value(&[]) == 0. {
            predicted = frame.keep("x_pred", p);
        } else {
            exec.source_locals(&frame, "tau")?;
            tau_t = if let Some(tau) = tau {
                tau(&next)?
            } else {
                let (start, end) = default_interval.unwrap();
                if start >= next.double_value(&[]) && next.double_value(&[]) >= end {
                    1.
                } else {
                    0.
                }
            };
            let times = frame.keep(
                "curr_lambdas",
                lambdas.f_narrow(0, i - predictor + 1, predictor)?,
            );
            let b = frame.keep(
                "b_coeffs",
                coefficients(
                    &next,
                    &times,
                    &lambdas.f_get(i)?,
                    &lambdas.f_get(i + 1)?,
                    tau_t,
                    settings.simple_order_2,
                    false,
                )?,
            );
            let residual = contraction(&mut frame, &history, predictor, &b)?;
            let residual = frame.keep("pred_res", residual);
            let step = frame.keep("h", lambdas.f_get(i + 1)?.f_sub(&lambdas.f_get(i)?)?);
            h = Some(step.shallow_clone());
            predicted = frame.keep(
                "x_pred",
                next.f_div(&sigma)?
                    .f_mul(&step.f_mul_scalar(-tau_t.powi(2))?.f_exp()?)?
                    .f_mul(&x)?
                    .f_add(&residual)?,
            );
            if tau_t > 0. && noise_scale > 0. {
                let n = frame.keep(
                    "noise",
                    exec.noise(&template, &sigma, &next, NoiseKind::Independent)?
                        .f_mul(&next)?
                        .f_mul(
                            &step
                                .f_mul_scalar(-2. * tau_t.powi(2))?
                                .f_expm1()?
                                .f_neg()?
                                .f_sqrt()?,
                        )?
                        .f_mul_scalar(noise_scale)?,
                );
                noise = Some(n.shallow_clone());
                predicted = frame.keep("x_pred", predicted.f_add(&n)?);
            }
        }
    }
    frame.finish(exec, predicted)
}
