//! SEEDS/exponential Heun and the general ER-SDE contract.
use super::{
    locals::SourceLocals,
    parameterization::Parameterization,
    sde::offset_first,
    solvers::{Execution, NoiseKind, half_log_snr, snr_to_sigma},
};
use crate::native::Error;
use tch::{Kind, Tensor};
fn phi1(h: &Tensor) -> Result<Tensor, Error> {
    Ok(h.f_expm1()?)
}
fn phi2(h: &Tensor) -> Result<Tensor, Error> {
    Ok(h.f_expm1()?.f_sub(h)?.f_div(h)?)
}
#[derive(Clone, Copy, Debug)]
pub struct Seeds2 {
    pub eta: f64,
    pub noise: f64,
    pub r: f64,
}
impl Default for Seeds2 {
    fn default() -> Self {
        Self {
            eta: 1.,
            noise: 1.,
            r: 0.5,
        }
    }
}
pub fn seeds_2(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: Seeds2,
    solver_type: &str,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    if !matches!(solver_type, "phi_1" | "phi_2") {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "solver_type must be 'phi_1' or 'phi_2'".into(),
        });
    }
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let template = x.shallow_clone();
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = settings.noise * model.noise_scale;
    let inject = settings.eta > 0. && noise > 0.;
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    if settings.r == 0. {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "float division by zero".into(),
        });
    }
    let factor = 1. / (2. * settings.r);
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        frame.callback(
            exec,
            crate::models::qwen_image21::EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        if next.double_value(&[]) == 0. {
            x = frame.keep("x", p.denoised);
            continue;
        }
        let ls = frame.keep("lambda_s", half_log_snr(&sigma, model)?);
        let lt = frame.keep("lambda_t", half_log_snr(&next, model)?);
        let h = frame.keep("h", lt.f_sub(&ls)?);
        let he = frame.keep("h_eta", h.f_mul_scalar(settings.eta + 1.)?);
        let ls1 = frame.keep("lambda_s_1", ls.f_lerp(&lt, settings.r)?);
        let ss1 = frame.keep("sigma_s_1", snr_to_sigma(&ls1, model)?);
        let alpha_s1 = frame.keep("alpha_s_1", ss1.f_mul(&ls1.f_exp()?)?);
        let alpha_t = frame.keep("alpha_t", next.f_mul(&lt.f_exp()?)?);
        let mut x2 = frame.keep(
            "x_2",
            ss1.f_div(&sigma)?
                .f_mul(
                    &h.f_mul_scalar(-settings.r)?
                        .f_mul_scalar(settings.eta)?
                        .f_exp()?,
                )?
                .f_mul(&x)?
                .f_sub(
                    &alpha_s1
                        .f_mul(&phi1(&he.f_mul_scalar(-settings.r)?)?)?
                        .f_mul(&p.denoised)?,
                )?,
        );
        let mut sde_noise = if inject {
            let sde = frame.keep(
                "sde_noise",
                h.f_mul_scalar(-2. * settings.r)?
                    .f_mul_scalar(settings.eta)?
                    .f_expm1()?
                    .f_neg()?
                    .f_sqrt()?
                    .f_mul(&exec.noise(&template, &sigma, &ss1, NoiseKind::Independent)?)?,
            );
            x2 = frame.keep("x_2", x2.f_add(&sde.f_mul(&ss1)?.f_mul_scalar(noise)?)?);
            Some(sde)
        } else {
            None
        };
        let p2 = frame.denoise(exec, &x2, &ss1.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised_2", p2.denoised.shallow_clone());
        if solver_type == "phi_1" {
            let predicted = frame.keep("denoised_d", p.denoised.f_lerp(&p2.denoised, factor)?);
            x = frame.keep(
                "x",
                next.f_div(&sigma)?
                    .f_mul(&h.f_neg()?.f_mul_scalar(settings.eta)?.f_exp()?)?
                    .f_mul(&x)?
                    .f_sub(&alpha_t.f_mul(&phi1(&he.f_neg()?)?)?.f_mul(&predicted)?)?,
            );
        } else {
            let b2 = frame.keep("b2", phi2(&he.f_neg()?)?.f_div_scalar(settings.r)?);
            let b1 = frame.keep("b1", phi1(&he.f_neg()?)?.f_sub(&b2)?);
            x = frame.keep(
                "x",
                next.f_div(&sigma)?
                    .f_mul(&h.f_neg()?.f_mul_scalar(settings.eta)?.f_exp()?)?
                    .f_mul(&x)?
                    .f_sub(
                        &alpha_t.f_mul(&b1.f_mul(&p.denoised)?.f_add(&b2.f_mul(&p2.denoised)?)?)?,
                    )?,
            );
        }
        if let Some(mut sde) = sde_noise.take() {
            let segment = frame.keep(
                "segment_factor",
                h.f_mul_scalar(settings.r - 1.)?
                    .f_mul_scalar(settings.eta)?,
            );
            sde = frame.keep("sde_noise", sde.f_mul(&segment.f_exp()?)?);
            sde = frame.keep(
                "sde_noise",
                sde.f_add(
                    &segment
                        .f_mul_scalar(2)?
                        .f_expm1()?
                        .f_neg()?
                        .f_sqrt()?
                        .f_mul(&exec.noise(&template, &ss1, &next, NoiseKind::Independent)?)?,
                )?,
            );
            x = frame.keep("x", x.f_add(&sde.f_mul(&next)?.f_mul_scalar(noise)?)?);
        }
    }
    frame.finish(exec, x)
}
#[derive(Clone, Copy, Debug)]
pub struct Seeds3 {
    pub eta: f64,
    pub noise: f64,
    pub r1: f64,
    pub r2: f64,
}
impl Default for Seeds3 {
    fn default() -> Self {
        Self {
            eta: 1.,
            noise: 1.,
            r1: 1. / 3.,
            r2: 2. / 3.,
        }
    }
}
pub fn seeds_3(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: Seeds3,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let template = x.shallow_clone();
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = settings.noise * model.noise_scale;
    let inject = settings.eta > 0. && noise > 0.;
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        frame.callback(
            exec,
            crate::models::qwen_image21::EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        if next.double_value(&[]) == 0. {
            x = frame.keep("x", p.denoised);
            continue;
        }
        let ls = frame.keep("lambda_s", half_log_snr(&sigma, model)?);
        let lt = frame.keep("lambda_t", half_log_snr(&next, model)?);
        let h = frame.keep("h", lt.f_sub(&ls)?);
        let he = frame.keep("h_eta", h.f_mul_scalar(settings.eta + 1.)?);
        let ls1 = frame.keep("lambda_s_1", ls.f_lerp(&lt, settings.r1)?);
        let ls2 = frame.keep("lambda_s_2", ls.f_lerp(&lt, settings.r2)?);
        let ss1 = frame.keep("sigma_s_1", snr_to_sigma(&ls1, model)?);
        let ss2 = frame.keep("sigma_s_2", snr_to_sigma(&ls2, model)?);
        let a1 = frame.keep("alpha_s_1", ss1.f_mul(&ls1.f_exp()?)?);
        let a2 = frame.keep("alpha_s_2", ss2.f_mul(&ls2.f_exp()?)?);
        let at = frame.keep("alpha_t", next.f_mul(&lt.f_exp()?)?);
        let mut x2 = frame.keep(
            "x_2",
            ss1.f_div(&sigma)?
                .f_mul(
                    &h.f_mul_scalar(-settings.r1)?
                        .f_mul_scalar(settings.eta)?
                        .f_exp()?,
                )?
                .f_mul(&x)?
                .f_sub(
                    &a1.f_mul(&phi1(&he.f_mul_scalar(-settings.r1)?)?)?
                        .f_mul(&p.denoised)?,
                )?,
        );
        let mut sde_noise = if inject {
            let n = frame.keep(
                "sde_noise",
                h.f_mul_scalar(-2. * settings.r1)?
                    .f_mul_scalar(settings.eta)?
                    .f_expm1()?
                    .f_neg()?
                    .f_sqrt()?
                    .f_mul(&exec.noise(&template, &sigma, &ss1, NoiseKind::Independent)?)?,
            );
            x2 = frame.keep("x_2", x2.f_add(&n.f_mul(&ss1)?.f_mul_scalar(noise)?)?);
            Some(n)
        } else {
            None
        };
        let p2 = frame.denoise(exec, &x2, &ss1.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised_2", p2.denoised.shallow_clone());
        if settings.r1 == 0. {
            return Err(Error::ReferenceException {
                class: "ZeroDivisionError",
                message: "float division by zero".into(),
            });
        }
        let a32 = frame.keep(
            "a3_2",
            phi2(&he.f_mul_scalar(-settings.r2)?)?.f_mul_scalar(settings.r2 / settings.r1)?,
        );
        let a31 = frame.keep("a3_1", phi1(&he.f_mul_scalar(-settings.r2)?)?.f_sub(&a32)?);
        let mut x3 = frame.keep(
            "x_3",
            ss2.f_div(&sigma)?
                .f_mul(
                    &h.f_mul_scalar(-settings.r2)?
                        .f_mul_scalar(settings.eta)?
                        .f_exp()?,
                )?
                .f_mul(&x)?
                .f_sub(&a2.f_mul(&a31.f_mul(&p.denoised)?.f_add(&a32.f_mul(&p2.denoised)?)?)?)?,
        );
        if let Some(n) = &mut sde_noise {
            let segment = frame.keep(
                "segment_factor",
                h.f_mul_scalar(settings.r1 - settings.r2)?
                    .f_mul_scalar(settings.eta)?,
            );
            *n = frame.keep("sde_noise", n.f_mul(&segment.f_exp()?)?);
            *n = frame.keep(
                "sde_noise",
                n.f_add(
                    &segment
                        .f_mul_scalar(2)?
                        .f_expm1()?
                        .f_neg()?
                        .f_sqrt()?
                        .f_mul(&exec.noise(&template, &ss1, &ss2, NoiseKind::Independent)?)?,
                )?,
            );
            x3 = frame.keep("x_3", x3.f_add(&n.f_mul(&ss2)?.f_mul_scalar(noise)?)?);
        }
        let p3 = frame.denoise(exec, &x3, &ss2.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised_3", p3.denoised.shallow_clone());
        let b3 = frame.keep("b3", phi2(&he.f_neg()?)?.f_div_scalar(settings.r2)?);
        let b1 = frame.keep("b1", phi1(&he.f_neg()?)?.f_sub(&b3)?);
        x = frame.keep(
            "x",
            next.f_div(&sigma)?
                .f_mul(&h.f_neg()?.f_mul_scalar(settings.eta)?.f_exp()?)?
                .f_mul(&x)?
                .f_sub(&at.f_mul(&b1.f_mul(&p.denoised)?.f_add(&b3.f_mul(&p3.denoised)?)?)?)?,
        );
        if let Some(mut n) = sde_noise {
            let segment = frame.keep(
                "segment_factor",
                h.f_mul_scalar(settings.r2 - 1.)?
                    .f_mul_scalar(settings.eta)?,
            );
            n = frame.keep("sde_noise", n.f_mul(&segment.f_exp()?)?);
            n = frame.keep(
                "sde_noise",
                n.f_add(
                    &segment
                        .f_mul_scalar(2)?
                        .f_expm1()?
                        .f_neg()?
                        .f_sqrt()?
                        .f_mul(&exec.noise(&template, &ss2, &next, NoiseKind::Independent)?)?,
                )?,
            );
            x = frame.keep("x", x.f_add(&n.f_mul(&next)?.f_mul_scalar(noise)?)?);
        }
    }
    frame.finish(exec, x)
}
pub type NoiseScaler<'a> = &'a dyn Fn(&Tensor) -> Result<Tensor, Error>;
fn er_noise_scaler(x: &Tensor) -> Result<Tensor, Error> {
    Ok(x.f_mul(&x.f_pow_tensor_scalar(0.3)?.f_exp()?.f_add_scalar(10.)?)?)
}
fn previous(value: &Option<Tensor>) -> Result<&Tensor, Error> {
    value.as_ref().ok_or_else(|| Error::ReferenceException {
        class: "TypeError",
        message: "unsupported operand type(s) for -: 'Tensor' and 'NoneType'".into(),
    })
}
fn scale_er_noise(
    exec: &mut dyn Execution,
    frame: &SourceLocals,
    scaler: NoiseScaler<'_>,
    value: &Tensor,
) -> Result<Tensor, Error> {
    exec.source_locals(frame, "scaler")?;
    scaler(value)
}
pub fn er_sde(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    noise: f64,
    max_stage: i64,
    noise_scaler: Option<NoiseScaler<'_>>,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let template = x.shallow_clone();
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = noise * model.noise_scale;
    let scaler = noise_scaler.unwrap_or(&er_noise_scaler);
    let points = frame.keep(
        "point_indice",
        Tensor::f_arange_start(0., 200., (Kind::Float, x.device()))?,
    );
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    let half_logs = frame.keep("half_log_snrs", half_log_snr(&sigmas, model)?);
    let lambdas = frame.keep("er_lambdas", half_logs.f_neg()?.f_exp()?);
    let (mut old, mut old_d): (Option<Tensor>, Option<Tensor>) = (None, None);
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        frame.callback(
            exec,
            crate::models::qwen_image21::EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        let stage = max_stage.min(i + 1);
        if next.double_value(&[]) == 0. {
            x = frame.keep("x", p.denoised.shallow_clone());
        } else {
            let s = frame.keep("er_lambda_s", lambdas.f_get(i)?);
            let t = frame.keep("er_lambda_t", lambdas.f_get(i + 1)?);
            let alpha_s = frame.keep("alpha_s", sigma.f_div(&s)?);
            let alpha_t = frame.keep("alpha_t", next.f_div(&t)?);
            let ratio_alpha = frame.keep("r_alpha", alpha_t.f_div(&alpha_s)?);
            let ratio = frame.keep(
                "r",
                scale_er_noise(exec, &frame, scaler, &t)?
                    .f_div(&scale_er_noise(exec, &frame, scaler, &s)?)?,
            );
            x = frame.keep(
                "x",
                ratio_alpha.f_mul(&ratio)?.f_mul(&x)?.f_add(
                    &alpha_t
                        .f_mul(&ratio.f_rsub_scalar(1)?)?
                        .f_mul(&p.denoised)?,
                )?,
            );
            if stage >= 2 {
                let dt = frame.keep("dt", t.f_sub(&s)?);
                let step = frame.keep("lambda_step_size", dt.f_neg()?.f_div_scalar(200.)?);
                let positions = frame.keep("lambda_pos", t.f_add(&points.f_mul(&step)?)?);
                let scaled = frame.keep(
                    "scaled_pos",
                    scale_er_noise(exec, &frame, scaler, &positions)?,
                );
                let integral = frame.keep(
                    "s",
                    scaled
                        .f_reciprocal()?
                        .f_mul_scalar(1)?
                        .f_sum(None)?
                        .f_mul(&step)?,
                );
                let derivative = frame.keep(
                    "denoised_d",
                    p.denoised
                        .f_sub(previous(&old)?)?
                        .f_div(&s.f_sub(&lambdas.f_get(i - 1)?)?)?,
                );
                x = frame.keep(
                    "x",
                    x.f_add(
                        &alpha_t
                            .f_mul(&dt.f_add(
                                &integral.f_mul(&scale_er_noise(exec, &frame, scaler, &t)?)?,
                            )?)?
                            .f_mul(&derivative)?,
                    )?,
                );
                if stage >= 3 {
                    let integral_u = frame.keep(
                        "s_u",
                        positions
                            .f_sub(&s)?
                            .f_div(&scaled)?
                            .f_sum(None)?
                            .f_mul(&step)?,
                    );
                    let second = frame.keep(
                        "denoised_u",
                        derivative
                            .f_sub(previous(&old_d)?)?
                            .f_div(&s.f_sub(&lambdas.f_get(i - 2)?)?.f_div_scalar(2)?)?,
                    );
                    x = frame.keep(
                        "x",
                        x.f_add(
                            &alpha_t
                                .f_mul(
                                    &dt.f_pow_tensor_scalar(2)?.f_div_scalar(2)?.f_add(
                                        &integral_u
                                            .f_mul(&scale_er_noise(exec, &frame, scaler, &t)?)?,
                                    )?,
                                )?
                                .f_mul(&second)?,
                        )?,
                    );
                }
                old_d = Some(derivative);
                let _ = frame.keep("old_denoised_d", old_d.as_ref().unwrap().shallow_clone());
            }
            if noise > 0. {
                x = frame.keep(
                    "x",
                    x.f_add(
                        &alpha_t
                            .f_mul(&exec.noise(
                                &template,
                                &sigma,
                                &next,
                                NoiseKind::Independent,
                            )?)?
                            .f_mul_scalar(noise)?
                            .f_mul(
                                &t.f_pow_tensor_scalar(2)?
                                    .f_sub(
                                        &s.f_pow_tensor_scalar(2)?
                                            .f_mul(&ratio.f_pow_tensor_scalar(2)?)?,
                                    )?
                                    .f_sqrt()?
                                    .f_nan_to_num(Some(0.), None, None)?,
                            )?,
                    )?,
                );
            }
        }
        old = Some(p.denoised);
        let _ = frame.keep("old_denoised", old.as_ref().unwrap().shallow_clone());
    }
    frame.finish(exec, x)
}
