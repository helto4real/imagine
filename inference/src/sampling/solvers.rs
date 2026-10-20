//! Fallible shared sampler loops. Operation order and callback sites follow the
//! frozen source; unimplemented names are never aliased to a different solver.
use super::locals::SourceLocals;
use crate::{
    models::qwen_image21::{EulerSettings, EulerStep},
    native::Error,
    sampling::parameterization::Parameterization,
};
use tch::Tensor;

pub struct Prediction {
    pub denoised: Tensor,
    pub uncond_denoised: Option<Tensor>,
    pub cond_denoised: Option<Tensor>,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NoiseKind {
    GlobalLike,
    Independent,
    BrownianCpu,
    BrownianGpu,
    Custom,
}
pub trait Execution {
    fn set_denoise_mask(&mut self, _mask: Option<&Tensor>) -> Result<(), Error> {
        Ok(())
    }
    /// External KSAMPLER callback; absence of a caller callback is the default.
    fn sample_callback(
        &mut self,
        _i: i64,
        _denoised: &Tensor,
        _x: &Tensor,
        _total: i64,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn scale_latent_inpaint(
        &mut self,
        p: Parameterization,
        _x: &Tensor,
        sigma: &Tensor,
        noise: &Tensor,
        latent: &Tensor,
        _mask: &Tensor,
    ) -> Result<Tensor, Error> {
        let mut shape = vec![sigma.size()[0]];
        shape.resize(noise.dim(), 1);
        p.noise_scaling(&sigma.f_reshape(shape.as_slice())?, noise, latent, false)
    }
    /// Optional metadata observer used to validate source local ownership.
    fn source_locals(
        &mut self,
        _locals: &super::locals::SourceLocals,
        _event: &str,
    ) -> Result<(), Error> {
        Ok(())
    }
    /// cfg_pp requires the source post-CFG capture and disables CFG1 skipping.
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg_pp: bool) -> Result<Prediction, Error>;
    fn denoise_with_capture(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        disable_cfg1: bool,
    ) -> Result<Prediction, Error> {
        if disable_cfg1 {
            self.denoise(x, sigma, true)
        } else {
            Err(Error::BindingGap {
                backend: "sampler_post_cfg".into(),
                operation: "capture preserving CFG1 optimization".into(),
            })
        }
    }
    fn noise(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        next: &Tensor,
        kind: NoiseKind,
    ) -> Result<Tensor, Error>;
    fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error>;
    fn callback_values(
        &mut self,
        fields: &[(String, crate::execution_value::Value)],
    ) -> Result<(), Error> {
        let get = |key: &str| {
            fields
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::BindingGap {
                    backend: "sampler_callback".into(),
                    operation: format!("optional-field callback {key}"),
                })
        };
        let x = get("x")?.as_tensor()?;
        let sigma = get("sigma")?.as_tensor()?;
        let hat = get("sigma_hat")?.as_tensor()?;
        let denoised = get("denoised")?.as_tensor()?;
        let crate::execution_value::Value::Int(i) = get("i")? else {
            return Err(Error::Native("Invalid callback step index".into()));
        };
        self.callback(EulerStep {
            x: &x,
            i: *i,
            sigma: &sigma,
            sigma_hat: &hat,
            denoised: &denoised,
        })
    }
    fn eps_callback(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn make_brownian(
        &mut self,
        _x: &Tensor,
        _min: &Tensor,
        _max: &Tensor,
        _cpu: bool,
    ) -> Result<super::brownian::BrownianNoise, Error> {
        Err(Error::BindingGap {
            backend: "sampler_noise".into(),
            operation: "Brownian generator/seed binding".into(),
        })
    }
    fn observe_noise(
        &mut self,
        _noise: &Tensor,
        _sigma: &Tensor,
        _next: &Tensor,
        _kind: NoiseKind,
    ) -> Result<(), Error> {
        Ok(())
    }
    fn warning(&mut self, _message: &str) {}
    fn percent_to_sigma(&self, _percent: f64) -> Result<f64, Error> {
        Err(Error::BindingGap {
            backend: "model_sampling".into(),
            operation: "percent_to_sigma".into(),
        })
    }
}
pub fn derivative(x: &Tensor, sigma: &Tensor, denoised: &Tensor) -> Result<Tensor, Error> {
    // k_diffusion append_dims retains a [1] sigma, unlike reshape_sigma.
    let mut shape = sigma.size();
    shape.resize(x.dim(), 1);
    Ok(x.f_sub(denoised)?.f_div(&sigma.f_reshape(&shape)?)?)
}
fn uncond(p: &Prediction) -> Result<&Tensor, Error> {
    p.uncond_denoised.as_ref().ok_or_else(|| Error::BindingGap {
        backend: "sampler_post_cfg".into(),
        operation: "uncond_denoised capture".into(),
    })
}
fn scalar(t: &Tensor) -> f64 {
    t.double_value(&[])
}
pub enum NoiseScale {
    Scalar(f64),
    Tensor(Tensor),
}
impl NoiseScale {
    fn number(&self) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Tensor(value) => scalar(value),
        }
    }
    pub(super) fn retain(self, frame: &mut SourceLocals, name: &str) -> Self {
        match self {
            Self::Tensor(value) => Self::Tensor(frame.keep(name, value)),
            Self::Scalar(value) => {
                frame.remove(name);
                Self::Scalar(value)
            }
        }
    }
}
pub(super) trait MultiplyNoiseScale {
    fn f_mul_noise_scale(&self, scale: &NoiseScale) -> Result<Tensor, Error>;
}
impl MultiplyNoiseScale for Tensor {
    fn f_mul_noise_scale(&self, scale: &NoiseScale) -> Result<Tensor, Error> {
        Ok(match scale {
            NoiseScale::Scalar(value) => self.f_mul_scalar(*value)?,
            NoiseScale::Tensor(value) => self.f_mul(value)?,
        })
    }
}
pub fn ancestral(from: &Tensor, to: &Tensor, eta: f64) -> Result<(Tensor, NoiseScale), Error> {
    if eta == 0. {
        return Ok((to.shallow_clone(), NoiseScale::Scalar(0.)));
    }
    let up = to
        .f_pow_tensor_scalar(2)?
        .f_mul(
            &from
                .f_pow_tensor_scalar(2)?
                .f_sub(&to.f_pow_tensor_scalar(2)?)?,
        )?
        .f_div(&from.f_pow_tensor_scalar(2)?)?
        .f_pow_tensor_scalar(0.5)?
        .f_mul_scalar(eta)?;
    // Python min keeps its first argument when the comparison is unordered.
    let up = if scalar(&up) < scalar(to) {
        up
    } else {
        to.shallow_clone()
    };
    let down = to
        .f_pow_tensor_scalar(2)?
        .f_sub(&up.f_pow_tensor_scalar(2)?)?
        .f_pow_tensor_scalar(0.5)?;
    Ok((down, NoiseScale::Tensor(up)))
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChurnSolver {
    Euler,
    Heun,
    Dpm2,
    HeunPP2,
}
pub fn churn(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    settings: EulerSettings,
    solver: ChurnSolver,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::named(match solver {
        ChurnSolver::Euler => "sample_euler",
        ChurnSolver::Heun => "sample_heun",
        ChurnSolver::Dpm2 => "sample_dpm_2",
        ChurnSolver::HeunPP2 => "sample_heunpp2",
    })
    .decorated(&[&x, sigmas]);
    let result = (|| {
        x = frame.keep("x", x);
        let _ = frame.keep("sigmas", sigmas.shallow_clone());
        let steps = sigmas.size()[0] - 1;
        let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
        let end = if solver == ChurnSolver::HeunPP2 {
            Some(frame.keep("s_end", sigmas.f_get(-1)?))
        } else {
            None
        };
        for i in 0..steps {
            let sigma = sigmas.f_get(i)?;
            let next = sigmas.f_get(i + 1)?;
            let gamma = if (settings.churn > 0. || solver == ChurnSolver::HeunPP2)
                && settings.tmin <= scalar(&sigma)
                && scalar(&sigma) <= settings.tmax
            {
                (settings.churn / steps as f64).min(2_f64.sqrt() - 1.)
            } else {
                0.
            };
            let eps = if solver == ChurnSolver::HeunPP2 {
                Some(
                    frame.keep(
                        "eps",
                        exec.noise(&x, &sigma, &next, NoiseKind::GlobalLike)?
                            .f_mul_scalar(settings.noise)?,
                    ),
                )
            } else {
                None
            };
            let hat = frame.keep(
                "sigma_hat",
                if settings.churn > 0. || matches!(solver, ChurnSolver::Heun | ChurnSolver::HeunPP2)
                {
                    sigma.f_mul_scalar(gamma + 1.)?
                } else {
                    sigma.shallow_clone()
                },
            );
            if gamma > 0. {
                let eps = if let Some(eps) = eps {
                    eps
                } else {
                    frame.keep(
                        "eps",
                        exec.noise(&x, &sigma, &next, NoiseKind::GlobalLike)?
                            .f_mul_scalar(settings.noise)?,
                    )
                };
                x = frame.keep(
                    "x",
                    x.f_add(
                        &eps.f_mul(
                            &hat.f_pow_tensor_scalar(2)?
                                .f_sub(&sigma.f_pow_tensor_scalar(2)?)?
                                .f_pow_tensor_scalar(0.5)?,
                        )?,
                    )?,
                );
            }
            let model_sigma = hat.f_mul(&ones)?;
            let denoised = frame.keep(
                "denoised",
                frame.denoise(exec, &x, &model_sigma, false)?.denoised,
            );
            drop(model_sigma);
            let d = frame.keep("d", derivative(&x, &hat, &denoised)?);
            frame.callback(
                exec,
                EulerStep {
                    x: &x,
                    i,
                    sigma: &sigma,
                    sigma_hat: &hat,
                    denoised: &denoised,
                },
            )?;
            let use_euler = solver == ChurnSolver::Euler
                || (solver != ChurnSolver::HeunPP2 && scalar(&next) == 0.)
                || end.as_ref().is_some_and(|end| scalar(&next) == scalar(end));
            if solver == ChurnSolver::Dpm2 && !use_euler {
                let mid = frame.keep(
                    "sigma_mid",
                    hat.f_log()?.f_lerp(&next.f_log()?, 0.5)?.f_exp()?,
                );
                let dt1 = frame.keep("dt_1", mid.f_sub(&hat)?);
                let dt2 = frame.keep("dt_2", next.f_sub(&hat)?);
                let x2 = frame.keep("x_2", x.f_add(&d.f_mul(&dt1)?)?);
                let model_sigma = mid.f_mul(&ones)?;
                let denoised2 = frame.keep(
                    "denoised_2",
                    frame.denoise(exec, &x2, &model_sigma, false)?.denoised,
                );
                drop(model_sigma);
                let d2 = frame.keep("d_2", derivative(&x2, &mid, &denoised2)?);
                x = frame.keep("x", x.f_add(&d2.f_mul(&dt2)?)?);
                continue;
            }
            let dt = frame.keep("dt", next.f_sub(&hat)?);
            if use_euler {
                x = frame.keep("x", x.f_add(&d.f_mul(&dt)?)?);
                continue;
            }
            let x2 = frame.keep("x_2", x.f_add(&d.f_mul(&dt)?)?);
            let model_sigma = next.f_mul(&ones)?;
            let denoised2 = frame.keep(
                "denoised_2",
                frame.denoise(exec, &x2, &model_sigma, false)?.denoised,
            );
            drop(model_sigma);
            let d2 = frame.keep("d_2", derivative(&x2, &next, &denoised2)?);
            let dp = if solver == ChurnSolver::Heun {
                d.f_add(&d2)?.f_div_scalar(2)?
            } else {
                let next2 = sigmas.f_get(i + 2)?;
                if scalar(&next2) == scalar(end.as_ref().unwrap()) {
                    let w = frame.keep("w", sigmas.f_get(0)?.f_mul_scalar(2)?);
                    let w2 = frame.keep("w2", next.f_div(&w)?);
                    let w1 = frame.keep("w1", w2.f_rsub_scalar(1)?);
                    d.f_mul(&w1)?.f_add(&d2.f_mul(&w2)?)?
                } else {
                    let dt2 = frame.keep("dt_2", next2.f_sub(&next)?);
                    let x3 = frame.keep("x_3", x2.f_add(&d2.f_mul(&dt2)?)?);
                    let model_sigma = next2.f_mul(&ones)?;
                    let denoised3 = frame.keep(
                        "denoised_3",
                        frame.denoise(exec, &x3, &model_sigma, false)?.denoised,
                    );
                    drop(model_sigma);
                    let d3 = frame.keep("d_3", derivative(&x3, &next2, &denoised3)?);
                    let w = frame.keep("w", sigmas.f_get(0)?.f_mul_scalar(3)?);
                    let w2 = frame.keep("w2", next.f_div(&w)?);
                    let w3 = frame.keep("w3", next2.f_div(&w)?);
                    let w1 = frame.keep("w1", w2.f_rsub_scalar(1)?.f_sub(&w3)?);
                    w1.f_mul(&d)?
                        .f_add(&w2.f_mul(&d2)?)?
                        .f_add(&w3.f_mul(&d3)?)?
                }
            };
            let dp = frame.keep("d_prime", dp);
            x = frame.keep("x", x.f_add(&dp.f_mul(&dt)?)?);
        }
        frame.finish(exec, x)
    })();
    result.map_err(|error| frame.retain_error(error))
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AncestralSolver {
    Euler,
    Dpm2,
}
pub fn sample_ancestral(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    eta: f64,
    noise: f64,
    solver: AncestralSolver,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = if model.is_flow() {
        noise * model.noise_scale
    } else {
        noise
    };
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let denoised = frame.keep(
            "denoised",
            frame
                .denoise(exec, &x, &sigma.f_mul(&ones)?, false)?
                .denoised,
        );
        // RF Euler computes no ancestral coefficients before its callback.
        if model.is_flow() && solver == AncestralSolver::Euler {
            frame.callback(
                exec,
                EulerStep {
                    x: &x,
                    i,
                    sigma: &sigma,
                    sigma_hat: &sigma,
                    denoised: &denoised,
                },
            )?;
            if scalar(&next) == 0. {
                x = frame.keep("x", denoised);
            } else {
                let ratio = frame.keep(
                    "downstep_ratio",
                    next.f_div(&sigma)?
                        .f_sub_scalar(1)?
                        .f_mul_scalar(eta)?
                        .f_add_scalar(1)?,
                );
                let down = frame.keep("sigma_down", next.f_mul(&ratio)?);
                let aip1 = frame.keep("alpha_ip1", next.f_rsub_scalar(1)?);
                let adown = frame.keep("alpha_down", down.f_rsub_scalar(1)?);
                let coeff = frame.keep(
                    "renoise_coeff",
                    next.f_pow_tensor_scalar(2)?
                        .f_sub(
                            &down
                                .f_pow_tensor_scalar(2)?
                                .f_mul(&aip1.f_pow_tensor_scalar(2)?)?
                                .f_div(&adown.f_pow_tensor_scalar(2)?)?,
                        )?
                        .f_pow_tensor_scalar(0.5)?,
                );
                let ratio = frame.keep("sigma_down_i_ratio", down.f_div(&sigma)?);
                x = frame.keep(
                    "x",
                    ratio
                        .f_mul(&x)?
                        .f_add(&ratio.f_rsub_scalar(1)?.f_mul(&denoised)?)?,
                );
                if eta > 0. {
                    x = frame.keep(
                        "x",
                        aip1.f_div(&adown)?.f_mul(&x)?.f_add(
                            &exec
                                .noise(&x, &sigma, &next, NoiseKind::Independent)?
                                .f_mul_scalar(noise)?
                                .f_mul(&coeff)?,
                        )?,
                    );
                }
            }
            continue;
        }
        let (down, up, alphas) = if model.is_flow() {
            let ratio = frame.keep(
                "downstep_ratio",
                next.f_div(&sigma)?
                    .f_sub_scalar(1)?
                    .f_mul_scalar(eta)?
                    .f_add_scalar(1)?,
            );
            let down = frame.keep("sigma_down", next.f_mul(&ratio)?);
            let aip1 = frame.keep("alpha_ip1", next.f_rsub_scalar(1)?);
            let adown = frame.keep("alpha_down", down.f_rsub_scalar(1)?);
            let coeff = frame.keep(
                "renoise_coeff",
                next.f_pow_tensor_scalar(2)?
                    .f_sub(
                        &down
                            .f_pow_tensor_scalar(2)?
                            .f_mul(&aip1.f_pow_tensor_scalar(2)?)?
                            .f_div(&adown.f_pow_tensor_scalar(2)?)?,
                    )?
                    .f_pow_tensor_scalar(0.5)?,
            );
            (down, NoiseScale::Tensor(coeff), Some((aip1, adown)))
        } else {
            let (down, up) = ancestral(&sigma, &next, eta)?;
            (
                frame.keep("sigma_down", down),
                up.retain(&mut frame, "sigma_up"),
                None,
            )
        };
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &denoised,
            },
        )?;
        if solver == AncestralSolver::Euler {
            if scalar(&down) == 0. {
                x = frame.keep("x", denoised);
            } else {
                let d = frame.keep("d", derivative(&x, &sigma, &denoised)?);
                let dt = frame.keep("dt", down.f_sub(&sigma)?);
                x = frame.keep(
                    "x",
                    x.f_add(&d.f_mul(&dt)?)?.f_add(
                        &exec
                            .noise(&x, &sigma, &next, NoiseKind::Independent)?
                            .f_mul_scalar(noise)?
                            .f_mul_noise_scale(&up)?,
                    )?,
                );
            }
        } else {
            let d = frame.keep("d", derivative(&x, &sigma, &denoised)?);
            if scalar(&down) == 0. {
                let dt = frame.keep("dt", down.f_sub(&sigma)?);
                x = frame.keep("x", x.f_add(&d.f_mul(&dt)?)?);
            } else {
                let mid = frame.keep(
                    "sigma_mid",
                    sigma.f_log()?.f_lerp(&down.f_log()?, 0.5)?.f_exp()?,
                );
                let dt1 = frame.keep("dt_1", mid.f_sub(&sigma)?);
                let dt2 = frame.keep("dt_2", down.f_sub(&sigma)?);
                let x2 = frame.keep("x_2", x.f_add(&d.f_mul(&dt1)?)?);
                let denoised2 = frame.keep(
                    "denoised_2",
                    frame
                        .denoise(exec, &x2, &mid.f_mul(&ones)?, false)?
                        .denoised,
                );
                let d2 = frame.keep("d_2", derivative(&x2, &mid, &denoised2)?);
                x = frame.keep("x", x.f_add(&d2.f_mul(&dt2)?)?);
                x = frame.keep(
                    "x",
                    if let Some((aip1, adown)) = alphas {
                        aip1.f_div(&adown)?.f_mul(&x)?.f_add(
                            &exec
                                .noise(&x, &sigma, &next, NoiseKind::Independent)?
                                .f_mul_scalar(noise)?
                                .f_mul_noise_scale(&up)?,
                        )?
                    } else {
                        x.f_add(
                            &exec
                                .noise(&x, &sigma, &next, NoiseKind::Independent)?
                                .f_mul_scalar(noise)?
                                .f_mul_noise_scale(&up)?,
                        )?
                    },
                );
            }
        }
    }
    frame.finish(exec, x)
}
pub fn dpmpp_2m(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    cfg_pp: bool,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let mut old: Option<Tensor> = None;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, cfg_pp)?;
        let denoised = frame.keep("denoised", p.denoised.shallow_clone());
        if cfg_pp {
            let _ = frame.keep("uncond_denoised", uncond(&p)?.shallow_clone());
        }
        frame.callback(
            exec,
            crate::models::qwen_image21::EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &denoised,
            },
        )?;
        let t = frame.keep("t", sigma.f_log()?.f_neg()?);
        let tn = frame.keep("t_next", next.f_log()?.f_neg()?);
        let h = frame.keep("h", tn.f_sub(&t)?);
        if cfg_pp {
            let mix = if let Some(old) = &old
                && scalar(&next) != 0.
            {
                let last = frame.keep("h_last", t.f_sub(&sigmas.f_get(i - 1)?.f_log()?.f_neg()?)?);
                let r = frame.keep("r", last.f_div(&h)?);
                h.f_neg()?.f_exp()?.f_neg()?.f_mul(uncond(&p)?)?.f_sub(
                    &h.f_neg()?
                        .f_expm1()?
                        .f_mul(&r.f_mul_scalar(2)?.f_reciprocal()?)?
                        .f_mul(&denoised.f_sub(old)?)?,
                )?
            } else {
                h.f_neg()?.f_exp()?.f_neg()?.f_mul(uncond(&p)?)?
            };
            let mix = frame.keep("denoised_mix", mix);
            x = frame.keep(
                "x",
                denoised
                    .f_add(&mix)?
                    .f_add(&h.f_neg()?.f_exp()?.f_mul(&x)?)?,
            );
            old = Some(uncond(&p)?.shallow_clone());
            let _ = frame.keep("old_uncond_denoised", old.as_ref().unwrap().shallow_clone());
        } else {
            let predicted = if let Some(old) = &old {
                if scalar(&next) != 0. {
                    let last =
                        frame.keep("h_last", t.f_sub(&sigmas.f_get(i - 1)?.f_log()?.f_neg()?)?);
                    let r = frame.keep("r", last.f_div(&h)?);
                    let factor = r.f_mul_scalar(2)?.f_reciprocal()?;
                    factor
                        .f_add_scalar(1)?
                        .f_mul(&denoised)?
                        .f_sub(&factor.f_mul(old)?)?
                } else {
                    denoised.shallow_clone()
                }
            } else {
                denoised.shallow_clone()
            };
            let predicted = if old.is_some() && scalar(&next) != 0. {
                frame.keep("denoised_d", predicted)
            } else {
                predicted
            };
            x = frame.keep(
                "x",
                tn.f_neg()?
                    .f_exp()?
                    .f_div(&t.f_neg()?.f_exp()?)?
                    .f_mul(&x)?
                    .f_sub(&h.f_neg()?.f_expm1()?.f_mul(&predicted)?)?,
            );
            old = Some(p.denoised);
            let _ = frame.keep("old_denoised", old.as_ref().unwrap().shallow_clone());
        }
    }
    frame.finish(exec, x)
}
pub fn gradient_estimation(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    gamma: f64,
    cfg_pp: bool,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let mut old: Option<Tensor> = None;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, cfg_pp)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        if cfg_pp {
            let _ = frame.keep("uncond_denoised", uncond(&p)?.shallow_clone());
        }
        let d = frame.keep(
            "d",
            derivative(&x, &sigma, if cfg_pp { uncond(&p)? } else { &p.denoised })?,
        );
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
        let dt = frame.keep("dt", next.f_sub(&sigma)?);
        if scalar(&next) == 0. {
            x = frame.keep("x", p.denoised);
        } else {
            x = frame.keep(
                "x",
                if cfg_pp {
                    p.denoised.f_add(&d.f_mul(&next)?)?
                } else {
                    x.f_add(&d.f_mul(&dt)?)?
                },
            );
            if let Some(old) = &old {
                let bar = frame.keep("d_bar", d.f_sub(old)?.f_mul_scalar(gamma - 1.)?);
                x = frame.keep("x", x.f_add(&bar.f_mul(&dt)?)?);
            }
        }
        old = Some(d);
        let _ = frame.keep("old_d", old.as_ref().unwrap().shallow_clone());
    }
    frame.finish(exec, x)
}

pub fn half_log_snr(sigma: &Tensor, model: Parameterization) -> Result<Tensor, Error> {
    Ok(if model.is_flow() {
        sigma.f_logit(None)?.f_neg()?
    } else {
        sigma.f_log()?.f_neg()?
    })
}
pub fn snr_to_sigma(time: &Tensor, model: Parameterization) -> Result<Tensor, Error> {
    Ok(if model.is_flow() {
        time.f_neg()?.f_sigmoid()?
    } else {
        time.f_neg()?.f_exp()?
    })
}
pub fn euler_cfg_pp(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    eta: f64,
    noise: f64,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = noise * model.noise_scale;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, true)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        let _ = frame.keep("uncond_denoised", uncond(&p)?.shallow_clone());
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        if scalar(&next) == 0. {
            x = frame.keep("x", p.denoised);
        } else {
            let alpha_s = frame.keep(
                "alpha_s",
                sigma.f_mul(&half_log_snr(&sigma, model)?.f_exp()?)?,
            );
            let alpha_t = frame.keep(
                "alpha_t",
                next.f_mul(&half_log_snr(&next, model)?.f_exp()?)?,
            );
            let d = frame.keep("d", derivative(&x, &sigma, &alpha_s.f_mul(uncond(&p)?)?)?);
            let (down, up) = ancestral(&sigma.f_div(&alpha_s)?, &next.f_div(&alpha_t)?, eta)?;
            let down = frame.keep("sigma_down", down);
            let up = up.retain(&mut frame, "sigma_up");
            let down = frame.keep("sigma_down", alpha_t.f_mul(&down)?);
            x = frame.keep("x", alpha_t.f_mul(&p.denoised)?.f_add(&down.f_mul(&d)?)?);
            if eta > 0. && noise > 0. {
                x = frame.keep(
                    "x",
                    x.f_add(
                        &alpha_t
                            .f_mul(&exec.noise(&x, &sigma, &next, NoiseKind::Independent)?)?
                            .f_mul_scalar(noise)?
                            .f_mul_noise_scale(&up)?,
                    )?,
                );
            }
        }
    }
    frame.finish(exec, x)
}
pub fn dpmpp_2s_ancestral(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    eta: f64,
    noise: f64,
    cfg_pp: bool,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = if model.is_flow() || cfg_pp {
        noise * model.noise_scale
    } else {
        noise
    };
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let mut p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, cfg_pp)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        if cfg_pp {
            let _ = frame.keep("temp[0]", uncond(&p)?.shallow_clone());
        }
        let (down, up, alpha) = if model.is_flow() && !cfg_pp {
            let ratio = frame.keep(
                "downstep_ratio",
                next.f_div(&sigma)?
                    .f_sub_scalar(1)?
                    .f_mul_scalar(eta)?
                    .f_add_scalar(1)?,
            );
            let down = frame.keep("sigma_down", next.f_mul(&ratio)?);
            let aip1 = frame.keep("alpha_ip1", next.f_rsub_scalar(1)?);
            let adown = frame.keep("alpha_down", down.f_rsub_scalar(1)?);
            let up = frame.keep(
                "renoise_coeff",
                next.f_pow_tensor_scalar(2)?
                    .f_sub(
                        &down
                            .f_pow_tensor_scalar(2)?
                            .f_mul(&aip1.f_pow_tensor_scalar(2)?)?
                            .f_div(&adown.f_pow_tensor_scalar(2)?)?,
                    )?
                    .f_pow_tensor_scalar(0.5)?,
            );
            (down, NoiseScale::Tensor(up), Some((aip1, adown)))
        } else {
            let (down, up) = ancestral(&sigma, &next, eta)?;
            (
                frame.keep("sigma_down", down),
                up.retain(&mut frame, "sigma_up"),
                None,
            )
        };
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        if scalar(&down) == 0. || (model.is_flow() && !cfg_pp && scalar(&next) == 0.) {
            let d = frame.keep(
                "d",
                derivative(&x, &sigma, if cfg_pp { uncond(&p)? } else { &p.denoised })?,
            );
            if cfg_pp {
                x = frame.keep("x", p.denoised.f_add(&d.f_mul(&down)?)?);
            } else {
                let dt = frame.keep("dt", down.f_sub(&sigma)?);
                x = frame.keep("x", x.f_add(&d.f_mul(&dt)?)?);
            }
        } else if model.is_flow() && !cfg_pp {
            let sigma_s = if scalar(&sigma) == 1. {
                None
            } else {
                let t = frame.keep("t_i", sigma.f_rsub_scalar(1)?.f_div(&sigma)?.f_log()?);
                let td = frame.keep("t_down", down.f_rsub_scalar(1)?.f_div(&down)?.f_log()?);
                let h = frame.keep("h", td.f_sub(&t)?);
                let s = frame.keep("s", t.f_add(&h.f_mul_scalar(0.5)?)?);
                Some(frame.keep(
                    "sigma_s",
                    s.f_exp()?.f_add_scalar(1)?.f_pow_tensor_scalar(-1)?,
                ))
            };
            if sigma_s.is_none() {
                frame.remove("sigma_s");
            }
            let ratio = if let Some(s) = &sigma_s {
                s.f_div(&sigma)?
            } else {
                sigma.f_reciprocal()?.f_mul_scalar(0.9999)?
            };
            let ratio = frame.keep("sigma_s_i_ratio", ratio);
            let u = frame.keep(
                "u",
                ratio
                    .f_mul(&x)?
                    .f_add(&ratio.f_rsub_scalar(1)?.f_mul(&p.denoised)?)?,
            );
            let model_sigma = if let Some(s) = sigma_s {
                s.f_mul(&ones)?
            } else {
                ones.f_mul_scalar(0.9999)?
            };
            let d = frame.keep(
                "D_i",
                frame.denoise(exec, &u, &model_sigma, false)?.denoised,
            );
            let ratio = frame.keep("sigma_down_i_ratio", down.f_div(&sigma)?);
            x = frame.keep(
                "x",
                ratio
                    .f_mul(&x)?
                    .f_add(&ratio.f_rsub_scalar(1)?.f_mul(&d)?)?,
            );
        } else {
            let t = frame.keep("t", sigma.f_log()?.f_neg()?);
            let tn = frame.keep("t_next", down.f_log()?.f_neg()?);
            let h = frame.keep("h", tn.f_sub(&t)?);
            let s = frame.keep("s", t.f_add(&h.f_mul_scalar(0.5)?)?);
            let base = if cfg_pp {
                x.f_add(&p.denoised.f_sub(uncond(&p)?)?)?
            } else {
                x.shallow_clone()
            };
            let x2 = frame.keep(
                "x_2",
                s.f_neg()?
                    .f_exp()?
                    .f_div(&t.f_neg()?.f_exp()?)?
                    .f_mul(&base)?
                    .f_sub(
                        &h.f_neg()?
                            .f_mul_scalar(0.5)?
                            .f_expm1()?
                            .f_mul(&p.denoised)?,
                    )?,
            );
            drop(base);
            let p2 = frame.denoise(exec, &x2, &s.f_neg()?.f_exp()?.f_mul(&ones)?, cfg_pp)?;
            let _ = frame.keep("denoised_2", p2.denoised.shallow_clone());
            if cfg_pp {
                let _ = frame.keep("temp[0]", uncond(&p2)?.shallow_clone());
                drop(p.uncond_denoised.take());
            }
            // The second model call replaces the captured unconditioned output.
            let base = if cfg_pp {
                x.f_add(&p.denoised.f_sub(uncond(&p2)?)?)?
            } else {
                x.shallow_clone()
            };
            x = frame.keep(
                "x",
                tn.f_neg()?
                    .f_exp()?
                    .f_div(&t.f_neg()?.f_exp()?)?
                    .f_mul(&base)?
                    .f_sub(&h.f_neg()?.f_expm1()?.f_mul(&p2.denoised)?)?,
            );
        }
        if scalar(&next) > 0. && (!model.is_flow() || cfg_pp || eta > 0.) {
            if let Some((aip1, adown)) = alpha {
                x = aip1.f_div(&adown)?.f_mul(&x)?;
            }
            x = frame.keep(
                "x",
                x.f_add(
                    &exec
                        .noise(&x, &sigma, &next, NoiseKind::Independent)?
                        .f_mul_scalar(noise)?
                        .f_mul_noise_scale(&up)?,
                )?,
            );
        }
    }
    frame.finish(exec, x)
}
pub fn res_multistep(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    eta: f64,
    noise: f64,
    cfg_pp: bool,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let noise = noise * model.noise_scale;
    let mut old_denoised: Option<Tensor> = None;
    let mut old_down: Option<Tensor> = None;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, cfg_pp)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        if cfg_pp {
            let _ = frame.keep("uncond_denoised", uncond(&p)?.shallow_clone());
        }
        let (down, up) = ancestral(&sigma, &next, eta)?;
        let down = frame.keep("sigma_down", down);
        let up = up.retain(&mut frame, "sigma_up");
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        if let (true, Some(old)) = (scalar(&down) != 0., old_denoised.as_ref()) {
            let t = frame.keep("t", sigma.f_log()?.f_neg()?);
            let to = frame.keep("t_old", old_down.as_ref().unwrap().f_log()?.f_neg()?);
            let tn = frame.keep("t_next", down.f_log()?.f_neg()?);
            let tp = frame.keep("t_prev", sigmas.f_get(i - 1)?.f_log()?.f_neg()?);
            let h = frame.keep("h", tn.f_sub(&t)?);
            let c2 = frame.keep("c2", tp.f_sub(&to)?.f_div(&h)?);
            let phi1 = frame.keep("phi1_val", h.f_neg()?.f_expm1()?.f_div(&h.f_neg()?)?);
            let phi2 = frame.keep("phi2_val", phi1.f_sub_scalar(1.)?.f_div(&h.f_neg()?)?);
            let b1 = frame.keep(
                "b1",
                phi1.f_sub(&phi2.f_div(&c2)?)?
                    .f_nan_to_num(Some(0.), None, None)?,
            );
            let b2 = frame.keep("b2", phi2.f_div(&c2)?.f_nan_to_num(Some(0.), None, None)?);
            if cfg_pp {
                x = frame.keep("x", x.f_add(&p.denoised.f_sub(uncond(&p)?)?)?);
            }
            x = frame.keep(
                "x",
                h.f_neg()?.f_exp()?.f_mul(&x)?.f_add(
                    &h.f_mul(
                        &b1.f_mul(if cfg_pp { uncond(&p)? } else { &p.denoised })?
                            .f_add(&b2.f_mul(old)?)?,
                    )?,
                )?,
            );
        } else {
            let d = frame.keep(
                "d",
                derivative(&x, &sigma, if cfg_pp { uncond(&p)? } else { &p.denoised })?,
            );
            let value = if cfg_pp {
                p.denoised.f_add(&d.f_mul(&down)?)?
            } else {
                let dt = frame.keep("dt", down.f_sub(&sigma)?);
                x.f_add(&d.f_mul(&dt)?)?
            };
            x = frame.keep("x", value);
        }
        if up.number() > 0. {
            x = frame.keep(
                "x",
                x.f_add(
                    &exec
                        .noise(&x, &sigma, &next, NoiseKind::Independent)?
                        .f_mul_scalar(noise)?
                        .f_mul_noise_scale(&up)?,
                )?,
            );
        }
        old_denoised = Some(if cfg_pp {
            uncond(&p)?.shallow_clone()
        } else {
            p.denoised
        });
        let _ = frame.keep(
            "old_denoised",
            old_denoised.as_ref().unwrap().shallow_clone(),
        );
        old_down = Some(down);
        let _ = frame.keep("old_sigma_down", old_down.as_ref().unwrap().shallow_clone());
    }
    frame.finish(exec, x)
}
#[derive(Clone, Copy, Debug)]
pub struct LcmSettings {
    pub noise: f64,
    pub noise_end: Option<f64>,
    pub noise_clip_std: f64,
}
impl Default for LcmSettings {
    fn default() -> Self {
        Self {
            noise: 1.,
            noise_end: None,
            noise_clip_std: 0.,
        }
    }
}
pub fn lcm(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: LcmSettings,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let steps = (sigmas.size()[0] - 1).max(1);
    let end = settings.noise_end.unwrap_or(settings.noise);
    for i in 0..steps {
        let sigma = sigmas.f_get(i)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        x = frame.keep("x", p.denoised);
        let next = sigmas.f_get(i + 1)?;
        if scalar(&next) > 0. {
            let mut noise = frame.keep(
                "noise",
                exec.noise(&x, &sigma, &next, NoiseKind::Independent)?,
            );
            if settings.noise_clip_std > 0. {
                let clip = frame.keep(
                    "clip_val",
                    noise.f_std(true)?.f_mul_scalar(settings.noise_clip_std)?,
                );
                noise = frame.keep(
                    "noise",
                    noise.f_clamp_tensor(Some(&clip.f_neg()?), Some(&clip))?,
                );
            }
            let t = if steps > 1 {
                i as f64 / (steps - 1) as f64
            } else {
                0.
            };
            let scale = settings.noise + (end - settings.noise) * t;
            if scale != 1. {
                noise = frame.keep("noise", noise.f_mul_scalar(scale)?);
            }
            x = frame.keep("x", model.noise_scaling(&next, &noise, &x, false)?);
        }
    }
    frame.finish(exec, x)
}
pub fn ddpm(exec: &mut dyn Execution, mut x: Tensor, sigmas: &Tensor) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, false)?;
        let denoised = frame.keep("denoised", p.denoised);
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &denoised,
            },
        )?;
        let mut next_x = {
            let normalized =
                x.f_div(&sigma.f_pow_tensor_scalar(2.)?.f_add_scalar(1.)?.f_sqrt()?)?;
            let noise = x.f_sub(&denoised)?.f_div(&sigma)?;
            ddpm_step(exec, &x, normalized, &sigma, &next, noise)?
        };
        if scalar(&next) != 0. {
            let _ = next_x.f_mul_(&next.f_pow_tensor_scalar(2.)?.f_add_scalar(1.)?.f_sqrt()?)?;
        }
        x = frame.keep("x", next_x);
    }
    frame.finish(exec, x)
}
fn ddpm_step(
    exec: &mut dyn Execution,
    template: &Tensor,
    x: Tensor,
    sigma: &Tensor,
    next: &Tensor,
    noise: Tensor,
) -> Result<Tensor, Error> {
    let ac = sigma
        .f_mul(sigma)?
        .f_add_scalar(1)?
        .f_reciprocal()?
        .f_mul_scalar(1)?;
    let ap = next
        .f_mul(next)?
        .f_add_scalar(1)?
        .f_reciprocal()?
        .f_mul_scalar(1)?;
    let alpha = ac.f_div(&ap)?;
    let mut mu = alpha.f_reciprocal()?.f_mul_scalar(1.)?.f_sqrt()?.f_mul(
        &x.f_sub(
            &alpha
                .f_rsub_scalar(1)?
                .f_mul(&noise)?
                .f_div(&ac.f_rsub_scalar(1)?.f_sqrt()?)?,
        )?,
    )?;
    if scalar(next) > 0. {
        let _ = mu.f_add_(
            &alpha
                .f_rsub_scalar(1)?
                .f_mul(&ap.f_rsub_scalar(1.)?)?
                .f_div(&ac.f_rsub_scalar(1.)?)?
                .f_sqrt()?
                .f_mul(&exec.noise(template, sigma, next, NoiseKind::Independent)?)?,
        )?;
    }
    Ok(mu)
}
pub fn ipndm(
    exec: &mut dyn Execution,
    x: Tensor,
    sigmas: &Tensor,
    max_order: i64,
) -> Result<Tensor, Error> {
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    let _ = frame.keep("x", x.shallow_clone());
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let mut next_x = frame.keep("x_next", x.shallow_clone());
    let mut history: Vec<Tensor> = Vec::new();
    for i in 0..sigmas.size()[0] - 1 {
        let t = frame.keep("t_cur", sigmas.f_get(i)?);
        let next = frame.keep("t_next", sigmas.f_get(i + 1)?);
        let current = frame.keep("x_cur", next_x.shallow_clone());
        let p = frame.denoise(exec, &current, &t.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        // Original callback retains the initial x for IPNDM/DEIS.
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &t,
                sigma_hat: &t,
                denoised: &p.denoised,
            },
        )?;
        let d = frame.keep("d_cur", current.f_sub(&p.denoised)?.f_div(&t)?);
        let order = max_order.min(i + 1);
        let len = history.len();
        let dt = next.f_sub(&t)?;
        if scalar(&next) == 0. {
            next_x = p.denoised;
        } else {
            next_x = match order {
                1 => current.f_add(&dt.f_mul(&d)?)?,
                2 => current.f_add(
                    &dt.f_mul(&d.f_mul_scalar(3)?.f_sub(&history[len - 1])?)?
                        .f_div_scalar(2)?,
                )?,
                3 => current.f_add(
                    &dt.f_mul(
                        &d.f_mul_scalar(23)?
                            .f_sub(&history[len - 1].f_mul_scalar(16)?)?
                            .f_add(&history[len - 2].f_mul_scalar(5)?)?,
                    )?
                    .f_div_scalar(12)?,
                )?,
                4 => current.f_add(
                    &dt.f_mul(
                        &d.f_mul_scalar(55)?
                            .f_sub(&history[len - 1].f_mul_scalar(59)?)?
                            .f_add(&history[len - 2].f_mul_scalar(37)?)?
                            .f_sub(&history[len - 3].f_mul_scalar(9)?)?,
                    )?
                    .f_div_scalar(24)?,
                )?,
                _ => next_x,
            };
        }
        if len as i64 == max_order - 1 {
            if len == 0 {
                return Err(Error::ReferenceException {
                    class: "IndexError",
                    message: "list assignment index out of range".into(),
                });
            }
            let _ = history.remove(0);
            history.push(d);
        } else {
            history.push(d);
        }
        let _ = frame.keep("x_next", next_x.shallow_clone());
        frame.keep_list("buffer_model", &history);
    }
    frame.finish(exec, next_x)
}
pub fn lms(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    order: i64,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    let _ = frame.keep("x", x.shallow_clone());
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    // The original exports the sigma buffer before making any model call.
    if sigmas.kind() == tch::Kind::BFloat16 {
        return Err(Error::ReferenceException {
            class: "TypeError",
            message: "Got unsupported ScalarType BFloat16".into(),
        });
    }
    let cpu = frame.keep(
        "sigmas_cpu",
        sigmas.f_detach()?.f_to_device(tch::Device::Cpu)?,
    );
    let mut history: Vec<Tensor> = Vec::new();
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        let p = frame.denoise(exec, &x, &sigma.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        history.push(frame.keep("d", derivative(&x, &sigma, &p.denoised)?));
        if history.len() as i64 > order {
            let _ = history.remove(0);
        }
        frame.keep_list("ds", &history);
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        if scalar(&next) == 0. {
            x = p.denoised;
        } else {
            let current_order = (i + 1).min(order).max(0) as usize;
            let mut sum: Option<Tensor> = None;
            for (j, d) in history.iter().rev().take(current_order).enumerate() {
                let c = super::quadrature::coefficient(current_order, &cpu, i as usize, j)?;
                if let Some(message) = c.warning() {
                    warning(message);
                }
                let term = d.f_mul_scalar(c.value)?;
                sum = Some(if let Some(sum) = sum {
                    sum.f_add(&term)?
                } else {
                    term.f_add_scalar(0)?
                });
            }
            x = if let Some(sum) = sum {
                x.f_add(&sum)?
            } else {
                x.f_add_scalar(0)?
            };
        }
        let _ = frame.keep("x", x.shallow_clone());
    }
    frame.finish(exec, x)
}
pub fn ipndm_v(
    exec: &mut dyn Execution,
    x: Tensor,
    sigmas: &Tensor,
    max_order: i64,
) -> Result<Tensor, Error> {
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    let _ = frame.keep("x", x.shallow_clone());
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let mut next_x = frame.keep("x_next", x.shallow_clone());
    let _ = frame.keep("t_steps", sigmas.shallow_clone());
    let mut history: Vec<Tensor> = Vec::new();
    for i in 0..sigmas.size()[0] - 1 {
        let t = frame.keep("t_cur", sigmas.f_get(i)?);
        let next = frame.keep("t_next", sigmas.f_get(i + 1)?);
        let current = frame.keep("x_cur", next_x.shallow_clone());
        let p = frame.denoise(exec, &current, &t.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &t,
                sigma_hat: &t,
                denoised: &p.denoised,
            },
        )?;
        let d = frame.keep("d_cur", current.f_sub(&p.denoised)?.f_div(&t)?);
        let order = max_order.min(i + 1);
        let dt = next.f_sub(&t)?;
        if next.double_value(&[]) == 0. {
            next_x = p.denoised;
        } else if order == 1 {
            next_x = current.f_add(&dt.f_mul(&d)?)?;
        } else if (2..=4).contains(&order) {
            let coefficients = super::ipndm_v_formulas::coefficients(&mut frame, sigmas, i, order)?;
            let mut sum = coefficients[0].f_mul(&d)?;
            for (c, previous) in coefficients[1..].iter().zip(history.iter().rev()) {
                sum = sum.f_add(&c.f_mul(previous)?)?;
            }
            next_x = current.f_add(&dt.f_mul(&sum)?)?;
        }
        if history.len() as i64 == max_order - 1 {
            if history.is_empty() {
                return Err(Error::ReferenceException {
                    class: "IndexError",
                    message: "list assignment index out of range".into(),
                });
            }
            let _ = history.remove(0);
            history.push(d.f_detach()?);
        } else {
            history.push(d.f_detach()?);
        }
        let _ = frame.keep("x_next", next_x.shallow_clone());
        frame.keep_list("buffer_model", &history);
    }
    frame.finish(exec, next_x)
}
#[derive(Clone, Copy, Debug)]
pub struct CfgHistory {
    pub history_weight: f64,
    pub zero_weight: Option<f64>,
    pub zero_order: i64,
    pub uncond_history_weight: f64,
}
impl CfgHistory {
    pub fn ud10_ab() -> Self {
        Self {
            history_weight: 0.25,
            zero_weight: Some(1.),
            zero_order: 1,
            uncond_history_weight: 0.1,
        }
    }
}
/// Registered UD10 is an undecorated wrapper whose x argument remains owned
/// while the shared history body replaces its own x local.
pub fn cfgpp_ud10_ab(
    exec: &mut dyn Execution,
    x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    cfgpp_history(
        exec,
        x.shallow_clone(),
        sigmas,
        model,
        CfgHistory::ud10_ab(),
        warning,
    )
}
pub fn cfgpp_history(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: CfgHistory,
    warning: &mut dyn FnMut(&str),
) -> Result<Tensor, Error> {
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    if sigmas.kind() == tch::Kind::BFloat16 {
        return Err(Error::ReferenceException {
            class: "TypeError",
            message: "Got unsupported ScalarType BFloat16".into(),
        });
    }
    let cpu = frame.keep(
        "sigmas_cpu",
        sigmas.f_detach()?.f_to_device(tch::Device::Cpu)?,
    );
    let mut derivatives: Vec<Tensor> = Vec::new();
    let mut denoised_history: Vec<Tensor> = Vec::new();
    let mut old_uncond: Option<Tensor> = None;
    for i in 0..sigmas.size()[0] - 1 {
        let sigma = sigmas.f_get(i)?;
        let next = sigmas.f_get(i + 1)?;
        exec.source_locals(&frame, "model")?;
        let p = exec.denoise_with_capture(&x, &sigma.f_mul(&ones)?, false)?;
        let _ = frame.keep("denoised", p.denoised.shallow_clone());
        let captured = p.uncond_denoised.as_ref().or(p.cond_denoised.as_ref());
        if let Some(captured) = captured {
            let _ = frame.keep("uncond_denoised", captured.shallow_clone());
        }
        frame.callback(
            exec,
            EulerStep {
                x: &x,
                i,
                sigma: &sigma,
                sigma_hat: &sigma,
                denoised: &p.denoised,
            },
        )?;
        let captured = captured.ok_or_else(|| Error::BindingGap {
            backend: "sampler_post_cfg".into(),
            operation: "uncond/cond fallback capture".into(),
        })?;
        let alpha_s = frame.keep(
            "alpha_s",
            sigma.f_mul(&half_log_snr(&sigma, model)?.f_exp()?)?,
        );
        let alpha_t = frame.keep(
            "alpha_t",
            if next.double_value(&[]) != 0. {
                next.f_mul(&half_log_snr(&next, model)?.f_exp()?)?
            } else {
                Tensor::f_ones([], (next.kind(), next.device()))?
            },
        );
        let current_uncond = frame.keep(
            "current_uncond_d",
            derivative(&x, &sigma, &alpha_s.f_mul(captured)?)?,
        );
        let mut uncond = frame.keep("uncond_d", current_uncond.shallow_clone());
        let dt = frame.keep("dt", next.f_sub(&sigma)?);
        if i > 0 && settings.uncond_history_weight != 0. {
            let ratio = frame.keep(
                "step_ratio",
                dt.f_div(&sigma.f_sub(&sigmas.f_get(i - 1)?)?)?,
            );
            uncond = frame.keep(
                "uncond_d",
                uncond.f_add(
                    &ratio
                        .f_mul_scalar(settings.uncond_history_weight)?
                        .f_mul(&current_uncond.f_sub(old_uncond.as_ref().unwrap())?)?,
                )?,
            );
        }
        let euler = frame.keep(
            "euler_step",
            alpha_t
                .f_mul(&p.denoised)?
                .f_add(&next.f_mul(&uncond)?)?
                .f_sub(&x)?,
        );
        let d = frame.keep("d", euler.f_div(&dt)?);
        derivatives.push(d);
        if derivatives.len() > 2 {
            let _ = derivatives.remove(0);
        }
        frame.keep_list("derivatives", &derivatives);
        let step = if derivatives.len() == 1 {
            euler
        } else {
            let mut sum: Option<Tensor> = None;
            for (j, d) in derivatives.iter().rev().enumerate() {
                let coefficient = super::quadrature::coefficient(2, &cpu, i as usize, j)?;
                if let Some(message) = coefficient.warning() {
                    warning(message);
                }
                let term = d.f_mul_scalar(coefficient.value)?;
                sum = Some(if let Some(sum) = sum {
                    sum.f_add(&term)?
                } else {
                    term.f_add_scalar(0)?
                });
            }
            let sum = frame.keep("history_step", sum.unwrap());
            euler.f_lerp(&sum, settings.history_weight)?
        };
        let step = frame.keep("step", step);
        x = frame.keep("x", x.f_add(&step)?);
        if let Some(weight) = settings.zero_weight
            && next.double_value(&[]) == 0.
            && !denoised_history.is_empty()
        {
            let n = denoised_history.len();
            let zero = if settings.zero_order == 2 && n > 1 {
                let s0 = frame.keep("sigma_0", sigmas.f_get(i - 2)?);
                let s1 = frame.keep("sigma_1", sigmas.f_get(i - 1)?);
                let s2 = frame.keep("sigma_2", sigma.shallow_clone());
                let w0 = frame.keep(
                    "weight_0",
                    s1.f_mul(&s2)?
                        .f_div(&s0.f_sub(&s1)?.f_mul(&s0.f_sub(&s2)?)?)?,
                );
                let w1 = frame.keep(
                    "weight_1",
                    s0.f_mul(&s2)?
                        .f_div(&s1.f_sub(&s0)?.f_mul(&s1.f_sub(&s2)?)?)?,
                );
                let w2 = frame.keep(
                    "weight_2",
                    s0.f_mul(&s1)?
                        .f_div(&s2.f_sub(&s0)?.f_mul(&s2.f_sub(&s1)?)?)?,
                );
                w0.f_mul(&denoised_history[n - 2])?
                    .f_add(&w1.f_mul(&denoised_history[n - 1])?)?
                    .f_add(&w2.f_mul(&p.denoised)?)?
            } else {
                let slope = frame.keep(
                    "denoised_slope",
                    p.denoised
                        .f_sub(&denoised_history[n - 1])?
                        .f_div(&sigma.f_sub(&sigmas.f_get(i - 1)?)?)?,
                );
                p.denoised.f_sub(&sigma.f_mul(&slope)?)?
            };
            let zero = frame.keep("zero_prediction", zero);
            x = frame.keep("x", x.f_lerp(&zero, weight)?);
        }
        denoised_history.push(p.denoised);
        if denoised_history.len() > 2 {
            let _ = denoised_history.remove(0);
        }
        frame.keep_list("denoised_history", &denoised_history);
        old_uncond = Some(current_uncond);
        let _ = frame.keep("old_uncond_d", old_uncond.as_ref().unwrap().shallow_clone());
    }
    frame.finish(exec, x)
}
