//! Original DPM++ SDE families. Default Brownian ownership is local to a sampler
//! call and drops on success/error; a caller-provided custom stream stays owned
//! by its caller, matching noise_sampler= in the reference.
use super::{
    brownian::BrownianNoise,
    locals::SourceLocals,
    parameterization::Parameterization,
    solvers::{Execution, MultiplyNoiseScale, NoiseKind, ancestral, half_log_snr, snr_to_sigma},
};
use crate::native::Error;
use tch::Tensor;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Source {
    Cpu,
    Gpu,
    Custom,
    /// A supplied custom sampler invoked through the registered GPU wrapper.
    CustomGpu,
}
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub eta: f64,
    pub noise: f64,
    pub source: Source,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            eta: 1.,
            noise: 1.,
            source: Source::Cpu,
        }
    }
}
struct Noise {
    source: Source,
    tree: Option<BrownianNoise>,
    // The source keeps both scalar bounds in its function frame, even for a
    // supplied sampler. GPU entry points additionally retain wrapper bounds.
    _inner_bounds: (Tensor, Tensor),
    _wrapper_bounds: Option<(Tensor, Tensor)>,
}
fn length(sigmas: &Tensor) -> Result<i64, Error> {
    sigmas
        .size()
        .first()
        .copied()
        .ok_or_else(|| Error::ReferenceException {
            class: "TypeError",
            message: "len() of a 0-d tensor".into(),
        })
}
fn bounds(sigmas: &Tensor) -> Result<(Tensor, Tensor), Error> {
    Ok((
        sigmas.f_masked_select(&sigmas.f_gt(0)?)?.f_min()?,
        sigmas.f_max()?,
    ))
}
struct WrapperNoise {
    bounds: (Tensor, Tensor),
    tree: Option<BrownianNoise>,
}
impl Noise {
    fn wrapper(
        exec: &mut dyn Execution,
        x: &Tensor,
        sigmas: &Tensor,
        source: Source,
    ) -> Result<Option<WrapperNoise>, Error> {
        if !matches!(source, Source::Gpu | Source::CustomGpu) {
            return Ok(None);
        }
        let bounds = bounds(sigmas)?;
        let tree = if source == Source::Gpu {
            Some(exec.make_brownian(x, &bounds.0, &bounds.1, false)?)
        } else {
            None
        };
        Ok(Some(WrapperNoise { bounds, tree }))
    }
    fn new(
        exec: &mut dyn Execution,
        x: &Tensor,
        sigmas: &Tensor,
        source: Source,
        wrapper: Option<WrapperNoise>,
    ) -> Result<Self, Error> {
        let inner_bounds = bounds(sigmas)?;
        let (tree, wrapper_bounds) = if let Some(wrapper) = wrapper {
            (wrapper.tree, Some(wrapper.bounds))
        } else if matches!(source, Source::Custom | Source::CustomGpu) {
            (None, None)
        } else {
            (
                Some(exec.make_brownian(x, &inner_bounds.0, &inner_bounds.1, true)?),
                None,
            )
        };
        Ok(Self {
            source,
            tree,
            _inner_bounds: inner_bounds,
            _wrapper_bounds: wrapper_bounds,
        })
    }
    fn sample(
        &mut self,
        exec: &mut dyn Execution,
        x: &Tensor,
        from: &Tensor,
        to: &Tensor,
    ) -> Result<Tensor, Error> {
        if let Some(tree) = &mut self.tree {
            let noise = tree.sample(from, to, &mut |message| exec.warning(message))?;
            exec.observe_noise(
                &noise,
                from,
                to,
                if self.source == Source::Cpu {
                    NoiseKind::BrownianCpu
                } else {
                    NoiseKind::BrownianGpu
                },
            )?;
            Ok(noise)
        } else {
            exec.noise(x, from, to, NoiseKind::Custom)
        }
    }
}
pub(super) fn offset_first(
    exec: &dyn Execution,
    sigmas: &Tensor,
    model: Parameterization,
) -> Result<Tensor, Error> {
    if sigmas.size()[0] > 1 && model.is_flow() && sigmas.double_value(&[0]) >= 1. {
        let copy = sigmas.copy();
        let _ = copy.f_get(0)?.f_fill_(exec.percent_to_sigma(1e-4)?)?;
        Ok(copy)
    } else {
        Ok(sigmas.shallow_clone())
    }
}
pub fn dpmpp_2m_sde(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: Settings,
    solver_type: &str,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    if length(sigmas)? <= 1 {
        return frame.finish(exec, x);
    }
    let wrapper = Noise::wrapper(exec, &x, sigmas, settings.source)?;
    if !matches!(solver_type, "heun" | "midpoint") {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "solver_type must be 'heun' or 'midpoint'".into(),
        });
    }
    let mut noise = Noise::new(exec, &x, sigmas, settings.source, wrapper)?;
    let _ = frame.keep("sigma_min", noise._inner_bounds.0.shallow_clone());
    let _ = frame.keep("sigma_max", noise._inner_bounds.1.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    let noise_scale = settings.noise * model.noise_scale;
    let mut old: Option<Tensor> = None;
    let mut previous_h: Option<Tensor> = None;
    let mut h: Option<Tensor> = None;
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
            x = frame.keep("x", p.denoised.shallow_clone());
        } else {
            let ls = frame.keep("lambda_s", half_log_snr(&sigma, model)?);
            let lt = frame.keep("lambda_t", half_log_snr(&next, model)?);
            let step = frame.keep("h", lt.f_sub(&ls)?);
            let h_eta = frame.keep("h_eta", step.f_mul_scalar(settings.eta + 1.)?);
            let alpha_t = frame.keep("alpha_t", next.f_mul(&lt.f_exp()?)?);
            x = frame.keep(
                "x",
                next.f_div(&sigma)?
                    .f_mul(&step.f_neg()?.f_mul_scalar(settings.eta)?.f_exp()?)?
                    .f_mul(&x)?
                    .f_add(
                        &alpha_t
                            .f_mul(&h_eta.f_neg()?.f_expm1()?.f_neg()?)?
                            .f_mul(&p.denoised)?,
                    )?,
            );
            if let Some(old) = &old {
                let r = frame.keep("r", previous_h.as_ref().unwrap().f_div(&step)?);
                let delta = p.denoised.f_sub(old)?;
                let factor = if solver_type == "heun" {
                    alpha_t.f_mul(
                        &h_eta
                            .f_neg()?
                            .f_expm1()?
                            .f_neg()?
                            .f_div(&h_eta.f_neg()?)?
                            .f_add_scalar(1)?,
                    )?
                } else {
                    alpha_t
                        .f_mul_scalar(0.5)?
                        .f_mul(&h_eta.f_neg()?.f_expm1()?.f_neg()?)?
                };
                x = frame.keep(
                    "x",
                    x.f_add(
                        &factor
                            .f_mul(&r.f_reciprocal()?.f_mul_scalar(1)?)?
                            .f_mul(&delta)?,
                    )?,
                );
            }
            if settings.eta > 0. && noise_scale > 0. {
                x = frame.keep(
                    "x",
                    x.f_add(
                        &noise
                            .sample(exec, &x, &sigma, &next)?
                            .f_mul(&next)?
                            .f_mul(
                                &step
                                    .f_mul_scalar(-2)?
                                    .f_mul_scalar(settings.eta)?
                                    .f_expm1()?
                                    .f_neg()?
                                    .f_sqrt()?,
                            )?
                            .f_mul_scalar(noise_scale)?,
                    )?,
                );
            }
            h = Some(step);
        }
        old = Some(p.denoised);
        let _ = frame.keep("old_denoised", old.as_ref().unwrap().shallow_clone());
        previous_h = h.as_ref().map(Tensor::shallow_clone);
        if let Some(value) = &previous_h {
            let _ = frame.keep("h_last", value.shallow_clone());
        }
    }
    frame.finish(exec, x)
}
pub fn dpmpp_3m_sde(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: Settings,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    if length(sigmas)? <= 1 {
        return frame.finish(exec, x);
    }
    let wrapper = Noise::wrapper(exec, &x, sigmas, settings.source)?;
    let mut noise = Noise::new(exec, &x, sigmas, settings.source, wrapper)?;
    let _ = frame.keep("sigma_min", noise._inner_bounds.0.shallow_clone());
    let _ = frame.keep("sigma_max", noise._inner_bounds.1.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    let noise_scale = settings.noise * model.noise_scale;
    let (mut d1, mut d2): (Option<Tensor>, Option<Tensor>) = (None, None);
    let (mut h, mut h1, mut h2): (Option<Tensor>, Option<Tensor>, Option<Tensor>) =
        (None, None, None);
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
            x = frame.keep("x", p.denoised.shallow_clone());
        } else {
            let ls = frame.keep("lambda_s", half_log_snr(&sigma, model)?);
            let lt = frame.keep("lambda_t", half_log_snr(&next, model)?);
            let step = frame.keep("h", lt.f_sub(&ls)?);
            let h_eta = frame.keep("h_eta", step.f_mul_scalar(settings.eta + 1.)?);
            let alpha_t = frame.keep("alpha_t", next.f_mul(&lt.f_exp()?)?);
            x = frame.keep(
                "x",
                next.f_div(&sigma)?
                    .f_mul(&step.f_neg()?.f_mul_scalar(settings.eta)?.f_exp()?)?
                    .f_mul(&x)?
                    .f_add(
                        &alpha_t
                            .f_mul(&h_eta.f_neg()?.f_expm1()?.f_neg()?)?
                            .f_mul(&p.denoised)?,
                    )?,
            );
            if let Some(h2) = &h2 {
                let r0 = frame.keep("r0", h1.as_ref().unwrap().f_div(&step)?);
                let r1 = frame.keep("r1", h2.f_div(&step)?);
                let d10 = frame.keep("d1_0", p.denoised.f_sub(d1.as_ref().unwrap())?.f_div(&r0)?);
                let d11 = frame.keep(
                    "d1_1",
                    d1.as_ref()
                        .unwrap()
                        .f_sub(d2.as_ref().unwrap())?
                        .f_div(&r1)?,
                );
                let first = frame.keep(
                    "d1",
                    d10.f_add(&d10.f_sub(&d11)?.f_mul(&r0)?.f_div(&r0.f_add(&r1)?)?)?,
                );
                let second = frame.keep("d2", d10.f_sub(&d11)?.f_div(&r0.f_add(&r1)?)?);
                let phi2 = frame.keep(
                    "phi_2",
                    h_eta.f_neg()?.f_expm1()?.f_div(&h_eta)?.f_add_scalar(1)?,
                );
                let phi3 = frame.keep("phi_3", phi2.f_div(&h_eta)?.f_sub_scalar(0.5)?);
                x = frame.keep(
                    "x",
                    x.f_add(&alpha_t.f_mul(&phi2)?.f_mul(&first)?)?
                        .f_sub(&alpha_t.f_mul(&phi3)?.f_mul(&second)?)?,
                );
            } else if let Some(h1) = &h1 {
                let r = frame.keep("r", h1.f_div(&step)?);
                let d = frame.keep("d", p.denoised.f_sub(d1.as_ref().unwrap())?.f_div(&r)?);
                let phi2 = frame.keep(
                    "phi_2",
                    h_eta.f_neg()?.f_expm1()?.f_div(&h_eta)?.f_add_scalar(1)?,
                );
                x = frame.keep("x", x.f_add(&alpha_t.f_mul(&phi2)?.f_mul(&d)?)?);
            }
            if settings.eta > 0. && noise_scale > 0. {
                x = frame.keep(
                    "x",
                    x.f_add(
                        &noise
                            .sample(exec, &x, &sigma, &next)?
                            .f_mul(&next)?
                            .f_mul(
                                &step
                                    .f_mul_scalar(-2)?
                                    .f_mul_scalar(settings.eta)?
                                    .f_expm1()?
                                    .f_neg()?
                                    .f_sqrt()?,
                            )?
                            .f_mul_scalar(noise_scale)?,
                    )?,
                );
            }
            h = Some(step);
        }
        d2 = d1;
        d1 = Some(p.denoised);
        let _ = frame.keep("denoised_1", d1.as_ref().unwrap().shallow_clone());
        if let Some(value) = &d2 {
            let _ = frame.keep("denoised_2", value.shallow_clone());
        }
        h2 = h1;
        h1 = h.as_ref().map(Tensor::shallow_clone);
        if let Some(value) = &h1 {
            let _ = frame.keep("h_1", value.shallow_clone());
        }
        if let Some(value) = &h2 {
            let _ = frame.keep("h_2", value.shallow_clone());
        }
    }
    frame.finish(exec, x)
}
pub fn dpmpp_sde(
    exec: &mut dyn Execution,
    mut x: Tensor,
    sigmas: &Tensor,
    model: Parameterization,
    settings: Settings,
    r: f64,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, sigmas]);
    let _grad = tch::no_grad_guard();
    let mut frame = SourceLocals::default();
    x = frame.keep("x", x);
    let _ = frame.keep("sigmas", sigmas.shallow_clone());
    if length(sigmas)? <= 1 {
        return frame.finish(exec, x);
    }
    let wrapper = Noise::wrapper(exec, &x, sigmas, settings.source)?;
    let mut noise = Noise::new(exec, &x, sigmas, settings.source, wrapper)?;
    let _ = frame.keep("sigma_min", noise._inner_bounds.0.shallow_clone());
    let _ = frame.keep("sigma_max", noise._inner_bounds.1.shallow_clone());
    let ones = frame.keep("s_in", x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?);
    let sigmas = frame.keep("sigmas", offset_first(exec, sigmas, model)?);
    let noise_scale = settings.noise * model.noise_scale;
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
        } else {
            let ls = frame.keep("lambda_s", half_log_snr(&sigma, model)?);
            let lt = frame.keep("lambda_t", half_log_snr(&next, model)?);
            let h = frame.keep("h", lt.f_sub(&ls)?);
            let ls1 = frame.keep("lambda_s_1", ls.f_add(&h.f_mul_scalar(r)?)?);
            if r == 0. {
                return Err(Error::ReferenceException {
                    class: "ZeroDivisionError",
                    message: "float division by zero".into(),
                });
            }
            let factor = 1. / (2. * r);
            let ss1 = frame.keep("sigma_s_1", snr_to_sigma(&ls1, model)?);
            let alpha_s = frame.keep("alpha_s", sigma.f_mul(&ls.f_exp()?)?);
            let alpha_s1 = frame.keep("alpha_s_1", ss1.f_mul(&ls1.f_exp()?)?);
            let alpha_t = frame.keep("alpha_t", next.f_mul(&lt.f_exp()?)?);
            let (down, up) =
                ancestral(&ls.f_neg()?.f_exp()?, &ls1.f_neg()?.f_exp()?, settings.eta)?;
            let down = frame.keep("sd", down);
            let up = up.retain(&mut frame, "su");
            let lambda_s1 = frame.keep("lambda_s_1_", down.f_log()?.f_neg()?);
            let h1 = frame.keep("h_", lambda_s1.f_sub(&ls)?);
            let mut x2 = frame.keep(
                "x_2",
                alpha_s1
                    .f_div(&alpha_s)?
                    .f_mul(&h1.f_neg()?.f_exp()?)?
                    .f_mul(&x)?
                    .f_sub(
                        &alpha_s1
                            .f_mul(&h1.f_neg()?.f_expm1()?)?
                            .f_mul(&p.denoised)?,
                    )?,
            );
            if settings.eta > 0. && noise_scale > 0. {
                x2 = frame.keep(
                    "x_2",
                    x2.f_add(
                        &alpha_s1
                            .f_mul(&noise.sample(exec, &x, &sigma, &ss1)?)?
                            .f_mul_scalar(noise_scale)?
                            .f_mul_noise_scale(&up)?,
                    )?,
                );
            }
            let p2 = frame.denoise(exec, &x2, &ss1.f_mul(&ones)?, false)?;
            let _ = frame.keep("denoised_2", p2.denoised.shallow_clone());
            let (down2, up2) =
                ancestral(&ls.f_neg()?.f_exp()?, &lt.f_neg()?.f_exp()?, settings.eta)?;
            let down2 = frame.keep("sd", down2);
            drop(down);
            let up2 = up2.retain(&mut frame, "su");
            drop(up);
            let lambda_t = frame.keep("lambda_t_", down2.f_log()?.f_neg()?);
            let h2 = frame.keep("h_", lambda_t.f_sub(&ls)?);
            drop(h1);
            let predicted = frame.keep(
                "denoised_d",
                p.denoised
                    .f_mul_scalar(1. - factor)?
                    .f_add(&p2.denoised.f_mul_scalar(factor)?)?,
            );
            x = frame.keep(
                "x",
                alpha_t
                    .f_div(&alpha_s)?
                    .f_mul(&h2.f_neg()?.f_exp()?)?
                    .f_mul(&x)?
                    .f_sub(&alpha_t.f_mul(&h2.f_neg()?.f_expm1()?)?.f_mul(&predicted)?)?,
            );
            if settings.eta > 0. && noise_scale > 0. {
                x = frame.keep(
                    "x",
                    x.f_add(
                        &alpha_t
                            .f_mul(&noise.sample(exec, &x, &sigma, &next)?)?
                            .f_mul_scalar(noise_scale)?
                            .f_mul_noise_scale(&up2)?,
                    )?,
                );
            }
        }
    }
    frame.finish(exec, x)
}
