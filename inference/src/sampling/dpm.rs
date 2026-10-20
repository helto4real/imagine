//! Original DPM-Solver stages and PID adaptive acceptance/rejection history.
use super::solvers::{Execution, NoiseKind, ancestral};
use crate::{
    boxed_aten,
    execution_value::Value as V,
    native::{Error, Kitchen},
};
use tch::Tensor;

#[derive(Debug)]
pub struct Pid {
    pub h: f64,
    b: [f64; 3],
    accept_safety: f64,
    eps: f64,
    errors: Vec<f64>,
}
impl Pid {
    pub fn new(
        h: f64,
        p: f64,
        i: f64,
        d: f64,
        order: f64,
        accept_safety: f64,
        eps: f64,
    ) -> Result<Self, Error> {
        if order == 0. {
            return Err(Error::ReferenceException {
                class: "ZeroDivisionError",
                message: "float division by zero".into(),
            });
        }
        Ok(Self {
            h,
            b: [(p + i + d) / order, -(p + 2. * d) / order, d / order],
            accept_safety,
            eps,
            errors: Vec::new(),
        })
    }
    pub fn propose(&mut self, error: f64) -> Result<bool, Error> {
        if error + self.eps == 0. {
            return Err(Error::ReferenceException {
                class: "ZeroDivisionError",
                message: "float division by zero".into(),
            });
        }
        let inverse = 1. / (error + self.eps);
        if self.errors.is_empty() {
            self.errors = vec![inverse; 3];
        }
        self.errors[0] = inverse;
        let factor = self.errors[0].powf(self.b[0])
            * self.errors[1].powf(self.b[1])
            * self.errors[2].powf(self.b[2]);
        let factor = 1. + (factor - 1.).atan();
        let accept = factor >= self.accept_safety;
        if accept {
            self.errors[2] = self.errors[1];
            self.errors[1] = self.errors[0];
        }
        self.h *= factor;
        Ok(accept)
    }
}
type Cache = Vec<(&'static str, Tensor)>;
fn sigma(t: &Tensor) -> Result<Tensor, Error> {
    Ok(t.f_neg()?.f_exp()?)
}
fn eps(
    exec: &mut dyn Execution,
    cache: &mut Cache,
    key: &'static str,
    x: &Tensor,
    t: &Tensor,
) -> Result<Tensor, Error> {
    if let Some((_, value)) = cache.iter().find(|(k, _)| *k == key) {
        return Ok(value.shallow_clone());
    }
    let s = sigma(t)?;
    let ones = x.f_new_ones([x.size()[0]], (x.kind(), x.device()))?;
    let output = x
        .f_sub(&exec.denoise(x, &s.f_mul(&ones)?, false)?.denoised)?
        .f_div(&s)?;
    exec.eps_callback()?;
    cache.insert(0, (key, output.shallow_clone()));
    Ok(output)
}
#[derive(Clone, Copy, Debug)]
pub struct Stage {
    pub order: i64,
    pub r1: f64,
    pub r2: f64,
}
pub fn step(
    exec: &mut dyn Execution,
    x: &Tensor,
    t: &Tensor,
    next: &Tensor,
    stage: Stage,
    cache: &mut Vec<(&'static str, Tensor)>,
) -> Result<Tensor, Error> {
    let Stage { order, r1, r2 } = stage;
    let h = next.f_sub(t)?;
    let e = eps(exec, cache, "eps", x, t)?;
    let base = |e: &Tensor| -> Result<Tensor, Error> {
        Ok(x.f_sub(&sigma(next)?.f_mul(&h.f_expm1()?)?.f_mul(e)?)?)
    };
    if order == 1 {
        return base(&e);
    }
    let s1 = t.f_add(&h.f_mul_scalar(r1)?)?;
    let s2 = if order != 2 {
        Some(t.f_add(&h.f_mul_scalar(r2)?)?)
    } else {
        None
    };
    let u1 = x.f_sub(
        &sigma(&s1)?
            .f_mul(&h.f_mul_scalar(r1)?.f_expm1()?)?
            .f_mul(&e)?,
    )?;
    let e1 = eps(exec, cache, "eps_r1", &u1, &s1)?;
    if order == 2 {
        return Ok(base(&e)?.f_sub(
            &sigma(next)?
                .f_div_scalar(2. * r1)?
                .f_mul(&h.f_expm1()?)?
                .f_mul(&e1.f_sub(&e)?)?,
        )?);
    }
    let s2 = s2.unwrap();
    if r1 == 0. {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "float division by zero".into(),
        });
    }
    let rh = h.f_mul_scalar(r2)?;
    let u2 = x
        .f_sub(&sigma(&s2)?.f_mul(&rh.f_expm1()?)?.f_mul(&e)?)?
        .f_sub(
            &sigma(&s2)?
                .f_mul_scalar(r2 / r1)?
                .f_mul(&rh.f_expm1()?.f_div(&rh)?.f_sub_scalar(1)?)?
                .f_mul(&e1.f_sub(&e)?)?,
        )?;
    let e2 = eps(exec, cache, "eps_r2", &u2, &s2)?;
    Ok(base(&e)?.f_sub(
        &sigma(next)?
            .f_div_scalar(r2)?
            .f_mul(&h.f_expm1()?.f_div(&h)?.f_sub_scalar(1)?)?
            .f_mul(&e2.f_sub(&e)?)?,
    )?)
}
fn endpoints(
    exec: &mut dyn Execution,
    min: &Tensor,
    max: &Tensor,
) -> Result<(Tensor, Tensor), Error> {
    if min.f_le(0.)?.f_is_nonzero()? || max.f_le(0.)?.f_is_nonzero()? {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "sigma_min and sigma_max must not be 0".into(),
        });
    }
    const COPY_WARNING: &str = "To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).";
    exec.warning(COPY_WARNING);
    let start = crate::native::clone_tensor(max)?.f_log()?.f_neg()?;
    exec.warning(COPY_WARNING);
    let end = crate::native::clone_tensor(min)?.f_log()?.f_neg()?;
    Ok((start, end))
}
fn callback(
    exec: &mut dyn Execution,
    x: &Tensor,
    i: i64,
    t: &Tensor,
    up: &Tensor,
    denoised: &Tensor,
    extra: Vec<(String, V)>,
) -> Result<(), Error> {
    let mut fields = vec![
        ("sigma".into(), V::tensor(sigma(t)?)),
        ("sigma_hat".into(), V::tensor(sigma(up)?)),
        ("x".into(), V::tensor(x.shallow_clone())),
        ("i".into(), V::Int(i)),
        ("t".into(), V::tensor(t.shallow_clone())),
        ("t_up".into(), V::tensor(up.shallow_clone())),
        ("denoised".into(), V::tensor(denoised.shallow_clone())),
    ];
    fields.extend(extra);
    exec.callback_values(&fields)
}
#[derive(Clone, Copy, Debug)]
pub struct Stochastic {
    pub eta: f64,
    pub noise: f64,
}
impl Default for Stochastic {
    fn default() -> Self {
        Self { eta: 0., noise: 1. }
    }
}
pub fn fast(
    exec: &mut dyn Execution,
    native: &Kitchen,
    mut x: Tensor,
    min: &Tensor,
    max: &Tensor,
    n: i64,
    settings: Stochastic,
) -> Result<Tensor, Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, min, max]);
    let _grad = tch::no_grad_guard();
    let (start, end) = endpoints(exec, min, max)?;
    if end.double_value(&[]) <= start.double_value(&[]) && settings.eta != 0. {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "eta must be 0 for reverse sampling".into(),
        });
    }
    let m = (n as f64 / 3.).floor() as i64 + 1;
    let times = Tensor::f_linspace_tensor_tensor(
        &start,
        &end,
        m + 1,
        (native.default_dtype()?, x.device()),
    )?;
    let mut orders = if n % 3 == 0 {
        vec![3; (m - 2).max(0) as usize]
    } else {
        vec![3; (m - 1).max(0) as usize]
    };
    if n % 3 == 0 {
        orders.extend([2, 1]);
    } else {
        orders.push(n % 3);
    }
    let template = x.shallow_clone();
    for (i, order) in orders.into_iter().enumerate() {
        let t = times.f_get(i as i64)?;
        let next = times.f_get(i as i64 + 1)?;
        let (mut down, mut up) = (next.shallow_clone(), None);
        if settings.eta != 0. {
            let (d, _) = ancestral(&sigma(&t)?, &sigma(&next)?, settings.eta)?;
            down = end.f_minimum(&d.f_log()?.f_neg()?)?;
            up = Some(
                sigma(&next)?
                    .f_pow_tensor_scalar(2)?
                    .f_sub(&sigma(&down)?.f_pow_tensor_scalar(2)?)?
                    .f_pow_tensor_scalar(0.5)?,
            );
        }
        let mut cache = Vec::new();
        let e = eps(exec, &mut cache, "eps", &x, &t)?;
        let denoised = x.f_sub(&sigma(&t)?.f_mul(&e)?)?;
        callback(
            exec,
            &x,
            i as i64,
            &times.f_get(i as i64)?,
            &t,
            &denoised,
            Vec::new(),
        )?;
        x = step(
            exec,
            &x,
            &t,
            &down,
            Stage {
                order,
                r1: if order == 3 { 1. / 3. } else { 0.5 },
                r2: 2. / 3.,
            },
            &mut cache,
        )?;
        // The original consumes a noise draw even when eta is zero.
        let noise = exec.noise(
            &template,
            &sigma(&t)?,
            &sigma(&next)?,
            NoiseKind::Independent,
        )?;
        x = if let Some(up) = up {
            x.f_add(&up.f_mul_scalar(settings.noise)?.f_mul(&noise)?)?
        } else {
            x.f_add(&noise.f_mul_scalar(0. * settings.noise)?)?
        };
    }
    Ok(x)
}
#[derive(Clone, Copy, Debug)]
pub struct Adaptive {
    pub order: i64,
    pub rtol: f64,
    pub atol: f64,
    pub initial_step: f64,
    pub p: f64,
    pub i: f64,
    pub d: f64,
    pub accept_safety: f64,
    pub stochastic: Stochastic,
}
impl Default for Adaptive {
    fn default() -> Self {
        Self {
            order: 3,
            rtol: 0.05,
            atol: 0.0078,
            initial_step: 0.05,
            p: 0.,
            i: 1.,
            d: 0.,
            accept_safety: 0.81,
            stochastic: Stochastic::default(),
        }
    }
}
#[derive(Default, Debug)]
pub struct Info {
    pub steps: i64,
    pub nfe: i64,
    pub accepted: i64,
    pub rejected: i64,
}
pub fn adaptive(
    exec: &mut dyn Execution,
    native: &Kitchen,
    mut x: Tensor,
    min: &Tensor,
    max: &Tensor,
    settings: Adaptive,
) -> Result<(Tensor, Info), Error> {
    let _source_arguments = super::locals::ArgumentRoots::new(&[&x, min, max]);
    let _grad = tch::no_grad_guard();
    let (start, end) = endpoints(exec, min, max)?;
    if !matches!(settings.order, 2 | 3) {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "order should be 2 or 3".into(),
        });
    }
    let forward = end.double_value(&[]) > start.double_value(&[]);
    if !forward && settings.stochastic.eta != 0. {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "eta must be 0 for reverse sampling".into(),
        });
    }
    let h = settings.initial_step.abs() * if forward { 1. } else { -1. };
    let kind = native.default_dtype()?;
    let atol = Tensor::from(settings.atol).f_to_kind(kind)?;
    let rtol = Tensor::from(settings.rtol).f_to_kind(kind)?;
    let mut s = start;
    let mut previous = x.shallow_clone();
    let template = x.shallow_clone();
    let mut info = Info::default();
    let mut pid = Pid::new(
        h,
        settings.p,
        settings.i,
        settings.d,
        if settings.stochastic.eta != 0. {
            1.5
        } else {
            settings.order as f64
        },
        settings.accept_safety,
        1e-8,
    )?;
    while if forward {
        s.double_value(&[]) < end.f_sub_scalar(1e-5)?.double_value(&[])
    } else {
        s.double_value(&[]) > end.f_add_scalar(1e-5)?.double_value(&[])
    } {
        let mut cache = Vec::new();
        let t = if forward {
            end.f_minimum(&s.f_add_scalar(pid.h)?)?
        } else {
            end.f_maximum(&s.f_add_scalar(pid.h)?)?
        };
        let (mut down, mut up) = (t.shallow_clone(), None);
        if settings.stochastic.eta != 0. {
            let (d, _) = ancestral(&sigma(&s)?, &sigma(&t)?, settings.stochastic.eta)?;
            down = end.f_minimum(&d.f_log()?.f_neg()?)?;
            up = Some(
                sigma(&t)?
                    .f_pow_tensor_scalar(2)?
                    .f_sub(&sigma(&down)?.f_pow_tensor_scalar(2)?)?
                    .f_pow_tensor_scalar(0.5)?,
            );
        }
        let e = eps(exec, &mut cache, "eps", &x, &s)?;
        let denoised = x.f_sub(&sigma(&s)?.f_mul(&e)?)?;
        let low = step(
            exec,
            &x,
            &s,
            &down,
            Stage {
                order: settings.order - 1,
                r1: if settings.order == 3 { 1. / 3. } else { 0.5 },
                r2: 2. / 3.,
            },
            &mut cache,
        )?;
        let high = step(
            exec,
            &x,
            &s,
            &down,
            Stage {
                order: settings.order,
                r1: if settings.order == 3 { 1. / 3. } else { 0.5 },
                r2: 2. / 3.,
            },
            &mut cache,
        )?;
        let delta = atol.f_maximum(&rtol.f_mul(&low.f_abs()?.f_maximum(&previous.f_abs()?)?)?)?;
        let normalized = low.f_sub(&high)?.f_div(&delta)?;
        let norm = unsafe {
            boxed_aten::call(
                native,
                "aten::linalg_norm",
                "",
                &[
                    V::tensor(normalized),
                    V::None,
                    V::None,
                    V::Bool(false),
                    V::None,
                ],
                &vec![],
            )?
        }
        .as_tensor()?;
        let error = norm.f_div_scalar((x.numel() as f64).powf(0.5))?;
        let accepted = pid.propose(error.double_value(&[]))?;
        if accepted {
            previous = low;
            let noise = exec.noise(&template, &sigma(&s)?, &sigma(&t)?, NoiseKind::Independent)?;
            x = if let Some(up) = up {
                high.f_add(&up.f_mul_scalar(settings.stochastic.noise)?.f_mul(&noise)?)?
            } else {
                high.f_add(&noise.f_mul_scalar(0. * settings.stochastic.noise)?)?
            };
            s = t;
            info.accepted += 1;
        } else {
            info.rejected += 1;
        }
        info.nfe += settings.order;
        info.steps += 1;
        callback(
            exec,
            &x,
            info.steps - 1,
            &s,
            &s,
            &denoised,
            vec![
                ("error".into(), V::tensor(error)),
                ("h".into(), V::Float(pid.h)),
                ("steps".into(), V::Int(info.steps)),
                ("nfe".into(), V::Int(info.nfe)),
                ("n_accept".into(), V::Int(info.accepted)),
                ("n_reject".into(), V::Int(info.rejected)),
            ],
        )?;
    }
    Ok((x, info))
}
/// KSampler's wrappers derive min/max/NFE from the supplied schedule. They do
/// not otherwise use its intermediate sigma points or append a denoising step.
pub fn bounds(sigmas: &Tensor) -> Result<Option<(Tensor, Tensor, i64)>, Error> {
    let n = sigmas
        .size()
        .first()
        .copied()
        .ok_or_else(|| Error::ReferenceException {
            class: "TypeError",
            message: "len() of a 0-d tensor".into(),
        })?;
    if n <= 1 {
        return Ok(None);
    }
    let mut min = sigmas.f_get(-1)?;
    if min.f_eq(0.)?.f_is_nonzero()? {
        min = sigmas.f_get(-2)?;
    }
    Ok(Some((min, sigmas.f_get(0)?, n - 1)))
}
