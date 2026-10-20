//! UniPC's registered SigmaConvert path and original BH/vary coefficient
//! updates. Includes callback shape and the reference's terminal-sigma mutation.
use super::solvers::Execution;
use crate::{
    boxed_aten,
    execution_value::Value as V,
    native::{Error, Kitchen},
};
use tch::{Device, Kind, Tensor};

pub trait NoiseSchedule {
    fn log_mean(&self, t: &Tensor) -> Result<Tensor, Error>;
    fn alpha(&self, t: &Tensor) -> Result<Tensor, Error> {
        Ok(self.log_mean(t)?.f_exp()?)
    }
    fn std(&self, t: &Tensor) -> Result<Tensor, Error> {
        Ok(self
            .log_mean(t)?
            .f_mul_scalar(2.)?
            .f_exp()?
            .f_rsub_scalar(1.)?
            .f_sqrt()?)
    }
    fn lambda(&self, t: &Tensor) -> Result<Tensor, Error> {
        let mean = self.log_mean(t)?;
        let std = mean
            .f_mul_scalar(2.)?
            .f_exp()?
            .f_rsub_scalar(1.)?
            .f_log()?
            .f_mul_scalar(0.5)?;
        Ok(mean.f_sub(&std)?)
    }
}
pub struct SigmaConvert;
impl NoiseSchedule for SigmaConvert {
    fn log_mean(&self, t: &Tensor) -> Result<Tensor, Error> {
        Ok(t.f_mul(t)?
            .f_add_scalar(1)?
            .f_reciprocal()?
            .f_mul_scalar(1)?
            .f_log()?
            .f_mul_scalar(0.5)?)
    }
}
fn expand(t: &Tensor, dims: usize) -> Result<Tensor, Error> {
    let mut shape = t.size();
    shape.extend(vec![1; dims - 1]);
    Ok(t.f_view(shape.as_slice())?)
}
fn tensor_list(native: &Kitchen, values: &[V], device: Device) -> Result<Tensor, Error> {
    let mut dtype = None;
    let mut data = Vec::new();
    for value in values {
        let (kind, scalar) = match value {
            V::Tensor(_) => {
                let t = value.as_tensor()?;
                if t.numel() != 1 {
                    return Err(Error::ReferenceException {
                        class: "ValueError",
                        message: "only one element tensors can be converted to Python scalars"
                            .into(),
                    });
                }
                (t.kind(), t.double_value(&[]))
            }
            V::Float(n) => (native.default_dtype()?, *n),
            V::Int(n) => (Kind::Int64, *n as f64),
            _ => {
                return Err(Error::Native(
                    "UniPC coefficient list expects numbers".into(),
                ));
            }
        };
        dtype = Some(if let Some(previous) = dtype {
            boxed_aten::promote_types(native, previous, kind)?
        } else {
            kind
        });
        data.push(scalar);
    }
    Ok(Tensor::f_from_slice(&data)?
        .f_to_kind(dtype.unwrap_or(native.default_dtype()?))?
        .f_to_device(device)?)
}
#[derive(Clone, Debug)]
pub struct Settings {
    pub predict_x0: bool,
    pub thresholding: bool,
    pub max_val: f64,
    pub variant: String,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            predict_x0: true,
            thresholding: false,
            max_val: 1.,
            variant: "bh1".into(),
        }
    }
}
pub struct Update<'a> {
    pub x: &'a Tensor,
    pub previous: &'a [Tensor],
    pub times: &'a [Tensor],
    pub time: &'a Tensor,
    pub order: usize,
    pub corrector: bool,
    pub provided: Option<&'a Tensor>,
}
pub struct UniPc<'a> {
    pub execution: &'a mut dyn Execution,
    pub native: &'a Kitchen,
    pub schedule: &'a dyn NoiseSchedule,
    pub settings: Settings,
}
impl UniPc<'_> {
    fn prediction(&mut self, x: &Tensor, time: &Tensor) -> Result<Tensor, Error> {
        let time = if time.numel() == 1 {
            time.f_expand([x.size()[0]], false)?
        } else {
            time.shallow_clone()
        };
        let s = expand(&time, x.dim())?;
        let scaled = x.f_mul(
            &s.f_pow_tensor_scalar(2)?
                .f_add_scalar(1.)?
                .f_pow_tensor_scalar(0.5)?,
        )?;
        let noise = scaled
            .f_sub(&self.execution.denoise(&scaled, &time, false)?.denoised)?
            .f_div(&s)?;
        if !self.settings.predict_x0 {
            return Ok(noise);
        }
        let mut x0 = x
            .f_sub(&expand(&self.schedule.std(&time)?, x.dim())?.f_mul(&noise)?)?
            .f_div(&expand(&self.schedule.alpha(&time)?, x.dim())?)?;
        if self.settings.thresholding {
            let flat = x0.f_abs()?.f_reshape([x0.size()[0], -1])?;
            let quantile = unsafe {
                boxed_aten::call(
                    self.native,
                    "aten::quantile",
                    "scalar",
                    &[V::tensor(flat), V::Float(0.995), V::Int(1), V::Bool(false)],
                    &vec![("interpolation".into(), V::Text("linear".into()))],
                )?
            }
            .as_tensor()?;
            let scale = expand(
                &quantile.f_maximum(
                    &quantile
                        .f_ones_like()?
                        .f_mul_scalar(self.settings.max_val)?,
                )?,
                x.dim(),
            )?;
            x0 = x0
                .f_clamp_tensor(Some(&scale.f_neg()?), Some(&scale))?
                .f_div(&scale)?;
        }
        Ok(x0)
    }
    pub fn update(&mut self, update: Update<'_>) -> Result<(Tensor, Option<Tensor>), Error> {
        let Update {
            x,
            previous,
            times,
            time,
            order,
            corrector,
            provided,
        } = update;
        let time = if time.dim() == 0 {
            time.f_view([-1])?
        } else {
            time.shallow_clone()
        };
        if self.settings.variant.contains("bh") {
            self.bh(Update {
                x,
                previous,
                times,
                time: &time,
                order,
                corrector,
                provided,
            })
        } else {
            if self.settings.variant != "vary_coeff" {
                return Err(Error::ReferenceException {
                    class: "AssertionError",
                    message: String::new(),
                });
            }
            self.vary(x, previous, times, &time, order, corrector)
        }
    }
    fn bh(&mut self, update: Update<'_>) -> Result<(Tensor, Option<Tensor>), Error> {
        let Update {
            x,
            previous,
            times,
            time,
            order,
            corrector,
            provided,
        } = update;
        if order > previous.len() {
            return Err(Error::ReferenceException {
                class: "AssertionError",
                message: String::new(),
            });
        }
        let n = previous.len();
        let t0 = &times[times.len() - 1];
        let l0 = self.schedule.lambda(t0)?;
        let lt = self.schedule.lambda(time)?;
        let m0 = &previous[n - 1];
        let std0 = self.schedule.std(t0)?;
        let stdt = self.schedule.std(time)?;
        let log0 = self.schedule.log_mean(t0)?;
        let logt = self.schedule.log_mean(time)?;
        let alpha = logt.f_exp()?;
        let h = lt.f_sub(&l0)?;
        let mut ratios = Vec::new();
        let mut differences = Vec::new();
        for i in 1..order {
            let r = self
                .schedule
                .lambda(&times[times.len() - 1 - i])?
                .f_sub(&l0)?
                .f_div(&h)?
                .f_get(0)?;
            differences.push(previous[n - 1 - i].f_sub(m0)?.f_div(&r)?);
            ratios.push(V::tensor(r));
        }
        ratios.push(V::Float(1.));
        let ratios = tensor_list(self.native, &ratios, x.device())?;
        let hh = if self.settings.predict_x0 {
            h.f_get(0)?.f_neg()?
        } else {
            h.f_get(0)?
        };
        let phi1 = hh.f_expm1()?;
        let mut phi = phi1.f_div(&hh)?.f_sub_scalar(1)?;
        let bh = match self.settings.variant.as_str() {
            "bh1" => hh.shallow_clone(),
            "bh2" => hh.f_expm1()?,
            _ => {
                return Err(Error::ReferenceException {
                    class: "NotImplementedError",
                    message: String::new(),
                });
            }
        };
        let mut rows = Vec::new();
        let mut b = Vec::new();
        let mut factorial = 1_i64;
        for i in 1..=order {
            rows.push(ratios.f_pow_tensor_scalar((i - 1) as i64)?);
            b.push(V::tensor(phi.f_mul_scalar(factorial)?.f_div(&bh)?));
            factorial *= (i + 1) as i64;
            phi = phi.f_div(&hh)?.f_sub_scalar(1. / factorial as f64)?;
        }
        let matrix = Tensor::f_stack(&rows, 0)?;
        drop(rows);
        let b = tensor_list(self.native, &b, x.device())?;
        let predictor = !differences.is_empty() && provided.is_none();
        let differences = if differences.is_empty() {
            None
        } else {
            Some(Tensor::f_stack(&differences, 1)?)
        };
        let rho_p = if differences.is_some() && provided.is_none() {
            Some(if order == 2 {
                tensor_list(self.native, &[V::Float(0.5)], b.device())?
            } else {
                Tensor::f_linalg_solve(
                    &matrix
                        .f_narrow(0, 0, order as i64 - 1)?
                        .f_narrow(1, 0, order as i64 - 1)?,
                    &b.f_narrow(0, 0, order as i64 - 1)?,
                    true,
                )?
            })
        } else {
            None
        };
        let rho_c = if corrector {
            Some(if order == 1 {
                tensor_list(self.native, &[V::Float(0.5)], b.device())?
            } else {
                Tensor::f_linalg_solve(&matrix, &b, true)?
            })
        } else {
            None
        };
        let (base, factor) = if self.settings.predict_x0 {
            (
                expand(&stdt.f_div(&std0)?, x.dim())?
                    .f_mul(x)?
                    .f_sub(&expand(&alpha.f_mul(&phi1)?, x.dim())?.f_mul(m0)?)?,
                expand(&alpha.f_mul(&bh)?, x.dim())?,
            )
        } else {
            (
                expand(&logt.f_sub(&log0)?.f_exp()?, x.dim())?
                    .f_mul(x)?
                    .f_sub(&expand(&stdt.f_mul(&phi1)?, x.dim())?.f_mul(m0)?)?,
                expand(&stdt.f_mul(&bh)?, x.dim())?,
            )
        };
        let predict_x0 = self.settings.predict_x0;
        let contract = |weights: &Tensor| -> Result<Tensor, Error> {
            let differences = differences.as_ref().unwrap();
            if predict_x0 {
                Ok(differences.f_tensordot(weights, [1], [0])?)
            } else {
                Ok(Tensor::f_einsum(
                    "k,bkchw->bchw",
                    &[weights, differences],
                    None::<&[i64]>,
                )?)
            }
        };
        let mut out = if let Some(provided) = provided {
            provided.shallow_clone()
        } else {
            let residual = if predictor {
                Some(contract(rho_p.as_ref().unwrap())?)
            } else {
                None
            };
            base.f_sub(&if let Some(residual) = residual {
                factor.f_mul(&residual)?
            } else {
                factor.f_mul_scalar(0)?
            })?
        };
        let mut model_t = None;
        if corrector {
            let model = self.prediction(&out, time)?;
            let rho = rho_c.as_ref().unwrap();
            let residual = if differences.is_some() {
                Some(contract(&rho.f_narrow(0, 0, order as i64 - 1)?)?)
            } else {
                None
            };
            let term = rho.f_get(-1)?.f_mul(&model.f_sub(m0)?)?;
            let correction = if let Some(residual) = residual {
                residual.f_add(&term)?
            } else {
                term.f_add_scalar(0)?
            };
            out = base.f_sub(&factor.f_mul(&correction)?)?;
            model_t = Some(model);
        }
        Ok((out, model_t))
    }
    fn vary(
        &mut self,
        x: &Tensor,
        previous: &[Tensor],
        times: &[Tensor],
        time: &Tensor,
        order: usize,
        corrector: bool,
    ) -> Result<(Tensor, Option<Tensor>), Error> {
        if order > previous.len() {
            return Err(Error::ReferenceException {
                class: "AssertionError",
                message: String::new(),
            });
        }
        let n = previous.len();
        let t0 = &times[times.len() - 1];
        let l0 = self.schedule.lambda(t0)?;
        let lt = self.schedule.lambda(time)?;
        let m0 = &previous[n - 1];
        let std0 = self.schedule.std(t0)?;
        let stdt = self.schedule.std(time)?;
        let logt = self.schedule.log_mean(time)?;
        let alpha = logt.f_exp()?;
        let h = lt.f_sub(&l0)?;
        let mut ratios = Vec::new();
        let mut differences = Vec::new();
        for i in 1..order {
            let r = self
                .schedule
                .lambda(&times[times.len() - 1 - i])?
                .f_sub(&l0)?
                .f_div(&h)?;
            differences.push(previous[n - 1 - i].f_sub(m0)?.f_div(&r)?);
            ratios.push(V::tensor(r));
        }
        ratios.push(V::Float(1.));
        let ratios = tensor_list(self.native, &ratios, x.device())?;
        let k = ratios.size()[0];
        let mut columns = Vec::new();
        let mut column = ratios.f_ones_like()?;
        for i in 1..=k {
            columns.push(column.shallow_clone());
            column = column.f_mul(&ratios)?.f_div_scalar(i + 1)?;
        }
        let matrix = Tensor::f_stack(&columns, 1)?;
        drop(columns);
        let differences = if differences.is_empty() {
            None
        } else {
            Some(Tensor::f_stack(&differences, 1)?)
        };
        let inverse_p = if differences.is_some() {
            Some(Tensor::f_linalg_inv(
                &matrix.f_narrow(0, 0, k - 1)?.f_narrow(1, 0, k - 1)?,
            )?)
        } else {
            None
        };
        let inverse_c = if corrector {
            Some(Tensor::f_linalg_inv(&matrix)?)
        } else {
            None
        };
        let hh = if self.settings.predict_x0 {
            h.f_neg()?
        } else {
            h.shallow_clone()
        };
        let phi1 = hh.f_expm1()?;
        let mut phis = Vec::new();
        let mut phi = phi1.shallow_clone();
        let mut factorial = 1_i64;
        for i in 1..=k + 1 {
            phis.push(phi.shallow_clone());
            phi = phi.f_div(&hh)?.f_sub_scalar(1. / factorial as f64)?;
            factorial *= i + 1;
        }
        let (base, scale) = if self.settings.predict_x0 {
            (
                stdt.f_div(&std0)?
                    .f_mul(x)?
                    .f_sub(&alpha.f_mul(&phi1)?.f_mul(m0)?)?,
                alpha,
            )
        } else {
            (
                self.schedule
                    .log_mean(time)?
                    .f_sub(&self.schedule.log_mean(t0)?)?
                    .f_exp()?
                    .f_mul(x)?
                    .f_sub(&stdt.f_mul(&phi1)?.f_mul(m0)?)?,
                stdt,
            )
        };
        let contract = |weights: &Tensor| -> Result<Tensor, Error> {
            Ok(Tensor::f_einsum(
                "bkchw,k->bchw",
                &[differences.as_ref().unwrap(), weights],
                None::<&[i64]>,
            )?)
        };
        let mut out = base.shallow_clone();
        if let Some(inverse) = &inverse_p {
            for j in 0..k - 1 {
                out = out.f_sub(
                    &scale
                        .f_mul(&phis[j as usize + 1])?
                        .f_mul(&contract(&inverse.f_get(j)?)?)?,
                )?;
            }
        }
        let mut model_t = None;
        if corrector {
            let model = self.prediction(&out, time)?;
            let delta = model.f_sub(m0)?;
            out = base;
            let inverse = inverse_c.as_ref().unwrap();
            let mut last = 0;
            for j in 0..k - 1 {
                out = out.f_sub(
                    &scale
                        .f_mul(&phis[j as usize + 1])?
                        .f_mul(&contract(&inverse.f_get(j)?.f_narrow(0, 0, k - 1)?)?)?,
                )?;
                last = j;
            }
            out = out.f_sub(
                &scale
                    .f_mul(&phis[k as usize])?
                    .f_mul(&delta.f_mul(&inverse.f_get(last)?.f_get(-1)?)?)?,
            )?;
            model_t = Some(model);
        }
        Ok((out, model_t))
    }
    pub fn multistep(
        &mut self,
        mut x: Tensor,
        times: &Tensor,
        order: i64,
        lower_final: bool,
    ) -> Result<Tensor, Error> {
        let steps = times.size()[0] - 1;
        if steps < order {
            return Err(Error::ReferenceException {
                class: "AssertionError",
                message: String::new(),
            });
        }
        let mut predictions: Vec<Tensor> = Vec::new();
        let mut previous: Vec<Tensor> = Vec::new();
        for index in 0..steps {
            if index == 0 {
                let t = times.f_get(0)?.f_expand([x.size()[0]], false)?;
                predictions.push(self.prediction(&x, &t)?);
                previous.push(t);
            } else if index < order {
                let t = times.f_get(index)?.f_expand([x.size()[0]], false)?;
                let (out, model) = self.update(Update {
                    x: &x,
                    previous: &predictions,
                    times: &previous,
                    time: &t,
                    order: index as usize,
                    corrector: true,
                    provided: None,
                })?;
                x = out;
                let model = if let Some(model) = model {
                    model
                } else {
                    self.prediction(&x, &t)?
                };
                predictions.push(model);
                previous.push(t);
            } else {
                let final_step = if index == steps - 1 { 1 } else { 0 };
                for step in index..index + 1 + final_step {
                    let t = times.f_get(step)?.f_expand([x.size()[0]], false)?;
                    let used = if lower_final {
                        order.min(steps + 1 - step)
                    } else {
                        order
                    };
                    let (out, model) = self.update(Update {
                        x: &x,
                        previous: &predictions,
                        times: &previous,
                        time: &t,
                        order: used as usize,
                        corrector: step != steps,
                        provided: None,
                    })?;
                    x = out;
                    for i in 0..order - 1 {
                        previous[i as usize] = previous[i as usize + 1].shallow_clone();
                        predictions[i as usize] = predictions[i as usize + 1].shallow_clone();
                    }
                    let last = previous.len() - 1;
                    previous[last] = t.shallow_clone();
                    if step < steps {
                        predictions[last] = if let Some(model) = model {
                            model
                        } else {
                            self.prediction(&x, &t)?
                        };
                    }
                }
            }
            self.execution.callback_values(&[
                ("x".into(), V::tensor(x.shallow_clone())),
                ("i".into(), V::Int(index)),
                (
                    "denoised".into(),
                    V::tensor(predictions[predictions.len() - 1].shallow_clone()),
                ),
            ])?;
        }
        Ok(x)
    }
}
pub fn sample(
    exec: &mut dyn Execution,
    native: &Kitchen,
    mut noise: Tensor,
    sigmas: &Tensor,
    variant: &str,
) -> Result<Tensor, Error> {
    let copy = |t: &Tensor| crate::native::clone_tensor(t);
    let initial = copy(sigmas)?;
    let times = if sigmas.f_get(-1)?.double_value(&[]) == 0. {
        let times = sigmas.f_slice(0, 0, sigmas.size()[0], 1)?;
        drop(initial);
        let _ = times.f_get(-1)?.f_fill_(0.001)?;
        times
    } else {
        let times = copy(sigmas)?;
        drop(initial);
        times
    };
    let input = noise.f_div(
        &times
            .f_get(0)?
            .f_pow_tensor_scalar(2.)?
            .f_add_scalar(1.)?
            .f_sqrt()?,
    )?;
    // Source rebinds noise to the scaled value; that outer local then owns the
    // initial multistep input until sample_unipc returns.
    drop(std::mem::replace(&mut noise, input));
    let mut outer = super::locals::SourceLocals::default();
    let _ = outer.keep("noise", noise.shallow_clone());
    let _ = outer.keep("sigmas", sigmas.shallow_clone());
    let _ = outer.keep("timesteps", times.shallow_clone());
    exec.source_locals(&outer, "unipc_outer")?;
    let mut pc = UniPc {
        execution: exec,
        native,
        schedule: &SigmaConvert,
        settings: Settings {
            variant: variant.into(),
            ..Default::default()
        },
    };
    let mut out = pc.multistep(
        noise.shallow_clone(),
        &times,
        3.min(times.size()[0] - 2),
        true,
    )?;
    let _ = out.f_div_(&pc.schedule.alpha(&times.f_get(-1)?)?)?;
    Ok(out)
}
/// The registered BH2 alias retains the original noise while sample_unipc
/// rebinds its inner noise parameter to the scaled tensor.
pub fn sample_bh2(
    exec: &mut dyn Execution,
    native: &Kitchen,
    noise: Tensor,
    sigmas: &Tensor,
) -> Result<Tensor, Error> {
    sample(exec, native, noise.shallow_clone(), sigmas, "bh2")
}
