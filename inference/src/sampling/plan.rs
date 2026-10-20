//! KSampler construction/sigma planning and sample-bound selection. Actual
//! execution uses the selected source sampler, never a fallback solver.
use super::schedules::{self, ScheduleModel};
use crate::{
    execution_value::Value,
    native::{Error, Kitchen},
};
use tch::{Device, Kind, Tensor};

pub const SAMPLERS: [&str; 45] = [
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "cfgpp_ud10_ab",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
];
pub trait Model {
    /// model.get_model_object("model_sampling") at the original call site.
    fn sampling(&self) -> Result<&dyn ScheduleModel, Error>;
}
pub struct Plan {
    pub sampler: String,
    pub scheduler: String,
    pub steps: i64,
    pub sigmas: Tensor,
    pub denoise: Option<f64>,
    pub device: Device,
}
pub struct Settings<'a> {
    pub steps: i64,
    pub device: Device,
    pub sampler: Option<&'a str>,
    pub scheduler: Option<&'a str>,
    pub denoise: Option<f64>,
}
fn error(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn len(t: &Tensor) -> Result<i64, Error> {
    t.size()
        .first()
        .copied()
        .ok_or_else(|| error("TypeError", "len() of a 0-d tensor"))
}
fn slice(t: &Tensor, start: i64, end: i64) -> Result<Tensor, Error> {
    let n = len(t)?;
    let bound = |i: i64| if i < 0 { (n + i).max(0) } else { i.min(n) };
    let start = bound(start);
    let end = bound(end).max(start);
    Ok(t.f_slice(0, start, end, 1)?)
}
impl Plan {
    pub fn new(
        model: &dyn Model,
        native: &Kitchen,
        settings: Settings<'_>,
        warning: &mut dyn FnMut(&str),
    ) -> Result<Self, Error> {
        let Settings {
            steps,
            device,
            sampler,
            scheduler,
            denoise,
        } = settings;
        let sampler = sampler
            .filter(|name| SAMPLERS.contains(name))
            .unwrap_or(SAMPLERS[0]);
        let scheduler = scheduler
            .filter(|name| schedules::NAMES.contains(name))
            .unwrap_or(schedules::NAMES[0]);
        let mut plan = Self {
            sampler: sampler.into(),
            scheduler: scheduler.into(),
            steps,
            sigmas: Tensor::new(),
            denoise,
            device,
        };
        plan.set_steps(model, native, steps, denoise, warning)?;
        Ok(plan)
    }
    pub fn calculate_sigmas(
        &self,
        model: &dyn Model,
        native: &Kitchen,
        mut steps: i64,
        warning: &mut dyn FnMut(&str),
    ) -> Result<Tensor, Error> {
        let discard = matches!(
            self.sampler.as_str(),
            "dpm_2" | "dpm_2_ancestral" | "uni_pc" | "uni_pc_bh2"
        );
        if discard {
            steps += 1;
        }
        let sigmas =
            schedules::calculate(model.sampling()?, &self.scheduler, steps, native, warning)?;
        if discard {
            let n = len(&sigmas)?;
            Ok(Tensor::f_cat(
                &[slice(&sigmas, 0, -2)?, slice(&sigmas, -1, n)?],
                0,
            )?)
        } else {
            Ok(sigmas)
        }
    }
    /// Source set_steps updates steps/sigmas, but leaves the constructor's
    /// denoise attribute unchanged when this method is called again.
    pub fn set_steps(
        &mut self,
        model: &dyn Model,
        native: &Kitchen,
        steps: i64,
        denoise: Option<f64>,
        warning: &mut dyn FnMut(&str),
    ) -> Result<(), Error> {
        self.steps = steps;
        self.sigmas = match denoise {
            None => self
                .calculate_sigmas(model, native, steps, warning)?
                .f_to(self.device)?,
            Some(d) if d > 0.9999 => self
                .calculate_sigmas(model, native, steps, warning)?
                .f_to(self.device)?,
            Some(d) if d <= 0. => Tensor::f_empty([0], (Kind::Float, Device::Cpu))?,
            Some(d) => {
                let count = steps as f64 / d;
                if count.is_nan() {
                    return Err(error("ValueError", "cannot convert float NaN to integer"));
                }
                if count.is_infinite() {
                    return Err(error(
                        "OverflowError",
                        "cannot convert float infinity to integer",
                    ));
                }
                if count < i64::MIN as f64 || count >= -(i64::MIN as f64) {
                    return Err(Error::BindingGap {
                        backend: "KSampler".into(),
                        operation: "sigma count beyond ATen int64 dimension".into(),
                    });
                }
                let sigmas = self
                    .calculate_sigmas(model, native, count as i64, warning)?
                    .f_to(self.device)?;
                slice(&sigmas, -(steps + 1), len(&sigmas)?)?
            }
        };
        Ok(())
    }
    pub fn select(
        &self,
        noise: &Tensor,
        latent: Option<&Value>,
        sigmas: Option<&Tensor>,
        start: Option<i64>,
        last: Option<i64>,
        force_full_denoise: bool,
    ) -> Result<Selection, Error> {
        let mut sigmas = sigmas.unwrap_or(&self.sigmas).shallow_clone();
        if let Some(last) = last
            && last < len(&sigmas)? - 1
        {
            sigmas = slice(&sigmas, 0, last + 1)?;
            if force_full_denoise {
                if len(&sigmas)? == 0 {
                    return Err(error(
                        "IndexError",
                        "index -1 is out of bounds for dimension 0 with size 0",
                    ));
                }
                let _ = sigmas.f_get(-1)?.f_fill_(0)?;
            }
        }
        if let Some(start) = start {
            if start < len(&sigmas)? - 1 {
                sigmas = slice(&sigmas, start, len(&sigmas)?)?;
            } else {
                return Ok(Selection::Early(match latent {
                    Some(v) if !matches!(v, Value::None) => v.clone(),
                    _ => Value::tensor(noise.f_zeros_like()?),
                }));
            }
        }
        Ok(Selection::Sampling(sigmas))
    }
}
pub enum Selection {
    Early(Value),
    Sampling(Tensor),
}
