//! Original ModelSamplingDiscreteFlow schedule, distinct from Qwen's FLUX shift.
use crate::native::{Error, Kitchen};
use tch::{Device, Kind, Tensor};
pub struct MossSampling {
    pub shift: f64,
    pub multiplier: f64,
    pub sigmas: Tensor,
}
impl MossSampling {
    pub fn new(shift: f64, multiplier: f64, timesteps: i64) -> Result<Self, Error> {
        let time = Tensor::f_arange_start(1, timesteps + 1, (Kind::Int64, Device::Cpu))?
            .f_div_scalar(timesteps)?
            .f_mul_scalar(multiplier)?;
        let mut model = Self {
            shift,
            multiplier,
            sigmas: Tensor::new(),
        };
        model.sigmas = model.sigma(&time)?;
        Ok(model)
    }
    pub fn moss() -> Result<Self, Error> {
        Self::new(5., 1., 1000)
    }
    pub fn sigma(&self, time: &Tensor) -> Result<Tensor, Error> {
        let t = time.f_div_scalar(self.multiplier)?;
        if self.shift == 1. {
            return Ok(t);
        }
        Ok(t.f_mul_scalar(self.shift)?
            .f_div(&t.f_mul_scalar(self.shift - 1.)?.f_add_scalar(1.)?)?)
    }
    pub fn timestep(&self, sigma: &Tensor) -> Result<Tensor, Error> {
        Ok(sigma.f_mul_scalar(self.multiplier)?)
    }
    pub fn percent_to_sigma(&self, percent: f64) -> f64 {
        if percent <= 0. {
            return 1.;
        }
        if percent >= 1. {
            return 0.;
        }
        let t = 1. - percent;
        if self.shift == 1. {
            t
        } else {
            self.shift * t / (1. + (self.shift - 1.) * t)
        }
    }
    /// Source normal_scheduler, including its explicit CPU FloatTensor result.
    pub fn normal(&self, steps: i64, sgm: bool, native: &Kitchen) -> Result<Tensor, Error> {
        let start = self.timestep(&self.sigmas.f_get(-1)?)?;
        let end = self.timestep(&self.sigmas.f_get(0)?)?;
        let mut count = steps;
        let mut append = true;
        if !sgm && self.sigma(&end)?.double_value(&[]).abs() <= 1e-5 {
            count += 1;
            append = false;
        }
        let time = Tensor::f_linspace(
            start.double_value(&[]),
            end.double_value(&[]),
            if sgm { count + 1 } else { count },
            (native.default_dtype()?, Device::Cpu),
        )?;
        let mut values = Vec::new();
        for i in 0..if sgm {
            time.size()[0] - 1
        } else {
            time.size()[0]
        } {
            values.push(self.sigma(&time.f_get(i)?)?.double_value(&[]) as f32);
        }
        if append {
            values.push(0.)
        }
        Ok(Tensor::f_from_slice(&values)?)
    }
}
pub fn moss_memory_required(shape: &[i64], dtype: Kind, flash_attention: bool) -> f64 {
    let area = shape[0] * shape[2..].iter().product::<i64>();
    if flash_attention {
        (area as f64 * dtype.elt_size_in_bytes() as f64 * 0.01 * 14.6) * (1024 * 1024) as f64
    } else {
        (area as f64 * 0.15 * 14.6) * (1024 * 1024) as f64
    }
}
