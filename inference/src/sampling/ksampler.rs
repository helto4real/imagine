//! KSAMPLER / KSamplerX0Inpaint ownership and masking, including DDIM's
//! source random-inpaint option. The selected solver is supplied explicitly.
use super::{
    parameterization::Parameterization,
    solvers::{Execution, NoiseKind, Prediction},
};
use crate::{
    execution_value::Value,
    models::qwen_image21::EulerStep,
    native::{Error, Kitchen},
};
use tch::{Device, Tensor};
#[derive(Clone, Copy, Debug)]
pub enum Seed {
    Missing,
    Null,
    Integer(i128),
}
/// Sampler.max_denoise's fixed math.isclose policy, including nonfinite values.
pub fn maximum_denoise(maximum: f64, sigma: f64) -> bool {
    sigma == maximum
        || (sigma.is_finite()
            && maximum.is_finite()
            && (maximum - sigma).abs() <= 1e-5 * maximum.abs().max(sigma.abs()))
        || sigma > maximum
}
pub trait MaskFunction {
    fn apply(
        &mut self,
        sigma: &Tensor,
        mask: &Tensor,
        execution: &mut dyn Execution,
        sigmas: &Tensor,
    ) -> Result<Tensor, Error>;
}
pub struct Inputs<'a> {
    pub sigmas: &'a Tensor,
    pub noise: &'a Tensor,
    pub latent: &'a Tensor,
    pub mask: Option<&'a Tensor>,
    pub random_inpaint: bool,
    pub seed: Seed,
    pub maximum_sigma: f64,
    pub parameterization: Parameterization,
    pub mask_function: Option<&'a mut dyn MaskFunction>,
}
pub struct Inpaint<'a> {
    pub inner: &'a mut dyn Execution,
    pub sigmas: &'a Tensor,
    pub latent: &'a Tensor,
    pub noise: Tensor,
    pub mask: Option<&'a Tensor>,
    pub parameterization: Parameterization,
    pub mask_function: Option<&'a mut dyn MaskFunction>,
}
impl Execution for Inpaint<'_> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg: bool) -> Result<Prediction, Error> {
        self.call(x, sigma, cfg, false)
    }
    fn denoise_with_capture(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        disable: bool,
    ) -> Result<Prediction, Error> {
        self.call(x, sigma, disable, true)
    }
    fn noise(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        next: &Tensor,
        kind: NoiseKind,
    ) -> Result<Tensor, Error> {
        self.inner.noise(x, sigma, next, kind)
    }
    fn callback(&mut self, step: EulerStep<'_>) -> Result<(), Error> {
        self.inner
            .sample_callback(step.i, step.denoised, step.x, self.sigmas.size()[0] - 1)
    }
    fn callback_values(&mut self, fields: &[(String, Value)]) -> Result<(), Error> {
        let get = |name: &str| {
            fields
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v)
                .ok_or_else(|| Error::ReferenceException {
                    class: "KeyError",
                    message: format!("'{name}'"),
                })
        };
        let Value::Int(index) = get("i")? else {
            return Err(Error::Native("Invalid sample callback index".into()));
        };
        self.inner.sample_callback(
            *index,
            &get("denoised")?.as_tensor()?,
            &get("x")?.as_tensor()?,
            self.sigmas.size()[0] - 1,
        )
    }
    fn eps_callback(&mut self) -> Result<(), Error> {
        self.inner.eps_callback()
    }
    fn make_brownian(
        &mut self,
        x: &Tensor,
        min: &Tensor,
        max: &Tensor,
        cpu: bool,
    ) -> Result<super::brownian::BrownianNoise, Error> {
        self.inner.make_brownian(x, min, max, cpu)
    }
    fn observe_noise(
        &mut self,
        n: &Tensor,
        s: &Tensor,
        t: &Tensor,
        k: NoiseKind,
    ) -> Result<(), Error> {
        self.inner.observe_noise(n, s, t, k)
    }
    fn percent_to_sigma(&self, p: f64) -> Result<f64, Error> {
        self.inner.percent_to_sigma(p)
    }
    fn warning(&mut self, message: &str) {
        self.inner.warning(message)
    }
    fn source_locals(&mut self, f: &super::locals::SourceLocals, event: &str) -> Result<(), Error> {
        self.inner.source_locals(f, event)
    }
    fn scale_latent_inpaint(
        &mut self,
        p: Parameterization,
        x: &Tensor,
        sigma: &Tensor,
        noise: &Tensor,
        latent: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, Error> {
        self.inner
            .scale_latent_inpaint(p, x, sigma, noise, latent, mask)
    }
}
impl Inpaint<'_> {
    fn call(
        &mut self,
        x: &Tensor,
        sigma: &Tensor,
        cfg: bool,
        capture: bool,
    ) -> Result<Prediction, Error> {
        let mask = if let Some(mask) = self.mask {
            Some(if let Some(function) = self.mask_function.as_mut() {
                function.apply(sigma, mask, self.inner, self.sigmas)?
            } else {
                mask.shallow_clone()
            })
        } else {
            None
        };
        let latent_mask = mask
            .as_ref()
            .map(|mask| mask.f_rsub_scalar(1.))
            .transpose()?;
        let input = if let Some(mask) = &mask {
            let scaled = self.inner.scale_latent_inpaint(
                self.parameterization,
                x,
                sigma,
                &self.noise,
                self.latent,
                mask,
            )?;
            x.f_mul(mask)?
                .f_add(&scaled.f_mul(latent_mask.as_ref().unwrap())?)?
        } else {
            x.shallow_clone()
        };
        let mut out = if capture {
            self.inner.denoise_with_capture(&input, sigma, cfg)?
        } else {
            self.inner.denoise(&input, sigma, cfg)?
        };
        if let Some(mask) = mask {
            out.denoised = out
                .denoised
                .f_mul(&mask)?
                .f_add(&self.latent.f_mul(latent_mask.as_ref().unwrap())?)?;
        }
        Ok(out)
    }
}
pub fn sample<'a>(
    execution: &'a mut dyn Execution,
    native: &Kitchen,
    input: Inputs<'a>,
    solver: impl FnOnce(&mut dyn Execution, Tensor, &Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let Inputs {
        sigmas,
        noise,
        latent,
        mask,
        random_inpaint,
        seed,
        maximum_sigma,
        parameterization,
        mask_function,
    } = input;
    // extra_args receives denoise_mask before creating the source inpaint wrapper.
    execution.set_denoise_mask(mask)?;
    let inpaint_noise = if random_inpaint {
        let seed = match seed {
            Seed::Missing => 42,
            Seed::Null => {
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: "unsupported operand type(s) for +: 'NoneType' and 'int'".into(),
                });
            }
            Seed::Integer(seed) => seed
                .checked_add(1)
                .filter(|s| *s >= i64::MIN as i128 && *s <= u64::MAX as i128)
                .ok_or_else(|| Error::ReferenceException {
                    class: "ValueError",
                    message: "Overflow when unpacking long long".into(),
                })?,
        };
        tch::manual_seed(seed as i64);
        Tensor::f_randn(noise.size(), (native.default_dtype()?, Device::Cpu))?
            .f_to_kind(noise.kind())?
            .f_to_device(noise.device())?
    } else {
        noise.shallow_clone()
    };
    let mut inpaint = Inpaint {
        inner: execution,
        sigmas,
        latent,
        noise: inpaint_noise,
        mask,
        parameterization,
        mask_function,
    };
    let first = sigmas.f_get(0)?;
    let sigma = first.double_value(&[]);
    // Python math.isclose uses max magnitude, zero absolute tolerance.
    let scaled = parameterization.noise_scaling(
        &first,
        noise,
        latent,
        maximum_denoise(maximum_sigma, sigma),
    )?;
    let samples = solver(&mut inpaint, scaled, sigmas)?;
    parameterization.inverse_noise_scaling(&sigmas.f_get(-1)?, &samples)
}
