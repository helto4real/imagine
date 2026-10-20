//! Prediction/noise algebra from comfy.model_sampling, separate from schedules.
use crate::native::Error;
use tch::Tensor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Prediction {
    Eps,
    V,
    VDDPM,
    Edm,
    Const,
    X0,
    ImageToImage,
    ImageToImageFlow,
    ImageToImageVelocity,
    CosmosRFlow,
}
#[derive(Clone, Copy, Debug)]
pub struct Parameterization {
    pub prediction: Prediction,
    pub sigma_data: f64,
    pub noise_scale: f64,
}
pub fn reshape_sigma(sigma: &Tensor, rank: usize) -> Result<Tensor, Error> {
    if sigma.numel() == 1 {
        return Ok(sigma.f_view(())?);
    }
    let mut shape = vec![sigma.size()[0]];
    shape.resize(rank, 1);
    Ok(sigma.f_view(shape.as_slice())?)
}
impl Parameterization {
    pub fn is_flow(&self) -> bool {
        matches!(
            self.prediction,
            Prediction::Const | Prediction::ImageToImageFlow | Prediction::ImageToImageVelocity
        )
    }
    pub fn input(&self, sigma: &Tensor, noise: &Tensor) -> Result<Tensor, Error> {
        use Prediction::*;
        match self.prediction {
            Const | ImageToImage | ImageToImageFlow | ImageToImageVelocity | VDDPM => {
                Ok(noise.shallow_clone())
            }
            CosmosRFlow => {
                let sigma = reshape_sigma(&sigma.f_div(&sigma.f_add_scalar(1)?)?, noise.dim())?;
                Ok(noise.f_mul(&sigma.f_rsub_scalar(1)?)?)
            }
            Eps | V | Edm | X0 => {
                let sigma = reshape_sigma(sigma, noise.dim())?;
                Ok(noise.f_div(
                    &sigma
                        .f_pow_tensor_scalar(2)?
                        .f_add_scalar(self.sigma_data.powi(2))?
                        .f_pow_tensor_scalar(0.5)?,
                )?)
            }
        }
    }
    pub fn denoised(
        &self,
        sigma: &Tensor,
        output: &Tensor,
        input: &Tensor,
    ) -> Result<Tensor, Error> {
        use Prediction::*;
        match self.prediction {
            X0 | ImageToImage | ImageToImageFlow => Ok(output.shallow_clone()),
            ImageToImageVelocity => Ok(input.f_sub(output)?),
            Eps | Const => Ok(input.f_sub(&output.f_mul(&reshape_sigma(sigma, output.dim())?)?)?),
            V | Edm => {
                let sigma = reshape_sigma(sigma, output.dim())?;
                let denominator = sigma
                    .f_pow_tensor_scalar(2)?
                    .f_add_scalar(self.sigma_data.powi(2))?;
                let a = input
                    .f_mul_scalar(self.sigma_data.powi(2))?
                    .f_div(&denominator)?;
                let b = output
                    .f_mul(&sigma)?
                    .f_mul_scalar(self.sigma_data)?
                    .f_div(&denominator.f_pow_tensor_scalar(0.5)?)?;
                Ok(if self.prediction == Edm {
                    a.f_add(&b)?
                } else {
                    a.f_sub(&b)?
                })
            }
            VDDPM => {
                let sigma = reshape_sigma(sigma, output.dim())?;
                let denominator = sigma
                    .f_pow_tensor_scalar(2)?
                    .f_add_scalar(1.)?
                    .f_pow_tensor_scalar(0.5)?;
                Ok(input
                    .f_div(&denominator)?
                    .f_sub(&output.f_mul(&sigma)?.f_div(&denominator)?)?)
            }
            CosmosRFlow => {
                let sigma = reshape_sigma(&sigma.f_div(&sigma.f_add_scalar(1)?)?, output.dim())?;
                Ok(input
                    .f_mul(&sigma.f_rsub_scalar(1.)?)?
                    .f_sub(&output.f_mul(&sigma)?)?)
            }
        }
    }
    pub fn noise_scaling(
        &self,
        sigma: &Tensor,
        noise: &Tensor,
        latent: &Tensor,
        max_denoise: bool,
    ) -> Result<Tensor, Error> {
        use Prediction::*;
        match self.prediction {
            ImageToImageFlow | ImageToImageVelocity => Ok(latent.shallow_clone()),
            Const => {
                let sigma = reshape_sigma(sigma, noise.dim())?;
                Ok(sigma
                    .f_mul(&noise.f_mul_scalar(self.noise_scale)?)?
                    .f_add(&sigma.f_rsub_scalar(1.)?.f_mul(latent)?)?)
            }
            CosmosRFlow => {
                let sigma = reshape_sigma(sigma, noise.dim())?;
                let mut scaled = noise.f_mul(&sigma)?;
                let _ = scaled.f_add_(latent)?;
                Ok(scaled)
            }
            Eps | V | VDDPM | Edm | X0 | ImageToImage => {
                let sigma = reshape_sigma(sigma, noise.dim())?;
                let scale = if max_denoise {
                    sigma.f_pow_tensor_scalar(2.)?.f_add_scalar(1.)?.f_sqrt()?
                } else {
                    sigma
                };
                let mut scaled = noise.f_mul(&scale)?;
                let _ = scaled.f_add_(latent)?;
                Ok(scaled)
            }
        }
    }
    pub fn inverse_noise_scaling(&self, sigma: &Tensor, latent: &Tensor) -> Result<Tensor, Error> {
        Ok(match self.prediction {
            Prediction::Const => {
                latent.f_div(&reshape_sigma(sigma, latent.dim())?.f_rsub_scalar(1.)?)?
            }
            Prediction::ImageToImageFlow => latent.f_rsub_scalar(1.)?,
            _ => latent.shallow_clone(),
        })
    }
}

/// nodes_model_advanced.LCM mixes its original x0 prediction with model input;
/// timestep conversion belongs to the actual (usually distilled) model object.
pub fn lcm_denoised(
    model: &dyn super::schedules::ScheduleModel,
    sigma: &Tensor,
    output: &Tensor,
    input: &Tensor,
) -> Result<Tensor, Error> {
    let mut shape = sigma.size().into_iter().take(1).collect::<Vec<_>>();
    shape.extend(vec![1; output.dim().saturating_sub(1)]);
    let timestep = model.timestep(sigma)?.f_view(shape.as_slice())?;
    let sigma = sigma.f_view(shape.as_slice())?;
    let x0 = input.f_sub(&output.f_mul(&sigma)?)?;
    let scaled = timestep.f_mul_scalar(10.)?;
    let denominator = scaled.f_pow_tensor_scalar(2)?.f_add_scalar(0.25)?;
    let skip = denominator.f_reciprocal()?.f_mul_scalar(0.25)?;
    let out = scaled.f_div(&denominator.f_pow_tensor_scalar(0.5)?)?;
    Ok(out.f_mul(&x0)?.f_add(&skip.f_mul(input)?)?)
}
