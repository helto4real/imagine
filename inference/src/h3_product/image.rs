//! Comfy VAE wrapper's process_input, applied once before native VideoVae::encode.
use crate::native::Error;
use tch::{Kind, Tensor};
pub fn vae_input(rgb_nhwc: &Tensor, vae_kind: Kind) -> Result<Tensor, Error> {
    process_input(&rgb_nhwc.f_permute([0, 3, 1, 2])?, vae_kind)
}
pub fn process_input(input: &Tensor, vae_kind: Kind) -> Result<Tensor, Error> {
    Ok(input
        .f_mul_scalar(2.)?
        .f_sub_scalar(1.)?
        .f_to_kind(vae_kind)?)
}
