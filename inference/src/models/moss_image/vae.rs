//! Flux.2 AutoencoderKL: original 2D residual VAE and latent BatchNorm packing.
use crate::models::qwen_image21::VaeExecution;
use crate::{checkpoint::PreparedParameters, execution::Runtime, native::Error};
use tch::Tensor;

#[derive(Clone)]
pub struct Flux2VaeConfig {
    pub dim: i64,
    pub decoder_dim: i64,
    pub z_dim: i64,
    pub multipliers: Vec<i64>,
    pub residual_blocks: usize,
}
impl Default for Flux2VaeConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            decoder_dim: 128,
            z_dim: 32,
            multipliers: vec![1, 2, 4, 4],
            residual_blocks: 2,
        }
    }
}
pub struct Flux2Vae {
    pub config: Flux2VaeConfig,
    state: PreparedParameters,
}
impl Flux2Vae {
    pub fn new(config: Flux2VaeConfig, state: PreparedParameters) -> Self {
        Self { config, state }
    }
    fn weight(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        self.state
            .get(name)
            .ok_or_else(|| Error::BindingGap {
                backend: "flux2_vae".into(),
                operation: format!("Missing prepared {name}"),
            })?
            .as_tensor()?
            .f_to_device_(x.device(), x.kind(), false, false)
            .map_err(Error::from)
    }
    fn conv<E: VaeExecution>(
        &self,
        name: &str,
        x: &Tensor,
        stride: i64,
        pad: i64,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        Ok(x.f_conv2d(
            &self.weight(&format!("{name}.weight"), x)?,
            Some(self.weight(&format!("{name}.bias"), x)?),
            [stride, stride],
            [pad, pad],
            [1, 1],
            1,
        )?)
    }
    fn norm<E: VaeExecution>(&self, name: &str, x: &Tensor, e: &mut E) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        let s = x.size();
        let batch = s[0] * s[1] / 32 * s[2] * s[3];
        if batch == 1 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: format!(
                    "Expected more than 1 value per channel when training, got input size {:?}",
                    vec![s[0] * s[1] / 32, 32, s[2], s[3]]
                ),
            });
        }
        Ok(x.f_group_norm(
            32,
            Some(self.weight(&format!("{name}.weight"), x)?),
            Some(self.weight(&format!("{name}.bias"), x)?),
            1e-6,
            true,
        )?)
    }
    fn residual<E: VaeExecution>(
        &self,
        name: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let h = self.norm(&format!("{name}.norm1"), x, e)?.f_silu_()?;
        let h = self.conv(&format!("{name}.conv1"), &h, 1, 1, e)?;
        let h = self.norm(&format!("{name}.norm2"), &h, e)?.f_silu_()?;
        // Source Dropout(0, inplace=True) is an identity in this image VAE.
        let h = self.conv(&format!("{name}.conv2"), &h, 1, 1, e)?;
        let shortcut = if self
            .state
            .contains_key(&format!("{name}.nin_shortcut.weight"))
        {
            self.conv(&format!("{name}.nin_shortcut"), x, 1, 0, e)?
        } else {
            x.shallow_clone()
        };
        Ok(shortcut.f_add(&h)?)
    }
    fn attention<E: VaeExecution>(
        &self,
        rt: &Runtime,
        name: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let h = self.norm(&format!("{name}.norm"), x, e)?;
        let q = self.conv(&format!("{name}.q"), &h, 1, 0, e)?;
        let k = self.conv(&format!("{name}.k"), &h, 1, 0, e)?;
        let v = self.conv(&format!("{name}.v"), &h, 1, 0, e)?;
        let h = e.attention(rt, &q, &k, &v)?;
        Ok(x.f_add(&self.conv(&format!("{name}.proj_out"), &h, 1, 0, e)?)?)
    }
    fn middle<E: VaeExecution>(
        &self,
        rt: &Runtime,
        path: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let x = self.residual(&format!("{path}.block_1"), x, e)?;
        let x = self.attention(rt, &format!("{path}.attn_1"), &x, e)?;
        self.residual(&format!("{path}.block_2"), &x, e)
    }
    pub fn encode<E: VaeExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let mut h = self.conv("encoder.conv_in", x, 1, 1, e)?;
        for level in 0..self.config.multipliers.len() {
            for block in 0..self.config.residual_blocks {
                h = self.residual(&format!("encoder.down.{level}.block.{block}"), &h, e)?;
            }
            if level + 1 < self.config.multipliers.len() {
                h = self.conv(
                    &format!("encoder.down.{level}.downsample.conv"),
                    &h.f_constant_pad_nd([0, 1, 0, 1])?,
                    2,
                    0,
                    e,
                )?;
            }
        }
        h = self.middle(rt, "encoder.mid", &h, e)?;
        h = self.norm("encoder.norm_out", &h, e)?.f_silu()?;
        h = self.conv("encoder.conv_out", &h, 1, 1, e)?;
        h = self.conv("quant_conv", &h, 1, 0, e)?;
        let parts = h.f_chunk(2, 1)?;
        let mean = {
            // DiagonalGaussianDistribution computes these even in mode();
            // retain their source lifetime before returning the mean view.
            let logvar = parts[1].f_clamp(-30., 20.)?;
            let _std = logvar.f_mul_scalar(0.5)?.f_exp()?;
            let _var = logvar.f_exp()?;
            parts[0].shallow_clone()
        };
        let s = mean.size();
        let z = mean
            .f_reshape([s[0], s[1], s[2] / 2, 2, s[3] / 2, 2])?
            .f_permute([0, 1, 3, 5, 2, 4])?
            .f_reshape([s[0], s[1] * 4, s[2] / 2, s[3] / 2])?;
        Ok(z.f_batch_norm(
            None::<Tensor>,
            None::<Tensor>,
            Some(self.weight("bn.running_mean", &z)?),
            Some(self.weight("bn.running_var", &z)?),
            false,
            0.1,
            1e-4,
            true,
        )?)
    }
    pub fn decode<E: VaeExecution>(
        &self,
        rt: &Runtime,
        z: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let scale = self
            .weight("bn.running_var", z)?
            .f_view([1, -1, 1, 1])?
            .f_add_scalar(1e-4)?
            .f_sqrt()?;
        let mean = self.weight("bn.running_mean", z)?.f_view([1, -1, 1, 1])?;
        let z = z.f_mul(&scale)?.f_add(&mean)?;
        let s = z.size();
        let z = z
            .f_reshape([s[0], self.config.z_dim, 2, 2, s[2], s[3]])?
            .f_permute([0, 1, 4, 2, 5, 3])?
            .f_reshape([s[0], self.config.z_dim, s[2] * 2, s[3] * 2])?;
        let z = self.conv("post_quant_conv", &z, 1, 0, e)?;
        let mut h = self.conv("decoder.conv_in", &z, 1, 1, e)?;
        h = self.middle(rt, "decoder.mid", &h, e)?;
        for level in (0..self.config.multipliers.len()).rev() {
            for block in 0..=self.config.residual_blocks {
                h = self.residual(&format!("decoder.up.{level}.block.{block}"), &h, e)?;
            }
            if level != 0 {
                let s = h.size();
                h = h.f_upsample_nearest2d([s[2] * 2, s[3] * 2], Some(2.), Some(2.))?;
                h = self.conv(&format!("decoder.up.{level}.upsample.conv"), &h, 1, 1, e)?;
            }
        }
        h = self.norm("decoder.norm_out", &h, e)?.f_silu()?;
        self.conv("decoder.conv_out", &h, 1, 1, e)
    }
}
