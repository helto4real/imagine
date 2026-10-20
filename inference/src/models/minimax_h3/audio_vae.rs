//! Original MiniMax H3 FP32 audio VAE: DAC encoder, causal projection and
//! alias-free BigVGAN decoder. The public AV boundary returns raw stereo PCM.
use crate::{
    checkpoint::PreparedParameters,
    execution::Runtime,
    models::qwen_image21::{Linear, TextEncoderExecution},
    native::Error,
};
use tch::{Kind, Tensor};

pub struct Config {
    pub encoder_rates: Vec<i64>,
    pub attention_heads: i64,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            encoder_rates: vec![2, 4, 4, 5, 5],
            attention_heads: 8,
        }
    }
}
pub struct AudioVae {
    parameters: PreparedParameters,
    pub config: Config,
}
impl AudioVae {
    pub fn new(parameters: PreparedParameters, config: Config) -> Self {
        Self { parameters, config }
    }
    fn get(&self, n: &str) -> Result<Tensor, Error> {
        self.parameters
            .get(n)
            .ok_or_else(|| Error::BindingGap {
                backend: "minimax_h3_audio_vae".into(),
                operation: format!("Missing {n}"),
            })?
            .as_tensor()
    }
    fn cast(&self, n: &str, x: &Tensor) -> Result<Tensor, Error> {
        Ok(self
            .get(n)?
            .f_to_device_(x.device(), x.kind(), false, false)?)
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the source convolution stride, padding, dilation and transpose operands explicit"
    )]
    fn conv<E: TextEncoderExecution>(
        &self,
        n: &str,
        x: &Tensor,
        stride: i64,
        padding: i64,
        dilation: i64,
        transpose: bool,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let w = self.cast(&format!("{n}.weight"), x)?;
        let b = self
            .parameters
            .get(&format!("{n}.bias"))
            .map(|v| {
                v.as_tensor()?
                    .f_to_device_(x.device(), x.kind(), false, false)
                    .map_err(Error::from)
            })
            .transpose()?;
        Ok(if transpose {
            x.f_conv_transpose1d(&w, b.as_ref(), [stride], [padding], [0], 1, [dilation])?
        } else {
            x.f_conv1d(&w, b.as_ref(), [stride], [padding], [dilation], 1)?
        })
    }
    unsafe fn linear<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        n: &str,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let layer = Linear::from_parameters(
            &self.parameters,
            n,
            self.parameters.contains_key(&format!("{n}.bias")),
        )?;
        unsafe { layer.forward(rt, x) }
    }
    fn layer_norm<E: TextEncoderExecution>(
        &self,
        n: &str,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let w = self.cast(&format!("{n}.weight"), x)?;
        let b = self.cast(&format!("{n}.bias"), x)?;
        Ok(x.f_layer_norm(w.size().as_slice(), Some(&w), Some(&b), 1e-5, true)?)
    }
    fn snake(&self, n: &str, x: &Tensor, log: bool) -> Result<Tensor, Error> {
        let alpha = self.cast(&format!("{n}.alpha"), x)?;
        let (alpha, beta) = if log {
            (
                alpha.f_exp()?.f_view([1, -1, 1])?,
                self.cast(&format!("{n}.beta"), x)?
                    .f_exp()?
                    .f_view([1, -1, 1])?,
            )
        } else {
            (alpha.shallow_clone(), alpha)
        };
        let mut t = alpha.f_mul(x)?.f_sin()?;
        let alias = t.shallow_clone();
        Ok(t.f_mul_(&alias)?
            .f_mul_(&beta.f_add_scalar(1e-9)?.f_reciprocal()?)?
            .f_add_(x)?)
    }
    fn activation(&self, n: &str, x: &Tensor) -> Result<Tensor, Error> {
        let channels = x.size()[1];
        let up = self
            .cast(&format!("{n}.upsample.filter"), x)?
            .f_expand([channels, -1, -1], false)?;
        let padded = x.f_pad([5, 5], "replicate", None)?;
        let up = padded
            .f_conv_transpose1d(&up, None::<&Tensor>, [2], [0], [0], channels, [1])?
            .f_mul_scalar_(2.)?;
        let up = up.f_slice(2, 15, -15, 1)?;
        let act = self.snake(&format!("{n}.act"), &up, true)?;
        let filter = self
            .cast(&format!("{n}.downsample.lowpass.filter"), &act)?
            .f_expand([channels, -1, -1], false)?;
        Ok(act.f_pad([5, 6], "replicate", None)?.f_conv1d(
            &filter,
            None::<&Tensor>,
            [2],
            [0],
            [1],
            channels,
        )?)
    }
    fn residual<E: TextEncoderExecution>(
        &self,
        n: &str,
        x: &Tensor,
        dilation: i64,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let y = self.snake(&format!("{n}.block.0"), x, false)?;
        let y = self.conv(
            &format!("{n}.block.1"),
            &y,
            1,
            3 * dilation,
            dilation,
            false,
            exec,
        )?;
        let y = self.snake(&format!("{n}.block.2"), &y, false)?;
        let mut y = self.conv(&format!("{n}.block.3"), &y, 1, 0, 1, false, exec)?;
        let pad = (x.size()[2] - y.size()[2]) / 2;
        let x = if pad > 0 {
            x.f_slice(2, pad, -pad, 1)?
        } else {
            x.shallow_clone()
        };
        Ok(y.f_add_(&x)?)
    }
    unsafe fn encoder<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let mut x = self.conv("encoder.block.0", x, 1, 3, 1, false, exec)?;
        for (i, stride) in self.config.encoder_rates.iter().enumerate() {
            let prefix = format!("encoder.block.{}.block", i + 1);
            for (j, d) in [1, 3, 9].into_iter().enumerate() {
                x = self.residual(&format!("{prefix}.{j}"), &x, d, exec)?;
            }
            x = self.snake(&format!("{prefix}.3"), &x, false)?;
            x = self.conv(
                &format!("{prefix}.4"),
                &x,
                *stride,
                (*stride + 1) / 2,
                1,
                false,
                exec,
            )?;
        }
        let n = self.config.encoder_rates.len() + 1;
        x = self.snake(&format!("encoder.block.{n}"), &x, false)?;
        x = self.conv(
            &format!("encoder.block.{}", n + 1),
            &x,
            1,
            1,
            1,
            false,
            exec,
        )?;
        let x = x.f_transpose(1, 2)?;
        let mut base = unsafe {
            self.linear(
                rt,
                "pre_block.proj",
                &self.layer_norm("pre_block.norm3", &x, exec)?,
                exec,
            )?
        };
        let attention_input = self.layer_norm("pre_block.norm1", &x, exec)?;
        let weight = self.cast("pre_block.attn.qkv.weight", &attention_input)?;
        let bias = Tensor::f_cat(
            &[
                self.get("pre_block.attn.q_bias")?,
                self.get("pre_block.attn.zero_k_bias")?,
                self.get("pre_block.attn.v_bias")?,
            ],
            0,
        )?
        .f_to_device_(x.device(), x.kind(), false, false)?;
        let qkv = attention_input.f_linear(&weight, Some(&bias))?;
        let shape = qkv.size();
        let heads = self.config.attention_heads;
        let head_dim = attention_input.size()[2] / heads;
        let qkv = qkv
            .f_reshape([shape[0], shape[1], 3, heads, head_dim])?
            .f_permute([2, 0, 3, 1, 4])?
            .f_unbind(0)?;
        // Original NVIDIA comfy.ops wrapper: small calls bypass the context;
        // large calls enable flash/cuDNN/efficient/math in that order. Preserve
        // the source's enabled-only prior order and context-exit reconstruction.
        let previous =
            if matches!(qkv[0].device(), tch::Device::Cuda(_)) && qkv[0].numel() >= 1024 * 128 {
                let state = rt.native.sdpa_state()?;
                let enabled = state[5..]
                    .iter()
                    .copied()
                    .filter(|b| state[*b as usize] != 0)
                    .collect::<Vec<_>>();
                rt.native.set_sdpa_state([1, 1, 1, 1, 0, 1, 3, 2, 0, 4])?;
                Some(enabled)
            } else {
                None
            };
        let attention = Tensor::f_scaled_dot_product_attention(
            &qkv[0],
            &qkv[1],
            &qkv[2],
            None::<&Tensor>,
            0.,
            true,
            None,
            false,
        );
        if let Some(previous) = previous {
            let current = rt.native.sdpa_state()?;
            let mut restored = [0; 10];
            for backend in &previous {
                restored[*backend as usize] = 1;
            }
            let mut order = previous;
            for backend in &current[5..] {
                if !order.contains(backend) {
                    order.push(*backend);
                }
            }
            restored[5..].copy_from_slice(&order);
            rt.native.set_sdpa_state(restored)?;
        }
        let attention = attention?;
        let out_dim = self.get("pre_block.attn.proj.weight")?.size()[0];
        let pooled = attention
            .f_mean_dim([1].as_slice(), false, None)?
            .f_adaptive_avg_pool1d([out_dim])?;
        let attn = unsafe { self.linear(rt, "pre_block.attn.proj", &pooled, exec)? };
        let mut x = base.f_add_(&attn)?;
        let normalized = self.layer_norm("pre_block.norm2", &x, exec)?;
        let normalized = self.layer_norm("pre_block.mlp.norm", &normalized, exec)?;
        let mut w0 =
            unsafe { self.linear(rt, "pre_block.mlp.w0", &normalized, exec)? }.f_gelu("tanh")?;
        let w1 = unsafe { self.linear(rt, "pre_block.mlp.w1", &normalized, exec)? };
        let mlp = unsafe { self.linear(rt, "pre_block.mlp.w2", &w0.f_mul_(&w1)?, exec)? };
        x = x.f_add_(&mlp)?;
        self.conv("mean_proj", &x.f_transpose(1, 2)?, 1, 0, 1, false, exec)
    }
    fn amp_block<E: TextEncoderExecution>(
        &self,
        n: &str,
        x: &Tensor,
        kernel: i64,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let mut x = x.shallow_clone();
        for (i, d) in [1, 3, 5].into_iter().enumerate() {
            let a = self.activation(&format!("{n}.activations.{}", i * 2), &x)?;
            let c = self.conv(
                &format!("{n}.convs1.{i}"),
                &a,
                1,
                (kernel * d - d) / 2,
                d,
                false,
                exec,
            )?;
            let a = self.activation(&format!("{n}.activations.{}", i * 2 + 1), &c)?;
            let mut c = self.conv(
                &format!("{n}.convs2.{i}"),
                &a,
                1,
                (kernel - 1) / 2,
                1,
                false,
                exec,
            )?;
            x = c.f_add_(&x)?;
        }
        Ok(x)
    }
    fn decoder<E: TextEncoderExecution>(&self, z: &Tensor, exec: &mut E) -> Result<Tensor, Error> {
        let mut x = self.conv("dec_in_proj", z, 1, 0, 1, false, exec)?;
        x = self.conv("decoder.conv_pre", &x, 1, 3, 1, false, exec)?;
        for (i, (stride, kernel)) in [(5, 9), (5, 9), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)]
            .into_iter()
            .enumerate()
        {
            x = self.conv(
                &format!("decoder.ups.{i}.0"),
                &x,
                stride,
                (kernel - stride) / 2,
                1,
                true,
                exec,
            )?;
            let mut sum = None;
            for (j, k) in [3, 7, 11].into_iter().enumerate() {
                let value =
                    self.amp_block(&format!("decoder.resblocks.{}", i * 3 + j), &x, k, exec)?;
                sum = Some(if let Some(mut s) = sum {
                    Tensor::f_add_(&mut s, &value)?
                } else {
                    value
                });
            }
            x = sum.unwrap().f_div_scalar_(3.)?;
        }
        x = self.activation("decoder.activation_post", &x)?;
        Ok(self
            .conv("decoder.conv_post", &x, 1, 3, 1, false, exec)?
            .f_clamp_(-1., 1.)?)
    }
    fn require_fp32(x: &Tensor) -> Result<(), Error> {
        if x.kind() != Kind::Float {
            Err(Error::BindingGap {
                backend: "minimax_h3_audio_vae".into(),
                operation: "Selected audio VAE wrapper supplies FP32 tensors".into(),
            })
        } else {
            Ok(())
        }
    }
    /// # Safety
    /// Original FP32 convolution/linear and callback contracts hold.
    pub unsafe fn encode<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        waveform: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        Self::require_fp32(waveform)?;
        let shape = waveform.size();
        let (b, s, length) = (shape[0], shape[1], shape[2]);
        let hop = self.config.encoder_rates.iter().product::<i64>();
        let pad = ((length + hop - 1) / hop) * hop - length;
        let padded = waveform.f_pad([0, pad], "constant", None)?;
        let z = unsafe { self.encoder(rt, &padded.f_reshape([b * s, 1, -1])?, exec)? };
        let mean = self.cast("latents_mean", &z)?.f_view([1, -1, 1])?;
        let std = self.cast("latents_std", &z)?.f_view([1, -1, 1])?;
        let z = z.f_sub(&mean)?.f_div(&std)?;
        Ok(z.f_reshape([b, s, z.size()[1], z.size()[2]])?
            .f_permute([0, 2, 1, 3])?)
    }
    /// # Safety
    /// Original FP32 convolution and callback contracts hold. Output is PCM at
    /// 32 kHz; container encoding remains outside the native model boundary.
    pub unsafe fn decode<E: TextEncoderExecution>(
        &self,
        _rt: &Runtime,
        latent: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        Self::require_fp32(latent)?;
        let shape = latent.size();
        let (b, c, s, t) = (shape[0], shape[1], shape[2], shape[3]);
        let z = latent.f_permute([0, 2, 1, 3])?.f_reshape([b * s, c, t])?;
        let mean = self.cast("latents_mean", &z)?.f_view([1, -1, 1])?;
        let std = self.cast("latents_std", &z)?.f_view([1, -1, 1])?;
        let z = z.f_mul(&std)?.f_add(&mean)?;
        Ok(self.decoder(&z, exec)?.f_reshape([b, s, -1])?)
    }
}
