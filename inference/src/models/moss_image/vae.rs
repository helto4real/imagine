//! Flux.2 AutoencoderKL: original 2D residual VAE and latent BatchNorm packing.
use crate::models::qwen_image21::VaeExecution;
use crate::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value as V, native::Error,
};
use tch::{Device, Kind, Tensor};

#[derive(Clone)]
pub struct Flux2VaeConfig {
    pub dim: i64,
    pub decoder_dim: i64,
    pub z_dim: i64,
    pub multipliers: Vec<i64>,
    pub residual_blocks: usize,
    /// ComfyUI d07dff45: fuse GroupNorm/SiLU/padding through Kitchen and run the
    /// convolutions in row strips when the device and dtype are served.
    pub fused_norm: bool,
    /// Elements per strip (`STRIP_ELEMS`).
    pub strip_elements: i64,
    /// PyTorch's fp16 accumulation opt-in, which routes strip convolutions
    /// through Kitchen's fp16 conv.
    pub fp16_accumulation: bool,
}
impl Default for Flux2VaeConfig {
    fn default() -> Self {
        Self {
            dim: 128,
            decoder_dim: 128,
            z_dim: 32,
            multipliers: vec![1, 2, 4, 4],
            residual_blocks: 2,
            fused_norm: true,
            strip_elements: 1 << 24,
            fp16_accumulation: false,
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
        if self.served(x) {
            return self.encode_fused(rt, x, e);
        }
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
        self.encode_tail(&h, e)
    }
    fn encode_tail<E: VaeExecution>(&self, h: &Tensor, e: &mut E) -> Result<Tensor, Error> {
        let h = self.conv("quant_conv", h, 1, 0, e)?;
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
        if self.served(&z) {
            let mut out = Vec::new();
            for i in 0..z.size()[0] {
                out.push(self.decode_fused(rt, &z.f_narrow(0, i, 1)?, e)?);
            }
            return Ok(if out.len() == 1 {
                out.pop().unwrap()
            } else {
                Tensor::f_cat(&out, 0)?
            });
        }
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

/// ComfyUI `region`: the channels_last tensor of that shape ending at element
/// `end` of the flat buffer.
fn region(buf: &Tensor, end: i64, channels: i64, h: i64, w: i64) -> Result<Tensor, Error> {
    let n = channels * h * w;
    Ok(buf
        .f_narrow(0, end - n, n)?
        .f_view([1, h, w, channels])?
        .f_permute([0, 3, 1, 2])?)
}
/// ComfyUI `interleaved_view`: x's storage as two half-width tensors whose rows
/// alternate inside each row slot.
fn interleaved_view(x: &Tensor, channels: i64, part: i64) -> Result<Tensor, Error> {
    let s = x.size();
    let (h, w, c) = (s[2], s[3], s[1]);
    let flat = x.f_permute([0, 2, 3, 1])?.f_reshape([-1])?;
    Ok(flat.f_as_strided(
        [1, channels, h, w],
        [h * w * c, 1, w * c, channels],
        Some(flat.storage_offset() + part * w * channels),
    )?)
}
/// `-(-a // b)` for positive b.
fn ceil_div(a: i64, b: i64) -> i64 {
    -(-a).div_euclid(b)
}
impl Flux2Vae {
    /// ComfyUI `fused_norm_served` on CUDA (not ROCm).
    fn served(&self, x: &Tensor) -> bool {
        self.config.fused_norm
            && matches!(x.device(), Device::Cuda(_))
            && matches!(x.kind(), Kind::Half | Kind::BFloat16)
    }
    fn channels_last(rt: &Runtime, x: &Tensor) -> Result<Tensor, Error> {
        unsafe {
            rt.op(
                "aten::contiguous",
                &[V::tensor(x.shallow_clone())],
                &vec![("memory_format".into(), V::Int(2))],
            )?
        }
        .as_tensor()
    }
    /// `torch.empty(shape, memory_format=torch.channels_last)`.
    fn empty_channels_last(shape: &[i64], like: &Tensor) -> Result<Tensor, Error> {
        Ok(Tensor::f_empty(
            [shape[0], shape[2], shape[3], shape[1]],
            (like.kind(), like.device()),
        )?
        .f_permute([0, 3, 1, 2])?)
    }
    /// ComfyUI `conv_weight`: the fp16-accumulation opt-in takes a 5-D weight.
    fn conv_weight(&self, rt: &Runtime, x: &Tensor, weight: Tensor) -> Result<Tensor, Error> {
        if self.config.fp16_accumulation && x.kind() == Kind::Half && x.device().is_cuda() {
            return unsafe {
                rt.op(
                    "aten::contiguous",
                    &[V::tensor(weight.f_unsqueeze(2)?)],
                    &vec![("memory_format".into(), V::Int(3))],
                )?
            }
            .as_tensor();
        }
        Ok(weight)
    }
    /// ComfyUI `padded_conv`: a 5-D weight runs Kitchen's fp16 conv on the padded
    /// strip as a depth-1 conv3d.
    fn padded_conv(
        rt: &Runtime,
        padded: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor, Error> {
        if weight.dim() == 5 {
            return unsafe {
                rt.call_kitchen(
                    "fp16_conv3d",
                    &vec![
                        ("x".into(), V::tensor(padded.shallow_clone())),
                        ("weight".into(), V::tensor(weight.shallow_clone())),
                        ("bias".into(), V::tensor(bias.shallow_clone())),
                        ("residual".into(), V::None),
                        ("stride".into(), V::Int(1)),
                    ],
                )?
            }
            .as_tensor()?
            .f_squeeze_dim(2)
            .map_err(Error::from);
        }
        Ok(padded
            .f_squeeze_dim(2)?
            .f_conv2d(weight, Some(bias), [1, 1], [0, 0], [1, 1], 1)?)
    }
    fn group_norm_stats(rt: &Runtime, x: &Tensor, groups: i64, eps: f64) -> Result<Tensor, Error> {
        unsafe {
            rt.call_kitchen(
                "group_norm_stats3d",
                &vec![
                    ("x".into(), V::tensor(x.f_unsqueeze(2)?)),
                    ("num_groups".into(), V::Int(groups)),
                    ("eps".into(), V::Float(eps)),
                ],
            )?
        }
        .as_tensor()
    }
    #[allow(clippy::too_many_arguments)]
    fn group_norm_pad(
        rt: &Runtime,
        rows: &Tensor,
        norm: Option<(&Tensor, &Tensor)>,
        groups: i64,
        eps: f64,
        pad: [i64; 5],
        silu: bool,
        stats: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (weight, bias) = match norm {
            Some((w, b)) => (V::tensor(w.shallow_clone()), V::tensor(b.shallow_clone())),
            None => (V::None, V::None),
        };
        unsafe {
            rt.call_kitchen(
                "group_norm_silu_pad3d",
                &vec![
                    ("x".into(), V::tensor(rows.f_unsqueeze(2)?)),
                    ("weight".into(), weight),
                    ("bias".into(), bias),
                    ("num_groups".into(), V::Int(groups)),
                    ("eps".into(), V::Float(eps)),
                    (
                        "pad".into(),
                        V::List(pad.iter().map(|&p| V::Int(p)).collect()),
                    ),
                    ("silu".into(), V::Bool(silu)),
                    ("zero_pad".into(), V::Bool(true)),
                    (
                        "stats".into(),
                        stats.map_or(V::None, |s| V::tensor(s.shallow_clone())),
                    ),
                ],
            )?
        }
        .as_tensor()
    }
    /// ComfyUI `strip_rows`: `f(rows, top, bottom)` runs a 3x3 conv on a strip
    /// with a one-row halo, padding only at the image edge.
    fn strip_rows(
        &self,
        f: &mut dyn FnMut(&Tensor, i64, i64) -> Result<Tensor, Error>,
        x: &Tensor,
        out_elements: i64,
        scale: i64,
        mut out: Option<Tensor>,
        accumulate: bool,
    ) -> Result<Tensor, Error> {
        let size = x.size()[2];
        let strips = ceil_div(
            x.numel().max(out_elements as usize) as i64,
            self.config.strip_elements,
        );
        let step = ceil_div(size, strips);
        let mut a = 0;
        while a < size {
            let b = size.min(a + step);
            let (lo, hi) = ((a - 1).max(0), size.min(b + 1));
            let top = i64::from(lo == 0);
            let rows = f(&x.f_narrow(2, lo, hi - lo)?, top, i64::from(hi == size))?;
            let rows = rows.f_narrow(2, scale * (a - lo) + top - 1, scale * (b - a))?;
            let target = match &out {
                Some(o) => o.shallow_clone(),
                None => {
                    let r = rows.size();
                    let o = Self::empty_channels_last(&[r[0], r[1], size * scale, r[3]], &rows)?;
                    out = Some(o.shallow_clone());
                    o
                }
            };
            let mut dst = target.f_narrow(2, scale * a, scale * (b - a))?;
            if accumulate {
                let _ = dst.f_add_(&rows)?;
            } else {
                dst.f_copy_(&rows)?;
            }
            a += step;
        }
        Ok(out.unwrap())
    }
    /// ComfyUI `strip_conv`: conv(silu(norm(x))) in row strips, after a nearest
    /// upsample when scale > 1.
    #[allow(clippy::too_many_arguments)]
    fn strip_conv<E: VaeExecution>(
        &self,
        rt: &Runtime,
        conv: &str,
        x: &Tensor,
        norm: Option<&str>,
        scale: i64,
        out: Option<Tensor>,
        accumulate: bool,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        let (groups, eps) = if norm.is_some() { (32, 1e-6) } else { (1, 0.) };
        let stats = norm
            .map(|_| Self::group_norm_stats(rt, x, groups, eps))
            .transpose()?;
        let affine = norm
            .map(|n| -> Result<_, Error> {
                Ok((
                    self.weight(&format!("{n}.weight"), x)?,
                    self.weight(&format!("{n}.bias"), x)?,
                ))
            })
            .transpose()?;
        let weight = self.conv_weight(rt, x, self.weight(&format!("{conv}.weight"), x)?)?;
        let bias = self.weight(&format!("{conv}.bias"), x)?;
        let out_channels = weight.size()[0];
        let mut f = |rows: &Tensor, top: i64, bottom: i64| -> Result<Tensor, Error> {
            let rows = if scale > 1 {
                let s = rows.size();
                rows.f_upsample_nearest2d(
                    [s[2] * scale, s[3] * scale],
                    Some(scale as f64),
                    Some(scale as f64),
                )?
            } else {
                rows.shallow_clone()
            };
            let padded = Self::group_norm_pad(
                rt,
                &rows,
                affine.as_ref().map(|(w, b)| (w, b)),
                groups,
                eps,
                [1, 1, top, bottom, 0],
                norm.is_some(),
                stats.as_ref(),
            )?;
            Self::padded_conv(rt, &padded, &weight, &bias)
        };
        let s = x.size();
        let out_elements = s[0] * out_channels * s[2] * s[3] * scale * scale;
        self.strip_rows(&mut f, x, out_elements, scale, out, accumulate)
    }
    /// ComfyUI `strip_down`: the stride-2 conv with its (0, 1, 0, 1) pad in strips.
    fn strip_down<E: VaeExecution>(
        &self,
        rt: &Runtime,
        conv: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        let s = x.size();
        let rows = s[2] / 2;
        let step = ceil_div(rows, ceil_div(x.numel() as i64, self.config.strip_elements));
        let weight = self.weight(&format!("{conv}.weight"), x)?;
        let bias = self.weight(&format!("{conv}.bias"), x)?;
        let mut out: Option<Tensor> = None;
        let mut a = 0;
        while a < rows {
            let b = rows.min(a + step);
            let hi = s[2].min(2 * b + 1);
            let padded = Self::group_norm_pad(
                rt,
                &x.f_narrow(2, 2 * a, hi - 2 * a)?,
                None,
                1,
                0.,
                [0, 1, 0, i64::from(hi == s[2]), 0],
                false,
                None,
            )?
            .f_squeeze_dim(2)?;
            let y = padded.f_conv2d(&weight, Some(&bias), [2, 2], [0, 0], [1, 1], 1)?;
            let target = match &out {
                Some(o) => o.shallow_clone(),
                None => {
                    let ys = y.size();
                    let o = Self::empty_channels_last(&[s[0], ys[1], rows, ys[3]], &y)?;
                    out = Some(o.shallow_clone());
                    o
                }
            };
            target.f_narrow(2, a, b - a)?.f_copy_(&y)?;
            a += step;
        }
        Ok(out.unwrap())
    }
    fn has_shortcut(&self, name: &str) -> bool {
        self.state
            .contains_key(&format!("{name}.nin_shortcut.weight"))
    }
    /// ComfyUI `ResnetBlock.halves_in_place`.
    fn halves_in_place(&self, name: &str, x: &Tensor) -> Result<bool, Error> {
        let (cin, cout) = self.block_channels(name)?;
        Ok(x.size()[0] == 1 && cin == 2 * cout && Self::is_channels_last(x))
    }
    fn block_channels(&self, name: &str) -> Result<(i64, i64), Error> {
        let w = self
            .state
            .get(&format!("{name}.conv1.weight"))
            .ok_or_else(|| Error::BindingGap {
                backend: "flux2_vae".into(),
                operation: format!("Missing prepared {name}.conv1.weight"),
            })?;
        let shape = match w {
            V::Quantized(_) => w.as_quantized()?.params.original_shape,
            _ => w.as_tensor()?.size(),
        };
        Ok((shape[1], shape[0]))
    }
    /// PyTorch's 4-d channels_last contiguity: dimensions C, W, H, N in that
    /// order, size-one dimensions ignored.
    fn is_channels_last(x: &Tensor) -> bool {
        let (size, stride) = (x.size(), x.stride());
        let mut expected = 1;
        for d in [1, 3, 2, 0] {
            if size[d] != 1 {
                if stride[d] != expected {
                    return false;
                }
                expected *= size[d];
            }
        }
        true
    }
    /// ComfyUI `ResnetBlock.halve_in_place`: C -> C/2 with the hidden and output
    /// tensors interleaved inside x; a strip's last row waits until the next strip
    /// has read its halo.
    fn halve_in_place<E: VaeExecution>(
        &self,
        rt: &Runtime,
        name: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let (_, cout) = self.block_channels(name)?;
        let h = interleaved_view(x, cout, 0)?;
        let out = interleaved_view(x, cout, 1)?;
        let stats = Self::group_norm_stats(rt, x, 32, 1e-6)?;
        let rows = x.size()[2];
        let step = ceil_div(rows, ceil_div(x.numel() as i64, self.config.strip_elements));
        let norm = (
            self.weight(&format!("{name}.norm1.weight"), x)?,
            self.weight(&format!("{name}.norm1.bias"), x)?,
        );
        let weight = self.conv_weight(rt, x, self.weight(&format!("{name}.conv1.weight"), x)?)?;
        let bias = self.weight(&format!("{name}.conv1.bias"), x)?;
        let sc_weight = self.weight(&format!("{name}.nin_shortcut.weight"), x)?;
        let sc_bias = self.weight(&format!("{name}.nin_shortcut.bias"), x)?;
        let mut held: Option<(Tensor, Tensor)> = None;
        let mut a = 0;
        while a < rows {
            e.check_interrupt()?;
            let b = rows.min(a + step);
            let (lo, hi) = ((a - 1).max(0), rows.min(b + 1));
            let padded = Self::group_norm_pad(
                rt,
                &x.f_narrow(2, lo, hi - lo)?,
                Some((&norm.0, &norm.1)),
                32,
                1e-6,
                [1, 1, i64::from(lo == 0), i64::from(hi == rows), 0],
                true,
                Some(&stats),
            )?;
            let h_rows = Self::padded_conv(rt, &padded, &weight, &bias)?;
            let out_rows = x.f_narrow(2, a, b - a)?.f_conv2d(
                &sc_weight,
                Some(&sc_bias),
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )?;
            if let Some((hh, ho)) = &held {
                h.f_narrow(2, a - 1, 1)?.f_copy_(hh)?;
                out.f_narrow(2, a - 1, 1)?.f_copy_(ho)?;
            }
            h.f_narrow(2, a, b - a - 1)?
                .f_copy_(&h_rows.f_narrow(2, 0, b - a - 1)?)?;
            out.f_narrow(2, a, b - a - 1)?
                .f_copy_(&out_rows.f_narrow(2, 0, b - a - 1)?)?;
            held = Some((
                h_rows.f_narrow(2, b - a - 1, 1)?.copy(),
                out_rows.f_narrow(2, b - a - 1, 1)?.copy(),
            ));
            a += step;
        }
        let (hh, ho) = held.unwrap();
        h.f_narrow(2, rows - 1, 1)?.f_copy_(&hh)?;
        out.f_narrow(2, rows - 1, 1)?.f_copy_(&ho)?;
        self.strip_conv(
            rt,
            &format!("{name}.conv2"),
            &h,
            Some(&format!("{name}.norm2")),
            1,
            Some(out),
            true,
            e,
        )
    }
    /// ComfyUI's fused `ResnetBlock.forward`: the block consumes x, its residual
    /// stream updated in place.
    fn residual_fused<E: VaeExecution>(
        &self,
        rt: &Runtime,
        name: &str,
        x: &Tensor,
        hidden: Option<Tensor>,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        // NCHW after the mid-block attention.
        let x = if x.stride()[1] != 1 {
            Self::channels_last(rt, x)?
        } else {
            x.shallow_clone()
        };
        let (n1, n2) = (format!("{name}.norm1"), format!("{name}.norm2"));
        let (c1, c2) = (format!("{name}.conv1"), format!("{name}.conv2"));
        if !self.has_shortcut(name) {
            let h = self.strip_conv(rt, &c1, &x, Some(&n1), 1, hidden, false, e)?;
            return self.strip_conv(rt, &c2, &h, Some(&n2), 1, Some(x), true, e);
        }
        if self.halves_in_place(name, &x)? {
            return self.halve_in_place(rt, name, &x, e);
        }
        let h = self.strip_conv(rt, &c1, &x, Some(&n1), 1, None, false, e)?;
        let shortcut = self.conv(&format!("{name}.nin_shortcut"), &x, 1, 0, e)?;
        self.strip_conv(rt, &c2, &h, Some(&n2), 1, Some(shortcut), true, e)
    }
    fn middle_fused<E: VaeExecution>(
        &self,
        rt: &Runtime,
        path: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let x = self.residual_fused(rt, &format!("{path}.block_1"), x, None, e)?;
        let x = self.attention(rt, &format!("{path}.attn_1"), &x, e)?;
        self.residual_fused(rt, &format!("{path}.block_2"), &x, None, e)
    }
    /// ComfyUI's fused `Encoder.forward`, then `quant_conv` and the latent packing.
    fn encode_fused<E: VaeExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let x = Self::channels_last(rt, x)?;
        let mut h = self.conv("encoder.conv_in", &x, 1, 1, e)?;
        let levels = self.config.multipliers.len();
        for level in 0..levels {
            for block in 0..self.config.residual_blocks {
                h = self.residual_fused(
                    rt,
                    &format!("encoder.down.{level}.block.{block}"),
                    &h,
                    None,
                    e,
                )?;
            }
            if level + 1 < levels {
                h = self.strip_down(rt, &format!("encoder.down.{level}.downsample.conv"), &h, e)?;
            }
        }
        h = self.middle_fused(rt, "encoder.mid", &h, e)?;
        let h = self.strip_conv(
            rt,
            "encoder.conv_out",
            &h,
            Some("encoder.norm_out"),
            1,
            None,
            false,
            e,
        )?;
        self.encode_tail(&h, e)
    }
    /// ComfyUI `Decoder.forward_fused` for one latent: one buffer the size of the
    /// largest upsample output holds every level; each upsample writes it from the
    /// front while reading its input at the back, and the blocks take their hidden
    /// tensors from the rest.
    fn decode_fused<E: VaeExecution>(
        &self,
        rt: &Runtime,
        z: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let z = Self::channels_last(rt, z)?;
        let mut h = self.conv("decoder.conv_in", &z, 1, 1, e)?;
        h = self.middle_fused(rt, "decoder.mid", &h, e)?;
        let n = self.config.multipliers.len();
        let (hh, hw) = (h.size()[2], h.size()[3]);
        let block_out = |level: usize| self.config.decoder_dim * self.config.multipliers[level];
        let size = (1..n)
            .map(|i| block_out(i) * (hh << (n - i)) * (hw << (n - i)))
            .max()
            .unwrap();
        let buf = Tensor::f_empty([size], (h.kind(), h.device()))?;
        for level in (0..n).rev() {
            let mut hidden: Option<Tensor> = None;
            for block in 0..=self.config.residual_blocks {
                let name = format!("decoder.up.{level}.block.{block}");
                if hidden.is_none() {
                    let out = self.block_channels(&name)?.1;
                    hidden = Some(if self.halves_in_place(&name, &h)? {
                        interleaved_view(&h, out, 0)?
                    } else {
                        let end =
                            buf.numel() as i64 - if level < n - 1 { h.numel() as i64 } else { 0 };
                        region(&buf, end, out, h.size()[2], h.size()[3])?
                    });
                }
                h = self.residual_fused(
                    rt,
                    &name,
                    &h,
                    hidden.as_ref().map(Tensor::shallow_clone),
                    e,
                )?;
            }
            if level != 0 {
                let s = h.size();
                let out = region(&buf, buf.numel() as i64, s[1], 2 * s[2], 2 * s[3])?;
                h = self.strip_conv(
                    rt,
                    &format!("decoder.up.{level}.upsample.conv"),
                    &h,
                    None,
                    2,
                    Some(out),
                    false,
                    e,
                )?;
            }
        }
        self.strip_conv(
            rt,
            "decoder.conv_out",
            &h,
            Some("decoder.norm_out"),
            1,
            None,
            false,
            e,
        )
    }
}
