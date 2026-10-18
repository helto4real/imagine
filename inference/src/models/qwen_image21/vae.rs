//! Qwen Image 2.1's Wan2.2 image branch (temporal_kernel=1, patch_size=1).
//! Temporal streaming and the outer memory/tile policy have separate bindings.
use crate::{
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value,
    native::{Error, SdpaOptions},
};
use tch::Tensor;

pub const VAE_STRIP_ELEMS: usize = 1 << 24;

/// Original row-strip workspace bound, including its in-place additive shortcut.
pub fn vae_strip_apply(
    x: &Tensor,
    scale: i64,
    halo: i64,
    mut out: Option<Tensor>,
    mut f: impl FnMut(&Tensor) -> Result<Tensor, Error>,
) -> Result<Tensor, Error> {
    let n = (x.numel() * (scale * scale) as usize).div_ceil(VAE_STRIP_ELEMS);
    if n <= 1 && out.is_none() {
        return f(x);
    }
    let add = out.is_some();
    let size = x.size()[x.dim() - 2];
    let step = (size + n as i64 - 1) / n as i64;
    let mut a = 0;
    while a < size {
        let b = size.min(a + step);
        let lo = 0.max(a - halo);
        let y = f(&x.f_narrow(-2, lo, size.min(b + halo) - lo)?)?.f_narrow(
            -2,
            (a - lo) * scale,
            (b - a) * scale,
        )?;
        if out.is_none() {
            let mut shape = y.size();
            let rank = shape.len();
            shape[rank - 2] = size * scale;
            out = Some(Tensor::f_empty(&shape, (y.kind(), y.device()))?);
        }
        let mut dst = out
            .as_ref()
            .unwrap()
            .f_narrow(-2, a * scale, (b - a) * scale)?;
        if add {
            let _ = dst.f_add_(&y)?;
        } else {
            dst.f_copy_(&y)?;
        }
        a = b;
    }
    out.ok_or_else(|| Error::BindingGap {
        backend: "wan_vae".into(),
        operation: "empty strip result".into(),
    })
}

#[derive(Clone, Debug)]
pub struct ImageVaeConfig {
    pub dim: i64,
    pub dec_dim: i64,
    pub z_dim: i64,
    pub image_channels: i64,
    pub dim_mult: Vec<i64>,
    pub num_res_blocks: usize,
    pub temporal_downsample: Vec<bool>,
}
impl Default for ImageVaeConfig {
    fn default() -> Self {
        Self {
            dim: 160,
            dec_dim: 256,
            z_dim: 64,
            image_channels: 4,
            dim_mult: vec![1, 2, 4, 8, 8],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true, true],
        }
    }
}
pub trait VaeExecution {
    fn check_interrupt(&mut self) -> Result<(), Error>;
    /// The source-selected VAE attention, bound at model construction.
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, Error>;
}
/// Original `pytorch_attention`, including contiguous B1SC inputs.
pub fn vae_pytorch_attention(
    rt: &Runtime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
) -> Result<Tensor, Error> {
    let s = q.size();
    let shape = |t: &Tensor| -> Result<Tensor, Error> {
        Ok(t.f_view([s[0], 1, s[1], -1])?
            .f_transpose(2, 3)?
            .f_contiguous()?)
    };
    let out = rt.native.comfy_sdpa(
        &shape(q)?,
        &shape(k)?,
        &shape(v)?,
        SdpaOptions {
            mask: None,
            scale: None,
            enable_gqa: false,
            priority_wrapper: true,
        },
    )?;
    Ok(out.f_transpose(2, 3)?.f_reshape(&s)?)
}
pub struct ImageVae {
    pub config: ImageVaeConfig,
    state: PreparedParameters,
}
impl ImageVae {
    pub fn new(config: ImageVaeConfig, state: PreparedParameters) -> Self {
        Self { config, state }
    }
    fn parameter(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        self.state
            .get(name)
            .ok_or_else(|| Error::BindingGap {
                backend: "wan_vae".into(),
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
        rank: usize,
        stride: i64,
        padding: i64,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        let w = self.parameter(&format!("{name}.weight"), x)?;
        let b = self.parameter(&format!("{name}.bias"), x)?;
        // Original Conv3d autopad slices the temporal weight even for a singleton.
        if rank == 3 {
            Ok(x.f_conv3d(
                &w.f_narrow(
                    2,
                    w.size()[2] - x.size()[2].min(w.size()[2]),
                    x.size()[2].min(w.size()[2]),
                )?,
                Some(&b),
                [1, stride, stride],
                [0, padding, padding],
                [1, 1, 1],
                1,
            )?)
        } else {
            Ok(x.f_conv2d(
                &w,
                Some(&b),
                [stride, stride],
                [padding, padding],
                [1, 1],
                1,
            )?)
        }
    }
    fn norm(&self, rt: &Runtime, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        // F.normalize: linalg_vector_norm(p=2), clamp_min(1e-12), expand, divide.
        let denom = unsafe {
            rt.op(
                "aten::linalg_vector_norm",
                &[
                    Value::tensor(x.shallow_clone()),
                    Value::Float(2.),
                    Value::List(vec![Value::Int(1)]),
                    Value::Bool(true),
                    Value::None,
                ],
                &vec![],
            )?
        }
        .as_tensor()?;
        let normalized = x.f_div(&denom.f_clamp_min(1e-12)?.f_expand_as(x)?)?;
        Ok(normalized
            .f_mul_scalar((x.size()[1] as f64).sqrt())?
            .f_mul(&self.parameter(&format!("{name}.gamma"), x)?)?
            .f_add_scalar(0)?)
    }
    fn residual<E: VaeExecution>(
        &self,
        rt: &Runtime,
        name: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        vae_strip_apply(x, 1, 2, None, |s| {
            let y = self.norm(rt, &format!("{name}.residual.0"), s)?.f_silu()?;
            let y = self.conv(&format!("{name}.residual.2"), &y, 3, 1, 1, e)?;
            let y = self.norm(rt, &format!("{name}.residual.3"), &y)?.f_silu()?;
            let mut y = self.conv(&format!("{name}.residual.6"), &y, 3, 1, 1, e)?;
            let shortcut = if s.size()[1] != y.size()[1] {
                self.conv(&format!("{name}.shortcut"), s, 3, 1, 0, e)?
            } else {
                s.shallow_clone()
            };
            let _ = y.f_add_(&shortcut)?;
            Ok(y)
        })
    }
    fn middle<E: VaeExecution>(
        &self,
        rt: &Runtime,
        name: &str,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let x = self.residual(rt, &format!("{name}.0"), x, e)?;
        let s = x.size();
        let to4 = |v: &Tensor| -> Result<Tensor, Error> {
            Ok(v.f_permute([0, 2, 1, 3, 4])?
                .f_reshape([s[0] * s[2], s[1], s[3], s[4]])?)
        };
        let y = self.norm(rt, &format!("{name}.1.norm"), &to4(&x)?)?;
        let qkv = self
            .conv(&format!("{name}.1.to_qkv"), &y, 2, 1, 0, e)?
            .f_chunk(3, 1)?;
        let y = e.attention(rt, &qkv[0], &qkv[1], &qkv[2])?;
        let y = self
            .conv(&format!("{name}.1.proj"), &y, 2, 1, 0, e)?
            .f_reshape([s[0], s[2], s[1], s[3], s[4]])?
            .f_permute([0, 2, 1, 3, 4])?
            .f_add(&x)?;
        self.residual(rt, &format!("{name}.2"), &y, e)
    }
    fn resample<E: VaeExecution>(
        &self,
        name: &str,
        x: &Tensor,
        up: bool,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let s = x.size();
        let y = x
            .f_permute([0, 2, 1, 3, 4])?
            .f_reshape([s[0] * s[2], s[1], s[3], s[4]])?;
        let y = if up {
            vae_strip_apply(&y, 2, 1, None, |v| {
                let shape = v.size();
                let v = v.f_internal_upsample_nearest_exact2d(
                    [shape[2] * 2, shape[3] * 2],
                    Some(2.),
                    Some(2.),
                )?;
                self.conv(&format!("{name}.resample.1"), &v, 2, 1, 1, e)
            })?
        } else {
            self.conv(
                &format!("{name}.resample.1"),
                &y.f_constant_pad_nd([0, 1, 0, 1])?,
                2,
                2,
                0,
                e,
            )?
        };
        let d = y.size();
        Ok(y.f_reshape([s[0], s[2], d[1], d[2], d[3]])?
            .f_permute([0, 2, 1, 3, 4])?)
    }
    /// Source image decode: no temporal cache is instantiated.
    pub fn decode<E: VaeExecution>(
        &self,
        rt: &Runtime,
        z: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        if z.dim() != 4 {
            return Err(Error::BindingGap {
                backend: "wan_vae".into(),
                operation: "temporal VAE decode".into(),
            });
        }
        let mut x = self.conv("conv2", &z.f_unsqueeze(2)?, 3, 1, 0, e)?;
        x = self.conv("decoder.conv1", &x, 3, 1, 1, e)?;
        x = self.middle(rt, "decoder.middle", &x, e)?;
        let mut dims = vec![*self.config.dim_mult.last().unwrap()];
        dims.extend(self.config.dim_mult.iter().rev().copied());
        for i in 0..self.config.dim_mult.len() {
            let input = x.shallow_clone();
            let name = format!("decoder.upsamples.{i}");
            for j in 0..=self.config.num_res_blocks {
                x = self.residual(rt, &format!("{name}.upsamples.{j}"), &x, e)?;
            }
            if i + 1 < self.config.dim_mult.len() {
                x = self.resample(
                    &format!("{name}.upsamples.{}", self.config.num_res_blocks + 1),
                    &x,
                    true,
                    e,
                )?;
                let ft = if self
                    .config
                    .temporal_downsample
                    .iter()
                    .rev()
                    .nth(i)
                    .copied()
                    .unwrap_or(false)
                {
                    2
                } else {
                    1
                };
                let out_c = self.config.dec_dim * dims[i + 1];
                x = vae_strip_apply(&input, 2, 0, Some(x), |s| dup_up(s, out_c, ft, 2, true))?;
            }
        }
        x = self.norm(rt, "decoder.head.0", &x)?.f_silu()?;
        Ok(self
            .conv("decoder.head.2", &x, 3, 1, 1, e)?
            .f_squeeze_dim(2)?)
    }
    pub fn encode<E: VaeExecution>(
        &self,
        rt: &Runtime,
        pixels: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        if pixels.dim() != 4 {
            return Err(Error::BindingGap {
                backend: "wan_vae".into(),
                operation: "temporal VAE encode".into(),
            });
        }
        let mut x = self.conv("encoder.conv1", &pixels.f_unsqueeze(2)?, 3, 1, 1, e)?;
        for (i, m) in self.config.dim_mult.iter().enumerate() {
            let original = x.shallow_clone();
            let name = format!("encoder.downsamples.{i}");
            for j in 0..self.config.num_res_blocks {
                x = self.residual(rt, &format!("{name}.downsamples.{j}"), &x, e)?;
            }
            let down = i + 1 < self.config.dim_mult.len();
            if down {
                x = self.resample(
                    &format!("{name}.downsamples.{}", self.config.num_res_blocks),
                    &x,
                    false,
                    e,
                )?;
            }
            let ft = if self
                .config
                .temporal_downsample
                .get(i)
                .copied()
                .unwrap_or(false)
            {
                2
            } else {
                1
            };
            x = x.f_add(&avg_down(
                &original,
                self.config.dim * m,
                ft,
                if down { 2 } else { 1 },
            )?)?;
        }
        x = self.middle(rt, "encoder.middle", &x, e)?;
        x = self.norm(rt, "encoder.head.0", &x)?.f_silu()?;
        x = self.conv("encoder.head.2", &x, 3, 1, 1, e)?;
        Ok(self.conv("conv1", &x, 3, 1, 0, e)?.f_chunk(2, 1)?[0].f_squeeze_dim(2)?)
    }
}
pub fn dup_up(x: &Tensor, out_c: i64, ft: i64, fs: i64, first: bool) -> Result<Tensor, Error> {
    let s = x.size();
    let repeats = out_c * ft * fs * fs / s[1];
    let y = x
        .f_repeat_interleave_self_int(repeats, Some(1), None)?
        .f_view(&[s[0], out_c, ft, fs, fs, s[2], s[3], s[4]][..])?
        .f_permute([0, 1, 5, 2, 6, 3, 7, 4])?
        .f_contiguous()?
        .f_view([s[0], out_c, s[2] * ft, s[3] * fs, s[4] * fs])?;
    Ok(if first {
        y.f_narrow(2, ft - 1, y.size()[2] - (ft - 1))?
    } else {
        y
    })
}
pub fn avg_down(x: &Tensor, out_c: i64, ft: i64, fs: i64) -> Result<Tensor, Error> {
    let pad = (ft - x.size()[2] % ft) % ft;
    let x = x.f_constant_pad_nd([0, 0, 0, 0, pad, 0])?;
    let s = x.size();
    let factor = ft * fs * fs;
    Ok(
        x.f_view(&[s[0], s[1], s[2] / ft, ft, s[3] / fs, fs, s[4] / fs, fs][..])?
            .f_permute([0, 1, 3, 5, 7, 2, 4, 6])?
            .f_contiguous()?
            .f_view([s[0], s[1] * factor, s[2] / ft, s[3] / fs, s[4] / fs])?
            .f_view([
                s[0],
                out_c,
                s[1] * factor / out_c,
                s[2] / ft,
                s[3] / fs,
                s[4] / fs,
            ])?
            .f_mean_dim(&[2i64][..], false, None)?,
    )
}
pub fn vae_process_output(image: &mut Tensor) -> Result<(), Error> {
    let _ = image.f_add_scalar_(1)?.f_div_scalar_(2)?.f_clamp_(0, 1)?;
    Ok(())
}
