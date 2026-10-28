//! Depth Anything V2 as transformers' `DepthAnythingForDepthEstimation`: the DINOv2
//! backbone, the DPT reassemble/fusion neck and the relative depth head; its
//! `DPTImageProcessor` input; and the depth map Imagine makes from a prediction.
use crate::{
    models::qwen_image21::{Pillow, Resampling},
    native::Error,
};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

/// The backbone's attention implementation as transformers selects it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DinoAttention {
    Sdpa,
    Eager,
}
/// How a reassemble layer resizes its stage: a transposed convolution up by a
/// factor, unchanged, or a stride-two convolution down.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reassemble {
    Up(i64),
    Same,
    Down,
}
#[derive(Clone, Debug)]
pub struct DepthAnythingConfig {
    pub hidden: i64,
    pub heads: i64,
    pub layers: usize,
    pub patch: i64,
    /// Encoder hidden states (0 = embeddings) the neck reads.
    pub out_indices: Vec<usize>,
    pub neck: Vec<i64>,
    pub reassemble: Vec<Reassemble>,
    pub fusion: i64,
    pub head_hidden: i64,
    pub eps: f64,
    pub attention: DinoAttention,
}
impl DepthAnythingConfig {
    /// `depth-anything/Depth-Anything-V2-Small-hf`.
    pub fn small(attention: DinoAttention) -> Self {
        Self {
            hidden: 384,
            heads: 6,
            layers: 12,
            patch: 14,
            out_indices: vec![3, 6, 9, 12],
            neck: vec![48, 96, 192, 384],
            reassemble: vec![
                Reassemble::Up(4),
                Reassemble::Up(2),
                Reassemble::Same,
                Reassemble::Down,
            ],
            fusion: 64,
            head_hidden: 32,
            eps: 1e-6,
            attention,
        }
    }
}
pub struct DepthAnything {
    pub config: DepthAnythingConfig,
    weights: BTreeMap<String, Tensor>,
}
fn missing(name: &str) -> Error {
    Error::BindingGap {
        backend: "depth_anything".into(),
        operation: format!("Missing weight {name}"),
    }
}
impl DepthAnything {
    pub fn new(config: DepthAnythingConfig, weights: BTreeMap<String, Tensor>) -> Self {
        Self { config, weights }
    }
    fn w(&self, name: &str) -> Result<&Tensor, Error> {
        self.weights.get(name).ok_or_else(|| missing(name))
    }
    fn linear(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.f_linear(
            self.w(&format!("{name}.weight"))?,
            Some(self.w(&format!("{name}.bias"))?),
        )?)
    }
    fn conv(&self, name: &str, x: &Tensor, stride: i64, padding: i64) -> Result<Tensor, Error> {
        Ok(x.f_conv2d(
            self.w(&format!("{name}.weight"))?,
            self.weights.get(&format!("{name}.bias")),
            [stride, stride],
            [padding, padding],
            [1, 1],
            1,
        )?)
    }
    fn norm(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.f_layer_norm(
            [self.config.hidden],
            Some(self.w(&format!("{name}.weight"))?),
            Some(self.w(&format!("{name}.bias"))?),
            self.config.eps,
            true,
        )?)
    }
    /// Dinov2Embeddings: patches, the class token and position embeddings,
    /// resized bicubically unless the grid is the trained square one.
    fn embeddings(&self, pixels: &Tensor) -> Result<Tensor, Error> {
        let [b, _, h, w] = pixels.size()[..] else {
            unreachable!()
        };
        let name = "backbone.embeddings";
        let patches = self
            .conv(
                &format!("{name}.patch_embeddings.projection"),
                pixels,
                self.config.patch,
                0,
            )?
            .f_flatten(2, -1)?
            .f_transpose(1, 2)?;
        let cls = self
            .w(&format!("{name}.cls_token"))?
            .f_expand([b, -1, -1], false)?;
        let x = Tensor::f_cat(&[cls, patches], 1)?;
        let position = self.w(&format!("{name}.position_embeddings"))?;
        let patches = x.size()[1] - 1;
        let positions = position.size()[1] - 1;
        let position = if patches == positions && h == w {
            position.shallow_clone()
        } else {
            let dim = x.size()[2];
            let side = (positions as f64).sqrt() as i64;
            let grid = position
                .f_narrow(1, 1, positions)?
                .f_reshape([1, side, side, dim])?
                .f_permute([0, 3, 1, 2])?
                .f_to_kind(Kind::Float)?
                .f_upsample_bicubic2d(
                    [h / self.config.patch, w / self.config.patch],
                    false,
                    None,
                    None,
                )?
                .f_to_kind(position.kind())?
                .f_permute([0, 2, 3, 1])?
                .f_view([1, -1, dim])?;
            Tensor::f_cat(&[position.f_narrow(1, 0, 1)?, grid], 1)?
        };
        Ok(x.f_add(&position)?)
    }
    fn attention(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        let (b, heads) = (x.size()[0], self.config.heads);
        let head = self.config.hidden / heads;
        let split = |t: Tensor| -> Result<Tensor, Error> {
            Ok(t.f_view([b, -1, heads, head])?.f_transpose(1, 2)?)
        };
        let key = split(self.linear(&format!("{name}.attention.key"), x)?)?;
        let value = split(self.linear(&format!("{name}.attention.value"), x)?)?;
        let query = split(self.linear(&format!("{name}.attention.query"), x)?)?;
        let scaling = (head as f64).powf(-0.5);
        let out = match self.config.attention {
            DinoAttention::Sdpa => Tensor::f_scaled_dot_product_attention(
                &query,
                &key,
                &value,
                None::<Tensor>,
                0.,
                false,
                scaling,
                false,
            )?,
            DinoAttention::Eager => query
                .f_matmul(&key.f_transpose(-1, -2)?)?
                .f_mul_scalar(scaling)?
                .f_softmax(-1, Kind::Float)?
                .f_to_kind(query.kind())?
                .f_matmul(&value)?,
        };
        let out = out.f_transpose(1, 2)?.f_contiguous()?;
        let out = out.f_reshape([b, -1, self.config.hidden])?;
        self.linear(&format!("{name}.output.dense"), &out)
    }
    fn layer(&self, i: usize, x: &Tensor) -> Result<Tensor, Error> {
        let name = format!("backbone.encoder.layer.{i}");
        let h = self.attention(
            &format!("{name}.attention"),
            &self.norm(&format!("{name}.norm1"), x)?,
        )?;
        let x = h
            .f_mul(self.w(&format!("{name}.layer_scale1.lambda1"))?)?
            .f_add(x)?;
        let h = self.linear(
            &format!("{name}.mlp.fc1"),
            &self.norm(&format!("{name}.norm2"), &x)?,
        )?;
        let h = self.linear(&format!("{name}.mlp.fc2"), &h.f_gelu("none")?)?;
        Ok(h.f_mul(self.w(&format!("{name}.layer_scale2.lambda1"))?)?
            .f_add(&x)?)
    }
    /// The relative depth `[batch, height, width]` for normalized pixels.
    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor, Error> {
        let [b, _, h, w] = pixels.size()[..] else {
            return Err(Error::Native("Expected NCHW pixels".into()));
        };
        let (ph, pw) = (h / self.config.patch, w / self.config.patch);
        let mut x = self.embeddings(pixels)?;
        let mut stages = Vec::new();
        for i in 1..=self.config.layers {
            x = self.layer(i - 1, &x)?;
            if self.config.out_indices.contains(&i) {
                stages.push(self.norm("backbone.layernorm", &x)?);
            }
        }
        let mut features = Vec::new();
        for (i, stage) in stages.iter().enumerate() {
            let name = format!("neck.reassemble_stage.layers.{i}");
            let s = stage.f_narrow(1, 1, stage.size()[1] - 1)?;
            let c = s.size()[2];
            let s = s
                .f_reshape([b, ph, pw, c])?
                .f_permute([0, 3, 1, 2])?
                .f_contiguous()?;
            let s = self.conv(&format!("{name}.projection"), &s, 1, 0)?;
            let s = match self.config.reassemble[i] {
                Reassemble::Up(f) => s.f_conv_transpose2d(
                    self.w(&format!("{name}.resize.weight"))?,
                    Some(self.w(&format!("{name}.resize.bias"))?),
                    [f, f],
                    [0, 0],
                    [0, 0],
                    1,
                    [1, 1],
                )?,
                Reassemble::Same => s,
                Reassemble::Down => self.conv(&format!("{name}.resize"), &s, 2, 1)?,
            };
            features.push(self.conv(&format!("neck.convs.{i}"), &s, 1, 1)?);
        }
        features.reverse();
        let mut fused: Option<Tensor> = None;
        for (i, feature) in features.iter().enumerate() {
            let size = features.get(i + 1).map(|f| [f.size()[2], f.size()[3]]);
            fused = Some(match fused {
                None => self.fusion(i, feature, None, size)?,
                Some(f) => self.fusion(i, &f, Some(feature), size)?,
            });
        }
        let x = fused.unwrap();
        let x = self.conv("head.conv1", &x, 1, 1)?;
        let x = x.f_upsample_bilinear2d(
            [ph * self.config.patch, pw * self.config.patch],
            true,
            None,
            None,
        )?;
        let x = self.conv("head.conv2", &x, 1, 1)?.f_relu()?;
        // max_depth is 1 for relative depth.
        let x = self
            .conv("head.conv3", &x, 1, 0)?
            .f_relu()?
            .f_mul_scalar(1)?;
        Ok(x.f_squeeze_dim(1)?)
    }
    fn residual(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        let h = self.conv(&format!("{name}.convolution1"), &x.f_relu()?, 1, 1)?;
        let h = self.conv(&format!("{name}.convolution2"), &h.f_relu()?, 1, 1)?;
        Ok(h.f_add(x)?)
    }
    /// DepthAnythingFeatureFusionLayer.
    fn fusion(
        &self,
        i: usize,
        x: &Tensor,
        residual: Option<&Tensor>,
        size: Option<[i64; 2]>,
    ) -> Result<Tensor, Error> {
        let name = format!("neck.fusion_stage.layers.{i}");
        let mut x = x.shallow_clone();
        if let Some(residual) = residual {
            let residual = if residual.size() != x.size() {
                residual.f_upsample_bilinear2d([x.size()[2], x.size()[3]], false, None, None)?
            } else {
                residual.shallow_clone()
            };
            x = x.f_add(&self.residual(&format!("{name}.residual_layer1"), &residual)?)?;
        }
        let x = self.residual(&format!("{name}.residual_layer2"), &x)?;
        let x = match size {
            Some(size) => x.f_upsample_bilinear2d(size, true, None, None)?,
            None => {
                let s = x.size();
                x.f_upsample_bilinear2d([s[2] * 2, s[3] * 2], true, Some(2.), Some(2.))?
            }
        };
        self.conv(&format!("{name}.projection"), &x, 1, 0)
    }
}
/// DPTImageProcessor's output size for Depth Anything: as little scaling as
/// possible towards 518 × 518, each side a multiple of 14.
pub fn depth_input_size(height: i64, width: i64) -> (i64, i64) {
    let (mut sh, mut sw) = (518. / height as f64, 518. / width as f64);
    if (1. - sw).abs() < (1. - sh).abs() {
        sh = sw;
    } else {
        sw = sh;
    }
    // Python's round() rounds half to even.
    let constrain = |v: f64| ((v / 14.).round_ties_even() * 14.) as i64;
    (constrain(sh * height as f64), constrain(sw * width as f64))
}
/// DPTImageProcessor on an RGB byte image `[height, width, 3]`: the Pillow bicubic
/// resize, the float64 rescale cast to float32, the ImageNet normalization and
/// channels first, as `[1, 3, h, w]`.
pub fn depth_pixels(pillow: &Pillow, rgb: &[u8], height: i64, width: i64) -> Result<Tensor, Error> {
    let (h, w) = depth_input_size(height, width);
    let resized = pillow.resize_rgb_filter(
        rgb,
        width as i32,
        height as i32,
        3,
        w as i32,
        h as i32,
        Resampling::Bicubic,
    )?;
    let scaled = resized
        .iter()
        .map(|&v| (v as f64 * 0.00392156862745098) as f32)
        .collect::<Vec<_>>();
    let mean = Tensor::from_slice(&[0.485f32, 0.456, 0.406]);
    let std = Tensor::from_slice(&[0.229f32, 0.224, 0.225]);
    Ok(Tensor::from_slice(&scaled)
        .f_view([h, w, 3])?
        .f_sub(&mean)?
        .f_div(&std)?
        .f_permute([2, 0, 1])?
        .f_contiguous()?
        .f_unsqueeze(0)?)
}
/// The prediction resized to the image bicubically and scaled so its maximum is
/// 255, as the transformers example does: `(resized, bytes [height, width])`.
pub fn depth_map(depth: &Tensor, height: i64, width: i64) -> Result<(Tensor, Tensor), Error> {
    let resized = depth
        .f_unsqueeze(0)?
        .f_unsqueeze(1)?
        .f_upsample_bicubic2d([height, width], false, None, None)?
        .f_squeeze()?;
    let scaled = resized
        .f_mul_scalar(255)?
        .f_div(&resized.f_max()?)?
        .f_clamp(0, 255)?;
    Ok((resized, scaled.f_to_kind(Kind::Uint8)?))
}
/// Weights from a safetensors state dictionary on `device`.
pub fn depth_weights(
    tensors: Vec<(String, Tensor)>,
    device: Device,
) -> Result<BTreeMap<String, Tensor>, Error> {
    tensors
        .into_iter()
        .map(|(name, t)| Ok((name, t.f_to_device(device)?)))
        .collect()
}
