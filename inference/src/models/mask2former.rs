//! Mask2Former as transformers' `Mask2FormerForUniversalSegmentation` with its Swin
//! backbone (`SwinBackbone`), multi-scale deformable pixel decoder and masked-attention
//! transformer decoder; its `Mask2FormerImageProcessor` input; and the panoptic
//! regions Imagine colors into a region map.
use crate::{
    models::qwen_image21::{Pillow, Resampling},
    native::Error,
};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

const BACKBONE: &str = "model.pixel_level_module.encoder";
const PIXEL_DECODER: &str = "model.pixel_level_module.decoder";
const TRANSFORMER: &str = "model.transformer_module";
/// Feature levels the pixel decoder's encoder and the transformer decoder attend.
const LEVELS: i64 = 3;
/// Sampling points per head and level in the deformable attention.
const POINTS: i64 = 4;
const EPS: f64 = 1e-5;

#[derive(Clone, Debug)]
pub struct Mask2FormerConfig {
    /// Swin: the patch embedding width, blocks and heads per stage, window and patch.
    pub embed: i64,
    pub depths: Vec<usize>,
    pub swin_heads: Vec<i64>,
    pub window: i64,
    pub patch: i64,
    /// The pixel decoder's feature size, the transformer's hidden size and the mask
    /// feature size, equal in every released checkpoint.
    pub feature: i64,
    pub encoder_layers: usize,
    /// Heads of the deformable, self- and cross-attention.
    pub heads: i64,
    /// `decoder_layers`: the transformer decoder runs one layer fewer.
    pub decoder_layers: usize,
}
impl Mask2FormerConfig {
    /// `facebook/mask2former-swin-large-coco-panoptic`.
    pub fn swin_large() -> Self {
        Self {
            embed: 192,
            depths: vec![2, 2, 18, 2],
            swin_heads: vec![6, 12, 24, 48],
            window: 12,
            patch: 4,
            feature: 256,
            encoder_layers: 6,
            heads: 8,
            decoder_layers: 10,
        }
    }
}
/// The final queries: class logits `[batch, queries, labels + 1]` (the last is "no
/// object") and mask logits `[batch, queries, height / 4, width / 4]`.
pub struct Prediction {
    pub class_logits: Tensor,
    pub mask_logits: Tensor,
}
pub struct Mask2Former {
    pub config: Mask2FormerConfig,
    weights: BTreeMap<String, Tensor>,
}
fn missing(name: &str) -> Error {
    Error::BindingGap {
        backend: "mask2former".into(),
        operation: format!("Missing weight {name}"),
    }
}
/// SwinSelfAttention's relative position index for a square window.
fn relative_position_index(window: i64) -> Tensor {
    let side = 2 * window - 1;
    let mut index = Vec::with_capacity((window * window * window * window) as usize);
    for i in 0..window * window {
        for j in 0..window * window {
            let dy = i / window - j / window + window - 1;
            let dx = i % window - j % window + window - 1;
            index.push(dy * side + dx);
        }
    }
    Tensor::from_slice(&index)
}
/// window_partition: `[batch, h, w, c]` into `[windows, ws, ws, c]`.
fn partition(x: &Tensor, ws: i64) -> Result<Tensor, Error> {
    let [b, h, w, c] = x.size()[..] else {
        unreachable!()
    };
    Ok(x.f_view([b, h / ws, ws, w / ws, ws, c])?
        .f_permute([0, 1, 3, 2, 4, 5])?
        .f_contiguous()?
        .f_view([-1, ws, ws, c])?)
}
/// window_reverse.
fn reverse(windows: &Tensor, ws: i64, h: i64, w: i64) -> Result<Tensor, Error> {
    let c = windows.size()[3];
    Ok(windows
        .f_view([-1, h / ws, w / ws, ws, ws, c])?
        .f_permute([0, 1, 3, 2, 4, 5])?
        .f_contiguous()?
        .f_view([-1, h, w, c])?)
}
/// SwinLayer.get_attn_mask for a shifted window over a padded `[h, w]` grid.
fn shifted_window_mask(
    h: i64,
    w: i64,
    ws: i64,
    shift: i64,
    kind: Kind,
    device: Device,
) -> Result<Tensor, Error> {
    let bands = |n: i64| [(0, n - ws), (n - ws, n - shift), (n - shift, n)];
    let mut regions = vec![0f32; (h * w) as usize];
    let mut count = 0.;
    for (y0, y1) in bands(h) {
        for (x0, x1) in bands(w) {
            for y in y0.max(0)..y1 {
                for x in x0.max(0)..x1 {
                    regions[(y * w + x) as usize] = count;
                }
            }
            count += 1.;
        }
    }
    let windows = partition(
        &Tensor::from_slice(&regions)
            .f_to_kind(kind)?
            .f_to_device(device)?
            .f_view([1, h, w, 1])?,
        ws,
    )?
    .f_view([-1, ws * ws])?;
    let mask = windows.f_unsqueeze(1)?.f_sub(&windows.f_unsqueeze(2)?)?;
    Ok(mask
        .f_masked_fill(&mask.f_ne(0)?, -100.)?
        .f_masked_fill(&mask.f_eq(0)?, 0.)?)
}
impl Mask2Former {
    /// Floating weights require gradients as `nn.Parameter`s do: matmul folds a
    /// strided batch into one `mm` when the other operand requires them, which
    /// changes the summation order of the queries' linears for batches above one.
    pub fn new(config: Mask2FormerConfig, weights: BTreeMap<String, Tensor>) -> Self {
        let weights = weights
            .into_iter()
            .map(|(name, t)| {
                let t = if t.is_floating_point() {
                    t.set_requires_grad(true)
                } else {
                    t
                };
                (name, t)
            })
            .collect();
        Self { config, weights }
    }
    fn w(&self, name: &str) -> Result<&Tensor, Error> {
        self.weights.get(name).ok_or_else(|| missing(name))
    }
    fn linear(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.f_linear(
            self.w(&format!("{name}.weight"))?,
            self.weights.get(&format!("{name}.bias")),
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
    fn layer_norm(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        let weight = self.w(&format!("{name}.weight"))?;
        Ok(x.f_layer_norm(
            weight.size(),
            Some(weight),
            Some(self.w(&format!("{name}.bias"))?),
            EPS,
            true,
        )?)
    }
    fn group_norm(&self, name: &str, x: &Tensor) -> Result<Tensor, Error> {
        Ok(x.f_group_norm(
            32,
            Some(self.w(&format!("{name}.weight"))?),
            Some(self.w(&format!("{name}.bias"))?),
            EPS,
            true,
        )?)
    }
    fn swin_attention(
        &self,
        name: &str,
        x: &Tensor,
        heads: i64,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let [windows, n, c] = x.size()[..] else {
            unreachable!()
        };
        let head = c / heads;
        let split = |t: Tensor| -> Result<Tensor, Error> {
            Ok(t.f_view([windows, n, -1, head])?.f_transpose(1, 2)?)
        };
        let query = split(self.linear(&format!("{name}.self.query"), x)?)?;
        let key = split(self.linear(&format!("{name}.self.key"), x)?)?;
        let value = split(self.linear(&format!("{name}.self.value"), x)?)?;
        let scores = query
            .f_matmul(&key.f_transpose(-1, -2)?)?
            .f_div_scalar((head as f64).sqrt())?;
        let ws = self.config.window;
        let bias = self
            .w(&format!("{name}.self.relative_position_bias_table"))?
            .f_index_select(
                0,
                &relative_position_index(ws).f_to_device(scores.device())?,
            )?
            .f_view([ws * ws, ws * ws, -1])?
            .f_permute([2, 0, 1])?
            .f_contiguous()?;
        let mut scores = scores.f_add(&bias.f_unsqueeze(0)?)?;
        if let Some(mask) = mask {
            let count = mask.size()[0];
            scores = scores
                .f_view([windows / count, count, heads, n, n])?
                .f_add(&mask.f_unsqueeze(1)?.f_unsqueeze(0)?)?
                .f_view([-1, heads, n, n])?;
        }
        let probs = scores.f_softmax(-1, scores.kind())?;
        let context = probs
            .f_matmul(&value)?
            .f_permute([0, 2, 1, 3])?
            .f_contiguous()?
            .f_view([windows, n, c])?;
        self.linear(&format!("{name}.output.dense"), &context)
    }
    /// SwinLayer with the backbone's `always_partition`: the configured window and
    /// shift at every resolution, the grid padded to whole windows.
    fn swin_layer(
        &self,
        name: &str,
        x: &Tensor,
        h: i64,
        w: i64,
        heads: i64,
        shift: i64,
    ) -> Result<Tensor, Error> {
        let ws = self.config.window;
        let [b, _, c] = x.size()[..] else {
            unreachable!()
        };
        let s = self
            .layer_norm(&format!("{name}.layernorm_before"), x)?
            .f_view([b, h, w, c])?;
        let (right, bottom) = ((ws - w % ws) % ws, (ws - h % ws) % ws);
        let s = s.f_constant_pad_nd([0, 0, 0, right, 0, bottom])?;
        let (hp, wp) = (h + bottom, w + right);
        let s = if shift > 0 {
            s.f_roll([-shift, -shift], [1, 2])?
        } else {
            s
        };
        let windows = partition(&s, ws)?.f_view([-1, ws * ws, c])?;
        let mask = if shift > 0 {
            Some(shifted_window_mask(
                hp,
                wp,
                ws,
                shift,
                x.kind(),
                x.device(),
            )?)
        } else {
            None
        };
        let a =
            self.swin_attention(&format!("{name}.attention"), &windows, heads, mask.as_ref())?;
        let s = reverse(&a.f_view([-1, ws, ws, c])?, ws, hp, wp)?;
        let s = if shift > 0 {
            s.f_roll([shift, shift], [1, 2])?
        } else {
            s
        };
        let s = if right > 0 || bottom > 0 {
            s.f_narrow(1, 0, h)?.f_narrow(2, 0, w)?.f_contiguous()?
        } else {
            s
        };
        let x = x.f_add(&s.f_view([b, h * w, c])?)?;
        let m = self.linear(
            &format!("{name}.intermediate.dense"),
            &self.layer_norm(&format!("{name}.layernorm_after"), &x)?,
        )?;
        Ok(x.f_add(&self.linear(&format!("{name}.output.dense"), &m.f_gelu("none")?)?)?)
    }
    /// SwinPatchMerging.
    fn merge(&self, name: &str, x: &Tensor, h: i64, w: i64) -> Result<Tensor, Error> {
        let [b, _, c] = x.size()[..] else {
            unreachable!()
        };
        let mut s = x.f_view([b, h, w, c])?;
        if h % 2 == 1 || w % 2 == 1 {
            s = s.f_constant_pad_nd([0, 0, 0, w % 2, 0, h % 2])?;
        }
        let part = |y: i64, x: i64| -> Result<Tensor, Error> {
            Ok(s.f_slice(1, y, None, 2)?.f_slice(2, x, None, 2)?)
        };
        let s = Tensor::f_cat(&[part(0, 0)?, part(1, 0)?, part(0, 1)?, part(1, 1)?], -1)?
            .f_view([b, -1, 4 * c])?;
        let s = self.layer_norm(&format!("{name}.norm"), &s)?;
        self.linear(&format!("{name}.reduction"), &s)
    }
    /// SwinBackbone's normalized feature maps of every stage, channels first.
    fn backbone(&self, pixels: &Tensor) -> Result<Vec<Tensor>, Error> {
        let [b, _, h, w] = pixels.size()[..] else {
            unreachable!()
        };
        let p = self.config.patch;
        let pixels = pixels.f_constant_pad_nd([0, (p - w % p) % p, 0, (p - h % p) % p])?;
        let x = self.conv(
            &format!("{BACKBONE}.embeddings.patch_embeddings.projection"),
            &pixels,
            p,
            0,
        )?;
        let (mut h, mut w) = (x.size()[2], x.size()[3]);
        let mut x = self.layer_norm(
            &format!("{BACKBONE}.embeddings.norm"),
            &x.f_flatten(2, -1)?.f_transpose(1, 2)?,
        )?;
        let mut features = Vec::new();
        let stages = self.config.depths.len();
        for (i, &depth) in self.config.depths.iter().enumerate() {
            let name = format!("{BACKBONE}.encoder.layers.{i}");
            for j in 0..depth {
                let shift = if j % 2 == 0 {
                    0
                } else {
                    self.config.window / 2
                };
                x = self.swin_layer(
                    &format!("{name}.blocks.{j}"),
                    &x,
                    h,
                    w,
                    self.config.swin_heads[i],
                    shift,
                )?;
            }
            let c = x.size()[2];
            features.push(
                self.layer_norm(
                    &format!("{BACKBONE}.hidden_states_norms.stage{}", i + 1),
                    &x,
                )?
                .f_view([b, h, w, c])?
                .f_permute([0, 3, 1, 2])?
                .f_contiguous()?,
            );
            if i + 1 < stages {
                x = self.merge(&format!("{name}.downsample"), &x, h, w)?;
                (h, w) = ((h + 1) / 2, (w + 1) / 2);
            }
        }
        Ok(features)
    }
    /// Mask2FormerSinePositionEmbedding (normalized, no mask): `[b, feature, h, w]`.
    fn sine(&self, b: i64, h: i64, w: i64, kind: Kind, device: Device) -> Result<Tensor, Error> {
        let features = self.config.feature / 2;
        let scale = 2. * std::f64::consts::PI;
        let not_mask = Tensor::f_ones([b, h, w], (kind, device))?;
        let y = not_mask.f_cumsum(1, None::<Kind>)?;
        let x = not_mask.f_cumsum(2, None::<Kind>)?;
        let y = y
            .f_div(&y.f_narrow(1, h - 1, 1)?.f_add_scalar(1e-6)?)?
            .f_mul_scalar(scale)?;
        let x = x
            .f_div(&x.f_narrow(2, w - 1, 1)?.f_add_scalar(1e-6)?)?
            .f_mul_scalar(scale)?;
        let dim = Tensor::f_arange(features, (Kind::Int64, device))?.f_to_kind(kind)?;
        let dim = Tensor::f_pow_scalar(
            10000,
            &dim.f_div_scalar_mode(2, "floor")?
                .f_mul_scalar(2)?
                .f_div_scalar(features)?,
        )?;
        let embed = |t: Tensor| -> Result<Tensor, Error> {
            let t = t.f_unsqueeze(-1)?.f_div(&dim)?;
            Ok(Tensor::f_stack(
                &[
                    t.f_slice(3, 0, None, 2)?.f_sin()?,
                    t.f_slice(3, 1, None, 2)?.f_cos()?,
                ],
                4,
            )?
            .f_flatten(3, -1)?)
        };
        Ok(Tensor::f_cat(&[embed(y)?, embed(x)?], 3)?.f_permute([0, 3, 1, 2])?)
    }
    /// Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention.
    fn deformable(
        &self,
        name: &str,
        x: &Tensor,
        position: &Tensor,
        reference: &Tensor,
        shapes: &[(i64, i64)],
    ) -> Result<Tensor, Error> {
        let heads = self.config.heads;
        let [b, q, d] = x.size()[..] else {
            unreachable!()
        };
        let head = d / heads;
        let query = x.f_add(position)?;
        let value = self
            .linear(&format!("{name}.value_proj"), x)?
            .f_view([b, q, heads, head])?;
        let offsets = self
            .linear(&format!("{name}.sampling_offsets"), &query)?
            .f_view([b, q, heads, LEVELS, POINTS, 2])?;
        let weights = self
            .linear(&format!("{name}.attention_weights"), &query)?
            .f_view([b, q, heads, LEVELS * POINTS])?;
        let weights = weights
            .f_softmax(-1, weights.kind())?
            .f_view([b, q, heads, LEVELS, POINTS])?;
        let normalizer = shapes.iter().flat_map(|&(h, w)| [w, h]).collect::<Vec<_>>();
        let normalizer = Tensor::from_slice(&normalizer)
            .f_to_device(x.device())?
            .f_view([1, 1, 1, LEVELS, 1, 2])?;
        let locations = reference
            .f_unsqueeze(2)?
            .f_unsqueeze(4)?
            .f_add(&offsets.f_div(&normalizer)?)?;
        // multi_scale_deformable_attention through grid_sample.
        let grids = locations.f_mul_scalar(2)?.f_sub_scalar(1)?;
        let mut sampled = Vec::new();
        let mut start = 0;
        for (level, &(h, w)) in shapes.iter().enumerate() {
            let v = value
                .f_narrow(1, start, h * w)?
                .f_flatten(2, -1)?
                .f_transpose(1, 2)?
                .f_reshape([b * heads, head, h, w])?;
            start += h * w;
            let grid = grids
                .f_select(3, level as i64)?
                .f_transpose(1, 2)?
                .f_flatten(0, 1)?;
            sampled.push(v.f_grid_sampler(&grid, 0, 0, false)?);
        }
        let weights = weights
            .f_transpose(1, 2)?
            .f_reshape([b * heads, 1, q, LEVELS * POINTS])?;
        let out = Tensor::f_stack(&sampled, -2)?
            .f_flatten(-2, -1)?
            .f_mul(&weights)?
            .f_sum_dim_intlist([-1].as_slice(), false, None::<Kind>)?
            .f_view([b, heads * head, q])?
            .f_transpose(1, 2)?
            .f_contiguous()?;
        self.linear(&format!("{name}.output_proj"), &out)
    }
    /// Mask2FormerPixelDecoderEncoderOnly.get_reference_points with unpadded levels.
    fn reference_points(
        &self,
        b: i64,
        shapes: &[(i64, i64)],
        device: Device,
    ) -> Result<Tensor, Error> {
        // get_valid_ratio of an empty padding mask: the valid count over the side.
        let ratio = |n: i64| -> Result<Tensor, Error> {
            Ok(Tensor::f_full([b], n, (Kind::Float, device))?.f_div_scalar(n)?)
        };
        let valid = shapes
            .iter()
            .map(|&(h, w)| Ok(Tensor::f_stack(&[ratio(w)?, ratio(h)?], -1)?))
            .collect::<Result<Vec<_>, Error>>()?;
        let valid = Tensor::f_stack(&valid, 1)?;
        let mut points = Vec::new();
        for (level, &(h, w)) in shapes.iter().enumerate() {
            let ys = Tensor::f_linspace(0.5, h as f64 - 0.5, h, (Kind::Float, device))?;
            let xs = Tensor::f_linspace(0.5, w as f64 - 0.5, w, (Kind::Float, device))?;
            let grid = Tensor::f_meshgrid_indexing(&[ys, xs], "ij")?;
            let ratio = valid.f_select(1, level as i64)?;
            let y = grid[0]
                .f_reshape([-1])?
                .f_unsqueeze(0)?
                .f_div(&ratio.f_select(1, 1)?.f_unsqueeze(1)?.f_mul_scalar(h)?)?;
            let x = grid[1]
                .f_reshape([-1])?
                .f_unsqueeze(0)?
                .f_div(&ratio.f_select(1, 0)?.f_unsqueeze(1)?.f_mul_scalar(w)?)?;
            points.push(Tensor::f_stack(&[x, y], -1)?);
        }
        Ok(Tensor::f_cat(&points, 1)?
            .f_unsqueeze(2)?
            .f_mul(&valid.f_unsqueeze(1)?)?)
    }
    /// Mask2FormerPixelDecoder: `(mask features, the three coarsest multi-scale features)`.
    fn pixel_decoder(&self, features: &[Tensor]) -> Result<(Tensor, Vec<Tensor>), Error> {
        let b = features[0].size()[0];
        let (kind, device) = (features[0].kind(), features[0].device());
        let mut embeds = Vec::new();
        let mut positions = Vec::new();
        let mut shapes = Vec::new();
        let level_embed = self.w(&format!("{PIXEL_DECODER}.level_embed"))?;
        for (level, x) in features.iter().rev().take(LEVELS as usize).enumerate() {
            let name = format!("{PIXEL_DECODER}.input_projections.{level}");
            let e = self.group_norm(
                &format!("{name}.1"),
                &self.conv(&format!("{name}.0"), x, 1, 0)?,
            )?;
            let (h, w) = (x.size()[2], x.size()[3]);
            shapes.push((h, w));
            embeds.push(e.f_flatten(2, -1)?.f_transpose(1, 2)?);
            positions.push(
                self.sine(b, h, w, kind, device)?
                    .f_flatten(2, -1)?
                    .f_transpose(1, 2)?
                    .f_add(&level_embed.f_select(0, level as i64)?.f_view([1, 1, -1])?)?,
            );
        }
        let mut x = Tensor::f_cat(&embeds, 1)?;
        let position = Tensor::f_cat(&positions, 1)?;
        let reference = self.reference_points(b, &shapes, device)?;
        for i in 0..self.config.encoder_layers {
            let name = format!("{PIXEL_DECODER}.encoder.layers.{i}");
            let h = self.deformable(
                &format!("{name}.self_attn"),
                &x,
                &position,
                &reference,
                &shapes,
            )?;
            x = self.layer_norm(&format!("{name}.self_attn_layer_norm"), &x.f_add(&h)?)?;
            let h = self.linear(&format!("{name}.fc1"), &x)?.f_relu()?;
            let h = self.linear(&format!("{name}.fc2"), &h)?;
            x = self.layer_norm(&format!("{name}.final_layer_norm"), &x.f_add(&h)?)?;
        }
        let mut outputs = Vec::new();
        let mut start = 0;
        for &(h, w) in &shapes {
            outputs.push(
                x.f_narrow(1, start, h * w)?
                    .f_transpose(1, 2)?
                    .f_view([b, -1, h, w])?,
            );
            start += h * w;
        }
        // The FPN levels below the encoder's: log2(8) - log2(4) for the released strides.
        let fpn = features.len() - LEVELS as usize;
        for (idx, feature) in features[..fpn].iter().rev().enumerate() {
            let n = fpn - idx;
            let lateral = self.group_norm(
                &format!("{PIXEL_DECODER}.adapter_{n}.1"),
                &self.conv(&format!("{PIXEL_DECODER}.adapter_{n}.0"), feature, 1, 0)?,
            )?;
            let size = [lateral.size()[2], lateral.size()[3]];
            let out = lateral.f_add(
                &outputs
                    .last()
                    .unwrap()
                    .f_upsample_bilinear2d(size, false, None, None)?,
            )?;
            let out = self
                .group_norm(
                    &format!("{PIXEL_DECODER}.layer_{n}.1"),
                    &self.conv(&format!("{PIXEL_DECODER}.layer_{n}.0"), &out, 1, 1)?,
                )?
                .f_relu()?;
            outputs.push(out);
        }
        let mask = self.conv(
            &format!("{PIXEL_DECODER}.mask_projection"),
            outputs.last().unwrap(),
            1,
            0,
        )?;
        outputs.truncate(LEVELS as usize);
        Ok((mask, outputs))
    }
    /// Mask2FormerMaskPredictor: `(mask logits, the cross-attention mask)`.
    fn predict_masks(
        &self,
        x: &Tensor,
        pixels: &Tensor,
        size: [i64; 2],
    ) -> Result<(Tensor, Tensor), Error> {
        let name = format!("{TRANSFORMER}.decoder.mask_predictor.mask_embedder");
        let mut e = x.f_transpose(0, 1)?;
        for i in 0..3 {
            e = self.linear(&format!("{name}.{i}.0"), &e)?;
            if i < 2 {
                e = e.f_relu()?;
            }
        }
        let masks = Tensor::f_einsum("bqc, bchw -> bqhw", &[&e, pixels], None::<&[i64]>)?;
        let attention = masks
            .f_upsample_bilinear2d(size, false, None, None)?
            .f_sigmoid()?
            .f_flatten(2, -1)?
            .f_unsqueeze(1)?
            .f_repeat([1, self.config.heads, 1, 1])?
            .f_flatten(0, 1)?
            .f_lt(0.5)?;
        Ok((masks, attention))
    }
    /// nn.MultiheadAttention (sequence first) with a boolean mask, as transformers
    /// calls it: separate projections and explicit weights.
    fn cross_attention(
        &self,
        name: &str,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, Error> {
        let [t, b, c] = query.size()[..] else {
            unreachable!()
        };
        let s = key.size()[0];
        let heads = self.config.heads;
        let head = c / heads;
        let w = self.w(&format!("{name}.in_proj_weight"))?.f_chunk(3, 0)?;
        let bias = self.w(&format!("{name}.in_proj_bias"))?.f_chunk(3, 0)?;
        let q = query.f_linear(&w[0], Some(&bias[0]))?;
        let k = key.f_linear(&w[1], Some(&bias[1]))?;
        let v = value.f_linear(&w[2], Some(&bias[2]))?;
        let mask = Tensor::f_zeros(mask.size(), (query.kind(), query.device()))?
            .f_masked_fill(mask, f64::NEG_INFINITY)?;
        let q = q.f_view([t, b * heads, head])?.f_transpose(0, 1)?;
        let k = k.f_view([s, b * heads, head])?.f_transpose(0, 1)?;
        let v = v.f_view([s, b * heads, head])?.f_transpose(0, 1)?;
        let q = q.f_mul_scalar((1. / head as f64).sqrt())?;
        let weights = mask.f_baddbmm(&q, &k.f_transpose(-2, -1)?, 1, 1)?;
        let weights = weights.f_softmax(-1, weights.kind())?;
        let out = weights
            .f_bmm(&v)?
            .f_transpose(0, 1)?
            .f_reshape([t * b, c])?;
        Ok(self
            .linear(&format!("{name}.out_proj"), &out)?
            .f_view([t, b, c])?)
    }
    /// Mask2FormerAttention over the queries (sequence first).
    fn self_attention(&self, name: &str, x: &Tensor, position: &Tensor) -> Result<Tensor, Error> {
        let x = x.f_permute([1, 0, 2])?;
        let position = position.f_permute([1, 0, 2])?;
        let [b, t, c] = x.size()[..] else {
            unreachable!()
        };
        let heads = self.config.heads;
        let head = c / heads;
        let shape = |t: Tensor| -> Result<Tensor, Error> {
            Ok(t.f_view([b, -1, heads, head])?
                .f_transpose(1, 2)?
                .f_contiguous()?
                .f_view([b * heads, -1, head])?)
        };
        let positioned = x.f_add(&position)?;
        let q = shape(
            self.linear(&format!("{name}.q_proj"), &positioned)?
                .f_mul_scalar((head as f64).powf(-0.5))?,
        )?;
        let k = shape(self.linear(&format!("{name}.k_proj"), &positioned)?)?;
        let v = shape(self.linear(&format!("{name}.v_proj"), &x)?)?;
        let weights = q.f_bmm(&k.f_transpose(1, 2)?)?;
        let weights = weights.f_softmax(-1, weights.kind())?;
        let out = weights
            .f_bmm(&v)?
            .f_view([b, heads, t, head])?
            .f_transpose(1, 2)?
            .f_reshape([b, t, c])?;
        Ok(self
            .linear(&format!("{name}.out_proj"), &out)?
            .f_permute([1, 0, 2])?)
    }
    /// Mask2FormerMaskedAttentionDecoderLayer, post-norm.
    #[allow(clippy::too_many_arguments)]
    fn decoder_layer(
        &self,
        i: usize,
        x: &Tensor,
        level: usize,
        positions: &[Tensor],
        query_position: &Tensor,
        features: &[Tensor],
        mask: &Tensor,
    ) -> Result<Tensor, Error> {
        let name = format!("{TRANSFORMER}.decoder.layers.{i}");
        let h = self.cross_attention(
            &format!("{name}.cross_attn"),
            &x.f_add(query_position)?,
            &features[level].f_add(&positions[level])?,
            &features[level],
            mask,
        )?;
        let x = self.layer_norm(&format!("{name}.cross_attn_layer_norm"), &x.f_add(&h)?)?;
        let h = self.self_attention(&format!("{name}.self_attn"), &x, query_position)?;
        let x = self.layer_norm(&format!("{name}.self_attn_layer_norm"), &x.f_add(&h)?)?;
        let h = self.linear(&format!("{name}.fc1"), &x)?.f_relu()?;
        let h = self.linear(&format!("{name}.fc2"), &h)?;
        self.layer_norm(&format!("{name}.final_layer_norm"), &x.f_add(&h)?)
    }
    /// The final class and mask logits for normalized pixels `[batch, 3, h, w]`.
    pub fn forward(&self, pixels: &Tensor) -> Result<Prediction, Error> {
        let _grad = tch::no_grad_guard();
        if pixels.dim() != 4 {
            return Err(Error::Native("Expected NCHW pixels".into()));
        }
        let features = self.backbone(pixels)?;
        let (mask_features, scales) = self.pixel_decoder(&features)?;
        let b = pixels.size()[0];
        let (kind, device) = (pixels.kind(), pixels.device());
        let level_embed = self.w(&format!("{TRANSFORMER}.level_embed.weight"))?;
        let mut sizes = Vec::new();
        let mut positions = Vec::new();
        let mut levels = Vec::new();
        for (i, x) in scales.iter().enumerate() {
            let (h, w) = (x.size()[2], x.size()[3]);
            sizes.push([h, w]);
            positions.push(
                self.sine(b, h, w, kind, device)?
                    .f_flatten(2, -1)?
                    .f_permute([2, 0, 1])?,
            );
            levels.push(
                x.f_flatten(2, -1)?
                    .f_add(&level_embed.f_select(0, i as i64)?.f_view([1, -1, 1])?)?
                    .f_permute([2, 0, 1])?,
            );
        }
        let repeat = |name: &str| -> Result<Tensor, Error> {
            Ok(self
                .w(&format!("{TRANSFORMER}.{name}.weight"))?
                .f_unsqueeze(1)?
                .f_repeat([1, b, 1])?)
        };
        let query_position = repeat("queries_embedder")?;
        let mut x = repeat("queries_features")?;
        let norm = format!("{TRANSFORMER}.decoder.layernorm");
        let mut intermediate = self.layer_norm(&norm, &x)?;
        let (mut masks, mut attention) =
            self.predict_masks(&intermediate, &mask_features, sizes[0])?;
        for i in 0..self.config.decoder_layers - 1 {
            let n = attention.size()[2];
            let any = attention
                .f_sum_dim_intlist([-1].as_slice(), false, None::<Kind>)?
                .f_ne(n)?;
            attention = attention.f_mul(&any.f_unsqueeze(-1)?)?;
            x = self.decoder_layer(
                i,
                &x,
                i % LEVELS as usize,
                &positions,
                &query_position,
                &levels,
                &attention,
            )?;
            intermediate = self.layer_norm(&norm, &x)?;
            (masks, attention) = self.predict_masks(
                &intermediate,
                &mask_features,
                sizes[(i + 1) % LEVELS as usize],
            )?;
        }
        Ok(Prediction {
            class_logits: self.linear("class_predictor", &intermediate.f_transpose(0, 1)?)?,
            mask_logits: masks,
        })
    }
}
/// Mask2FormerImageProcessor's resize with Detectron2's COCO test setting: the
/// shortest edge 800 unless the longest would exceed 1333, each side then rounded
/// up to a multiple of 32.
pub fn segmentation_input_size(height: i64, width: i64) -> (i64, i64) {
    let (size, max_size, divisor) = (800, 1333, 32);
    let (short, long) = if width <= height {
        (width, height)
    } else {
        (height, width)
    };
    let (mut new_short, mut new_long) = (size, ((size * long) as f64 / short as f64) as i64);
    if new_long > max_size {
        (new_short, new_long) = (
            ((max_size * new_short) as f64 / new_long as f64) as i64,
            max_size,
        );
    }
    let (h, w) = if width <= height {
        (new_long, new_short)
    } else {
        (new_short, new_long)
    };
    let round = |v: i64| (v as f64 / divisor as f64).ceil() as i64 * divisor;
    (round(h), round(w))
}
/// Mask2FormerImageProcessor on an RGB byte image `[height, width, 3]`: the Pillow
/// bilinear resize, the float64 rescale cast to float32, the ImageNet normalization
/// and channels first, as `[1, 3, h, w]`.
pub fn segmentation_pixels(
    pillow: &Pillow,
    rgb: &[u8],
    height: i64,
    width: i64,
) -> Result<Tensor, Error> {
    let (h, w) = segmentation_input_size(height, width);
    let resized = pillow.resize_rgb_filter(
        rgb,
        width as i32,
        height as i32,
        3,
        w as i32,
        h as i32,
        Resampling::Bilinear,
    )?;
    let scaled = resized
        .iter()
        .map(|&v| (v as f64 * 0.00392156862745098) as f32)
        .collect::<Vec<_>>();
    // The checkpoint's processor stores the green deviation one float32 below 0.224.
    let mean = Tensor::from_slice(&[0.485f32, 0.456, 0.406]);
    let std = Tensor::from_slice(&[0.229f32, 0.223_999_99, 0.225]);
    Ok(Tensor::from_slice(&scaled)
        .f_view([h, w, 3])?
        .f_sub(&mean)?
        .f_div(&std)?
        .f_permute([2, 0, 1])?
        .f_contiguous()?
        .f_unsqueeze(0)?)
}
/// COCO panoptic labels below this are things; the rest are stuff, fused per label.
pub const COCO_PANOPTIC_THINGS: i64 = 80;
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub id: i32,
    pub label: i64,
    pub fused: bool,
    pub score: f32,
}
/// Panoptic segments: ids `[height, width]` (0 where no segment won) and one entry
/// per accepted query, as transformers' `compute_segments` lists them.
pub struct Panoptic {
    pub segmentation: Tensor,
    pub segments: Vec<Segment>,
}
/// Panoptic segments of one image at `[height, width]`, with transformers'
/// `remove_low_and_no_objects` and `compute_segments`; the kept mask logits are
/// first resized to the target bilinearly (Detectron2 resizes logits, not
/// probabilities) and `fuse` names the labels merged into one segment.
pub fn panoptic(
    class_logits: &Tensor,
    mask_logits: &Tensor,
    height: i64,
    width: i64,
    threshold: f64,
    overlap: f64,
    fuse: impl Fn(i64) -> bool,
) -> Result<Panoptic, Error> {
    let labels_count = class_logits.size()[1] - 1;
    let (scores, labels) = class_logits
        .f_softmax(-1, class_logits.kind())?
        .f_max_dim(-1, false)?;
    let keep = labels
        .f_ne(labels_count)?
        .f_logical_and(&scores.f_gt(threshold)?)?;
    let masks = mask_logits.f_index(&[Some(&keep)])?;
    let scores = scores.f_index(&[Some(&keep)])?;
    let labels = labels.f_index(&[Some(&keep)])?;
    let device = mask_logits.device();
    let mut segmentation = Tensor::f_zeros([height, width], (Kind::Int, device))?;
    let mut segments = Vec::new();
    let n = masks.size()[0];
    if n == 0 {
        return Ok(Panoptic {
            segmentation,
            segments,
        });
    }
    let probs = masks
        .f_unsqueeze(0)?
        .f_upsample_bilinear2d([height, width], false, None, None)?
        .f_squeeze_dim(0)?
        .f_sigmoid()?
        .f_mul(&scores.f_view([-1, 1, 1])?)?;
    let winners = probs.f_argmax(0, false)?;
    let labels = Vec::<i64>::try_from(labels.f_to_device(Device::Cpu)?)?;
    let scores = Vec::<f32>::try_from(scores.f_to_device(Device::Cpu)?)?;
    let mut stuff = BTreeMap::new();
    let mut current = 0;
    for k in 0..n {
        let label = labels[k as usize];
        let fused = fuse(label);
        let mask = winners.f_eq(k)?;
        let area = mask.f_sum(Kind::Int64)?.int64_value(&[]);
        let original = probs
            .f_select(0, k)?
            .f_ge(0.5)?
            .f_sum(Kind::Int64)?
            .int64_value(&[]);
        let exists = area > 0 && original > 0 && f64::from(area as f32 / original as f32) > overlap;
        if !exists {
            continue;
        }
        let id = match stuff.get(&label) {
            Some(&id) => id,
            None => {
                current += 1;
                current
            }
        };
        segmentation = segmentation.f_masked_fill(&mask, id as i64)?;
        segments.push(Segment {
            id,
            label,
            fused,
            score: scores[k as usize],
        });
        if fused {
            stuff.insert(label, id);
        }
    }
    Ok(Panoptic {
        segmentation,
        segments,
    })
}
/// A distinct flat color per segment id, black where no segment won: the color map
/// the MOSS control adapter reads (colors only identify regions).
pub fn region_colors(segmentation: &Tensor, palette: &[[u8; 3]]) -> Result<Tensor, Error> {
    let top = segmentation.f_max()?.int64_value(&[]).max(0);
    let mut table = vec![0u8; 3 * (top as usize + 1)];
    for id in 1..=top as usize {
        table[3 * id..3 * id + 3].copy_from_slice(&palette[(id - 1) % palette.len()]);
    }
    Ok(Tensor::from_slice(&table)
        .f_view([-1, 3])?
        .f_to_device(segmentation.device())?
        .f_index_select(
            0,
            &segmentation
                .f_clamp_min(0)?
                .f_flatten(0, -1)?
                .f_to_kind(Kind::Int64)?,
        )?
        .f_view([segmentation.size()[0], segmentation.size()[1], 3])?)
}
/// Weights from a safetensors state dictionary on `device`.
pub fn segmentation_weights(
    tensors: Vec<(String, Tensor)>,
    device: Device,
) -> Result<BTreeMap<String, Tensor>, Error> {
    tensors
        .into_iter()
        .map(|(name, t)| Ok((name, t.f_to_device(device)?)))
        .collect()
}
