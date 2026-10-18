//! Original Qwen3-VL vision transformer shared with Qwen35, including DeepStack.
use super::{Linear, TextEncoderExecution};
use crate::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value, native::Error,
};
use tch::{Device, Kind, Tensor};

pub struct VisionEncoderConfig {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub depth: usize,
    pub heads: i64,
    pub patch_size: i64,
    pub temporal_patch_size: i64,
    pub channels: i64,
    pub merge_size: i64,
    pub position_embeddings: i64,
    pub out_hidden_size: i64,
    pub deepstack_indexes: Vec<usize>,
}
impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            depth: 27,
            heads: 16,
            patch_size: 16,
            temporal_patch_size: 2,
            channels: 3,
            merge_size: 2,
            position_embeddings: 2304,
            out_hidden_size: 4096,
            deepstack_indexes: vec![8, 16, 24],
        }
    }
}
struct Norm {
    weight: Tensor,
    bias: Tensor,
}
impl Norm {
    fn from_parameters(state: &PreparedParameters, name: &str) -> Result<Self, Error> {
        Ok(Self {
            weight: get(state, &format!("{name}.weight"))?.as_tensor()?,
            bias: get(state, &format!("{name}.bias"))?.as_tensor()?,
        })
    }
    fn forward<E: TextEncoderExecution>(&self, x: &Tensor, exec: &mut E) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let weight = self
            .weight
            .f_to_device_(x.device(), x.kind(), false, false)?;
        let bias = self.bias.f_to_device_(x.device(), x.kind(), false, false)?;
        Ok(x.f_layer_norm(
            weight.size().as_slice(),
            Some(&weight),
            Some(&bias),
            1e-6,
            true,
        )?)
    }
}
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}
struct Merger {
    norm: Norm,
    fc1: Linear,
    fc2: Linear,
    postshuffle: bool,
}
impl Merger {
    fn from_parameters(
        state: &PreparedParameters,
        name: &str,
        postshuffle: bool,
    ) -> Result<Self, Error> {
        Ok(Self {
            norm: Norm::from_parameters(state, &format!("{name}.norm"))?,
            fc1: Linear::from_parameters(state, &format!("{name}.linear_fc1"), true)?,
            fc2: Linear::from_parameters(state, &format!("{name}.linear_fc2"), true)?,
            postshuffle,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        merge_dim: i64,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let x = if self.postshuffle {
            self.norm.forward(&x.f_view([-1, merge_dim])?, exec)?
        } else {
            self.norm.forward(x, exec)?.f_view([-1, merge_dim])?
        };
        exec.check_interrupt()?;
        // The two merger classes use exact GELU, unlike the vision block MLP.
        let x = unsafe { self.fc1.forward(rt, &x)? }.f_gelu("none")?;
        exec.check_interrupt()?;
        unsafe { self.fc2.forward(rt, &x) }
    }
}
fn get(state: &PreparedParameters, name: &str) -> Result<Value, Error> {
    state.get(name).cloned().ok_or_else(|| Error::BindingGap {
        backend: "qwen3vl_vision".into(),
        operation: format!("Missing prepared {name}"),
    })
}
pub struct QwenVisionEncoder {
    pub config: VisionEncoderConfig,
    patch_weight: Tensor,
    patch_bias: Tensor,
    pos_weight: Value,
    inv_freq: Tensor,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
}
impl QwenVisionEncoder {
    pub fn from_parameters(
        config: VisionEncoderConfig,
        state: &PreparedParameters,
    ) -> Result<Self, Error> {
        let mut blocks = vec![];
        for i in 0..config.depth {
            let p = format!("visual.blocks.{i}");
            let l = |name: &str| Linear::from_parameters(state, &format!("{p}.{name}"), true);
            blocks.push(Block {
                norm1: Norm::from_parameters(state, &format!("{p}.norm1"))?,
                norm2: Norm::from_parameters(state, &format!("{p}.norm2"))?,
                qkv: l("attn.qkv")?,
                proj: l("attn.proj")?,
                fc1: l("mlp.linear_fc1")?,
                fc2: l("mlp.linear_fc2")?,
            });
        }
        let deepstack = (0..config.deepstack_indexes.len())
            .map(|i| {
                Merger::from_parameters(state, &format!("visual.deepstack_merger_list.{i}"), true)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            patch_weight: get(state, "visual.patch_embed.proj.weight")?.as_tensor()?,
            patch_bias: get(state, "visual.patch_embed.proj.bias")?.as_tensor()?,
            pos_weight: get(state, "visual.pos_embed.weight")?,
            inv_freq: get(state, "visual.rotary_pos_emb.inv_freq")?.as_tensor()?,
            blocks,
            merger: Merger::from_parameters(state, "visual.merger", false)?,
            deepstack,
            config,
        })
    }
    /// Preserve the constructor buffer's device/dtype; do not recompute it on
    /// the execution device with a different exponent/rounding policy.
    pub fn rotary_positions(&self, grid: &[[i64; 3]]) -> Result<Tensor, Error> {
        let c = &self.config;
        let dev = self.inv_freq.device();
        let max_hw =
            grid.iter()
                .map(|g| g[1].max(g[2]))
                .max()
                .ok_or_else(|| Error::ReferenceException {
                    class: "ValueError",
                    message: "max() iterable argument is empty".into(),
                })?;
        let table =
            Tensor::f_arange(max_hw, (self.inv_freq.kind(), dev))?.f_outer(&self.inv_freq)?;
        let mut ids = vec![];
        for &[t, h, w] in grid {
            for _ in 0..t {
                for row in 0..h / c.merge_size {
                    for col in 0..w / c.merge_size {
                        for rr in 0..c.merge_size {
                            for cc in 0..c.merge_size {
                                ids.extend([row * c.merge_size + rr, col * c.merge_size + cc]);
                            }
                        }
                    }
                }
            }
        }
        let ids = Tensor::f_from_slice(&ids)?
            .f_to_device(dev)?
            .f_view([-1, 2])?;
        Ok(table.f_index(&[Some(&ids)])?.f_flatten(1, -1)?)
    }
    /// Source bilinear interpolation of learned positions with its original
    /// host list round-trip before casting weights to the embedding dtype.
    /// # Safety
    /// Prepared position embedding and indices satisfy the original lookup ABI.
    pub unsafe fn learned_positions<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        grid: &[[i64; 3]],
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let c = &self.config;
        let (dev, kind) = match &self.pos_weight {
            Value::Quantized(_) => {
                let q = self.pos_weight.as_quantized()?;
                (q.payload.device(), q.params.original_kind)
            }
            _ => {
                let t = self.pos_weight.as_tensor()?;
                (t.device(), t.kind())
            }
        };
        let side = (c.position_embeddings as f64).sqrt() as i64;
        let mut indices: [Vec<i64>; 4] = std::array::from_fn(|_| vec![]);
        let mut weights: [Vec<f32>; 4] = std::array::from_fn(|_| vec![]);
        for &[_, h, w] in grid {
            let hi = Tensor::f_linspace(0., (side - 1) as f64, h, (Kind::Float, dev))?;
            let wi = Tensor::f_linspace(0., (side - 1) as f64, w, (Kind::Float, dev))?;
            let hf = hi.f_to_kind(Kind::Int)?;
            let wf = wi.f_to_kind(Kind::Int)?;
            let hc = hf.f_add_scalar(1)?.f_clamp_max(side - 1)?;
            let wc = wf.f_add_scalar(1)?.f_clamp_max(side - 1)?;
            let dh = hi.f_sub(&hf)?;
            let dw = wi.f_sub(&wf)?;
            let base = hf.f_mul_scalar(side)?;
            let ceil = hc.f_mul_scalar(side)?;
            let ii = [
                base.f_unsqueeze(1)?.f_add(&wf.f_unsqueeze(0)?)?,
                base.f_unsqueeze(1)?.f_add(&wc.f_unsqueeze(0)?)?,
                ceil.f_unsqueeze(1)?.f_add(&wf.f_unsqueeze(0)?)?,
                ceil.f_unsqueeze(1)?.f_add(&wc.f_unsqueeze(0)?)?,
            ];
            let inv_h = dh.f_rsub_scalar(1.)?;
            let inv_w = dw.f_rsub_scalar(1.)?;
            let ww = [
                inv_h.f_unsqueeze(1)?.f_mul(&inv_w.f_unsqueeze(0)?)?,
                inv_h.f_unsqueeze(1)?.f_mul(&dw.f_unsqueeze(0)?)?,
                dh.f_unsqueeze(1)?.f_mul(&inv_w.f_unsqueeze(0)?)?,
                dh.f_unsqueeze(1)?.f_mul(&dw.f_unsqueeze(0)?)?,
            ];
            for j in 0..4 {
                let a = ii[j]
                    .f_flatten(0, -1)?
                    .f_to_device(Device::Cpu)?
                    .f_to_kind(Kind::Int64)?;
                let mut v = vec![0_i64; a.numel()];
                let n = v.len();
                a.f_copy_data(&mut v, n)?;
                indices[j].extend(v);
                let a = ww[j].f_flatten(0, -1)?.f_to_device(Device::Cpu)?;
                let mut v = vec![0_f32; a.numel()];
                let n = v.len();
                a.f_copy_data(&mut v, n)?;
                weights[j].extend(v);
            }
        }
        let n = indices[0].len() as i64;
        let ids = Tensor::f_from_slice(&indices.into_iter().flatten().collect::<Vec<_>>())?
            .f_view([4, n])?
            .f_to_device(dev)?;
        let weights = Tensor::f_from_slice(&weights.into_iter().flatten().collect::<Vec<_>>())?
            .f_view([4, n])?
            .f_to_device_(dev, kind, false, false)?;
        exec.check_interrupt()?;
        let pos = unsafe { super::layers::embedding_rows(rt, &self.pos_weight, &ids, None)? }
            .f_mul(&weights.f_unsqueeze(2)?)?;
        let pos = pos
            .f_get(0)?
            .f_add(&pos.f_get(1)?)?
            .f_add(&pos.f_get(2)?)?
            .f_add(&pos.f_get(3)?)?;
        let parts =
            pos.f_split_with_sizes(grid.iter().map(|g| g[1] * g[2]).collect::<Vec<_>>(), 0)?;
        let mut out = vec![];
        for (p, &[t, h, w]) in parts.iter().zip(grid) {
            out.push(
                p.f_repeat([t, 1])?
                    .f_view([
                        t,
                        h / c.merge_size,
                        c.merge_size,
                        w / c.merge_size,
                        c.merge_size,
                        -1,
                    ])?
                    .f_permute([0, 1, 3, 2, 4, 5])?
                    .f_flatten(0, 4)?,
            );
        }
        Ok(Tensor::f_cat(&out, 0)?)
    }
    /// # Safety
    /// Resident prepared weights, original attention and Kitchen RoPE contracts hold.
    pub unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        patches: &Tensor,
        grid: &[[i64; 3]],
        exec: &mut E,
    ) -> Result<(Tensor, Vec<Tensor>), Error> {
        let c = &self.config;
        exec.check_interrupt()?;
        let x = patches.f_view([
            -1,
            c.channels,
            c.temporal_patch_size,
            c.patch_size,
            c.patch_size,
        ])?;
        let weight = self
            .patch_weight
            .f_to_device_(x.device(), x.kind(), false, false)?;
        let bias = self
            .patch_bias
            .f_to_device_(x.device(), x.kind(), false, false)?;
        let mut x = x
            .f_conv3d(
                &weight,
                Some(&bias),
                [c.temporal_patch_size, c.patch_size, c.patch_size],
                [0, 0, 0],
                [1, 1, 1],
                1,
            )?
            .f_view([-1, c.hidden_size])?;
        exec.observe("vision_patch", &x)?;
        let pos = unsafe { self.learned_positions(rt, grid, exec)? }.f_to_device(x.device())?;
        exec.observe("vision_learned_positions", &pos)?;
        x = x.f_add(&pos)?;
        let rotary = self
            .rotary_positions(grid)?
            .f_to_device(x.device())?
            .f_reshape([x.size()[0], -1])?;
        let emb = Tensor::f_cat(&[rotary.shallow_clone(), rotary], -1)?;
        let cos = emb.f_cos()?.f_unsqueeze(-2)?;
        let sin = emb.f_sin()?.f_unsqueeze(-2)?;
        let half = sin.size()[sin.dim() - 1] / 2;
        let matrix = Tensor::f_stack(
            &[
                cos.f_slice(-1, 0, half, 1)?,
                sin.f_slice(-1, half, i64::MAX, 1)?.f_neg()?,
                sin.f_slice(-1, 0, half, 1)?,
                cos.f_slice(-1, half, i64::MAX, 1)?,
            ],
            -1,
        )?
        .f_reshape([1, x.size()[0], 1, half, 2, 2])?;
        exec.observe("vision_rope", &matrix)?;
        let lengths = grid
            .iter()
            .flat_map(|g| std::iter::repeat_n(g[1] * g[2], g[0] as usize))
            .collect::<Vec<_>>();
        let mut ds = vec![];
        for (i, b) in self.blocks.iter().enumerate() {
            let n = b.norm1.forward(&x, exec)?;
            exec.check_interrupt()?;
            let qkv = unsafe { b.qkv.forward(rt, &n)? }
                .f_reshape([x.size()[0], 3, c.heads, -1])?
                .f_permute([1, 0, 2, 3])?
                .f_unbind(0)?;
            let qk = unsafe {
                rt.call_kitchen(
                    "apply_rope_split_half",
                    &vec![
                        ("xq".into(), Value::tensor(qkv[0].f_unsqueeze(0)?)),
                        ("xk".into(), Value::tensor(qkv[1].f_unsqueeze(0)?)),
                        ("freqs_cis".into(), Value::tensor(matrix.shallow_clone())),
                    ],
                )?
            };
            let Value::Tuple(qk) = qk else {
                return Err(Error::Native("Invalid vision RoPE output".into()));
            };
            let qs = qk[0]
                .as_tensor()?
                .f_squeeze_dim(0)?
                .f_split_with_sizes(&lengths, 0)?;
            let ks = qk[1]
                .as_tensor()?
                .f_squeeze_dim(0)?
                .f_split_with_sizes(&lengths, 0)?;
            let vs = qkv[2].f_split_with_sizes(&lengths, 0)?;
            let mut outputs = vec![];
            for ((q, k), v) in qs.iter().zip(&ks).zip(&vs) {
                outputs.push(exec.attention(
                    rt,
                    &q.f_transpose(0, 1)?.f_unsqueeze(0)?,
                    &k.f_transpose(0, 1)?.f_unsqueeze(0)?,
                    &v.f_transpose(0, 1)?.f_unsqueeze(0)?,
                    None,
                    c.heads,
                    false,
                )?);
            }
            let attn = Tensor::f_cat(&outputs, 1)?.f_reshape([x.size()[0], -1])?;
            exec.check_interrupt()?;
            x = x.f_add(&unsafe { b.proj.forward(rt, &attn)? })?;
            let n = b.norm2.forward(&x, exec)?;
            exec.check_interrupt()?;
            let n = unsafe { b.fc1.forward(rt, &n)? }.f_gelu("tanh")?;
            exec.check_interrupt()?;
            x = x.f_add(&unsafe { b.fc2.forward(rt, &n)? })?;
            exec.observe(&format!("vision_block_{i}"), &x)?;
            if let Some(j) = c.deepstack_indexes.iter().position(|n| *n == i) {
                ds.push(unsafe {
                    self.deepstack[j].forward(
                        rt,
                        &x,
                        c.hidden_size * c.merge_size * c.merge_size,
                        exec,
                    )?
                });
            }
        }
        let merged = unsafe {
            self.merger
                .forward(rt, &x, c.hidden_size * c.merge_size * c.merge_size, exec)?
        };
        Ok((merged, ds))
    }
}
