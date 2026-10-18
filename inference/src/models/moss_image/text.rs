//! Qwen3.5-9B language prefill for MOSS conditioning, without autoregressive decode.
use super::moss_gated_delta_rule;
use crate::models::qwen_image21::{
    Linear, PromptToken, TextConditioning, TextConditioningOptions, TextEncoderConfig,
    TextEncoderExecution, TokenizedPrompt, WeightedToken, embedding_rows, text_rope,
};
use crate::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value, native::Error,
};
use tch::{Kind, Tensor};

pub struct Qwen35Config {
    pub hidden: i64,
    pub layers: usize,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub rotary_dim: i64,
    pub key_heads: i64,
    pub value_heads: i64,
    pub key_dim: i64,
    pub value_dim: i64,
    pub conv_kernel: i64,
    pub eps: f64,
    pub theta: f64,
    pub mrope: [i64; 3],
}
impl Default for Qwen35Config {
    fn default() -> Self {
        Self {
            hidden: 4096,
            layers: 32,
            heads: 16,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            key_heads: 16,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
            conv_kernel: 4,
            eps: 1e-6,
            theta: 10000000.,
            mrope: [11, 11, 10],
        }
    }
}
fn parameter(p: &PreparedParameters, name: &str) -> Result<Value, Error> {
    p.get(name).cloned().ok_or_else(|| Error::BindingGap {
        backend: "qwen35".into(),
        operation: format!("Missing prepared {name}"),
    })
}
fn rms(x: &Tensor, w: &Tensor, add: bool, eps: f64) -> Result<Tensor, Error> {
    // llama.RMSNorm.scale runs in the parameter dtype before cast_to(x).
    let w = if add {
        w.f_add_scalar(1.)?
    } else {
        w.shallow_clone()
    };
    let w = w.f_to_device_(x.device(), x.kind(), false, false)?;
    Ok(x.f_rms_norm([*x.size().last().unwrap()], Some(&w), Some(eps))?)
}
unsafe fn project<E: TextEncoderExecution>(
    linear: &Linear,
    rt: &Runtime,
    x: &Tensor,
    e: &mut E,
) -> Result<Tensor, Error> {
    e.check_interrupt()?;
    unsafe { linear.forward(rt, x) }
}
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Mlp {
    fn load(p: &PreparedParameters, path: &str) -> Result<Self, Error> {
        let l = |n| Linear::from_parameters(p, &format!("{path}.{n}"), false);
        Ok(Self {
            gate: l("gate_proj")?,
            up: l("up_proj")?,
            down: l("down_proj")?,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let gate = unsafe { project(&self.gate, rt, x, e)? }.f_silu()?;
        let up = unsafe { project(&self.up, rt, x, e)? };
        unsafe { project(&self.down, rt, &gate.f_mul(&up)?, e) }
    }
}
struct Delta {
    qkv: Linear,
    z: Linear,
    b: Linear,
    a: Linear,
    out: Linear,
    conv: Tensor,
    decay: Tensor,
    bias: Tensor,
    norm: Tensor,
}
impl Delta {
    fn load(p: &PreparedParameters, path: &str) -> Result<Self, Error> {
        let l = |n| Linear::from_parameters(p, &format!("{path}.{n}"), false);
        let w = |n| parameter(p, &format!("{path}.{n}"))?.as_tensor();
        Ok(Self {
            qkv: l("in_proj_qkv")?,
            z: l("in_proj_z")?,
            b: l("in_proj_b")?,
            a: l("in_proj_a")?,
            out: l("out_proj")?,
            conv: w("conv1d.weight")?,
            decay: w("A_log")?,
            bias: w("dt_bias")?,
            norm: w("norm.weight")?,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        c: &Qwen35Config,
        x: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let shape = x.size();
        let (batch, seq) = (shape[0], shape[1]);
        let proj = unsafe { project(&self.qkv, rt, x, e)? };
        let z = unsafe { project(&self.z, rt, x, e)? };
        let mixed = proj.f_transpose(1, 2)?;
        let b = unsafe { project(&self.b, rt, x, e)? };
        let a = unsafe { project(&self.a, rt, x, e)? };
        e.check_interrupt()?;
        let conv = self.conv.f_to_device_(x.device(), x.kind(), false, false)?;
        let mixed = mixed
            .f_conv1d(
                &conv,
                None::<&Tensor>,
                [1],
                [c.conv_kernel - 1],
                [1],
                mixed.size()[1],
            )?
            .f_slice(2, 0, seq, 1)?
            .f_silu()?
            .f_transpose(1, 2)?;
        let kd = c.key_heads * c.key_dim;
        let vd = c.value_heads * c.value_dim;
        let parts = mixed.f_split_with_sizes([kd, kd, vd], -1)?;
        let decay = self
            .decay
            .f_to_device_(x.device(), Kind::Float, false, false)?
            .f_exp()?
            .f_neg()?;
        let bias = self
            .bias
            .f_to_device_(x.device(), Kind::Float, false, false)?;
        let beta = b.f_sigmoid()?;
        let g = decay.f_mul(&a.f_to_kind(Kind::Float)?.f_add(&bias)?.f_softplus()?)?;
        let mut q = parts[0].f_reshape([batch, seq, c.key_heads, c.key_dim])?;
        let mut k = parts[1].f_reshape([batch, seq, c.key_heads, c.key_dim])?;
        let v = parts[2].f_reshape([batch, seq, c.value_heads, c.value_dim])?;
        if c.value_heads != c.key_heads {
            q = q.f_repeat_interleave_self_int(c.value_heads / c.key_heads, Some(2), None)?;
            k = k.f_repeat_interleave_self_int(c.value_heads / c.key_heads, Some(2), None)?;
        }
        let (out, _) = moss_gated_delta_rule(&rt.native, &q, &k, &v, &g, &beta, 64, None, false)?;
        let out = rms(&out.f_reshape([-1, c.value_dim])?, &self.norm, false, c.eps)?.f_mul(
            &z.f_reshape([-1, c.value_dim])?
                .f_to_kind(out.kind())?
                .f_silu()?,
        )?;
        unsafe { project(&self.out, rt, &out.f_reshape([batch, seq, -1])?, e) }
    }
}
struct FullAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    qnorm: Tensor,
    knorm: Tensor,
}
impl FullAttention {
    fn load(p: &PreparedParameters, path: &str) -> Result<Self, Error> {
        let l = |n| Linear::from_parameters(p, &format!("{path}.{n}"), false);
        let w = |n| parameter(p, &format!("{path}.{n}.weight"))?.as_tensor();
        Ok(Self {
            q: l("q_proj")?,
            k: l("k_proj")?,
            v: l("v_proj")?,
            out: l("o_proj")?,
            qnorm: w("q_norm")?,
            knorm: w("k_norm")?,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        c: &Qwen35Config,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let shape = x.size();
        let (b, n) = (shape[0], shape[1]);
        let qg = unsafe { project(&self.q, rt, x, e)? }.f_view([b, n, c.heads, c.head_dim * 2])?;
        let q = qg.f_slice(-1, 0, c.head_dim, 1)?;
        let gate = qg
            .f_slice(-1, c.head_dim, i64::MAX, 1)?
            .f_reshape([b, n, -1])?;
        let k = unsafe { project(&self.k, rt, x, e)? };
        let v = unsafe { project(&self.v, rt, x, e)? };
        let q = rms(&q, &self.qnorm, true, c.eps)?.f_transpose(1, 2)?;
        let k = rms(
            &k.f_view([b, n, c.kv_heads, c.head_dim])?,
            &self.knorm,
            true,
            c.eps,
        )?
        .f_transpose(1, 2)?;
        let v = v
            .f_view([b, n, c.kv_heads, c.head_dim])?
            .f_transpose(1, 2)?;
        let result = unsafe {
            rt.call_kitchen(
                "apply_rope_split_half",
                &vec![
                    (
                        "xq".into(),
                        Value::tensor(q.f_slice(-1, 0, c.rotary_dim, 1)?),
                    ),
                    (
                        "xk".into(),
                        Value::tensor(k.f_slice(-1, 0, c.rotary_dim, 1)?),
                    ),
                    ("freqs_cis".into(), Value::tensor(rope.shallow_clone())),
                ],
            )?
        };
        let Value::Tuple(rot) = result else {
            return Err(Error::Native("Invalid Qwen3.5 partial RoPE result".into()));
        };
        let q = Tensor::f_cat(
            &[
                rot[0].as_tensor()?,
                q.f_slice(-1, c.rotary_dim, i64::MAX, 1)?,
            ],
            -1,
        )?;
        let k = Tensor::f_cat(
            &[
                rot[1].as_tensor()?,
                k.f_slice(-1, c.rotary_dim, i64::MAX, 1)?,
            ],
            -1,
        )?;
        let out = e.attention(rt, &q, &k, &v, mask, c.heads, c.heads != c.kv_heads)?;
        unsafe { project(&self.out, rt, &out.f_mul(&gate.f_sigmoid()?)?, e) }
    }
}
enum Attention {
    Linear(Box<Delta>),
    Full(Box<FullAttention>),
}
struct Block {
    attention: Attention,
    mlp: Mlp,
    input_norm: Tensor,
    post_norm: Tensor,
}
pub struct Qwen35TextEncoder {
    pub config: Qwen35Config,
    pub embedding: Value,
    blocks: Vec<Block>,
    norm: Tensor,
}
impl Qwen35TextEncoder {
    /// Original SDClip token/mask processing and ClipTokenWeightEncoder. MOSS
    /// uses the entire final normalized state, with no Qwen-Image system filter.
    /// # Safety
    /// Prepared embedding/linear storage and selected execution services satisfy
    /// the original source contracts; image conditioning has a separate binding.
    pub unsafe fn encode_prompt<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        prompt: &TokenizedPrompt,
        options: TextConditioningOptions,
        e: &mut E,
    ) -> Result<TextConditioning, Error> {
        let sections = prompt.batches.len();
        let weighted = prompt.batches.iter().flatten().any(|t| t.weight != 1.);
        let length = prompt.batches.iter().map(Vec::len).max().unwrap_or(0);
        let mut batches = prompt.batches.clone();
        if weighted || sections == 0 {
            batches.push(
                (0..length)
                    .map(|_| WeightedToken {
                        token: PromptToken::Id(248044),
                        weight: 1.,
                        word_id: 0,
                    })
                    .collect(),
            );
        }
        let mut embeddings = Vec::new();
        let mut masks = Vec::new();
        for batch in &batches {
            let mut ids = Vec::new();
            let mut mask = Vec::new();
            let mut eos = false;
            let mut left = false;
            for (i, token) in batch.iter().enumerate() {
                let PromptToken::Id(id) = token.token else {
                    return Err(Error::BindingGap {
                        backend: "moss_text".into(),
                        operation: "Qwen3.5 vision conditioning is not bound".into(),
                    });
                };
                if i == 0 && id == 248044 {
                    left = true;
                }
                if eos || left && id == 248044 {
                    mask.push(0_i64);
                } else {
                    mask.push(1_i64);
                    left = false;
                }
                ids.push(id);
                if !eos && id == 248044 && !left {
                    *mask.last_mut().unwrap() = 0;
                    eos = true;
                }
            }
            let ids = Tensor::f_from_slice(&ids)?
                .f_unsqueeze(0)?
                .f_to_device(options.execution_device)?;
            embeddings.push(unsafe { self.embed(rt, &ids, e)? });
            masks.push(mask);
        }
        let embeddings = Tensor::f_cat(&embeddings, 0)?;
        let width = masks.first().map_or(0, Vec::len);
        let mask = if width == 0 {
            Tensor::f_empty(
                [masks.len() as i64, 0],
                (Kind::Int64, options.execution_device),
            )?
        } else {
            Tensor::f_from_slice(&masks.iter().flatten().copied().collect::<Vec<_>>())?
                .f_reshape([masks.len() as i64, width as i64])?
                .f_to_device(options.execution_device)?
        };
        let out = unsafe { self.forward(rt, &embeddings, Some(&mask), None, e)? }
            .f_to_kind(Kind::Float)?;
        let mut parts = Vec::new();
        for (i, batch) in prompt.batches.iter().enumerate() {
            let z = out.f_slice(0, i as i64, i as i64 + 1, 1)?;
            if weighted {
                let empty = out.f_get(out.size()[0] - 1)?;
                for (j, token) in batch.iter().enumerate() {
                    if token.weight != 1. {
                        let mut row = z.f_get(0)?.f_get(j as i64)?;
                        let neutral = empty.f_get(j as i64)?;
                        let value = row
                            .f_sub(&neutral)?
                            .f_mul_scalar(token.weight)?
                            .f_add(&neutral)?;
                        row.f_copy_(&value)?;
                    }
                }
            }
            parts.push(z);
        }
        let context = if parts.is_empty() {
            out.f_slice(0, -1, i64::MAX, 1)?
        } else {
            Tensor::f_cat(&parts, -2)?
        }
        .f_to_device(options.intermediate_device)?;
        let mask = mask
            .f_slice(0, 0, sections as i64, 1)?
            .f_flatten(0, -1)?
            .f_unsqueeze(0)?
            .f_to_device(options.intermediate_device)?;
        Ok(TextConditioning {
            context,
            pooled: None,
            attention_mask: Some(mask),
            image_slots: Vec::new(),
        })
    }
    pub fn from_parameters(c: Qwen35Config, p: &PreparedParameters) -> Result<Self, Error> {
        let mut blocks = Vec::new();
        for i in 0..c.layers {
            let path = format!("model.layers.{i}");
            let attention = if (i + 1) % 4 == 0 {
                Attention::Full(Box::new(FullAttention::load(
                    p,
                    &format!("{path}.self_attn"),
                )?))
            } else {
                Attention::Linear(Box::new(Delta::load(p, &format!("{path}.linear_attn"))?))
            };
            blocks.push(Block {
                attention,
                mlp: Mlp::load(p, &format!("{path}.mlp"))?,
                input_norm: parameter(p, &format!("{path}.input_layernorm.weight"))?.as_tensor()?,
                post_norm: parameter(p, &format!("{path}.post_attention_layernorm.weight"))?
                    .as_tensor()?,
            });
        }
        Ok(Self {
            embedding: parameter(p, "model.embed_tokens.weight")?,
            norm: parameter(p, "model.norm.weight")?.as_tensor()?,
            blocks,
            config: c,
        })
    }
    /// # Safety
    /// The original prepared embedding and selected execution kernels are valid.
    pub unsafe fn embed<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        ids: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        unsafe { embedding_rows(rt, &self.embedding, ids, Some(Kind::Float)) }
    }
    /// Language prefill only; caller binds full-attention preferences at each
    /// indexed execution block. Source add(out=x) mutates the embedding buffer.
    /// # Safety
    /// Original dtype/device/stream contracts and static service policy hold.
    pub unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: Option<&Tensor>,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let c = &self.config;
        let mut x = embeds.shallow_clone();
        let seq = x.size()[1];
        let positions = match positions {
            Some(p) => p.shallow_clone(),
            None => Tensor::f_arange(seq, (Kind::Int64, x.device()))?.f_unsqueeze(0)?,
        };
        let rope = text_rope(
            &TextEncoderConfig {
                head_dim: c.rotary_dim,
                theta: c.theta,
                rope_dims: c.mrope,
                interleaved_mrope: true,
                ..TextEncoderConfig::default()
            },
            &positions,
            x.device(),
            &rt.native,
        )?;
        let minimum = match x.kind() {
            Kind::Half => -65504.,
            Kind::BFloat16 => -(f32::from_bits(0x7f7f0000) as f64),
            Kind::Float => -(f32::MAX as f64),
            _ => -f64::MAX,
        };
        let mut mask = attention_mask
            .map(|mask| -> Result<Tensor, Error> {
                let s = mask.size();
                let m = mask
                    .f_to_kind(x.kind())?
                    .f_reshape([s[0], 1, -1, *s.last().unwrap()])?
                    .f_expand([s[0], 1, seq, *s.last().unwrap()], false)?
                    .f_rsub_scalar(1.)?;
                Ok(m.f_masked_fill(&m.f_to_kind(Kind::Bool)?, minimum / 4.)?)
            })
            .transpose()?;
        if seq > 1 {
            let causal = Tensor::f_empty([seq, seq], (x.kind(), x.device()))?
                .f_fill_(minimum / 4.)?
                .f_triu_(1)?;
            mask = Some(match mask {
                Some(mut m) => m.f_add_(&causal)?,
                None => causal,
            });
        }
        e.observe("embeddings", &x)?;
        e.queue_begin(self.blocks.len(), x.device())?;
        for (i, block) in self.blocks.iter().enumerate() {
            let mut core = |e: &mut E| -> Result<(), Error> {
                let input = rms(&x, &block.input_norm, true, c.eps)?;
                let h = match &block.attention {
                    Attention::Linear(a) => unsafe { a.forward(rt, c, &input, e)? },
                    Attention::Full(a) => unsafe {
                        a.forward(rt, c, &input, mask.as_ref(), &rope, e)?
                    },
                };
                x = x.f_add_out(&x, &h)?;
                let post = rms(&x, &block.post_norm, true, c.eps)?;
                let h = unsafe { block.mlp.forward(rt, &post, e)? };
                x = x.f_add_out(&x, &h)?;
                Ok(())
            };
            e.block(i, &mut core)?;
            e.observe(&format!("block-{i}"), &x)?;
        }
        e.queue_end(x.device())?;
        rms(&x, &self.norm, true, c.eps)
    }
}
