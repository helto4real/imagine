//! Qwen3-VL language prefill from the frozen llama.py implementation.
//! Vision preprocessing, autoregressive KV decoding and the model patcher have
//! separate bindings; this is the text-conditioning forward, without Python.
use super::{Linear, PromptToken, TokenizedPrompt, WeightedToken};
use crate::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value, native::Error,
};
use tch::{Device, Kind, Tensor};

pub struct TextEncoderConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub layers: usize,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub eps: f64,
    pub theta: f64,
    pub rope_dims: [i64; 3],
    pub interleaved_mrope: bool,
    pub merged_qkv: bool,
    pub merged_mlp: bool,
    pub qkv_bias: bool,
}
impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            intermediate_size: 12288,
            layers: 36,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            eps: 1e-6,
            theta: 5_000_000.,
            rope_dims: [24, 20, 20],
            interleaved_mrope: true,
            merged_qkv: false,
            merged_mlp: false,
            qkv_bias: false,
        }
    }
}
pub trait TextEncoderExecution {
    /// Optional isolated differential observer; the default does no tensor work.
    fn observe(&mut self, _name: &str, _tensor: &Tensor) -> Result<(), Error> {
        Ok(())
    }
    fn check_interrupt(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn queue_begin(&mut self, layers: usize, device: Device) -> Result<(), Error>;
    fn queue_end(&mut self, device: Device) -> Result<(), Error>;
    fn block(
        &mut self,
        index: usize,
        core: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error>
    where
        Self: Sized;
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserve the original shaped attention callback and its GQA flag"
    )]
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        heads: i64,
        gqa: bool,
    ) -> Result<Tensor, Error>;
}
#[derive(Clone)]
pub enum Intermediate {
    None,
    Layer(i64),
    All,
    Layers(Vec<i64>),
}
pub struct TextForwardOptions<'a> {
    pub attention_mask: Option<&'a Tensor>,
    pub position_ids: Option<&'a Tensor>,
    pub intermediate: Intermediate,
    pub final_layer_norm_intermediate: bool,
    pub deepstack: &'a [Tensor],
    pub visual_mask: Option<&'a Tensor>,
}
impl Default for TextForwardOptions<'_> {
    fn default() -> Self {
        Self {
            attention_mask: None,
            position_ids: None,
            intermediate: Intermediate::Layer(-1),
            final_layer_norm_intermediate: false,
            deepstack: &[],
            visual_mask: None,
        }
    }
}
pub struct TextEncoderOutput {
    pub last: Tensor,
    pub intermediate: Option<Tensor>,
}
pub struct TextConditioningOptions {
    pub execution_device: Device,
    pub intermediate_device: Device,
}
pub struct TextConditioning {
    pub context: Tensor,
    pub pooled: Option<Tensor>,
    pub attention_mask: Option<Tensor>,
    pub image_slots: Vec<i64>,
}
struct LanguageBlock {
    q: Linear,
    k: Option<Linear>,
    v: Option<Linear>,
    out: Linear,
    gate: Linear,
    up: Option<Linear>,
    down: Linear,
    input_norm: Tensor,
    post_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}
pub struct QwenTextEncoder {
    pub config: TextEncoderConfig,
    pub embedding: Value,
    blocks: Vec<LanguageBlock>,
    norm: Option<Tensor>,
}
fn parameter(state: &PreparedParameters, name: &str) -> Result<Value, Error> {
    state.get(name).cloned().ok_or_else(|| Error::BindingGap {
        backend: "qwen3vl".into(),
        operation: format!("Missing prepared {name}"),
    })
}
fn rms(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor, Error> {
    let w = weight.f_to_device_(x.device(), x.kind(), false, false)?;
    Ok(x.f_rms_norm(w.size().as_slice(), Some(&w), Some(eps))?)
}
fn min_value(kind: Kind) -> f64 {
    match kind {
        Kind::Half => -65504.,
        Kind::BFloat16 => -(f32::from_bits(0x7f7f0000) as f64),
        Kind::Float => -(f32::MAX as f64),
        _ => -f64::MAX,
    }
}
/// Original precompute_freqs_cis + rope_matrix for the Qwen3-VL theta/config.
pub fn text_rope(
    config: &TextEncoderConfig,
    positions: &Tensor,
    device: Device,
    native: &crate::native::Kitchen,
) -> Result<Tensor, Error> {
    let numerator = Tensor::f_arange_start_step(0, config.head_dim, 2, (Kind::Int64, device))?
        .f_to_kind(Kind::Float)?;
    let inv = Tensor::f_pow_scalar(config.theta, &numerator.f_div_scalar(config.head_dim)?)?
        .f_reciprocal()?
        .f_mul_scalar(1.)?;
    let inv = inv
        .f_unsqueeze(0)?
        .f_unsqueeze(-1)?
        .f_expand([positions.size()[0], -1, 1], false)?;
    let pos = positions.f_unsqueeze(1)?.f_to_kind(Kind::Float)?;
    let freqs = inv
        .f_to_kind(Kind::Float)?
        .f_matmul(&pos.f_to_kind(Kind::Float)?)?
        .f_transpose(1, 2)?;
    let (cos, sin) = if positions.size()[0] > 1 && config.interleaved_mrope {
        let inter = native.clone_tensor(&freqs.f_get(0)?)?;
        for (axis, offset) in [(1, 1), (2, 2)] {
            inter
                .f_slice(-1, offset, config.rope_dims[axis] * 3, 3)?
                .f_copy_(&freqs.f_get(axis as i64)?.f_slice(
                    -1,
                    offset,
                    config.rope_dims[axis] * 3,
                    3,
                )?)?;
        }
        let emb = Tensor::f_cat(&[inter.shallow_clone(), inter], -1)?;
        (emb.f_cos()?.f_unsqueeze(0)?, emb.f_sin()?.f_unsqueeze(0)?)
    } else {
        let emb = Tensor::f_cat(&[freqs.shallow_clone(), freqs], -1)?;
        let cos = emb.f_cos()?;
        let sin = emb.f_sin()?;
        if positions.size()[0] > 1 {
            let sizes = config
                .rope_dims
                .iter()
                .chain(config.rope_dims.iter())
                .copied()
                .collect::<Vec<_>>();
            let select = |t: Tensor| -> Result<Tensor, Error> {
                let chunks = t.f_split_with_sizes(&sizes, -1)?;
                let selected = chunks
                    .iter()
                    .enumerate()
                    .map(|(i, t)| t.f_get((i % 3) as i64))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Tensor::f_cat(&selected, -1)?.f_unsqueeze(0)?)
            };
            (select(cos)?, select(sin)?)
        } else {
            (cos.f_unsqueeze(1)?, sin.f_unsqueeze(1)?)
        }
    };
    let half = sin.size()[sin.dim() - 1] / 2;
    let matrix = Tensor::f_stack(
        &[
            cos.f_slice(-1, 0, half, 1)?,
            sin.f_slice(-1, half, i64::MAX, 1)?.f_neg()?,
            sin.f_slice(-1, 0, half, 1)?,
            cos.f_slice(-1, half, i64::MAX, 1)?,
        ],
        -1,
    )?;
    let mut shape = matrix.size();
    shape.pop();
    shape.extend([2, 2]);
    let matrix = matrix.f_reshape(shape.as_slice())?;
    Ok(if matrix.dim() == 5 {
        matrix.f_unsqueeze(0)?
    } else {
        matrix
    })
}
impl QwenTextEncoder {
    /// Source SDClipModel.process_tokens / ClipTokenWeightEncoder, followed by
    /// QwenImage21TEModel's system-turn removal. Image preprocessing has its
    /// own acceptance gate and is never silently skipped by this text path.
    /// # Safety
    /// All active original model, embedding and attention kernel contracts hold.
    pub unsafe fn encode_prompt<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        prompt: &TokenizedPrompt,
        options: TextConditioningOptions,
        exec: &mut E,
    ) -> Result<TextConditioning, Error> {
        unsafe { self.encode_prompt_with_images(rt, prompt, options, None, exec) }
    }
    /// Original SDClip image expansion followed by Qwen3-VL multimodal prefill.
    /// # Safety
    /// Active language/vision weights, image tensors and original kernels satisfy
    /// their selected device/layout contracts.
    pub unsafe fn encode_prompt_with_images<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        prompt: &TokenizedPrompt,
        options: TextConditioningOptions,
        vision: Option<(&super::QwenVisionEncoder, &[Tensor])>,
        exec: &mut E,
    ) -> Result<TextConditioning, Error> {
        let sections = prompt.batches.len();
        let has_weights = prompt
            .batches
            .iter()
            .any(|batch| batch.iter().any(|t| t.weight != 1.));
        let max_length = prompt.batches.iter().map(Vec::len).max().unwrap_or(0);
        let mut batches = prompt.batches.clone();
        if has_weights || sections == 0 {
            batches.push(
                (0..max_length)
                    .map(|_| WeightedToken {
                        token: PromptToken::Id(151643),
                        weight: 1.,
                        word_id: 0,
                    })
                    .collect(),
            );
        }
        let mut embeddings = vec![];
        let mut masks = vec![];
        let mut image_info = vec![];
        for batch in &batches {
            let mut other = vec![];
            let mut ids = vec![];
            let mut mask = vec![];
            let mut eos = false;
            let mut left_pad = false;
            for (i, token) in batch.iter().enumerate() {
                let PromptToken::Id(id) = token.token else {
                    if let PromptToken::Image(image) = token.token {
                        other.push((i as i64, image));
                    }
                    continue;
                };
                if i == 0 && id == 151643 {
                    left_pad = true;
                }
                if eos || (left_pad && id == 151643) {
                    mask.push(0_i64);
                } else {
                    mask.push(1_i64);
                    left_pad = false;
                }
                ids.push(id);
                if !eos && id == 151643 && !left_pad {
                    *mask.last_mut().unwrap() = 0;
                    eos = true;
                }
            }
            let ids = Tensor::f_from_slice(&ids)?
                .f_unsqueeze(0)?
                .f_to_device(options.execution_device)?;
            let mut embedded = unsafe { self.embed(rt, &ids, exec)? };
            let mut offset = 0;
            let mut pad_extra = 0;
            image_info.clear(); // original process_tokens returns the last section's info
            for (index, image) in other {
                let (vision, images) = vision.ok_or_else(|| Error::BindingGap {
                    backend: "qwen3vl".into(),
                    operation: "Vision owner required for image tokens".into(),
                })?;
                let image = images.get(image).ok_or_else(|| Error::ReferenceException {
                    class: "IndexError",
                    message: "list index out of range".into(),
                })?;
                let (patches, grid) = super::VisionPreprocess::default().patches(image)?;
                let patches =
                    patches.f_to_device_(options.execution_device, Kind::Float, false, false)?;
                let rows = grid
                    .f_to_device(Device::Cpu)?
                    .f_to_kind(Kind::Int64)?
                    .f_view([-1])?;
                let mut grid = vec![0_i64; rows.numel()];
                let len = grid.len();
                rows.f_copy_data(&mut grid, len)?;
                let grid = grid
                    .chunks_exact(3)
                    .map(|r| [r[0], r[1], r[2]])
                    .collect::<Vec<_>>();
                let (merged, deepstack) = unsafe { vision.forward(rt, &patches, &grid, exec)? };
                let emb = merged
                    .f_view([1, -1, merged.size()[merged.dim() - 1]])?
                    .f_to_device_(options.execution_device, Kind::Float, false, false)?;
                let size = emb.size()[1];
                if emb.size()[2] == embedded.size()[2] {
                    let start = index + offset;
                    embedded = Tensor::f_cat(
                        &[
                            embedded.f_slice(1, 0, start, 1)?,
                            emb,
                            embedded.f_slice(1, start, i64::MAX, 1)?,
                        ],
                        1,
                    )?;
                    mask.splice(
                        start as usize..start as usize,
                        std::iter::repeat_n(1_i64, size as usize),
                    );
                    offset += size - 1;
                    image_info.push(super::ImageEmbeddingInfo {
                        index: start,
                        size,
                        grid,
                        deepstack,
                    });
                } else {
                    offset -= 1;
                    pad_extra += size;
                }
            }
            if pad_extra > 0 {
                let pad = Tensor::f_full(
                    [1, pad_extra],
                    151643,
                    (Kind::Int64, options.execution_device),
                )?;
                embedded = Tensor::f_cat(&[embedded, unsafe { self.embed(rt, &pad, exec)? }], 1)?;
                mask.extend(std::iter::repeat_n(0_i64, pad_extra as usize));
            }
            embeddings.push(embedded);
            masks.push(mask);
        }
        let embeds = Tensor::f_cat(&embeddings, 0)?;
        let width = masks.first().map(Vec::len).unwrap_or(0);
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
        let image_inputs =
            super::build_image_inputs(&image_info, embeds.size()[1], options.execution_device)?;
        let result = unsafe {
            self.forward(
                rt,
                &embeds,
                TextForwardOptions {
                    attention_mask: Some(&mask),
                    position_ids: image_inputs.position_ids.as_ref(),
                    deepstack: image_inputs.deepstack.as_deref().unwrap_or(&[]),
                    visual_mask: image_inputs.visual_mask.as_ref(),
                    ..TextForwardOptions::default()
                },
                exec,
            )?
        };
        let output = result
            .intermediate
            .ok_or_else(|| Error::BindingGap {
                backend: "qwen3vl".into(),
                operation: "Default pre-final-norm hidden state is absent".into(),
            })?
            .f_to_kind(Kind::Float)?;
        let mut parts = vec![];
        for (section, batch) in prompt.batches.iter().enumerate() {
            let z = output.f_slice(0, section as i64, section as i64 + 1, 1)?;
            if has_weights {
                let empty = output.f_get(output.size()[0] - 1)?;
                for (j, token) in batch.iter().enumerate() {
                    if token.weight != 1. {
                        let mut row = z.f_get(0)?.f_get(j as i64)?;
                        let neutral = empty.f_get(j as i64)?;
                        let weighted = row
                            .f_sub(&neutral)?
                            .f_mul_scalar(token.weight)?
                            .f_add(&neutral)?;
                        row.f_copy_(&weighted)?;
                    }
                }
            }
            parts.push(z);
        }
        let context = if parts.is_empty() {
            output.f_slice(0, -1, i64::MAX, 1)?
        } else {
            Tensor::f_cat(&parts, -2)?
        }
        .f_to_device(options.intermediate_device)?;
        let mask = mask
            .f_slice(0, 0, sections as i64, 1)?
            .f_flatten(0, -1)?
            .f_unsqueeze(0)?
            .f_to_device(options.intermediate_device)?;
        let first = prompt
            .batches
            .first()
            .ok_or_else(|| Error::ReferenceException {
                class: "IndexError",
                message: "list index out of range".into(),
            })?;
        super::filter_image_conditioning(
            &context,
            &mask,
            &first.iter().map(|t| t.token.clone()).collect::<Vec<_>>(),
            &image_info
                .iter()
                .map(|e| (e.index, e.size))
                .collect::<Vec<_>>(),
            prompt.keep_vision,
        )
    }

    pub fn from_parameters(
        config: TextEncoderConfig,
        state: &PreparedParameters,
    ) -> Result<Self, Error> {
        Self::from_parameters_with_final_norm(config, state, true)
    }
    /// The source Qwen3-VL constructor makes the final norm optional. H3's
    /// truncated language checkpoint selects false; existing image models use true.
    pub fn from_parameters_with_final_norm(
        config: TextEncoderConfig,
        state: &PreparedParameters,
        final_norm: bool,
    ) -> Result<Self, Error> {
        let mut blocks = vec![];
        for i in 0..config.layers {
            let prefix = format!("model.layers.{i}");
            let l = |suffix: &str, bias: bool| {
                Linear::from_parameters(state, &format!("{prefix}.{suffix}"), bias)
            };
            let w =
                |suffix: &str| parameter(state, &format!("{prefix}.{suffix}.weight"))?.as_tensor();
            blocks.push(LanguageBlock {
                q: l(
                    if config.merged_qkv {
                        "self_attn.qkv_proj"
                    } else {
                        "self_attn.q_proj"
                    },
                    config.qkv_bias,
                )?,
                k: if config.merged_qkv {
                    None
                } else {
                    Some(l("self_attn.k_proj", config.qkv_bias)?)
                },
                v: if config.merged_qkv {
                    None
                } else {
                    Some(l("self_attn.v_proj", config.qkv_bias)?)
                },
                out: l("self_attn.o_proj", false)?,
                gate: l(
                    if config.merged_mlp {
                        "mlp.gate_up_proj"
                    } else {
                        "mlp.gate_proj"
                    },
                    false,
                )?,
                up: if config.merged_mlp {
                    None
                } else {
                    Some(l("mlp.up_proj", false)?)
                },
                down: l("mlp.down_proj", false)?,
                input_norm: w("input_layernorm")?,
                post_norm: w("post_attention_layernorm")?,
                q_norm: w("self_attn.q_norm")?,
                k_norm: w("self_attn.k_norm")?,
            });
        }
        Ok(Self {
            embedding: parameter(state, "model.embed_tokens.weight")?,
            norm: if final_norm {
                Some(parameter(state, "model.norm.weight")?.as_tensor()?)
            } else {
                None
            },
            blocks,
            config,
        })
    }
    /// # Safety
    /// The source-selected embedding kernel must receive valid model indices,
    /// quantized storage and stream/device capacities.
    pub unsafe fn embed<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        ids: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        unsafe { super::layers::embedding_rows(rt, &self.embedding, ids, Some(Kind::Float)) }
    }

    /// # Safety
    /// Source model, operation and supplied shared execution contracts must hold.
    pub unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        embeds: &Tensor,
        options: TextForwardOptions<'_>,
        exec: &mut E,
    ) -> Result<TextEncoderOutput, Error> {
        let mut x = embeds.shallow_clone();
        let shape = x.size();
        let seq = shape[1];
        let positions = match options.position_ids {
            Some(p) => p.shallow_clone(),
            None => Tensor::f_arange(seq, (Kind::Int64, x.device()))?.f_unsqueeze(0)?,
        };
        let rope = text_rope(&self.config, &positions, x.device(), &rt.native)?;
        exec.observe("embeddings", &x)?;
        exec.observe("rope", &rope)?;
        let mut mask = options
            .attention_mask
            .map(|m| -> Result<Tensor, Error> {
                let s = m.size();
                let m = m
                    .f_to_kind(x.kind())?
                    .f_reshape([s[0], 1, -1, s[s.len() - 1]])?
                    .f_expand([s[0], 1, seq, s[s.len() - 1]], false)?;
                let m = m.f_rsub_scalar(1.)?;
                Ok(m.f_masked_fill(&m.f_to_kind(Kind::Bool)?, min_value(x.kind()) / 4.)?)
            })
            .transpose()?;
        if seq > 1 {
            let causal = Tensor::f_empty([seq, seq], (x.kind(), x.device()))?
                .f_fill_(min_value(x.kind()) / 4.)?
                .f_triu_(1)?;
            mask = Some(match mask {
                Some(mut m) => m.f_add_(&causal)?,
                None => causal,
            });
        }
        let selected = match &options.intermediate {
            Intermediate::Layer(n) => Some(if *n < 0 {
                self.blocks.len() as i64 + n
            } else {
                *n
            }),
            _ => None,
        };
        let collect = matches!(
            options.intermediate,
            Intermediate::All | Intermediate::Layers(_)
        );
        let wants = |i: usize| match &options.intermediate {
            Intermediate::All => true,
            Intermediate::Layers(l) => l.contains(&(i as i64)),
            _ => false,
        };
        let mut all = vec![];
        let mut intermediate = None;
        exec.queue_begin(self.blocks.len(), x.device())?;
        for (i, block) in self.blocks.iter().enumerate() {
            if wants(i) {
                all.push(rt.native.clone_tensor(&x.f_unsqueeze(1)?)?);
            }
            let mut core = |execution: &mut E| -> Result<(), Error> {
                x = unsafe {
                    block.forward(rt, &self.config, &x, mask.as_ref(), &rope, execution)?
                };
                Ok(())
            };
            exec.block(i, &mut core)?;
            exec.observe(&format!("block-{i}"), &x)?;
            if let Some(deep) = options.deepstack.get(i) {
                let mask = options.visual_mask.ok_or_else(|| Error::BindingGap {
                    backend: "qwen3vl".into(),
                    operation: "Missing visual position mask".into(),
                })?;
                let values = x.f_index(&[Some(mask)])?.f_add(&deep.f_to_device_(
                    x.device(),
                    x.kind(),
                    false,
                    false,
                )?)?;
                x = x.f_index_put_(&[Some(mask)], &values, false)?;
            }
            if selected == Some(i as i64) {
                intermediate = Some(rt.native.clone_tensor(&x)?);
            }
        }
        exec.queue_end(x.device())?;
        if let Some(norm) = &self.norm {
            x = rms(&x, norm, self.config.eps)?;
        }
        if wants(self.blocks.len()) {
            all.push(rt.native.clone_tensor(&x.f_unsqueeze(1)?)?);
        }
        if collect {
            intermediate = Some(Tensor::f_cat(&all, 1)?);
        }
        if let Some(norm) = &self.norm
            && options.final_layer_norm_intermediate
        {
            intermediate = intermediate
                .map(|t| rms(&t, norm, self.config.eps))
                .transpose()?;
        }
        Ok(TextEncoderOutput {
            last: x,
            intermediate,
        })
    }
}
impl LanguageBlock {
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        config: &TextEncoderConfig,
        input: &Tensor,
        mask: Option<&Tensor>,
        rope: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let project = |linear: &Linear, x: &Tensor, exec: &mut E| -> Result<Tensor, Error> {
            exec.check_interrupt()?;
            unsafe { linear.forward(rt, x) }
        };
        let shape = input.size();
        let b = shape[0];
        let seq = shape[1];
        let norm = rms(input, &self.input_norm, config.eps)?;
        exec.observe("input_norm", &norm)?;
        let (q, k, v) = if let Some(k) = &self.k {
            (
                project(&self.q, &norm, exec)?,
                project(k, &norm, exec)?,
                project(self.v.as_ref().unwrap(), &norm, exec)?,
            )
        } else {
            let qkv = project(&self.q, &norm, exec)?.f_split_with_sizes(
                [
                    config.heads * config.head_dim,
                    config.kv_heads * config.head_dim,
                    config.kv_heads * config.head_dim,
                ],
                -1,
            )?;
            (
                qkv[0].shallow_clone(),
                qkv[1].shallow_clone(),
                qkv[2].shallow_clone(),
            )
        };
        exec.observe("q_proj", &q)?;
        exec.observe("k_proj", &k)?;
        exec.observe("v_proj", &v)?;
        let q = rms(
            &q.f_view([b, seq, config.heads, config.head_dim])?
                .f_transpose(1, 2)?,
            &self.q_norm,
            config.eps,
        )?;
        let k = rms(
            &k.f_view([b, seq, config.kv_heads, config.head_dim])?
                .f_transpose(1, 2)?,
            &self.k_norm,
            config.eps,
        )?;
        let v = v
            .f_view([b, seq, config.kv_heads, config.head_dim])?
            .f_transpose(1, 2)?;
        exec.observe("q_norm", &q)?;
        exec.observe("k_norm", &k)?;
        let rope = unsafe {
            rt.call_kitchen(
                "apply_rope_split_half",
                &vec![
                    ("xq".into(), Value::tensor(q)),
                    ("xk".into(), Value::tensor(k)),
                    ("freqs_cis".into(), Value::tensor(rope.shallow_clone())),
                ],
            )?
        };
        let Value::Tuple(qk) = rope else {
            return Err(Error::Native(
                "Invalid source split-half RoPE result".into(),
            ));
        };
        exec.observe("q_rope", &qk[0].as_tensor()?)?;
        exec.observe("k_rope", &qk[1].as_tensor()?)?;
        let attention = exec.attention(
            rt,
            &qk[0].as_tensor()?,
            &qk[1].as_tensor()?,
            &v,
            mask,
            config.heads,
            config.heads != config.kv_heads,
        )?;
        exec.observe("attention", &attention)?;
        let x = input.f_add(&project(&self.out, &attention, exec)?)?;
        let norm = rms(&x, &self.post_norm, config.eps)?;
        let gate = project(&self.gate, &norm, exec)?;
        let output = if let Some(up) = &self.up {
            let gate = gate.f_silu()?;
            let up = project(up, &norm, exec)?;
            project(&self.down, &gate.f_mul(&up)?, exec)?
        } else {
            unsafe {
                self.down
                    .swiglu(rt, &gate, false, &mut || exec.check_interrupt())?
            }
        };
        Ok(x.f_add_out(input, &output)?)
    }
}
