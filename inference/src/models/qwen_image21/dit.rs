//! Complete model.py DiT computation, with explicit shared runtime boundaries.
use super::{
    geometry::assemble_sequence,
    layers::{self, Linear, Modulation},
    *,
};
use crate::checkpoint::PreparedParameters;
use crate::{execution::Runtime, execution_value::Value, native::Error};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

pub struct DiTConfig {
    pub in_channels: i64,
    pub out_channels: i64,
    pub num_layers: usize,
    pub head_dim: i64,
    pub heads: i64,
    pub context_dim: i64,
    pub mlp_ratio: i64,
    pub axes: [i64; 3],
    pub eps: f64,
    pub fused_mlp: bool,
}
impl Default for DiTConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            num_layers: 32,
            head_dim: 128,
            heads: 32,
            context_dim: 4096,
            mlp_ratio: 3,
            axes: [16, 56, 56],
            eps: 1e-6,
            fused_mlp: true,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lifecycle {
    Queue,
    GraphBegin,
    Block(usize),
    PauseGraph,
    ResumeGraph,
    QueueEnd,
    GraphEnd,
}
#[derive(Default)]
pub struct ForwardState {
    pub total_blocks: usize,
    pub block_index: usize,
    pub block_type: &'static str,
}

/// Shared runtime integration points. Implementations must use the selected
/// original attention, hook, allocation, prefetch and cache services. None of
/// these callbacks implicitly selects a substitute implementation.
pub trait DiTExecution {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn training(&self) -> bool {
        false
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: i64,
        mask: Option<&Tensor>,
        block: usize,
        preferred: Option<&serde_json::Value>,
    ) -> Result<Tensor, Error>;
    fn lifecycle(&mut self, event: Lifecycle) -> Result<(), Error>;
    fn forward_metadata(&mut self, _total_blocks: usize) -> Result<(), Error> {
        Ok(())
    }
    fn forward_state(&mut self, _state: &ForwardState) -> Result<(), Error> {
        Ok(())
    }
    fn post_input(&mut self, x: Tensor, pe: Tensor) -> Result<(Tensor, Tensor), Error> {
        Ok((x, pe))
    }
    fn single_block(
        &mut self,
        x: Tensor,
        _input: &Tensor,
        _state: &ForwardState,
    ) -> Result<Tensor, Error> {
        Ok(x)
    }
    fn has_attn_patch(&self) -> bool {
        false
    }
    fn attn_patch(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pe: Tensor,
        _img_slice: [i64; 2],
        _state: &ForwardState,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), Error> {
        Ok((q, k, v, pe))
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    fn replace_block(
        &mut self,
        _state: &ForwardState,
        _x: &Tensor,
        _temb: &Tensor,
        _pe: &Tensor,
        _modulation: &Modulation,
        _prefix: i64,
        _original: &mut dyn FnMut(Tensor, Tensor, &mut Self) -> Result<Tensor, Error>,
    ) -> Result<Option<Tensor>, Error>
    where
        Self: Sized,
    {
        Ok(None)
    }
    /// WrapperExecutor boundary, including chained calls of the original body.
    fn wrapper(
        &mut self,
        input: &ForwardInputs<'_>,
        original: &mut dyn FnMut(&ForwardInputs<'_>, &mut Self) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error>
    where
        Self: Sized,
    {
        original(input, self)
    }
    /// The source disables prefix caching whenever any of its four hook groups
    /// is installed, including block replacements. The host owns this metadata.
    fn hooked(&self) -> bool {
        self.has_attn_patch()
    }
}

/// The source PoseBranchCache contract. The sampling owner supplies the same
/// memory, dtype, slot, transfer and lifecycle implementation to the DiT.
pub trait PrefixKv {
    fn select(
        &mut self,
        key: &Tensor,
        cache_bytes: usize,
        num_blocks: usize,
        device: Device,
    ) -> Result<CacheSelection, Error>;
    fn put(&mut self, block: usize, kv: Tensor) -> Result<(), Error>;
    fn take(
        &mut self,
        block: usize,
        device: Device,
        dtype: Kind,
        batch: i64,
    ) -> Result<Tensor, Error>;
    fn prefetch(&mut self, block: usize, device: Device, dtype: Kind) -> Result<(), Error>;
    fn free(&mut self) -> Result<(), Error>;
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CacheSelection {
    Disabled,
    Fill,
    Hit,
}

pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    heads: i64,
    eps: f64,
    preferred: Option<serde_json::Value>,
}
struct Mlp {
    gate: Linear,
    up: Option<Linear>,
    out: Linear,
}
pub struct TransformerBlock {
    attn: Attention,
    mlp: Mlp,
    eps: f64,
}
pub struct ForwardInputs<'a> {
    pub x: &'a Tensor,
    pub timesteps: &'a Tensor,
    pub context: &'a Tensor,
    pub refs: &'a [Tensor],
    pub slots: &'a [i64],
}
pub struct DiT {
    pub config: DiTConfig,
    text_norm: Tensor,
    text_in: Linear,
    text_out: Linear,
    image: Linear,
    time1: Linear,
    time2: Linear,
    modulation: Linear,
    pub blocks: Vec<TransformerBlock>,
    last: Linear,
    output: Linear,
    pub prefix_cache_enabled: bool,
}
unsafe fn project<E: DiTExecution>(
    linear: &Linear,
    rt: &Runtime,
    x: &Tensor,
    exec: &mut E,
) -> Result<Tensor, Error> {
    exec.check_interrupt()?;
    unsafe { linear.forward(rt, x) }
}
fn missing(name: &str) -> Error {
    Error::Native(format!("Missing prepared Qwen Image 2.1 parameter: {name}"))
}
fn parameter(state: &BTreeMap<String, Value>, name: &str) -> Result<Value, Error> {
    state.get(name).cloned().ok_or_else(|| missing(name))
}
fn preference(
    state: &PreparedParameters,
    prefix: &str,
) -> Result<Option<serde_json::Value>, Error> {
    state.attention_config(&format!("{prefix}.attn.comfy_attention.config"))
}
fn linear(state: &PreparedParameters, name: &str, bias: bool) -> Result<Linear, Error> {
    Linear::from_parameters(state, name, bias)
}
impl TransformerBlock {
    fn load(state: &PreparedParameters, prefix: &str, config: &DiTConfig) -> Result<Self, Error> {
        let l = |name: &str| linear(state, &format!("{prefix}.{name}"), false);
        Ok(Self {
            attn: Attention {
                q: l("attn.to_q")?,
                k: l("attn.to_k")?,
                v: l("attn.to_v")?,
                out: l("attn.to_out.0")?,
                q_norm: parameter(state, &format!("{prefix}.attn.norm_q.weight"))?.as_tensor()?,
                k_norm: parameter(state, &format!("{prefix}.attn.norm_k.weight"))?.as_tensor()?,
                heads: config.heads,
                eps: config.eps,
                preferred: preference(state, prefix)?,
            },
            mlp: Mlp {
                gate: l(if config.fused_mlp {
                    "img_mlp.gate_up"
                } else {
                    "img_mlp.gate_layer"
                })?,
                up: if config.fused_mlp {
                    None
                } else {
                    Some(l("img_mlp.proj")?)
                },
                out: l("img_mlp.out")?,
            },
            eps: config.eps,
        })
    }
    /// # Safety
    /// Inputs, resident parameters and all execution callbacks must satisfy the
    /// selected original operators' contracts, including shared storage mutations.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    pub unsafe fn forward<E: DiTExecution>(
        &self,
        rt: &Runtime,
        x: Tensor,
        m: &Modulation,
        pe: &Tensor,
        route: &mut AttentionRoute<'_>,
        prefix: i64,
        state: &ForwardState,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        if exec.training() {
            exec.check_interrupt()?;
        }
        let norm = unsafe {
            modulated_norm(
                rt,
                &x,
                (&m.scale1.0, &m.scale1.1),
                prefix,
                &m.zero,
                self.eps,
                exec.training(),
            )?
        };
        let attn = unsafe {
            self.attn
                .forward(rt, &norm, pe, route, prefix, state, exec)?
        };
        let x = gated_residual(&x, &attn, (&m.gate1.0, &m.gate1.1), prefix)?;
        if exec.training() {
            exec.check_interrupt()?;
        }
        let norm = unsafe {
            modulated_norm(
                rt,
                &x,
                (&m.scale2.0, &m.scale2.1),
                prefix,
                &m.zero,
                self.eps,
                exec.training(),
            )?
        };
        let y = unsafe { project(&self.mlp.gate, rt, &norm, exec)? };
        let out = if let Some(up) = &self.mlp.up {
            let other = unsafe { project(up, rt, &norm, exec)? };
            unsafe { project(&self.mlp.out, rt, &y.f_silu()?.f_mul(&other)?, exec)? }
        } else {
            unsafe {
                self.mlp
                    .out
                    .swiglu(rt, &y, exec.training(), &mut || exec.check_interrupt())?
            }
        };
        layers::clamp_fp16(gated_residual(&x, &out, (&m.gate2.0, &m.gate2.1), prefix)?)
    }
}
/// Original block-causal, prefix-fill and prefix-hit attention routes.
pub enum AttentionRoute<'a> {
    Segments {
        segments: &'a [Segment],
        store: Option<(&'a mut dyn PrefixKv, usize, i64)>,
    },
    Cached {
        key: Tensor,
        value: Tensor,
    },
}
impl AttentionRoute<'_> {
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    fn run<E: DiTExecution>(
        &mut self,
        rt: &Runtime,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        heads: i64,
        state: &ForwardState,
        preferred: Option<&serde_json::Value>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        match self {
            Self::Segments { segments, store } => {
                if let Some((cache, index, prefix)) = store {
                    cache.put(
                        *index,
                        Tensor::f_stack(
                            &[k.f_slice(1, 0, *prefix, 1)?, v.f_slice(1, 0, *prefix, 1)?],
                            1,
                        )?,
                    )?;
                }
                let mut outs = vec![];
                for s in *segments {
                    outs.push(exec.attention(
                        rt,
                        &q.f_slice(1, s.start, s.end, 1)?.f_flatten(2, -1)?,
                        &k.f_slice(1, 0, s.end, 1)?.f_flatten(2, -1)?,
                        &v.f_slice(1, 0, s.end, 1)?.f_flatten(2, -1)?,
                        heads,
                        s.mask.as_ref(),
                        state.block_index,
                        preferred,
                    )?);
                }
                if outs.len() == 1 {
                    Ok(outs.remove(0))
                } else {
                    Ok(Tensor::f_cat(&outs, 1)?)
                }
            }
            Self::Cached { key, value } => exec.attention(
                rt,
                &q.f_flatten(2, -1)?,
                &Tensor::f_cat(&[key.shallow_clone(), k], 1)?.f_flatten(2, -1)?,
                &Tensor::f_cat(&[value.shallow_clone(), v], 1)?.f_flatten(2, -1)?,
                heads,
                None,
                state.block_index,
                preferred,
            ),
        }
    }
}
impl Attention {
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    unsafe fn forward<E: DiTExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        pe: &Tensor,
        route: &mut AttentionRoute<'_>,
        prefix: i64,
        state: &ForwardState,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let shape = x.size();
        let view = |t: Tensor| t.f_view([shape[0], shape[1], self.heads, -1]);
        let mut q = view(unsafe { project(&self.q, rt, x, exec)? })?;
        let mut k = view(unsafe { project(&self.k, rt, x, exec)? })?;
        let mut v = view(unsafe { project(&self.v, rt, x, exec)? })?;
        if exec.training() || exec.has_attn_patch() {
            exec.check_interrupt()?;
            let qw = self
                .q_norm
                .f_to_device_(x.device(), x.kind(), false, false)?;
            q = q.f_rms_norm([q.size()[3]], Some(&qw), Some(self.eps))?;
            exec.check_interrupt()?;
            let kw = self
                .k_norm
                .f_to_device_(x.device(), x.kind(), false, false)?;
            k = k.f_rms_norm([k.size()[3]], Some(&kw), Some(self.eps))?;
            let mut pe = pe.shallow_clone();
            if exec.has_attn_patch() {
                (q, k, v, pe) = exec.attn_patch(
                    q.f_transpose(1, 2)?,
                    k.f_transpose(1, 2)?,
                    v.f_transpose(1, 2)?,
                    pe.f_transpose(1, 2)?,
                    [prefix, shape[1]],
                    state,
                )?;
                q = q.f_transpose(1, 2)?.f_contiguous()?;
                k = k.f_transpose(1, 2)?.f_contiguous()?;
                v = v.f_transpose(1, 2)?.f_contiguous()?;
                pe = pe.f_transpose(1, 2)?.f_contiguous()?;
            }
            if exec.training() {
                q = layers::training_rope(&q, &pe)?;
                k = layers::training_rope(&k, &pe)?;
            } else {
                q = unsafe {
                    rt.call_kitchen(
                        "apply_rope1",
                        &vec![
                            ("x".into(), Value::tensor(q)),
                            ("freqs_cis".into(), Value::tensor(pe.shallow_clone())),
                        ],
                    )?
                    .as_tensor()?
                };
                k = unsafe {
                    rt.call_kitchen(
                        "apply_rope1",
                        &vec![
                            ("x".into(), Value::tensor(k)),
                            ("freqs_cis".into(), Value::tensor(pe)),
                        ],
                    )?
                    .as_tensor()?
                };
            }
        } else {
            let qw = self
                .q_norm
                .f_to_device_(x.device(), x.kind(), false, false)?;
            let kw = self
                .k_norm
                .f_to_device_(x.device(), x.kind(), false, false)?;
            let out = unsafe {
                rt.call_kitchen(
                    "rms_rope",
                    &vec![
                        ("q".into(), Value::tensor(q)),
                        ("k".into(), Value::tensor(k)),
                        ("freqs_cis".into(), Value::tensor(pe.shallow_clone())),
                        ("q_scale".into(), Value::tensor(qw)),
                        ("k_scale".into(), Value::tensor(kw)),
                        ("epsilon".into(), Value::Float(self.eps)),
                    ],
                )?
            };
            let Value::Tuple(mut outputs) = out else {
                return Err(Error::Native("Invalid rms_rope result".into()));
            };
            k = outputs.pop().unwrap().as_tensor()?;
            q = outputs.pop().unwrap().as_tensor()?;
        }
        let out = route.run(
            rt,
            q,
            k,
            v,
            self.heads,
            state,
            self.preferred.as_ref(),
            exec,
        )?;
        unsafe { project(&self.out, rt, &out, exec) }
    }
}
impl DiT {
    /// Bind an already prepared state dictionary. Loading model files and
    /// patching remain the shared loader's responsibility; layout metadata and
    /// parameter identity are retained rather than materialized to BF16.
    pub fn from_prepared(
        config: DiTConfig,
        state: &BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        Self::from_parameters(config, &PreparedParameters::plain(state.clone()))
    }
    pub fn from_parameters(config: DiTConfig, state: &PreparedParameters) -> Result<Self, Error> {
        let l = |name| linear(state, name, false);
        let blocks = (0..config.num_layers)
            .map(|i| TransformerBlock::load(state, &format!("transformer_blocks.{i}"), &config))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            text_norm: parameter(state, "txt_in.text_norm.weight")?.as_tensor()?,
            text_in: l("txt_in.in_layer")?,
            text_out: l("txt_in.out_layer")?,
            image: l("img_in")?,
            time1: l("time_text_embed.timestep_embedder.linear_1")?,
            time2: l("time_text_embed.timestep_embedder.linear_2")?,
            modulation: l("modulation.1")?,
            last: l("norm_out.linear")?,
            output: l("proj_out")?,
            blocks,
            config,
            prefix_cache_enabled: false,
        })
    }
    pub fn reset_prefix_cache(
        &mut self,
        enabled: bool,
        cache: &mut dyn PrefixKv,
    ) -> Result<(), Error> {
        cache.free()?;
        self.prefix_cache_enabled = enabled;
        Ok(())
    }
    /// # Safety
    /// All source model shape/dtype/parameter contracts and the supplied shared
    /// runtime's kernel, callback, cache and lifecycle contracts must hold.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    pub unsafe fn forward<E: DiTExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        timesteps: &Tensor,
        context: &Tensor,
        refs: &[Tensor],
        slots: &[i64],
        mut cache: Option<&mut dyn PrefixKv>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let input = ForwardInputs {
            x,
            timesteps,
            context,
            refs,
            slots,
        };
        let mut original = |input: &ForwardInputs<'_>, execution: &mut E| unsafe {
            self.forward_original(rt, input, &mut cache, execution)
        };
        exec.wrapper(&input, &mut original)
    }
    /// The original unwrapped _forward body.
    /// # Safety
    /// Same contracts as forward, including all provided shared runtime services.
    pub unsafe fn forward_original<E: DiTExecution>(
        &self,
        rt: &Runtime,
        input: &ForwardInputs<'_>,
        cache: &mut Option<&mut dyn PrefixKv>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let ForwardInputs {
            x,
            timesteps,
            context,
            refs,
            slots,
        } = *input;
        let shape = x.size();
        let dtype = x.kind();
        let txt = zero_centered_rms_norm(context, &self.text_norm, self.config.eps)?;
        let txt = unsafe { project(&self.text_in, rt, &txt, exec)? }.f_gelu("tanh")?;
        let txt = unsafe { project(&self.text_out, rt, &txt, exec)? };
        let pe = PositionEncoding {
            axes: self.config.axes,
            theta: 10000.,
            supports_fp64: true,
        };
        let sequence = assemble_sequence(x, &txt, refs, slots, &pe, |input| unsafe {
            project(&self.image, rt, input, exec)
        })?;
        let (mut hidden, mut pe) = exec.post_input(sequence.hidden, sequence.position_encoding)?;
        let mut prefix = hidden.size()[1] - shape[2] * shape[3];
        let t = round_timesteps(timesteps, dtype)?;
        let t = Tensor::f_cat(
            &[t.shallow_clone(), t.f_new_zeros([1], (dtype, x.device()))?],
            0,
        )?;
        let embedding =
            timestep_embedding(&t.f_to_kind(Kind::Float)?, 256, 10000., 1000.)?.f_to_kind(dtype)?;
        let temb = unsafe {
            project(
                &self.time2,
                rt,
                &project(&self.time1, rt, &embedding, exec)?.f_silu()?,
                exec,
            )?
        };
        let modulation = Modulation::from_projection(unsafe {
            project(&self.modulation, rt, &temb.f_silu()?, exec)?
        })?;
        let selection = if self.prefix_cache_enabled && prefix > 0 && !exec.hooked() {
            if let Some(c) = cache.as_deref_mut() {
                let bytes = 2
                    * self.blocks.len()
                    * shape[0] as usize
                    * prefix as usize
                    * (self.config.heads * self.config.head_dim) as usize
                    * hidden.kind().elt_size_in_bytes();
                c.select(
                    &prefix_cache_key(x, context, refs, slots)?,
                    bytes,
                    self.blocks.len(),
                    x.device(),
                )?
            } else {
                return Err(Error::BindingGap {
                    backend: "qwen_image21".into(),
                    operation: "enabled prefix cache requires the shared cache owner".into(),
                });
            }
        } else {
            CacheSelection::Disabled
        };
        let mut prefix_states = None;
        if selection == CacheSelection::Hit {
            hidden = hidden.f_slice(1, prefix, i64::MAX, 1)?;
            pe = pe.f_slice(1, prefix, i64::MAX, 1)?;
            prefix = 0;
        } else if selection == CacheSelection::Fill {
            prefix_states = Some((
                hidden.f_slice(1, 0, prefix, 1)?,
                pe.f_slice(1, 0, prefix, 1)?,
            ));
            hidden = hidden.f_slice(1, prefix, i64::MAX, 1)?;
            pe = pe.f_slice(1, prefix, i64::MAX, 1)?;
            prefix = 0;
        }
        let mut state = ForwardState {
            total_blocks: self.blocks.len(),
            block_index: 0,
            block_type: "single",
        };
        exec.forward_metadata(state.total_blocks)?;
        exec.lifecycle(Lifecycle::Queue)?;
        exec.lifecycle(Lifecycle::GraphBegin)?;
        for (i, block) in self.blocks.iter().enumerate() {
            exec.lifecycle(Lifecycle::Block(i))?;
            state.block_index = i;
            exec.forward_state(&state)?;
            let mut route = if selection != CacheSelection::Disabled {
                let c = cache.as_deref_mut().unwrap();
                if selection == CacheSelection::Fill {
                    exec.lifecycle(Lifecycle::PauseGraph)?;
                    let (old, pp) = prefix_states.take().unwrap();
                    let plen = old.size()[1];
                    let mut prefix_route = AttentionRoute::Segments {
                        segments: &sequence.segments[..sequence.segments.len() - 1],
                        store: Some((c, i, plen)),
                    };
                    let result = unsafe {
                        block.forward(
                            rt,
                            old,
                            &modulation,
                            &pp,
                            &mut prefix_route,
                            plen,
                            &state,
                            exec,
                        )
                    };
                    // The source context manager resumes even if the prefix
                    // block is interrupted or fails. GraphEnd remains outside
                    // a finally block, exactly as in the original _forward.
                    exec.lifecycle(Lifecycle::ResumeGraph)?;
                    prefix_states = Some((result?, pp));
                }
                let kv = c.take(i, x.device(), dtype, shape[0])?;
                if selection == CacheSelection::Hit {
                    c.prefetch(i + 1, x.device(), dtype)?;
                }
                AttentionRoute::Cached {
                    key: kv.f_select(1, 0)?,
                    value: kv.f_select(1, 1)?,
                }
            } else {
                AttentionRoute::Segments {
                    segments: &sequence.segments,
                    store: None,
                }
            };
            let mut original = |input: Tensor, position: Tensor, execution: &mut E| unsafe {
                block.forward(
                    rt,
                    input,
                    &modulation,
                    &position,
                    &mut route,
                    prefix,
                    &state,
                    execution,
                )
            };
            hidden = if let Some(replaced) = exec.replace_block(
                &state,
                &hidden,
                &temb,
                &pe,
                &modulation,
                prefix,
                &mut original,
            )? {
                replaced
            } else {
                original(hidden, pe.shallow_clone(), exec)?
            };
            hidden = exec.single_block(hidden, x, &state)?;
        }
        exec.lifecycle(Lifecycle::QueueEnd)?;
        exec.lifecycle(Lifecycle::GraphEnd)?;
        hidden = hidden.f_slice(1, prefix, i64::MAX, 1)?;
        let scale =
            unsafe { project(&self.last, rt, &temb.f_slice(0, 0, -1, 1)?.f_silu()?, exec)? }
                .f_unsqueeze(1)?;
        hidden = if exec.training() {
            exec.check_interrupt()?;
            hidden
                .f_layer_norm(
                    [self.config.heads * self.config.head_dim],
                    None::<&Tensor>,
                    None::<&Tensor>,
                    self.config.eps,
                    true,
                )?
                .f_mul(&scale.f_add_scalar(1)?)?
        } else {
            unsafe {
                rt.call_kitchen(
                    "adaln",
                    &vec![
                        ("x".into(), Value::tensor(hidden)),
                        ("scale".into(), Value::tensor(scale.shallow_clone())),
                        (
                            "shift".into(),
                            Value::tensor(scale.f_slice(0, 0, 1, 1)?.f_zeros_like()?),
                        ),
                        ("eps".into(), Value::Float(self.config.eps)),
                    ],
                )?
                .as_tensor()?
            }
        };
        let out = unsafe { project(&self.output, rt, &hidden, exec)? };
        Ok(out.f_transpose(1, 2)?.f_reshape([
            shape[0],
            self.config.out_channels,
            shape[2],
            shape[3],
        ])?)
    }
}

/// Original VACE-style control stream; skip application is owned by its patch.
pub struct FunControl {
    image: Linear,
    before: Linear,
    blocks: Vec<TransformerBlock>,
    after: Vec<Linear>,
}
impl FunControl {
    pub fn from_prepared(
        config: &DiTConfig,
        state: &BTreeMap<String, Value>,
    ) -> Result<Self, Error> {
        Self::from_parameters(config, &PreparedParameters::plain(state.clone()))
    }
    pub fn from_parameters(config: &DiTConfig, state: &PreparedParameters) -> Result<Self, Error> {
        let blocks = (0..config.num_layers)
            .map(|i| TransformerBlock::load(state, &format!("control_blocks.{i}"), config))
            .collect::<Result<_, _>>()?;
        let after = (0..config.num_layers)
            .map(|i| linear(state, &format!("control_blocks.{i}.after_proj"), true))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            image: linear(state, "control_img_in", true)?,
            before: linear(state, "control_blocks.0.before_proj", true)?,
            blocks,
            after,
        })
    }
    /// # Safety
    /// Same original resident linear and sequence contracts as DiT::forward.
    pub unsafe fn init_stream<E: DiTExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        control: &Tensor,
        prefix: i64,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let joint = x.f_zeros_like()?;
        joint
            .f_slice(1, prefix, i64::MAX, 1)?
            .f_copy_(&unsafe { project(&self.image, rt, control, exec)? })?;
        Ok(unsafe { project(&self.before, rt, &joint, exec)? }.f_add(x)?)
    }
    /// # Safety
    /// Same original transformer, callback and attention contracts as a base block.
    #[expect(
        clippy::too_many_arguments,
        reason = "Keep the frozen model callback arguments explicit across the Rust runtime boundary"
    )]
    pub unsafe fn step<E: DiTExecution>(
        &self,
        rt: &Runtime,
        index: usize,
        c: Tensor,
        m: &Modulation,
        pe: &Tensor,
        route: &mut AttentionRoute<'_>,
        prefix: i64,
        state: &ForwardState,
        exec: &mut E,
    ) -> Result<(Tensor, Tensor), Error> {
        let c = unsafe { self.blocks[index].forward(rt, c, m, pe, route, prefix, state, exec)? };
        let skip = unsafe { project(&self.after[index], rt, &c, exec)? };
        Ok((c, skip))
    }
}
