//! `comfy/ldm/moss_image/model.py`, retaining operation and in-place boundaries.
use crate::models::qwen_image21::{Linear, timestep_embedding};
use crate::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value, native::Error,
};
use tch::{Device, Kind, Tensor};

pub struct MossConfig {
    pub dim: i64,
    pub ffn_dim: i64,
    pub heads: i64,
    pub layers: usize,
    pub text_dim: i64,
    pub latent_channels: i64,
    pub fused_qkv: bool,
}
impl Default for MossConfig {
    fn default() -> Self {
        Self {
            dim: 3072,
            ffn_dim: 8192,
            heads: 24,
            layers: 42,
            text_dim: 4096,
            latent_channels: 128,
            fused_qkv: false,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MossLifecycle {
    TextQueue,
    TextBlock(usize),
    TextQueueEnd,
    Queue,
    GraphBegin,
    Block(usize),
    QueueEnd,
    GraphEnd,
}
pub struct MossState {
    pub total_blocks: usize,
    pub block_index: usize,
}
pub struct MossInputs<'a> {
    pub latent: &'a Tensor,
    pub timestep: &'a Tensor,
    pub context: &'a Tensor,
}
pub type MossOriginalBlock<'a, E> =
    dyn FnMut(&Tensor, &[Tensor], &Tensor, &mut E) -> Result<Tensor, Error> + 'a;
pub trait MossExecution {
    fn training(&self) -> bool {
        false
    }
    fn check_interrupt(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn lifecycle(&mut self, event: MossLifecycle) -> Result<(), Error>;
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: i64,
        module: &str,
    ) -> Result<Tensor, Error>;
    fn replace_block(
        &mut self,
        _state: &MossState,
        _x: &Tensor,
        _vec: &Tensor,
        _modulation: &[Tensor],
        _pe: &Tensor,
        _original: &mut MossOriginalBlock<'_, Self>,
    ) -> Result<Option<Tensor>, Error>
    where
        Self: Sized,
    {
        Ok(None)
    }
    fn wrapper(
        &mut self,
        inputs: &MossInputs<'_>,
        original: &mut dyn FnMut(&MossInputs<'_>, &mut Self) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error>
    where
        Self: Sized,
    {
        original(inputs, self)
    }
}

fn tensor(state: &PreparedParameters, name: &str) -> Result<Tensor, Error> {
    state
        .get(name)
        .ok_or_else(|| Error::BindingGap {
            backend: "moss_image".into(),
            operation: format!("Missing prepared {name}"),
        })?
        .as_tensor()
}
fn cast(weight: &Tensor, x: &Tensor) -> Result<Tensor, Error> {
    Ok(weight.f_to_device_(x.device(), x.kind(), false, false)?)
}
fn norm(x: &Tensor, weight: &Tensor) -> Result<Tensor, Error> {
    let w = cast(weight, x)?;
    Ok(x.f_rms_norm([*x.size().last().unwrap()], Some(&w), Some(1e-6))?)
}
pub fn moss_situ(hidden: &Tensor, gate: &Tensor) -> Result<Tensor, Error> {
    Ok(gate
        .f_div_scalar(4.)?
        .f_tanh()?
        .f_mul_scalar(4.)?
        .f_mul(&gate.f_sigmoid()?)?
        .f_mul(&hidden.f_div_scalar(25.)?.f_tanh()?.f_mul_scalar(25.)?)?)
}
fn residual(x: &Tensor, y: &Tensor, gate: &Tensor, training: bool) -> Result<Tensor, Error> {
    Ok(if training {
        x.f_addcmul(y, gate)?
    } else {
        x.shallow_clone().f_addcmul_(y, gate)?
    })
}
unsafe fn mod_norm<E: MossExecution>(
    rt: &Runtime,
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    e: &mut E,
) -> Result<Tensor, Error> {
    if e.training() {
        e.check_interrupt()?;
        return Ok(norm(x, weight)?
            .f_mul(&scale.f_add_scalar(1.)?)?
            .f_add(shift)?);
    }
    let scale = cast(weight, x)?
        .f_mul(&scale.f_add_scalar(1.)?)?
        .f_sub_scalar(1.)?;
    unsafe {
        rt.call_kitchen(
            "rms_adaln",
            &vec![
                ("x".into(), Value::tensor(x.shallow_clone())),
                ("scale".into(), Value::tensor(scale)),
                ("shift".into(), Value::tensor(shift.shallow_clone())),
                ("eps".into(), Value::Float(1e-6)),
            ],
        )?
        .as_tensor()
    }
}
pub fn moss_rope_split_half(x: &Tensor, pe: &Tensor) -> Result<Tensor, Error> {
    let shape = x.size();
    let view = x
        .f_to_kind(pe.kind())?
        .f_unflatten(-1, [2, -1])?
        .f_movedim_int(-2, -1)?
        .f_unsqueeze(-2)?;
    Ok(pe
        .f_select(-1, 0)?
        .f_mul(&view.f_select(-1, 0)?)?
        .f_add(&pe.f_select(-1, 1)?.f_mul(&view.f_select(-1, 1)?)?)?
        .f_movedim_int(-1, -2)?
        .f_flatten(-2, -1)?
        .f_reshape(shape.as_slice())?
        .f_to_kind(x.kind())?)
}
/// Original four-axis EmbedND. Image coordinates and text-length displacement
/// are retained; unlike Qwen Image, cond/uncond lengths cannot be repeated.
pub fn moss_rope(
    h: i64,
    w: i64,
    text: i64,
    head_dim: i64,
    device: Device,
    fp64: bool,
) -> Result<Tensor, Error> {
    let ids = Tensor::f_zeros([h * w + text, 4], (Kind::Float, device))?;
    let _ = ids.f_narrow(0, 0, h * w)?.f_select(1, 1)?.f_fill_(text)?;
    let rows = Tensor::f_arange(h, (Kind::Float, device))?
        .f_sub_scalar((h - 1) as f64 / 2.)?
        .f_repeat_interleave_self_int(w, Some(0), None)?;
    let cols = Tensor::f_arange(w, (Kind::Float, device))?
        .f_sub_scalar((w - 1) as f64 / 2.)?
        .f_repeat([h])?;
    ids.f_narrow(0, 0, h * w)?.f_select(1, 2)?.f_copy_(&rows)?;
    ids.f_narrow(0, 0, h * w)?.f_select(1, 3)?.f_copy_(&cols)?;
    ids.f_narrow(0, h * w, text)?
        .f_select(1, 1)?
        .f_copy_(&Tensor::f_arange(text, (Kind::Float, device))?)?;
    let ids = ids.f_unsqueeze(0)?;
    let dim = head_dim / 4;
    let mut axes = Vec::new();
    for axis in 0..4 {
        let dev = if fp64 { device } else { Device::Cpu };
        let omega = Tensor::f_pow_scalar(
            10000.,
            &Tensor::f_linspace(
                0.,
                (dim - 2) as f64 / dim as f64,
                dim / 2,
                (Kind::Double, dev),
            )?,
        )?
        .f_reciprocal()?
        .f_mul_scalar(1.)?;
        let pos = ids
            .f_select(-1, axis)?
            .f_to_device_(dev, Kind::Float, false, false)?;
        let out = Tensor::f_einsum("...n,d->...nd", &[pos, omega], None::<&[i64]>)?;
        axes.push(
            Tensor::f_stack(
                &[
                    out.f_cos()?,
                    out.f_sin()?.f_neg()?,
                    out.f_sin()?,
                    out.f_cos()?,
                ],
                -1,
            )?
            .f_reshape([1, h * w + text, dim / 2, 2, 2])?
            .f_to_device_(device, Kind::Float, false, false)?,
        );
    }
    Ok(Tensor::f_cat(&axes, -3)?
        .f_unsqueeze(1)?
        .f_transpose(1, 2)?
        .f_contiguous()?)
}
struct AttentionNorms {
    q: Tensor,
    k: Tensor,
    heads: i64,
    path: String,
}
impl AttentionNorms {
    fn load(p: &PreparedParameters, path: &str, heads: i64) -> Result<Self, Error> {
        Ok(Self {
            q: tensor(p, &format!("{path}.q_norm.weight"))?,
            k: tensor(p, &format!("{path}.k_norm.weight"))?,
            heads,
            path: path.into(),
        })
    }
    unsafe fn forward<E: MossExecution>(
        &self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        pe: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let shape = q.size();
        let mut q = q.f_view([shape[0], shape[1], self.heads, -1])?;
        let mut k = k.f_view([shape[0], shape[1], self.heads, -1])?;
        if e.training() {
            e.check_interrupt()?;
            q = moss_rope_split_half(&norm(&q, &self.q)?, pe)?;
            e.check_interrupt()?;
            k = moss_rope_split_half(&norm(&k, &self.k)?, pe)?;
        } else {
            let output = unsafe {
                rt.call_kitchen(
                    "rms_rope_split_half",
                    &vec![
                        ("q".into(), Value::tensor(q.shallow_clone())),
                        ("k".into(), Value::tensor(k.shallow_clone())),
                        ("freqs_cis".into(), Value::tensor(pe.shallow_clone())),
                        ("q_scale".into(), Value::tensor(cast(&self.q, &q)?)),
                        ("k_scale".into(), Value::tensor(cast(&self.k, &k)?)),
                        ("epsilon".into(), Value::Float(1e-6)),
                    ],
                )?
            };
            let Value::Tuple(mut out) = output else {
                return Err(Error::Native("Invalid rms_rope_split_half result".into()));
            };
            k = out.pop().unwrap().as_tensor()?;
            q = out.pop().unwrap().as_tensor()?;
        }
        e.attention(
            rt,
            &q.f_flatten(2, -1)?,
            &k.f_flatten(2, -1)?,
            v,
            self.heads,
            &self.path,
        )
    }
}
unsafe fn linear<E: MossExecution>(
    p: &Linear,
    rt: &Runtime,
    x: &Tensor,
    e: &mut E,
) -> Result<Tensor, Error> {
    e.check_interrupt()?;
    unsafe { p.forward(rt, x) }
}
struct TextBlock {
    norms: [Tensor; 2],
    attn: AttentionNorms,
    qkv: Linear,
    proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl TextBlock {
    fn load(p: &PreparedParameters, path: &str, heads: i64) -> Result<Self, Error> {
        let lin = |name| Linear::from_parameters(p, &format!("{path}.{name}"), true);
        Ok(Self {
            norms: [
                tensor(p, &format!("{path}.norm1.weight"))?,
                tensor(p, &format!("{path}.norm2.weight"))?,
            ],
            attn: AttentionNorms::load(p, path, heads)?,
            qkv: lin("qkv")?,
            proj: lin("proj")?,
            gate: lin("ffn_gate")?,
            up: lin("ffn_up")?,
            down: lin("ffn_down")?,
        })
    }
    unsafe fn forward<E: MossExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        pe: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        e.check_interrupt()?;
        let qkv = unsafe { linear(&self.qkv, rt, &norm(x, &self.norms[0])?, e)? }.f_chunk(3, -1)?;
        let attn = unsafe { self.attn.forward(rt, &qkv[0], &qkv[1], &qkv[2], pe, e)? };
        let x = x.f_add(&unsafe { linear(&self.proj, rt, &attn, e)? })?;
        // Release attention operands before the FFN, as the original does.
        drop((qkv, attn));
        e.check_interrupt()?;
        let y = norm(&x, &self.norms[1])?;
        let hidden = unsafe { linear(&self.up, rt, &y, e)? };
        let gate = unsafe { linear(&self.gate, rt, &y, e)? };
        Ok(x.f_add(&unsafe { linear(&self.down, rt, &moss_situ(&hidden, &gate)?, e)? })?)
    }
}
struct Block {
    norms: [Tensor; 2],
    attn: AttentionNorms,
    q: Linear,
    k: Option<Linear>,
    v: Option<Linear>,
    o: Linear,
    up: Linear,
    down: Linear,
}
impl Block {
    fn load(p: &PreparedParameters, path: &str, c: &MossConfig) -> Result<Self, Error> {
        let lin = |name| Linear::from_parameters(p, &format!("{path}.proj.{name}"), false);
        Ok(Self {
            norms: [
                tensor(p, &format!("{path}.norm1.weight"))?,
                tensor(p, &format!("{path}.norm2.weight"))?,
            ],
            attn: AttentionNorms::load(p, path, c.heads)?,
            q: lin(if c.fused_qkv { "qkv" } else { "q" })?,
            k: if c.fused_qkv { None } else { Some(lin("k")?) },
            v: if c.fused_qkv { None } else { Some(lin("v")?) },
            o: lin("o")?,
            up: lin("ffn_up_gate")?,
            down: lin("ffn_down")?,
        })
    }
    unsafe fn forward<E: MossExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        m: &[Tensor],
        pe: &Tensor,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let y = unsafe { mod_norm(rt, x, &self.norms[0], &m[1], &m[0], e)? };
        // A fused projection's output lives only in its chunks.
        let parts = {
            let q = unsafe { linear(&self.q, rt, &y, e)? };
            if let (Some(k), Some(v)) = (&self.k, &self.v) {
                vec![q, unsafe { linear(k, rt, &y, e)? }, unsafe {
                    linear(v, rt, &y, e)?
                }]
            } else {
                q.f_chunk(3, -1)?
            }
        };
        // Each intermediate is released once used, as the original does, which
        // lowers peak memory.
        drop(y);
        let attn = unsafe {
            self.attn
                .forward(rt, &parts[0], &parts[1], &parts[2], pe, e)?
        };
        drop(parts);
        let out = unsafe { linear(&self.o, rt, &attn, e)? };
        drop(attn);
        let x = residual(x, &out, &m[2], e.training())?;
        drop(out);
        let y = unsafe { mod_norm(rt, &x, &self.norms[1], &m[4], &m[3], e)? };
        let hidden = unsafe { linear(&self.up, rt, &y, e)? };
        drop(y);
        let training = e.training();
        let y = unsafe {
            self.down
                .input_act(rt, &hidden, "situ", None, 0., training, &mut || {
                    e.check_interrupt()
                })?
        };
        drop(hidden);
        residual(&x, &y, &m[5], training)
    }
}
pub struct MossDiT {
    pub config: MossConfig,
    video: Linear,
    text: Linear,
    text_blocks: Vec<TextBlock>,
    time: [Linear; 2],
    modulation: Linear,
    blocks: Vec<Block>,
    final_norm: Tensor,
    final_linear: Linear,
    head: Linear,
}
impl MossDiT {
    pub fn from_parameters(c: MossConfig, p: &PreparedParameters) -> Result<Self, Error> {
        let lin = |name, bias| Linear::from_parameters(p, name, bias);
        let blocks = (0..c.layers)
            .map(|i| Block::load(p, &format!("layers.{i}"), &c))
            .collect::<Result<_, _>>()?;
        let text_blocks = (0..2)
            .map(|i| TextBlock::load(p, &format!("text_embedder.blocks.{i}"), c.heads))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            blocks,
            text_blocks,
            video: lin("video_embedder", true)?,
            text: lin("text_embedder.proj", true)?,
            time: [
                lin("time_embedding.0", true)?,
                lin("time_embedding.2", true)?,
            ],
            modulation: lin("shared_block_modulation", true)?,
            head: lin("video_head_modulation", true)?,
            final_norm: tensor(p, "final_norm_video.weight")?,
            final_linear: lin("final_linear_video", false)?,
            config: c,
        })
    }
    /// # Safety
    /// Prepared parameters, selected backends and allocation services satisfy
    /// the original ComfyUI/Kitchen contracts on the current device/stream.
    pub unsafe fn forward<E: MossExecution>(
        &self,
        rt: &Runtime,
        inputs: &MossInputs<'_>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.wrapper(inputs, &mut |inputs, exec| unsafe {
            self.forward_inner(rt, inputs, exec)
        })
    }
    unsafe fn forward_inner<E: MossExecution>(
        &self,
        rt: &Runtime,
        a: &MossInputs<'_>,
        e: &mut E,
    ) -> Result<Tensor, Error> {
        let shape = a.latent.size();
        let (b, h, w) = (shape[0], shape[2], shape[3]);
        let pe = moss_rope(
            h,
            w,
            a.context.size()[1],
            self.config.dim / self.config.heads,
            a.latent.device(),
            true,
        )?;
        let t = timestep_embedding(a.timestep, 256, 10000., 1000.)?.f_to_kind(a.latent.kind())?;
        let vec = unsafe { linear(&self.time[0], rt, &t, e)? }.f_silu()?;
        let vec = unsafe { linear(&self.time[1], rt, &vec, e)? };
        let modulation = unsafe { linear(&self.modulation, rt, &vec, e)? }
            .f_unflatten(-1, [6, -1])?
            .f_unsqueeze(2)?
            .f_unbind(1)?;
        let head = unsafe { linear(&self.head, rt, &vec, e)? }
            .f_unflatten(-1, [2, -1])?
            .f_unsqueeze(2)?
            .f_unbind(1)?;
        let mut txt = unsafe { linear(&self.text, rt, a.context, e)? };
        e.lifecycle(MossLifecycle::TextQueue)?;
        for (i, block) in self.text_blocks.iter().enumerate() {
            e.lifecycle(MossLifecycle::TextBlock(i))?;
            txt = unsafe { block.forward(rt, &txt, &pe.f_slice(1, h * w, i64::MAX, 1)?, e)? };
        }
        e.lifecycle(MossLifecycle::TextQueueEnd)?;
        let image = unsafe {
            linear(
                &self.video,
                rt,
                &a.latent.f_flatten(2, -1)?.f_transpose(1, 2)?,
                e,
            )?
        };
        let mut x = Tensor::f_cat(&[image, txt], 1)?;
        e.lifecycle(MossLifecycle::Queue)?;
        e.lifecycle(MossLifecycle::GraphBegin)?;
        for (i, block) in self.blocks.iter().enumerate() {
            e.lifecycle(MossLifecycle::Block(i))?;
            let state = MossState {
                total_blocks: self.blocks.len(),
                block_index: i,
            };
            x = match e.replace_block(
                &state,
                &x,
                &vec,
                &modulation,
                &pe,
                &mut |x, m, pe, e| unsafe { block.forward(rt, x, m, pe, e) },
            )? {
                Some(x) => x,
                None => unsafe { block.forward(rt, &x, &modulation, &pe, e)? },
            };
        }
        e.lifecycle(MossLifecycle::QueueEnd)?;
        e.lifecycle(MossLifecycle::GraphEnd)?;
        let x = unsafe {
            mod_norm(
                rt,
                &x.f_slice(1, 0, h * w, 1)?,
                &self.final_norm,
                &head[1],
                &head[0],
                e,
            )?
        };
        Ok(unsafe { linear(&self.final_linear, rt, &x, e)? }
            .f_transpose(1, 2)?
            .f_reshape([b, self.config.latent_channels, h, w])?)
    }
}
