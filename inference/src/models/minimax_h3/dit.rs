//! Selected dense Ref2VA DiT. Source H3 geometry and operations remain separate
//! from Qwen Image's transformer; learned parameters use the common prepared
//! payloads and execution/offload callbacks. The selected checkpoint uses the
//! eight-coordinate AdaLN curve basis and one output head per stream.
use super::layout::{self, Keyframe, PackedLayout, ReferenceBlock, SegmentKind};
use crate::{
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value as V,
    models::qwen_image21::{Linear, TextEncoderExecution},
    native::Error,
};
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

#[derive(Deserialize)]
pub struct Config {
    pub hidden_size: i64,
    pub num_layers: usize,
    pub token_refiner_num_layers: usize,
    pub num_attention_heads: i64,
    pub attention_head_dim: i64,
    pub ffn_hidden_size: i64,
    pub latents_dim: i64,
    pub audio_latents_dim: i64,
    pub text_dim: i64,
    pub adaln_curve_grid: i64,
    pub time_embed_dim: i64,
    pub rope_inv_freq_len: i64,
}
impl Config {
    pub fn selected() -> Result<Self, Error> {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("../../../resources/h3-schema.json"))
                .map_err(|e| Error::Native(e.to_string()))?;
        serde_json::from_value(v["dit_config"].clone()).map_err(|e| Error::Native(e.to_string()))
    }
}
fn get(p: &PreparedParameters, n: &str) -> Result<Tensor, Error> {
    p.get(n)
        .ok_or_else(|| Error::BindingGap {
            backend: "minimax_h3_dit".into(),
            operation: format!("Missing {n}"),
        })?
        .as_tensor()
}
fn norm(x: &Tensor, w: &Tensor) -> Result<Tensor, Error> {
    let w = w.f_to_device_(x.device(), x.kind(), false, false)?;
    Ok(x.f_rms_norm(w.size().as_slice(), Some(&w), Some(1e-5))?)
}
fn linear(p: &PreparedParameters, n: &str, bias: bool) -> Result<Linear, Error> {
    Linear::from_parameters(p, n, bias)
}

pub fn time_shift(sigma: &Tensor, from: f64, to: f64) -> Result<Tensor, Error> {
    let base = sigma.f_div(&sigma.f_mul_scalar(1. - from)?.f_add_scalar(from)?)?;
    Ok(base
        .f_mul_scalar(to)?
        .f_div(&base.f_mul_scalar(to - 1.)?.f_add_scalar(1.)?)?)
}
pub fn interpolate_basis(table: &Tensor, t: &Tensor) -> Result<Tensor, Error> {
    let table = table.f_to_device(t.device())?;
    let pos = t.f_clamp(0., 1.)?.f_mul_scalar(table.size()[0] - 1)?;
    let i0 = pos
        .f_floor()?
        .f_to_kind(Kind::Int64)?
        .f_clamp_max(table.size()[0] - 2)?;
    let fraction = pos.f_sub(&i0)?.f_unsqueeze(1)?;
    Ok(table
        .f_index(&[Some(&i0)])?
        .f_lerp_tensor(&table.f_index(&[Some(&i0.f_add_scalar(1)?)])?, &fraction)?)
}
#[derive(Clone)]
pub enum Row {
    Scalar(i64),
    PerToken(V),
}
impl Row {
    fn values(&self, vecs: &Tensor, dtype: Kind) -> Result<Tensor, Error> {
        Ok(match self {
            Self::Scalar(i) => vecs.f_get(*i)?,
            Self::PerToken(t) => vecs.f_index(&[Some(&t.as_tensor()?)])?,
        }
        .f_to_kind(dtype)?)
    }
}
pub struct ModSegment {
    pub start: i64,
    pub stop: i64,
    pub row: Row,
}
fn scale_shift(
    h: &Tensor,
    shift: &Tensor,
    scale: &Tensor,
    segments: &[ModSegment],
) -> Result<(), Error> {
    for seg in segments {
        let mut part = h.f_slice(0, seg.start, seg.stop, 1)?;
        let _ = part.f_mul_(&seg.row.values(scale, h.kind())?.f_add_scalar(1.)?)?;
        let _ = part.f_add_(&seg.row.values(shift, h.kind())?)?;
    }
    Ok(())
}
fn gate(x: &Tensor, gate: &Tensor, other: &Tensor, segments: &[ModSegment]) -> Result<(), Error> {
    for seg in segments {
        let mut part = x.f_slice(0, seg.start, seg.stop, 1)?;
        let _ = part.f_addcmul_(
            &other.f_slice(0, seg.start, seg.stop, 1)?,
            &seg.row.values(gate, x.kind())?,
        )?;
    }
    Ok(())
}
struct Attention {
    qkv: Linear,
    out: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    heads: i64,
    dim: i64,
}
impl Attention {
    fn new(p: &PreparedParameters, n: &str, c: &Config) -> Result<Self, Error> {
        Ok(Self {
            qkv: linear(p, &format!("{n}.qkv_proj"), false)?,
            out: linear(p, &format!("{n}.out_proj"), false)?,
            q_norm: get(p, &format!("{n}.q_norm.weight"))?,
            k_norm: get(p, &format!("{n}.k_norm.weight"))?,
            heads: c.num_attention_heads,
            dim: c.attention_head_dim,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        rope: Option<&Tensor>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let qkv = unsafe { self.qkv.forward(rt, x)? }.f_split(self.heads * self.dim, -1)?;
        let s = x.size()[0];
        let v = qkv[2].f_view([s, self.heads, self.dim])?;
        let (q, k) = if let Some(rope) = rope {
            let q = qkv[0].f_view([1, s, self.heads, self.dim])?;
            let k = qkv[1].f_view([1, s, self.heads, self.dim])?;
            let _ = unsafe {
                rt.call_kitchen(
                    "rms_rope_split_half_",
                    &vec![
                        ("q".into(), V::tensor(q.shallow_clone())),
                        ("k".into(), V::tensor(k.shallow_clone())),
                        ("freqs_cis".into(), V::tensor(rope.shallow_clone())),
                        (
                            "q_scale".into(),
                            V::tensor(self.q_norm.f_to_device(x.device())?),
                        ),
                        (
                            "k_scale".into(),
                            V::tensor(self.k_norm.f_to_device(x.device())?),
                        ),
                        ("epsilon".into(), V::Float(1e-5)),
                        ("rot_dim".into(), V::Int(rope.size()[rope.dim() - 3] * 2)),
                    ],
                )?
            };
            (q.f_get(0)?, k.f_get(0)?)
        } else {
            (
                norm(&qkv[0].f_view([s, self.heads, self.dim])?, &self.q_norm)?,
                norm(&qkv[1].f_view([s, self.heads, self.dim])?, &self.k_norm)?,
            )
        };
        let out = exec.attention(
            rt,
            &q.f_transpose(0, 1)?.f_unsqueeze(0)?,
            &k.f_transpose(0, 1)?.f_unsqueeze(0)?,
            &v.f_transpose(0, 1)?.f_unsqueeze(0)?,
            None,
            self.heads,
            false,
        )?;
        exec.check_interrupt()?;
        unsafe { self.out.forward(rt, &out.f_squeeze_dim(0)?) }
    }
}
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}
impl Mlp {
    fn new(p: &PreparedParameters, n: &str) -> Result<Self, Error> {
        Ok(Self {
            fc1: linear(p, &format!("{n}.fc1"), false)?,
            fc2: linear(p, &format!("{n}.fc2"), false)?,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let h = unsafe { self.fc1.forward(rt, x)? };
        unsafe {
            self.fc2
                .swiglu(rt, &h, false, &mut || exec.check_interrupt())
        }
    }
}
struct Refiner {
    norm1: Tensor,
    norm2: Tensor,
    attn: Attention,
    mlp: Mlp,
}
impl Refiner {
    fn new(p: &PreparedParameters, n: &str, c: &Config) -> Result<Self, Error> {
        Ok(Self {
            norm1: get(p, &format!("{n}.norm1.weight"))?,
            norm2: get(p, &format!("{n}.norm2.weight"))?,
            attn: Attention::new(p, &format!("{n}.attn"), c)?,
            mlp: Mlp::new(p, &format!("{n}.mlp"))?,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let x = unsafe { self.attn.forward(rt, &norm(x, &self.norm1)?, None, exec)? }.f_add_(x)?;
        Ok(unsafe { self.mlp.forward(rt, &norm(&x, &self.norm2)?, exec)? }.f_add_(&x)?)
    }
}
struct Block {
    norm1: Tensor,
    norm2: Tensor,
    attn: Attention,
    mlp: Mlp,
    adaln: Linear,
}
impl Block {
    fn new(p: &PreparedParameters, n: &str, c: &Config) -> Result<Self, Error> {
        Ok(Self {
            norm1: get(p, &format!("{n}.norm1.weight"))?,
            norm2: get(p, &format!("{n}.norm2.weight"))?,
            attn: Attention::new(p, &format!("{n}.attn"), c)?,
            mlp: Mlp::new(p, &format!("{n}.mlp"))?,
            adaln: linear(p, &format!("{n}.adaln_proj.linear"), true)?,
        })
    }
    unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        t: &Tensor,
        segments: &[ModSegment],
        rope: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        exec.check_interrupt()?;
        let projected = unsafe { self.adaln.forward(rt, t)? };
        let hidden = x.size()[1];
        let mods = projected
            .f_view([projected.size()[0] * 3, 6 * hidden])?
            .f_chunk(6, -1)?;
        let h = norm(x, &self.norm1)?;
        scale_shift(&h, &mods[0], &mods[1], segments)?;
        gate(
            x,
            &mods[2],
            &unsafe { self.attn.forward(rt, &h, Some(rope), exec)? },
            segments,
        )?;
        let h = norm(x, &self.norm2)?;
        scale_shift(&h, &mods[3], &mods[4], segments)?;
        gate(
            x,
            &mods[5],
            &unsafe { self.mlp.forward(rt, &h, exec)? },
            segments,
        )?;
        Ok(x.shallow_clone())
    }
}
#[derive(Clone, Copy)]
pub struct Payload<'a> {
    pub layout: Option<&'a PackedLayout>,
    pub keyframes: &'a [Keyframe],
    pub refs: &'a [ReferenceBlock],
    pub video_conditions: &'a [Tensor],
    pub audio_conditions: &'a [Tensor],
    pub text_tags: Option<&'a Tensor>,
    pub visual_aug: f64,
    pub audio_aug: f64,
    pub seed: u64,
    pub audio_scale: f64,
}
impl Default for Payload<'_> {
    fn default() -> Self {
        Self {
            layout: None,
            keyframes: &[],
            refs: &[],
            video_conditions: &[],
            audio_conditions: &[],
            text_tags: None,
            visual_aug: 0.999,
            audio_aug: 1.,
            seed: 0,
            audio_scale: 1.,
        }
    }
}
pub struct Forward<'a> {
    pub video: &'a Tensor,
    pub audio: &'a Tensor,
    pub timestep: &'a Tensor,
    pub context: &'a Tensor,
    pub payload: Payload<'a>,
    pub denoise_mask: Option<&'a Tensor>,
    pub audio_denoise_mask: Option<&'a Tensor>,
    pub shifts: [f64; 2],
}
pub struct Denoiser {
    pub config: Config,
    video: Linear,
    audio: Linear,
    condition: Linear,
    refiners: Vec<Refiner>,
    refiner_norm: Tensor,
    blocks: Vec<Block>,
    table: Tensor,
    inv_freq: Tensor,
    final_norm: Tensor,
    final_adaln: Linear,
    video_out: Linear,
    audio_out: Linear,
}
impl Denoiser {
    pub fn from_parameters(config: Config, p: &PreparedParameters) -> Result<Self, Error> {
        let refiners = (0..config.token_refiner_num_layers)
            .map(|i| Refiner::new(p, &format!("token_refiner.blocks.{i}"), &config))
            .collect::<Result<_, _>>()?;
        let blocks = (0..config.num_layers)
            .map(|i| Block::new(p, &format!("blocks.{i}"), &config))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            video: linear(p, "video_patch_proj", true)?,
            audio: linear(p, "audio_patch_proj", true)?,
            condition: linear(p, "condition_proj", true)?,
            refiners,
            refiner_norm: get(p, "token_refiner.final_norm.weight")?,
            blocks,
            table: get(p, "adaln_t_table")?,
            inv_freq: get(p, "rope.inv_freq")?,
            final_norm: get(p, "final_layer.norm.weight")?,
            final_adaln: linear(p, "final_layer.adaln_proj.linear", true)?,
            video_out: linear(p, "final_layer.video_out", true)?,
            audio_out: linear(p, "final_layer.audio_out", true)?,
            config,
        })
    }
    /// # Safety
    /// Parameters, callbacks and native kernels satisfy the selected H3 ABI.
    pub unsafe fn preprocess_text<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        x: &Tensor,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        if x.size()[x.dim() - 1] == self.config.hidden_size {
            return Ok(x.shallow_clone());
        }
        let mut x = unsafe { self.condition.forward(rt, &x.f_get(0)?)? };
        for block in &self.refiners {
            x = unsafe { block.forward(rt, &x, exec)? };
        }
        Ok(norm(&x, &self.refiner_norm)?.f_unsqueeze(0)?)
    }
    fn rope(&self, layout: &PackedLayout, device: Device, kind: Kind) -> Result<Tensor, Error> {
        let pos = layout
            .position_ids
            .f_to_kind(Kind::Float)?
            .f_to_device(device)?;
        let freq = pos
            .f_unsqueeze(-1)?
            .f_mul(&self.inv_freq.f_to_device(device)?.f_view([1, 1, -1])?)?;
        let half = Tensor::f_cat(&freq.f_unbind(1)?, -1)?;
        let angles = Tensor::f_cat(&[half.shallow_clone(), half], -1)?;
        let half = angles.size()[1] / 2;
        let ang = angles.f_slice(1, 0, half, 1)?;
        let c = ang.f_cos()?;
        let s = ang.f_sin()?;
        Ok(Tensor::f_stack(&[c.shallow_clone(), s.f_neg()?, s, c], -1)?
            .f_reshape([1, angles.size()[0], 1, half, 2, 2])?
            .f_to_kind(kind)?)
    }
    fn condition_rows(
        &self,
        rt: &Runtime,
        items: &[Tensor],
        video: bool,
        aug: f64,
        seed: u64,
        device: Device,
    ) -> Result<Option<Tensor>, Error> {
        let mut rows = vec![];
        for z in items {
            let z = z.f_to_kind(Kind::Float)?;
            let mut r = if video {
                layout::patchify_video(&z, [1, 2, 2])?
            } else {
                layout::pack_audio(&z)?
            };
            if aug < 1. {
                let mut generator = rt.native.generator(Device::Cpu, seed)?;
                let noise = generator.randn(&r.size(), Kind::Float)?;
                r = r
                    .f_mul_scalar(aug)?
                    .f_add(&noise.f_to_device(r.device())?.f_mul_scalar(1. - aug)?)?;
            }
            rows.push(r.f_to_device(device)?);
        }
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(Tensor::f_cat(&rows, 0)?))
        }
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Preserve the original H3 stream, layout and payload boundary"
    )]
    unsafe fn embed<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        video: &Tensor,
        audio: &Tensor,
        context: &Tensor,
        layout: &PackedLayout,
        payload: &Payload<'_>,
        exec: &mut E,
    ) -> Result<Tensor, Error> {
        let device = video.device();
        let kind = context.kind();
        let vu = layout.img_update.f_to_device(device)?;
        let au = layout.audio_update.f_to_device(device)?;
        let vr = layout::patchify_video(&video.f_to_kind(Kind::Float)?, [1, 2, 2])?;
        let ar = layout::pack_audio(&audio.f_to_kind(Kind::Float)?)?;
        let crv = self.condition_rows(
            rt,
            payload.video_conditions,
            true,
            payload.visual_aug,
            payload.seed,
            device,
        )?;
        let cra = self.condition_rows(
            rt,
            payload.audio_conditions,
            false,
            payload.audio_aug,
            payload.seed.wrapping_add(1),
            device,
        )?;
        let all_v = if let Some(c) = crv {
            let mut all = Tensor::f_empty([vu.size()[0], vr.size()[1]], (Kind::Float, device))?;
            let _ = all.f_index_put_(&[Some(&vu.f_logical_not()?)], &c, false)?;
            let _ = all.f_index_put_(&[Some(&vu)], &vr, false)?;
            all
        } else {
            vr
        };
        let all_a = if let Some(c) = cra {
            let mut all = Tensor::f_empty([au.size()[0], ar.size()[1]], (Kind::Float, device))?;
            let _ = all.f_index_put_(&[Some(&au.f_logical_not()?)], &c, false)?;
            let _ = all.f_index_put_(&[Some(&au)], &ar, false)?;
            all
        } else {
            ar
        };
        let v = unsafe { self.video.forward(rt, &all_v)? }.f_to_kind(kind)?;
        let a = unsafe { self.audio.forward(rt, &all_a)? }.f_to_kind(kind)?;
        let text = unsafe { self.preprocess_text(rt, context, exec)? }.f_get(0)?;
        let h = Tensor::f_empty([layout.seq_len, self.config.hidden_size], (kind, device))?;
        let (mut vi, mut ai) = (0, 0);
        for seg in &layout.segments {
            let n = seg.stop - seg.start;
            let values = match seg.kind {
                SegmentKind::Text => text.shallow_clone(),
                SegmentKind::Cond | SegmentKind::RefImg | SegmentKind::Video => {
                    let r = v.f_slice(0, vi, vi + n, 1)?;
                    vi += n;
                    r
                }
                _ => {
                    let r = a.f_slice(0, ai, ai + n, 1)?;
                    ai += n;
                    r
                }
            };
            h.f_slice(0, seg.start, seg.stop, 1)?.f_copy_(&values)?;
        }
        Ok(h)
    }
}

fn doubles(t: &Tensor) -> Result<Vec<f64>, Error> {
    let t = t
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Double)?
        .f_reshape([-1])?;
    let mut data = vec![0.; t.numel()];
    let n = data.len();
    t.f_copy_data(&mut data, n)?;
    Ok(data)
}
fn longs(t: &Tensor) -> Result<Vec<i64>, Error> {
    let t = t
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Int64)?
        .f_reshape([-1])?;
    let mut data = vec![0; t.numel()];
    let n = data.len();
    t.f_copy_data(&mut data, n)?;
    Ok(data)
}
fn time_index(values: &[f64], t: f64) -> Result<i64, Error> {
    values
        .iter()
        .position(|v| *v == t)
        .map(|i| i as i64)
        .ok_or_else(|| Error::BindingGap {
            backend: "minimax_h3_dit".into(),
            operation: "Missing original timestep row".into(),
        })
}
fn row_indices(rows: &Tensor, tag: i64, times: &[f64]) -> Result<Tensor, Error> {
    let (levels, _, _) = rows.f_internal_unique2(true, false, false)?;
    let base = doubles(&levels)?
        .into_iter()
        .map(|v| Ok(time_index(times, v)? * 3 + tag))
        .collect::<Result<Vec<_>, Error>>()?;
    let base = Tensor::f_from_slice(&base)?.f_to_device(rows.device())?;
    let indices = rows.f_searchsorted(&levels, false, false, "left", None::<&Tensor>)?;
    Ok(base.f_index(&[Some(&indices)])?)
}
struct TimePlan {
    times: Vec<f64>,
    segments: Vec<ModSegment>,
    video: ModSegment,
    audio: ModSegment,
}
fn time_plan(
    layout: &PackedLayout,
    sigma: &Tensor,
    shifts: [f64; 2],
    payload: &Payload<'_>,
    video_mask: Option<&Tensor>,
    audio_mask: Option<&Tensor>,
) -> Result<TimePlan, Error> {
    let tv = sigma.f_rsub_scalar(1.)?.double_value(&[]);
    let ta = time_shift(sigma, shifts[0], shifts[1])?
        .f_rsub_scalar(1.)?
        .double_value(&[]);
    let mut video_time = tv;
    let mut audio_time = ta;
    let mut video_rows = None;
    let mut audio_rows = None;
    if let Some(mask) = video_mask
        && let Some(m) = layout::mask_row_values(
            &mask.f_get(0)?.f_get(0)?.f_to_kind(Kind::Float)?,
            layout.signature[1],
            layout.signature[2],
            layout.signature[3],
        )?
    {
        let rows = m
            .f_mul(&sigma.f_to_device(m.device())?)?
            .f_rsub_scalar(1.)?
            .f_clamp_max(tv.max(0.999))?;
        if rows.f_internal_unique2(true, false, false)?.0.numel() == 1 {
            video_time = rows.f_get(0)?.double_value(&[]);
        } else {
            video_rows = Some(rows);
        }
    }
    if let Some(mask) = audio_mask {
        let m = mask
            .f_get(0)?
            .f_get(0)?
            .f_to_kind(Kind::Float)?
            .f_reshape([-1])?;
        if m.f_ge(1. - 1e-3)?.f_all()?.int64_value(&[]) == 0 {
            let rows = m
                .f_mul_scalar(1. - ta)?
                .f_rsub_scalar(1.)?
                .f_clamp_max(ta.max(1.))?;
            if rows.f_internal_unique2(true, false, false)?.0.numel() == 1 {
                audio_time = rows.f_get(0)?.double_value(&[]);
            } else {
                audio_rows = Some(rows);
            }
        }
    }
    let segment_time = |kind| match kind {
        SegmentKind::Text => tv,
        SegmentKind::Video => video_time,
        SegmentKind::Audio => audio_time,
        SegmentKind::Cond | SegmentKind::RefImg => tv.max(payload.visual_aug),
        SegmentKind::CondAudio | SegmentKind::RefAudio => ta.max(payload.audio_aug),
    };
    let mut times = vec![tv, ta];
    for seg in &layout.segments {
        times.push(segment_time(seg.kind));
    }
    for rows in [&video_rows, &audio_rows].into_iter().flatten() {
        times.extend(doubles(&rows.f_internal_unique2(true, false, false)?.0)?);
    }
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| *a == *b);
    let mut segments = vec![];
    for seg in &layout.segments {
        let base = time_index(&times, segment_time(seg.kind))? * 3;
        if seg.kind == SegmentKind::Text
            && let Some(tags) = payload.text_tags
        {
            let tags = longs(tags)?;
            let mut start = 0;
            for i in 1..=(seg.stop - seg.start) as usize {
                if i == (seg.stop - seg.start) as usize || tags[i] != tags[start] {
                    segments.push(ModSegment {
                        start: seg.start + start as i64,
                        stop: seg.start + i as i64,
                        row: Row::Scalar(base + tags[start]),
                    });
                    start = i;
                }
            }
        } else {
            let row = match seg.kind {
                SegmentKind::Video if video_rows.is_some() => Row::PerToken(V::tensor(
                    row_indices(video_rows.as_ref().unwrap(), 0, &times)?,
                )),
                SegmentKind::Audio if audio_rows.is_some() => Row::PerToken(V::tensor(
                    row_indices(audio_rows.as_ref().unwrap(), 2, &times)?,
                )),
                SegmentKind::Text => Row::Scalar(base + 1),
                SegmentKind::Audio | SegmentKind::CondAudio | SegmentKind::RefAudio => {
                    Row::Scalar(base + 2)
                }
                _ => Row::Scalar(base),
            };
            segments.push(ModSegment {
                start: seg.start,
                stop: seg.stop,
                row,
            });
        }
    }
    let final_segment = |kind, rows: &Option<Tensor>| -> Result<ModSegment, Error> {
        let seg = layout.segments.iter().find(|s| s.kind == kind).unwrap();
        let row = if let Some(rows) = rows {
            Row::PerToken(V::tensor(
                row_indices(rows, 0, &times)?.f_floor_divide_scalar(3)?,
            ))
        } else {
            Row::Scalar(time_index(&times, segment_time(kind))?)
        };
        Ok(ModSegment {
            start: seg.start,
            stop: seg.stop,
            row,
        })
    };
    let video = final_segment(SegmentKind::Video, &video_rows)?;
    let audio = final_segment(SegmentKind::Audio, &audio_rows)?;
    Ok(TimePlan {
        times,
        segments,
        video,
        audio,
    })
}
impl Denoiser {
    /// Selected dense `_forward`, including full row-wise denoise-mask and
    /// reference conditioning. No model-sized weights are allocated here.
    /// # Safety
    /// Source H3 parameter, kernel, memory-owner and callback contracts hold.
    pub unsafe fn forward<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        input: Forward<'_>,
        exec: &mut E,
    ) -> Result<[Tensor; 2], Error> {
        let sigma = input
            .timestep
            .f_flatten(0, -1)?
            .f_get(0)?
            .f_div_scalar(1000.)?
            .f_to_kind(Kind::Float)?
            .f_clamp_min(1e-6)?;
        let (mut out, carry, audio_src) = if input.payload.audio_scale != 1. {
            let sigma_a = time_shift(&sigma, input.shifts[0], input.shifts[1])?;
            let carry = sigma_a.f_div(&sigma)?.f_to_kind(input.audio.kind())?;
            let carried = input.audio.f_mul(&carry)?;
            let out = unsafe { self.core(rt, input.video, &carried, &input, &sigma, exec)? };
            (out, Some((carry, sigma_a)), input.audio)
        } else {
            (
                unsafe { self.core(rt, input.video, input.audio, &input, &sigma, exec)? },
                None,
                input.audio,
            )
        };
        if let Some(mask) = input.denoise_mask {
            out[0] = out[0].f_mul(mask)?;
        }
        if let Some(mask) = input.audio_denoise_mask {
            out[1] = out[1].f_mul(mask)?;
        }
        if let Some((carry, sigma_a)) = carry {
            let scale = input.payload.audio_scale;
            out[1] = audio_src.f_mul(&carry)?.f_mul_scalar(1. - scale)?.f_add(
                &sigma_a
                    .f_mul_scalar(scale - 1.)?
                    .f_add_scalar(1.)?
                    .f_to_kind(out[1].kind())?
                    .f_mul(&out[1])?,
            )?;
        }
        Ok(out)
    }
    unsafe fn core<E: TextEncoderExecution>(
        &self,
        rt: &Runtime,
        video: &Tensor,
        audio: &Tensor,
        input: &Forward<'_>,
        sigma: &Tensor,
        exec: &mut E,
    ) -> Result<[Tensor; 2], Error> {
        let original = video.size();
        let video = video.f_pad(
            [
                0,
                (2 - original[4] % 2) % 2,
                0,
                (2 - original[3] % 2) % 2,
                0,
                0,
            ],
            "circular",
            None,
        )?;
        if video.size()[0] != 1 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "MiniMax H3 supports batch size 1".into(),
            });
        }
        let shape = video.size();
        let signature = [
            input.context.size()[1],
            shape[2],
            shape[3],
            shape[4],
            audio.size()[3],
        ];
        let owned;
        let layout = match input.payload.layout {
            Some(layout) if layout.signature == signature => layout,
            _ => {
                owned = PackedLayout::new(signature, input.payload.keyframes, input.payload.refs)?;
                &owned
            }
        };
        let plan = time_plan(
            layout,
            sigma,
            input.shifts,
            &input.payload,
            input.denoise_mask,
            input.audio_denoise_mask,
        )?;
        let times = Tensor::f_from_slice(&plan.times)?
            .f_to_device(video.device())?
            .f_to_kind(Kind::Float)?;
        let t = interpolate_basis(&self.table, &times)?;
        let rope = self.rope(layout, video.device(), input.context.kind())?;
        let mut h = unsafe {
            self.embed(
                rt,
                &video,
                audio,
                input.context,
                layout,
                &input.payload,
                exec,
            )?
        };
        exec.observe("h3-packed", &h)?;
        exec.observe("h3-time-basis", &t)?;
        exec.observe("h3-rope", &rope)?;
        exec.queue_begin(self.blocks.len(), video.device())?;
        for (i, block) in self.blocks.iter().enumerate() {
            let mut core = |exec: &mut E| -> Result<(), Error> {
                h = unsafe { block.forward(rt, &h, &t, &plan.segments, &rope, exec)? };
                Ok(())
            };
            exec.block(i, &mut core)?;
            exec.observe(&format!("h3-block-{i}"), &h)?;
        }
        exec.queue_end(video.device())?;
        let projected = unsafe { self.final_adaln.forward(rt, &t)? };
        let mods = projected
            .f_view([projected.size()[0], 2 * self.config.hidden_size])?
            .f_chunk(2, -1)?;
        let modulate = |seg: &ModSegment| -> Result<Tensor, Error> {
            let normed = norm(&h.f_slice(0, seg.start, seg.stop, 1)?, &self.final_norm)?;
            Ok(normed
                .f_mul(&seg.row.values(&mods[1], mods[1].kind())?.f_add_scalar(1.)?)?
                .f_add(&seg.row.values(&mods[0], mods[0].kind())?)?
                .f_to_kind(Kind::Float)?)
        };
        let v = unsafe { self.video_out.forward(rt, &modulate(&plan.video)?)? };
        let a = unsafe { self.audio_out.forward(rt, &modulate(&plan.audio)?)? };
        let v = layout::unpatchify_video(
            &v,
            [shape[2], shape[3] / 2, shape[4] / 2],
            self.config.latents_dim,
            [1, 2, 2],
        )?
        .f_slice(2, 0, original[2], 1)?
        .f_slice(3, 0, original[3], 1)?
        .f_slice(4, 0, original[4], 1)?;
        let a = layout::unpack_audio(&a, 2)?;
        Ok([
            v.f_to_kind(video.kind())?.f_neg()?,
            a.f_to_kind(audio.kind())?.f_neg()?,
        ])
    }
}
