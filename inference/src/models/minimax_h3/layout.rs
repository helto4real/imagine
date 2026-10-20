//! Locked MiniMax H3 packed token layout, latent row transforms and mask pooling.
//! The typed interface describes the valid tensor geometry used by H3; it does
//! not implement arbitrary Python argument binding.
use crate::native::Error;
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

const FRAME_RESCALE: f64 = 5.0 / 3.0;
const FRAME_PER_TOKEN: [f64; 5] = [1., 4., 4., 4., 4.];

pub fn patchify_video(latent: &Tensor, patch: [i64; 3]) -> Result<Tensor, Error> {
    let shape = latent.size();
    let [b, c, t, h, w] = five(&shape)?;
    let [pt, ph, pw] = patch;
    Ok(latent
        .f_reshape([b, c, t / pt, pt, h / ph, ph, w / pw, pw])?
        .f_permute([0, 2, 4, 6, 1, 3, 5, 7])?
        .f_reshape([b * (t / pt) * (h / ph) * (w / pw), c * pt * ph * pw])?)
}

pub fn unpatchify_video(
    rows: &Tensor,
    geometry: [i64; 3],
    channels: i64,
    patch: [i64; 3],
) -> Result<Tensor, Error> {
    let [t, h, w] = geometry;
    let [pt, ph, pw] = patch;
    Ok(rows
        .f_reshape([-1, t, h, w, channels, pt, ph, pw])?
        .f_permute([0, 4, 1, 5, 2, 6, 3, 7])?
        .f_reshape([-1, channels, t * pt, h * ph, w * pw])?)
}

pub fn pack_audio(latent: &Tensor) -> Result<Tensor, Error> {
    let shape = latent.size();
    let [_, c, channels, t]: [i64; 4] = shape.try_into().map_err(|_| geometry_error())?;
    Ok(latent
        .f_get(0)?
        .f_permute([1, 2, 0])?
        .f_reshape([channels * t, c])?)
}

pub fn unpack_audio(rows: &Tensor, channels: i64) -> Result<Tensor, Error> {
    let shape = rows.size();
    Ok(rows
        .f_reshape([channels, shape[0] / channels, shape[1]])?
        .f_permute([2, 0, 1])?
        .f_unsqueeze(0)?)
}

fn geometry_error() -> Error {
    Error::BindingGap {
        backend: "minimax_h3".into(),
        operation: "typed H3 latent tensor rank".into(),
    }
}

fn five(shape: &[i64]) -> Result<[i64; 5], Error> {
    shape.try_into().map_err(|_| geometry_error())
}

pub fn mask_row_values(
    mask: &Tensor,
    latent_t: i64,
    latent_h: i64,
    latent_w: i64,
) -> Result<Option<Tensor>, Error> {
    let shape = mask.size();
    let m = mask.f_replication_pad2d([
        0,
        latent_w - shape[shape.len() - 1],
        0,
        latent_h - shape[shape.len() - 2],
    ])?;
    let values = m
        .f_reshape([latent_t, latent_h / 2, 2, latent_w / 2, 2])?
        .f_amax([2, 4], false)?
        .f_reshape([-1])?;
    Ok(if values.f_ge(1.0 - 1e-3)?.f_all()?.int64_value(&[]) != 0 {
        None
    } else {
        Some(values)
    })
}

fn axis(dim: i64, sqrt_area: f64) -> Result<Tensor, Error> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / 2;
    Ok(Tensor::f_arange(n, (Kind::Double, Device::Cpu))?
        .f_mul_scalar(ratio / n as f64)?
        .f_add_scalar((1.0 - ratio) / 2.0)?
        .f_mul_scalar(32.0)?)
}

fn frame_grid(h: i64, w: i64) -> Result<(Tensor, Tensor), Error> {
    let area = ((h * w) as f64).sqrt();
    let axes = [axis(h, area)?, axis(w, area)?];
    let grid = Tensor::f_meshgrid_indexing(&axes, "ij")?;
    Ok((
        Tensor::f_stack(&[grid[0].f_reshape([-1])?, grid[1].f_reshape([-1])?], -1)?,
        axis(w, area)?,
    ))
}

fn video_spans(n: i64) -> Vec<f64> {
    (0..n)
        .map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k as usize % 5])
        .collect()
}

fn video_span_sum(n: i64) -> f64 {
    // CPython v3.12.13 Python/bltinmodule.c builtin_sum_impl float branch:
    // preserve its compensated summation for the original sum(_video_t_spans).
    let mut total: f64 = 0.;
    let mut correction = 0.;
    for x in video_spans(n) {
        let next = total + x;
        correction += if total.abs() >= x.abs() {
            (total - next) + x
        } else {
            (x - next) + total
        };
        total = next;
    }
    if correction != 0. && correction.is_finite() {
        total += correction;
    }
    total
}

fn video_t_grid(n: i64, origin: f64) -> Result<Tensor, Error> {
    let spans = Tensor::f_from_slice(&video_spans(n))?;
    Ok(Tensor::f_cat(
        &[
            Tensor::f_zeros([1], (Kind::Double, Device::Cpu))?,
            spans.f_slice(0, 0, -1, 1)?.f_cumsum(0, None)?,
        ],
        0,
    )?
    .f_add_scalar(origin)?)
}

fn audio_grid(cursor: f64, t: i64, low: f64, high: f64) -> Result<Tensor, Error> {
    let g = Tensor::f_zeros([t * 2, 3], (Kind::Double, Device::Cpu))?;
    g.f_select(1, 0)?.f_copy_(
        &Tensor::f_arange(t, (Kind::Double, Device::Cpu))?
            .f_add_scalar(cursor)?
            .f_repeat([2])?,
    )?;
    let _ = g.f_narrow(0, 0, t)?.f_select(1, 2)?.f_fill_(low)?;
    let _ = g.f_narrow(0, t, t)?.f_select(1, 2)?.f_fill_(high)?;
    Ok(g)
}

fn video_grid(vt: i64, frame: &Tensor, cursor: f64) -> Result<Tensor, Error> {
    let g = Tensor::f_empty([vt, frame.size()[0], 3], (Kind::Double, Device::Cpu))?;
    g.f_select(2, 0)?
        .f_copy_(&video_t_grid(vt, cursor)?.f_unsqueeze(1)?)?;
    g.f_narrow(2, 1, 2)?.f_copy_(&frame.f_unsqueeze(0)?)?;
    Ok(g.f_reshape([-1, 3])?)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    Image,
    Audio,
    Video,
    VideoAudio,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceBlock {
    pub kind: ReferenceKind,
    #[serde(default)]
    pub latent_t: i64,
    #[serde(default)]
    pub latent_h: i64,
    #[serde(default)]
    pub latent_w: i64,
    #[serde(default)]
    pub ref_audio_t: i64,
}

impl ReferenceBlock {
    fn span(&self) -> f64 {
        match self.kind {
            ReferenceKind::Image => 1.,
            ReferenceKind::Audio => self.ref_audio_t as f64,
            ReferenceKind::Video | ReferenceKind::VideoAudio => {
                (self.ref_audio_t as f64).max(video_span_sum(self.latent_t))
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Keyframe {
    pub resolved_frame_index: i64,
    pub video_t: Option<i64>,
    pub audio_t: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    Text,
    Cond,
    CondAudio,
    RefImg,
    RefAudio,
    Audio,
    Video,
}

impl SegmentKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Cond => "cond",
            Self::CondAudio => "cond_audio",
            Self::RefImg => "ref_img",
            Self::RefAudio => "ref_audio",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

pub struct Segment {
    pub start: i64,
    pub stop: i64,
    pub kind: SegmentKind,
}

pub struct PackedLayout {
    pub seq_len: i64,
    pub signature: [i64; 5],
    pub position_ids: Tensor,
    pub img_pos: Tensor,
    pub img_update: Tensor,
    pub audio_pos: Tensor,
    pub audio_update: Tensor,
    pub segments: Vec<Segment>,
}

struct Builder {
    row: i64,
    positions: Vec<Tensor>,
    img_pos: Vec<Tensor>,
    img_update: Vec<Tensor>,
    audio_pos: Vec<Tensor>,
    audio_update: Vec<Tensor>,
    segments: Vec<Segment>,
}

impl Builder {
    fn append(&mut self, kind: SegmentKind, grid: Tensor, update: bool) -> Result<(), Error> {
        let count = grid.size()[0];
        self.segments.push(Segment {
            start: self.row,
            stop: self.row + count,
            kind,
        });
        self.positions.push(grid);
        if kind != SegmentKind::Text {
            let positions =
                Tensor::f_arange_start(self.row, self.row + count, (Kind::Int64, Device::Cpu))?;
            let updates = Tensor::f_full([count], i64::from(update), (Kind::Bool, Device::Cpu))?;
            if matches!(
                kind,
                SegmentKind::Cond | SegmentKind::RefImg | SegmentKind::Video
            ) {
                self.img_pos.push(positions);
                self.img_update.push(updates);
            } else {
                self.audio_pos.push(positions);
                self.audio_update.push(updates);
            }
        }
        self.row += count;
        Ok(())
    }
}

impl PackedLayout {
    pub fn new(
        signature: [i64; 5],
        keyframes: &[Keyframe],
        refs: &[ReferenceBlock],
    ) -> Result<Self, Error> {
        let [text_len, latent_t, latent_h, latent_w, audio_t] = signature;
        let (frame, w_grid) = frame_grid(latent_h, latent_w)?;
        let low = w_grid.double_value(&[0]);
        let high = w_grid.double_value(&[w_grid.size()[0] - 1]);
        let mut b = Builder {
            row: 0,
            positions: vec![],
            img_pos: vec![],
            img_update: vec![],
            audio_pos: vec![],
            audio_update: vec![],
            segments: vec![],
        };
        let g = Tensor::f_zeros([text_len, 3], (Kind::Double, Device::Cpu))?;
        g.f_select(1, 0)?
            .f_copy_(&Tensor::f_arange(text_len, (Kind::Double, Device::Cpu))?)?;
        b.append(SegmentKind::Text, g, false)?;
        let mut cursor = text_len as f64;
        for block in refs {
            cursor += block.span();
        }
        for keyframe in keyframes {
            let cond_t = cursor + FRAME_RESCALE * keyframe.resolved_frame_index as f64;
            if let Some(vt) = keyframe.video_t {
                b.append(SegmentKind::Cond, video_grid(vt, &frame, cond_t)?, false)?;
            }
            if let Some(rt) = keyframe.audio_t {
                b.append(
                    SegmentKind::CondAudio,
                    audio_grid(cond_t, rt, low, high)?,
                    false,
                )?;
            }
        }
        if !refs.is_empty() {
            cursor = text_len as f64;
            for block in refs {
                match block.kind {
                    ReferenceKind::Image => {
                        let (r_frame, _) = frame_grid(block.latent_h, block.latent_w)?;
                        let g =
                            Tensor::f_empty([r_frame.size()[0], 3], (Kind::Double, Device::Cpu))?;
                        let _ = g.f_select(1, 0)?.f_fill_(cursor)?;
                        g.f_narrow(1, 1, 2)?.f_copy_(&r_frame)?;
                        b.append(SegmentKind::RefImg, g, false)?;
                        cursor += 1.;
                    }
                    ReferenceKind::Audio => {
                        if block.ref_audio_t > 0 {
                            b.append(
                                SegmentKind::RefAudio,
                                audio_grid(cursor, block.ref_audio_t, low, high)?,
                                false,
                            )?;
                        }
                        cursor += block.ref_audio_t as f64;
                    }
                    ReferenceKind::Video | ReferenceKind::VideoAudio => {
                        let (r_frame, r_w) = frame_grid(block.latent_h, block.latent_w)?;
                        if block.ref_audio_t > 0 {
                            b.append(
                                SegmentKind::RefAudio,
                                audio_grid(
                                    cursor,
                                    block.ref_audio_t,
                                    r_w.double_value(&[0]),
                                    r_w.double_value(&[r_w.size()[0] - 1]),
                                )?,
                                false,
                            )?;
                        }
                        b.append(
                            SegmentKind::RefImg,
                            video_grid(block.latent_t, &r_frame, cursor)?,
                            false,
                        )?;
                        cursor += block.span();
                    }
                }
            }
        }
        b.append(
            SegmentKind::Audio,
            audio_grid(cursor, audio_t, low, high)?,
            true,
        )?;
        b.append(
            SegmentKind::Video,
            video_grid(latent_t, &frame, cursor)?,
            true,
        )?;
        Ok(Self {
            seq_len: b.row,
            signature,
            position_ids: Tensor::f_cat(&b.positions, 0)?,
            img_pos: Tensor::f_cat(&b.img_pos, 0)?,
            img_update: Tensor::f_cat(&b.img_update, 0)?,
            audio_pos: Tensor::f_cat(&b.audio_pos, 0)?,
            audio_update: Tensor::f_cat(&b.audio_update, 0)?,
            segments: b.segments,
        })
    }
}
