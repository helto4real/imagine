//! Director's latent-tail, masked-prefix and model-local16+16 attention contracts.
use super::protocol::{self as p, continuation as c};
use crate::native::Error;
use tch::{Device, Kind, Tensor};
fn error() -> Error {
    Error::Native("H3 continuation contract failed".into())
}
pub fn decode(input: &c::Input, bytes: &[u8]) -> Result<(Tensor, Tensor), Error> {
    if !input.descriptor.valid_bytes(bytes) {
        return Err(error());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    let split = input.descriptor.video_values();
    let s = input.descriptor.shape;
    Ok((
        Tensor::f_from_slice(&values[..split])?.f_view([
            1,
            24,
            7,
            i64::from(s.height / 16),
            i64::from(s.width / 16),
        ])?,
        Tensor::f_from_slice(&values[split..])?.f_view([1, 32, 2, 37])?,
    ))
}
pub fn capture(
    video: &Tensor,
    audio: &Tensor,
    shape: p::Shape,
    sampled: p::Shape,
    head: u32,
) -> Result<c::Artifact, Error> {
    let end = head + shape.frames;
    let (step, frame) = c::endpoint(sampled.video_latent_t() as u32, end);
    let descriptor = c::Descriptor {
        schema_version: 1,
        shape,
        sampled_frames: sampled.frames,
        visible_end_frame: end,
        selected_end_frame: frame,
    };
    if !descriptor.valid()
        || video.size()
            != [
                1,
                24,
                sampled.video_latent_t(),
                i64::from(shape.height / 16),
                i64::from(shape.width / 16),
            ]
        || audio.size() != [1, 32, 2, sampled.audio_latent_t()]
    {
        return Err(error());
    }
    let audio_end = (f64::from(frame) * 5. / 3.).round_ties_even() as i64;
    if audio_end < 37 || audio_end > sampled.audio_latent_t() {
        return Err(error());
    }
    let slices = [
        video.f_narrow(2, i64::from(step) - 7, 7)?,
        audio.f_narrow(3, audio_end - 37, 37)?,
    ];
    let mut bytes = Vec::with_capacity(descriptor.bytes());
    for tensor in slices {
        let tensor = tensor
            .f_to_device_(Device::Cpu, Kind::Float, false, false)?
            .f_contiguous()?
            .f_view([-1])?;
        for value in Vec::<f32>::try_from(&tensor)? {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    let artifact = c::Artifact { descriptor, bytes };
    if !artifact.valid() {
        return Err(error());
    }
    Ok(artifact)
}
pub fn prefix(
    video: &Tensor,
    audio: &Tensor,
    context: &(Tensor, Tensor),
    include_audio: bool,
) -> Result<Tensor, Error> {
    video.f_narrow(2, 0, 7)?.f_copy_(&context.0.f_to_device_(
        video.device(),
        video.kind(),
        false,
        false,
    )?)?;
    if include_audio {
        audio.f_narrow(3, 0, 37)?.f_copy_(&context.1.f_to_device_(
            audio.device(),
            audio.kind(),
            false,
            false,
        )?)?;
    }
    let vm = video.f_ones_like()?;
    let _ = vm.f_narrow(2, 0, 7)?.f_zero_()?;
    let am = audio.f_ones_like()?;
    if include_audio {
        let _ = am.f_narrow(3, 0, 37)?.f_zero_()?;
    }
    let mask = crate::models::minimax_h3::streams::NestedTensor::new(vec![
        crate::execution_value::Value::tensor(vm),
        crate::execution_value::Value::tensor(am),
    ]);
    mask.pack()?.0.as_tensor()
}
pub struct Window {
    minimum: i64,
    video_rows: i64,
    limited: Vec<(i64, i64)>,
    weights: Vec<f32>,
}
impl Window {
    pub(super) fn from_layout(
        minimum: i64,
        video_rows: i64,
        limited: Vec<(i64, i64)>,
        weights: Vec<f32>,
    ) -> Self {
        Self {
            minimum,
            video_rows,
            limited,
            weights,
        }
    }
    pub fn new(input: &c::Input, shape: p::Shape, text: i64) -> Self {
        let frame = i64::from(shape.height / 32) * i64::from(shape.width / 32);
        let video_rows = shape.video_latent_t() * frame;
        let audio_rows = shape.audio_latent_t() * 2;
        let guide_rows = if input.method == c::Method::MotionContext {
            7 * frame + if input.include_audio { 74 } else { 0 }
        } else {
            0
        };
        let minimum = text + guide_rows + audio_rows + video_rows;
        let limited = if input.method == c::Method::MotionContext {
            vec![(text, text + guide_rows)]
        } else {
            let mut ranges = vec![(text + audio_rows, text + audio_rows + 7 * frame)];
            if input.include_audio {
                for channel in 0..2 {
                    let start = text + channel * shape.audio_latent_t();
                    ranges.push((start, start + 37));
                }
            }
            ranges.sort_unstable();
            ranges
        };
        let head = if input.method == c::Method::MotionContext {
            34
        } else {
            21
        };
        let weights = c::step_weights(shape.video_latent_t() as u32, head)
            .into_iter()
            .flat_map(|w| std::iter::repeat_n(w, frame as usize))
            .collect();
        Self {
            minimum,
            video_rows,
            limited,
            weights,
        }
    }
    pub fn run(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mut backend: impl FnMut(&Tensor, &Tensor, &Tensor) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error> {
        let n = q.size()[2];
        if n < self.minimum || k.size()[2] != n || v.size()[2] != n {
            return backend(q, k, v);
        }
        if n != self.minimum {
            return Err(error());
        }
        let mut keep = vec![];
        let mut cursor = 0;
        for &(start, stop) in &self.limited {
            if start < cursor || stop <= start || stop > n {
                return Err(error());
            }
            if cursor < start {
                keep.push((cursor, start));
            }
            cursor = stop;
        }
        if cursor < n {
            keep.push((cursor, n));
        }
        if keep.is_empty() {
            return Err(error());
        }
        let slice = |tensor: &Tensor| -> Result<Tensor, Error> {
            let parts = keep
                .iter()
                .map(|&(a, b)| tensor.f_narrow(2, a, b - a).map_err(Error::from))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::f_cat(&parts, 2)?)
        };
        let restricted_k = slice(k)?;
        let restricted_v = slice(v)?;
        let fade = self
            .weights
            .iter()
            .position(|w| *w < 1.)
            .unwrap_or(self.weights.len());
        let blocked = self
            .weights
            .iter()
            .position(|w| *w <= 0.)
            .unwrap_or(self.weights.len());
        let a = n - self.video_rows + fade as i64;
        let b = n - self.video_rows + blocked as i64;
        let mut out = vec![];
        if a > 0 {
            out.push(backend(&q.f_narrow(2, 0, a)?, k, v)?);
        }
        if b > a {
            let queries = q.f_narrow(2, a, b - a)?;
            let full = backend(&queries, k, v)?;
            let restricted = backend(&queries, &restricted_k, &restricted_v)?;
            let weights = Tensor::f_from_slice(&self.weights[fade..blocked])?
                .f_to_device_(full.device(), full.kind(), false, false)?
                .f_view([1, -1, 1])?;
            out.push(restricted.f_lerp_tensor(&full, &weights)?);
        }
        if b < n {
            out.push(backend(
                &q.f_narrow(2, b, n - b)?,
                &restricted_k,
                &restricted_v,
            )?);
        }
        Ok(Tensor::f_cat(&out, 1)?)
    }
}
