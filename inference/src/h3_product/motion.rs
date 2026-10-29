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
/// Director's ephemeral Stage 1 proxy. Keep the original high-resolution AV
/// context separately: only spatial cells are area-pooled, never time or audio.
pub fn area_context(context: &(Tensor, Tensor), base: p::Shape) -> Result<(Tensor, Tensor), Error> {
    let size = context.0.size();
    if size.len() != 5 || size[..3] != [1, 24, 7] || !base.valid() {
        return Err(error());
    }
    let height = i64::from(base.height / 16);
    let width = i64::from(base.width / 16);
    let video = context
        .0
        .f_to_kind(Kind::Float)?
        .f_reshape([-1, 1, size[3], size[4]])?
        .f_adaptive_avg_pool2d([height, width])?
        .f_reshape([size[0], size[1], size[2], height, width])?
        .f_to_kind(context.0.kind())?;
    Ok((video, context.1.shallow_clone()))
}
pub fn capture(
    video: &Tensor,
    audio: &Tensor,
    shape: p::Shape,
    sampled: p::Shape,
    head: u32,
    handoff_upscale: Option<c::HandoffUpscale>,
) -> Result<c::Artifact, Error> {
    let end = head + shape.frames;
    let (step, frame) = c::endpoint(sampled.video_latent_t() as u32, end);
    let descriptor = c::Descriptor {
        schema_version: 1,
        shape,
        sampled_frames: sampled.frames,
        visible_end_frame: end,
        selected_end_frame: frame,
        handoff_upscale,
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
    pub(super) fn matches_layout(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> bool {
        q.size()[2] == self.minimum && k.size()[2] == self.minimum && v.size()[2] == self.minimum
    }
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
    /// The sequence is [text][keyframes][reference blocks][audio][video].
    /// `keyframes` counts every keyframe row, including a MotionContext guide,
    /// which is the last keyframe; `references` counts the reference block rows.
    pub fn new(
        input: &c::Input,
        shape: p::Shape,
        text: i64,
        keyframes: i64,
        references: i64,
    ) -> Result<Self, Error> {
        let frame = i64::from(shape.height / 32) * i64::from(shape.width / 32);
        let video_rows = shape.video_latent_t() * frame;
        let audio_rows = shape.audio_latent_t() * 2;
        let guide_rows = if input.method == c::Method::MotionContext {
            7 * frame + if input.include_audio { 74 } else { 0 }
        } else {
            0
        };
        if keyframes < guide_rows || references < 0 {
            return Err(error());
        }
        let target = text + keyframes + references;
        let minimum = target + audio_rows + video_rows;
        let limited = if input.method == c::Method::MotionContext {
            vec![(text + keyframes - guide_rows, text + keyframes)]
        } else {
            let mut ranges = vec![(target + audio_rows, target + audio_rows + 7 * frame)];
            if input.include_audio {
                for channel in 0..2 {
                    let start = target + channel * shape.audio_latent_t();
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
        Ok(Self {
            minimum,
            video_rows,
            limited,
            weights,
        })
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
        // A query slice is a strided view; SDPA receives it dense so that a
        // fused kernel, not the materializing math kernel, can take it.
        let rows = |start: i64, length: i64| -> Result<Tensor, Error> {
            Ok(q.f_narrow(2, start, length)?.f_contiguous()?)
        };
        let mut out = vec![];
        if a > 0 {
            out.push(backend(&rows(0, a)?, k, v)?);
        }
        if b > a {
            let queries = rows(a, b - a)?;
            let full = backend(&queries, k, v)?;
            let restricted = backend(&queries, &restricted_k, &restricted_v)?;
            let weights = Tensor::f_from_slice(&self.weights[fade..blocked])?
                .f_to_device_(full.device(), full.kind(), false, false)?
                .f_view([1, -1, 1])?;
            out.push(restricted.f_lerp_tensor(&full, &weights)?);
        }
        if b < n {
            out.push(backend(&rows(b, n - b)?, &restricted_k, &restricted_v)?);
        }
        Ok(Tensor::f_cat(&out, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn values(tensor: &Tensor) -> Vec<f32> {
        Vec::<f32>::try_from(
            &tensor
                .f_to_kind(Kind::Float)
                .unwrap()
                .f_contiguous()
                .unwrap()
                .f_view([-1])
                .unwrap(),
        )
        .unwrap()
    }
    fn source() -> (Tensor, Tensor) {
        let data = (0..24 * 7 * 4 * 6)
            .map(|v| (v % 31) as f32 / 16.)
            .collect::<Vec<_>>();
        (
            Tensor::f_from_slice(&data)
                .unwrap()
                .f_view([1, 24, 7, 4, 6])
                .unwrap(),
            Tensor::f_from_slice(
                &(0..32 * 2 * 37)
                    .map(|v| (v % 17) as f32 / 8.)
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .f_view([1, 32, 2, 37])
            .unwrap(),
        )
    }
    #[test]
    fn area_proxy_preserves_planes_audio_and_original_high_resolution_tail() {
        let original = source();
        let before = values(&original.0);
        let base = p::Shape {
            width: 64,
            height: 32,
            ..p::Shape::default()
        };
        let proxy = area_context(&original, base).unwrap();
        assert_eq!(proxy.0.size(), [1, 24, 7, 2, 4]);
        // Independent adaptive-area bins for the noninteger spatial scale 6 -> 4.
        let mut expected = vec![];
        for plane in before.chunks_exact(24) {
            for row in [0, 2] {
                for col in [0, 1, 3, 4] {
                    expected.push(
                        (plane[row * 6 + col]
                            + plane[row * 6 + col + 1]
                            + plane[(row + 1) * 6 + col]
                            + plane[(row + 1) * 6 + col + 1])
                            / 4.,
                    );
                }
            }
        }
        assert_eq!(values(&proxy.0), expected);
        assert_eq!(values(&proxy.1), values(&original.1));
        assert_eq!(values(&original.0), before);
        let half = (
            original.0.f_to_kind(Kind::Half).unwrap(),
            original.1.shallow_clone(),
        );
        let half_proxy = area_context(&half, base).unwrap();
        assert_eq!(half_proxy.0.kind(), Kind::Half);
        assert_eq!(values(&half_proxy.0), expected);
    }
    #[test]
    fn stage_two_restores_original_prefix_after_renoise_without_resizing_audio() {
        let original = source();
        let proxy = area_context(
            &original,
            p::Shape {
                width: 64,
                height: 32,
                ..p::Shape::default()
            },
        )
        .unwrap();
        let first_video = Tensor::zeros([1, 24, 12, 2, 4], (Kind::Float, Device::Cpu));
        let first_audio = Tensor::ones([1, 32, 2, 80], (Kind::Float, Device::Cpu));
        let _ = prefix(&first_video, &first_audio, &proxy, true).unwrap();
        assert_eq!(values(&first_video.narrow(2, 0, 7)), values(&proxy.0));
        assert_eq!(values(&first_audio.narrow(3, 0, 37)), values(&original.1));
        for include_audio in [false, true] {
            let prediction = Tensor::ones([1, 24, 12, 4, 6], (Kind::Float, Device::Cpu));
            let noise = Tensor::full([1, 24, 12, 4, 6], 3., (Kind::Float, Device::Cpu));
            let video = super::super::sampling::renoise_video(&prediction, &noise, 0.5).unwrap();
            let audio = Tensor::full([1, 32, 2, 80], 9., (Kind::Float, Device::Cpu));
            let mask = prefix(&video, &audio, &original, include_audio).unwrap();
            assert_eq!(values(&video.narrow(2, 0, 7)), values(&original.0));
            assert!(values(&video.narrow(2, 7, 5)).iter().all(|v| *v == 4.));
            if include_audio {
                assert_eq!(values(&audio.narrow(3, 0, 37)), values(&original.1));
            } else {
                assert!(values(&audio).iter().all(|v| *v == 9.));
            }
            assert!(values(&audio.narrow(3, 37, 43)).iter().all(|v| *v == 9.));
            let mask = values(&mask);
            assert!(mask.iter().all(|v| *v == 0. || *v == 1.));
            assert_eq!(
                mask.iter().filter(|v| **v == 0.).count(),
                24 * 7 * 4 * 6 + if include_audio { 32 * 2 * 37 } else { 0 }
            );
        }
    }
    #[test]
    fn capture_retains_final_geometry_and_roundtrips_original_tail() {
        let shape = p::Shape {
            width: 96,
            height: 64,
            ..p::Shape::default()
        };
        let (sampled, head, _) = c::runtime(shape, c::Method::MotionContextExp).unwrap();
        let video = Tensor::ones(
            [1, 24, sampled.video_latent_t(), 4, 6],
            (Kind::Float, Device::Cpu),
        );
        let audio = Tensor::full(
            [1, 32, 2, sampled.audio_latent_t()],
            0.25,
            (Kind::Float, Device::Cpu),
        );
        let handoff = c::HandoffUpscale {
            base_width: 64,
            base_height: 32,
            factor: 1.5,
        };
        let artifact =
            capture(&video, &audio, shape, sampled, head, Some(handoff.clone())).unwrap();
        assert!(artifact.valid());
        assert_eq!(artifact.descriptor.handoff_upscale, Some(handoff));
        let input = c::Input {
            method: c::Method::MotionContextExp,
            include_audio: true,
            descriptor: artifact.descriptor,
            sha256: "a".repeat(64),
        };
        let decoded = decode(&input, &artifact.bytes).unwrap();
        assert_eq!(decoded.0.size(), [1, 24, 7, 4, 6]);
        assert!(values(&decoded.0).iter().all(|v| *v == 1.));
        assert!(values(&decoded.1).iter().all(|v| *v == 0.25));
    }
    fn window_input(method: c::Method, include_audio: bool) -> c::Input {
        let shape = p::Shape {
            width: 64,
            height: 32,
            frames: 56,
            fps: 24,
        };
        c::Input {
            method,
            include_audio,
            descriptor: c::Descriptor {
                schema_version: 1,
                shape,
                sampled_frames: 56,
                visible_end_frame: 56,
                selected_end_frame: 56,
                handoff_upscale: None,
            },
            sha256: "a".repeat(64),
        }
    }
    #[test]
    fn motion_windows_place_the_guide_and_prefix_after_keyframes_and_reference_blocks() {
        use crate::models::minimax_h3::layout::{ReferenceBlock, ReferenceKind};
        let shape = p::Shape {
            width: 64,
            height: 32,
            frames: 56,
            fps: 24,
        };
        let frame = 2;
        let (audio_rows, video_rows) = (shape.audio_latent_t() * 2, shape.video_latent_t() * frame);
        let image = ReferenceBlock {
            kind: ReferenceKind::Image,
            latent_t: 1,
            latent_h: 4,
            latent_w: 4,
            ref_audio_t: 0,
        };
        let references = super::super::reference_window::rows(&[image.clone(), image.clone()]);
        assert_eq!(references, 8);
        // MotionContext: the guide is the last keyframe, before the reference blocks.
        let guide = 7 * frame;
        let context = window_input(c::Method::MotionContext, false);
        for (keyframes, start) in [(guide, 2), (guide + 4, 6)] {
            let window = Window::new(&context, shape, 2, keyframes, references).unwrap();
            assert_eq!(window.limited, vec![(start, start + guide)]);
            assert_eq!(
                window.minimum,
                2 + keyframes + references + audio_rows + video_rows
            );
        }
        assert!(Window::new(&context, shape, 2, guide - 1, references).is_err());
        // Exp/Combo/Influence: the restricted prefix is inside the target, after the references.
        let exp = window_input(c::Method::MotionContextExp, true);
        let window = Window::new(&exp, shape, 2, 0, references).unwrap();
        let target = 2 + references;
        let mut expected = vec![
            (target, target + 37),
            (
                target + shape.audio_latent_t(),
                target + shape.audio_latent_t() + 37,
            ),
            (target + audio_rows, target + audio_rows + 7 * frame),
        ];
        expected.sort_unstable();
        assert_eq!(window.limited, expected);
        assert_eq!(window.minimum, target + audio_rows + video_rows);
        // The sequence now matches; the old length without reference rows did not.
        let attention = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor, Error> {
            let d = q.size();
            Ok(q.f_matmul(&k.f_transpose(2, 3)?)?
                .f_softmax(-1, Kind::Double)?
                .f_matmul(v)?
                .f_transpose(1, 2)?
                .f_contiguous()?
                .f_view([d[0], d[2], d[1] * d[3]])?)
        };
        let tokens = |n: i64| Tensor::zeros([1, 1, n, 3], (Kind::Double, Device::Cpu));
        let full = tokens(window.minimum);
        assert!(window.run(&full, &full, &full, attention).is_ok());
        let legacy = Window::new(&exp, shape, 2, 0, 0).unwrap();
        assert!(legacy.run(&full, &full, &full, attention).is_err());
    }
    #[test]
    fn reference_rows_match_the_boundary_window_layout() {
        use crate::models::minimax_h3::layout::{ReferenceBlock, ReferenceKind};
        let shape = p::Shape {
            width: 64,
            height: 32,
            frames: 56,
            fps: 24,
        };
        let block = |kind, latent_t, latent_h, latent_w, ref_audio_t| ReferenceBlock {
            kind,
            latent_t,
            latent_h,
            latent_w,
            ref_audio_t,
        };
        let blocks = [
            block(ReferenceKind::Image, 1, 4, 4, 0),
            block(ReferenceKind::Audio, 0, 0, 0, 3),
            block(ReferenceKind::VideoAudio, 2, 2, 4, 2),
            block(ReferenceKind::Video, 3, 4, 2, 0),
        ];
        let window = super::super::reference_window::prepare(shape, 2, 3, &blocks).unwrap();
        let rows = super::super::reference_window::rows(&blocks);
        assert_eq!(
            window.minimum,
            2 + 3 + rows + shape.audio_latent_t() * 2 + shape.video_latent_t() * 2
        );
    }
}
