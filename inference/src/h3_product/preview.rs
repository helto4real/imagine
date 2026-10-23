//! Director's fast H3 denoised-video preview. No VAE, output mutation or media files.
//!
//! Input is the unpacked video stream `[B,C,T,H,W]` at the sampler callback,
//! before latent-format output processing. Audio and padded channels are ignored.
//! The caller owns cadence start, unpacking, transport, latest-preview replacement
//! and non-fatal error presentation. This module returns RGB8, not encoded WebP.
use crate::native::Error;
use tch::{Device, Kind, Tensor};

pub const DEFAULT_INTERVAL_SECONDS: f64 = 10.;
pub const MAX_RESOLUTION: u32 = 768;
pub const MAX_FRAMES: u32 = 48;
pub const MAX_TOTAL_PIXELS: usize = 8_000_000;

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub maximum_resolution: u32,
    pub preview_frames: u32,
    pub source_fps: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            maximum_resolution: 512,
            preview_frames: 24,
            source_fps: 24.,
        }
    }
}

/// No wall clock is read here. Pass elapsed seconds from this sampling invocation,
/// not from worker launch/model loading; one slot is attempted at most once.
#[derive(Debug)]
pub struct Cadence {
    interval_seconds: f64,
    emitted_slot: u64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Due {
    pub slot: u64,
    pub elapsed_seconds: f64,
}
impl Cadence {
    pub fn new(interval_seconds: f64) -> Result<Self, Error> {
        if !interval_seconds.is_finite() || interval_seconds <= 0. {
            return Err(value_error(
                "Preview interval must be positive finite seconds",
            ));
        }
        Ok(Self {
            interval_seconds,
            emitted_slot: 0,
        })
    }
    pub fn due(&self, elapsed_seconds: f64) -> Option<Due> {
        if !elapsed_seconds.is_finite() {
            return None;
        }
        let elapsed_seconds = elapsed_seconds.max(0.);
        // Python's float // corrects division using fmod; floor(a / b) alone
        // differs at representable decimal boundaries such as 1.0 // 0.1 == 9.
        let quotient =
            (elapsed_seconds - elapsed_seconds % self.interval_seconds) / self.interval_seconds;
        let floor = quotient.floor();
        let slot = (floor + if quotient - floor > 0.5 { 1. } else { 0. }) as u64;
        (slot > self.emitted_slot).then_some(Due {
            slot,
            elapsed_seconds,
        })
    }
    /// Match Director: call after obtaining a valid video latent, before rendering.
    /// A rendering failure is non-fatal and is not retried on every callback in a slot.
    pub fn mark_emitted(&mut self, due: Due) {
        self.emitted_slot = self.emitted_slot.max(due.slot);
    }
}

pub struct Rendered {
    pub width: u32,
    pub height: u32,
    /// HWC RGB8 frames in batch/temporal order, with no retained tensors or audio.
    pub frames: Vec<Vec<u8>>,
    pub source_frames: u64,
    pub source_duration_seconds: f64,
    /// Director's unrounded effective fps (before animated-WebP millisecond timing).
    pub fps: f64,
    /// Director's animated-WebP frame delay. Raw transport may use this same clock.
    pub frame_duration_ms: u64,
    pub bounded: bool,
}
impl Rendered {
    pub fn playback_duration_seconds(&self) -> f64 {
        self.frames.len() as f64 * self.frame_duration_ms as f64 / 1000.
    }
}

// comfy.latent_formats.MiniMaxH3Video; Director's LatentRgbDecoder uses these
// first 24 channels even when MiniMaxH3AV has 32 padded channels.
const RGB_FACTORS: [[f32; 3]; 24] = [
    [-0.018555, 0.024344, -0.017536],
    [0.150164, 0.137244, 0.129221],
    [0.027367, -0.050369, -0.208606],
    [-0.000793, -0.164622, -0.323161],
    [-0.048556, 0.013970, -0.074286],
    [0.011740, 0.014172, -0.006906],
    [0.061517, 0.061212, 0.110025],
    [0.035321, 0.086879, 0.110059],
    [-0.017426, 0.002997, 0.035356],
    [0.531539, 0.548819, 0.624404],
    [-0.024968, -0.040234, -0.034302],
    [-0.032549, -0.029096, -0.017221],
    [0.022609, 0.020286, 0.050661],
    [-0.084001, -0.038131, -0.020805],
    [-0.018830, 0.010412, 0.061120],
    [0.020777, 0.011196, -0.030994],
    [-0.008390, -0.012201, -0.025687],
    [-0.013281, -0.002924, 0.006331],
    [0.000260, 0.001833, -0.011038],
    [0.105471, 0.100482, 0.132106],
    [0.016529, 0.015213, 0.009999],
    [-0.014015, -0.017438, -0.019134],
    [-0.033787, -0.009984, -0.019725],
    [0.004224, 0.017284, 0.027196],
];
const RGB_BIAS: [f32; 3] = [0.057426, -0.022078, -0.071449];

/// Director's H3 temporal inverse, including the fallback for intermediate shapes.
pub fn pixel_frame_count(latent_frames: u64) -> u64 {
    if latent_frames <= 2 {
        5
    } else if (latent_frames - 2).is_multiple_of(5) {
        17 * ((latent_frames - 2) / 5) + 5
    } else {
        ((latent_frames as f64 * 17. / 5.).round_ties_even() as u64).max(1)
    }
}

fn validate_video(video: &Tensor) -> Result<[i64; 5], Error> {
    let shape = video.size();
    if shape.len() != 5 || shape.iter().any(|n| *n <= 0) || shape[1] < 24 {
        return Err(value_error("Expected H3 video latent [B,C>=24,T,H,W]"));
    }
    Ok([shape[0], shape[1], shape[2], shape[3], shape[4]])
}

/// Use the same ATen linspace/round/index_select path as Director. In particular,
/// half indices use ties-to-even, and a single requested latent frame is the first.
pub fn thin_video_frames(video: &Tensor, maximum_frames: u32) -> Result<Tensor, Error> {
    let shape = validate_video(video)?;
    if shape[2] <= i64::from(maximum_frames) || maximum_frames == 0 {
        return Ok(video.shallow_clone());
    }
    let indices = Tensor::f_linspace(
        0.,
        (shape[2] - 1) as f64,
        i64::from(maximum_frames),
        (Kind::Float, video.device()),
    )?
    .f_round()?
    .f_to_kind(Kind::Int64)?;
    // maximum_frames < T, so linspace's rounded indices are already unique/sorted.
    Ok(video.f_index_select(2, &indices)?)
}

/// Director's FP32 linear projection, before RGB8 conversion and image resizing.
/// Uses f_linear rather than an alternate scalar accumulation/kernel.
pub fn latent_rgb(video: &Tensor) -> Result<Tensor, Error> {
    let [batch, channels, frames, height, width] = validate_video(video)?;
    let flat = video
        .f_movedim_int(2, 1)?
        .f_reshape([batch * frames, channels, height, width])?
        .f_narrow(1, 0, 24)?
        .f_to_kind(Kind::Float)?;
    let factors: Vec<f32> = RGB_FACTORS.into_iter().flatten().collect();
    let weights = Tensor::f_from_slice(&factors)?
        .f_reshape([24, 3])?
        .f_transpose(0, 1)?
        .f_to_device(video.device())?;
    let bias = Tensor::f_from_slice(&RGB_BIAS)?.f_to_device(video.device())?;
    Ok(flat
        .f_movedim_int(1, -1)?
        .f_linear(&weights, Some(&bias))?
        .f_add_scalar(1.)?
        .f_div_scalar(2.)?
        .f_clamp(0., 1.)?)
}

pub fn render_fast(video: &Tensor, options: &Options) -> Result<Rendered, Error> {
    let [_, _, temporal, height, width] = validate_video(video)?;
    if !options.source_fps.is_finite() || options.source_fps <= 0. {
        return Err(value_error(
            "Preview source fps must be positive and finite",
        ));
    }
    let edge = options.maximum_resolution.clamp(1, MAX_RESOLUTION);
    let count = options.preview_frames.clamp(1, MAX_FRAMES);
    let selected = thin_video_frames(video, count)?;
    let rgb = latent_rgb(&selected)?;
    // Pillow receives uint8 pixels, not float pixels to be resized then quantized.
    let bytes = rgb
        .f_mul_scalar(255.)?
        .f_to_kind(Kind::Uint8)?
        .f_to_device(Device::Cpu)?
        .f_contiguous()?;
    let frame_count = usize::try_from(bytes.size()[0])
        .map_err(|_| value_error("Preview frame count overflow"))?;
    let width = u32::try_from(width).map_err(|_| value_error("Preview width overflow"))?;
    let height = u32::try_from(height).map_err(|_| value_error("Preview height overflow"))?;
    let frame_bytes = rgb_len(width, height)?;
    let total = frame_bytes
        .checked_mul(frame_count)
        .ok_or_else(|| value_error("Preview buffer overflow"))?;
    let mut raw = vec![0u8; total];
    bytes.f_copy_data_u8(&mut raw, total)?;
    let (target_width, target_height) = output_size(width, height, edge);
    let filter = if edge < width.max(height) {
        Filter::Lanczos
    } else {
        Filter::Bicubic
    };
    let mut frames = raw
        .chunks_exact(frame_bytes)
        .map(|pixels| resize_rgb(pixels, width, height, target_width, target_height, filter))
        .collect::<Result<Vec<_>, _>>()?;
    let source_frames = pixel_frame_count(temporal as u64);
    let source_duration_seconds = source_frames as f64 / options.source_fps;
    let mut fps = (options.source_fps * frames.len() as f64 / source_frames as f64).max(0.1);
    let rendered_count = frames.len();
    let pixels = rgb_len(target_width, target_height)? / 3;
    let maximum = (MAX_TOTAL_PIXELS / pixels).max(1);
    if frames.len() > maximum {
        let indices = sample_frame_indices(frames.len(), maximum);
        let mut owned: Vec<Option<Vec<u8>>> = frames.into_iter().map(Some).collect();
        frames = indices
            .into_iter()
            .map(|index| owned[index].take().unwrap())
            .collect();
        fps = (fps * frames.len() as f64 / rendered_count as f64).max(0.1);
    }
    Ok(Rendered {
        width: target_width,
        height: target_height,
        source_frames,
        source_duration_seconds,
        fps,
        frame_duration_ms: (1000. / fps.max(0.1)).round_ties_even().max(1.) as u64,
        bounded: edge != options.maximum_resolution
            || count != options.preview_frames
            || frames.len() != rendered_count,
        frames,
    })
}

fn value_error(message: &str) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
fn rgb_len(width: u32, height: u32) -> Result<usize, Error> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| value_error("Preview image buffer overflow"))
}
fn output_size(width: u32, height: u32, edge: u32) -> (u32, u32) {
    let scale = f64::from(edge) / f64::from(width.max(height));
    (
        (f64::from(width) * scale).round_ties_even().max(1.) as u32,
        (f64::from(height) * scale).round_ties_even().max(1.) as u32,
    )
}
// Director's later PIL-frame budget sampling differs from latent sampling when
// maximum == 1: retain the LAST rendered frame, not the first latent frame.
fn sample_frame_indices(length: usize, maximum: usize) -> Vec<usize> {
    if maximum >= length {
        return (0..length).collect();
    }
    if maximum <= 1 {
        return vec![length - 1];
    }
    (0..maximum)
        .map(|i| (i as f64 * (length - 1) as f64 / (maximum - 1) as f64).round_ties_even() as usize)
        .collect()
}

// Adapted from Imagine src/video/rgb_resize.rs and Pillow 12.3.0
// src/libImaging/Resample.c (HPND). Kept local to this dependency-free inference
// crate: identical 22-bit separable RGB8 accumulation, now with Bicubic upsampling.
#[derive(Clone, Copy)]
enum Filter {
    Lanczos,
    Bicubic,
}
impl Filter {
    fn support(self) -> f64 {
        match self {
            Self::Lanczos => 3.,
            Self::Bicubic => 2.,
        }
    }
    fn weight(self, x: f64) -> f64 {
        match self {
            Self::Lanczos if (-3. ..3.).contains(&x) => sinc(x) * sinc(x / 3.),
            Self::Lanczos => 0.,
            Self::Bicubic => {
                let x = x.abs();
                if x < 1. {
                    ((1.5 * x - 2.5) * x) * x + 1.
                } else if x < 2. {
                    (((x - 5.) * x + 8.) * x - 4.) * -0.5
                } else {
                    0.
                }
            }
        }
    }
}
fn sinc(x: f64) -> f64 {
    if x == 0. {
        1.
    } else {
        let p = x * std::f64::consts::PI;
        p.sin() / p
    }
}
const PRECISION: u32 = 22;
fn coefficients(input: usize, output: usize, filter: Filter) -> Vec<(usize, Vec<i64>)> {
    // Pillow's box is float32, then filter arithmetic is double precision.
    let scale = f64::from(input as f32) / output as f64;
    let filter_scale = scale.max(1.);
    let inverse = 1. / filter_scale;
    let support = filter.support() * filter_scale;
    (0..output)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            let first = ((centre - support + 0.5) as i64).max(0) as usize;
            let end = ((centre + support + 0.5) as usize).min(input);
            let weights: Vec<f64> = (first..end)
                .map(|j| filter.weight((j as f64 - centre + 0.5) * inverse))
                .collect();
            let sum = weights.iter().sum::<f64>();
            let fixed = weights
                .into_iter()
                .map(|w| {
                    let normalized = if sum == 0. { w } else { w / sum };
                    let scaled = normalized * (1u64 << PRECISION) as f64;
                    (scaled + if scaled < 0. { -0.5 } else { 0.5 }) as i64
                })
                .collect();
            (first, fixed)
        })
        .collect()
}
fn filtered(
    source: &[u8],
    width: usize,
    height: usize,
    target: usize,
    horizontal: bool,
    filter: Filter,
) -> Vec<u8> {
    let input = if horizontal { width } else { height };
    if input == target {
        return source.to_vec();
    }
    let table = coefficients(input, target, filter);
    let (w, h) = if horizontal {
        (target, height)
    } else {
        (width, target)
    };
    let mut result = vec![0; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (first, weights) = &table[if horizontal { x } else { y }];
            for channel in 0..3 {
                let mut sum = 1i64 << (PRECISION - 1);
                for (j, weight) in weights.iter().enumerate() {
                    let (sx, sy) = if horizontal {
                        (first + j, y)
                    } else {
                        (x, first + j)
                    };
                    sum += i64::from(source[(sy * width + sx) * 3 + channel]) * weight;
                }
                result[(y * w + x) * 3 + channel] = (sum >> PRECISION).clamp(0, 255) as u8;
            }
        }
    }
    result
}
fn resize_rgb(
    source: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    filter: Filter,
) -> Result<Vec<u8>, Error> {
    if source.len() != rgb_len(width, height)? {
        return Err(value_error("Invalid RGB8 preview buffer length"));
    }
    rgb_len(target_width, height)?;
    rgb_len(target_width, target_height)?;
    let horizontal = filtered(
        source,
        width as usize,
        height as usize,
        target_width as usize,
        true,
        filter,
    );
    Ok(filtered(
        &horizontal,
        target_width as usize,
        height as usize,
        target_height as usize,
        false,
        filter,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cadence_is_elapsed_time_not_step_count_and_does_not_catch_up() {
        let mut c = Cadence::new(10.).unwrap();
        assert!(c.due(9.999).is_none());
        c.mark_emitted(c.due(10.).unwrap());
        assert!(c.due(19.999).is_none());
        let due = c.due(45.).unwrap();
        assert_eq!(due.slot, 4);
        c.mark_emitted(due);
        assert!(c.due(45.1).is_none());
        assert_eq!(c.due(50.).unwrap().slot, 5);
        assert_eq!(Cadence::new(0.1).unwrap().due(1.).unwrap().slot, 9);
        assert!(Cadence::new(0.).is_err());
    }
    #[test]
    fn whole_clip_clock_and_two_sampling_geometries_are_preserved() {
        assert_eq!([2, 7, 12, 37].map(pixel_frame_count), [5, 22, 39, 124]);
        assert_eq!(output_size(5, 3, 3), (3, 2));
        assert_eq!(output_size(5, 3, 10), (10, 6));
        assert_eq!(sample_frame_indices(6, 3), [0, 2, 5]);
        assert_eq!(sample_frame_indices(6, 1), [5]);
    }
    #[test]
    fn projection_ignores_padded_av_channels_without_changing_input() {
        let video = Tensor::zeros([1, 32, 7, 2, 3], (Kind::Float, Device::Cpu));
        let _ = video.narrow(1, 24, 8).fill_(1000.);
        let before = video.copy();
        let rgb = latent_rgb(&video).unwrap();
        assert_eq!(rgb.size(), [7, 2, 3, 3]);
        for (channel, bias) in RGB_BIAS.iter().enumerate() {
            assert!(
                (rgb.double_value(&[0, 0, 0, channel as i64]) - f64::from((*bias + 1.) / 2.)).abs()
                    < 1e-7
            );
        }
        assert!(video.equal(&before));
        let selected = thin_video_frames(&video, 1).unwrap();
        assert!(selected.equal(&video.narrow(2, 0, 1)));
    }
    #[test]
    #[ignore = "Set IMAGINE_H3_PREVIEW_ORACLE to the own encrypted source-executed fixture"]
    fn director_source_executed_fast_rgb_resize_and_timing_oracle() {
        let path = std::env::var_os("IMAGINE_H3_PREVIEW_ORACLE").expect("own oracle path");
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let shape: Vec<i64> = serde_json::from_value(fixture["shape"].clone()).unwrap();
        let input: Vec<f32> = serde_json::from_value(fixture["input"].clone()).unwrap();
        let video = Tensor::from_slice(&input).reshape(&shape);
        let expected_float: Vec<f32> = serde_json::from_value(fixture["rgb"].clone()).unwrap();
        let rgb = latent_rgb(&video).unwrap().contiguous();
        let mut actual_float = vec![0f32; expected_float.len()];
        let n = actual_float.len();
        rgb.copy_data(&mut actual_float, n);
        let max = actual_float
            .iter()
            .zip(&expected_float)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-6, "FP32 projection max {max}");
        for case in fixture["cases"].as_array().unwrap() {
            let options = Options {
                maximum_resolution: case["edge"].as_u64().unwrap() as u32,
                preview_frames: case["requested_frames"].as_u64().unwrap() as u32,
                source_fps: 24.,
            };
            let rendered = render_fast(&video, &options).unwrap();
            let expected: Vec<Vec<u8>> = serde_json::from_value(case["frames"].clone()).unwrap();
            assert_eq!(rendered.frames, expected, "{}", case["name"]);
            assert_eq!(u64::from(rendered.width), case["width"].as_u64().unwrap());
            assert_eq!(u64::from(rendered.height), case["height"].as_u64().unwrap());
            assert_eq!(
                rendered.source_frames,
                case["source_frames"].as_u64().unwrap()
            );
            assert!((rendered.fps - case["fps"].as_f64().unwrap()).abs() < 1e-12);
            assert_eq!(
                rendered.frame_duration_ms,
                case["frame_duration_ms"].as_u64().unwrap()
            );
        }
    }
}
