//! Director H3 sizing: target area, nearest-32 (half up), portrait swap.
//! Source: comfyui-helto-director ab8a643, shared/h3/{config,planner}.py.
use serde::{Deserialize, Serialize};
pub const RATIOS: [&str; 5] = ["16:9", "4:3", "3:2", "21:9", "1:1"];
pub const QUALITIES: [&str; 5] = [
    "Quick Draft",
    "Draft",
    "Standard",
    "High",
    "Native Resolution",
];
/// Bounded in-memory product transport, not a GPU-fit promise. Supports native
/// 1 MP / 15 s; reject larger decoded results before any model is constructed.
pub const MAX_RAW_BYTES: u64 = 2 * 1024 * 1024 * 1024;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
}
impl Default for Shape {
    fn default() -> Self {
        Self {
            width: 384,
            height: 256,
            frames: 39,
            fps: 24,
        }
    }
}
impl Shape {
    pub fn frame_bytes(self) -> usize {
        self.width as usize * self.height as usize * 3
    }
    pub fn audio_samples(self) -> u32 {
        ((u64::from(self.frames) * 4000 + 1) / 3) as u32
    }
    pub fn video_latent_t(self) -> i64 {
        i64::from((self.frames - 5) / 17 * 5 + 2)
    }
    pub fn audio_latent_t(self) -> i64 {
        (f64::from(self.frames) * (5. / 3.)).round_ties_even() as i64
    }
    pub fn native_audio_samples(self) -> u32 {
        self.audio_latent_t() as u32 * 800
    }
    pub fn seconds(self) -> f64 {
        f64::from(self.frames) / 24.
    }
    pub fn raw_bytes(self) -> u64 {
        u64::from(self.width) * u64::from(self.height) * 3 * u64::from(self.frames)
            + u64::from(self.audio_samples()) * 8
    }
    pub fn valid(self) -> bool {
        self.width >= 32
            && self.height >= 32
            && self.width <= 4096
            && self.height <= 4096
            && self.width.is_multiple_of(32)
            && self.height.is_multiple_of(32)
            && self.fps == 24
            && self.frames >= 5
            && self.frames <= 86_405
            && self.frames % 17 == 5
            && u64::from(self.width) * u64::from(self.height) * 3 <= super::MAX_PAYLOAD as u64
            && self.raw_bytes() <= MAX_RAW_BYTES
    }
    /// H3 audio's 40 Hz latent grid has 800-sample cells. Crop/pad only the
    /// exported tail (at most 267 samples), keeping the requested video duration.
    pub fn fit_audio(self, mut samples: Vec<f32>) -> Result<Vec<f32>, &'static str> {
        if !self.valid()
            || samples.len() != self.native_audio_samples() as usize * 2
            || !samples.iter().all(|v| v.is_finite())
        {
            return Err("h3_invalid_audio");
        }
        samples.resize(self.audio_samples() as usize * 2, 0.);
        Ok(samples)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub ratio: String,
    pub portrait: bool,
    pub quality: String,
    pub seconds: f64,
}
impl Default for Selection {
    fn default() -> Self {
        Self {
            ratio: "3:2".into(),
            portrait: false,
            quality: "Quick Draft".into(),
            seconds: 1.625,
        }
    }
}
impl Selection {
    pub fn resolve(&self) -> Result<Shape, &'static str> {
        self.resolve_with_source(None)
    }
    pub fn resolve_with_source(&self, source: Option<(u32, u32)>) -> Result<Shape, &'static str> {
        if !self.seconds.is_finite() || self.seconds < 0.25 || self.seconds > 3600. {
            return Err("invalid_h3_duration");
        }
        let ratios = [(16., 9.), (4., 3.), (3., 2.), (21., 9.), (1., 1.)];
        let (mut w, mut h) = if self.ratio == "Source" {
            let (w, h) = source
                .filter(|(w, h)| *w > 0 && *h > 0 && *w <= 8192 && *h <= 8192)
                .ok_or("invalid_h3_source_ratio")?;
            (f64::from(w), f64::from(h))
        } else {
            let i = RATIOS
                .iter()
                .position(|r| *r == self.ratio)
                .ok_or("invalid_h3_ratio")?;
            ratios[i]
        };
        if self.ratio != "Source" && self.portrait && w > h {
            std::mem::swap(&mut w, &mut h);
        }
        let q = QUALITIES
            .iter()
            .position(|p| *p == self.quality)
            .ok_or("invalid_h3_quality")?;
        let pixels: f64 = [100_000., 200_000., 400_000., 500_000., 1_000_000.][q];
        let snap = |v: f64| ((v / 32. + 0.5).floor() as u32 * 32).max(32);
        let requested = (self.seconds * 24.).ceil() as u32;
        let n = requested.max(5);
        let frames = n + (22 - n % 17) % 17;
        let shape = Shape {
            width: snap((pixels * w / h).sqrt()),
            height: snap((pixels * h / w).sqrt()),
            frames,
            fps: 24,
        };
        if !shape.valid() {
            return Err("h3_result_exceeds_memory_limit");
        }
        Ok(shape)
    }
}
