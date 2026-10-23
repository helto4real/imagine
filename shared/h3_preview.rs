//! Sampling-preview presentation contract. RGB bytes are separate, transient RAM data.
use serde::{Deserialize, Serialize};

pub const MAX_RESOLUTION: u32 = 768;
pub const MAX_FRAMES: u32 = 48;
pub const MAX_PIXELS: usize = 8_000_000;
pub const MAX_BYTES: usize = MAX_PIXELS * 3;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    pub enabled: bool,
    pub interval_seconds: f64,
    pub max_resolution: u32,
    pub frames: u32,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: 10.,
            max_resolution: 512,
            frames: 24,
        }
    }
}
impl Settings {
    pub fn valid(&self) -> bool {
        self.interval_seconds.is_finite() && self.interval_seconds > 0.
    }
    /// Preserve user settings; apply Director's documented preview runtime bounds.
    pub fn effective(&self) -> Self {
        Self {
            max_resolution: self.max_resolution.clamp(1, MAX_RESOLUTION),
            frames: self.frames.clamp(1, MAX_FRAMES),
            ..*self
        }
    }
}

/// One COMPLETE, silent RGB8 animation, not a frame of the final generated video.
/// Payload order is temporal HWC RGB8, with `frames` contiguous images.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub sequence: u64,
    /// Sampler progress label; zero is valid for a zero-based callback.
    pub step: u32,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub frame_duration_ms: u64,
    pub source_duration_seconds: f64,
}
impl Metadata {
    pub fn bytes(&self) -> Option<usize> {
        if self.width == 0
            || self.width > MAX_RESOLUTION
            || self.height == 0
            || self.height > MAX_RESOLUTION
            || self.frames == 0
            || self.frames > MAX_FRAMES
        {
            return None;
        }
        let pixels = usize::try_from(self.width)
            .ok()?
            .checked_mul(usize::try_from(self.height).ok()?)?
            .checked_mul(usize::try_from(self.frames).ok()?)?;
        if pixels > MAX_PIXELS {
            return None;
        }
        pixels.checked_mul(3).filter(|bytes| *bytes <= MAX_BYTES)
    }
    pub fn valid_payload(&self, bytes: usize) -> bool {
        self.sequence > 0
            && self.frame_duration_ms > 0
            && self.source_duration_seconds.is_finite()
            && self.source_duration_seconds > 0.
            && self.bytes() == Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preview_settings_preserve_requested_values_and_only_bound_runtime_size() {
        let defaults: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults, Settings::default());
        let requested = Settings {
            max_resolution: u32::MAX,
            frames: u32::MAX,
            ..defaults
        };
        assert!(requested.valid());
        assert_eq!(requested.max_resolution, u32::MAX);
        assert_eq!(requested.effective().max_resolution, MAX_RESOLUTION);
        assert_eq!(requested.effective().frames, MAX_FRAMES);
        for interval in [0., -1., f64::INFINITY, f64::NAN] {
            assert!(
                !Settings {
                    interval_seconds: interval,
                    ..defaults
                }
                .valid()
            );
        }
    }
    #[test]
    fn full_rgb_animation_has_checked_exact_size_and_finite_timing() {
        let mut m = Metadata {
            sequence: 1,
            step: 0,
            width: 500,
            height: 500,
            frames: 32,
            frame_duration_ms: 40,
            source_duration_seconds: 1.28,
        };
        assert_eq!(m.bytes(), Some(MAX_BYTES));
        assert!(m.valid_payload(MAX_BYTES));
        assert!(!m.valid_payload(MAX_BYTES - 1));
        m.frames += 1;
        assert_eq!(m.bytes(), None);
        m.frames = 1;
        m.width = 769;
        assert_eq!(m.bytes(), None);
        m.width = 1;
        m.height = 1;
        assert!(m.valid_payload(3));
        m.frame_duration_ms = 0;
        assert!(!m.valid_payload(3));
        m.frame_duration_ms = 1;
        m.source_duration_seconds = f64::NAN;
        assert!(!m.valid_payload(3));
    }
}
