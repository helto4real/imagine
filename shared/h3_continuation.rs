//! Versioned physical FP32 AV tail, without paths, pickle or device-dependent tensors.
use super::geometry::Shape;
use serde::{Deserialize, Serialize};
pub const VIDEO_STEPS: u32 = 7;
pub const AUDIO_STEPS: u32 = 37;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    MotionContext,
    MotionContextExp,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Descriptor {
    pub schema_version: u32,
    pub shape: Shape,
    pub sampled_frames: u32,
    pub visible_end_frame: u32,
    pub selected_end_frame: u32,
}
impl Descriptor {
    pub fn valid(&self) -> bool {
        let sampled = Shape {
            frames: self.sampled_frames,
            ..self.shape
        };
        self.schema_version == 1
            && self.shape.valid()
            && sampled.valid()
            && self.visible_end_frame >= 22
            && self.visible_end_frame <= self.sampled_frames
            && self.selected_end_frame == self.visible_end_frame
            && endpoint(sampled.video_latent_t() as u32, self.visible_end_frame).0 >= VIDEO_STEPS
            && endpoint(sampled.video_latent_t() as u32, self.visible_end_frame).1
                == self.selected_end_frame
            && self.bytes() <= super::MAX_PAYLOAD
    }
    pub fn video_values(&self) -> usize {
        24 * VIDEO_STEPS as usize
            * (self.shape.height / 16) as usize
            * (self.shape.width / 16) as usize
    }
    pub fn bytes(&self) -> usize {
        (self.video_values() + 32 * 2 * AUDIO_STEPS as usize) * 4
    }
    pub fn valid_bytes(&self, bytes: &[u8]) -> bool {
        self.valid()
            && bytes.len() == self.bytes()
            && bytes
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub method: Method,
    #[serde(default)]
    pub include_audio: bool,
    pub descriptor: Descriptor,
    pub sha256: String,
}
impl Input {
    pub fn valid(&self, target: Shape) -> bool {
        self.descriptor.valid()
            && super::is_sha256(&self.sha256)
            && (self.descriptor.shape.width, self.descriptor.shape.height)
                == (target.width, target.height)
            && runtime(target, self.method).is_some()
    }
}
pub struct Artifact {
    pub descriptor: Descriptor,
    pub bytes: Vec<u8>,
}
impl Artifact {
    pub fn valid(&self) -> bool {
        self.descriptor.valid_bytes(&self.bytes)
    }
}
pub fn endpoint(steps: u32, end: u32) -> (u32, u32) {
    let mut frames = 0;
    let mut selected = (0, 0);
    for step in 0..steps {
        frames += [1, 4, 4, 4, 4][step as usize % 5];
        if frames > end {
            break;
        }
        selected = (step + 1, frames);
    }
    selected
}
pub fn runtime(shape: Shape, method: Method) -> Option<(Shape, u32, u32)> {
    if !shape.valid() {
        return None;
    }
    let sampled = Shape {
        frames: shape.frames.checked_add(34)?,
        ..shape
    };
    sampled.valid().then_some(match method {
        Method::MotionContext => (sampled, 34, 14),
        Method::MotionContextExp => (sampled, 21, 0),
    })
}
/// Fit the decoder's sampled grid before selecting the visible stereo interval.
pub fn trim_audio(
    sampled: Shape,
    visible: Shape,
    head: u32,
    samples: Vec<f32>,
) -> Result<Vec<f32>, &'static str> {
    if !visible.valid()
        || (sampled.width, sampled.height, sampled.fps)
            != (visible.width, visible.height, visible.fps)
        || u64::from(head) + u64::from(visible.frames) > u64::from(sampled.frames)
    {
        return Err("h3_invalid_audio_trim");
    }
    let mut samples = sampled.fit_audio(samples)?;
    let start = (f64::from(head) * 32000. / 24.).round_ties_even() as usize * 2;
    let wanted = visible.audio_samples() as usize * 2;
    if start + wanted > samples.len() {
        return Err("h3_invalid_audio_trim");
    }
    samples.drain(..start);
    samples.truncate(wanted);
    Ok(samples)
}
/// Native packed target-video rows; context influence is full16 then linearly fades16.
pub fn step_weights(steps: u32, visible_start: u32) -> Vec<f32> {
    let mut cursor = 0;
    (0..steps)
        .map(|index| {
            let span = [1, 4, 4, 4, 4][index as usize % 5];
            let center = cursor as f64 + (span - 1) as f64 / 2.;
            cursor += span;
            ((f64::from(visible_start + 32) - center) / 16.).clamp(0., 1.) as f32
        })
        .collect()
}
