//! Private H3 AV wire contract. Included by path by product and inference.
//! Only serde/serde_json/std: no tensor, CUDA, filesystem or async dependencies.
use serde::{Deserialize, Serialize};

pub const VERSION: u32 = 1;
pub const CORE_REVISION: &str = "40ff91114a6d71782aa1a79a76df6940546570bb";
pub const MODEL_LOCK_SHA256: &str =
    "9cb461e3b118be5775602a81e4ddbd7e727b178f45aba9ce8fab0a3917c296b4";
pub const MAX_HEADER: usize = 64 * 1024;
pub const MAX_PAYLOAD: usize = 4 * 1024 * 1024;
pub const MAX_PROMPT_BYTES: usize = 16 * 1024;
pub const WIDTH: u32 = 384;
pub const HEIGHT: u32 = 256;
pub const FRAMES: u32 = 39;
pub const FPS: u32 = 24;
pub const SAMPLE_RATE: u32 = 32_000;
pub const CHANNELS: u32 = 2;
pub const AUDIO_SAMPLES: u32 = 52_000;
pub const FRAME_BYTES: usize = WIDTH as usize * HEIGHT as usize * 3;
pub const STEPS: u32 = 4;

/// Immutable experimental profile: no references, LoRA, negative prompt or overrides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Profile {
    #[serde(rename = "h3-short-av-noref-v1")]
    ShortAvNoReferenceV1,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    pub protocol: u32,
    pub profile: Profile,
    pub core_revision: String,
    pub model_lock_sha256: String,
    pub worker_sha256: String,
    /// Digest of canonical JSON BTreeMap<absolute dependency path, sha256>.
    /// This map excludes the worker itself and the four model files.
    pub dependencies_sha256: String,
}
impl Deployment {
    pub fn valid(&self) -> bool {
        self.protocol == VERSION
            && self.core_revision == CORE_REVISION
            && self.model_lock_sha256 == MODEL_LOCK_SHA256
            && is_sha256(&self.worker_sha256)
            && is_sha256(&self.dependencies_sha256)
    }
}
pub fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequestKind {
    #[serde(rename = "generate_video")]
    GenerateVideo,
}
/// Deliberately no Debug: a request contains private authoring text.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub protocol: u32,
    pub kind: RequestKind,
    pub id: String,
    pub profile: Profile,
    pub prompt: String,
    pub seed: u64,
}
impl Request {
    pub fn valid(&self) -> bool {
        self.protocol == VERSION
            && valid_id(&self.id)
            && !self.prompt.trim().is_empty()
            && self.prompt.len() <= MAX_PROMPT_BYTES
            && self.seed == 42
            && serde_json::to_vec(self).is_ok_and(|header| header.len() <= MAX_HEADER)
    }
}
fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    LoadingText,
    EncodingText,
    LoadingDit,
    Sampling,
    LoadingVideoVae,
    DecodingVideo,
    LoadingAudioVae,
    DecodingAudio,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    DeploymentMismatch,
    ModelIdentityMismatch,
    ComponentLoadFailed,
    OutOfMemory,
    GenerationFailed,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    Ready {
        deployment: Deployment,
    },
    Phase {
        id: String,
        phase: Phase,
    },
    Progress {
        id: String,
        completed: u32,
        total: u32,
        elapsed_seconds: f64,
    },
    /// One HWC RGB8 frame using export_rgb8. Index is zero-based, increasing.
    VideoFrame {
        id: String,
        index: u32,
    },
    /// Interleaved stereo IEEE f32 LE; start/samples count frames per channel.
    AudioChunk {
        id: String,
        start: u32,
        samples: u32,
    },
    Complete {
        id: String,
        frames: u32,
        audio_samples: u32,
    },
    Error {
        id: Option<String>,
        code: ErrorCode,
    },
}
/// Product uses this before accepting any bytes. No partial result is publishable.
/// Ready is checked against the verified deployment before constructing a stream.
pub struct Stream {
    id: String,
    frames: u32,
    audio_samples: u32,
    steps: u32,
    terminal: bool,
}
impl Stream {
    pub fn new(id: String) -> Option<Self> {
        valid_id(&id).then_some(Self {
            id,
            frames: 0,
            audio_samples: 0,
            steps: 0,
            terminal: false,
        })
    }
    pub fn accept(&mut self, event: &Event, payload: &[u8]) -> Result<(), &'static str> {
        if self.terminal || payload.len() > MAX_PAYLOAD {
            return Err("h3_protocol_error");
        }
        match event {
            Event::Phase { id, .. } if id == &self.id && payload.is_empty() => {}
            Event::Progress {
                id,
                completed,
                total,
                elapsed_seconds,
            } if id == &self.id
                && payload.is_empty()
                && *total == STEPS
                && *completed > self.steps
                && *completed <= STEPS
                && elapsed_seconds.is_finite()
                && *elapsed_seconds >= 0. =>
            {
                self.steps = *completed;
            }
            Event::VideoFrame { id, index }
                if id == &self.id
                    && *index == self.frames
                    && *index < FRAMES
                    && payload.len() == FRAME_BYTES =>
            {
                self.frames += 1;
            }
            Event::AudioChunk { id, start, samples }
                if id == &self.id
                    && *start == self.audio_samples
                    && *samples > 0
                    && *samples <= AUDIO_SAMPLES - self.audio_samples
                    && payload.len() == *samples as usize * CHANNELS as usize * 4 =>
            {
                if !payload
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                {
                    return Err("h3_invalid_audio");
                }
                self.audio_samples += samples;
            }
            Event::Complete {
                id,
                frames,
                audio_samples,
            } if id == &self.id
                && payload.is_empty()
                && *frames == FRAMES
                && *audio_samples == AUDIO_SAMPLES
                && self.frames == FRAMES
                && self.audio_samples == AUDIO_SAMPLES
                && self.steps == STEPS =>
            {
                self.terminal = true;
            }
            Event::Error { id: Some(id), .. } if id == &self.id && payload.is_empty() => {
                self.terminal = true;
            }
            _ => return Err("h3_protocol_error"),
        }
        Ok(())
    }
}

/// H3 frozen viewing-export boundary: f32 clamp, scale, ties-to-even rounding.
/// This does not modify inference tensors or their numerical acceptance.
pub fn export_rgb8(pixels: &[f32]) -> Result<Vec<u8>, &'static str> {
    pixels
        .iter()
        .map(|v| {
            if !v.is_finite() {
                return Err("h3_invalid_video");
            }
            Ok((v.clamp(0., 1.) * 255.).round_ties_even() as u8)
        })
        .collect()
}
