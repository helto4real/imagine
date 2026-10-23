//! Private H3 AV wire contract. Included by path by product and inference.
//! Only serde/serde_json/std: no tensor, CUDA, filesystem or async dependencies.
use serde::{Deserialize, Serialize};

// Version 13 adds disposable sampling previews, separate from final AV output.
// v15 carries trained-upscale continuation geometry. v14 readers reject that
// descriptor field; negotiate a matching worker before sending physical tails.
// v16 reports actual adaptive-solver callback progress, including total corrections.
pub const VERSION: u32 = 16;
#[path = "h3_reference_cache.rs"]
pub mod cache;
#[path = "h3_continuation.rs"]
pub mod continuation;
#[path = "h3_geometry.rs"]
pub mod geometry;
#[path = "h3_options.rs"]
pub mod options;
#[path = "h3_preview.rs"]
pub mod preview;
#[path = "h3_progress.rs"]
pub mod progress;
#[path = "h3_references.rs"]
pub mod references;
#[path = "h3_sampling.rs"]
pub mod sampling;
pub use geometry::{Selection, Shape};
pub const CORE_REVISION: &str = "40ff91114a6d71782aa1a79a76df6940546570bb";
pub const MODEL_LOCK_SHA256: &str =
    "4ee404910332151d529509d3ca6cbfa7c829d0df34109fc0e93166f3dccdcc18";
/// Checkpoint hashes are independent even when tensor headers happen to match.
pub fn profile_checkpoints() -> std::collections::BTreeMap<String, String> {
    std::collections::BTreeMap::from([
        (
            "h3-av-noref-v2".into(),
            "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779".into(),
        ),
        (
            "h3-fl2va-i2va-v1".into(),
            "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a".into(),
        ),
    ])
}
pub const MAX_HEADER: usize = 512 * 1024;
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
pub const STEPS: u32 = 20;
pub const SAMPLER: &str = "res_multistep";
pub const SCHEDULER: &str = "simple";

/// Checkpoint family; optional executable sampling is carried separately from creative inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Profile {
    #[serde(rename = "h3-short-av-noref-v1")]
    ShortAvNoReferenceV1,
    #[serde(rename = "h3-av-noref-v2")]
    AvNoReferenceV2,
    #[serde(rename = "h3-fl2va-i2va-v1")]
    ImageToAvV1,
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
            // Revisions and model-lock digests describe build provenance, not ABI.
            && is_sha256(&self.model_lock_sha256)
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preview: Option<preview::Settings>,
    #[serde(default)]
    pub options: options::Options,
    pub protocol: u32,
    pub kind: RequestKind,
    pub id: String,
    pub profile: Profile,
    pub prompt: String,
    pub seed: u64,
    #[serde(default)]
    pub shape: Shape,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<ImageInput>,
    #[serde(skip)]
    pub image_rgb: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<sampling::Sampling>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub capture_continuation: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation: Option<continuation::Input>,
    #[serde(skip)]
    pub continuation_bytes: Vec<u8>,
    #[serde(default, skip_serializing_if = "references::Input::is_empty")]
    pub references: references::Input,
}
impl Request {
    pub fn total_steps(&self) -> u32 {
        self.sampling
            .as_ref()
            .map_or(STEPS, sampling::Sampling::total_steps)
    }
    pub fn valid(&self) -> bool {
        self.valid_settings()
            && (self.profile != Profile::ImageToAvV1 || self.image.is_some())
            && (self.profile != Profile::ShortAvNoReferenceV1 || self.image.is_none())
            && self.image.as_ref().is_none_or(ImageInput::valid)
    }
    pub fn valid_settings(&self) -> bool {
        self.protocol == VERSION
            && self.preview.as_ref().is_none_or(preview::Settings::valid)
            && self.options.valid()
            && valid_id(&self.id)
            && !self.prompt.trim().is_empty()
            && self.prompt.len() <= MAX_PROMPT_BYTES
            && self.shape.valid()
            && (self.profile != Profile::ShortAvNoReferenceV1 || self.shape == Shape::default())
            && (self.profile != Profile::ShortAvNoReferenceV1
                || (self.seed == 42 && self.sampling.is_none()))
            && self
                .sampling
                .as_ref()
                .is_none_or(|s| s.valid() && s.valid_shape(self.shape))
            && self.references.valid()
            && (!self.capture_continuation || self.shape.frames >= 22)
            && (self.profile != Profile::ShortAvNoReferenceV1
                || (!self.capture_continuation && self.continuation.is_none()))
            && self.continuation.as_ref().is_none_or(|v| {
                self.profile == Profile::AvNoReferenceV2
                    && v.valid_sampling(self.shape, self.sampling.as_ref())
            })
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
    LoadingUpscaler,
    UpscalingLatents,
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
    LatentUpscaleFailed,
    LoraLoadFailed,
    UpscaleContextUnavailable,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Event {
    SamplingPreview {
        id: String,
        metadata: preview::Metadata,
    },
    PreviewWarning {
        id: String,
        code: String,
    },
    ReferenceCache {
        id: String,
        command: cache::Command,
    },
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
    Continuation {
        id: String,
        descriptor: continuation::Descriptor,
    },
    BoundaryApplied {
        id: String,
        receipt: serde_json::Value,
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
impl Event {
    pub fn payload_limit(&self) -> usize {
        if matches!(self, Self::SamplingPreview { .. }) {
            preview::MAX_BYTES
        } else {
            MAX_PAYLOAD
        }
    }
}
/// Product uses this before accepting any bytes. No partial result is publishable.
/// Ready is checked against the verified deployment before constructing a stream.
pub struct Stream {
    shape: Shape,
    id: String,
    frames: u32,
    audio_samples: u32,
    steps: u32,
    expected_steps: u32,
    terminal: bool,
    continuation: bool,
    preview_sequence: u64,
    boundary_applied: bool,
}
impl Stream {
    pub fn captured_continuation(&self) -> bool {
        self.continuation
    }
    pub fn boundary_applied(&self) -> bool {
        self.boundary_applied
    }
    pub fn new(id: String) -> Option<Self> {
        Self::with_shape(id, Shape::default())
    }
    pub fn with_shape(id: String, shape: Shape) -> Option<Self> {
        Self::with_steps(id, shape, STEPS)
    }
    pub fn with_steps(id: String, shape: Shape, expected_steps: u32) -> Option<Self> {
        (valid_id(&id) && shape.valid() && expected_steps > 0).then_some(Self {
            shape,
            id,
            frames: 0,
            audio_samples: 0,
            steps: 0,
            expected_steps,
            terminal: false,
            continuation: false,
            preview_sequence: 0,
            boundary_applied: false,
        })
    }
    pub fn accept(&mut self, event: &Event, payload: &[u8]) -> Result<(), &'static str> {
        if self.terminal || payload.len() > event.payload_limit() {
            return Err("h3_protocol_error");
        }
        match event {
            Event::BoundaryApplied { id, receipt }
                if id == &self.id
                    && !self.boundary_applied
                    && payload.is_empty()
                    && self.steps == self.expected_steps
                    && receipt["applied"] == true
                    && receipt["dependency"].is_object() =>
            {
                self.boundary_applied = true;
            }
            Event::SamplingPreview { id, metadata }
                if id == &self.id
                    && metadata.sequence > self.preview_sequence
                    && metadata.step > 0
                    && metadata.step <= self.steps
                    && metadata.valid_payload(payload.len()) =>
            {
                self.preview_sequence = metadata.sequence;
            }
            Event::PreviewWarning { id, code }
                if id == &self.id && payload.is_empty() && !code.is_empty() => {}
            Event::Continuation { id, descriptor }
                if id == &self.id
                    && !self.continuation
                    && descriptor.shape == self.shape
                    && descriptor.valid_bytes(payload)
                    && self.steps == self.expected_steps =>
            {
                self.continuation = true;
            }
            Event::ReferenceCache { id, command }
                if id == &self.id && command.valid_payload(payload.len()) => {}
            Event::Phase { id, .. } if id == &self.id && payload.is_empty() => {}
            Event::Progress {
                id,
                completed,
                total,
                elapsed_seconds,
            } if id == &self.id
                && payload.is_empty()
                && *completed > 0
                && *completed >= self.steps
                && (*completed > self.steps || *total != self.expected_steps)
                && *completed <= *total
                && elapsed_seconds.is_finite()
                && *elapsed_seconds >= 0. =>
            {
                self.steps = *completed;
                self.expected_steps = *total;
            }
            Event::VideoFrame { id, index }
                if id == &self.id
                    && *index == self.frames
                    && *index < self.shape.frames
                    && payload.len() == self.shape.frame_bytes() =>
            {
                self.frames += 1;
            }
            Event::AudioChunk { id, start, samples }
                if id == &self.id
                    && *start == self.audio_samples
                    && *samples > 0
                    && *samples <= self.shape.audio_samples() - self.audio_samples
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
                && *frames == self.shape.frames
                && *audio_samples == self.shape.audio_samples()
                && self.frames == self.shape.frames
                && self.audio_samples == self.shape.audio_samples()
                && self.steps == self.expected_steps =>
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

/// RGB8 transport only; no paths or encoded-image decoder in the worker.
pub const MAX_INPUT_IMAGE: usize = 32 * 1024 * 1024;
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageInput {
    pub width: u32,
    pub height: u32,
    pub sha256: String,
}
impl ImageInput {
    pub fn bytes(&self) -> usize {
        self.width as usize * self.height as usize * 3
    }
    pub fn valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.width <= 8192
            && self.height <= 8192
            && self.bytes() <= MAX_INPUT_IMAGE
            && is_sha256(&self.sha256)
    }
}
