//! Ordered project reference presentation and raw media, independent of tensors.
use serde::{Deserialize, Serialize};
// Use the existing product raw-media memory/framing bound, including video batches.
pub const MAX_BYTES: usize = super::geometry::MAX_RAW_BYTES as usize;
fn standard() -> String {
    "Standard".into()
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Input {
    pub quality: String,
    pub manifest: serde_json::Value,
    pub images: Vec<Image>,
    pub videos: Vec<Video>,
    pub audios: Vec<Audio>,
}
impl Default for Input {
    fn default() -> Self {
        Self {
            quality: standard(),
            manifest: serde_json::Value::Null,
            images: vec![],
            videos: vec![],
            audios: vec![],
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    #[serde(skip)]
    pub rgb: Vec<u8>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Audio {
    pub sample_rate: u32,
    pub samples: u32,
    #[serde(skip)]
    pub pcm: Vec<u8>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Video {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub source_fps: f64,
    #[serde(default)]
    pub tail_frames: Option<u32>,
    pub audio: Option<Audio>,
    #[serde(skip)]
    pub rgb: Vec<u8>,
}
impl Video {
    pub fn rgb_bytes(&self) -> usize {
        (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(self.frames as usize)
            .saturating_mul(3)
    }
    pub fn bytes(&self) -> usize {
        self.rgb_bytes()
            .saturating_add(self.audio.as_ref().map_or(0, Audio::bytes))
    }
    fn valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.frames > 0
            && self.source_fps.is_finite()
            && self.source_fps > 0.
            && self.bytes() <= MAX_BYTES
            && self.audio.as_ref().is_none_or(Audio::valid)
    }
}
impl Image {
    pub fn bytes(&self) -> usize {
        (self.width as usize)
            .saturating_mul(self.height as usize)
            .saturating_mul(3)
    }
    fn valid(&self) -> bool {
        self.width > 0 && self.height > 0 && self.bytes() <= super::MAX_INPUT_IMAGE
    }
}
impl Audio {
    pub fn bytes(&self) -> usize {
        (self.samples as usize).saturating_mul(8)
    }
    fn valid(&self) -> bool {
        self.sample_rate == 44_100 && self.samples > 0 && self.bytes() <= MAX_BYTES
    }
    fn valid_payload(&self) -> bool {
        self.valid()
            && self.pcm.len() == self.bytes()
            && self
                .pcm
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
    }
}
impl Input {
    pub fn boundary_receipt(&self) -> Option<serde_json::Value> {
        self.manifest["boundary_dependency"].is_object().then(|| {
            serde_json::json!({
                "dependency":self.manifest["boundary_dependency"],
                "selection":self.manifest["boundary_selection"],"applied":true
            })
        })
    }
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
            && self.videos.is_empty()
            && self.audios.is_empty()
            && self.manifest.is_null()
    }
    pub fn bytes(&self) -> usize {
        self.images
            .iter()
            .map(Image::bytes)
            .chain(self.videos.iter().map(Video::bytes))
            .chain(self.audios.iter().map(Audio::bytes))
            .fold(0, usize::saturating_add)
    }
    pub fn valid(&self) -> bool {
        self.bytes() <= MAX_BYTES
            && self.images.iter().all(Image::valid)
            && self.videos.iter().all(Video::valid)
            && self.audios.iter().all(Audio::valid)
    }
    pub fn valid_payload(&self) -> bool {
        self.valid()
            && self.images.iter().all(|i| i.rgb.len() == i.bytes())
            && self.videos.iter().all(|v| {
                v.rgb.len() == v.rgb_bytes() && v.audio.as_ref().is_none_or(Audio::valid_payload)
            })
            && self.audios.iter().all(|a| {
                a.pcm.len() == a.bytes()
                    && a.pcm
                        .chunks_exact(4)
                        .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite())
            })
    }
}
