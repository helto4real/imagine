//! Session-bound, RAM-only sampling preview transport. RGB never enters JSON or Debug.
use super::Client;
use crate::h3_protocol::preview::MAX_BYTES;
pub use crate::h3_protocol::preview::{Metadata, Settings};

#[derive(Clone, serde::Deserialize)]
pub struct Status {
    pub settings: Settings,
    pub effective: Settings,
    pub active_job: Option<String>,
    pub metadata: Option<Metadata>,
    pub warning: Option<String>,
}

/// Exactly one complete silent animation; no file path or durable media identity.
pub struct Clip {
    pub metadata: Metadata,
    rgb: Vec<u8>,
}
impl std::fmt::Debug for Clip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplingPreview")
            .field("metadata", &self.metadata)
            .field("rgb", &"<transient pixels>")
            .finish()
    }
}
impl Clip {
    pub fn new(metadata: Metadata, rgb: Vec<u8>) -> Result<Self, String> {
        if !metadata.valid_payload(rgb.len()) {
            return Err("The sampling preview has an invalid frame layout.".into());
        }
        Ok(Self { metadata, rgb })
    }
    pub fn frame(&self, index: usize) -> Option<&[u8]> {
        if index >= self.metadata.frames as usize {
            return None;
        }
        let bytes = (self.metadata.width as usize)
            .checked_mul(self.metadata.height as usize)?
            .checked_mul(3)?;
        let start = index.checked_mul(bytes)?;
        self.rgb.get(start..start.checked_add(bytes)?)
    }
}

fn invalid_payload() -> String {
    "The sampling preview response is incomplete or invalid.".into()
}
fn transport_error(error: reqwest::Error) -> String {
    if error.is_timeout() {
        "Sampling preview timed out; generation continues.".into()
    } else if error.is_connect() {
        "Sampling preview could not connect to the server.".into()
    } else {
        "Sampling preview was interrupted; generation continues.".into()
    }
}

// Metadata cannot carry pixels. Its bound uses the existing transport memory bound,
// rather than a new user setting or a restriction on generated video dimensions.
const MAX_WIRE_BYTES: usize = 4 + MAX_BYTES + MAX_BYTES;
fn decode_wire(mut bytes: Vec<u8>) -> Result<Clip, String> {
    let prefix: [u8; 4] = bytes
        .get(..4)
        .ok_or_else(invalid_payload)?
        .try_into()
        .map_err(|_| invalid_payload())?;
    let metadata_len = u32::from_be_bytes(prefix) as usize;
    let end = 4usize
        .checked_add(metadata_len)
        .ok_or_else(invalid_payload)?;
    if metadata_len > MAX_BYTES || bytes.len() > MAX_WIRE_BYTES {
        return Err(invalid_payload());
    }
    let metadata: Metadata = serde_json::from_slice(bytes.get(4..end).ok_or_else(invalid_payload)?)
        .map_err(|_| invalid_payload())?;
    let rgb_len = bytes.len().checked_sub(end).ok_or_else(invalid_payload)?;
    if !metadata.valid_payload(rgb_len) {
        return Err(invalid_payload());
    }
    let rgb = bytes.split_off(end);
    Clip::new(metadata, rgb)
}

impl Client {
    pub async fn sampling_preview_status(&self, session: &str) -> Result<Status, String> {
        let value = self
            .timeline_request(session, false, "/v1/runtime/sampling-preview", None)
            .await
            .map_err(|error| error.message)?;
        serde_json::from_value(value).map_err(|_| invalid_payload())
    }
    pub async fn sampling_preview_settings(
        &self,
        session: &str,
        settings: &Settings,
    ) -> Result<Status, String> {
        let value = serde_json::to_value(settings).map_err(|_| {
            "Preview interval must be a finite number greater than zero.".to_owned()
        })?;
        let value = self
            .timeline_request(session, true, "/v1/runtime/sampling-preview", Some(&value))
            .await
            .map_err(|error| error.message)?;
        serde_json::from_value(value).map_err(|_| invalid_payload())
    }
    pub async fn sampling_preview_frame(
        &self,
        session: &str,
        job: &str,
        after: u64,
    ) -> Result<Option<Clip>, String> {
        // Identifiers use the same admitted-job representation as the Timeline.
        super::identifier(&serde_json::json!(job)).map_err(|_| invalid_payload())?;
        let mut response = self
            .get(&format!("/v1/jobs/{job}/sampling-preview?after={after}"))
            .header("X-Imagine-Session", session)
            .header("Accept", "application/octet-stream")
            .header("Cache-Control", "no-store")
            .send()
            .await
            .map_err(transport_error)?;
        if response.status() == reqwest::StatusCode::NO_CONTENT {
            return Ok(None);
        }
        if !response.status().is_success() {
            return Err("Sampling preview is unavailable; generation continues.".into());
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next());
        if content_type != Some("application/octet-stream") {
            return Err(invalid_payload());
        }
        if response
            .content_length()
            .is_some_and(|len| len > MAX_WIRE_BYTES as u64)
        {
            return Err(invalid_payload());
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
            if bytes.len().saturating_add(chunk.len()) > MAX_WIRE_BYTES {
                return Err(invalid_payload());
            }
            bytes.extend_from_slice(&chunk);
        }
        decode_wire(bytes).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata() -> Metadata {
        Metadata {
            sequence: 1,
            step: 0,
            width: 2,
            height: 1,
            frames: 2,
            frame_duration_ms: 500,
            source_duration_seconds: 1.,
        }
    }
    fn wire(metadata: &Metadata, rgb: &[u8]) -> Vec<u8> {
        let json = serde_json::to_vec(metadata).unwrap();
        let mut bytes = (json.len() as u32).to_be_bytes().to_vec();
        bytes.extend(json);
        bytes.extend(rgb);
        bytes
    }
    #[test]
    fn sampling_preview_wire_preserves_all_frames_and_redacts_debug() {
        let rgb = [17, 29, 83, 49, 117, 201, 77, 92, 109, 39, 33, 47];
        let clip = decode_wire(wire(&metadata(), &rgb)).unwrap();
        assert_eq!(clip.frame(0), Some(rgb[..6].as_ref()));
        assert_eq!(clip.frame(1), Some(rgb[6..].as_ref()));
        assert!(clip.frame(2).is_none());
        let debug = format!("{clip:?}");
        assert!(debug.contains("<transient pixels>"));
        assert!(!debug.contains("117, 201"));
    }
    #[test]
    fn sampling_preview_wire_rejects_incomplete_and_wrong_layout() {
        for bytes in [
            vec![],
            vec![0, 0, 0],
            vec![255; 4],
            wire(&metadata(), &[0; 11]),
        ] {
            assert!(decode_wire(bytes).is_err());
        }
        let mut invalid = metadata();
        invalid.frames = 0;
        assert!(decode_wire(wire(&invalid, &[])).is_err());
    }
    #[test]
    fn sampling_preview_settings_preserve_requested_runtime_clamps() {
        let settings = Settings {
            max_resolution: 1234,
            frames: 123,
            ..Settings::default()
        };
        let status: Status = serde_json::from_value(serde_json::json!({
            "settings": settings, "effective": settings.effective(),
            "active_job": null, "metadata": null, "warning": null
        }))
        .unwrap();
        assert_eq!(status.settings, settings);
        assert_eq!(status.effective.max_resolution, 768);
        assert_eq!(status.effective.frames, 48);
    }
}
