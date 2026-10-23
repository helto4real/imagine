//! Native image sampling display only. Final PNG bytes never enter this decoder.
use crate::{
    h3_protocol::preview::{Metadata, Settings},
    sampling_preview::Hub,
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::Value;
use std::{
    io::Cursor,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    pass: i64,
    step: u32,
    total: u32,
}
pub(crate) struct Controller {
    hub: Arc<Hub>,
    job: String,
    started: Instant,
    last_attempt: Duration,
    seen: Option<Key>,
    observed: Option<Key>,
    generation: Option<u64>,
}
struct Attempt {
    generation: u64,
    step: u32,
    settings: Settings,
    encoded: String,
}
impl Controller {
    pub fn new(hub: Arc<Hub>, job: String) -> Self {
        Self {
            hub,
            job,
            started: Instant::now(),
            last_attempt: Duration::ZERO,
            seen: None,
            observed: None,
            generation: None,
        }
    }
    pub fn enabled(&self) -> bool {
        self.hub.image_context(&self.job).is_some()
    }
    fn prepare(&mut self, status: &Value, elapsed: Duration) -> Option<Attempt> {
        if status["status"] != "generating" || !status["preview"].is_object() {
            return None;
        }
        let preview = &status["preview"];
        let key = (|| {
            Some(Key {
                pass: preview["pass"].as_i64()?,
                step: u32::try_from(preview["step"].as_u64()?).ok()?,
                total: u32::try_from(preview["total_steps"].as_u64()?).ok()?,
            })
        })();
        // Observe labels even while disabled, so re-enable/clear cannot replay a
        // previously polled native callback. A later callback remains eligible.
        let observed = self.observed;
        self.observed = key;
        let Some((generation, settings)) = self.hub.image_context(&self.job) else {
            self.seen = key;
            return None;
        };
        if self
            .generation
            .is_some_and(|previous| previous != generation)
        {
            self.seen = observed;
        }
        self.generation = Some(generation);
        if key.is_some() && self.seen == key {
            return None;
        }
        if elapsed.saturating_sub(self.last_attempt).as_secs_f64() < settings.interval_seconds {
            return None;
        }
        self.last_attempt = elapsed;
        self.seen = key;
        let Some((key, encoded)) = key.zip(preview["b64_json"].as_str()) else {
            self.hub.warn_image(&self.job, generation);
            return None;
        };
        Some(Attempt {
            generation,
            step: key.step,
            settings,
            encoded: encoded.to_owned(),
        })
    }
    pub async fn update(&mut self, status: &Value) {
        self.update_at(status, self.started.elapsed()).await;
    }
    async fn update_at(&mut self, status: &Value, elapsed: Duration) {
        let Some(attempt) = self.prepare(status, elapsed) else {
            return;
        };
        let generation = attempt.generation;
        match tokio::task::spawn_blocking(move || render(attempt)).await {
            Ok(Ok((metadata, bytes))) => {
                self.hub
                    .publish_image(&self.job, generation, metadata, bytes);
            }
            _ => self.hub.warn_image(&self.job, generation),
        }
    }
}
fn render(attempt: Attempt) -> Result<(Metadata, Vec<u8>), ()> {
    if attempt.encoded.len() > crate::h3_protocol::MAX_INPUT_IMAGE.div_ceil(3) * 4 {
        return Err(());
    }
    let bytes = STANDARD.decode(attempt.encoded).map_err(|_| ())?;
    if bytes.len() > crate::h3_protocol::MAX_INPUT_IMAGE {
        return Err(());
    }
    // The native callback's wire format is PNG. ImageReader retains its existing
    // decoder allocation limits; the output uses the shared preview dimensions.
    let mut reader = image::ImageReader::with_format(Cursor::new(bytes), image::ImageFormat::Png);
    // Same image allocation/dimension bounds as storage::inspect_image.
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(8192);
    limits.max_image_height = Some(8192);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let image = reader.decode().map_err(|_| ())?;
    if u64::from(image.width()) * u64::from(image.height()) > 16_777_216 {
        return Err(());
    }
    let limit = attempt.settings.effective().max_resolution;
    let image = if image.width() > limit || image.height() > limit {
        image.resize(limit, limit, image::imageops::FilterType::Triangle)
    } else {
        image
    };
    // This is a static image, not a video-duration/fps estimate.
    let metadata = Metadata {
        sequence: 0,
        step: attempt.step,
        width: image.width(),
        height: image.height(),
        frames: 1,
        frame_duration_ms: 1000,
        source_duration_seconds: 1.,
    };
    let rgb = image.into_rgb8().into_raw();
    if metadata.bytes() != Some(rgb.len()) {
        return Err(());
    }
    Ok((metadata, rgb))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn status(step: u32, encoded: &str) -> Value {
        serde_json::json!({"status":"generating","preview":{"pass":0,"step":step,"total_steps":20,"b64_json":encoded}})
    }
    fn png() -> String {
        STANDARD.encode(
            crate::workflow_image::encode(&image::DynamicImage::ImageRgb8(
                image::RgbImage::from_pixel(80, 40, image::Rgb([30, 80, 150])),
            ))
            .unwrap(),
        )
    }
    #[tokio::test]
    async fn cadence_precedes_decode_and_failures_do_not_fail_generation() {
        let hub = Arc::new(Hub::default());
        hub.begin("public-job");
        let mut controller = Controller::new(hub.clone(), "public-job".into());
        controller
            .update_at(&status(1, "invalid base64"), Duration::from_secs(9))
            .await;
        assert!(hub.status().warning.is_none());
        controller
            .update_at(&status(1, "invalid base64"), Duration::from_secs(10))
            .await;
        assert_eq!(hub.status().warning.as_deref(), Some("fast_preview_failed"));
        controller
            .update_at(&status(3, &png()), Duration::from_secs(20))
            .await;
        let clip = hub.snapshot("public-job", 0).unwrap();
        assert_eq!(
            (
                clip.metadata.width,
                clip.metadata.height,
                clip.metadata.frames,
                clip.metadata.sequence
            ),
            (80, 40, 1, 1)
        );
        assert_eq!(&clip.bytes[..3], [30, 80, 150]);
        assert!(hub.status().warning.is_none());
    }
    #[tokio::test]
    async fn image_passes_share_sequence_and_clear_disable_and_obsolete_tickets_cannot_republish() {
        let hub = Arc::new(Hub::default());
        hub.configure(Settings {
            max_resolution: 32,
            ..Settings::default()
        })
        .unwrap();
        hub.begin("public-job");
        let source = png();
        let mut first = Controller::new(hub.clone(), "public-job".into());
        first
            .update_at(&status(4, &source), Duration::from_secs(10))
            .await;
        assert_eq!(hub.snapshot("public-job", 0).unwrap().metadata.width, 32);
        let mut next = Controller::new(hub.clone(), "public-job".into());
        next.update_at(&status(4, &source), Duration::from_secs(10))
            .await;
        assert_eq!(hub.snapshot("public-job", 1).unwrap().metadata.sequence, 2);
        let old = next
            .prepare(&status(5, &source), Duration::from_secs(20))
            .unwrap();
        hub.clear();
        let generation = old.generation;
        let (metadata, bytes) = render(old).unwrap();
        assert!(!hub.publish_image("public-job", generation, metadata, bytes));
        next.update_at(&status(5, &source), Duration::from_secs(30))
            .await;
        assert!(hub.snapshot("public-job", 0).is_none());
        hub.configure(Settings {
            enabled: false,
            ..hub.settings()
        })
        .unwrap();
        next.update_at(&status(6, &source), Duration::from_secs(40))
            .await;
        hub.configure(Settings {
            enabled: true,
            ..hub.settings()
        })
        .unwrap();
        next.update_at(&status(6, &source), Duration::from_secs(50))
            .await;
        assert!(hub.snapshot("public-job", 0).is_none());
        next.update_at(&status(7, &source), Duration::from_secs(60))
            .await;
        assert_eq!(hub.snapshot("public-job", 2).unwrap().metadata.sequence, 3);
        let old = next
            .prepare(&status(8, &source), Duration::from_secs(70))
            .unwrap();
        let generation = old.generation;
        hub.begin("new-public-job");
        let (metadata, bytes) = render(old).unwrap();
        assert!(!hub.publish_image("public-job", generation, metadata, bytes));
        hub.warn_image("public-job", generation);
        assert!(hub.status().warning.is_none());
        hub.end("new-public-job");
        assert!(hub.status().active_job.is_none());
    }
}
