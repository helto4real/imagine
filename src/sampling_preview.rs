//! Latest-only sampling preview owner. No files, source text or persistent assets.
use crate::h3_protocol::preview::{Metadata, Settings};
use serde::Serialize;
use std::{
    fmt,
    sync::{Arc, Mutex, MutexGuard},
};

/// Intentionally no Debug or Serialize: pixels must not enter diagnostics/JSON.
pub struct Clip {
    pub metadata: Metadata,
    pub bytes: Vec<u8>,
}
#[derive(Default)]
struct State {
    settings: Settings,
    active_job: Option<String>,
    sequence: u64,
    image_generation: u64,
    latest: Option<Arc<Clip>>,
    warning: Option<String>,
}
#[derive(Default)]
pub struct Hub {
    state: Mutex<State>,
}
impl fmt::Debug for Hub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("SamplingPreviewHub")
            .field("settings", &state.settings)
            .field("active", &state.active_job.is_some())
            .field("has_clip", &state.latest.is_some())
            .field("has_warning", &state.warning.is_some())
            .finish()
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub settings: Settings,
    pub effective: Settings,
    pub active_job: Option<String>,
    pub metadata: Option<Metadata>,
    pub warning: Option<String>,
}
impl Hub {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
    pub fn settings(&self) -> Settings {
        self.lock().settings
    }
    pub fn configure(&self, settings: Settings) -> Result<Settings, &'static str> {
        if !settings.valid() {
            return Err("invalid_sampling_preview_settings");
        }
        let mut state = self.lock();
        if state.settings != settings {
            state.image_generation = state.image_generation.wrapping_add(1);
        }
        state.settings = settings;
        if !settings.enabled {
            state.latest = None;
            state.warning = None;
        }
        Ok(settings)
    }
    /// Public scheduler job ID, not a possibly different native request ID.
    pub fn begin(&self, job_id: &str) -> Settings {
        let mut state = self.lock();
        state.image_generation = state.image_generation.wrapping_add(1);
        state.active_job = Some(job_id.to_owned());
        state.sequence = 0;
        state.latest = None;
        state.warning = None;
        state.settings.effective()
    }
    pub fn end(&self, job_id: &str) {
        let mut state = self.lock();
        if state.active_job.as_deref() == Some(job_id) {
            state.image_generation = state.image_generation.wrapping_add(1);
            state.active_job = None;
            state.sequence = 0;
            state.latest = None;
            state.warning = None;
        }
    }
    /// Release current data but retain the active job's sequence high-water mark.
    pub fn clear(&self) {
        let mut state = self.lock();
        state.image_generation = state.image_generation.wrapping_add(1);
        state.latest = None;
        state.warning = None;
    }
    /// Invalid/stale previews are ignored; a preview failure is never a job failure.
    pub fn publish(&self, job_id: &str, metadata: Metadata, bytes: Vec<u8>) -> bool {
        if !metadata.valid_payload(bytes.len()) {
            return false;
        }
        let mut state = self.lock();
        if !state.settings.enabled
            || state.active_job.as_deref() != Some(job_id)
            || metadata.sequence <= state.sequence
        {
            return false;
        }
        state.sequence = metadata.sequence;
        state.latest = Some(Arc::new(Clip { metadata, bytes }));
        state.warning = None;
        true
    }
    /// Ticket for an in-flight native image preview, invalidated by clear/settings/job changes.
    pub(crate) fn image_context(&self, job_id: &str) -> Option<(u64, Settings)> {
        let state = self.lock();
        (state.settings.enabled && state.active_job.as_deref() == Some(job_id))
            .then(|| (state.image_generation, state.settings.effective()))
    }
    /// Allocate sequence at publication, across all image passes of one public job.
    pub(crate) fn publish_image(
        &self,
        job_id: &str,
        generation: u64,
        mut metadata: Metadata,
        bytes: Vec<u8>,
    ) -> bool {
        let mut state = self.lock();
        if !state.settings.enabled
            || state.active_job.as_deref() != Some(job_id)
            || state.image_generation != generation
        {
            return false;
        }
        let Some(sequence) = state.sequence.checked_add(1) else {
            return false;
        };
        metadata.sequence = sequence;
        if !metadata.valid_payload(bytes.len()) {
            return false;
        }
        state.sequence = sequence;
        state.latest = Some(Arc::new(Clip { metadata, bytes }));
        state.warning = None;
        true
    }
    pub(crate) fn warn_image(&self, job_id: &str, generation: u64) {
        let mut state = self.lock();
        if state.settings.enabled
            && state.active_job.as_deref() == Some(job_id)
            && state.image_generation == generation
        {
            state.warning = Some("fast_preview_failed".into());
        }
    }
    /// Code only (e.g. fast_preview_failed), never a renderer exception/prompt/path.
    pub fn warn(&self, job_id: &str, code: &str) {
        let mut state = self.lock();
        if state.settings.enabled && state.active_job.as_deref() == Some(job_id) {
            state.warning = Some(code.to_owned());
        }
    }
    pub fn snapshot(&self, job_id: &str, after: u64) -> Option<Arc<Clip>> {
        let state = self.lock();
        if !state.settings.enabled || state.active_job.as_deref() != Some(job_id) {
            return None;
        }
        state
            .latest
            .as_ref()
            .filter(|clip| clip.metadata.sequence > after)
            .cloned()
    }
    pub fn status(&self) -> Status {
        let state = self.lock();
        Status {
            settings: state.settings,
            effective: state.settings.effective(),
            active_job: state.active_job.clone(),
            metadata: state.latest.as_ref().map(|clip| clip.metadata.clone()),
            warning: state.warning.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata(sequence: u64) -> Metadata {
        Metadata {
            sequence,
            step: 0,
            width: 1,
            height: 1,
            frames: 2,
            frame_duration_ms: 500,
            source_duration_seconds: 1.,
        }
    }
    #[test]
    fn latest_only_releases_old_clip_and_clear_keeps_monotonicity() {
        let hub = Hub::default();
        assert!(hub.begin("own-a").enabled);
        assert!(hub.publish("own-a", metadata(1), vec![1; 6]));
        let old = Arc::downgrade(&hub.snapshot("own-a", 0).unwrap());
        assert!(hub.publish("own-a", metadata(2), vec![2; 6]));
        assert!(old.upgrade().is_none());
        assert!(hub.snapshot("own-a", 2).is_none());
        assert!(!hub.publish("own-a", metadata(2), vec![3; 6]));
        hub.clear();
        assert!(hub.snapshot("own-a", 0).is_none());
        assert!(!hub.publish("own-a", metadata(1), vec![4; 6]));
        assert!(hub.publish("own-a", metadata(3), vec![5; 6]));
    }
    #[test]
    fn obsolete_jobs_warnings_and_disable_do_not_retain_pixels() {
        let hub = Hub::default();
        let settings = Settings {
            interval_seconds: 3.,
            ..Settings::default()
        };
        hub.configure(settings).unwrap();
        hub.begin("own-a");
        hub.warn("own-a", "fast_preview_failed");
        assert_eq!(hub.status().warning.as_deref(), Some("fast_preview_failed"));
        hub.begin("own-b");
        hub.end("own-a");
        assert!(!hub.publish("own-a", metadata(10), vec![0; 6]));
        hub.warn("own-a", "obsolete");
        assert!(hub.status().warning.is_none());
        assert_eq!(hub.settings(), settings);
        assert!(hub.publish("own-b", metadata(1), vec![7; 6]));
        let weak = Arc::downgrade(&hub.snapshot("own-b", 0).unwrap());
        hub.configure(Settings {
            enabled: false,
            ..settings
        })
        .unwrap();
        assert!(weak.upgrade().is_none());
        assert!(hub.snapshot("own-b", 0).is_none());
        assert!(!hub.publish("own-b", metadata(2), vec![8; 6]));
        assert!(
            hub.configure(Settings {
                interval_seconds: 0.,
                ..settings
            })
            .is_err()
        );
        hub.end("own-b");
        assert!(hub.status().active_job.is_none());
        assert!(!format!("{hub:?}").contains("own-b"));
    }
}
