//! Image-worker display events share the public job's transient RAM owner.
use crate::{
    engine::EngineState,
    h3_protocol::preview::{Metadata, Settings},
    sampling_preview::Hub,
};
use serde_json::Value;
use std::sync::Arc;

pub(super) struct Context {
    hub: Arc<Hub>,
    job: String,
    generation: u64,
    sequence: u64,
    pub settings: Settings,
}
impl Context {
    pub fn new(state: &EngineState) -> Option<Self> {
        let job = state.job_id.clone()?;
        let (generation, settings) = state.sampling_preview.image_context(&job)?;
        Some(Self {
            hub: state.sampling_preview.clone(),
            job,
            generation,
            sequence: 0,
            settings,
        })
    }
    pub fn receive(&mut self, value: Value, pixels: Vec<u8>, steps: u32) {
        let frame = serde_json::from_value::<Metadata>(value).ok().filter(|m| {
            m.frames == 1
                && m.step > 0
                && m.step <= steps
                && m.sequence > self.sequence
                && m.valid_payload(pixels.len())
                && m.width <= self.settings.max_resolution
                && m.height <= self.settings.max_resolution
        });
        let Some(metadata) = frame else {
            self.failed();
            return;
        };
        self.sequence = metadata.sequence;
        self.hub
            .publish_image(&self.job, self.generation, metadata, pixels);
    }
    pub fn failed(&self) {
        self.hub.warn_image(&self.job, self.generation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn frame(sequence: u64) -> Value {
        json!({"sequence":sequence,"step":2,"width":1,"height":1,"frames":1,"frame_duration_ms":1000,"source_duration_seconds":1.})
    }
    #[test]
    fn worker_frame_failure_clear_cancel_and_multi_pass_share_owned_public_identity() {
        let state = EngineState {
            job_id: Some("public-job".into()),
            ..Default::default()
        };
        state.sampling_preview.begin("public-job");
        let mut first = Context::new(&state).unwrap();
        first.receive(json!({"invalid":"private detail"}), vec![], 20);
        assert_eq!(
            state.sampling_preview.status().warning.as_deref(),
            Some("fast_preview_failed")
        );
        first.receive(frame(1), vec![1, 2, 3], 20);
        assert_eq!(
            state
                .sampling_preview
                .snapshot("public-job", 0)
                .unwrap()
                .metadata
                .sequence,
            1
        );
        let mut second = Context::new(&state).unwrap();
        second.receive(frame(1), vec![4, 5, 6], 20);
        assert_eq!(
            state
                .sampling_preview
                .snapshot("public-job", 1)
                .unwrap()
                .metadata
                .sequence,
            2
        );
        state.sampling_preview.clear();
        second.receive(frame(2), vec![7, 8, 9], 20);
        assert!(state.sampling_preview.snapshot("public-job", 0).is_none());
        state.sampling_preview.end("public-job");
        first.failed();
        assert!(Context::new(&state).is_none());
        assert!(state.sampling_preview.status().warning.is_none());
    }
}
