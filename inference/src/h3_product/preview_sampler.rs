//! Timed sampler callback previews. Owns no model or latent tensor between calls.
use super::{preview, protocol};
use crate::{
    context_windows::LatentShapes, execution_value::Value,
    models::minimax_h3::streams::NestedTensor, native::Error,
};
use std::time::Instant;
use tch::Tensor;

pub struct Sampler {
    settings: protocol::preview::Settings,
    shapes: LatentShapes,
    cadence: preview::Cadence,
    started: Instant,
    fps: f64,
    sequence: u64,
    warned: bool,
}
impl Sampler {
    pub fn new(
        settings: protocol::preview::Settings,
        shapes: LatentShapes,
        fps: f64,
        sequence: u64,
    ) -> Result<Self, Error> {
        let settings = settings.effective();
        Ok(Self {
            cadence: preview::Cadence::new(settings.interval_seconds)?,
            settings,
            shapes,
            fps,
            sequence,
            started: Instant::now(),
            warned: false,
        })
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    /// Director creates this clock/cadence for each sampler invocation, including
    /// a second sampling stage. The transport sequence remains unique per job.
    pub fn observe(
        &mut self,
        packed: &Tensor,
        step: u32,
    ) -> Result<Option<(protocol::preview::Metadata, Vec<u8>)>, Error> {
        self.observe_at(packed, step, self.started.elapsed().as_secs_f64())
    }
    pub fn warn_once(&mut self) -> bool {
        !std::mem::replace(&mut self.warned, true)
    }
    fn observe_at(
        &mut self,
        packed: &Tensor,
        step: u32,
        elapsed: f64,
    ) -> Result<Option<(protocol::preview::Metadata, Vec<u8>)>, Error> {
        let Some(due) = self.cadence.due(elapsed) else {
            return Ok(None);
        };
        let streams = NestedTensor::unpack(&Value::tensor(packed.shallow_clone()), &self.shapes)?;
        let video = streams.unbind().borrow()[0].as_tensor()?;
        // Invalid/missing video is not an attempted render slot in Director.
        if video.size().len() != 5 {
            return Ok(None);
        }
        self.cadence.mark_emitted(due);
        let rendered = preview::render_fast(
            &video,
            &preview::Options {
                maximum_resolution: self.settings.max_resolution,
                preview_frames: self.settings.frames,
                source_fps: self.fps,
            },
        )?;
        self.sequence += 1;
        let metadata = protocol::preview::Metadata {
            sequence: self.sequence,
            step,
            width: rendered.width,
            height: rendered.height,
            frames: rendered.frames.len() as u32,
            frame_duration_ms: rendered.frame_duration_ms,
            source_duration_seconds: rendered.source_duration_seconds,
        };
        let pixels = rendered.frames.into_iter().flatten().collect();
        Ok(Some((metadata, pixels)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};
    #[test]
    fn packed_av_callback_preview_preserves_sampling_input_and_full_clock() {
        let video = Tensor::arange(32 * 12 * 2 * 4, (Kind::Float, Device::Cpu))
            .reshape([1, 32, 12, 2, 4])
            / 1000.;
        let audio = Tensor::ones([1, 64, 1, 8], (Kind::Float, Device::Cpu));
        let streams = NestedTensor::new(vec![Value::tensor(video), Value::tensor(audio)]);
        let (packed, shapes) = streams.pack().unwrap();
        let packed = packed.as_tensor().unwrap();
        let before = packed.copy();
        let mut sampler = Sampler::new(
            protocol::preview::Settings {
                max_resolution: 64,
                frames: 2,
                ..Default::default()
            },
            shapes,
            24.,
            7,
        )
        .unwrap();
        assert!(sampler.observe_at(&packed, 1, 9.99).unwrap().is_none());
        let (meta, rgb) = sampler.observe_at(&packed, 2, 10.).unwrap().unwrap();
        assert_eq!(meta.sequence, 8);
        assert_eq!(meta.frames, 2);
        assert!(meta.valid_payload(rgb.len()));
        assert_eq!(meta.source_duration_seconds, 39. / 24.);
        assert_eq!(meta.frame_duration_ms, 812);
        assert!(packed.equal(&before));
        assert!(sampler.observe_at(&packed, 3, 10.01).unwrap().is_none());
        assert!(sampler.observe_at(&packed, 4, 20.).unwrap().is_some());
        assert_eq!(sampler.sequence(), 9);
        assert!(sampler.warn_once());
        assert!(!sampler.warn_once());
    }
}
