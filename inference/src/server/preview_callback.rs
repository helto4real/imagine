//! Best-effort image display callbacks; sampler inputs and RNG are untouched.
use super::{Event, Observer, image_preview, preview};
use crate::native::Error;
use std::time::{Duration, Instant};
use tch::Tensor;

pub(super) struct State {
    settings: Option<preview::Settings>,
    started: Instant,
    attempted: Duration,
    sequence: u64,
}
impl State {
    pub fn new(settings: Option<preview::Settings>) -> Self {
        Self {
            settings: settings
                .filter(|s| s.enabled && s.valid())
                .map(|s| s.effective()),
            started: Instant::now(),
            attempted: Duration::ZERO,
            sequence: 0,
        }
    }
    pub fn update(
        &mut self,
        model: &str,
        step: i64,
        denoised: &Tensor,
        observer: &mut Observer<'_>,
    ) -> Result<(), Error> {
        self.update_at(self.started.elapsed(), step, observer, |edge| {
            image_preview::render(model, denoised, edge)
        })
    }
    fn update_at(
        &mut self,
        elapsed: Duration,
        step: i64,
        observer: &mut Observer<'_>,
        render: impl FnOnce(u32) -> Result<(u32, u32, Vec<u8>), Error>,
    ) -> Result<(), Error> {
        let Some(settings) = self.settings else {
            return Ok(());
        };
        if elapsed.saturating_sub(self.attempted).as_secs_f64() < settings.interval_seconds {
            return Ok(());
        }
        self.attempted = elapsed;
        // Rendering failures cannot change the generation result. Broken output
        // transport still propagates, allowing the owned worker to terminate.
        let frame = render(settings.max_resolution)
            .ok()
            .and_then(|(width, height, pixels)| {
                let metadata = preview::Metadata {
                    sequence: self.sequence.checked_add(1)?,
                    step: u32::try_from(step).ok()?,
                    width,
                    height,
                    frames: 1,
                    frame_duration_ms: 1000,
                    source_duration_seconds: 1.,
                };
                metadata
                    .valid_payload(pixels.len())
                    .then_some((metadata, pixels))
            });
        match frame {
            Some((metadata, pixels)) => {
                self.sequence = metadata.sequence;
                observer(Event::Preview { metadata, pixels })
            }
            None => observer(Event::PreviewWarning),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_and_not_due_never_render_and_failure_keeps_later_callbacks_alive() {
        let mut captured = vec![];
        let mut observer = |event| {
            captured.push(event);
            Ok(())
        };
        let mut state = State::new(None);
        state
            .update_at(Duration::from_secs(60), 1, &mut observer, |_| {
                panic!("disabled")
            })
            .unwrap();
        let mut state = State::new(Some(preview::Settings {
            interval_seconds: 2.,
            max_resolution: 1,
            ..Default::default()
        }));
        state
            .update_at(Duration::from_secs(1), 1, &mut observer, |_| {
                panic!("not due")
            })
            .unwrap();
        state
            .update_at(Duration::from_secs(2), 2, &mut observer, |_| {
                Err(Error::Native("private renderer detail".into()))
            })
            .unwrap();
        state
            .update_at(Duration::from_secs(3), 3, &mut observer, |_| {
                panic!("failed attempt still throttles")
            })
            .unwrap();
        state
            .update_at(Duration::from_secs(4), 4, &mut observer, |edge| {
                assert_eq!(edge, 1);
                Ok((1, 1, vec![1, 2, 3]))
            })
            .unwrap();
        assert_eq!(captured.len(), 2);
        assert!(matches!(captured[0], Event::PreviewWarning));
        assert!(
            matches!(&captured[1], Event::Preview { metadata, pixels } if metadata.sequence == 1 && metadata.step == 4 && pixels == &[1,2,3])
        );
    }
}
