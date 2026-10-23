use super::{
    dsp::{self, Stereo},
    *,
};
use std::{collections::VecDeque, fs::File, io, sync::atomic::AtomicBool};
const RAW_LIMIT: u64 = crate::video::media::MAX_UPLOAD;
fn selected_range(clip: &ClipPlan, count: usize) -> (usize, usize, f64, f64) {
    let fps = clip.source.fps.map(seconds).unwrap_or(24.);
    let duration = count as f64 / fps;
    let start = seconds(clip.source_in).min(duration);
    let end = clip
        .source_out
        .map(seconds)
        .unwrap_or(duration)
        .min(duration);
    (
        (start * fps).floor() as usize,
        ((end * fps).ceil() as usize).min(count),
        start,
        end,
    )
}
fn decode_video(
    file: &File,
    clip: &ClipPlan,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, &'static str> {
    let bytes = media::decode(
        file,
        &[
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
        ],
        RAW_LIMIT,
        cancel,
    )?;
    let size = clip.source.width as usize * clip.source.height as usize * 3;
    if size == 0 || !bytes.len().is_multiple_of(size) || bytes.is_empty() {
        return Err("invalid_video_frames");
    }
    Ok(bytes)
}
pub fn fit_frame(
    bytes: &[u8],
    width: u32,
    height: u32,
    delivery: &crate::video::contract::DeliverySettings,
) -> Vec<f32> {
    if width == delivery.width && height == delivery.height {
        return bytes.iter().map(|v| *v as f32 / 255.).collect();
    }
    resize::pad(bytes, width, height, delivery.width, delivery.height)
        .into_iter()
        .map(|v| v as f32 / 255.)
        .collect()
}
struct Decoded {
    bytes: Vec<u8>,
    clip: ClipPlan,
    start: usize,
    count: usize,
    cursor: u64,
    correction: Vec<Vec<f32>>,
}
impl Decoded {
    fn frame(&self, index: u64, delivery: &crate::video::contract::DeliverySettings) -> Vec<f32> {
        let size = self.clip.source.width as usize * self.clip.source.height as usize * 3;
        let source = self.start + fit_index(index, self.count as u64, self.clip.frames) as usize;
        fit_frame(
            &self.bytes[source * size..(source + 1) * size],
            self.clip.source.width,
            self.clip.source.height,
            delivery,
        )
    }
    fn next(&mut self, delivery: &crate::video::contract::DeliverySettings) -> Option<Vec<f32>> {
        if self.cursor >= self.clip.frames {
            return None;
        }
        let frame = if (self.cursor as usize) < self.correction.len() {
            std::mem::take(&mut self.correction[self.cursor as usize])
        } else {
            self.frame(self.cursor, delivery)
        };
        self.cursor += 1;
        Some(frame)
    }
}
struct Frames {
    frozen: Frozen,
    cancel: std::sync::Arc<AtomicBool>,
    progress: Progress,
    error: std::sync::Arc<std::sync::Mutex<Option<&'static str>>>,
    clip_index: usize,
    current: Option<Decoded>,
    tail: VecDeque<Vec<f32>>,
    produced: u64,
    failed: bool,
}
impl Frames {
    fn load(&mut self) -> Result<bool, &'static str> {
        let Some(clip) = self.frozen.plan.clips.get(self.clip_index).cloned() else {
            return Ok(false);
        };
        let file = &self.frozen.files[&clip.source.resource_id];
        self.progress.set(
            "decoding_clip",
            self.clip_index as u64,
            self.frozen.plan.clips.len() as u64,
        );
        let bytes = decode_video(file, &clip, &self.cancel)?;
        let count = bytes.len() / (clip.source.width as usize * clip.source.height as usize * 3);
        let (start, end, _, _) = selected_range(&clip, count);
        if end <= start {
            return Err("assembly_empty_source_range");
        }
        let mut decoded = Decoded {
            bytes,
            clip: clip.clone(),
            start,
            count: end - start,
            cursor: 0,
            correction: vec![],
        };
        let n = if clip.boundary == BoundaryMode::ContinuousShot
            && self.frozen.plan.options.continuous_tone_match
        {
            clip.frames.min(
                seconds(self.frozen.plan.delivery.fps)
                    .round_ties_even()
                    .max(3.) as u64,
            )
        } else {
            0
        };
        if n > 0 {
            let mut first = (0..n)
                .map(|i| decoded.frame(i, &self.frozen.plan.delivery))
                .collect::<Vec<_>>();
            let left = self.tail.iter().cloned().collect::<Vec<_>>();
            dsp::tone_match(&left, &mut first, seconds(self.frozen.plan.delivery.fps));
            decoded.correction = first;
        }
        if clip.overlap > 0 {
            let split = self
                .tail
                .len()
                .checked_sub(clip.overlap as usize)
                .ok_or("assembly_blend_exceeds_clip")?;
            let mut previous = self.tail.split_off(split);
            for i in 0..clip.overlap {
                let mut left = previous.pop_front().ok_or("assembly_blend_exceeds_clip")?;
                let right = decoded
                    .next(&self.frozen.plan.delivery)
                    .ok_or("assembly_blend_exceeds_clip")?;
                dsp::blend(&mut left, &right, i, clip.overlap);
                self.tail.push_back(left);
            }
        }
        self.current = Some(decoded);
        self.clip_index += 1;
        Ok(true)
    }
    fn next_frame(&mut self) -> Result<Option<Vec<f32>>, &'static str> {
        loop {
            if cancelled(&self.cancel) {
                return Err("media_process_cancelled");
            }
            if self.current.is_none() && !self.load()? {
                return Ok(self.tail.pop_front());
            }
            // A later seam can reach across a one-frame clip into an earlier clip.
            // Keep the largest requested tail, including already blended frames.
            let keep = self
                .frozen
                .plan
                .clips
                .iter()
                .map(|c| c.overlap as usize)
                .max()
                .unwrap_or(0)
                .max(3);
            if self.tail.len() > keep {
                return Ok(self.tail.pop_front());
            }
            let current = self.current.as_mut().unwrap();
            if let Some(frame) = current.next(&self.frozen.plan.delivery) {
                self.tail.push_back(frame);
            } else {
                self.current = None;
            }
        }
    }
}
impl Iterator for Frames {
    type Item = io::Result<Vec<f32>>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        match self.next_frame() {
            Ok(Some(frame)) => {
                self.produced += 1;
                self.progress
                    .set("encoding_video", self.produced, self.frozen.plan.frames);
                Some(Ok(frame))
            }
            Ok(None) => None,
            Err(code) => {
                self.failed = true;
                *self.error.lock().unwrap() = Some(code);
                Some(Err(io::Error::other(code)))
            }
        }
    }
}
fn audio_slice(samples: &[Stereo], start: f64, end: f64) -> &[Stereo] {
    let a = (round_samples(start, 44100.) as usize).min(samples.len());
    let b = (round_samples(end, 44100.) as usize)
        .min(samples.len())
        .max(a);
    &samples[a..b]
}
fn mix(frozen: &Frozen, cancel: &AtomicBool, progress: &Progress) -> Result<Vec<u8>, &'static str> {
    let plan = &frozen.plan;
    let fps = seconds(plan.delivery.fps);
    let mut generated = vec![[0.; 2]; plan.samples as usize];
    let mut timeline = generated.clone();
    if plan.options.audio_mode != MixMode::Timeline {
        for (index, clip) in plan
            .clips
            .iter()
            .enumerate()
            .filter(|(_, c)| c.include_audio && c.source.audio)
        {
            if cancelled(cancel) {
                return Err("media_process_cancelled");
            }
            progress.set(
                "mixing_generated_audio",
                index as u64,
                plan.clips.len() as u64,
            );
            let source = media::audio(&frozen.files[&clip.source.resource_id], None, cancel)?;
            let source_end = clip
                .source_out
                .map(seconds)
                .unwrap_or(seconds(clip.source.duration));
            let samples = audio_slice(&source, seconds(clip.source_in), source_end);
            let start = round_samples(clip.output_start as f64 / fps, 44100.) as usize;
            let end =
                round_samples((clip.output_start + clip.frames) as f64 / fps, 44100.) as usize;
            let mut fitted = dsp::fit_audio(samples, end - start);
            let fade_in = round_samples(clip.overlap as f64 / fps, 44100.) as usize;
            let fade_out = plan
                .clips
                .get(index + 1)
                .map(|n| round_samples(n.overlap as f64 / fps, 44100.) as usize)
                .unwrap_or(0);
            dsp::gain_fades(&mut fitted, 1., fade_in, fade_out);
            dsp::add(&mut generated, &fitted, start);
        }
    }
    if plan.options.audio_mode != MixMode::Generated {
        for (index, clip) in plan.audio.iter().enumerate() {
            if cancelled(cancel) {
                return Err("media_process_cancelled");
            }
            progress.set(
                "mixing_timeline_audio",
                index as u64,
                plan.audio.len() as u64,
            );
            let source = media::audio(&frozen.files[&clip.source.resource_id], None, cancel)?;
            let a = (clip
                .source_start_sample
                .unwrap_or_else(|| round_samples(seconds(clip.source_in), 44100.))
                as usize)
                .min(source.len());
            let b = (clip
                .source_end_sample
                .unwrap_or_else(|| round_samples(seconds(clip.source_out), 44100.))
                as usize)
                .min(source.len())
                .max(a);
            let samples = &source[a..b];
            let mut fitted = dsp::fit_audio(samples, clip.samples as usize);
            // Director's sequence mix ignores per-clip normalization overrides.
            // Keep the authored values for other uses, and normalize only the final mix.
            dsp::gain_fades(
                &mut fitted,
                (clip.volume / 100.) as f32,
                round_samples(clip.fade_in, 44100.) as usize,
                round_samples(clip.fade_out, 44100.) as usize,
            );
            dsp::add(&mut timeline, &fitted, clip.start_sample as usize);
        }
    }
    dsp::combine(&mut generated, &timeline, &plan.options);
    if generated.iter().flatten().any(|v| !v.is_finite()) {
        return Err("invalid_audio_mix");
    }
    Ok(generated
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect())
}
/// Prepared raw delivery. The encoder consumes this same stream; probes can inspect
/// it before lossy encoding without duplicating any assembly or mixing arithmetic.
pub struct PreparedMedia {
    pub spec: crate::short_av::EncodeSpec,
    pub frames: Box<dyn Iterator<Item = io::Result<Vec<f32>>> + Send>,
    pub pcm: Vec<u8>,
    error: std::sync::Arc<std::sync::Mutex<Option<&'static str>>>,
}
pub fn quantize_rgb_frame(frame: &[f32], bit_depth: u8) -> Vec<u8> {
    if bit_depth == 10 {
        frame
            .iter()
            .flat_map(|v| ((v.clamp(0., 1.) * 65535. + 0.5) as u16).to_le_bytes())
            .collect()
    } else {
        frame
            .iter()
            .map(|v| (v.clamp(0., 1.) * 255. + 0.5) as u8)
            .collect()
    }
}
pub fn prepare_media(
    frozen: Frozen,
    cancel: std::sync::Arc<AtomicBool>,
    progress: Progress,
) -> Result<PreparedMedia, &'static str> {
    let pcm = mix(&frozen, &cancel, &progress)?;
    let plan = &frozen.plan;
    let spec = crate::short_av::EncodeSpec {
        width: plan.delivery.width,
        height: plan.delivery.height,
        frames: plan.frames,
        fps_num: plan.delivery.fps.num,
        fps_den: plan.delivery.fps.den,
        sample_rate: plan.sample_rate,
        bit_depth: plan.options.bit_depth,
    };
    let error = std::sync::Arc::new(std::sync::Mutex::new(None));
    let frames = Frames {
        frozen,
        cancel: cancel.clone(),
        progress: progress.clone(),
        error: error.clone(),
        clip_index: 0,
        current: None,
        tail: VecDeque::new(),
        produced: 0,
        failed: false,
    };
    Ok(PreparedMedia {
        spec,
        frames: Box::new(frames),
        pcm,
        error,
    })
}
pub fn render(
    frozen: Frozen,
    cancel: std::sync::Arc<AtomicBool>,
    progress: Progress,
) -> Result<Vec<u8>, &'static str> {
    let PreparedMedia {
        spec,
        frames,
        pcm,
        error,
    } = prepare_media(frozen, cancel.clone(), progress.clone())?;
    let frames =
        frames.map(move |result| result.map(|frame| quantize_rgb_frame(&frame, spec.bit_depth)));
    let result = crate::short_av::encode_stream(spec, frames, pcm, &cancel);
    let result = match *error.lock().unwrap() {
        Some(code) => Err(code),
        None => result,
    };
    progress.set(
        if result.is_ok() {
            "encoded"
        } else if cancelled(&cancel) {
            "cancelled"
        } else {
            "failed"
        },
        u64::from(result.is_ok()),
        1,
    );
    result
}
