//! Bounded CPU preview of owned bytes. Sources and PCM stay in memory; no preview files.
use super::*;
use std::{
    fs::File,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::process::CommandExt,
    },
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
mod comparison;
#[cfg(test)]
#[path = "player/lifecycle_tests.rs"]
mod lifecycle_tests;
pub(in crate::app) use comparison::{
    Comparison, Entry as ComparisonEntry, SelectionState as ComparisonState,
};
pub struct Preview {
    width: usize,
    height: usize,
    frames: Vec<Vec<u8>>,
    pcm: Vec<u8>,
    duration: f64,
}
impl Preview {
    /// Silent synthetic frames for tests.
    #[cfg(test)]
    pub fn synthetic(seconds: f64) -> Self {
        let frames = (seconds * 24.).ceil() as usize;
        Self {
            width: 8,
            height: 8,
            frames: vec![vec![40; 8 * 8 * 3]; frames],
            pcm: vec![],
            duration: seconds,
        }
    }
}
pub struct Player {
    media: Preview,
    frame: usize,
    texture: Option<(usize, egui::TextureHandle)>,
    started: Option<(Instant, usize)>,
    audio: Option<super::super::video_ui::Audio>,
    muted: bool,
}
impl Player {
    pub fn is_playing(&self) -> bool {
        self.started.is_some()
    }
    /// Seconds into the media, including the running clock.
    pub fn position(&self) -> f64 {
        match self.started {
            Some((instant, start)) => {
                (start as f64 / 24. + instant.elapsed().as_secs_f64()).min(self.media.duration)
            }
            None => self.frame as f64 / 24.,
        }
    }
    /// Pauses and moves to `seconds` into the media.
    pub fn seek(&mut self, seconds: f64) {
        self.pause();
        let last = (self.media.duration * 24.).ceil().max(1.) as usize - 1;
        self.frame = ((seconds.max(0.) * 24.).round() as usize).min(last);
    }
    /// Starts playback from the current frame.
    pub fn start(&mut self) -> Result<(), String> {
        self.play()
    }
    /// Played to its last frame (the clock has stopped there).
    pub fn finished(&mut self) -> bool {
        self.advance();
        let last = (self.media.duration * 24.).ceil().max(1.) as usize - 1;
        self.started.is_none() && self.frame >= last
    }
    #[cfg(feature = "ui-preview")]
    pub fn audio_running(&mut self) -> bool {
        self.audio.as_mut().is_some_and(|audio| audio.is_running())
    }
    pub fn new(media: Preview) -> Self {
        Self {
            media,
            frame: 0,
            texture: None,
            started: None,
            audio: None,
            muted: false,
        }
    }
    pub fn pause(&mut self) {
        self.advance();
        self.started = None;
        self.audio = None;
    }
    fn advance(&mut self) {
        if let Some((instant, start)) = self.started {
            let position = start as f64 / 24. + instant.elapsed().as_secs_f64();
            let frame = (position * 24.) as usize;
            let last = (self.media.duration * 24.).ceil().max(1.) as usize - 1;
            self.frame = frame.min(last);
            if position >= self.media.duration {
                self.started = None;
                self.audio = None;
            }
        }
    }
    fn play(&mut self) -> Result<(), String> {
        if self.frame as f64 / 24. >= self.media.duration - 1. / 24. {
            self.frame = 0;
        }
        let sample = (self.frame as u64 * 32000 / 24) as usize * 8;
        self.audio = if !self.muted && sample < self.media.pcm.len() {
            Some(
                super::super::video_ui::Audio::start(self.media.pcm[sample..].to_vec())
                    .map_err(str::to_owned)?,
            )
        } else {
            None
        };
        self.started = Some((Instant::now(), self.frame));
        Ok(())
    }
    pub fn show(&mut self, ui: &mut egui::Ui, privacy: bool) -> Result<(), String> {
        self.advance();
        let size = egui::vec2(ui.available_width(), (ui.available_height() - 38.).max(40.));
        let (rect, _response) = ui.allocate_exact_size(size, egui::Sense::hover());
        let boundary = egui::Rect::from_min_max(rect.min, rect.max + egui::vec2(0., 38.));
        let revealed = helto::revealed(ui, boundary, privacy);
        if !revealed {
            self.pause();
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Preview hidden",
                egui::FontId::proportional(14.),
                helto::DIM,
            );
        } else {
            self.paint_frame(ui, rect);
        }
        if self.media.duration == 0. {
            return Ok(());
        }
        let mut play = false;
        ui.horizontal(|ui| {
            let (icon, label) = if self.started.is_some() {
                (crate::icons::Icon::Pause, "Pause")
            } else {
                (crate::icons::Icon::Play, "Play")
            };
            let response = helto::action(
                ui,
                icon,
                label,
                false,
                (!revealed).then_some("Reveal the preview to play it."),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-play".into(), &response);
            if response.clicked() {
                if self.started.is_some() {
                    self.pause();
                } else {
                    play = true;
                }
            }
            let mut position = self.frame as f64 / 24.;
            let seek = ui.add_enabled(
                revealed,
                egui::Slider::new(&mut position, 0.0..=self.media.duration).show_value(false),
            );
            if seek.changed() {
                self.pause();
                self.frame = (position * 24.).round() as usize;
            }
            ui.weak(format!(
                "{:.2} / {:.2}",
                self.frame as f64 / 24.,
                self.media.duration
            ));
            if ui.checkbox(&mut self.muted, "Mute").changed() {
                let playing = self.started.is_some();
                self.pause();
                play = playing;
            }
        });
        if play {
            self.play()?;
        }
        if self.started.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(20));
        }
        Ok(())
    }
    /// Rendering only: A/B owns a common clock and a single independent audio source.
    fn paint_frame(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        if !self.media.frames.is_empty() {
            let index = self.frame.min(self.media.frames.len() - 1);
            if self.texture.as_ref().is_none_or(|(n, _)| *n != index) {
                let image = egui::ColorImage::from_rgb(
                    [self.media.width, self.media.height],
                    &self.media.frames[index],
                );
                self.texture = Some((
                    index,
                    ui.ctx()
                        .load_texture("timeline-frame", image, egui::TextureOptions::LINEAR),
                ));
            }
            let texture = &self.texture.as_ref().unwrap().1;
            let original = texture.size_vec2();
            let scale = (rect.width() / original.x).min(rect.height() / original.y);
            let image = egui::Rect::from_center_size(rect.center(), original * scale);
            ui.painter().image(
                texture.id(),
                image,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Audio preview",
                egui::FontId::proportional(14.),
                helto::DIM,
            );
        }
    }
}
pub fn decode(bytes: Vec<u8>, info: Value, cancel: Arc<AtomicBool>) -> Result<Preview, String> {
    if cancel.load(Ordering::Acquire) {
        return Err("Preview cancelled.".into());
    }
    let kind = info["media"]["kind"].as_str().unwrap_or("");
    if kind == "image" {
        let image = image::load_from_memory(&bytes)
            .map_err(|_| "Could not decode the image.")?
            .thumbnail(960, 540)
            .to_rgb8();
        return Ok(Preview {
            width: image.width() as usize,
            height: image.height() as usize,
            frames: vec![image.into_raw()],
            pcm: vec![],
            duration: 0.,
        });
    }
    let duration: Time = serde_json::from_value(info["media"]["duration"].clone())
        .map_err(|_| "Media duration unavailable. Check media.")?;
    let duration = super::editing::seconds(duration);
    if duration <= 0. || !duration.is_finite() {
        return Err("Media duration unavailable. Check media.".into());
    }
    let mime = info["mime_type"].as_str().unwrap_or("");
    let format = match mime {
        "video/mp4" => "mov",
        "video/webm" | "video/x-matroska" => "matroska",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/flac" => "flac",
        "audio/mpeg" => "mp3",
        "audio/ogg" => "ogg",
        _ => return Err("Unsupported preview format.".into()),
    };
    let fd = unsafe { libc::memfd_create(c"imagine-owned-preview".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err("Could not allocate preview memory.".into());
    }
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(&bytes)
        .map_err(|_| "Could not prepare preview.")?;
    drop(bytes);
    let (width, height, frames) = if kind == "video" {
        let width = info["width"].as_u64().unwrap_or(0) as usize;
        let height = info["height"].as_u64().unwrap_or(0) as usize;
        if width == 0 || height == 0 {
            return Err("Invalid video dimensions.".into());
        }
        let budget = 512. * 1024. * 1024.;
        let scale = (960. / width as f64)
            .min(540. / height as f64)
            .min((budget / (duration * 24. * width as f64 * height as f64 * 3.)).sqrt())
            .min(1.);
        let width = (width as f64 * scale).floor().max(1.) as usize;
        let height = (height as f64 * scale).floor().max(1.) as usize;
        let filter = format!("scale={width}:{height},fps=24");
        let out = run(
            &file,
            format,
            &[
                "-map", "0:v:0", "-vf", &filter, "-an", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "pipe:1",
            ],
            512 * 1024 * 1024,
            &cancel,
        )?;
        let size = width * height * 3;
        if out.is_empty() || out.len() % size != 0 {
            return Err("Could not decode the complete video preview.".into());
        }
        let frames = out.chunks_exact(size).map(<[u8]>::to_vec).collect();
        (width, height, frames)
    } else {
        (0, 0, vec![])
    };
    let pcm = if info["media"]["audio"] == true {
        run(
            &file,
            format,
            &[
                "-map",
                "0:a:0",
                "-vn",
                "-acodec",
                "pcm_f32le",
                "-ar",
                "32000",
                "-ac",
                "2",
                "-f",
                "f32le",
                "pipe:1",
            ],
            512 * 1024 * 1024,
            &cancel,
        )?
    } else {
        vec![]
    };
    Ok(Preview {
        width,
        height,
        frames,
        pcm,
        duration,
    })
}
fn run(
    file: &File,
    format: &str,
    args: &[&str],
    limit: usize,
    cancel: &AtomicBool,
) -> Result<Vec<u8>, String> {
    let mut command = Command::new("/usr/bin/ffmpeg");
    command
        .args([
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-f",
            format,
            "-i",
            "/proc/self/fd/3",
        ])
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let source = file.as_raw_fd();
    unsafe {
        command.pre_exec(move || {
            if libc::dup2(source, 3) < 0 || libc::fcntl(3, libc::F_SETFD, 0) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|_| "Could not start video preview. Check FFmpeg.")?;
    let stdout = child.stdout.take().ok_or("Could not read preview.")?;
    let read = std::thread::spawn(move || {
        let mut bytes = vec![];
        stdout
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes)
    });
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline && !cancel.load(Ordering::Acquire) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let _ = child.wait();
    let out = read
        .join()
        .map_err(|_| "Preview failed.")?
        .map_err(|_| "Preview failed.")?;
    if status.is_none_or(|s| !s.success()) || out.len() > limit {
        return Err(if cancel.load(Ordering::Acquire) {
            "Preview cancelled."
        } else {
            "Could not decode preview within its memory or time limit."
        }
        .into());
    }
    Ok(out)
}
use crate::app::workspace as chrome;
use egui::{Rect, Sense};
impl ImagineApp {
    pub(super) fn timeline_viewer(&mut self, ui: &mut egui::Ui, rect: Rect, d: &Snapshot) {
        let current_scene = self.timeline.primary_scene(&d.project);
        if self.timeline.comparison.as_ref().is_some_and(|c| {
            c.ids.iter().any(|id| {
                !d.view.takes.iter().any(|t| {
                    t.id.0 == *id
                        && Some(&t.scene_id) == current_scene.as_ref()
                        && review::attached(d, t)
                })
            })
        }) {
            self.timeline.clear_preview();
            self.timeline.comparison_selection = [None, None];
        }
        chrome::panel(ui, rect, true);
        let inner = rect.shrink(10.);
        chrome::region(
            ui,
            "timeline-viewer-heading",
            Rect::from_min_size(inner.min, egui::vec2(inner.width(), 24.)),
            |ui| {
                let number = self
                    .timeline
                    .scene()
                    .and_then(|s| d.project.scenes.iter().position(|v| v.id == s.id))
                    .map(|n| n + 1);
                ui.horizontal(|ui| {
                    ui.strong(
                        number
                            .map(|n| format!("Scene {n}"))
                            .unwrap_or_else(|| "Viewer".into()),
                    );
                    if self.timeline.loading.is_some() {
                        ui.spinner();
                    }
                    self.sampling_preview_controls(ui);
                });
            },
        );
        let picture = Rect::from_min_max(
            inner.min + egui::vec2(0., 28.),
            egui::pos2(inner.right(), (inner.bottom() - 78.).max(inner.top() + 50.)),
        );
        chrome::region(ui, "timeline-player", picture, |ui| {
            if self.timeline.comparison.is_some() {
                self.sampling_preview.hide_texture();
                ui.centered_and_justified(|ui| ui.weak("A/B Take review"));
            } else if self.sampling_preview_viewer(ui) {
                if let Some(player) = &mut self.timeline.player {
                    player.pause();
                }
            } else if let Some(player) = &mut self.timeline.player {
                if let Err(error) = player.show(ui, self.prefs.privacy_mode) {
                    self.timeline.error = Some(error);
                }
            } else {
                let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::hover());
                let reveal =
                    !self.prefs.privacy_mode || (response.hovered() && ui.input(|i| i.focused));
                let text = if !reveal {
                    "Preview hidden"
                } else if self.timeline.loading.is_some() {
                    "Loading preview…"
                } else {
                    "Choose a Take or project media"
                };
                let galley = ui.painter().layout(
                    text.into(),
                    egui::FontId::proportional(14.),
                    helto::DIM,
                    (rect.width() - 8.).max(1.),
                );
                let position = rect.center() - galley.size() * 0.5;
                ui.painter().galley(position, galley, helto::DIM);
            }
        });
        let strip = Rect::from_min_max(egui::pos2(inner.left(), inner.bottom() - 72.), inner.max);
        chrome::region(ui, "timeline-takes", strip, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("timeline-take-strip")
                .show(ui, |ui| {
                    #[cfg(feature = "ui-preview")]
                    {
                        let r = ui.interact(
                            ui.clip_rect(),
                            ui.id().with("take-strip-scroll"),
                            egui::Sense::hover(),
                        );
                        super::qa::control(ui, "take-strip-scroll".into(), &r);
                    }
                    self.timeline_take_strip(ui, d);
                    ui.add_space(4.); // Keep a bottom-row keyboard focus ring inside the scroll clip.
                });
        });

        if self.timeline.comparison.is_some() {
            self.timeline_comparison_dialog(ui.ctx(), d, current_scene);
        }
        if self.timeline.captures.open {
            self.timeline_captures_dialog(ui.ctx(), d);
        }
    }
    fn timeline_comparison_dialog(
        &mut self,
        ctx: &egui::Context,
        d: &Snapshot,
        current_scene: Option<SceneId>,
    ) {
        let bounds = ctx.content_rect();
        let mut interaction = None;
        let response =
            egui::Modal::new(egui::Id::new("timeline-comparison-dialog")).show(ctx, |ui| {
                ui.set_width((bounds.width() - 48.).clamp(260., 1100.));
                ui.set_height((bounds.height() - 80.).max(240.));
                if let Some(comparison) = &mut self.timeline.comparison {
                    let entries = current_scene
                        .as_ref()
                        .map(|id| super::review::comparison_entries(d, id))
                        .unwrap_or_default();
                    match comparison.show(
                        ui,
                        self.prefs.privacy_mode,
                        &entries,
                        !self.timeline.busy && !self.timeline.admitting,
                    ) {
                        Ok(value) => interaction = Some(value),
                        Err(error) => self.timeline.error = Some(error),
                    }
                    let pair = comparison.ids.clone().map(Some);
                    self.timeline.comparison_explicit |= self.timeline.comparison_selection != pair;
                    self.timeline.comparison_selection = pair;
                    // Recipe metadata follows Hide previews like the pictures.
                    ui.collapsing("Differences", |ui| {
                        if helto::reveal_in_viewport(ui, self.prefs.privacy_mode) {
                            let takes = comparison
                                .ids
                                .each_ref()
                                .map(|id| d.take(&TakeId(id.clone())));
                            super::review::take_differences(ui, takes);
                        } else {
                            ui.label("Hidden metadata");
                        }
                    });
                }
            });
        if response.should_close() {
            self.timeline.clear_preview();
            return;
        }
        if let Some(value) = interaction {
            if value.close {
                self.timeline.clear_preview();
            } else if let Some((side, id)) = value.select {
                self.timeline.comparison_explicit = true;
                let entries = current_scene
                    .as_ref()
                    .map(|id| super::review::comparison_entries(d, id))
                    .unwrap_or_default();
                let ids = entries.into_iter().map(|e| e.id).collect::<Vec<_>>();
                let accepted = current_scene
                    .as_ref()
                    .and_then(|id| d.project.scene(id).ok())
                    .and_then(|s| s.accepted_take.as_ref())
                    .map(|t| t.id.0.as_str());
                self.timeline.comparison_selection = super::take_selection::update(
                    &ids,
                    accepted,
                    &self.timeline.comparison_selection,
                    side,
                    &id,
                );
                self.timeline_compare(ctx, d);
            } else if let Some((id, action)) = value.review
                && let Some(scene) = &current_scene
            {
                self.timeline_apply_review(ctx, scene.clone(), TakeId(id), action);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_frame_keeps_its_full_interval_and_concealment_pauses() {
        let mut player = Player::new(Preview {
            width: 32,
            height: 32,
            frames: vec![vec![90; 32 * 32 * 3]; 39],
            pcm: vec![],
            duration: 39. / 24.,
        });
        player.started = Some((Instant::now() - Duration::from_millis(1600), 0));
        player.advance();
        assert_eq!(player.frame, 38);
        assert!(player.is_playing());
        player.started = Some((Instant::now() - Duration::from_millis(1700), 0));
        player.advance();
        assert!(!player.is_playing());
        player.started = Some((Instant::now(), 0));
        let ctx = egui::Context::default();
        let mut result = ctx.run_ui(
            egui::RawInput {
                focused: false,
                ..Default::default()
            },
            |ui| player.show(ui, true).unwrap(),
        );
        result.textures_delta.clear();
        assert!(!player.is_playing());
    }
    #[test]
    fn generic_non_h3_video_previews_and_precancel_never_decodes() {
        let output = Command::new("/usr/bin/ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30:duration=0.2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "matroska",
                "pipe:1",
            ])
            .output()
            .unwrap();
        assert!(output.status.success());
        let info = json!({"mime_type":"video/x-matroska","width":64,"height":48,"media":{"kind":"video","audio":false,"duration":{"num":1,"den":5}}});
        let preview = decode(output.stdout, info, Arc::new(AtomicBool::new(false))).unwrap();
        assert_eq!((preview.width, preview.height), (64, 48));
        assert!(!preview.frames.is_empty());
        assert_eq!(preview.duration, 0.2);
        assert!(decode(vec![], Value::Null, Arc::new(AtomicBool::new(true))).is_err());
    }
}
