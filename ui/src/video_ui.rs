//! Compact video desk. Results are persistent on the server; previews stay in memory.
use super::*;
use crate::{h3_protocol as h3, short_av};
use std::{
    io::Write,
    process::{Child, Command, Stdio},
    time::Instant,
};
#[derive(Default)]
pub(super) struct VideoDesk {
    pub prompt: String,
    i2v: bool,
    start_image: Option<ResultView>,
    pub selection: h3::Selection,
    pub available: bool,
    pub checked: bool,
    pub working: bool,
    pub cancelling: bool,
    pub pending: Option<Submission>,
    pub status: String,
    pub error: Option<String>,
    result: Option<VideoView>,
}
struct VideoView {
    av: short_av::Av,
    frame: usize,
    texture: Option<(usize, egui::TextureHandle)>,
    started: Option<(Instant, usize)>,
    audio: Option<Audio>,
    saved: PathBuf,
}
pub(super) struct Audio {
    child: Child,
    writer: Option<std::thread::JoinHandle<()>>,
}
impl Audio {
    #[cfg(feature = "ui-preview")]
    pub(super) fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
    pub(super) fn start(bytes: Vec<u8>) -> Result<Self, &'static str> {
        Self::start_command(Self::command(), bytes)
    }
    fn command() -> Command {
        use std::os::unix::process::CommandExt;
        let mut command = Command::new("/usr/bin/ffplay");
        command
            .args([
                "-nodisp",
                "-autoexit",
                "-hide_banner",
                "-loglevel",
                "quiet",
                "-f",
                "f32le",
                "-sample_rate",
                "32000",
                "-ch_layout",
                "stereo",
                "-i",
                "pipe:0",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let parent = std::process::id() as libc::pid_t;
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                    || libc::getppid() != parent
                {
                    return Err(std::io::Error::other("parent exited"));
                }
                Ok(())
            });
        }
        command
    }
    fn start_command(mut command: Command, bytes: Vec<u8>) -> Result<Self, &'static str> {
        let mut child = command
            .spawn()
            .map_err(|_| "Could not start audio preview. Check ffplay and the audio device.")?;
        let Some(mut input) = child.stdin.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return Err("Could not start audio preview.");
        };
        let writer = std::thread::spawn(move || {
            let _ = input.write_all(&bytes);
        });
        Ok(Self {
            child,
            writer: Some(writer),
        })
    }
}
impl Drop for Audio {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}
impl VideoView {
    fn pause(&mut self) {
        self.advance();
        self.started = None;
        self.audio = None;
    }
    fn advance(&mut self) {
        if let Some((start, base)) = self.started {
            let frame = base + (start.elapsed().as_secs_f64() * h3::FPS as f64) as usize;
            self.frame = frame.min(self.av.shape.frames as usize - 1);
            if frame >= self.av.shape.frames as usize {
                self.started = None;
                self.audio = None;
            }
        }
    }
    fn play(&mut self) -> Result<(), &'static str> {
        if self.frame == self.av.shape.frames as usize - 1 {
            self.frame = 0;
        }
        let sample = self.frame * h3::SAMPLE_RATE as usize / h3::FPS as usize;
        self.audio = Some(Audio::start(
            self.av.pcm[sample * h3::CHANNELS as usize * 4..].to_vec(),
        )?);
        self.started = Some((Instant::now(), self.frame));
        Ok(())
    }
}
pub(super) enum Event {
    Available(Result<bool, Failure>),
    Progress(Progress),
    Finished(Result<client::VideoCompleted, Failure>),
    Cancelled(Result<(), Failure>),
}
impl VideoDesk {
    fn source_dimensions(&self) -> Option<(u32, u32)> {
        self.start_image.as_ref().map(|v| {
            let [w, h] = v.texture.size();
            (w as u32, h as u32)
        })
    }
    fn shape(&self) -> Result<h3::Shape, &'static str> {
        if self.i2v && self.start_image.is_none() {
            return Err("Choose a generated image with Animate first.");
        }
        self.selection.resolve_with_source(if self.i2v {
            self.source_dimensions()
        } else {
            None
        })
    }
    fn request(&self) -> Result<serde_json::Value, &'static str> {
        if self.i2v {
            let image = self
                .start_image
                .as_ref()
                .ok_or("Choose a generated image with Animate first.")?;
            client::i2v_request(
                &self.prompt,
                &self.selection,
                &image.asset_id,
                self.source_dimensions().unwrap(),
            )
        } else {
            client::video_request_with_selection(&self.prompt, &self.selection)
        }
    }
}
impl ImagineApp {
    pub(super) fn can_animate_image(&self) -> bool {
        self.can_edit_image() && !self.video.working && self.video.pending.is_none()
    }
    pub(super) fn start_animating(&mut self) {
        if !self.can_animate_image() {
            return;
        }
        self.video.start_image = self.image.selected_result().cloned();
        self.timeline.quick_clip = true;
        self.video.i2v = true;
        self.video.selection.ratio = "Source".into();
        self.video.selection.portrait = false;
        self.video.error = None;
        self.workspace = Workspace::Video;
    }

    pub(super) fn refresh_video(&mut self, ctx: &egui::Context) {
        let Some(client) = self.client.clone() else {
            return;
        };
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.video_available());
            send(&tx, &ctx, Message::Video(Event::Available(result)));
        });
    }
    pub(super) fn video_message(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::Available(result) => {
                self.video.checked = true;
                match result {
                    Ok(v) => self.video.available = v,
                    Err(e) => self.video.error = Some(e.message),
                }
            }
            Event::Progress(Progress::Submitted(id)) => {
                if let Some(s) = &mut self.video.pending {
                    s.job = Some(id);
                }
            }
            Event::Progress(Progress::Status { label, sampling }) => {
                self.video.status = if label == "Sampling" {
                    sampling.map_or(label.clone(), |p| {
                        format!("{label} {}/{}", p.completed, p.total)
                    })
                } else {
                    label
                };
            }
            Event::Cancelled(result) => {
                if let Err(e) = result {
                    self.video.cancelling = false;
                    self.video.error = Some(e.message);
                }
            }
            Event::Finished(result) => {
                self.video.working = false;
                self.video.cancelling = false;
                match result {
                    Ok(done) => {
                        self.video.result = Some(VideoView {
                            av: done.av,
                            frame: 0,
                            texture: None,
                            started: None,
                            audio: None,
                            saved: done.saved,
                        });
                        self.video.pending = None;
                        self.video.status = "Saved".into();
                        self.video.error = None;
                    }
                    Err(e) => {
                        self.video.error = Some(e.message);
                        self.video.status.clear();
                        if !e.retryable {
                            self.video.pending = None;
                        }
                    }
                }
            }
        }
        ctx.request_repaint();
    }
    pub(super) fn generate_video(&mut self, ctx: &egui::Context) {
        if self.video.working || !self.video.available || self.video.prompt.trim().is_empty() {
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        if self.video.pending.is_none() {
            self.video.pending = Some(Submission {
                upscale_source: None,
                request: match self.video.request() {
                    Ok(request) => request,
                    Err(error) => {
                        self.video.error = Some(error.into());
                        return;
                    }
                },
                key: uuid::Uuid::new_v4().to_string(),
                folder: self.prefs.output_folder.clone(),
                job: None,
            });
        }
        if let Some(view) = &mut self.video.result {
            view.pause();
        }
        self.video.working = true;
        self.video.error = None;
        self.video.status = "Submitting".into();
        let submission = self.video.pending.clone().unwrap();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.generate_video(submission, |p| {
                send(&tx, &ctx, Message::Video(Event::Progress(p)))
            }));
            send(&tx, &ctx, Message::Video(Event::Finished(result)));
        });
    }
    pub(super) fn cancel_video(&mut self, ctx: &egui::Context) {
        let Some(job) = self.video.pending.as_ref().and_then(|s| s.job.clone()) else {
            return;
        };
        let Some(client) = self.client.clone() else {
            return;
        };
        self.video.cancelling = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.cancel(&job));
            send(&tx, &ctx, Message::Video(Event::Cancelled(result)));
        });
    }
    pub(super) fn video_workspace(&mut self, ui: &mut egui::Ui, bounds: egui::Rect) {
        if !self.timeline.quick_clip {
            self.timeline_workspace(ui, bounds);
            return;
        }
        workspace::panel(ui, bounds, false);
        let inner = bounds.shrink(12.);
        workspace::region(ui, "video-desk", inner, |ui| {
            egui::ScrollArea::vertical().id_salt("video-scroll").show(ui,|ui|{
                ui.horizontal(|ui|{ui.heading("H3");ui.weak("Experimental · 24 fps · Audio");if ui.button("Timeline").clicked(){self.timeline.quick_clip=false;}});
                ui.weak(format!("{} steps · {} / {} · CFG 1 · Shifts 12 / 3 · Full denoise", h3::STEPS, h3::SAMPLER, h3::SCHEDULER));
                ui.weak("Size presets change resolution only. The 20-step profile still needs GPU quality validation.");
                ui.add_enabled_ui(!self.video.working && self.video.pending.is_none(), |ui| {
                    let old_i2v=self.video.i2v;
                    egui::ComboBox::from_id_salt("h3-profile").selected_text(if self.video.i2v {"Animate image · I2V"} else {"Text · Ref2VA"}).show_ui(ui,|ui| {
                        ui.selectable_value(&mut self.video.i2v,false,"Text · Ref2VA");
                        ui.selectable_value(&mut self.video.i2v,true,"Animate image · I2V");
                    });
                    if old_i2v != self.video.i2v {self.video.selection.ratio=if self.video.i2v {"Source"}else{"3:2"}.into();}
                    if self.video.i2v {
                        ui.horizontal(|ui| {
                            if let Some(image)=&self.video.start_image {
                                if !self.prefs.privacy_mode {ui.add(egui::Image::new(&image.texture).max_size(egui::vec2(96.,64.)));} else {ui.weak("Start image hidden");}
                                if helto::select_button(ui,"Remove start image",false).clicked() {self.video.start_image=None;}
                            } else {ui.weak("Select a generated image in Image, then choose Animate.");}
                        });
                        ui.weak("First-frame conditioning; generated audio at gain 1.");
                        ui.weak("Source ratio preserves the image aspect after 32-pixel alignment. Other ratios stretch without cropping.");
                    }
                    ui.horizontal_wrapped(|ui| {
                        egui::ComboBox::from_id_salt("h3-ratio").selected_text(if self.video.selection.ratio=="Source" {"Source image ratio"} else {&self.video.selection.ratio}).show_ui(ui, |ui| {
                            if self.video.i2v {ui.selectable_value(&mut self.video.selection.ratio,"Source".into(),"Source image ratio");}
                            for ratio in h3::geometry::RATIOS { ui.selectable_value(&mut self.video.selection.ratio, ratio.into(), ratio); }
                        });
                        ui.add_enabled_ui(self.video.selection.ratio!="Source",|ui| {egui::ComboBox::from_id_salt("h3-orientation").selected_text(if self.video.selection.portrait {"Portrait"} else {"Landscape"}).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.video.selection.portrait, false, "Landscape");
                            ui.selectable_value(&mut self.video.selection.portrait, true, "Portrait");
                        });});
                        egui::ComboBox::from_id_salt("h3-size").selected_text(&self.video.selection.quality).show_ui(ui, |ui| {
                            for quality in h3::geometry::QUALITIES { ui.selectable_value(&mut self.video.selection.quality, quality.into(), quality); }
                        });
                        ui.label("Seconds");
                        ui.add(egui::DragValue::new(&mut self.video.selection.seconds).range(0.25..=3600.).speed(0.25).max_decimals(3));
                    });
                });
                let geometry = self.video.shape();
                match geometry {
                    Ok(shape) => {
                        ui.label(format!("{} × {} · {} frames · {:.3} s · Audio {:.3} s", shape.width, shape.height, shape.frames, shape.seconds(), shape.audio_samples() as f64 / 32000.));
                        if !self.video.i2v && shape != h3::Shape::default() { ui.weak(format!("GPU untested: may exceed VRAM. Decoded AV {:.0} MiB. Verified: 384 × 256 / 39 frames.", shape.raw_bytes() as f64 / 1048576.)); }
                        if shape.frames < 124 || shape.frames > 362 { ui.weak("Outside H3's reported training range (~5–15 s); motion or audio quality may degrade."); }
                    },
                    Err(error) => { ui.colored_label(helto::ERROR, match error {"invalid_h3_source_ratio"=>"Choose a generated image with Animate first.", "invalid_h3_duration"=>"Enter a duration between 0.25 and 3600 seconds.", "h3_result_exceeds_memory_limit"=>"Decoded video exceeds the 2 GiB host-memory limit. Reduce size or duration.",other=>other}); }
                }
                ui.label("Prompt");
                ui.add_enabled_ui(!self.video.working&&self.video.pending.is_none(),|ui|{
                    helto::textarea(ui,&mut self.video.prompt,3,"Video prompt",self.prefs.privacy_mode,true);
                });
                ui.horizontal(|ui|{
                    let label=if self.video.pending.is_some()&&!self.video.working{"Retry"}else{"Generate video"};
                    if ui.add_enabled_ui(geometry.is_ok()&&self.video.available&&!self.video.working&&!self.video.prompt.trim().is_empty(),|ui|helto::select_button(ui,label,true)).inner.clicked(){self.generate_video(ui.ctx());}
                    if self.video.working {
                        if ui.add_enabled_ui(!self.video.cancelling&&self.video.pending.as_ref().is_some_and(|s|s.job.is_some()),|ui|helto::select_button(ui,"Cancel video",false)).inner.clicked(){self.cancel_video(ui.ctx());}
                        ui.spinner();
                    }
                    if !self.video.available&&self.video.checked {
                        ui.weak("H3 unavailable").on_hover_text("Check the H3 deployment and GPU coordinator, then reconnect.");
                        if helto::select_button(ui,"Reconnect",false).clicked(){self.connect(ui.ctx());}
                    }
                    ui.label(&self.video.status);
                });
                if let Some(error)=&self.video.error {ui.colored_label(helto::ERROR,error);}
                let size=egui::vec2(ui.available_width(),(ui.available_height()-48.).clamp(160.,360.));
                let (media,_) =ui.allocate_exact_size(size,egui::Sense::hover());
                ui.painter().rect_filled(media,5,helto::MEDIA);
                let revealed=!self.prefs.privacy_mode||(ui.input(|i|i.focused)&&ui.rect_contains_pointer(egui::Rect::from_min_max(media.min,media.max+egui::vec2(0.,40.)).intersect(ui.clip_rect())));
                let mut play=false;
                if let Some(view)=&mut self.video.result {
                    if !revealed {view.pause();}
                    view.advance();
                    if view.started.is_some() {
                        if let Some(audio)=&mut view.audio {
                            match audio.child.try_wait() {
                                Ok(Some(status)) if !status.success()=>{view.pause();self.video.error=Some("Could not play audio. Check the audio device and retry Play.".into());}
                                Err(_)=>{view.pause();self.video.error=Some("Audio preview stopped. Retry Play.".into());}
                                _=>{}
                            }
                        }
                        ui.ctx().request_repaint_after(Duration::from_secs_f64(1./h3::FPS as f64));
                    }
                    if revealed {
                        if view.texture.as_ref().is_none_or(|(frame,_)|*frame!=view.frame){
                            let image=egui::ColorImage::from_rgb([view.av.shape.width as usize,view.av.shape.height as usize],&view.av.frames[view.frame]);
                            view.texture=Some((view.frame,ui.ctx().load_texture("video-preview",image,egui::TextureOptions::LINEAR)));
                        }
                        let (_,texture)=view.texture.as_ref().unwrap();
                        let fit=egui::vec2(view.av.shape.width as f32,view.av.shape.height as f32)*(media.width()/view.av.shape.width as f32).min(media.height()/view.av.shape.height as f32);
                        ui.painter().image(texture.id(),egui::Rect::from_center_size(media.center(),fit),egui::Rect::from_min_max(egui::Pos2::ZERO,egui::pos2(1.,1.)),egui::Color32::WHITE);
                    }else {ui.painter().text(media.center(),egui::Align2::CENTER_CENTER,"Preview hidden",egui::FontId::proportional(14.),helto::DIM);}
                    ui.horizontal(|ui|{
                        if helto::select_button(ui,if view.started.is_some(){"Pause"}else{"Play"},false).clicked(){play=true;}
                        let mut frame=view.frame;
                        if ui.add(egui::Slider::new(&mut frame,0..=view.av.shape.frames as usize-1).text("Frame")).changed(){view.pause();view.frame=frame;}
                        ui.label("Saved automatically").on_hover_text(view.saved.display().to_string());
                    });
                    if play {if view.started.is_some(){view.pause();}else if let Err(error)=view.play(){self.video.error=Some(error.into());}}
                }else {ui.painter().text(media.center(),egui::Align2::CENTER_CENTER,"Video preview",egui::FontId::proportional(14.),helto::DIM);}
                if let Some(view)=&self.video.result {ui.label(format!("Saved: {}",view.saved.display()));}
            });
        });
    }
    pub(super) fn pause_video(&mut self) {
        if let Some(view) = &mut self.video.result {
            view.pause();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn done() -> client::VideoCompleted {
        client::VideoCompleted {
            saved: PathBuf::from("/synthetic/encrypted/video_0001.mp4"),
            av: short_av::Av {
                continuation: None,
                shape: Default::default(),
                frames: vec![vec![0; h3::FRAME_BYTES]; h3::FRAMES as usize],
                pcm: vec![0; h3::AUDIO_SAMPLES as usize * 8],
            },
        }
    }
    fn app() -> (tempfile::TempDir, ImagineApp, egui::Context) {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.timeline.quick_clip = true;
        app.workspace = Workspace::Video;
        app.video.available = true;
        app.video.checked = true;
        app.video.prompt = "SYNTHETIC_PRIVATE_VIDEO_PROMPT".into();
        app.prefs.output_folder = root.path().join("explicit-save-only");
        (root, app, ctx)
    }
    #[test]
    fn animate_keeps_source_alive_and_never_submits_until_requested() {
        let (_root, mut app, _ctx) = app();
        app.workspace = Workspace::Image;
        assert!(app.can_animate_image());
        let original = app.image.selected_result().unwrap().asset_id.clone();
        app.start_animating();
        assert_eq!(app.workspace, Workspace::Video);
        assert_eq!(app.video.selection.ratio, "Source");
        assert!(app.video.i2v);
        app.image.result = None;
        assert_eq!(app.video.start_image.as_ref().unwrap().asset_id, original);
        assert!(app.video.pending.is_none());
        assert!(!app.video.working);
        assert!(app.video.shape().is_ok());
        let request = app.video.request().unwrap();
        assert_eq!(request["inputs"]["start_image"], original);
        app.video.start_image = None;
        assert!(app.video.request().is_err());
        app.video.i2v = false;
        app.video.selection = Default::default();
        assert_eq!(
            app.video.request().unwrap()["video"]["profile"],
            "h3-av-noref-v2"
        );
    }
    #[test]
    fn audio_helper_uses_valid_stereo_flags_and_reaps_with_dummy_devices() {
        let mut command = Audio::command();
        command
            .env("SDL_VIDEODRIVER", "dummy")
            .env("SDL_AUDIODRIVER", "dummy");
        let mut audio =
            Audio::start_command(command, vec![0; h3::AUDIO_SAMPLES as usize * 8]).unwrap();
        let pid = audio.child.id();
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            if let Some(status) = audio.child.try_wait().unwrap() {
                assert!(status.success(), "own dummy audio helper failed");
                break;
            }
            assert!(
                Instant::now() < deadline,
                "own dummy audio helper exceeded bound"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(audio);
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
    #[test]
    fn completed_preview_reports_persistent_path_without_another_save() {
        let (root, mut app, ctx) = app();
        app.video_message(&ctx, Event::Finished(Ok(done())));
        assert_eq!(
            app.video.result.as_ref().unwrap().saved,
            PathBuf::from("/synthetic/encrypted/video_0001.mp4")
        );
        assert!(!root.path().join("explicit-save-only").exists());
        assert_eq!(app.video.status, "Saved");
    }
    #[test]
    fn video_layout_privacy_and_controls_are_headless_and_prompt_stays_private() {
        let (_root, mut app, ctx) = app();
        app.video_message(&ctx, Event::Finished(Ok(done())));
        ctx.enable_accesskit();
        for privacy in [false, true] {
            app.prefs.privacy_mode = privacy;
            let mut result = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
                    )),
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            result.textures_delta.clear();
            let nodes = result.platform_output.accesskit_update.unwrap().nodes;
            assert!(!nodes.iter().any(|(_, n)| n.label() == Some("Save video")));
            assert!(nodes.iter().any(|(_, n)| n.label() == Some("Play")));
            if privacy {
                assert!(!nodes.iter().any(|(_, n)| {
                    n.value()
                        .is_some_and(|v| v.contains("SYNTHETIC_PRIVATE_VIDEO_PROMPT"))
                }));
            }
        }
        assert!(app.video.result.as_ref().unwrap().audio.is_none());
    }
    #[test]
    fn playback_reaches_full_last_frame_interval_and_pauses_on_workspace_leave() {
        let (_root, mut app, ctx) = app();
        app.video_message(&ctx, Event::Finished(Ok(done())));
        let view = app.video.result.as_mut().unwrap();
        view.started = Some((Instant::now() - Duration::from_millis(1600), 0));
        view.advance();
        assert_eq!(view.frame, 38);
        assert!(view.started.is_some());
        view.started = Some((Instant::now() - Duration::from_millis(1700), 0));
        view.advance();
        assert!(view.started.is_none());
        view.started = Some((Instant::now(), 0));
        app.workspace = Workspace::Image;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| app.content(ui));
        output.textures_delta.clear();
        assert!(app.video.result.as_ref().unwrap().started.is_none());
    }
}
