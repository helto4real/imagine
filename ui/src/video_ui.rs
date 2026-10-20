//! Compact fixed-profile video desk. Generated AV stays in memory until Save.
use super::*;
use crate::{h3_protocol as h3, settings, short_av};
use std::{
    io::Write,
    process::{Child, Command, Stdio},
    sync::Arc,
    time::Instant,
};
#[derive(Default)]
pub(super) struct VideoDesk {
    pub prompt: String,
    pub selection: h3::Selection,
    pub available: bool,
    pub checked: bool,
    pub working: bool,
    pub saving: bool,
    pub cancelling: bool,
    pub pending: Option<Submission>,
    pub status: String,
    pub error: Option<String>,
    result: Option<VideoView>,
    revision: u64,
}
struct VideoView {
    bytes: Arc<Vec<u8>>,
    av: short_av::Av,
    frame: usize,
    texture: Option<(usize, egui::TextureHandle)>,
    started: Option<(Instant, usize)>,
    audio: Option<Audio>,
    saved: Option<PathBuf>,
}
struct Audio {
    child: Child,
    writer: Option<std::thread::JoinHandle<()>>,
}
impl Audio {
    fn start(bytes: Vec<u8>) -> Result<Self, &'static str> {
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
    Saved(u64, Result<PathBuf, String>),
}
impl ImagineApp {
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
                        self.video.revision += 1;
                        self.video.result = Some(VideoView {
                            bytes: done.bytes,
                            av: done.av,
                            frame: 0,
                            texture: None,
                            started: None,
                            audio: None,
                            saved: None,
                        });
                        self.video.pending = None;
                        self.video.status = "Ready".into();
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
            Event::Saved(revision, result) => {
                if revision != self.video.revision {
                    return;
                }
                self.video.saving = false;
                match result {
                    Ok(path) => {
                        if let Some(r) = &mut self.video.result {
                            r.saved = Some(path);
                        }
                        self.video.error = None;
                    }
                    Err(error) => self.video.error = Some(error),
                }
            }
        }
        ctx.request_repaint();
    }
    pub(super) fn generate_video(&mut self, ctx: &egui::Context) {
        if self.video.working
            || self.video.saving
            || !self.video.available
            || self.video.prompt.trim().is_empty()
        {
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        if self.video.pending.is_none() {
            self.video.pending = Some(Submission {
                upscale_source: None,
                request: match client::video_request_with_selection(
                    &self.video.prompt,
                    &self.video.selection,
                ) {
                    Ok(request) => request,
                    Err(error) => {
                        self.video.error = Some(error.into());
                        return;
                    }
                },
                key: uuid::Uuid::new_v4().to_string(),
                folder: PathBuf::new(),
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
    fn save_video(&mut self, ctx: &egui::Context) {
        if self.video.saving || self.video.working {
            return;
        }
        let Some(view) = &self.video.result else {
            return;
        };
        if view.saved.is_some() {
            return;
        }
        let bytes = view.bytes.clone();
        let folder = self.prefs.output_folder.clone();
        let revision = self.video.revision;
        self.video.saving = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = settings::save_video(&folder, &bytes).map_err(|_| {
                "Could not save the video. Check the output folder and retry Save.".into()
            });
            send(&tx, &ctx, Message::Video(Event::Saved(revision, result)));
        });
    }
    pub(super) fn video_workspace(&mut self, ui: &mut egui::Ui, bounds: egui::Rect) {
        workspace::panel(ui, bounds, false);
        let inner = bounds.shrink(12.);
        workspace::region(ui, "video-desk", inner, |ui| {
            egui::ScrollArea::vertical().id_salt("video-scroll").show(ui,|ui|{
                ui.horizontal(|ui|{ui.heading("H3");ui.weak("Experimental · 24 fps · Audio");});
                ui.add_enabled_ui(!self.video.working && self.video.pending.is_none(), |ui| {
                    ui.horizontal_wrapped(|ui| {
                        egui::ComboBox::from_id_salt("h3-ratio").selected_text(&self.video.selection.ratio).show_ui(ui, |ui| {
                            for ratio in h3::geometry::RATIOS { ui.selectable_value(&mut self.video.selection.ratio, ratio.into(), ratio); }
                        });
                        egui::ComboBox::from_id_salt("h3-orientation").selected_text(if self.video.selection.portrait {"Portrait"} else {"Landscape"}).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.video.selection.portrait, false, "Landscape");
                            ui.selectable_value(&mut self.video.selection.portrait, true, "Portrait");
                        });
                        egui::ComboBox::from_id_salt("h3-size").selected_text(&self.video.selection.quality).show_ui(ui, |ui| {
                            for quality in h3::geometry::QUALITIES { ui.selectable_value(&mut self.video.selection.quality, quality.into(), quality); }
                        });
                        ui.label("Seconds");
                        ui.add(egui::DragValue::new(&mut self.video.selection.seconds).range(0.25..=3600.).speed(0.25).max_decimals(3));
                    });
                });
                let geometry = self.video.selection.resolve();
                match geometry {
                    Ok(shape) => {
                        ui.label(format!("{} × {} · {} frames · {:.3} s · Audio {:.3} s", shape.width, shape.height, shape.frames, shape.seconds(), shape.audio_samples() as f64 / 32000.));
                        if shape != h3::Shape::default() { ui.weak(format!("GPU untested: may exceed VRAM. Decoded AV {:.0} MiB. Verified: 384 × 256 / 39 frames.", shape.raw_bytes() as f64 / 1048576.)); }
                        if shape.frames < 124 || shape.frames > 362 { ui.weak("Outside H3's reported training range (~5–15 s); motion or audio quality may degrade."); }
                    },
                    Err(_) => { ui.colored_label(helto::ERROR, "Invalid duration or decoded result exceeds the 2 GiB host-memory limit. Reduce size or duration."); }
                }
                ui.label("Prompt");
                ui.add_enabled_ui(!self.video.working&&self.video.pending.is_none(),|ui|{
                    helto::textarea(ui,&mut self.video.prompt,3,"Video prompt",self.prefs.privacy_mode,true);
                });
                ui.horizontal(|ui|{
                    let label=if self.video.pending.is_some()&&!self.video.working{"Retry"}else{"Generate video"};
                    if ui.add_enabled_ui(geometry.is_ok()&&self.video.available&&!self.video.working&&!self.video.saving&&!self.video.prompt.trim().is_empty(),|ui|helto::select_button(ui,label,true)).inner.clicked(){self.generate_video(ui.ctx());}
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
                let mut play=false;let mut save=false;
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
                        let label=if self.video.saving{"Saving"}else if view.saved.is_some(){"Saved"}else{"Save video"};
                        if ui.add_enabled_ui(!self.video.saving&&!self.video.working&&view.saved.is_none(),|ui|helto::select_button(ui,label,false)).inner.clicked(){save=true;}
                    });
                    if play {if view.started.is_some(){view.pause();}else if let Err(error)=view.play(){self.video.error=Some(error.into());}}
                }else {ui.painter().text(media.center(),egui::Align2::CENTER_CENTER,"Video preview",egui::FontId::proportional(14.),helto::DIM);}
                if save {self.save_video(ui.ctx());}
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
            bytes: Arc::new(vec![1, 2, 3]),
            av: short_av::Av {
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
        app.workspace = Workspace::Video;
        app.video.available = true;
        app.video.checked = true;
        app.video.prompt = "SYNTHETIC_PRIVATE_VIDEO_PROMPT".into();
        app.prefs.output_folder = root.path().join("explicit-save-only");
        (root, app, ctx)
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
    fn completed_preview_never_saves_until_explicit_save_and_failure_keeps_bytes() {
        let (root, mut app, ctx) = app();
        let blocked = root.path().join("blocked");
        std::fs::write(&blocked, b"own marker").unwrap();
        app.video_message(&ctx, Event::Finished(Ok(done())));
        assert!(!app.prefs.output_folder.exists());
        assert!(app.video.result.as_ref().unwrap().saved.is_none());
        let original = app.video.result.as_ref().unwrap().bytes.clone();
        app.prefs.output_folder = blocked;
        app.save_video(&ctx);
        for _ in 0..100 {
            app.receive(&ctx);
            if !app.video.saving {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!app.video.saving);
        assert!(app.video.error.is_some());
        assert!(Arc::ptr_eq(
            &original,
            &app.video.result.as_ref().unwrap().bytes
        ));
        assert!(app.video.pending.is_none());
        assert!(!app.video.working);
        app.prefs.output_folder = root.path().join("saved");
        app.save_video(&ctx);
        for _ in 0..100 {
            app.receive(&ctx);
            if !app.video.saving {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let path = app.video.result.as_ref().unwrap().saved.as_ref().unwrap();
        assert_eq!(path.extension().unwrap(), "mkv");
        assert_eq!(std::fs::read(path).unwrap(), *original);
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
            assert!(nodes.iter().any(|(_, n)| n.label() == Some("Save video")));
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
