//! Transient sampling images/animations shared by image desks and the Timeline.
use super::*;
use crate::client::sampling_preview::{Clip, Settings, Status};
use imagine_video_contract::contract::{ProjectId, SceneId};
#[cfg(any(test, feature = "ui-preview"))]
use serde_json::json;
use std::time::{Duration, Instant};

#[derive(Clone, PartialEq, Eq)]
enum Scope {
    Timeline { project: ProjectId, scene: SceneId },
    Image { workspace: Workspace },
}
#[derive(Clone, PartialEq, Eq)]
pub(super) struct Identity {
    epoch: u64,
    session: String,
    scope: Scope,
}
pub(super) enum Reply {
    Status {
        identity: Identity,
        saved: bool,
        result: Result<Status, String>,
    },
    Frame {
        identity: Identity,
        job: String,
        result: Result<Option<Clip>, String>,
    },
}

#[derive(Default)]
pub(super) struct State {
    identity: Option<Identity>,
    nonce: u64,
    status_pending: Option<u64>,
    frame_pending: Option<(u64, String)>,
    status_at: Option<Instant>,
    frame_at: Option<Instant>,
    status: Option<Status>,
    job: Option<String>,
    sequence: u64,
    clip: Option<Clip>,
    texture: Option<(usize, egui::TextureHandle)>,
    started: Option<Instant>,
    open: bool,
    draft: Settings,
    dirty: bool,
    saving: bool,
    notice: Option<String>,
}
impl State {
    pub(super) fn hide_texture(&mut self) {
        self.texture = None;
    }
    fn warning_text(&self) -> Option<&str> {
        self.notice.as_deref().or_else(|| {
            let status = self.status.as_ref()?;
            if self.job.is_none() || self.job != status.active_job {
                return None;
            }
            status.warning.as_deref().map(|code| match code {
                "image_preview_unavailable" => "This image runtime does not provide sampling previews yet; generation continues.",
                "fast_preview_failed" => "The sampling preview could not be rendered; generation continues.",
                _ => "The sampling preview is unavailable; generation continues.",
            })
        })
    }
    #[cfg(feature = "ui-preview")]
    pub(super) fn qa_state(&self) -> Value {
        json!({
            "job": self.job,
            "sequence": self.sequence,
            "frames": self.clip.as_ref().map(|clip| clip.metadata.frames),
            "frame_duration_ms": self.clip.as_ref().map(|clip| clip.metadata.frame_duration_ms),
            "has_texture": self.texture.is_some(),
            "has_clip": self.clip.is_some(),
            "current_animation_frame": self.frame_index(
                self.started.map_or(Duration::ZERO, |start| start.elapsed())
            ),
        })
    }
    fn next_nonce(&mut self) -> u64 {
        self.nonce = self.nonce.wrapping_add(1);
        self.nonce
    }
    /// Drop textures, RGB and all outstanding identities on navigation/session reset.
    pub(super) fn clear(&mut self) {
        let nonce = self.next_nonce();
        *self = Self {
            nonce,
            ..Self::default()
        };
    }
    fn bind(&mut self, identity: Identity) {
        if self.identity.as_ref() != Some(&identity) {
            self.clear();
            self.identity = Some(identity);
        }
    }
    fn set_job(&mut self, job: Option<String>) {
        if self.job != job {
            self.frame_pending = None;
            self.frame_at = None;
            self.sequence = 0;
            self.clip = None;
            self.texture = None;
            self.started = None;
            self.job = job;
        }
    }
    fn receive_status(
        &mut self,
        identity: &Identity,
        nonce: u64,
        saved: bool,
        result: Result<Status, String>,
    ) {
        if self.identity.as_ref() != Some(identity) || self.status_pending != Some(nonce) {
            return;
        }
        self.status_pending = None;
        self.saving = false;
        self.status_at = Some(Instant::now() + Duration::from_secs(1));
        match result {
            Ok(status) => {
                if saved || !self.dirty {
                    self.draft = status.settings;
                    self.dirty = false;
                }
                self.status = Some(status);
                self.notice = None;
            }
            Err(message) => self.notice = Some(message),
        }
    }
    fn receive_frame(
        &mut self,
        identity: &Identity,
        nonce: u64,
        job: &str,
        result: Result<Option<Clip>, String>,
    ) {
        if self.identity.as_ref() != Some(identity)
            || self.job.as_deref() != Some(job)
            || self.frame_pending.as_ref().map(|p| (p.0, p.1.as_str())) != Some((nonce, job))
        {
            return;
        }
        self.frame_pending = None;
        self.frame_at = Some(Instant::now() + Duration::from_millis(500));
        match result {
            Ok(Some(clip)) if clip.metadata.sequence > self.sequence => {
                self.sequence = clip.metadata.sequence;
                self.clip = Some(clip);
                self.texture = None;
                self.started = Some(Instant::now());
                self.notice = None;
            }
            Ok(_) => {}
            Err(message) => self.notice = Some(message),
        }
    }
    fn frame_index(&self, elapsed: Duration) -> usize {
        self.clip.as_ref().map_or(0, |clip| {
            ((elapsed.as_millis() / u128::from(clip.metadata.frame_duration_ms))
                % u128::from(clip.metadata.frames)) as usize
        })
    }
    fn settings_window(&mut self, ctx: &egui::Context) -> Option<Settings> {
        if !self.open {
            return None;
        }
        let image = self
            .identity
            .as_ref()
            .is_some_and(|id| matches!(id.scope, Scope::Image { .. }));
        let mut open = true;
        let mut close_requested = false;
        let mut save = None;
        egui::Window::new("Sampling preview")
            .id(egui::Id::new("sampling-preview-settings"))
            .open(&mut open)
            .default_width(340.)
            .resizable(false)
            .show(ctx, |ui| {
                let _canvas = helto::window_canvas(ui);
                ui.add_enabled_ui(!self.saving, |ui| {
                    let enabled = ui.checkbox(&mut self.draft.enabled, "Sampling preview");
                    self.dirty |= enabled.changed();
                    #[cfg(feature = "ui-preview")]
                    crate::app::timeline::qa::control(
                        ui,
                        "sampling-preview-enabled".into(),
                        &enabled,
                    );
                    egui::Grid::new("sampling-preview-settings-fields")
                        .num_columns(2)
                        .spacing(egui::vec2(16., 10.))
                        .show(ui, |ui| {
                            ui.label("Interval (seconds)");
                            let response = ui.add(
                                egui::DragValue::new(&mut self.draft.interval_seconds).speed(0.5),
                            );
                            self.dirty |= response.changed();
                            #[cfg(feature = "ui-preview")]
                            crate::app::timeline::qa::control(
                                ui,
                                "sampling-preview-interval".into(),
                                &response,
                            );
                            ui.end_row();
                            ui.label("Maximum edge (pixels)");
                            let response =
                                ui.add(egui::DragValue::new(&mut self.draft.max_resolution));
                            self.dirty |= response.changed();
                            #[cfg(feature = "ui-preview")]
                            crate::app::timeline::qa::control(
                                ui,
                                "sampling-preview-edge".into(),
                                &response,
                            );
                            ui.end_row();
                            if !image {
                                ui.label("Preview frames");
                                let response = ui.add(egui::DragValue::new(&mut self.draft.frames));
                                self.dirty |= response.changed();
                                #[cfg(feature = "ui-preview")]
                                crate::app::timeline::qa::control(
                                    ui,
                                    "sampling-preview-frames".into(),
                                    &response,
                                );
                                ui.end_row();
                            }
                        });
                });
                if let Some(status) = &self.status
                    && (status.settings.max_resolution != status.effective.max_resolution
                        || (!image && status.settings.frames != status.effective.frames))
                {
                    ui.weak(if image {
                        format!(
                            "Runtime preview: up to {} px",
                            status.effective.max_resolution
                        )
                    } else {
                        format!(
                            "Runtime preview: up to {} px · {} frames",
                            status.effective.max_resolution, status.effective.frames
                        )
                    });
                }
                if let Some(warning) = self.warning_text() {
                    ui.weak(warning);
                }
                if !self.draft.valid() {
                    ui.weak("Interval must be greater than zero.");
                }
                ui.weak("Applies to the next sampling job. Does not change seeds or the recipe.");
                ui.horizontal(|ui| {
                    let blocked = if self.status.is_none() {
                        Some("Settings have not loaded yet.")
                    } else if self.saving {
                        Some("Saving.")
                    } else if !self.draft.valid() {
                        Some("Fix the invalid values first.")
                    } else {
                        None
                    };
                    let apply =
                        helto::action(ui, crate::icons::Icon::Check, "Apply", true, blocked);
                    #[cfg(feature = "ui-preview")]
                    crate::app::timeline::qa::control(ui, "sampling-preview-apply".into(), &apply);
                    if apply.clicked() {
                        save = Some(self.draft);
                    }
                    let close = helto::icon_button(ui, crate::icons::Icon::Close, "Close", false);
                    #[cfg(feature = "ui-preview")]
                    crate::app::timeline::qa::control(ui, "sampling-preview-close".into(), &close);
                    if close.clicked() {
                        close_requested = true;
                    }
                    if self.saving || self.status_pending.is_some() {
                        ui.spinner();
                    }
                });
            });
        self.open = open && !close_requested;
        save
    }
    /// A frame has arrived for the running job, so the preview owns the viewer.
    pub(super) fn showing(&self) -> bool {
        self.job.is_some() && self.clip.is_some()
    }
    /// Paints a complete animation silently; at most one GPU texture is retained.
    fn viewer(
        &mut self,
        ui: &mut egui::Ui,
        privacy: bool,
        zoom: crate::settings::MediaZoom,
    ) -> bool {
        // A waiting job must not hide an existing Take. Only actual RGB takes over.
        if !self.showing() {
            self.texture = None;
            return false;
        }
        let available = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(available, egui::Sense::hover());
        #[cfg(feature = "ui-preview")]
        crate::app::timeline::qa::control(ui, "sampling-preview-canvas".into(), &response);
        let reveal = !privacy || (response.hovered() && ui.input(|input| input.focused));
        if !reveal {
            self.texture = None;
        }
        let footer_height = 38_f32.min(rect.height());
        let picture = egui::Rect::from_min_max(rect.min, rect.max - egui::vec2(0., footer_height));
        helto::private_scope(ui, !reveal, |ui| {
            if reveal {
                if let Some(clip) = &self.clip {
                    let elapsed = self.started.map_or(Duration::ZERO, |start| start.elapsed());
                    let index = self.frame_index(elapsed);
                    if self
                        .texture
                        .as_ref()
                        .is_none_or(|(current, _)| *current != index)
                        && let Some(frame) = clip.frame(index)
                    {
                        let image = egui::ColorImage::from_rgb(
                            [clip.metadata.width as usize, clip.metadata.height as usize],
                            frame,
                        );
                        if let Some((current, texture)) = &mut self.texture {
                            texture.set(image, egui::TextureOptions::LINEAR);
                            *current = index;
                        } else {
                            self.texture = Some((
                                index,
                                ui.ctx().load_texture(
                                    "sampling-preview-frame",
                                    image,
                                    egui::TextureOptions::LINEAR,
                                ),
                            ));
                        }
                    }
                    if let Some((_, texture)) = &self.texture {
                        let destination = helto::media_rect(
                            picture,
                            texture.size_vec2(),
                            zoom,
                            ui.ctx().pixels_per_point(),
                        );
                        ui.painter().with_clip_rect(picture).image(
                            texture.id(),
                            destination,
                            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
                            egui::Color32::WHITE,
                        );
                    }
                    let text = clip_label(&clip.metadata);
                    paint_label(ui, rect, &text);
                    if clip.metadata.frames > 1 {
                        let remaining = u128::from(clip.metadata.frame_duration_ms)
                            - elapsed.as_millis() % u128::from(clip.metadata.frame_duration_ms);
                        ui.ctx()
                            .request_repaint_after(Duration::from_millis(remaining as u64));
                    }
                } else {
                    paint_label(ui, rect, "Sampling preview · waiting for first clip");
                }
                if let Some(warning) = self.warning_text() {
                    let warning_rect = egui::Rect::from_min_max(
                        egui::pos2(rect.left(), rect.bottom() - 18.),
                        rect.max,
                    );
                    let galley = ui.painter().layout(
                        warning.to_owned(),
                        egui::FontId::proportional(11.),
                        helto::DIM,
                        rect.width().max(1.),
                    );
                    ui.painter().with_clip_rect(warning_rect).galley(
                        warning_rect.min,
                        galley,
                        helto::DIM,
                    );
                }
            }
        });
        if !reveal {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Sampling preview hidden",
                egui::FontId::proportional(13.),
                helto::DIM,
            );
        }
        true
    }
}
fn paint_label(ui: &egui::Ui, rect: egui::Rect, text: &str) {
    let position = egui::pos2(rect.left(), rect.bottom() - 36.);
    let galley = ui.painter().layout(
        text.into(),
        egui::FontId::proportional(12.),
        helto::DIM,
        rect.width().max(1.),
    );
    ui.painter()
        .with_clip_rect(rect)
        .galley(position, galley, helto::DIM);
}

fn clip_label(metadata: &crate::client::sampling_preview::Metadata) -> String {
    if metadata.frames == 1 {
        format!("Sampling preview · step {}", metadata.step)
    } else {
        format!(
            "Sampling preview · step {} · {:.2} s · silent",
            metadata.step, metadata.source_duration_seconds
        )
    }
}
fn own_running_job(working: bool, cancelling: bool, candidate: Option<&str>, active: &str) -> bool {
    working && !cancelling && candidate == Some(active)
}

impl ImagineApp {
    fn sampling_preview_identity(&self) -> Option<Identity> {
        let (epoch, session) = self.timeline.sampling_preview_session()?;
        let scope = if self.workspace == Workspace::Video {
            let (project, scene) = self.timeline.sampling_preview_selection()?;
            Scope::Timeline { project, scene }
        } else {
            Scope::Image {
                workspace: self.workspace,
            }
        };
        Some(Identity {
            epoch,
            session,
            scope,
        })
    }
    fn sampling_preview_current_job(&self) -> Option<String> {
        let state = &self.sampling_preview;
        let identity = state.identity.as_ref()?;
        let status = state.status.as_ref()?;
        if !status.settings.enabled {
            return None;
        }
        let job = status.active_job.as_ref()?;
        let matches = match &identity.scope {
            Scope::Timeline { project, scene } => self
                .timeline
                .sampling_preview_job_matches(job, project, scene),
            Scope::Image { workspace } => {
                let desk = self.desk_for(*workspace);
                own_running_job(
                    desk.working,
                    desk.cancelling,
                    desk.pending
                        .as_ref()
                        .and_then(|pending| pending.job.as_deref()),
                    job,
                ) || (*workspace == Workspace::Image
                    && self.queue.jobs.iter().any(|queued| {
                        own_running_job(
                            queued.working,
                            queued.cancelling,
                            queued.submission.job.as_deref(),
                            job,
                        )
                    }))
            }
        };
        matches.then(|| job.clone())
    }
    /// One poll owner for whichever workspace is currently visible.
    pub(super) fn sampling_preview_poll(&mut self, ctx: &egui::Context) {
        self.timeline_session(ctx);
        let (Some(identity), Some(client)) =
            (self.sampling_preview_identity(), self.client.clone())
        else {
            self.sampling_preview.clear();
            return;
        };
        self.sampling_preview.bind(identity.clone());
        let job = self.sampling_preview_current_job();
        self.sampling_preview.set_job(job);
        let now = Instant::now();
        let state = &mut self.sampling_preview;
        ctx.request_repaint_after(Duration::from_millis(500));
        if state.status_pending.is_none() && state.status_at.is_none_or(|at| at <= now) {
            let nonce = state.next_nonce();
            state.status_pending = Some(nonce);
            let identity = identity.clone();
            let client = client.clone();
            let tx = self.tx.clone();
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let result = runtime().block_on(client.sampling_preview_status(&identity.session));
                send(
                    &tx,
                    &ctx,
                    Message::SamplingPreview {
                        nonce,
                        reply: Reply::Status {
                            identity,
                            saved: false,
                            result,
                        },
                    },
                );
            });
        }
        let Some(job) = state.job.clone() else {
            return;
        };
        if state.frame_pending.is_some() || state.frame_at.is_some_and(|at| at > now) {
            return;
        }
        if state
            .status
            .as_ref()
            .and_then(|s| s.metadata.as_ref())
            .is_some_and(|m| m.sequence <= state.sequence)
        {
            return;
        }
        let nonce = state.next_nonce();
        state.frame_pending = Some((nonce, job.clone()));
        let after = state.sequence;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.sampling_preview_frame(&identity.session, &job, after));
            send(
                &tx,
                &ctx,
                Message::SamplingPreview {
                    nonce,
                    reply: Reply::Frame {
                        identity,
                        job,
                        result,
                    },
                },
            );
        });
    }
    pub(super) fn sampling_preview_reply(&mut self, ctx: &egui::Context, nonce: u64, reply: Reply) {
        // Navigation/cancel can occur before the next content poll.
        let Some(identity) = self.sampling_preview_identity() else {
            self.sampling_preview.clear();
            return;
        };
        self.sampling_preview.bind(identity);
        match reply {
            Reply::Status {
                identity,
                saved,
                result,
            } => {
                self.sampling_preview
                    .receive_status(&identity, nonce, saved, result);
                let current = self.sampling_preview_current_job();
                self.sampling_preview.set_job(current);
            }
            Reply::Frame {
                identity,
                job,
                result,
            } => {
                let current = self.sampling_preview_current_job();
                self.sampling_preview.set_job(current);
                self.sampling_preview
                    .receive_frame(&identity, nonce, &job, result);
            }
        }
        ctx.request_repaint();
    }
    pub(super) fn sampling_preview_controls(&mut self, ui: &mut egui::Ui) {
        let tooltip = self.sampling_preview.warning_text().map_or_else(
            || "Sampling preview settings".to_owned(),
            |warning| format!("Sampling preview settings\n{warning}"),
        );
        let response = helto::icon_button(
            ui,
            crate::icons::Icon::Eye,
            &tooltip,
            self.sampling_preview.open,
        );
        #[cfg(feature = "ui-preview")]
        crate::app::timeline::qa::control(ui, "sampling-preview-settings".into(), &response);
        if response.clicked() {
            self.sampling_preview.open = !self.sampling_preview.open;
        }
        let Some(settings) = self.sampling_preview.settings_window(ui.ctx()) else {
            return;
        };
        let (Some(identity), Some(client)) =
            (self.sampling_preview_identity(), self.client.clone())
        else {
            return;
        };
        let state = &mut self.sampling_preview;
        let nonce = state.next_nonce();
        state.status_pending = Some(nonce);
        state.saving = true;
        let tx = self.tx.clone();
        let ctx = ui.ctx().clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.sampling_preview_settings(&identity.session, &settings));
            send(
                &tx,
                &ctx,
                Message::SamplingPreview {
                    nonce,
                    reply: Reply::Status {
                        identity,
                        saved: true,
                        result,
                    },
                },
            );
        });
    }
    pub(super) fn sampling_preview_viewer(&mut self, ui: &mut egui::Ui) -> bool {
        self.sampling_preview
            .viewer(ui, self.prefs.privacy_mode, self.prefs.media_zoom)
    }
    /// Actual pixels or fit for the video viewer and sampling previews.
    pub(super) fn media_zoom_toggle(&mut self, ui: &mut egui::Ui, picture: bool) {
        let response = helto::zoom_toggle(
            ui,
            &mut self.prefs.media_zoom,
            (!picture).then_some("Nothing with a picture is in the viewer."),
        );
        #[cfg(feature = "ui-preview")]
        crate::app::timeline::qa::control(ui, "media-zoom".into(), &response);
        if response.clicked() {
            self.save_preferences();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::sampling_preview::Metadata;
    fn identity() -> Identity {
        Identity {
            epoch: 7,
            session: "session-a".into(),
            scope: Scope::Timeline {
                project: ProjectId("project-a".into()),
                scene: SceneId("scene-a".into()),
            },
        }
    }
    fn clip(sequence: u64, frames: u32, duration_ms: u64) -> Clip {
        Clip::new(
            Metadata {
                sequence,
                step: 2,
                width: 1,
                height: 1,
                frames,
                frame_duration_ms: duration_ms,
                source_duration_seconds: f64::from(frames) * duration_ms as f64 / 1000.,
            },
            vec![37; frames as usize * 3],
        )
        .unwrap()
    }
    fn running() -> State {
        let mut state = State::default();
        state.bind(identity());
        state.set_job(Some("job-a".into()));
        state
    }
    fn deliver(state: &mut State, sequence: u64) {
        let nonce = state.next_nonce();
        state.frame_pending = Some((nonce, "job-a".into()));
        state.receive_frame(
            &identity(),
            nonce,
            "job-a",
            Ok(Some(clip(sequence, 24, 417))),
        );
    }
    fn image_identity(workspace: Workspace) -> Identity {
        Identity {
            epoch: 7,
            session: "session-a".into(),
            scope: Scope::Image { workspace },
        }
    }
    fn status() -> Status {
        Status {
            settings: Settings::default(),
            effective: Settings::default(),
            active_job: Some("job-a".into()),
            metadata: None,
            warning: None,
        }
    }
    #[test]
    fn image_preview_scope_rejects_other_desks_and_late_timeline_replies() {
        for old in [
            identity(),
            image_identity(Workspace::ImageEdit),
            image_identity(Workspace::CharacterSheet),
        ] {
            let mut state = State::default();
            let current = image_identity(Workspace::Image);
            state.bind(current.clone());
            state.set_job(Some("job-a".into()));
            let nonce = state.next_nonce();
            state.frame_pending = Some((nonce, "job-a".into()));
            state.status_pending = Some(nonce);
            state.receive_frame(&old, nonce, "job-a", Ok(Some(clip(1, 1, 1000))));
            state.receive_status(&old, nonce, false, Ok(status()));
            assert!(state.clip.is_none() && state.status.is_none());
            state.receive_frame(&current, nonce, "job-a", Ok(Some(clip(1, 1, 1000))));
            assert_eq!(state.sequence, 1);
            assert_eq!(state.frame_index(Duration::from_secs(99)), 0);
            assert_eq!(
                clip_label(&state.clip.as_ref().unwrap().metadata),
                "Sampling preview · step 2"
            );
            state.bind(old);
            assert!(state.clip.is_none() && state.texture.is_none() && state.job.is_none());
        }
    }
    #[test]
    fn image_preview_requires_own_active_non_cancelled_desk_or_queue_job() {
        let preferences =
            serde_json::from_value(json!({"output_folder":"/own-preview-fixture/output"})).unwrap();
        let mut app = ImagineApp::empty(
            PathBuf::from("/own-preview-fixture"),
            Connection {
                url: "http://127.0.0.1:1".into(),
            },
            preferences,
        );
        app.sampling_preview.bind(image_identity(Workspace::Image));
        app.sampling_preview.status = Some(status());
        let submission = Submission {
            upscale_source: None,
            request: json!({}),
            key: "own-key".into(),
            folder: PathBuf::from("/own-preview-fixture/output"),
            job: Some("job-a".into()),
        };
        app.image.pending = Some(submission.clone());
        assert!(app.sampling_preview_current_job().is_none());
        app.image.working = true;
        assert_eq!(app.sampling_preview_current_job().as_deref(), Some("job-a"));
        app.image.cancelling = true;
        assert!(app.sampling_preview_current_job().is_none());
        app.image.cancelling = false;
        app.image.pending.as_mut().unwrap().job = Some("other-job".into());
        assert!(app.sampling_preview_current_job().is_none());
        app.image.working = false;
        app.queue.stage(vec![(
            submission,
            queue::Snapshot {
                prompt: String::new(),
                preferences: app.prefs.clone(),
                references: vec![],
            },
        )]);
        assert_eq!(app.sampling_preview_current_job().as_deref(), Some("job-a"));
        app.queue.jobs[0].cancelling = true;
        assert!(app.sampling_preview_current_job().is_none());
        app.queue.jobs[0].cancelling = false;
        app.sampling_preview
            .bind(image_identity(Workspace::CharacterSheet));
        app.sampling_preview.status = Some(status());
        assert!(app.sampling_preview_current_job().is_none());
        app.sampling_preview.bind(image_identity(Workspace::Image));
        let mut disabled = status();
        disabled.settings.enabled = false;
        app.sampling_preview.status = Some(disabled);
        assert!(app.sampling_preview_current_job().is_none());
    }
    #[test]
    fn image_preview_warning_is_nonfatal_scoped_and_does_not_expose_unknown_code() {
        let mut state = running();
        let mut status = status();
        status.warning = Some("image_preview_unavailable".into());
        state.status = Some(status);
        assert_eq!(
            state.warning_text(),
            Some(
                "This image runtime does not provide sampling previews yet; generation continues."
            )
        );
        state.status.as_mut().unwrap().warning = Some("private-unexpected-diagnostic".into());
        assert_eq!(
            state.warning_text(),
            Some("The sampling preview is unavailable; generation continues.")
        );
        state.set_job(None);
        assert!(state.warning_text().is_none());
    }
    #[test]
    fn image_preview_settings_preserve_video_frame_setting_in_small_windows() {
        for width in [1100., 600.] {
            let ctx = egui::Context::default();
            helto::install(&ctx);
            let mut state = State::default();
            state.bind(image_identity(Workspace::Image));
            state.open = true;
            state.draft.frames = 37;
            state.status = Some(status());
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width, 760.),
                    )),
                    focused: true,
                    ..Default::default()
                },
                |ui| {
                    assert!(state.settings_window(ui.ctx()).is_none());
                },
            );
            output.textures_delta.clear();
            assert_eq!(state.draft.frames, 37);
        }
    }
    #[test]
    fn sampling_preview_keeps_only_latest_and_loops_complete_animation() {
        let mut state = running();
        deliver(&mut state, 2);
        deliver(&mut state, 1);
        assert_eq!(state.sequence, 2);
        assert_eq!(state.clip.as_ref().unwrap().metadata.sequence, 2);
        assert_eq!(state.frame_index(Duration::from_millis(417 * 23)), 23);
        assert_eq!(state.frame_index(Duration::from_millis(417 * 24)), 0);
        assert_eq!(state.frame_index(Duration::from_secs(1)), 2);
    }
    #[test]
    fn sampling_preview_rejects_old_epoch_session_project_scene_job_nonce() {
        for field in 0..5 {
            let mut state = running();
            let nonce = state.next_nonce();
            state.frame_pending = Some((nonce, "job-a".into()));
            let mut old = identity();
            match field {
                0 => old.epoch -= 1,
                1 => old.session = "session-old".into(),
                2 => {
                    old.scope = Scope::Timeline {
                        project: ProjectId("project-old".into()),
                        scene: SceneId("scene-a".into()),
                    }
                }
                3 => {
                    old.scope = Scope::Timeline {
                        project: ProjectId("project-a".into()),
                        scene: SceneId("scene-old".into()),
                    }
                }
                _ => {}
            }
            state.receive_frame(
                &old,
                if field == 4 { nonce + 1 } else { nonce },
                "job-a",
                Ok(Some(clip(1, 1, 40))),
            );
            assert!(state.clip.is_none());
        }
        let mut state = running();
        let nonce = state.next_nonce();
        state.frame_pending = Some((nonce, "job-a".into()));
        state.set_job(Some("job-b".into()));
        state.receive_frame(&identity(), nonce, "job-a", Ok(Some(clip(1, 1, 40))));
        assert!(state.clip.is_none());
    }
    #[test]
    fn sampling_preview_cancel_navigation_and_new_job_clear_pixels() {
        let mut state = running();
        deliver(&mut state, 99);
        state.set_job(None);
        assert!(state.clip.is_none() && state.texture.is_none());
        state.set_job(Some("job-a".into()));
        deliver(&mut state, 1);
        assert_eq!(state.sequence, 1);
        state.clear();
        assert!(state.clip.is_none() && state.texture.is_none() && state.identity.is_none());
    }
    #[test]
    fn sampling_preview_status_does_not_overwrite_dirty_settings_or_accept_old_response() {
        let mut state = running();
        let nonce = state.next_nonce();
        state.status_pending = Some(nonce);
        state.draft.frames = 47;
        state.dirty = true;
        state.receive_status(
            &identity(),
            nonce,
            false,
            Ok(Status {
                settings: Settings::default(),
                effective: Settings::default(),
                active_job: Some("job-a".into()),
                metadata: None,
                warning: None,
            }),
        );
        assert_eq!(state.draft.frames, 47);
        state.clear();
        state.receive_status(
            &identity(),
            nonce,
            true,
            Ok(Status {
                settings: Settings::default(),
                effective: Settings::default(),
                active_job: None,
                metadata: None,
                warning: None,
            }),
        );
        assert!(state.status.is_none());
    }
    #[test]
    fn sampling_preview_egui_hidden_drops_texture_and_never_uploads_pixels() {
        let ctx = egui::Context::default();
        helto::install(&ctx);
        let mut state = running();
        deliver(&mut state, 1);
        let show = |state: &mut State, privacy| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(600., 300.),
                    )),
                    focused: true,
                    ..Default::default()
                },
                |ui| {
                    assert!(state.viewer(ui, privacy, Default::default()));
                },
            )
        };
        show(&mut state, false).textures_delta.clear();
        assert!(state.texture.is_some());
        let previous_texture = state.texture.as_ref().unwrap().1.id();
        let mut output = show(&mut state, true);
        assert!(state.texture.is_none());
        // New font glyphs may upload an atlas; no sampling pixels may upload.
        assert!(
            output
                .textures_delta
                .set
                .iter()
                .all(|(id, _)| *id != previous_texture)
        );
        output.textures_delta.clear();
    }
    #[test]
    fn sampling_preview_settings_window_normal_and_small_stays_in_viewport() {
        for width in [1100., 600.] {
            let ctx = egui::Context::default();
            helto::install(&ctx);
            let mut state = running();
            state.open = true;
            state.status = Some(Status {
                settings: Settings::default(),
                effective: Settings::default(),
                active_job: None,
                metadata: None,
                warning: None,
            });
            let mut bounds = None;
            for _ in 0..2 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 760.),
                        )),
                        focused: true,
                        ..Default::default()
                    },
                    |ui| {
                        assert!(state.settings_window(ui.ctx()).is_none());
                        bounds = ui.ctx().memory(|memory| {
                            memory.area_rect(egui::Id::new("sampling-preview-settings"))
                        });
                    },
                );
                output.textures_delta.clear();
            }
            let rect = bounds.unwrap();
            assert!(rect.left() >= 0. && rect.right() <= width);
            assert!(rect.top() >= 0. && rect.bottom() <= 760.);
        }
    }
}
