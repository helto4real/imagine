use crate::{
    client::{self, Client, Completed, Failure, Preview, Progress, Reference, Submission},
    helto,
    settings::{Connection, Preferences},
};
use eframe::egui;
use serde_json::Value;
use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, Sender},
    time::Duration,
};

struct ReferenceView {
    id: String,
    name: String,
    texture: egui::TextureHandle,
}
enum Message {
    Connected(Result<(Client, Value), String>),
    Picked(Vec<PathBuf>),
    Folder(Option<PathBuf>),
    Uploaded(Result<Reference, Failure>),
    UploadsFinished,
    Progress(Progress),
    Finished(Result<Completed, Failure>),
    Cancelled(String, Result<(), Failure>),
}
fn send(tx: &Sender<Message>, ctx: &egui::Context, msg: Message) {
    let _ = tx.send(msg);
    ctx.request_repaint();
}
fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("client runtime")
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Workspace {
    Image,
    Video,
}

pub struct ImagineApp {
    project: PathBuf,
    connection: Connection,
    prefs: Preferences,
    client: Option<Client>,
    metadata: Option<Value>,
    prompt: String,
    references: Vec<ReferenceView>,
    reference_directory: Option<PathBuf>,
    pending: Option<Submission>,
    last_generation: Option<Value>,
    working: bool,
    uploading: bool,
    connecting: bool,
    picking_folder: bool,
    cancelling: bool,
    status: String,
    attempt_started: Option<std::time::Instant>,
    last_elapsed: Option<Duration>,
    sampling: Option<client::SamplingProgress>,
    error: Option<String>,
    result: Option<(egui::TextureHandle, PathBuf, Option<u64>)>,
    actual_pixels: bool,
    pan_enabled: bool,
    workspace: Workspace,
    settings_open: bool,
    settings_error: Option<String>,
    image_pan: egui::Vec2,
    tx: Sender<Message>,
    rx: Receiver<Message>,
}
impl ImagineApp {
    pub fn new(
        project: PathBuf,
        connection: Connection,
        prefs: Preferences,
        ctx: &egui::Context,
    ) -> Self {
        helto::install(ctx);
        let mut app = Self::empty(project, connection, prefs);
        app.connect(ctx);
        app
    }
    fn empty(project: PathBuf, connection: Connection, prefs: Preferences) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            project,
            connection,
            prefs,
            client: None,
            metadata: None,
            prompt: String::new(),
            references: Vec::new(),
            reference_directory: None,
            pending: None,
            last_generation: None,
            working: false,
            uploading: false,
            connecting: false,
            picking_folder: false,
            cancelling: false,
            status: String::new(),
            attempt_started: None,
            last_elapsed: None,
            sampling: None,
            error: None,
            result: None,
            actual_pixels: false,
            pan_enabled: true,
            workspace: Workspace::Image,
            settings_open: false,
            settings_error: None,
            image_pan: egui::Vec2::ZERO,
            tx,
            rx,
        }
    }
    fn connect(&mut self, ctx: &egui::Context) {
        self.connecting = true;
        self.error = None;
        let connection = self.connection.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                let client = Client::connect(&connection).map_err(|e| e.to_string())?;
                let data = client.metadata().await.map_err(|e| e.message)?;
                if data["available"] != true {
                    return Err("The model is unavailable on the server.".into());
                }
                Ok((client, data))
            });
            send(&tx, &ctx, Message::Connected(result));
        });
    }
    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.rx.try_recv() {
            match message {
                Message::Connected(value) => {
                    self.connecting = false;
                    match value {
                        Ok((client, data)) => {
                            self.client = Some(client);
                            self.metadata = Some(data);
                        }
                        Err(e) => {
                            self.client = None;
                            self.metadata = None;
                            self.error = Some(e);
                        }
                    }
                }
                Message::Picked(paths) => {
                    if paths.is_empty() {
                        self.uploading = false;
                        continue;
                    }
                    self.reference_directory = paths
                        .last()
                        .and_then(|path| path.parent())
                        .map(|directory| directory.to_path_buf());
                    if paths.len() + self.references.len() > 10 {
                        self.uploading = false;
                        self.error = Some("Up to ten reference images. Select fewer files.".into());
                        continue;
                    }
                    let Some(client) = self.client.clone() else {
                        self.uploading = false;
                        continue;
                    };
                    let tx = self.tx.clone();
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        runtime().block_on(async {
                            for path in paths {
                                send(&tx, &ctx, Message::Uploaded(client.upload(&path).await));
                            }
                        });
                        send(&tx, &ctx, Message::UploadsFinished);
                    });
                }
                Message::Uploaded(result) => match result {
                    Ok(reference) => {
                        let texture = texture(ctx, &reference.preview, &reference.id);
                        self.references.push(ReferenceView {
                            id: reference.id,
                            name: reference.name,
                            texture,
                        });
                    }
                    Err(e) => self.error = Some(e.message),
                },
                Message::UploadsFinished => self.uploading = false,
                Message::Folder(Some(folder)) => {
                    self.picking_folder = false;
                    let prefs = Preferences {
                        output_folder: folder,
                        ..self.prefs.clone()
                    };
                    match prefs.save(&self.project) {
                        Ok(()) => { self.prefs = prefs; self.settings_error = None; }
                        Err(_) => self.settings_error = Some("Could not save the folder choice. Check write permissions for .imagine.".into()),
                    }
                }
                Message::Folder(None) => self.picking_folder = false,
                Message::Progress(Progress::Submitted(job)) => {
                    if let Some(p) = &mut self.pending {
                        p.job = Some(job);
                    }
                }
                Message::Progress(Progress::Status { label, sampling }) => {
                    self.status = label;
                    self.sampling = sampling;
                }
                Message::Finished(result) => {
                    let elapsed = self.attempt_started.take().map(|start| start.elapsed());
                    self.working = false;
                    self.sampling = None;
                    self.cancelling = false;
                    match result {
                        Ok(done) => {
                            self.last_elapsed = elapsed;
                            self.result =
                                Some((texture(ctx, &done.preview, "result"), done.file, done.seed));
                            self.image_pan = egui::Vec2::ZERO;
                            self.last_generation = self.pending.take().map(|job| job.request);
                            self.status = "Saved".into();
                            self.error = None;
                        }
                        Err(error) => {
                            self.error = Some(error.message);
                            self.status.clear();
                            if !error.retryable {
                                self.pending = None;
                            }
                        }
                    }
                }
                Message::Cancelled(job, result) => {
                    if self.pending.as_ref().and_then(|p| p.job.as_ref()) != Some(&job) {
                        continue;
                    }
                    if let Err(e) = result {
                        self.cancelling = false;
                        self.error = Some(e.message);
                    }
                }
            }
        }
    }
    fn generate(&mut self, ctx: &egui::Context) {
        let Some(client) = self.client.clone() else {
            return;
        };
        if self.pending.is_none() {
            let refs: Vec<_> = self.references.iter().map(|r| r.id.clone()).collect();
            let Some(metadata) = &self.metadata else {
                return;
            };
            let request = match client::make_request(
                &self.prompt,
                &self.prefs.negative_prompt,
                &refs,
                metadata,
                &self.prefs.generation,
            ) {
                Ok(v) => v,
                Err(e) => {
                    self.error = Some(e.to_string());
                    return;
                }
            };
            self.pending = Some(Submission {
                request,
                key: uuid::Uuid::new_v4().to_string(),
                folder: self.prefs.output_folder.clone(),
                job: None,
            });
        }
        let pending = self.pending.as_mut().unwrap();
        pending.folder = self.prefs.output_folder.clone();
        let submission = pending.clone();
        self.attempt_started = Some(std::time::Instant::now());
        self.working = true;
        self.status = "Submitting".into();
        self.sampling = None;
        self.error = None;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime()
                .block_on(client.generate(submission, |p| send(&tx, &ctx, Message::Progress(p))));
            send(&tx, &ctx, Message::Finished(result));
        });
    }
    fn regeneration(&self) -> Option<Submission> {
        if self.working || self.pending.is_some() {
            return None;
        }
        Some(Submission {
            request: self.last_generation.clone()?,
            key: uuid::Uuid::new_v4().to_string(),
            folder: self.prefs.output_folder.clone(),
            job: None,
        })
    }
    fn choose_references(&mut self, ctx: &egui::Context) {
        self.uploading = true;
        self.error = None;
        let directory = self.reference_directory.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut dialog = rfd::FileDialog::new()
                .set_title("Select reference images")
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"]);
            if let Some(directory) = directory {
                dialog = dialog.set_directory(directory);
            }
            let files = dialog.pick_files().unwrap_or_default();
            send(&tx, &ctx, Message::Picked(files));
        });
    }
    fn choose_folder(&mut self, ctx: &egui::Context) {
        self.picking_folder = true;
        let folder = self.prefs.output_folder.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let folder = rfd::FileDialog::new()
                .set_title("Save images to")
                .set_directory(folder)
                .pick_folder();
            send(&tx, &ctx, Message::Folder(folder));
        });
    }
    fn cancel(&mut self, ctx: &egui::Context) {
        let Some(job) = self.pending.as_ref().and_then(|p| p.job.clone()) else {
            return;
        };
        let Some(client) = self.client.clone() else {
            return;
        };
        self.cancelling = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.cancel(&job));
            send(&tx, &ctx, Message::Cancelled(job, result));
        });
    }
    fn save_preferences(&mut self) {
        self.settings_error = self
            .prefs
            .save(&self.project)
            .err()
            .map(|_| "Could not save settings. Check write permissions for .imagine.".into());
    }
}

#[path = "workspace.rs"]
mod workspace;

#[cfg(any(test, feature = "ui-preview"))]
#[path = "preview.rs"]
mod preview;

fn local_time(time: std::time::SystemTime) -> String {
    let Ok(duration) = time.duration_since(std::time::UNIX_EPOCH) else {
        return "—".into();
    };
    let Ok(seconds) = libc::time_t::try_from(duration.as_secs()) else {
        return "—".into();
    };
    let mut local = std::mem::MaybeUninit::<libc::tm>::uninit();
    // localtime_r writes into caller-owned storage and uses the host timezone.
    unsafe {
        if libc::localtime_r(&seconds, local.as_mut_ptr()).is_null() {
            return "—".into();
        }
        let local = local.assume_init();
        format!(
            "{:02}:{:02}:{:02}",
            local.tm_hour, local.tm_min, local.tm_sec
        )
    }
}
fn private_image(
    ui: &mut egui::Ui,
    texture: &egui::TextureHandle,
    size: egui::Vec2,
    privacy_mode: bool,
) -> egui::Response {
    let mut pan = egui::Vec2::ZERO;
    image_view(ui, texture, size, privacy_mode, false, false, &mut pan)
}
fn image_view(
    ui: &mut egui::Ui,
    texture: &egui::TextureHandle,
    size: egui::Vec2,
    privacy_mode: bool,
    actual_pixels: bool,
    pan_enabled: bool,
    pan: &mut egui::Vec2,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        size,
        if actual_pixels && pan_enabled {
            egui::Sense::drag()
        } else {
            egui::Sense::hover()
        },
    );
    let visible_rect = rect.intersect(ui.clip_rect());
    let reveal = !privacy_mode
        || response.hovered()
            && ui.input(|input| {
                input.focused
                    && input
                        .pointer
                        .hover_pos()
                        .is_some_and(|pos| visible_rect.contains(pos))
            });
    ui.painter().rect_filled(rect, 0, helto::MEDIA);
    if reveal {
        let original = texture.size_vec2();
        let pixels_per_point = ui.ctx().pixels_per_point();
        let scale = if actual_pixels {
            1. / pixels_per_point
        } else {
            (rect.width() / original.x).min(rect.height() / original.y)
        };
        let image_size = original * scale;
        if actual_pixels {
            if pan_enabled {
                *pan += response.drag_delta();
            }
            let limit = ((image_size - rect.size()) * 0.5).max(egui::Vec2::ZERO);
            *pan = pan.clamp(-limit, limit);
        }
        let mut origin = rect.center() - image_size * 0.5;
        if actual_pixels {
            origin += *pan;
            // Align texels to physical pixels, including on HiDPI displays.
            origin = egui::pos2(
                (origin.x * pixels_per_point).round() / pixels_per_point,
                (origin.y * pixels_per_point).round() / pixels_per_point,
            );
        }
        let image_rect = egui::Rect::from_min_size(origin, image_size);
        ui.painter().with_clip_rect(visible_rect).image(
            texture.id(),
            image_rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.)),
            egui::Color32::WHITE,
        );
    } else {
        // Do not submit the private image to the painter while concealed.
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Hidden",
            egui::FontId::proportional(12.),
            helto::DIM,
        );
    }
    if actual_pixels && pan_enabled {
        let cursor = if response.dragged() {
            egui::CursorIcon::Grabbing
        } else {
            egui::CursorIcon::Grab
        };
        response.on_hover_cursor(cursor)
    } else {
        response
    }
}
fn texture(ctx: &egui::Context, preview: &Preview, id: &str) -> egui::TextureHandle {
    ctx.load_texture(
        id,
        egui::ColorImage::from_rgba_unmultiplied([preview.width, preview.height], &preview.pixels),
        egui::TextureOptions::LINEAR,
    )
}
impl eframe::App for ImagineApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.receive(ctx);
    }
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.content(ui);
    }
}

#[cfg(test)]
#[path = "preview_tests.rs"]
mod tests;
