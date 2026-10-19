use crate::{
    client::{
        self, Client, Completed, Failure, OutputRole, Preview, Progress, Reference, Submission,
    },
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

#[derive(Clone)]
struct ReferenceView {
    number: usize,
    id: String,
    name: String,
    texture: egui::TextureHandle,
}
enum Message {
    Queue(queue::Event),
    Enhancement(enhancement::Event),
    Connected(Result<(Client, crate::model_catalog::LibrarySnapshot), String>),
    Library(Result<crate::model_catalog::LibrarySnapshot, String>),
    ModelFolder(Option<PathBuf>),
    Picked(Workspace, Vec<PathBuf>),
    Folder(Option<PathBuf>),
    Uploaded(Workspace, Result<Reference, Failure>),
    UploadsFinished(Workspace),
    Progress(Workspace, Progress),
    Finished(Workspace, Result<Completed, Failure>),
    Cancelled(Workspace, String, Result<(), Failure>),
    ClipboardFailed,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Workspace {
    Image,
    ImageEdit,
    CharacterSheet,
    Video,
}

#[derive(Clone)]
struct ResultView {
    asset_id: String,
    role: OutputRole,
    texture: egui::TextureHandle,
    file: PathBuf,
}
#[derive(Clone)]
struct Results {
    assets: Vec<ResultView>,
    seed: Option<u64>,
}
struct GenerationDesk {
    prompt: String,
    character_description: String,
    clothing_description: String,
    references: Vec<ReferenceView>,
    reference_directory: Option<PathBuf>,
    pending: Option<Submission>,
    working: bool,
    uploading: bool,
    cancelling: bool,
    status: String,
    attempt_started: Option<std::time::Instant>,
    last_elapsed: Option<Duration>,
    sampling: Option<client::SamplingProgress>,
    error: Option<String>,
    result: Option<Results>,
    selected_role: OutputRole,
    actual_pixels: bool,
    pan_enabled: bool,
    image_pan: egui::Vec2,
}
impl Default for GenerationDesk {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            character_description: String::new(),
            clothing_description: String::new(),
            references: Vec::new(),
            reference_directory: None,
            pending: None,
            working: false,
            uploading: false,
            cancelling: false,
            status: String::new(),
            attempt_started: None,
            last_elapsed: None,
            sampling: None,
            error: None,
            result: None,
            selected_role: OutputRole::Image,
            actual_pixels: false,
            pan_enabled: true,
            image_pan: egui::Vec2::ZERO,
        }
    }
}
impl GenerationDesk {
    fn selected_result(&self) -> Option<&ResultView> {
        self.result
            .as_ref()?
            .assets
            .iter()
            .find(|asset| asset.role == self.selected_role)
    }
    fn select_result(&mut self, role: OutputRole) {
        if self.selected_role != role {
            self.selected_role = role;
            self.actual_pixels = false;
            self.pan_enabled = true;
            self.image_pan = egui::Vec2::ZERO;
        }
    }
}
pub struct ImagineApp {
    clipboard: Option<crate::clipboard::Clipboard>,
    project: PathBuf,
    connection: Connection,
    prefs: Preferences,
    client: Option<Client>,
    metadata: Option<Value>,
    models: models::ModelUi,
    enhancement: enhancement::EnhancementUi,
    image: GenerationDesk,
    edit: editing::EditSession,
    sheet: GenerationDesk,
    connecting: bool,
    picking_folder: bool,
    connection_error: Option<String>,
    workspace: Workspace,
    settings_open: bool,
    settings_error: Option<String>,
    queue: queue::Queue,
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
        let tx = app.tx.clone();
        app.clipboard = crate::clipboard::Clipboard::install(ctx, move || {
            let _ = tx.send(Message::ClipboardFailed);
        });
        app.connect(ctx);
        app
    }
    fn empty(project: PathBuf, connection: Connection, prefs: Preferences) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            clipboard: None,
            project,
            connection,
            prefs,
            client: None,
            metadata: None,
            models: Default::default(),
            enhancement: Default::default(),
            image: GenerationDesk::default(),
            edit: editing::EditSession::default(),
            sheet: GenerationDesk {
                selected_role: OutputRole::Sheet,
                ..Default::default()
            },
            connecting: false,
            picking_folder: false,
            connection_error: None,
            workspace: Workspace::Image,
            settings_open: false,
            settings_error: None,
            queue: queue::Queue::default(),
            tx,
            rx,
        }
    }
    fn desk_for(&self, workspace: Workspace) -> &GenerationDesk {
        match workspace {
            Workspace::ImageEdit => &self.edit.desk,
            Workspace::CharacterSheet => &self.sheet,
            _ => &self.image,
        }
    }
    fn desk_for_mut(&mut self, workspace: Workspace) -> &mut GenerationDesk {
        match workspace {
            Workspace::ImageEdit => &mut self.edit.desk,
            Workspace::CharacterSheet => &mut self.sheet,
            _ => &mut self.image,
        }
    }
    fn desk(&self) -> &GenerationDesk {
        self.desk_for(self.workspace)
    }
    fn desk_mut(&mut self) -> &mut GenerationDesk {
        self.desk_for_mut(self.workspace)
    }
    fn busy(&self) -> bool {
        self.image.working
            || self.sheet.working
            || self.image.uploading
            || self.sheet.uploading
            || self.edit.desk.working
            || self.edit.desk.uploading
    }
    fn reference_limit(&self) -> usize {
        if self.workspace == Workspace::ImageEdit {
            9
        } else if self.workspace == Workspace::CharacterSheet {
            8
        } else {
            10
        }
    }
    fn can_generate(&self) -> bool {
        if self.models.catalog.is_some() && self.model_selection().is_err() {
            return false;
        }
        match self.workspace {
            Workspace::Image => !self.image.prompt.trim().is_empty(),
            Workspace::ImageEdit => !self.edit.desk.prompt.trim().is_empty(),
            Workspace::CharacterSheet => {
                self.metadata
                    .as_ref()
                    .is_some_and(client::supports_character_sheets)
                    && (!self.sheet.character_description.trim().is_empty()
                        || !self.sheet.references.is_empty())
            }
            Workspace::Video => false,
        }
    }
    fn connect(&mut self, ctx: &egui::Context) {
        self.connecting = true;
        self.connection_error = None;
        let connection = self.connection.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                let client = Client::connect(&connection).map_err(|e| e.to_string())?;
                let data = client.library(None, false).await.map_err(|e| e.message)?;
                Ok((client, data))
            });
            send(&tx, &ctx, Message::Connected(result));
        });
    }
    fn receive(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.rx.try_recv() {
            match message {
                Message::Queue(event) => self.queue_message(ctx, event),
                Message::Enhancement(event) => self.enhancement_message(ctx, event),
                Message::Library(result) => {
                    self.models.loading = false;
                    match result {
                        Ok(data) => {
                            self.models.error = None;
                            self.accept_library(data);
                        }
                        Err(error) => self.models.error = Some(error),
                    }
                }
                Message::ModelFolder(path) => {
                    self.models.loading = false;
                    if let Some(path) = path
                        && let Some(library) = &self.models.catalog
                    {
                        let mut settings = library.settings.clone();
                        settings.root = path;
                        self.load_library(ctx, Some(settings));
                    }
                }
                Message::ClipboardFailed => {
                    self.desk_mut().error = Some("Could not preserve the clipboard after closing Imagine. Keep the app open and copy again.".into());
                }
                Message::Connected(value) => {
                    self.connecting = false;
                    match value {
                        Ok((client, data)) => {
                            self.client = Some(client);
                            self.accept_library(data);
                            self.load_enhancement_profiles(ctx);
                        }
                        Err(e) => {
                            self.client = None;
                            self.metadata = None;
                            self.connection_error = Some(e);
                        }
                    }
                }
                Message::Picked(workspace, paths) => {
                    let state = self.desk_for_mut(workspace);
                    if paths.is_empty() {
                        state.uploading = false;
                        continue;
                    }
                    state.reference_directory = paths
                        .last()
                        .and_then(|path| path.parent())
                        .map(|directory| directory.to_path_buf());
                    let limit = if workspace == Workspace::ImageEdit {
                        9
                    } else if workspace == Workspace::CharacterSheet {
                        8
                    } else {
                        10
                    };
                    if paths.len() + state.references.len() > limit {
                        state.uploading = false;
                        state.error = Some(format!(
                            "Up to {limit} reference images. Select fewer files."
                        ));
                        continue;
                    }
                    let Some(client) = self.client.clone() else {
                        self.desk_for_mut(workspace).uploading = false;
                        continue;
                    };
                    let tx = self.tx.clone();
                    let ctx = ctx.clone();
                    std::thread::spawn(move || {
                        runtime().block_on(async {
                            for path in paths {
                                send(
                                    &tx,
                                    &ctx,
                                    Message::Uploaded(workspace, client.upload(&path).await),
                                );
                            }
                        });
                        send(&tx, &ctx, Message::UploadsFinished(workspace));
                    });
                }
                Message::Uploaded(workspace, result) => match result {
                    Ok(reference) => {
                        let texture = texture(ctx, &reference.preview, &reference.id);
                        let number = if workspace == Workspace::ImageEdit {
                            let number = self.edit.next_reference_number;
                            self.edit.next_reference_number += 1;
                            number
                        } else {
                            0
                        };
                        self.desk_for_mut(workspace).references.push(ReferenceView {
                            number,
                            id: reference.id,
                            name: reference.name,
                            texture,
                        });
                    }
                    Err(e) => self.desk_for_mut(workspace).error = Some(e.message),
                },
                Message::UploadsFinished(workspace) => {
                    self.desk_for_mut(workspace).uploading = false
                }
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
                Message::Progress(workspace, Progress::Submitted(job)) => {
                    if let Some(p) = &mut self.desk_for_mut(workspace).pending {
                        p.job = Some(job);
                    }
                }
                Message::Progress(workspace, Progress::Status { label, sampling }) => {
                    self.desk_for_mut(workspace).status = label;
                    self.desk_for_mut(workspace).sampling = sampling;
                }
                Message::Finished(workspace, result) => {
                    if matches!(workspace, Workspace::Image | Workspace::ImageEdit)
                        && let Ok(done) = &result
                    {
                        self.archive_completed(ctx, done);
                    }
                    let state = self.desk_for_mut(workspace);
                    let elapsed = state.attempt_started.take().map(|start| start.elapsed());
                    state.working = false;
                    state.sampling = None;
                    state.cancelling = false;
                    match result {
                        Ok(done) => {
                            state.last_elapsed = elapsed;
                            state.selected_role = if workspace == Workspace::CharacterSheet {
                                OutputRole::Sheet
                            } else {
                                OutputRole::Image
                            };
                            let upscale = state
                                .pending
                                .as_ref()
                                .is_some_and(|p| p.request["profile"] == "image_upscale");
                            let assets: Vec<_> = done
                                .assets
                                .into_iter()
                                .map(|asset| ResultView {
                                    asset_id: asset.asset_id,
                                    role: if upscale {
                                        state.selected_role
                                    } else {
                                        asset.role
                                    },
                                    texture: texture(ctx, &asset.preview, asset.role.name()),
                                    file: asset.file,
                                })
                                .collect();
                            if upscale && let Some(previous) = &mut state.result {
                                for asset in assets {
                                    if let Some(slot) =
                                        previous.assets.iter_mut().find(|a| a.role == asset.role)
                                    {
                                        *slot = asset;
                                    }
                                }
                                previous.seed = done.seed;
                            } else {
                                state.result = Some(Results {
                                    assets,
                                    seed: done.seed,
                                });
                            }
                            state.actual_pixels = false;
                            state.image_pan = egui::Vec2::ZERO;
                            state.pending = None;
                            state.status = "Saved".into();
                            state.error = None;
                            if workspace == Workspace::ImageEdit {
                                self.edit.versions.push(editing::EditVersion {
                                    result: self.edit.desk.result.clone().unwrap(),
                                    source: Some(self.edit.selected),
                                });
                                self.edit.selected = self.edit.versions.len() - 1;
                            } else if workspace == Workspace::Image {
                                self.edit = editing::EditSession::default();
                            }
                        }
                        Err(error) => {
                            state.error = Some(error.message);
                            state.status.clear();
                            if !error.retryable {
                                state.pending = None;
                            }
                        }
                    }
                }
                Message::Cancelled(workspace, job, result) => {
                    let state = self.desk_for_mut(workspace);
                    if state.pending.as_ref().and_then(|p| p.job.as_ref()) != Some(&job) {
                        continue;
                    }
                    if let Err(e) = result {
                        state.cancelling = false;
                        state.error = Some(e.message);
                    }
                }
            }
        }
    }
    fn generate(&mut self, ctx: &egui::Context) {
        if self.busy() || self.connecting || self.workspace == Workspace::Video {
            return;
        }
        if self.workspace == Workspace::Image && self.image.pending.is_none() {
            self.enqueue_prompts(ctx, vec![self.image.prompt.clone()]);
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        let workspace = self.workspace;
        if self.desk().pending.is_none() {
            let state = self.desk();
            let refs: Vec<_> = state.references.iter().map(|r| r.id.clone()).collect();
            let Some(metadata) = &self.metadata else {
                return;
            };
            let request = if workspace == Workspace::ImageEdit {
                self.edit_request()
            } else if workspace == Workspace::CharacterSheet {
                client::make_sheet_request(
                    &state.character_description,
                    &state.clothing_description,
                    self.prefs.sheet_width,
                    &self.prefs.negative_prompt,
                    &refs,
                    metadata,
                    &self.prefs.generation,
                )
            } else {
                client::make_request(
                    &state.prompt,
                    &self.prefs.negative_prompt,
                    &refs,
                    metadata,
                    &self.prefs.generation,
                )
            };
            let request = match request.and_then(|r| self.apply_model_selection(r)) {
                Ok(v) => v,
                Err(e) => {
                    self.desk_mut().error = Some(e.to_string());
                    return;
                }
            };
            self.desk_mut().pending = Some(Submission {
                upscale_source: None,
                request,
                key: uuid::Uuid::new_v4().to_string(),
                folder: self.prefs.output_folder.clone(),
                job: None,
            });
        }
        let folder = self.prefs.output_folder.clone();
        let state = self.desk_mut();
        let pending = state.pending.as_mut().unwrap();
        pending.folder = folder;
        let submission = pending.clone();
        state.attempt_started = Some(std::time::Instant::now());
        state.working = true;
        state.status = "Submitting".into();
        state.sampling = None;
        state.error = None;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.generate(submission, |p| {
                send(&tx, &ctx, Message::Progress(workspace, p))
            }));
            send(&tx, &ctx, Message::Finished(workspace, result));
        });
    }
    fn upscale_ratio(&self) -> &str {
        if self.workspace == Workspace::CharacterSheet {
            "16:9"
        } else {
            &self.prefs.generation.aspect_ratio
        }
    }
    fn upscale_source(&self) -> Option<&ResultView> {
        if self.queue.loading.is_some() {
            return None;
        }
        let role = match self.workspace {
            Workspace::Image => OutputRole::Image,
            Workspace::CharacterSheet => OutputRole::Sheet,
            Workspace::Video | Workspace::ImageEdit => return None,
        };
        self.desk()
            .result
            .as_ref()?
            .assets
            .iter()
            .find(|a| a.role == role && !a.asset_id.is_empty())
    }
    fn upscale_submission(&self) -> anyhow::Result<Submission> {
        if self.busy() || self.desk().pending.is_some() {
            anyhow::bail!("Wait for the current job before upscaling.");
        }
        let source = self
            .upscale_source()
            .ok_or_else(|| anyhow::anyhow!("Generate an image before upscaling."))?;
        Ok(Submission {
            upscale_source: Some(source.file.clone()),
            request: client::make_upscale_request(
                &source.asset_id,
                self.prefs.upscale_target,
                self.upscale_ratio(),
                &self.prefs.generation,
            )?,
            key: uuid::Uuid::new_v4().to_string(),
            folder: self.prefs.output_folder.clone(),
            job: None,
        })
    }
    fn upscale(&mut self, ctx: &egui::Context) {
        match self.upscale_submission() {
            Ok(submission) => {
                self.desk_mut().pending = Some(submission);
                self.generate(ctx);
            }
            Err(error) => self.desk_mut().error = Some(error.to_string()),
        }
    }
    fn choose_references(&mut self, ctx: &egui::Context) {
        self.desk_mut().uploading = true;
        let workspace = self.workspace;
        self.desk_mut().error = None;
        let directory = self.desk().reference_directory.clone();
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
            send(&tx, &ctx, Message::Picked(workspace, files));
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
        let Some(job) = self.desk().pending.as_ref().and_then(|p| p.job.clone()) else {
            return;
        };
        let Some(client) = self.client.clone() else {
            return;
        };
        self.desk_mut().cancelling = true;
        let workspace = self.workspace;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.cancel(&job));
            send(&tx, &ctx, Message::Cancelled(workspace, job, result));
        });
    }
    fn save_preferences(&mut self) {
        if self
            .prefs
            .model_groups
            .values()
            .flat_map(|g| &g.loras)
            .any(|l| !l.strength.is_finite())
        {
            self.settings_error = Some("Enter a finite LoRA strength before saving.".into());
            return;
        }
        self.settings_error = self
            .prefs
            .save(&self.project)
            .err()
            .map(|_| "Could not save settings. Check write permissions for .imagine.".into());
    }
}

#[path = "workspace.rs"]
mod workspace;

#[path = "editing.rs"]
mod editing;

#[path = "queue.rs"]
mod queue;

#[path = "queue_ui.rs"]
mod queue_ui;

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
    image_view_layers(
        ui,
        (texture, None),
        size,
        privacy_mode,
        actual_pixels,
        pan_enabled,
        pan,
    )
}

fn image_view_layers(
    ui: &mut egui::Ui,
    (texture, before): (&egui::TextureHandle, Option<&egui::TextureHandle>),
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
        let visible_image = image_rect.intersect(visible_rect);
        let split = before.filter(|_| response.hovered()).and_then(|before| {
            ui.input(|input| {
                input
                    .pointer
                    .hover_pos()
                    .filter(|pos| input.focused && visible_image.contains(*pos))
                    .map(|pos| (before, pos.x))
            })
        });
        let uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1., 1.));
        if let Some((before, x)) = split {
            let left =
                egui::Rect::from_min_max(visible_image.min, egui::pos2(x, visible_image.bottom()));
            let right =
                egui::Rect::from_min_max(egui::pos2(x, visible_image.top()), visible_image.max);
            ui.painter().with_clip_rect(left).image(
                before.id(),
                image_rect,
                uv,
                egui::Color32::WHITE,
            );
            ui.painter().with_clip_rect(right).image(
                texture.id(),
                image_rect,
                uv,
                egui::Color32::WHITE,
            );
            ui.painter().with_clip_rect(visible_image).vline(
                x,
                visible_image.y_range(),
                egui::Stroke::new(1., helto::ACCENT),
            );
        } else {
            ui.painter().with_clip_rect(visible_rect).image(
                texture.id(),
                image_rect,
                uv,
                egui::Color32::WHITE,
            );
        }
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
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.models.dirty {
            self.save_preferences();
        }
        if let Some(clipboard) = &self.clipboard {
            clipboard.finish();
        }
    }
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

#[path = "models.rs"]
mod models;

#[path = "enhancement.rs"]
mod enhancement;
