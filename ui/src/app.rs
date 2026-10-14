use crate::{
    client::{self, Client, Completed, Failure, Preview, Progress, Reference, Submission},
    helto,
    settings::{Connection, Preferences},
};
use eframe::egui::{self, RichText};
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

pub struct ImagineApp {
    project: PathBuf,
    connection: Connection,
    prefs: Preferences,
    client: Option<Client>,
    metadata: Option<Value>,
    prompt: String,
    references: Vec<ReferenceView>,
    pending: Option<Submission>,
    working: bool,
    uploading: bool,
    connecting: bool,
    picking_folder: bool,
    cancelling: bool,
    status: String,
    sampling: Option<client::SamplingProgress>,
    error: Option<String>,
    result: Option<(egui::TextureHandle, PathBuf, Option<u64>)>,
    actual_pixels: bool,
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
            pending: None,
            working: false,
            uploading: false,
            connecting: false,
            picking_folder: false,
            cancelling: false,
            status: String::new(),
            sampling: None,
            error: None,
            result: None,
            actual_pixels: false,
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
                    match prefs.save(&self.project){Ok(())=>self.prefs=prefs,Err(_)=>self.error=Some("Could not save the folder choice. Check write permissions for .imagine.".into())}
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
                    self.working = false;
                    self.sampling = None;
                    self.cancelling = false;
                    match result {
                        Ok(done) => {
                            self.result =
                                Some((texture(ctx, &done.preview, "result"), done.file, done.seed));
                            self.image_pan = egui::Vec2::ZERO;
                            self.pending = None;
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
    fn choose_references(&mut self, ctx: &egui::Context) {
        self.uploading = true;
        self.error = None;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let files = rfd::FileDialog::new()
                .set_title("Select reference images")
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_files()
                .unwrap_or_default();
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
    fn editor(&mut self, ui: &mut egui::Ui) {
        let editable =
            !self.working && !self.uploading && !self.connecting && self.pending.is_none();
        let prompt_label = ui.label("Prompt");
        ui.add_enabled_ui(editable, |ui| {
            helto::textarea(ui, &mut self.prompt, 8, "prompt").labelled_by(prompt_label.id);
        });
        ui.add_enabled_ui(editable, |ui| {
            egui::CollapsingHeader::new("Negative prompt").show(ui, |ui| {
                if helto::textarea(ui, &mut self.prefs.negative_prompt, 3, "negative-prompt")
                    .changed()
                {
                    self.save_preferences();
                }
            });
        });
        ui.add_space(6.);
        ui.add_enabled_ui(editable, |ui| self.generation_controls(ui));
        ui.add_space(6.);
        ui.horizontal(|ui| {
            ui.label(format!("References · {}/10", self.references.len()));
            let enabled = editable && self.client.is_some() && self.references.len() < 10;
            if ui
                .add_enabled_ui(enabled, |ui| helto::button(ui, "Add", false))
                .inner
                .clicked()
            {
                self.choose_references(ui.ctx());
            }
            if self.uploading {
                ui.spinner();
            }
        });
        let mut remove = None;
        let mut swap = None;
        egui::ScrollArea::vertical()
            .id_salt("references")
            .max_height(245.)
            .show(ui, |ui| {
                for (index, reference) in self.references.iter().enumerate() {
                    ui.push_id(&reference.id, |ui| {
                        egui::Frame::new()
                            .fill(helto::BG)
                            .stroke(egui::Stroke::new(1., helto::FIELD))
                            .corner_radius(6)
                            .inner_margin(6)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    private_image(
                                        ui,
                                        &reference.texture,
                                        egui::vec2(64., 48.),
                                        self.prefs.privacy_mode,
                                    );
                                    ui.vertical(|ui| {
                                        ui.add(
                                            egui::Label::new(format!(
                                                "{}. {}",
                                                index + 1,
                                                reference.name
                                            ))
                                            .truncate(),
                                        )
                                        .on_hover_text(&reference.name);
                                        ui.add_enabled_ui(editable, |ui| {
                                            ui.horizontal(|ui| {
                                                if ui
                                                    .add_enabled_ui(index > 0, |ui| {
                                                        helto::button(ui, "Up", false)
                                                    })
                                                    .inner
                                                    .on_hover_text("Move up")
                                                    .clicked()
                                                {
                                                    swap = Some((index, index - 1));
                                                }
                                                if ui
                                                    .add_enabled_ui(
                                                        index + 1 < self.references.len(),
                                                        |ui| helto::button(ui, "Down", false),
                                                    )
                                                    .inner
                                                    .on_hover_text("Move down")
                                                    .clicked()
                                                {
                                                    swap = Some((index, index + 1));
                                                }
                                                if helto::button(ui, "Remove", false).clicked() {
                                                    remove = Some(index);
                                                }
                                            });
                                        });
                                    });
                                });
                            });
                    });
                }
            });
        if let Some(i) = remove {
            self.references.remove(i);
        } else if let Some((a, b)) = swap {
            self.references.swap(a, b);
        }
        ui.add_space(6.);
        ui.label("Save to");
        ui.horizontal(|ui| {
            ui.add(
                egui::Label::new(
                    RichText::new(self.prefs.output_folder.display().to_string()).color(helto::DIM),
                )
                .truncate(),
            )
            .on_hover_text(self.prefs.output_folder.display().to_string());
        });
        if ui
            .add_enabled_ui(!self.working && !self.picking_folder, |ui| {
                helto::button(ui, "Choose folder", false)
            })
            .inner
            .clicked()
        {
            self.choose_folder(ui.ctx());
        }
        ui.add_space(8.);
        ui.horizontal(|ui| {
            let enabled = self.client.is_some()
                && !self.connecting
                && !self.working
                && !self.uploading
                && !self.picking_folder
                && (!self.prompt.trim().is_empty() || self.pending.is_some());
            let label = if self.pending.is_some() && !self.working {
                "Retry"
            } else {
                "Generate"
            };
            if ui
                .add_enabled_ui(enabled, |ui| helto::button(ui, label, true))
                .inner
                .clicked()
            {
                self.generate(ui.ctx());
            }
            if self.working {
                ui.spinner();
                if ui
                    .add_enabled_ui(
                        !self.cancelling && self.pending.as_ref().is_some_and(|p| p.job.is_some()),
                        |ui| helto::button(ui, "Cancel", false),
                    )
                    .inner
                    .clicked()
                {
                    self.cancel(ui.ctx());
                }
            }
        });
        if self.working
            && let Some(progress) = &self.sampling
        {
            ui.add(
                egui::ProgressBar::new(progress.completed as f32 / progress.total as f32)
                    .fill(helto::ACCENT_TOP)
                    .text(format!(
                        "Sampling {} / {} · {:.0}%",
                        progress.completed,
                        progress.total,
                        100. * progress.completed as f32 / progress.total as f32
                    )),
            );
            if !self.cancelling
                && self.status != "Cancelling"
                && let Some(finish) = progress.finish_time()
            {
                let remaining = finish
                    .duration_since(std::time::SystemTime::now())
                    .unwrap_or_default()
                    .as_secs();
                ui.label(
                    RichText::new(format!(
                        "Sampling ETA ~{}:{:02} · ends ~{}",
                        remaining / 60,
                        remaining % 60,
                        local_time(finish)
                    ))
                    .color(helto::DIM),
                );
            }
        }
        if !self.status.is_empty() && (self.status != "Sampling" || self.sampling.is_none()) {
            ui.label(RichText::new(&self.status).color(helto::DIM));
        }
        if let Some(error) = &self.error {
            ui.add_space(4.);
            ui.colored_label(helto::ERROR, error);
        }
    }
    fn generation_controls(&mut self, ui: &mut egui::Ui) {
        let settings = &mut self.prefs.generation;
        let before = settings.clone();
        egui::Grid::new("generation-settings")
            .spacing([10., 6.])
            .show(ui, |ui| {
                let label = ui.label("Steps");
                ui.add(egui::DragValue::new(&mut settings.steps).range(1..=100))
                    .labelled_by(label.id);
                ui.end_row();
                let label = ui.label("CFG");
                ui.add(
                    egui::DragValue::new(&mut settings.cfg_scale)
                        .range(0.0..=20.0)
                        .speed(0.1),
                )
                .labelled_by(label.id);
                ui.end_row();
                let label = ui.label("Seed");
                egui::ComboBox::from_id_salt("seed-mode")
                    .selected_text(if settings.random_seed {
                        "Random each time"
                    } else {
                        "Fixed seed"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut settings.random_seed, true, "Random each time");
                        ui.selectable_value(&mut settings.random_seed, false, "Fixed seed");
                    })
                    .response
                    .labelled_by(label.id);
                ui.end_row();
                if !settings.random_seed {
                    let label = ui.label("Seed value");
                    ui.add(egui::DragValue::new(&mut settings.seed).range(0..=u32::MAX))
                        .labelled_by(label.id);
                    ui.end_row();
                }
                let label = ui.label("Ratio");
                egui::ComboBox::from_id_salt("aspect-ratio")
                    .selected_text(&settings.aspect_ratio)
                    .show_ui(ui, |ui| {
                        for ratio in crate::settings::ASPECT_RATIOS {
                            ui.selectable_value(
                                &mut settings.aspect_ratio,
                                ratio.to_owned(),
                                ratio,
                            );
                        }
                    })
                    .response
                    .labelled_by(label.id);
                ui.end_row();
                let label = ui.label("Max side (px)");
                ui.add(
                    egui::DragValue::new(&mut settings.max_side)
                        .range(256..=4096)
                        .speed(32.),
                )
                .labelled_by(label.id)
                .on_hover_text("Multiples of 32");
                ui.end_row();
            });
        match settings.dimensions() {
            Ok((w, h)) => {
                ui.label(RichText::new(format!("{w} × {h} px")).color(helto::DIM));
            }
            Err(e) => {
                ui.colored_label(helto::ERROR, e.to_string());
            }
        }
        if *settings != before {
            self.save_preferences();
        }
    }
    fn viewer(&mut self, ui: &mut egui::Ui) {
        let height = (ui.available_height() - 42.).max(230.);
        egui::Frame::new()
            .fill(helto::MEDIA)
            .stroke(egui::Stroke::new(1., helto::FIELD))
            .corner_radius(6)
            .inner_margin(9)
            .show(ui, |ui| {
                let size = egui::vec2(ui.available_width().max(1.), height);
                if let Some((texture, _, _)) = &self.result {
                    image_view(
                        ui,
                        texture,
                        size,
                        self.prefs.privacy_mode,
                        self.actual_pixels,
                        &mut self.image_pan,
                    );
                } else {
                    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "No image yet",
                        egui::FontId::proportional(12.),
                        helto::DIM,
                    );
                }
            });
        if let Some((_, path, seed)) = &self.result {
            ui.horizontal_wrapped(|ui| {
                if helto::button(
                    ui,
                    if self.actual_pixels {
                        "Zoom: Actual pixels"
                    } else {
                        "Zoom: Fit"
                    },
                    self.actual_pixels,
                )
                .clicked()
                {
                    self.actual_pixels = !self.actual_pixels;
                    self.image_pan = egui::Vec2::ZERO;
                }
                ui.label(
                    RichText::new(path.file_name().unwrap_or_default().to_string_lossy())
                        .color(helto::DIM),
                )
                .on_hover_text(path.display().to_string());
                if let Some(seed) = seed {
                    ui.label(RichText::new(format!("Seed {seed}")).color(helto::DIM));
                }
            });
        }
    }
    fn save_preferences(&mut self) {
        if self.prefs.save(&self.project).is_err() {
            self.error =
                Some("Could not save settings. Check write permissions for .imagine.".into());
        }
    }
    pub fn content(&mut self, ui: &mut egui::Ui) {
        self.receive(ui.ctx());
        if self.working || self.uploading || self.connecting {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        egui::Frame::new()
            .fill(helto::SURFACE)
            .inner_margin(12)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Imagine");
                    ui.label(RichText::new("Qwen Image 2.1").color(helto::DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if helto::button(
                            ui,
                            if self.prefs.privacy_mode {
                                "Privacy: on"
                            } else {
                                "Privacy: off"
                            },
                            self.prefs.privacy_mode,
                        )
                        .clicked()
                        {
                            self.prefs.privacy_mode = !self.prefs.privacy_mode;
                            self.save_preferences();
                        }
                        if (self.client.is_none() || self.error.is_some())
                            && !self.working
                            && !self.connecting
                            && !self.uploading
                            && self.pending.is_none()
                            && helto::button(ui, "Reconnect", false).clicked()
                        {
                            self.connect(ui.ctx());
                        }
                        ui.colored_label(
                            if self.client.is_some() {
                                helto::OK
                            } else {
                                helto::DIM
                            },
                            if self.connecting {
                                "Connecting"
                            } else if self.client.is_some() {
                                "Connected"
                            } else {
                                "Disconnected"
                            },
                        );
                    });
                });
                ui.add_space(8.);
                if ui.available_width() < 800. {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.editor(ui);
                        ui.add_space(12.);
                        self.viewer(ui);
                    });
                } else {
                    egui::Panel::left("compose")
                        .resizable(false)
                        .default_size(330.)
                        .frame(
                            egui::Frame::new()
                                .fill(helto::SURFACE)
                                .inner_margin(egui::Margin {
                                    left: 0,
                                    right: 12,
                                    top: 0,
                                    bottom: 0,
                                }),
                        )
                        .show(ui, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| self.editor(ui));
                        });
                    egui::CentralPanel::default()
                        .frame(egui::Frame::new().fill(helto::SURFACE))
                        .show(ui, |ui| self.viewer(ui));
                }
            });
    }
}
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
    image_view(ui, texture, size, privacy_mode, false, &mut pan)
}
fn image_view(
    ui: &mut egui::Ui,
    texture: &egui::TextureHandle,
    size: egui::Vec2,
    privacy_mode: bool,
    actual_pixels: bool,
    pan: &mut egui::Vec2,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(
        size,
        if actual_pixels {
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
            *pan += response.drag_delta();
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
    if actual_pixels {
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
