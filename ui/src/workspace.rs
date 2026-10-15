//! Reference desk: ordered inputs, an image viewport, and the prompt inspector.
use super::*;
use crate::icons::{self, Icon};
use egui::{Align, Layout, Rect, RichText, Stroke, Ui, UiBuilder, Vec2};

fn region<R>(ui: &mut Ui, id: &str, rect: Rect, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.scope_builder(
        UiBuilder::new()
            .id_salt(id)
            .max_rect(rect)
            .layout(Layout::top_down(Align::Min)),
        |ui| {
            ui.set_clip_rect(ui.clip_rect().intersect(rect));
            add(ui)
        },
    )
    .inner
}

fn panel(ui: &Ui, rect: Rect, media: bool) {
    helto::gradient(
        ui.painter(),
        rect,
        6.,
        if media { helto::MEDIA } else { helto::SURFACE },
        if media { helto::MEDIA } else { helto::BG },
    );
    ui.painter().rect_stroke(
        rect,
        6,
        Stroke::new(1., helto::FIELD),
        egui::StrokeKind::Inside,
    );
}

impl ImagineApp {
    fn editable(&self) -> bool {
        !self.working && !self.uploading && !self.connecting && self.pending.is_none()
    }

    pub fn content(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
        if self.working || self.uploading || self.connecting {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let bounds = ui.max_rect();
        ui.painter().rect_filled(bounds, 0, helto::BG);
        let header = Rect::from_min_size(bounds.min, egui::vec2(bounds.width(), 64.));
        self.header(ui, header);
        let body = Rect::from_min_max(
            egui::pos2(bounds.left() + 8., header.bottom() + 8.),
            bounds.max - egui::vec2(8., 8.),
        );
        region(ui, "workspace", body, |ui| match self.workspace {
            Workspace::Image if body.width() >= 900. => self.image_workspace(ui, body),
            Workspace::Image => {
                egui::ScrollArea::vertical()
                    .id_salt("compact-workspace")
                    .show(ui, |ui| {
                        let size = egui::vec2(ui.available_width(), body.height().max(920.));
                        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                        self.image_workspace(ui, rect);
                    });
            }
            Workspace::Video => {
                panel(ui, body, true);
                ui.painter().text(
                    body.center(),
                    egui::Align2::CENTER_CENTER,
                    "Video editing is not available yet",
                    egui::FontId::proportional(14.),
                    helto::DIM,
                );
            }
        });
        if self.settings_open {
            self.settings_window(ui.ctx());
        }
    }

    fn header(&mut self, ui: &mut Ui, rect: Rect) {
        helto::gradient(ui.painter(), rect, 0., helto::FIELD, helto::SURFACE);
        ui.painter()
            .hline(rect.x_range(), rect.bottom(), Stroke::new(1., helto::FIELD));
        region(
            ui,
            "brand",
            Rect::from_min_size(
                rect.min + egui::vec2(18., 0.),
                egui::vec2(280., rect.height()),
            ),
            |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.label(
                        RichText::new("Imagine")
                            .size(20.)
                            .strong()
                            .color(helto::TEXT),
                    );
                    if rect.width() >= 760. {
                        ui.add_space(8.);
                        ui.separator();
                        ui.add_space(8.);
                        ui.label(RichText::new("Qwen Image 2.1").color(helto::DIM));
                    }
                });
            },
        );
        let tabs = Rect::from_center_size(rect.center(), egui::vec2(72., 32.));
        region(ui, "tabs", tabs, |ui| {
            ui.horizontal(|ui| {
                if helto::icon_button(ui, Icon::Image, "Image", self.workspace == Workspace::Image)
                    .clicked()
                {
                    self.workspace = Workspace::Image;
                }
                if helto::icon_button(ui, Icon::Video, "Video", self.workspace == Workspace::Video)
                    .clicked()
                {
                    self.workspace = Workspace::Video;
                }
            });
        });
        region(
            ui,
            "header-actions",
            Rect::from_min_max(
                egui::pos2(rect.right() - 240., rect.top()),
                rect.max - egui::vec2(16., 0.),
            ),
            |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let privacy = self.prefs.privacy_mode;
                    if helto::icon_button(
                        ui,
                        if privacy { Icon::EyeOff } else { Icon::Eye },
                        "Hide previews",
                        privacy,
                    )
                    .clicked()
                    {
                        self.prefs.privacy_mode = !privacy;
                        self.save_preferences();
                    }
                    if helto::icon_button(ui, Icon::Settings, "Settings", self.settings_open)
                        .clicked()
                    {
                        self.settings_open = !self.settings_open;
                    }
                    let reconnect = (self.client.is_none() || self.error.is_some())
                        && !self.working
                        && !self.connecting
                        && !self.uploading
                        && self.pending.is_none();
                    if reconnect
                        && helto::icon_button(ui, Icon::Retry, "Reconnect", false).clicked()
                    {
                        self.connect(ui.ctx());
                    }
                    let text = if self.connecting {
                        "Connecting"
                    } else if self.client.is_some() {
                        "Connected"
                    } else {
                        "Disconnected"
                    };
                    if rect.width() >= 760. {
                        ui.label(RichText::new(text).color(helto::DIM));
                    }
                    let (dot, response) =
                        ui.allocate_exact_size(egui::vec2(10., 10.), egui::Sense::hover());
                    ui.painter().circle_filled(
                        dot.center(),
                        3.5,
                        if self.client.is_some() {
                            helto::OK
                        } else {
                            helto::DIM
                        },
                    );
                    response.on_hover_text(text);
                });
            },
        );
    }

    fn image_workspace(&mut self, ui: &mut Ui, rect: Rect) {
        let (references, viewer, prompt) = if rect.width() >= 900. {
            let refs_width = (rect.width() * 0.195).clamp(180., 320.);
            let prompt_width = (rect.width() * 0.235).clamp(240., 360.);
            let references = Rect::from_min_size(rect.min, egui::vec2(refs_width, rect.height()));
            let prompt = Rect::from_min_max(
                egui::pos2(rect.right() - prompt_width, rect.top()),
                rect.max,
            );
            let viewer = Rect::from_min_max(
                egui::pos2(references.right() + 8., rect.top()),
                egui::pos2(prompt.left() - 8., rect.bottom()),
            );
            (references, viewer, prompt)
        } else {
            let viewer = Rect::from_min_size(rect.min, egui::vec2(rect.width(), 360.));
            let top = viewer.bottom() + 8.;
            let refs_width = (rect.width() * 0.38).max(175.);
            let references = Rect::from_min_max(
                egui::pos2(rect.left(), top),
                egui::pos2(rect.left() + refs_width, rect.bottom()),
            );
            let prompt = Rect::from_min_max(egui::pos2(references.right() + 8., top), rect.max);
            (references, viewer, prompt)
        };
        panel(ui, references, false);
        panel(ui, viewer, true);
        panel(ui, prompt, false);
        region(ui, "references", references.shrink(10.), |ui| {
            self.reference_panel(ui)
        });
        region(ui, "viewer", viewer.shrink(10.), |ui| self.viewer(ui));
        region(ui, "prompt", prompt.shrink(12.), |ui| self.prompt_panel(ui));
    }

    fn reference_panel(&mut self, ui: &mut Ui) {
        let editable = self.editable();
        ui.horizontal(|ui| {
            ui.label(format!("References · {}/10", self.references.len()));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled_ui(
                        editable && self.client.is_some() && self.references.len() < 10,
                        |ui| helto::icon_button(ui, Icon::Plus, "Add references", false),
                    )
                    .inner
                    .clicked()
                {
                    self.choose_references(ui.ctx());
                }
            });
        });
        ui.separator();
        if self.uploading {
            ui.spinner();
        }
        let mut remove = None;
        let mut swap = None;
        egui::ScrollArea::vertical()
            .id_salt("reference-list")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (index, reference) in self.references.iter().enumerate() {
                    ui.push_id(&reference.id, |ui| {
                        ui.horizontal(|ui| {
                            let width = (ui.available_width() - 40.).max(64.);
                            let image = private_image(
                                ui,
                                &reference.texture,
                                egui::vec2(width, width * 1.05),
                                self.prefs.privacy_mode,
                            );
                            let badge = Rect::from_min_size(image.rect.min, egui::vec2(24., 24.));
                            ui.painter().rect_filled(badge, 5, helto::SURFACE);
                            ui.painter().text(
                                badge.center(),
                                egui::Align2::CENTER_CENTER,
                                (index + 1).to_string(),
                                egui::FontId::proportional(12.),
                                helto::TEXT,
                            );
                            image.on_hover_text(&reference.name);
                            ui.vertical(|ui| {
                                ui.add_space((width * 1.05 - 100.).max(0.) * 0.5);
                                if ui
                                    .add_enabled_ui(editable && index > 0, |ui| {
                                        helto::icon_button(ui, Icon::Up, "Move reference up", false)
                                    })
                                    .inner
                                    .clicked()
                                {
                                    swap = Some((index, index - 1));
                                }
                                if ui
                                    .add_enabled_ui(
                                        editable && index + 1 < self.references.len(),
                                        |ui| {
                                            helto::icon_button(
                                                ui,
                                                Icon::Down,
                                                "Move reference down",
                                                false,
                                            )
                                        },
                                    )
                                    .inner
                                    .clicked()
                                {
                                    swap = Some((index, index + 1));
                                }
                                if ui
                                    .add_enabled_ui(editable, |ui| {
                                        helto::icon_button(
                                            ui,
                                            Icon::Trash,
                                            "Remove reference",
                                            false,
                                        )
                                    })
                                    .inner
                                    .clicked()
                                {
                                    remove = Some(index);
                                }
                            });
                        });
                        ui.add_space(6.);
                    });
                }
            });
        if let Some(index) = remove {
            self.references.remove(index);
        } else if let Some((a, b)) = swap {
            self.references.swap(a, b);
        }
    }

    fn prompt_panel(&mut self, ui: &mut Ui) {
        let bottom = ui.max_rect().bottom();
        let content = Rect::from_min_max(
            ui.max_rect().min,
            egui::pos2(ui.max_rect().right(), bottom - 30.),
        );
        region(ui, "prompt-fields", content, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("prompt-scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let label = ui.label("Prompt");
                    ui.scope(|ui| {
                        helto::textarea(
                            ui,
                            &mut self.prompt,
                            8,
                            "prompt",
                            self.prefs.privacy_mode,
                            true,
                        )
                        .labelled_by(label.id);
                        ui.add_space(8.);
                        egui::Frame::new()
                            .fill(helto::FIELD)
                            .stroke(Stroke::new(1., helto::RAISED))
                            .corner_radius(5)
                            .inner_margin(8)
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                egui::CollapsingHeader::new("Negative prompt").show(ui, |ui| {
                                    if helto::textarea(
                                        ui,
                                        &mut self.prefs.negative_prompt,
                                        3,
                                        "negative-prompt",
                                        self.prefs.privacy_mode,
                                        false,
                                    )
                                    .changed()
                                    {
                                        self.save_preferences();
                                    }
                                });
                            });
                        ui.add_space(18.);
                        self.image_controls(ui);
                    });
                });
        });
        if let Some((_, _, Some(seed))) = &self.result {
            let footer = Rect::from_min_max(
                egui::pos2(ui.max_rect().left(), bottom - 26.),
                ui.max_rect().max,
            );
            region(ui, "resolved-seed", footer, |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(
                        RichText::new(format!("Seed {seed}"))
                            .small()
                            .color(helto::DIM),
                    );
                });
            });
        }
    }

    fn image_controls(&mut self, ui: &mut Ui) {
        let before = self.prefs.generation.clone();
        let settings = &mut self.prefs.generation;
        let field_width = (ui.available_width() - 80.).max(90.);
        egui::Grid::new("image-settings")
            .num_columns(2)
            .spacing([10., 10.])
            .show(ui, |ui| {
                let label = ui.label("Ratio");
                egui::ComboBox::from_id_salt("aspect-ratio")
                    .width(field_width)
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
                let label = ui.label("Max side");
                ui.add_sized(
                    [field_width, 32.],
                    egui::DragValue::new(&mut settings.max_side)
                        .range(256..=4096)
                        .speed(32.),
                )
                .labelled_by(label.id)
                .on_hover_text("Pixels · multiples of 32");
                ui.end_row();
                ui.label("");
                if let Ok((w, h)) = settings.dimensions() {
                    ui.label(RichText::new(format!("{w} × {h} px")).color(helto::DIM));
                }
                ui.end_row();
                let label = ui.label("Seed");
                egui::ComboBox::from_id_salt("seed-mode")
                    .width(field_width)
                    .selected_text(if settings.random_seed {
                        "Random"
                    } else {
                        "Fixed"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut settings.random_seed, true, "Random");
                        ui.selectable_value(&mut settings.random_seed, false, "Fixed");
                    })
                    .response
                    .labelled_by(label.id);
                ui.end_row();
                if !settings.random_seed {
                    let label = ui.label("Seed value");
                    ui.add_sized(
                        [field_width, 32.],
                        egui::DragValue::new(&mut settings.seed).range(0..=u32::MAX),
                    )
                    .labelled_by(label.id);
                    ui.end_row();
                }
            });
        if let Err(e) = settings.dimensions() {
            ui.colored_label(helto::ERROR, e.to_string());
        }
        if *settings != before {
            self.save_preferences();
        }
    }

    fn viewer(&mut self, ui: &mut Ui) {
        let bounds = ui.max_rect();
        region(
            ui,
            "view-tools",
            Rect::from_min_size(bounds.min, egui::vec2(bounds.width(), 32.)),
            |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.add_enabled_ui(self.result.is_some(), |ui| {
                        if helto::icon_button(
                            ui,
                            Icon::Hand,
                            "Pan image",
                            self.actual_pixels && self.pan_enabled,
                        )
                        .clicked()
                        {
                            if !self.actual_pixels {
                                self.actual_pixels = true;
                                self.pan_enabled = true;
                            } else {
                                self.pan_enabled = !self.pan_enabled;
                            }
                        }
                        if helto::icon_button(ui, Icon::Actual, "Actual pixels", self.actual_pixels)
                            .clicked()
                        {
                            self.actual_pixels = true;
                            self.image_pan = Vec2::ZERO;
                        }
                        if helto::icon_button(ui, Icon::Fit, "Fit image", !self.actual_pixels)
                            .clicked()
                        {
                            self.actual_pixels = false;
                            self.image_pan = Vec2::ZERO;
                        }
                    });
                });
            },
        );
        let error = self.error.clone().or_else(|| self.settings_error.clone());
        let footer_height = if error.is_some() {
            132.
        } else if self.sampling.is_some() {
            92.
        } else {
            66.
        };
        let image_rect = Rect::from_min_max(
            bounds.min + egui::vec2(0., 38.),
            bounds.max - egui::vec2(0., footer_height),
        );
        region(ui, "image", image_rect, |ui| {
            if let Some((texture, _, _)) = &self.result {
                image_view(
                    ui,
                    texture,
                    image_rect.size(),
                    self.prefs.privacy_mode,
                    self.actual_pixels,
                    self.pan_enabled,
                    &mut self.image_pan,
                );
            } else {
                icons::paint(
                    ui.painter(),
                    Rect::from_center_size(
                        image_rect.center() - egui::vec2(0., 12.),
                        egui::vec2(32., 32.),
                    ),
                    Icon::Image,
                    helto::DIM,
                );
                ui.painter().text(
                    image_rect.center() + egui::vec2(0., 24.),
                    egui::Align2::CENTER_CENTER,
                    "No image yet",
                    egui::FontId::proportional(12.),
                    helto::DIM,
                );
            }
        });
        let footer = Rect::from_min_max(
            egui::pos2(bounds.left(), image_rect.bottom() + 10.),
            bounds.max,
        );
        region(ui, "generation-actions", footer, |ui| {
            ui.separator();
            let ready = self.client.is_some()
                && !self.connecting
                && !self.working
                && !self.uploading
                && !self.picking_folder;
            ui.horizontal(|ui| {
                if ui
                    .add_enabled_ui(
                        ready && self.pending.is_none() && !self.prompt.trim().is_empty(),
                        |ui| helto::icon_button(ui, Icon::Generate, "Generate", true),
                    )
                    .inner
                    .clicked()
                {
                    self.generate(ui.ctx());
                }
                if ui
                    .add_enabled_ui(
                        ready && self.pending.is_none() && self.last_generation.is_some(),
                        |ui| helto::icon_button(ui, Icon::Retry, "Generate again", false),
                    )
                    .inner
                    .clicked()
                    && let Some(submission) = self.regeneration()
                {
                    self.pending = Some(submission);
                    self.generate(ui.ctx());
                }
                if ui
                    .add_enabled_ui(
                        self.working
                            && !self.cancelling
                            && self.pending.as_ref().is_some_and(|p| p.job.is_some()),
                        |ui| helto::icon_button(ui, Icon::Stop, "Cancel generation", false),
                    )
                    .inner
                    .clicked()
                {
                    self.cancel(ui.ctx());
                }
                ui.add_space(6.);
                if self.working
                    && let Some(progress) = &self.sampling
                {
                    ui.add(
                        egui::ProgressBar::new(progress.completed as f32 / progress.total as f32)
                            .desired_width((ui.available_width() - 48.).max(40.))
                            .desired_height(5.)
                            .fill(helto::ACCENT),
                    );
                    ui.label(format!(
                        "{:.0}%",
                        100. * progress.completed as f32 / progress.total as f32
                    ));
                } else {
                    if self.working {
                        ui.spinner();
                    }
                    if self.status == "Saved" && self.result.is_some() {
                        let progress = ui.add(
                            egui::ProgressBar::new(1.)
                                .desired_width((ui.available_width() - 110.).max(40.))
                                .desired_height(5.)
                                .fill(helto::ACCENT),
                        );
                        progress.on_hover_text("Saved");
                        ui.label("100%");
                        if let Some(elapsed) = self.last_elapsed {
                            ui.label(
                                RichText::new(format!("{:.1}s", elapsed.as_secs_f64()))
                                    .color(helto::DIM),
                            )
                            .on_hover_text("Generation and saving time in this client");
                        }
                        if let Some((_, path, _)) = &self.result {
                            ui.response().on_hover_text(path.display().to_string());
                        }
                    } else {
                        ui.label(RichText::new(&self.status).color(helto::DIM));
                    }
                }
            });
            if self.working
                && let Some(progress) = &self.sampling
            {
                let mut text = format!("Sampling {} / {}", progress.completed, progress.total);
                if !self.cancelling
                    && self.status != "Cancelling"
                    && let Some(finish) = progress.finish_time()
                {
                    let remaining = finish
                        .duration_since(std::time::SystemTime::now())
                        .unwrap_or_default()
                        .as_secs();
                    text.push_str(&format!(
                        " · ETA ~{}:{:02} · ends ~{}",
                        remaining / 60,
                        remaining % 60,
                        local_time(finish)
                    ));
                }
                ui.label(RichText::new(text).small().color(helto::DIM));
            }
            if let Some(error) = error {
                ui.horizontal(|ui| {
                    if self.pending.is_some()
                        && !self.working
                        && ui
                            .add_enabled_ui(ready, |ui| {
                                helto::icon_button(ui, Icon::Retry, "Retry", false)
                            })
                            .inner
                            .clicked()
                    {
                        self.generate(ui.ctx());
                    }
                    egui::ScrollArea::vertical()
                        .id_salt("generation-error")
                        .max_height(38.)
                        .show(ui, |ui| {
                            ui.colored_label(helto::ERROR, error);
                        });
                });
            }
        });
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("imagine-settings"),
            egui::ViewportBuilder::default()
                .with_title("Imagine · Settings")
                .with_inner_size([360., 680.])
                .with_min_inner_size([360., 680.])
                .with_max_inner_size([360., 680.])
                .with_resizable(false)
                .with_app_id("imagine-settings")
                .with_window_type(egui::X11WindowType::Dialog)
                .with_decorations(false),
            |ui, class| {
                if ui.input(|i| i.viewport().close_requested() || i.key_pressed(egui::Key::Escape))
                {
                    self.settings_open = false;
                }
                self.settings_content(ui, class != egui::ViewportClass::EmbeddedWindow);
            },
        );
    }

    pub(super) fn settings_content(&mut self, ui: &mut Ui, native: bool) {
        let rect = ui.max_rect();
        helto::gradient(ui.painter(), rect, 10., helto::FIELD, helto::BG);
        ui.painter().rect_stroke(
            rect,
            10,
            Stroke::new(1., helto::RAISED),
            egui::StrokeKind::Inside,
        );
        let header = Rect::from_min_size(rect.min, egui::vec2(rect.width(), 56.));
        let drag = ui.interact(
            header.shrink2(egui::vec2(48., 0.)),
            ui.id().with("settings-drag"),
            egui::Sense::drag(),
        );
        if native && drag.drag_started() {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }
        region(
            ui,
            "settings-title",
            header.shrink2(egui::vec2(18., 0.)),
            |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.label(
                        RichText::new("Settings")
                            .size(18.)
                            .strong()
                            .color(helto::TEXT),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if helto::icon_button(ui, Icon::Close, "Close settings", false).clicked() {
                            self.settings_open = false;
                        }
                    });
                });
            },
        );
        ui.painter().hline(
            rect.x_range(),
            header.bottom(),
            Stroke::new(1., helto::RAISED),
        );
        let body = Rect::from_min_max(
            egui::pos2(rect.left() + 22., header.bottom() + 24.),
            rect.max - egui::vec2(22., 48.),
        );
        region(ui, "settings-body", body, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("settings-scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.label(
                        RichText::new("Generation")
                            .strong()
                            .size(14.)
                            .color(helto::TEXT),
                    );
                    ui.add_space(14.);
                    let before = self.prefs.generation.clone();
                    ui.scope(|ui| {
                        egui::Grid::new("generation-settings")
                            .num_columns(2)
                            .spacing([18., 10.])
                            .show(ui, |ui| {
                                let label = ui.label("Steps");
                                ui.add_sized(
                                    [ui.available_width().max(120.), 28.],
                                    egui::DragValue::new(&mut self.prefs.generation.steps)
                                        .range(1..=100),
                                )
                                .labelled_by(label.id);
                                ui.end_row();
                                let label = ui.label("CFG");
                                ui.add_sized(
                                    [ui.available_width().max(120.), 28.],
                                    egui::DragValue::new(&mut self.prefs.generation.cfg_scale)
                                        .range(0.0..=20.0)
                                        .speed(0.1),
                                )
                                .labelled_by(label.id);
                                ui.end_row();
                                let label = ui.label("Reference size");
                                let mut megapixels =
                                    f64::from(self.prefs.generation.reference_max_pixels_per_image)
                                        / 1_000_000.;
                                if ui
                                    .add_sized(
                                        [ui.available_width().max(120.), 28.],
                                        egui::DragValue::new(&mut megapixels)
                                            .range(0.262144..=4.0)
                                            .speed(0.1)
                                            .min_decimals(1)
                                            .max_decimals(6)
                                            .suffix(" MP"),
                                    )
                                    .labelled_by(label.id)
                                    .on_hover_text(
                                        "Maximum megapixels per reference image during generation.",
                                    )
                                    .changed()
                                {
                                    self.prefs.generation.reference_max_pixels_per_image =
                                        (megapixels * 1_000_000.).round() as u32;
                                }
                                ui.end_row();
                            });
                    });
                    if self.prefs.generation != before {
                        self.save_preferences();
                    }
                    ui.add_space(24.);
                    ui.separator();
                    ui.add_space(22.);
                    ui.label(
                        RichText::new("Storage")
                            .strong()
                            .size(14.)
                            .color(helto::TEXT),
                    );
                    ui.add_space(12.);
                    ui.label("Save to");
                    ui.horizontal(|ui| {
                        let width = (ui.available_width() - 40.).max(80.);
                        let text = self.prefs.output_folder.display().to_string();
                        egui::Frame::new()
                            .fill(helto::FIELD)
                            .stroke(Stroke::new(1., helto::RAISED))
                            .corner_radius(5)
                            .inner_margin(egui::Margin::symmetric(8, 6))
                            .show(ui, |ui| {
                                ui.set_width(width - 18.);
                                ui.add(
                                    egui::Label::new(RichText::new(&text).monospace()).truncate(),
                                )
                                .on_hover_text(text);
                            });
                        if ui
                            .add_enabled_ui(!self.picking_folder, |ui| {
                                helto::icon_button(ui, Icon::Folder, "Choose output folder", false)
                            })
                            .inner
                            .clicked()
                        {
                            self.choose_folder(ui.ctx());
                        }
                    });
                    if self.picking_folder {
                        ui.spinner();
                    }
                    ui.add_space(24.);
                    ui.separator();
                    ui.add_space(22.);
                    ui.label(
                        RichText::new("Privacy")
                            .strong()
                            .size(14.)
                            .color(helto::TEXT),
                    );
                    ui.add_space(14.);
                    ui.horizontal(|ui| {
                        ui.label("Hide previews");
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if helto::toggle(ui, &mut self.prefs.privacy_mode, "Hide previews")
                                .changed()
                            {
                                self.save_preferences();
                            }
                        });
                    });
                    if let Some(error) = &self.settings_error {
                        ui.add_space(14.);
                        ui.colored_label(helto::ERROR, error);
                    }
                });
        });
        ui.painter().hline(
            body.x_range(),
            rect.bottom() - 40.,
            Stroke::new(1., helto::RAISED),
        );
        let footer = Rect::from_min_max(
            egui::pos2(body.left(), rect.bottom() - 36.),
            rect.max - egui::vec2(22., 4.),
        );
        region(ui, "settings-save-state", footer, |ui| {
            ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                ui.label(
                    RichText::new(if self.settings_error.is_some() {
                        "Not saved"
                    } else {
                        "Saved"
                    })
                    .small()
                    .color(if self.settings_error.is_some() {
                        helto::ERROR
                    } else {
                        helto::DIM
                    }),
                );
            });
        });
    }
}
