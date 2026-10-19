use super::*;
use crate::icons::Icon;
use egui::{Align, Layout, Rect, RichText, Stroke};
use workspace::region;

/// A single reveal boundary for the entire dock, including headers and gaps.
fn panel_revealed(ui: &egui::Ui, rect: Rect, privacy: bool) -> bool {
    !privacy
        || (ui.input(|i| i.focused) && ui.rect_contains_pointer(rect.intersect(ui.clip_rect())))
}

impl ImagineApp {
    pub(super) fn queue_panel(&mut self, ui: &mut egui::Ui, rect: Rect) {
        workspace::panel(ui, rect, false);
        let revealed = panel_revealed(ui, rect, self.prefs.privacy_mode);
        let header = Rect::from_min_max(
            rect.min + egui::vec2(10., 6.),
            egui::pos2(rect.right() - 10., rect.top() + 38.),
        );
        region(ui, "queue-header", header, |ui| {
            ui.horizontal(|ui| {
                ui.label("Queue & results");
                let pending = self
                    .queue
                    .jobs
                    .iter()
                    .filter(|j| j.working && matches!(j.status.as_str(), "Queued" | "Submitting"))
                    .count();
                let active = self
                    .queue
                    .jobs
                    .iter()
                    .filter(|j| j.working && !matches!(j.status.as_str(), "Queued" | "Submitting"))
                    .count();
                ui.label(
                    RichText::new(format!("{active} active · {pending} waiting"))
                        .small()
                        .color(helto::DIM),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if helto::icon_button(
                        ui,
                        if self.queue.collapsed {
                            Icon::Up
                        } else {
                            Icon::Down
                        },
                        if self.queue.collapsed {
                            "Expand queue"
                        } else {
                            "Collapse queue"
                        },
                        false,
                    )
                    .clicked()
                    {
                        self.queue.collapsed = !self.queue.collapsed;
                    }
                });
            });
        });
        if self.queue.collapsed {
            return;
        }
        let top = rect.top() + 44.;
        let split = rect.left() + (rect.width() * 0.36).clamp(140., 360.);
        let queue_rect = Rect::from_min_max(
            egui::pos2(rect.left() + 10., top),
            egui::pos2(split - 8., rect.bottom() - 8.),
        );
        let results_rect =
            Rect::from_min_max(egui::pos2(split + 8., top), rect.max - egui::vec2(10., 8.));
        ui.painter().vline(
            split,
            top..=rect.bottom() - 8.,
            Stroke::new(1., helto::FIELD),
        );
        let mut cancel = None;
        let mut retry = None;
        region(ui, "queued-jobs", queue_rect, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("queue-rows")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for (index, job) in self
                        .queue
                        .jobs
                        .iter()
                        .enumerate()
                        .filter(|(_, j)| j.working || j.error.is_some())
                    {
                        ui.push_id(&job.id, |ui| {
                            ui.horizontal(|ui| {
                                let name = if revealed {
                                    job.snapshot.prompt.lines().next().unwrap_or("")
                                } else {
                                    "Hidden prompt"
                                };
                                let action_width = 38.;
                                ui.allocate_ui_with_layout(
                                    egui::vec2((ui.available_width() - action_width).max(30.), 0.),
                                    Layout::top_down(Align::Min),
                                    |ui| {
                                        ui.add(
                                            egui::Label::new(format!("#{} · {name}", index + 1))
                                                .truncate(),
                                        );
                                        ui.label(RichText::new(&job.status).small().color(
                                            if job.error.is_some() {
                                                helto::ERROR
                                            } else {
                                                helto::DIM
                                            },
                                        ));
                                        if let Some(p) = &job.sampling {
                                            ui.add(
                                                egui::ProgressBar::new(
                                                    p.completed as f32 / p.total.max(1) as f32,
                                                )
                                                .desired_width(ui.available_width())
                                                .desired_height(4.)
                                                .fill(helto::ACCENT),
                                            );
                                            ui.label(
                                                RichText::new(format!(
                                                    "{} / {} steps",
                                                    p.completed, p.total
                                                ))
                                                .small()
                                                .color(helto::DIM),
                                            );
                                        }
                                    },
                                );
                                if job.working {
                                    if ui
                                        .add_enabled_ui(
                                            job.submission.job.is_some() && !job.cancelling,
                                            |ui| {
                                                helto::icon_button(
                                                    ui,
                                                    Icon::Stop,
                                                    &format!("Cancel job {}", index + 1),
                                                    false,
                                                )
                                            },
                                        )
                                        .inner
                                        .clicked()
                                    {
                                        cancel = Some(job.id.clone());
                                    }
                                } else if ui
                                    .add_enabled_ui(
                                        self.client.is_some() && !self.connecting,
                                        |ui| {
                                            helto::icon_button(
                                                ui,
                                                Icon::Retry,
                                                &format!("Retry job {}", index + 1),
                                                false,
                                            )
                                        },
                                    )
                                    .inner
                                    .clicked()
                                {
                                    retry = Some(job.id.clone());
                                }
                            });
                            if let Some(error) = &job.error {
                                // Error details can contain file names. They share the panel's reveal boundary.
                                ui.label(
                                    RichText::new(if revealed {
                                        error.as_str()
                                    } else {
                                        "Hidden error"
                                    })
                                    .small()
                                    .color(helto::ERROR),
                                );
                            }
                            ui.separator();
                        });
                    }
                    if !self
                        .queue
                        .jobs
                        .iter()
                        .any(|j| j.working || j.error.is_some())
                    {
                        ui.label(RichText::new("Queue empty").color(helto::DIM));
                    }
                });
        });
        let mut selection = None;
        let mut follow_latest = false;
        let mut reuse = None;
        region(ui, "queue-results", results_rect, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("Results · {}", self.queue.gallery.len()))
                        .small()
                        .color(helto::DIM),
                );
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .checkbox(&mut self.queue.follow, "Follow latest")
                        .changed()
                        && self.queue.follow
                    {
                        selection = self.queue.gallery.last().map(|r| r.id.clone());
                        follow_latest = true;
                    }
                });
            });
            let can_select = !self.busy() && self.image.pending.is_none();
            egui::ScrollArea::horizontal()
                .id_salt("result-thumbnails")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for (index, item) in self.queue.gallery.iter().enumerate() {
                            let Some(asset) = item.result.assets.first() else {
                                continue;
                            };
                            ui.push_id(&item.id, |ui| {
                                ui.vertical(|ui| {
                                    let (thumb, response) = ui.allocate_exact_size(
                                        egui::vec2(116., 76.),
                                        egui::Sense::click(),
                                    );
                                    let selected = self
                                        .queue
                                        .loading
                                        .as_ref()
                                        .or(self.queue.selected.as_ref())
                                        == Some(&item.id);
                                    response.widget_info(|| {
                                        egui::WidgetInfo::selected(
                                            egui::WidgetType::Button,
                                            can_select,
                                            selected,
                                            format!("View result {}", index + 1),
                                        )
                                    });
                                    ui.painter().rect_filled(thumb, 4, helto::MEDIA);
                                    if revealed {
                                        let size = asset.texture.size_vec2();
                                        let scale = ((thumb.width() - 8.) / size.x)
                                            .min((thumb.height() - 8.) / size.y);
                                        ui.painter()
                                            .with_clip_rect(thumb.intersect(ui.clip_rect()))
                                            .image(
                                                asset.texture.id(),
                                                Rect::from_center_size(
                                                    thumb.center(),
                                                    size * scale,
                                                ),
                                                Rect::from_min_max(
                                                    egui::Pos2::ZERO,
                                                    egui::pos2(1., 1.),
                                                ),
                                                egui::Color32::WHITE,
                                            );
                                    } else {
                                        ui.painter().text(
                                            thumb.center(),
                                            egui::Align2::CENTER_CENTER,
                                            "Hidden",
                                            egui::FontId::proportional(12.),
                                            helto::DIM,
                                        );
                                    }
                                    ui.painter().rect_stroke(
                                        thumb,
                                        4,
                                        Stroke::new(
                                            1.,
                                            if selected {
                                                helto::ACCENT
                                            } else if response.has_focus() {
                                                helto::FOCUS
                                            } else {
                                                helto::RAISED
                                            },
                                        ),
                                        egui::StrokeKind::Inside,
                                    );
                                    if can_select && response.clicked() {
                                        selection = Some(item.id.clone());
                                    }
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(format!("#{}", index + 1))
                                                .small()
                                                .color(helto::DIM),
                                        );
                                        if item.snapshot.is_some()
                                            && ui
                                                .add_enabled_ui(self.editable(), |ui| {
                                                    helto::icon_button(
                                                        ui,
                                                        Icon::Duplicate,
                                                        &format!("Reuse settings {}", index + 1),
                                                        false,
                                                    )
                                                })
                                                .inner
                                                .clicked()
                                        {
                                            reuse = Some(item.id.clone());
                                        }
                                    });
                                });
                            });
                        }
                        if self.queue.gallery.is_empty() {
                            ui.label(RichText::new("No results yet").color(helto::DIM));
                        }
                    });
                });
        });
        if let Some(id) = cancel {
            self.cancel_queue_job(ui.ctx(), &id);
        }
        if let Some(id) = retry {
            self.retry_queue_job(ui.ctx(), &id);
        }
        if let Some(id) = selection {
            self.select_queue_result(ui.ctx(), &id);
            if follow_latest {
                self.queue.follow = true;
            }
        }
        if let Some(id) = reuse {
            self.reuse_queue_settings(&id);
        }
    }

    pub(super) fn multiple_prompts_window(&mut self, ctx: &egui::Context) {
        let Some(mut drafts) = self.queue.drafts.take() else {
            return;
        };
        let mut open = true;
        if ctx.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            return;
        }
        let mut enqueue = false;
        let mut remove = None;
        egui::Window::new("Multiple prompts")
            .id(egui::Id::new("multiple-prompts"))
            .open(&mut open)
            .default_width(480.)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("batch-prompts")
                    .max_height(340.)
                    .show(ui, |ui| {
                        for (index, prompt) in drafts.iter_mut().enumerate() {
                            ui.push_id(index, |ui| {
                                let label = ui
                                    .horizontal(|ui| {
                                        let label = ui.label(format!("Prompt {}", index + 1));
                                        if helto::icon_button(
                                            ui,
                                            Icon::Trash,
                                            &format!("Remove prompt {}", index + 1),
                                            false,
                                        )
                                        .clicked()
                                        {
                                            remove = Some(index);
                                        }
                                        label
                                    })
                                    .inner;
                                helto::textarea(
                                    ui,
                                    prompt,
                                    3,
                                    "batch-prompt",
                                    self.prefs.privacy_mode,
                                    true,
                                )
                                .labelled_by(label.id);
                            });
                        }
                    });
                ui.horizontal(|ui| {
                    if helto::icon_button(ui, Icon::Plus, "Add prompt", false).clicked() {
                        drafts.push(String::new());
                    }
                    enqueue = ui
                        .add_enabled(
                            !drafts.is_empty()
                                && drafts.iter().all(|p| !p.trim().is_empty())
                                && self.editable()
                                && self.client.is_some()
                                && !self.picking_folder,
                            egui::Button::new("Add all to queue"),
                        )
                        .clicked();
                });
                if let Some(error) = &self.queue.draft_error {
                    ui.colored_label(helto::ERROR, error);
                }
            });
        if let Some(index) = remove {
            drafts.remove(index);
        }
        if enqueue && self.enqueue_prompts(ctx, drafts.clone()) {
            open = false;
        }
        if open {
            self.queue.drafts = Some(drafts);
        }
    }
}
