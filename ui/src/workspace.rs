//! Reference desk: ordered inputs, an image viewport, and the prompt inspector.
use super::enhancement::Field;
use super::*;
use crate::icons::{self, Icon};
use egui::{Align, Layout, Rect, RichText, Stroke, Ui, UiBuilder, Vec2};

pub(super) fn region<R>(ui: &mut Ui, id: &str, rect: Rect, add: impl FnOnce(&mut Ui) -> R) -> R {
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

pub(super) fn panel(ui: &Ui, rect: Rect, media: bool) {
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

fn consume_ctrl_key(ctx: &egui::Context, key: egui::Key) -> bool {
    ctx.input_mut(|input| {
        let mut pressed = false;
        input.events.retain(|event| {
            if let egui::Event::Key {
                key: event_key,
                modifiers,
                pressed: true,
                repeat,
                ..
            } = event
                && *event_key == key
                && modifiers.ctrl
                && !modifiers.shift
                && !modifiers.alt
                && !modifiers.mac_cmd
            {
                pressed |= !repeat;
                return false;
            }
            true
        });
        pressed
    })
}

fn focus_prompt_shortcut(response: egui::Response) {
    let ctx = &response.ctx;
    if ctx.input(|input| input.focused)
        && !ctx.text_edit_focused()
        && consume_ctrl_key(ctx, egui::Key::P)
    {
        helto::focus_and_reveal_textarea(&response);
        response.scroll_to_me(Some(Align::Center));
    }
}

fn sampling_setting(
    ui: &mut Ui,
    metadata: Option<&Value>,
    name: &str,
    field: &str,
    selected: &mut Option<String>,
) {
    let label = ui.label(name);
    let default = metadata
        .and_then(|m| m["defaults"]["sampling"][field].as_str())
        .unwrap_or("Unavailable");
    let constraint = format!("{field}s");
    let choices = metadata.and_then(|m| m["constraints"][&constraint].as_array());
    let manual = field == "scheduler"
        && metadata.is_some_and(|m| {
            m["defaults"]["sampling"]["custom_sigmas"]
                .as_array()
                .is_some_and(|s| !s.is_empty())
        });
    let text = if manual {
        "Manual sigmas"
    } else {
        selected.as_deref().unwrap_or(default)
    }
    .to_owned();
    let response = ui
        .add_enabled_ui(!manual && choices.is_some(), |ui| {
            egui::ComboBox::from_id_salt(field)
                .width(ui.available_width().max(120.))
                .selected_text(text)
                .show_ui(ui, |ui| {
                    ui.selectable_value(selected, None, format!("Default ({default})"));
                    if let Some(choices) = choices {
                        for choice in choices.iter().filter_map(Value::as_str) {
                            ui.selectable_value(selected, Some(choice.to_owned()), choice);
                        }
                    }
                })
                .response
                .labelled_by(label.id)
        })
        .inner;
    if manual {
        response
            .on_disabled_hover_text("This model uses a fixed sigma list instead of a scheduler.");
    }
    ui.end_row();
}

struct ViewerProgress {
    working: bool,
    status: String,
    sampling: Option<client::SamplingProgress>,
    cancelling: bool,
    can_cancel: bool,
    queue_job: Option<(String, usize)>,
}

// Saving a downloaded result can overlap the next server job. Prefer the
// inference phase over local saving/retrieval and jobs still awaiting execution.
fn progress_priority(status: &str) -> u8 {
    match status {
        "Queued" | "Submitting" => 2,
        "Saving" | "Retrieving" => 1,
        _ => 0,
    }
}

impl ViewerProgress {
    fn label(&self) -> String {
        match &self.queue_job {
            Some((_, number)) => format!("#{number} · {}", self.status),
            None => self.status.clone(),
        }
    }
}

impl ImagineApp {
    fn viewer_progress(&self) -> ViewerProgress {
        let desk = self.desk();
        let mut progress = ViewerProgress {
            working: desk.working,
            status: desk.status.clone(),
            sampling: desk.sampling.clone(),
            cancelling: desk.cancelling,
            can_cancel: desk.working
                && !desk.cancelling
                && desk.pending.as_ref().is_some_and(|p| p.job.is_some()),
            queue_job: None,
        };
        if self.workspace == Workspace::Image
            && let Some((index, job)) = self
                .queue
                .jobs
                .iter()
                .enumerate()
                .filter(|(_, job)| job.working)
                .min_by_key(|(_, job)| progress_priority(&job.status))
            && (!desk.working || progress_priority(&job.status) < progress_priority(&desk.status))
        {
            progress = ViewerProgress {
                working: true,
                status: job.status.clone(),
                sampling: job.sampling.clone(),
                cancelling: job.cancelling,
                can_cancel: !job.cancelling && job.submission.job.is_some(),
                queue_job: Some((job.id.clone(), index + 1)),
            };
        }
        progress
    }

    pub(super) fn editable(&self) -> bool {
        !self.busy() && !self.connecting && self.desk().pending.is_none()
    }

    pub fn content(&mut self, ui: &mut Ui) {
        self.receive(ui.ctx());
        self.keyboard_shortcuts(ui.ctx());
        if self.busy()
            || self.connecting
            || self.queue.jobs.iter().any(|job| job.working)
            || self.queue.loading.is_some()
        {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }
        let bounds = ui.max_rect();
        ui.painter().rect_filled(bounds, 0, helto::BG);
        let header = Rect::from_min_size(bounds.min, egui::vec2(bounds.width(), 64.));
        self.header(ui, header);
        let dock_height = if self.queue.collapsed {
            44.
        } else {
            (bounds.height() * 0.26).clamp(204., 240.)
        };
        let dock = Rect::from_min_max(
            egui::pos2(bounds.left() + 8., bounds.bottom() - dock_height - 8.),
            bounds.max - egui::vec2(8., 8.),
        );
        let body = Rect::from_min_max(
            egui::pos2(bounds.left() + 8., header.bottom() + 8.),
            egui::pos2(bounds.right() - 8., dock.top() - 8.),
        );
        region(ui, "workspace", body, |ui| match self.workspace {
            Workspace::Image | Workspace::ImageEdit | Workspace::CharacterSheet
                if body.width() >= 900. =>
            {
                ui.push_id(self.workspace, |ui| self.image_workspace(ui, body));
            }
            Workspace::Image | Workspace::ImageEdit | Workspace::CharacterSheet => {
                egui::ScrollArea::vertical()
                    .id_salt("compact-workspace")
                    .show(ui, |ui| {
                        let size = egui::vec2(ui.available_width(), body.height().max(920.));
                        let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                        ui.push_id(self.workspace, |ui| self.image_workspace(ui, rect));
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
        self.queue_panel(ui, dock);
        self.multiple_prompts_window(ui.ctx());
        if self.models.open {
            self.loras_window(ui.ctx());
        }
        if self.settings_open {
            self.settings_window(ui.ctx());
        }
        self.enhancement_proposal(ui.ctx());
    }

    fn keyboard_shortcuts(&mut self, ctx: &egui::Context) {
        if !ctx.input(|input| input.focused) || ctx.text_edit_focused() {
            return;
        }
        if consume_ctrl_key(ctx, egui::Key::Space) {
            self.prefs.privacy_mode = !self.prefs.privacy_mode;
            self.save_preferences();
        }
        if self.queue.drafts.is_some() {
            return;
        }
        if self.workspace == Workspace::Video {
            return;
        }
        if consume_ctrl_key(ctx, egui::Key::Z) && self.desk().result.is_some() {
            let desk = self.desk_mut();
            desk.actual_pixels = !desk.actual_pixels;
            desk.image_pan = Vec2::ZERO;
        }
        if consume_ctrl_key(ctx, egui::Key::Enter)
            && self.editable()
            && self.client.is_some()
            && !self.picking_folder
            && self.can_generate()
        {
            self.generate(ctx);
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
                        ui.label(RichText::new(self.selected_model_name()).color(helto::DIM));
                    }
                });
            },
        );
        let tabs = Rect::from_center_size(rect.center(), egui::vec2(110., 32.));
        region(ui, "tabs", tabs, |ui| {
            ui.horizontal(|ui| {
                if helto::icon_button(
                    ui,
                    Icon::Image,
                    "Image",
                    matches!(self.workspace, Workspace::Image | Workspace::ImageEdit),
                )
                .clicked()
                {
                    self.workspace = if self.edit.open {
                        Workspace::ImageEdit
                    } else {
                        Workspace::Image
                    };
                }
                if helto::icon_button(
                    ui,
                    Icon::CharacterSheet,
                    "Character sheets",
                    self.workspace == Workspace::CharacterSheet,
                )
                .clicked()
                {
                    self.workspace = Workspace::CharacterSheet;
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
                    let reconnect = (self.client.is_none()
                        || self.connection_error.is_some()
                        || self.desk().error.is_some())
                        && !self.busy()
                        && !self.connecting
                        && self.desk().pending.is_none();
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
        if self.workspace == Workspace::ImageEdit {
            self.edit_workspace(ui, rect);
            return;
        }
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

    fn edit_workspace(&mut self, ui: &mut Ui, rect: Rect) {
        let (viewer, inspector) = if rect.width() >= 900. {
            let width = (rect.width() * 0.28).clamp(300., 380.);
            (
                Rect::from_min_max(
                    rect.min,
                    egui::pos2(rect.right() - width - 8., rect.bottom()),
                ),
                Rect::from_min_max(egui::pos2(rect.right() - width, rect.top()), rect.max),
            )
        } else {
            (
                Rect::from_min_size(rect.min, egui::vec2(rect.width(), 360.)),
                Rect::from_min_max(egui::pos2(rect.left(), rect.top() + 368.), rect.max),
            )
        };
        panel(ui, viewer, true);
        panel(ui, inspector, false);
        region(ui, "edit-viewer", viewer.shrink(10.), |ui| self.viewer(ui));
        region(ui, "edit-inspector", inspector.shrink(12.), |ui| {
            egui::ScrollArea::vertical()
                .id_salt("edit-inspector-scroll")
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Edit image");
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui
                                .add_enabled_ui(self.editable(), |ui| {
                                    helto::icon_button(ui, Icon::Close, "Finish editing", false)
                                })
                                .inner
                                .clicked()
                            {
                                self.finish_editing();
                            }
                        });
                    });
                    if self.workspace != Workspace::ImageEdit {
                        return;
                    }
                    ui.add_space(12.);
                    self.reference_panel(ui);
                    ui.add_space(12.);
                    self.model_controls(ui);
                    ui.add_space(8.);
                    let label = self.enhancement_label(ui, "Prompt", Field::Prompt);
                    let response = helto::textarea(
                        ui,
                        &mut self.edit.desk.prompt,
                        6,
                        "edit-prompt",
                        self.prefs.privacy_mode,
                        true,
                    )
                    .labelled_by(label.id)
                    .on_hover_text("Use Reference 1, Reference 2, … in the prompt.");
                    self.enhancement_field(ui, &response, Field::Prompt);
                    focus_prompt_shortcut(response);
                    ui.add_space(12.);
                    if ui
                        .add_enabled_ui(
                            self.editable()
                                && self.client.is_some()
                                && !self.picking_folder
                                && self.can_generate(),
                            |ui| helto::icon_button(ui, Icon::Generate, "Create version", true),
                        )
                        .inner
                        .clicked()
                    {
                        self.generate(ui.ctx());
                    }
                    if let Some(result) = self.edit.desk.selected_result() {
                        let [w, h] = result.texture.size();
                        ui.label(
                            RichText::new(format!("{w} × {h} px"))
                                .small()
                                .color(helto::DIM),
                        );
                    }
                });
        });
    }

    fn edit_version_selector(&mut self, ui: &mut Ui) {
        let mut selected = self.edit.selected;
        let name = |index| {
            if index == 0 {
                "Original".to_string()
            } else {
                format!("Version {index}")
            }
        };
        ui.add_enabled_ui(self.editable(), |ui| {
            egui::ComboBox::from_id_salt("edit-version")
                .selected_text(name(selected))
                .show_ui(ui, |ui| {
                    for index in 0..self.edit.versions.len() {
                        ui.selectable_value(&mut selected, index, name(index));
                    }
                });
        });
        if selected != self.edit.selected {
            self.select_edit_version(selected);
        }
    }

    fn reference_panel(&mut self, ui: &mut Ui) {
        let editable = self.editable();
        let editing = self.workspace == Workspace::ImageEdit;
        ui.horizontal(|ui| {
            ui.label(format!(
                "References · {}/{}",
                self.desk().references.len(),
                self.reference_limit()
            ));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui
                    .add_enabled_ui(
                        editable
                            && self.client.is_some()
                            && self.desk().references.len() < self.reference_limit(),
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
        if self.desk().uploading {
            ui.spinner();
        }
        let mut remove = None;
        let mut swap = None;
        egui::ScrollArea::vertical()
            .id_salt("reference-list")
            .max_height(if editing { 230. } else { f32::INFINITY })
            .auto_shrink([false, editing])
            .show(ui, |ui| {
                for (index, reference) in self.desk().references.iter().enumerate() {
                    ui.push_id(&reference.id, |ui| {
                        if editing {
                            ui.horizontal(|ui| {
                                private_image(
                                    ui,
                                    &reference.texture,
                                    egui::vec2(72., 56.),
                                    self.prefs.privacy_mode,
                                );
                                ui.label(format!("Reference {}", reference.number));
                                if ui
                                    .add_enabled_ui(editable, |ui| {
                                        helto::icon_button(
                                            ui,
                                            Icon::Trash,
                                            &format!("Remove reference {}", reference.number),
                                            false,
                                        )
                                    })
                                    .inner
                                    .clicked()
                                {
                                    remove = Some(index);
                                }
                            });
                            ui.add_space(6.);
                            return;
                        }
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
                                        editable && index + 1 < self.desk().references.len(),
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
            self.desk_mut().references.remove(index);
        } else if let Some((a, b)) = swap {
            self.desk_mut().references.swap(a, b);
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
                    ui.scope(|ui| {
                        self.model_controls(ui);
                        ui.add_space(8.);
                        let privacy = self.prefs.privacy_mode;
                        if self.workspace == Workspace::CharacterSheet {
                            let label = self.enhancement_label(ui, "Character", Field::Character);
                            let response = helto::textarea(
                                ui,
                                &mut self.sheet.character_description,
                                6,
                                "character",
                                privacy,
                                true,
                            )
                            .labelled_by(label.id);
                            self.enhancement_field(ui, &response, Field::Character);
                            focus_prompt_shortcut(response);
                            ui.add_space(8.);
                            let label = self.enhancement_label(
                                ui,
                                "Clothing & accessories",
                                Field::Clothing,
                            );
                            let response = helto::textarea(
                                ui,
                                &mut self.sheet.clothing_description,
                                5,
                                "clothing",
                                privacy,
                                true,
                            )
                            .labelled_by(label.id);
                            self.enhancement_field(ui, &response, Field::Clothing);
                        } else {
                            let label = self.enhancement_label(ui, "Prompt", Field::Prompt);
                            let response = helto::textarea(
                                ui,
                                &mut self.image.prompt,
                                8,
                                "prompt",
                                privacy,
                                true,
                            )
                            .labelled_by(label.id);
                            self.enhancement_field(ui, &response, Field::Prompt);
                            focus_prompt_shortcut(response);
                        }
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
        if let Some(seed) = self.desk().result.as_ref().and_then(|r| r.seed) {
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
        let before_width = self.prefs.sheet_width;
        let sheet = self.workspace == Workspace::CharacterSheet;
        let settings = &mut self.prefs.generation;
        let field_width = (ui.available_width() - 80.).max(90.);
        egui::Grid::new("image-settings")
            .num_columns(2)
            .spacing([10., 10.])
            .show(ui, |ui| {
                if sheet {
                    let label = ui.label("Ratio");
                    ui.add_enabled(
                        false,
                        egui::Button::new("16:9").min_size(egui::vec2(field_width, 32.)),
                    )
                    .labelled_by(label.id);
                    ui.end_row();
                    let label = ui.label("Width");
                    ui.add_sized(
                        [field_width, 32.],
                        egui::DragValue::new(&mut self.prefs.sheet_width)
                            .range(512..=4096)
                            .speed(512.)
                            .custom_parser(|s| {
                                let value: u32 = s.parse().ok()?;
                                (value.is_multiple_of(512)).then_some(value as f64)
                            }),
                    )
                    .labelled_by(label.id)
                    .on_hover_text("Pixels · multiples of 512");
                    self.prefs.sheet_width =
                        (self.prefs.sheet_width.clamp(512, 4096) + 256) / 512 * 512;
                    ui.end_row();
                    ui.label("");
                    ui.label(
                        RichText::new(format!(
                            "{} × {} px",
                            self.prefs.sheet_width,
                            self.prefs.sheet_width * 9 / 16
                        ))
                        .color(helto::DIM),
                    );
                    ui.end_row();
                } else {
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
                }
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
        if !sheet && let Err(e) = settings.dimensions() {
            ui.colored_label(helto::ERROR, e.to_string());
        }
        if *settings != before || self.prefs.sheet_width != before_width {
            self.save_preferences();
        }
    }

    fn viewer(&mut self, ui: &mut Ui) {
        let bounds = ui.max_rect();
        let activity = self.viewer_progress();
        let sheet = self.workspace == Workspace::CharacterSheet;
        let editing = self.workspace == Workspace::ImageEdit;
        let stacked_tools = sheet && bounds.width() < 360.;
        let tools_height = if stacked_tools { 70. } else { 32. };
        if sheet {
            let selectors = Rect::from_min_size(
                bounds.min,
                egui::vec2(
                    if stacked_tools {
                        bounds.width()
                    } else {
                        bounds.width() - 118.
                    },
                    32.,
                ),
            );
            region(ui, "result-selector", selectors, |ui| {
                ui.horizontal(|ui| {
                    for (role, label) in [
                        (OutputRole::Sheet, "Sheet"),
                        (OutputRole::Portrait, "Portrait"),
                        (OutputRole::Views, "Views"),
                    ] {
                        if helto::select_button(ui, label, self.desk().selected_role == role)
                            .clicked()
                        {
                            self.desk_mut().select_result(role);
                        }
                    }
                });
            });
        }
        region(
            ui,
            "view-tools",
            Rect::from_min_size(
                bounds.min + egui::vec2(0., if stacked_tools { 38. } else { 0. }),
                egui::vec2(bounds.width(), 32.),
            ),
            |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if self.workspace == Workspace::Image
                        && ui
                            .add_enabled_ui(self.can_edit_image(), |ui| {
                                helto::icon_button(ui, Icon::Edit, "Edit image", false)
                            })
                            .inner
                            .clicked()
                    {
                        self.start_editing();
                    }
                    ui.add_enabled_ui(self.desk().result.is_some(), |ui| {
                        if helto::icon_button(
                            ui,
                            Icon::Hand,
                            "Pan image",
                            self.desk().actual_pixels && self.desk().pan_enabled,
                        )
                        .clicked()
                        {
                            if !self.desk().actual_pixels {
                                self.desk_mut().actual_pixels = true;
                                self.desk_mut().pan_enabled = true;
                            } else {
                                self.desk_mut().pan_enabled = !self.desk().pan_enabled;
                            }
                        }
                        if helto::icon_button(
                            ui,
                            Icon::Actual,
                            "Actual pixels",
                            self.desk().actual_pixels,
                        )
                        .clicked()
                        {
                            self.desk_mut().actual_pixels = true;
                            self.desk_mut().image_pan = Vec2::ZERO;
                        }
                        if helto::icon_button(
                            ui,
                            Icon::Fit,
                            "Fit image",
                            !self.desk().actual_pixels,
                        )
                        .clicked()
                        {
                            self.desk_mut().actual_pixels = false;
                            self.desk_mut().image_pan = Vec2::ZERO;
                        }
                    });
                });
            },
        );
        let error = self.desk().error.clone().or_else(|| self.connection_error.clone()).or_else(|| self.settings_error.clone()).or_else(|| {
            (sheet && self.metadata.as_ref().is_some_and(|m| !client::supports_character_sheets(m))).then(|| "This server does not support character sheets. Update the server and reconnect.".into())
        });
        let footer_height = (if error.is_some() {
            132.
        } else if activity.working && activity.sampling.is_some() {
            92.
        } else {
            66.
        }) + if editing { 38. } else { 0. };
        let image_rect = Rect::from_min_max(
            bounds.min + egui::vec2(0., tools_height + 6.),
            bounds.max - egui::vec2(0., footer_height),
        );
        region(ui, "image", image_rect, |ui| {
            let privacy = self.prefs.privacy_mode;
            let before = self
                .edit_comparison_source()
                .map(|source| source.texture.clone());
            let state = self.desk_mut();
            if let Some(texture) = state.selected_result().map(|r| r.texture.clone()) {
                image_view_layers(
                    ui,
                    (&texture, before.as_ref()),
                    image_rect.size(),
                    privacy,
                    state.actual_pixels,
                    state.pan_enabled,
                    &mut state.image_pan,
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
                    if sheet {
                        "No character sheet yet"
                    } else {
                        "No image yet"
                    },
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
            if editing {
                self.edit_version_selector(ui);
            }
            ui.separator();
            let ready =
                self.client.is_some() && !self.connecting && !self.busy() && !self.picking_folder;
            ui.horizontal(|ui| {
                if !editing
                    && ui
                        .add_enabled_ui(
                            ready && self.desk().pending.is_none() && self.can_generate(),
                            |ui| {
                                helto::icon_button(
                                    ui,
                                    Icon::Generate,
                                    if self.workspace == Workspace::Image {
                                        "Add to queue"
                                    } else {
                                        "Generate"
                                    },
                                    true,
                                )
                            },
                        )
                        .inner
                        .clicked()
                {
                    self.generate(ui.ctx());
                }
                if self.workspace == Workspace::Image
                    && ui
                        .add_enabled_ui(ready && self.image.pending.is_none(), |ui| {
                            helto::icon_button(ui, Icon::Duplicate, "Multiple prompts", false)
                        })
                        .inner
                        .clicked()
                {
                    self.queue.drafts = Some(vec![self.image.prompt.clone(), String::new()]);
                    self.queue.draft_error = None;
                }
                if !editing
                    && ui
                        .add_enabled_ui(
                            ready
                                && self.desk().pending.is_none()
                                && self.upscale_source().is_some(),
                            |ui| {
                                helto::icon_button(ui, Icon::Upscale, "Upscale", false)
                                    .on_hover_text(
                                        self.prefs
                                            .upscale_target
                                            .dimensions(self.upscale_ratio())
                                            .map(|(w, h)| {
                                                format!(
                                                    "{} · {w} × {h} px",
                                                    if sheet { "Character sheet" } else { "Image" }
                                                )
                                            })
                                            .unwrap_or_else(|e| e.to_string()),
                                    )
                            },
                        )
                        .inner
                        .clicked()
                {
                    self.upscale(ui.ctx());
                }
                if ui
                    .add_enabled_ui(activity.can_cancel, |ui| {
                        helto::icon_button(ui, Icon::Stop, "Cancel generation", false)
                    })
                    .inner
                    .clicked()
                {
                    if let Some((id, _)) = &activity.queue_job {
                        self.cancel_queue_job(ui.ctx(), id);
                    } else {
                        self.cancel(ui.ctx());
                    }
                }
                ui.add_space(6.);
                if activity.working
                    && let Some(progress) = &activity.sampling
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
                    if activity.working {
                        ui.spinner();
                    }
                    if !activity.working
                        && self.desk().status == "Saved"
                        && self.desk().result.is_some()
                    {
                        let progress = ui.add(
                            egui::ProgressBar::new(1.)
                                .desired_width((ui.available_width() - 110.).max(40.))
                                .desired_height(5.)
                                .fill(helto::ACCENT),
                        );
                        progress.on_hover_text("Saved");
                        ui.label("100%");
                        if let Some(elapsed) = self.desk().last_elapsed {
                            ui.label(
                                RichText::new(format!("{:.1}s", elapsed.as_secs_f64()))
                                    .color(helto::DIM),
                            )
                            .on_hover_text("Generation and saving time in this client");
                        }
                        if let Some(result) = self.desk().selected_result() {
                            ui.response()
                                .on_hover_text(result.file.display().to_string());
                        }
                    } else {
                        ui.label(RichText::new(activity.label()).color(helto::DIM));
                    }
                }
            });
            if activity.working
                && let Some(progress) = &activity.sampling
            {
                let mut text = format!(
                    "{} {} / {}",
                    activity.label(),
                    progress.completed,
                    progress.total
                );
                if !activity.cancelling
                    && activity.status != "Cancelling"
                    && let Some(finish) = progress.finish_time()
                {
                    let remaining = finish
                        .duration_since(std::time::SystemTime::now())
                        .unwrap_or_default()
                        .as_secs();
                    text.push_str(&format!(
                        " · {}ETA ~{}:{:02} · ends ~{}",
                        if sheet { "phase " } else { "" },
                        remaining / 60,
                        remaining % 60,
                        local_time(finish)
                    ));
                }
                ui.label(RichText::new(text).small().color(helto::DIM));
            }
            if let Some(error) = error {
                ui.horizontal(|ui| {
                    if self.desk().pending.is_some()
                        && !self.desk().working
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
                let popup_open = egui::Popup::is_any_open(ui.ctx());
                if ui.input(|i| {
                    i.viewport().close_requested()
                        || (i.key_pressed(egui::Key::Escape) && !popup_open)
                }) {
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
                    self.model_settings(ui);
                    ui.add_space(18.);
                    ui.label(
                        RichText::new("Generation")
                            .strong()
                            .size(14.)
                            .color(helto::TEXT),
                    );
                    ui.add_space(14.);
                    let before = self.prefs.generation.clone();
                    let before_upscale = self.prefs.upscale_target;
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
                                sampling_setting(
                                    ui,
                                    self.metadata.as_ref(),
                                    "Sampler",
                                    "sampler",
                                    &mut self.prefs.generation.sampler,
                                );
                                sampling_setting(
                                    ui,
                                    self.metadata.as_ref(),
                                    "Scheduler",
                                    "scheduler",
                                    &mut self.prefs.generation.scheduler,
                                );
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
                                let label = ui.label("Upscale target");
                                egui::ComboBox::from_id_salt("upscale-target")
                                    .width(ui.available_width().max(120.))
                                    .selected_text(self.prefs.upscale_target.label())
                                    .show_ui(ui, |ui| {
                                        for target in crate::settings::UpscaleTarget::ALL {
                                            ui.selectable_value(
                                                &mut self.prefs.upscale_target,
                                                target,
                                                target.label(),
                                            );
                                        }
                                    })
                                    .response
                                    .labelled_by(label.id);
                                ui.end_row();
                            });
                    });
                    if self.prefs.generation != before
                        || self.prefs.upscale_target != before_upscale
                    {
                        self.save_preferences();
                    }
                    ui.add_space(24.);
                    ui.separator();
                    ui.add_space(22.);
                    self.enhancement_settings(ui);
                    ui.add_space(16.);
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
                        self.enhancement_save_status().unwrap_or("Saved")
                    })
                    .small()
                    .color(
                        if self.settings_error.is_some()
                            || self.enhancement_save_status() == Some("Profile not saved")
                        {
                            helto::ERROR
                        } else {
                            helto::DIM
                        },
                    ),
                );
            });
        });
    }
}
