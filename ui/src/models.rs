//! Model selection and a persistent, group-scoped LoRA desk.
use super::*;
use crate::{
    icons::Icon,
    model_catalog::{
        Components, LibrarySettings, LibrarySnapshot, ModelFile, SelectedLora, Selection,
    },
    settings::{LoraSetting, ModelGroupSettings},
};
use anyhow::{Result, bail};
use egui::{RichText, Ui};
use serde_json::json;

#[cfg(test)]
#[path = "component_picker_tests.rs"]
mod component_picker_tests;

#[derive(Default)]
pub(super) struct ModelUi {
    pub catalog: Option<LibrarySnapshot>,
    pub loading: bool,
    pub error: Option<String>,
    pub open: bool,
    pub dirty: bool,
    search: String,
}

#[derive(Clone, Default)]
struct ComponentSearch {
    query: String,
    highlighted: usize,
}

fn component_picker(
    ui: &mut Ui,
    label: &str,
    files: &[ModelFile],
    kind: &str,
    selected: &mut String,
) {
    let id = ui.make_persistent_id("component-search");
    let popup = id.with("results");
    let mut search = ui.data_mut(|data| data.get_temp::<ComponentSearch>(id).unwrap_or_default());
    let mut open = egui::Popup::is_id_open(ui.ctx(), popup);
    let (mut down, mut up, mut choose, mut dismiss) = (false, false, false, false);
    if open {
        ui.input_mut(|input| {
            down = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown);
            up = input.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp);
            choose = input.consume_key(egui::Modifiers::NONE, egui::Key::Enter);
            dismiss = input.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
        });
    }
    if dismiss {
        egui::Popup::close_id(ui.ctx(), popup);
        open = false;
        ui.memory_mut(|memory| memory.surrender_focus(id));
    }
    let name = files
        .iter()
        .find(|f| f.id == *selected && f.kind == kind)
        .map(|f| f.path.as_str());
    let mut text = if open {
        search.query.clone()
    } else {
        name.unwrap_or(if selected.is_empty() {
            ""
        } else {
            "Missing file"
        })
        .to_owned()
    };
    let label = ui.label(label);
    let response = ui
        .add(
            egui::TextEdit::singleline(&mut text)
                .id(id)
                .hint_text("Select…")
                .desired_width(f32::INFINITY),
        )
        .labelled_by(label.id);
    if !dismiss && (response.gained_focus() || response.clicked()) && !open {
        search.query.clear();
        search.highlighted = files
            .iter()
            .filter(|f| f.kind == kind)
            .position(|f| f.id == *selected)
            .unwrap_or(0);
        egui::Popup::open_id(ui.ctx(), popup);
        open = true;
        ui.ctx().request_repaint();
    } else if open && response.changed() {
        search.query = text;
        search.highlighted = 0;
    }
    if response.lost_focus()
        && !ui.input(|input| input.pointer.any_down() || input.pointer.any_released())
    {
        egui::Popup::close_id(ui.ctx(), popup);
        open = false;
    }
    let query = search.query.to_lowercase();
    let terms: Vec<_> = query.split_whitespace().collect();
    let results: Vec<_> = files
        .iter()
        .filter(|file| {
            let path = file.path.to_lowercase();
            file.kind == kind && terms.iter().all(|term| path.contains(term))
        })
        .collect();
    let last = results.len().saturating_sub(1);
    search.highlighted = search.highlighted.min(last);
    if down {
        search.highlighted = (search.highlighted + 1).min(last);
    }
    if up {
        search.highlighted = search.highlighted.saturating_sub(1);
    }
    let mut chosen = (choose && open)
        .then(|| results.get(search.highlighted).map(|file| file.id.clone()))
        .flatten();
    egui::Popup::from_response(&response)
        .id(popup)
        .open_memory(None)
        .width(response.rect.width())
        .close_behavior(if response.clicked() {
            egui::PopupCloseBehavior::IgnoreClicks
        } else {
            egui::PopupCloseBehavior::CloseOnClickOutside
        })
        .show(|ui| {
            egui::ScrollArea::vertical()
                .id_salt(popup)
                .max_height(180.)
                .show(ui, |ui| {
                    if results.is_empty() {
                        ui.label("No matching files");
                    }
                    for (index, file) in results.iter().enumerate() {
                        let row = ui
                            .add(
                                egui::Button::selectable(index == search.highlighted, &file.path)
                                    .truncate(),
                            )
                            .on_hover_text(&file.path);
                        if index == search.highlighted && (down || up || response.changed()) {
                            row.scroll_to_me(Some(egui::Align::Center));
                        }
                        if row.clicked() {
                            chosen = Some(file.id.clone());
                        }
                    }
                });
        });
    if let Some(chosen) = chosen {
        *selected = chosen;
        egui::Popup::close_id(ui.ctx(), popup);
        ui.memory_mut(|memory| memory.surrender_focus(id));
    }
    ui.data_mut(|data| data.insert_temp(id, search));
}

impl ImagineApp {
    fn group_settings(&self) -> ModelGroupSettings {
        self.prefs
            .model_groups
            .get(&self.prefs.model_group)
            .cloned()
            .unwrap_or_default()
    }
    pub(super) fn selected_model(&self) -> Option<&Value> {
        let selected = self.prefs.model_groups.get(&self.prefs.model_group)?;
        self.models
            .catalog
            .as_ref()?
            .models
            .iter()
            .find(|m| m["id"] == selected.variant && m["group"] == self.prefs.model_group)
    }
    pub(super) fn selected_model_name(&self) -> String {
        self.selected_model()
            .and_then(|m| m["name"].as_str())
            .unwrap_or(if self.models.catalog.is_some() {
                "Select model"
            } else {
                "Qwen Image 2.1"
            })
            .into()
    }
    pub(super) fn accept_library(&mut self, library: LibrarySnapshot) {
        let group = self.prefs.model_group.clone();
        let choices = self.prefs.model_groups.entry(group.clone()).or_default();
        let initial = choices.variant.is_empty();
        if initial {
            // Prefer the existing INT8 behavior on first use; never replace a missing saved selection.
            if let Some(model) = library
                .models
                .iter()
                .filter(|m| m["group"] == group)
                .min_by_key(|m| m["model"] == "qwen-image-2.1-turbo")
            {
                choices.variant = model["id"].as_str().unwrap().into();
            }
        }
        self.models.catalog = Some(library);
        self.metadata = self.selected_model().cloned();
        if initial && self.metadata.is_some() {
            if self
                .metadata
                .as_ref()
                .is_some_and(|m| m["model"] == "qwen-image-2.1-turbo")
            {
                self.prefs.generation.steps = 8;
                self.prefs.generation.cfg_scale = 1.;
            }
            self.save_preferences();
        }
    }
    pub(super) fn select_model(&mut self, id: String) {
        if self.group_settings().variant == id {
            return;
        }
        if let Some(group) = self
            .models
            .catalog
            .as_ref()
            .and_then(|c| c.models.iter().find(|m| m["id"] == id))
            .and_then(|m| m["group"].as_str())
        {
            self.prefs.model_group = group.to_owned();
        }
        self.prefs
            .model_groups
            .entry(self.prefs.model_group.clone())
            .or_default()
            .variant = id;
        self.metadata = self.selected_model().cloned();
        if let Some(meta) = &self.metadata {
            self.prefs.generation.sampler = None;
            self.prefs.generation.scheduler = None;
            if let Some(steps) = meta["defaults"]["sampling"]["steps"].as_u64() {
                self.prefs.generation.steps = steps as u32;
            }
            if let Some(cfg) = meta["defaults"]["sampling"]["cfg_scale"].as_f64() {
                self.prefs.generation.cfg_scale = cfg as f32;
            }
        }
        if self
            .metadata
            .as_ref()
            .is_some_and(|m| m["constraints"]["loras"] == false)
        {
            self.models.open = false;
        }
        self.save_preferences();
    }
    pub(super) fn model_selection(&self) -> Result<Selection> {
        let library = self
            .models
            .catalog
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Reconnect to load models."))?;
        let model = self
            .selected_model()
            .ok_or_else(|| anyhow::anyhow!("Select an available model."))?;
        if model["available"] != true {
            bail!(
                "{}",
                client::error_message(
                    model["problem"]
                        .as_str()
                        .unwrap_or("model_component_missing")
                )
            );
        }
        let group = self.group_settings();
        let mut loras = Vec::new();
        for lora in group.loras.into_iter().filter(|l| l.enabled) {
            let file = library
                .files
                .iter()
                .find(|f| f.id == lora.id && f.kind == "loras");
            if file.is_none() {
                bail!(
                    "LoRA unavailable: {}. Disable it or update Models in Settings.",
                    lora.name
                );
            }
            if !lora.strength.is_finite() {
                bail!("Enter a finite LoRA strength.");
            }
            loras.push(SelectedLora {
                id: lora.id,
                strength: lora.strength,
            });
        }
        Ok(Selection {
            diffusion: group.variant,
            loras,
        })
    }
    pub(super) fn apply_model_selection(&self, mut request: Value) -> Result<Value> {
        if self.models.catalog.is_some() {
            request["model_selection"] = json!(self.model_selection()?);
        }
        Ok(request)
    }
    pub(super) fn load_library(&mut self, ctx: &egui::Context, settings: Option<LibrarySettings>) {
        if self.models.loading {
            return;
        }
        let Some(client) = self.client.clone() else {
            return;
        };
        self.models.loading = true;
        self.models.error = None;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime()
                .block_on(client.library(settings.as_ref(), true))
                .map_err(|e| e.message);
            send(&tx, &ctx, Message::Library(result));
        });
    }
    fn choose_model_folder(&mut self, ctx: &egui::Context) {
        if self.models.loading {
            return;
        }
        let Some(library) = &self.models.catalog else {
            return;
        };
        let root = library.settings.root.clone();
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        self.models.loading = true;
        std::thread::spawn(move || {
            let path = rfd::FileDialog::new()
                .set_title("Choose models folder")
                .set_directory(root)
                .pick_folder();
            send(&tx, &ctx, Message::ModelFolder(path));
        });
    }
    pub(super) fn model_controls(&mut self, ui: &mut Ui) {
        let Some(library) = self.models.catalog.clone() else {
            return;
        };
        let mut selected = self.group_settings().variant;
        let enabled = self.editable() && !self.models.loading;
        ui.add_enabled_ui(enabled, |ui| {
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 65.).max(100.);
                ui.allocate_ui_with_layout(
                    egui::vec2(width, ui.spacing().interact_size.y),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        egui::ComboBox::from_id_salt("model-selection")
                            .width(width)
                            .truncate()
                            .selected_text(self.selected_model_name())
                            .show_ui(ui, |ui| {
                                for model in &library.models {
                                    let id = model["id"].as_str().unwrap_or_default();
                                    let name = model["name"].as_str().unwrap_or_default();
                                    let label = if model["available"] == true {
                                        name.to_owned()
                                    } else {
                                        format!("{name} · Setup required")
                                    };
                                    ui.selectable_value(&mut selected, id.into(), label)
                                        .on_hover_text(model["path"].as_str().unwrap_or_default());
                                }
                            })
                    },
                );
                let count = self
                    .group_settings()
                    .loras
                    .iter()
                    .filter(|l| l.enabled)
                    .count();
                if ui
                    .add_enabled_ui(
                        self.metadata
                            .as_ref()
                            .is_none_or(|m| m["constraints"]["loras"] != false),
                        |ui| helto::icon_button(ui, Icon::Lora, "LoRAs", self.models.open),
                    )
                    .inner
                    .clicked()
                {
                    self.models.open = !self.models.open;
                }
                if count > 0 {
                    ui.label(RichText::new(count.to_string()).small());
                }
            });
        });
        self.select_model(selected);
        if let Some(model) = &self.metadata
            && let (Some(architecture), Some(format)) = (
                model["architecture"].as_str(),
                model["checkpoint"]["format"].as_str(),
            )
        {
            ui.label(
                RichText::new(format!(
                    "{architecture} · {}",
                    if format == "bf16" { "BF16" } else { "INT8" }
                ))
                .small(),
            );
        }
        if let Err(error) = self.model_selection() {
            ui.label(RichText::new(error.to_string()).small().color(helto::ERROR));
        }
    }
    pub(super) fn model_settings(&mut self, ui: &mut Ui) {
        ui.label(helto::section("Models"));
        let Some(library) = self.models.catalog.clone() else {
            ui.label("Reconnect to load models.");
            return;
        };
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(!self.models.loading, |ui| {
                    if helto::icon_button(ui, Icon::Retry, "Refresh models", false).clicked() {
                        self.load_library(ui.ctx(), None);
                    }
                    if helto::icon_button(ui, Icon::Folder, "Choose models folder", false).clicked()
                    {
                        self.choose_model_folder(ui.ctx());
                    }
                });
                ui.add(egui::Label::new(library.settings.root.display().to_string()).truncate())
                    .on_hover_text(library.settings.root.display().to_string());
            });
        });
        if self.models.loading {
            ui.spinner();
        }
        if let Some(error) = &self.models.error {
            ui.label(RichText::new(error).small().color(helto::ERROR));
        }
        if let Some(model) = self.selected_model().cloned() {
            let id = model["id"].as_str().unwrap().to_owned();
            if let Some(profiles) = model["execution_profiles"]
                .as_array()
                .filter(|p| p.len() > 1)
            {
                let mut profile = model["execution_profile"]
                    .as_str()
                    .unwrap_or("standard")
                    .to_owned();
                let before = profile.clone();
                ui.add_enabled_ui(!self.models.loading, |ui| {
                    egui::ComboBox::from_id_salt("checkpoint-execution-profile")
                        .selected_text(&profile)
                        .show_ui(ui, |ui| {
                            for value in profiles.iter().filter_map(Value::as_str) {
                                ui.selectable_value(&mut profile, value.to_owned(), value);
                            }
                        });
                });
                if profile != before {
                    let mut settings = library.settings.clone();
                    settings.execution_profiles.insert(id.clone(), profile);
                    self.load_library(ui.ctx(), Some(settings));
                }
            }
            let mut components = library
                .settings
                .components
                .get(&id)
                .cloned()
                .or_else(|| serde_json::from_value::<Components>(model["components"].clone()).ok())
                .unwrap_or_default();
            let before = components.clone();
            ui.add_enabled_ui(!self.models.loading, |ui| {
                for (kind, label, selected) in [
                    (
                        "text_encoders",
                        "Text encoder",
                        &mut components.text_encoder,
                    ),
                    ("vae", "VAE", &mut components.vae),
                ] {
                    ui.push_id((&id, kind), |ui| {
                        component_picker(ui, label, &library.files, kind, selected);
                    });
                }
            });
            if components != before {
                let mut settings = library.settings.clone();
                settings.components.insert(id, components);
                self.load_library(ui.ctx(), Some(settings));
            }
        }
        if !library.warnings.is_empty() {
            egui::CollapsingHeader::new(format!("Scan warnings ({})", library.warnings.len()))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("model-scan-warnings")
                        .max_height(140.)
                        .show(ui, |ui| {
                            for warning in &library.warnings {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(warning).small().color(helto::ERROR),
                                    )
                                    .truncate(),
                                )
                                .on_hover_text(warning);
                            }
                        });
                });
        }
        let unsupported: Vec<_> = library
            .files
            .iter()
            .filter(|f| {
                !matches!(f.kind.as_str(), "loras" | "text_encoders" | "vae") && !f.supported
            })
            .collect();
        if !unsupported.is_empty() {
            egui::CollapsingHeader::new(format!("Unsupported files ({})", unsupported.len())).show(
                ui,
                |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("unsupported-model-files")
                        .max_height(140.)
                        .show(ui, |ui| {
                            for file in unsupported {
                                ui.add(egui::Label::new(&file.path).truncate())
                                    .on_hover_text(&file.path);
                            }
                        });
                },
            );
        }
    }
    pub(super) fn loras_window(&mut self, ctx: &egui::Context) {
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("imagine-loras"),
            egui::ViewportBuilder::default()
                .with_title("Imagine · LoRAs")
                .with_inner_size([520., 420.])
                .with_min_inner_size([420., 280.])
                .with_app_id("imagine-loras")
                .with_window_type(egui::X11WindowType::Dialog),
            |ui, _| {
                if ui.input(|i| i.viewport().close_requested() || i.key_pressed(egui::Key::Escape))
                {
                    self.models.open = false;
                }
                self.loras_content(ui);
            },
        );
        if !self.models.open && self.models.dirty {
            self.save_preferences();
            self.models.dirty = false;
        }
    }
    pub(super) fn loras_content(&mut self, ui: &mut Ui) {
        egui::Frame::new()
            .inner_margin(12)
            .show(ui, |ui| self.loras_inner(ui));
    }
    fn loras_inner(&mut self, ui: &mut Ui) {
        let Some(library) = self.models.catalog.clone() else {
            return;
        };
        let mut group = self.group_settings();
        let before = group.clone();
        ui.horizontal(|ui| {
            ui.heading(format!(
                "LoRAs · {}",
                model_group_name(&self.prefs.model_group)
            ));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if helto::icon_button(ui, Icon::Close, "Close LoRAs", false).clicked() {
                    self.models.open = false;
                }
            });
        });
        // LoRA file names follow Hide previews, like the video LoRA catalog.
        let hidden = !helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
        let editable = self.editable() && !self.models.loading;
        helto::private_scope(ui, hidden, |ui| {
            ui.add_enabled_ui(editable, |ui| {
                let search = ui.label("Add LoRA");
                let search_changed = ui
                    .add(
                        egui::TextEdit::singleline(&mut self.models.search)
                            .hint_text("Search…")
                            .desired_width(f32::INFINITY),
                    )
                    .labelled_by(search.id)
                    .changed();
                let search = self.models.search.to_lowercase();
                let terms: Vec<_> = search.split_whitespace().collect();
                let mut results = egui::ScrollArea::vertical()
                    .id_salt("lora-search")
                    .max_height(110.);
                if search_changed {
                    results = results.vertical_scroll_offset(0.);
                }
                results.show(ui, |ui| {
                    for file in library.files.iter().filter(|f| {
                        if f.kind != "loras" {
                            return false;
                        }
                        let path = f.path.to_lowercase();
                        terms.iter().all(|term| path.contains(term))
                    }) {
                        if group.loras.iter().any(|l| l.id == file.id) {
                            continue;
                        }
                        ui.horizontal(|ui| {
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if helto::icon_button(
                                        ui,
                                        Icon::Plus,
                                        &format!("Add {}", file.path),
                                        false,
                                    )
                                    .clicked()
                                    {
                                        group.loras.push(LoraSetting {
                                            id: file.id.clone(),
                                            name: file.path.clone(),
                                            strength: 1.,
                                            enabled: true,
                                        });
                                    }
                                    ui.with_layout(
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            ui.add(egui::Label::new(&file.path).truncate())
                                                .on_hover_text(&file.path);
                                        },
                                    );
                                },
                            );
                        });
                    }
                });
                ui.separator();
                let mut remove = None;
                let mut shift = None;
                egui::ScrollArea::vertical()
                    .id_salt("selected-loras")
                    .show(ui, |ui| {
                        let count = group.loras.len();
                        for (index, lora) in group.loras.iter_mut().enumerate() {
                            ui.push_id(index, |ui| {
                                let status = (!library
                                    .files
                                    .iter()
                                    .any(|f| f.id == lora.id && f.kind == "loras"))
                                .then_some("Missing");
                                ui.horizontal(|ui| {
                                    let enabled = ui.checkbox(&mut lora.enabled, "");
                                    let label = ui
                                        .add(egui::Label::new(&lora.name).truncate())
                                        .on_hover_text(&lora.name);
                                    enabled.labelled_by(label.id).on_hover_text("Enable LoRA");
                                    if let Some(status) = status {
                                        ui.label(RichText::new(status).color(helto::ERROR));
                                    }
                                    if helto::icon_button(
                                        ui,
                                        Icon::Trash,
                                        &format!("Remove {}", lora.name),
                                        false,
                                    )
                                    .clicked()
                                    {
                                        remove = Some(index);
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::Slider::new(&mut lora.strength, -2.0..=2.0)
                                            .clamping(egui::SliderClamping::Never)
                                            .text("Strength"),
                                    );
                                    if ui
                                        .add_enabled_ui(index > 0, |ui| {
                                            helto::icon_button(ui, Icon::Up, "Move LoRA up", false)
                                        })
                                        .inner
                                        .clicked()
                                    {
                                        shift = Some((index, index - 1));
                                    }
                                    if ui
                                        .add_enabled_ui(index + 1 < count, |ui| {
                                            helto::icon_button(
                                                ui,
                                                Icon::Down,
                                                "Move LoRA down",
                                                false,
                                            )
                                        })
                                        .inner
                                        .clicked()
                                    {
                                        shift = Some((index, index + 1));
                                    }
                                });
                            });
                        }
                    });
                if let Some(index) = remove {
                    group.loras.remove(index);
                } else if let Some((a, b)) = shift {
                    group.loras.swap(a, b);
                }
            })
        });
        if group != before {
            self.prefs
                .model_groups
                .insert(self.prefs.model_group.clone(), group);
            self.models.dirty = true;
        }
        if self.models.dirty && !ui.input(|i| i.pointer.any_down()) {
            self.save_preferences();
            self.models.dirty = false;
        }
        if let Some(error) = self.models.error.as_ref().or(self.settings_error.as_ref()) {
            ui.label(helto::error_text(error, !hidden));
        }
    }
}

/// The display name of a model group.
pub(super) fn model_group_name(group: &str) -> &str {
    match group {
        crate::model_catalog::QWEN_GROUP => "Qwen Image 2.1",
        crate::model_catalog::MOSS_GROUP => "MOSS Image",
        other => other,
    }
}
