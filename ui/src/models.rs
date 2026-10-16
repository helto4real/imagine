//! Model selection and a persistent, group-scoped LoRA desk.
use super::*;
use crate::{
    icons::Icon,
    model_catalog::{
        Components, LibrarySettings, LibrarySnapshot, QWEN_GROUP, SelectedLora, Selection,
    },
    settings::{LoraSetting, ModelGroupSettings},
};
use anyhow::{Result, bail};
use egui::{RichText, Ui};
use serde_json::json;

#[derive(Default)]
pub(super) struct ModelUi {
    pub catalog: Option<LibrarySnapshot>,
    pub loading: bool,
    pub error: Option<String>,
    pub open: bool,
    pub dirty: bool,
    search: String,
}
impl ImagineApp {
    fn group_settings(&self) -> ModelGroupSettings {
        self.prefs
            .model_groups
            .get(&self.prefs.model_group)
            .cloned()
            .unwrap_or_default()
    }
    fn selected_model(&self) -> Option<&Value> {
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
        self.prefs
            .model_groups
            .entry(self.prefs.model_group.clone())
            .or_default()
            .variant = id;
        self.metadata = self.selected_model().cloned();
        if let Some(meta) = &self.metadata {
            if let Some(steps) = meta["defaults"]["sampling"]["steps"].as_u64() {
                self.prefs.generation.steps = steps as u32;
            }
            if let Some(cfg) = meta["defaults"]["sampling"]["cfg_scale"].as_f64() {
                self.prefs.generation.cfg_scale = cfg as f32;
            }
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
                    });
                let count = self
                    .group_settings()
                    .loras
                    .iter()
                    .filter(|l| l.enabled)
                    .count();
                if helto::icon_button(ui, Icon::Lora, "LoRAs", self.models.open).clicked() {
                    self.models.open = !self.models.open;
                }
                if count > 0 {
                    ui.label(RichText::new(count.to_string()).small());
                }
            });
        });
        self.select_model(selected);
        if let Err(error) = self.model_selection() {
            ui.label(RichText::new(error.to_string()).small().color(helto::ERROR));
        }
    }
    pub(super) fn model_settings(&mut self, ui: &mut Ui) {
        ui.label(RichText::new("Models").strong());
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
                    ui.label(label);
                    let name = library
                        .files
                        .iter()
                        .find(|f| &f.id == selected)
                        .map(|f| f.path.as_str())
                        .unwrap_or("Select…");
                    egui::ComboBox::from_id_salt(kind)
                        .width(ui.available_width())
                        .truncate()
                        .selected_text(name)
                        .show_ui(ui, |ui| {
                            for file in library.files.iter().filter(|f| {
                                f.kind == kind
                                    && f.supported
                                    && f.group.as_deref() == Some(QWEN_GROUP)
                            }) {
                                ui.selectable_value(selected, file.id.clone(), &file.path);
                            }
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
            .filter(|f| f.kind != "loras" && !f.supported)
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
            ui.heading("LoRAs · Qwen Image 2.1");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if helto::icon_button(ui, Icon::Close, "Close LoRAs", false).clicked() {
                    self.models.open = false;
                }
            });
        });
        ui.add_enabled_ui(self.editable() && !self.models.loading, |ui| {
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
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
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
                        });
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
                                        helto::icon_button(ui, Icon::Down, "Move LoRA down", false)
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
            ui.label(RichText::new(error).color(helto::ERROR));
        }
    }
}
