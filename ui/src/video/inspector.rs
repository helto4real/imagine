//! Contextual media/reference and scene/Section inspector.
use super::editing::{resize_scene, seconds, time};
use super::*;
use crate::app::workspace as chrome;
use egui::Rect;
impl ImagineApp {
    pub(super) fn timeline_references(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        chrome::region(ui, "timeline-references", rect.shrink(10.), |ui| {
            ui.horizontal(|ui| {
                ui.strong("References / Media");
            });
            ui.separator();
            let r = ui.add_enabled(enabled, egui::Button::new("Cast & references…"));
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "cast-open".into(), &r);
            if r.clicked() {
                self.timeline_open_cast();
            }
            if ui
                .add_enabled(enabled, egui::Button::new("Import media…"))
                .clicked()
            {
                self.timeline_choose_media(ui.ctx());
            }
            egui::ScrollArea::vertical()
                .id_salt("project-assets")
                .show(ui, |ui| {
                    let mut kind_counts = std::collections::BTreeMap::new();
                    for asset in d.project.assets.values() {
                        let count = kind_counts.entry(asset.kind.clone()).or_insert(0usize);
                        *count += 1;
                        let ordinal = *count;
                        ui.group(|ui| {
                            ui.label(format!(
                                "{} {}",
                                match asset.kind {
                                    MediaKind::Image => "Image",
                                    MediaKind::Video => "Video",
                                    MediaKind::Audio => "Audio",
                                },
                                ordinal
                            ));
                            if asset.kind == MediaKind::Audio {
                                let response =
                                    ui.add_enabled(enabled, egui::Button::new("Add to timeline"));
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(
                                    ui,
                                    format!("add-audio-asset-{ordinal}"),
                                    &response,
                                );
                                if response.clicked() {
                                    commands.extend(self.timeline.draft_edits());
                                    commands.push(
                                        self.timeline.add_audio(&d.project, Some(asset.id.clone())),
                                    );
                                }
                            }
                            if let Some(scene) = self.timeline.scene().cloned() {
                                if asset.kind == MediaKind::Video {
                                    let r = ui.add_enabled(
                                        enabled,
                                        egui::Button::new("Use as scene clip"),
                                    );
                                    #[cfg(feature = "ui-preview")]
                                    super::qa::control(
                                        ui,
                                        format!("assembly-use-video-{ordinal}"),
                                        &r,
                                    );
                                    if r.clicked() {
                                        commands.extend(self.timeline.draft_edits());
                                        let action = PostAction::UseClip {
                                            scene: scene.id.clone(),
                                            asset: asset.id.clone(),
                                        };
                                        if commands.is_empty() {
                                            self.timeline_action(ui.ctx(), action);
                                        } else {
                                            self.timeline.after_edit = Some(action);
                                        }
                                    }
                                }
                                if ui.button("Preview").clicked()
                                    && let Some(id) = d.raw["runtime_assets"][&asset.id.0].as_str()
                                {
                                    self.timeline.selected_take = None;
                                    self.timeline_load_media(ui.ctx(), id.to_owned());
                                }
                                self.timeline_media_binding(
                                    ui, d, &scene, asset, enabled, commands,
                                );
                            }
                        });
                    }
                });
        });
    }
    pub(super) fn timeline_inspector(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
        d: &Snapshot,
        enabled: bool,
        commands: &mut Vec<Edit>,
    ) {
        chrome::panel(ui, rect, false);
        chrome::region(ui, "timeline-inspector", rect.shrink(10.), |ui| {
            egui::ScrollArea::vertical()
                .id_salt("video-inspector-scroll")
                .show(ui, |ui| {
                    if self.timeline_audio_inspector(ui, d, enabled, commands) {
                        return;
                    }
                    let Some(scene) = self.timeline.scene().cloned() else {
                        ui.strong("Project");
                        return;
                    };
                    if self.timeline.draft_scene.as_ref() != Some(&scene.id) {
                        self.timeline.sync_draft();
                    }
                    if self.timeline.boundary {
                        self.timeline_assembly_boundary(ui, d, &scene, commands);
                        ui.strong("Boundary");
                        ui.add_enabled_ui(enabled, |ui| {
                            self.timeline_continuity(ui, d, &scene, commands)
                        });
                        return;
                    }
                    ui.strong(if self.timeline.section.is_some() {
                        "Section"
                    } else {
                        "Scene"
                    });
                    ui.separator();
                    ui.add_enabled_ui(enabled, |ui| {
                        ui.label("Duration");
                        let mut duration = seconds(scene.duration);
                        if ui
                            .add(
                                egui::DragValue::new(&mut duration)
                                    .range(0.25..=3600.)
                                    .speed(0.25)
                                    .suffix(" s"),
                            )
                            .changed()
                        {
                            commands.extend(resize_scene(
                                &d.project,
                                &scene,
                                duration,
                                self.timeline.mode,
                            ));
                        }
                        ui.label("Description");
                        self.timeline_assembly_clip(ui, d, &scene, commands);
                        let r = helto::textarea(
                            ui,
                            &mut self.timeline.draft_description,
                            3,
                            "Scene description",
                            self.prefs.privacy_mode,
                            true,
                        );
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, "scene-description".into(), &r);
                        if r.lost_focus() {
                            commands.extend(self.timeline.draft_edits());
                        }
                        let source_header = ui.collapsing("Scene prompt fields", |ui| {
                            let p = &mut self.timeline.draft_scene_prompt;
                            let mut save = false;
                            for (name, label, value) in [
                                ("title", "Title", &mut p.title),
                                ("summary", "Summary", &mut p.summary),
                                ("ambient", "Ambient sound", &mut p.ambient_sound),
                                ("music", "Audience-only music", &mut p.non_diegetic_music),
                            ] {
                                ui.label(label);
                                let r = helto::textarea(
                                    ui,
                                    value,
                                    2,
                                    label,
                                    self.prefs.privacy_mode,
                                    true,
                                );
                                #[cfg(feature = "ui-preview")]
                                super::qa::control(ui, format!("scene-source-{name}"), &r);
                                let _ = name;
                                save |= r.lost_focus();
                            }
                            if save {
                                commands.extend(self.timeline.draft_edits());
                            }
                        });
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(
                            ui,
                            "scene-source-fields".into(),
                            &source_header.header_response,
                        );
                        let _ = source_header;
                        if let Some(section) = self
                            .timeline
                            .section
                            .as_ref()
                            .and_then(|id| scene.sections.iter().find(|s| &s.id == id))
                        {
                            ui.label("Section prompt");
                            let r = helto::textarea(
                                ui,
                                &mut self.timeline.draft_section,
                                3,
                                "Section prompt",
                                self.prefs.privacy_mode,
                                true,
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, "section-prompt".into(), &r);
                            if r.lost_focus() {
                                commands.extend(self.timeline.draft_edits());
                            }
                            ui.weak(format!(
                                "{:.3}–{:.3} s",
                                seconds(section.start),
                                seconds(section.start) + seconds(section.duration)
                            ));
                        }
                        ui.label("Global prompt");
                        let r = helto::textarea(
                            ui,
                            &mut self.timeline.draft_global,
                            2,
                            "Project prompt",
                            self.prefs.privacy_mode,
                            true,
                        );
                        if r.lost_focus() {
                            commands.extend(self.timeline.draft_edits());
                        }
                        ui.separator();
                        ui.label("Model");
                        ui.weak(format!(
                            "{} · {}",
                            d.project.effective_model(&scene).model,
                            d.project.effective_model(&scene).variant
                        ));
                        let model = d.project.effective_model(&scene);
                        if model.adapter == "h3"
                            && model.variant == "reference"
                            && let Some(slot) =
                                scene.settings.iter().find(|slot| &slot.model == model)
                        {
                            let mut use_audio =
                                slot.values["audio_reference_mode"].as_str() != Some("Disabled");
                            let response =
                                ui.checkbox(&mut use_audio, "Use timeline audio references");
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(
                                ui,
                                "h3-timeline-audio-references".into(),
                                &response,
                            );
                            if response.changed() {
                                let mut settings = slot.clone();
                                settings.values["audio_reference_mode"] =
                                    serde_json::json!(if use_audio {
                                        "Timeline Audio References"
                                    } else {
                                        "Disabled"
                                    });
                                commands.push(Edit::SetModelSettings {
                                    scene_id: scene.id.clone(),
                                    settings,
                                });
                            }
                        }
                        if d.project
                            .scenes
                            .first()
                            .is_some_and(|first| first.id != scene.id)
                            && ui.button("Boundary settings").clicked()
                        {
                            commands.extend(self.timeline.draft_edits());
                            self.timeline.boundary = true;
                            self.timeline.section = None;
                        }
                        if ui.button("Duplicate scene").clicked() {
                            commands.push(Edit::DuplicateScene {
                                scene_id: scene.id.clone(),
                                new_scene_id: SceneId(format!(
                                    "scene_{}",
                                    uuid::Uuid::new_v4().simple()
                                )),
                                start: time(seconds(scene.start) + seconds(scene.duration)),
                            });
                        }
                        if ui.button("Remove scene").clicked() {
                            commands.push(Edit::RemoveScene {
                                scene_id: scene.id.clone(),
                            });
                        }
                    });
                });
        });
    }
}
