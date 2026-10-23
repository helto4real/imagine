//! Locked Timeline composition using existing native Helto surfaces.
use super::editing::{seconds, time};
use super::*;
use crate::app::workspace as chrome;
use egui::{Rect, RichText};
impl ImagineApp {
    pub(in crate::app) fn timeline_workspace(&mut self, ui: &mut egui::Ui, bounds: Rect) {
        self.timeline_session(ui.ctx());
        self.timeline_poll_runs(ui.ctx());
        if self.timeline.zoom == 0. {
            self.timeline.zoom = 68.;
        }
        let toolbar = Rect::from_min_size(bounds.min, egui::vec2(bounds.width(), 38.));
        let timeline_h = (bounds.height() * 0.29).clamp(126., 180.);
        let tracks = Rect::from_min_max(
            egui::pos2(bounds.left(), bounds.bottom() - timeline_h),
            bounds.max,
        );
        let main = Rect::from_min_max(
            egui::pos2(
                bounds.left(),
                toolbar.bottom()
                    + 6.
                    + if self.timeline.error.is_some() {
                        24.
                    } else {
                        0.
                    },
            ),
            egui::pos2(bounds.right(), tracks.top() - 6.),
        );
        let enabled =
            self.timeline.session.is_some() && !self.timeline.busy && !self.timeline.admitting;
        let document = self.timeline.document.clone();
        let mut commands = vec![];
        let mut action = None;
        chrome::panel(ui, toolbar, false);
        chrome::region(
            ui,
            "video-project-toolbar",
            toolbar.shrink2(egui::vec2(8., 3.)),
            |ui| {
                ui.horizontal(|ui| {
                    ui.add_enabled_ui(enabled, |ui| {
                        let _menu = ui.menu_button("Project", |ui| {
                            let response = ui.add_enabled(
                                document.is_some(),
                                egui::Button::new("Project audio…"),
                            );
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, "project-audio".into(), &response);
                            if response.clicked()
                                && let Some(d) = &document
                            {
                                commands.extend(self.timeline.draft_edits());
                                self.timeline.clear_audio_selection();
                                self.timeline.project_audio = true;
                                self.timeline.audio_settings_draft = Some(d.project.audio.clone());
                                ui.close();
                            }
                            if ui.button("New project").clicked() {
                                action = Some(PostAction::New);
                                ui.close();
                            }
                            if ui.button("Open project…").clicked() {
                                action = Some(PostAction::Open);
                                ui.close();
                            }
                            let r=ui.button("Project & Cast Library…");
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui,"library-open-window".into(),&r);
                            if r.clicked(){action=Some(PostAction::Library);ui.close();}
                            if ui.button("Single clip").clicked() {
                                self.timeline.quick_clip = true;
                                ui.close();
                            }
                            ui.add_enabled_ui(document.is_some(), |ui| {
                                if ui.button("Save").clicked() {
                                    action = Some(PostAction::Save);
                                    ui.close();
                                }
                                if ui.button("Save as…").clicked() {
                                    action = Some(PostAction::SaveAs);
                                    ui.close();
                                }
                                if ui.button("Refresh").clicked() {
                                    self.timeline_refresh(ui.ctx());
                                    ui.close();
                                }
                            });
                        });
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, "project-menu".into(), &_menu.response);
                    });
                    let saved = document
                        .as_ref()
                        .and_then(|d| d.raw["persistence"]["state"].as_str())
                        .unwrap_or("No project");
                    ui.label(
                        RichText::new(match saved {
                            "session_only" => "Unsaved project",
                            "saved" => "Saved",
                            "not_saved" => "Not saved",
                            "saving" => "Saving…",
                            s => s,
                        })
                        .small()
                        .color(if saved == "not_saved" {
                            helto::ERROR
                        } else {
                            helto::DIM
                        }),
                    );
                    ui.separator();
                    ui.add_enabled_ui(enabled && document.is_some(), |ui| {
                        if helto::select_button(ui, "+ Scene", false).clicked() {
                            self.timeline_add_scene(ui.ctx());
                        }
                        let audio_button = helto::select_button(ui, "+ Audio", false);
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, "add-audio".into(), &audio_button);
                        if audio_button.clicked()
                            && let Some(d) = &document
                        {
                            commands.extend(self.timeline.draft_edits());
                            commands.push(self.timeline.add_audio(&d.project, None));
                        }
                        if helto::select_button(ui, "+ Section", false).clicked()
                            && let Some(scene) = self.timeline.scene()
                        {
                            let start = scene
                                .sections
                                .iter()
                                .map(|s| seconds(s.start) + seconds(s.duration))
                                .fold(0., f64::max);
                            let duration = (seconds(scene.duration) - start).min(1.);
                            if duration >= 0.25 {
                                commands.push(Edit::AddSection {
                                    scene_id: scene.id.clone(),
                                    section: Section {
                                        id: SectionId(format!(
                                            "section_{}",
                                            uuid::Uuid::new_v4().simple()
                                        )),
                                        start: time(start),
                                        duration: time(duration),
                                        prompt: String::new(),
                                    },
                                });
                            } else {
                                self.timeline.error = Some(
                                    "Extend the scene or shorten its last section first.".into(),
                                );
                            }
                        }
                        if ui
                            .add_enabled_ui(!self.timeline.history.undo.is_empty(), |ui| {
                                helto::icon_button(ui, crate::icons::Icon::Undo, "Undo", false)
                            })
                            .inner
                            .on_hover_text("Undo · Ctrl+Z")
                            .clicked()
                            && let Some(d) = &document
                        {
                            let edits = self.timeline.history.undo_commands(&d.project);
                            self.timeline_edit(ui.ctx(), edits, -1);
                        }
                        if ui
                            .add_enabled_ui(!self.timeline.history.redo.is_empty(), |ui| {
                                helto::icon_button(ui, crate::icons::Icon::Redo, "Redo", false)
                            })
                            .inner
                            .on_hover_text("Redo · Ctrl+Shift+Z")
                            .clicked()
                            && let Some(d) = &document
                        {
                            let edits = self.timeline.history.redo_commands(&d.project);
                            self.timeline_edit(ui.ctx(), edits, 1);
                        }
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let response = ui
                            .add_enabled_ui(
                                enabled
                                    && self.video.available
                                    && self.timeline.selected.len() == 1
                                    && self.timeline.audio_selected.is_empty()
                                    && self.timeline.scene().is_some()
                                    && !self.timeline.scene_active(),
                                |ui| {
                                    helto::select_button(
                                        ui,
                                        if self.timeline.pending.is_some() {
                                            "Retry submission"
                                        } else {
                                            "Generate"
                                        },
                                        true,
                                    )
                                },
                            )
                            .inner;
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, "timeline-generate".into(), &response);
                        if response.clicked() {
                            action = Some(PostAction::Generate);
                        }
                        let r=ui.add_enabled(enabled&&self.timeline.scene().is_some(),egui::Button::new("Review"));
                        #[cfg(feature="ui-preview")]super::qa::control(ui,"compiled-open".into(),&r);
                        if r.clicked(){action=Some(PostAction::Compiled);}
                        let response=ui.add_enabled(document.is_some(),egui::Button::new("Final video…"));
                        #[cfg(feature="ui-preview")]super::qa::control(ui,"assembly-open".into(),&response);
                        if response.clicked(){action=Some(PostAction::Assembly);}
                        let menu=ui.add_enabled_ui(enabled && document.is_some() && self.timeline.pending.is_none(),|ui|ui.menu_button("Runs",|ui| {
                            for (mode,name,label) in [
                                (imagine_video_contract::runs::Mode::MissingOnly,"run-missing","Generate missing scenes"),
                                (imagine_video_contract::runs::Mode::AllSameSeeds,"run-same","Regenerate all · same seeds"),
                                (imagine_video_contract::runs::Mode::AllFreshSeeds,"run-fresh","Regenerate all · fresh seeds"),
                                (imagine_video_contract::runs::Mode::SelectedWithDependents,"run-dependents","Selected + dependent scenes"),
                            ] {
                                let ready=!self.timeline.has_active_run() && (mode!=imagine_video_contract::runs::Mode::SelectedWithDependents || self.timeline.selected.len()==1);
                                let response=ui.add_enabled(ready,egui::Button::new(label));
                                #[cfg(feature="ui-preview")]
                                super::qa::control(ui,name.into(),&response);
                                let _=name;
                                if response.clicked(){action=Some(PostAction::Run(mode));ui.close();}
                            }
                        })).inner;
                        #[cfg(feature="ui-preview")]
                        super::qa::control(ui,"run-menu".into(),&menu.response);
                        let _=menu;
                        let menu=ui.add_enabled_ui(enabled&&document.is_some(),|ui|ui.menu_button("Story",|ui|{
                            for (operation,name,label) in [(imagine_video_contract::authoring::Operation::Outline,"story-open","Story outline…"),(imagine_video_contract::authoring::Operation::EnhanceScene,"enhance-scene","Enhance scene…"),(imagine_video_contract::authoring::Operation::EnhanceSection,"enhance-section","Enhance section…")] {
                                let ready=operation==imagine_video_contract::authoring::Operation::Outline || self.timeline.scene().is_some() && (operation!=imagine_video_contract::authoring::Operation::EnhanceSection||self.timeline.section.is_some());
                                let r=ui.add_enabled(ready,egui::Button::new(label));
                                #[cfg(feature="ui-preview")] super::qa::control(ui,name.into(),&r);
                                let _=name;
                                if r.clicked(){action=Some(PostAction::Authoring(operation));ui.close();}
                            }
                        })).inner;
                        #[cfg(feature="ui-preview")]super::qa::control(ui,"story-menu".into(),&menu.response);
                        let _=menu;
                        if self.timeline.busy || self.timeline.admitting {
                            ui.spinner();
                        }
                        if self.timeline.session.is_none()
                            && !self.timeline.busy
                            && ui.button("Reconnect").clicked()
                        {
                            self.timeline.session_checked = false;
                            self.timeline_session(ui.ctx());
                        }
                    });
                });
            },
        );
        if let Some(document) = &document {
            let left_w = (main.width() * 0.20).clamp(176., 230.);
            let right_w = (main.width() * 0.25).clamp(220., 310.);
            let left =
                Rect::from_min_max(main.min, egui::pos2(main.left() + left_w, main.bottom()));
            let right =
                Rect::from_min_max(egui::pos2(main.right() - right_w, main.top()), main.max);
            let center = Rect::from_min_max(
                egui::pos2(left.right() + 6., main.top()),
                egui::pos2(right.left() - 6., main.bottom()),
            );
            self.timeline_references(ui, left, document, enabled, &mut commands);
            self.timeline_viewer(ui, center, document);
            self.timeline_inspector(ui, right, document, enabled, &mut commands);
            self.timeline_tracks(ui, tracks, &document.project, enabled, &mut commands);
        } else {
            chrome::panel(ui, main, false);
            chrome::region(ui, "timeline-empty", main.shrink(24.), |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(36.);
                    ui.heading("Video Timeline");
                    ui.add_space(12.);
                    ui.add_enabled_ui(enabled, |ui| {
                        if helto::select_button(ui, "New project", true).clicked() {
                            action = Some(PostAction::New);
                        }
                        if ui.button("Open project…").clicked() {
                            action = Some(PostAction::Open);
                        }
                    });
                });
            });
            chrome::panel(ui, tracks, false);
        }
        if let Some(error) = &self.timeline.error {
            let strip = Rect::from_min_size(
                egui::pos2(bounds.left() + 10., toolbar.bottom() + 6.),
                egui::vec2(bounds.width() - 20., 22.),
            );
            let reveal = !self.prefs.privacy_mode
                || (ui.input(|i| i.focused) && ui.rect_contains_pointer(strip));
            chrome::region(ui, "timeline-error", strip, |ui| {
                ui.add(
                    egui::Label::new(
                        RichText::new(if reveal {
                            error.as_str()
                        } else {
                            "Hidden error"
                        })
                        .color(helto::ERROR),
                    )
                    .truncate(),
                );
            });
        }
        if enabled && !ui.ctx().text_edit_focused() {
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::S)) {
                action = Some(PostAction::Save);
            }
            let redo = egui::Modifiers {
                ctrl: true,
                command: true,
                shift: true,
                ..Default::default()
            };
            if ui.input_mut(|i| i.consume_key(redo, egui::Key::Z))
                && let Some(d) = &document
            {
                self.timeline_edit(ui.ctx(), self.timeline.history.redo_commands(&d.project), 1);
            }
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Z))
                && let Some(d) = &document
            {
                self.timeline_edit(
                    ui.ctx(),
                    self.timeline.history.undo_commands(&d.project),
                    -1,
                );
            }
        }
        if let Some(next) = action {
            commands.extend(self.timeline.draft_edits());
            if commands.is_empty() && !self.timeline.busy {
                self.timeline_action(ui.ctx(), next);
            } else {
                self.timeline.after_edit = Some(next);
            }
        }
        if !commands.is_empty() && !self.timeline.busy {
            self.timeline_edit(ui.ctx(), commands, 0);
        }
        self.timeline_authoring_window(ui.ctx());
        self.timeline_cast_window(ui.ctx());
        self.timeline_compiled_window(ui.ctx());
        self.timeline_library_window(ui.ctx());
        self.timeline_assembly_window(ui.ctx());
    }
}
