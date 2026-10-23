//! Locked Timeline composition using existing native Helto surfaces.
use super::*;
use crate::app::workspace as chrome;
use crate::icons::Icon;
use egui::{Rect, RichText};
impl ImagineApp {
    pub(in crate::app) fn timeline_workspace(&mut self, ui: &mut egui::Ui, bounds: Rect) {
        self.timeline_catalog_install(ui.ctx());
        self.timeline_session(ui.ctx());
        self.timeline_poll_runs(ui.ctx());
        if self.timeline.zoom == 0. {
            self.timeline.zoom = 68.;
        }
        let stacked_toolbar = bounds.width() < 1000.;
        let toolbar = Rect::from_min_size(
            bounds.min,
            egui::vec2(bounds.width(), if stacked_toolbar { 78. } else { 38. }),
        );
        let timeline_h = if bounds.width() < 850. {
            (bounds.height() * 0.35).clamp(224., 280.)
        } else {
            (bounds.height() * 0.29).clamp(160., 220.)
        };
        let tracks = Rect::from_min_max(
            egui::pos2(bounds.left(), bounds.bottom() - timeline_h),
            bounds.max,
        );
        let main = Rect::from_min_max(
            egui::pos2(
                bounds.left(),
                toolbar.bottom()
                    + 6.
                    + if self.timeline.needs_reopen {
                        62.
                    } else if self.timeline.error.is_some() {
                        24.
                    } else {
                        0.
                    },
            ),
            egui::pos2(bounds.right(), tracks.top() - 6.),
        );
        let enabled = self.timeline.session.is_some()
            && !self.timeline.needs_reopen
            && !self.timeline.busy
            && !self.timeline.admitting;
        let document = self.timeline.document.clone();
        let mut commands = vec![];
        let mut action = None;
        chrome::panel(ui, toolbar, false);
        chrome::region(
            ui,
            "video-project-toolbar",
            toolbar.shrink2(egui::vec2(8., 3.)),
            |ui| {
                if stacked_toolbar {
                    ui.horizontal(|ui| {
                        self.timeline_toolbar_left(
                            ui,
                            &document,
                            enabled,
                            &mut commands,
                            &mut action,
                        )
                    });
                    ui.horizontal(|ui| {
                        self.timeline_toolbar_right(
                            ui,
                            &document,
                            enabled,
                            &mut commands,
                            &mut action,
                        )
                    });
                } else {
                    ui.horizontal(|ui| {
                        self.timeline_toolbar_left(
                            ui,
                            &document,
                            enabled,
                            &mut commands,
                            &mut action,
                        );
                        self.timeline_toolbar_right(
                            ui,
                            &document,
                            enabled,
                            &mut commands,
                            &mut action,
                        );
                    });
                }
            },
        );
        if let Some(document) = &document {
            let left_w = if self.timeline.shelf_collapsed {
                0.
            } else {
                (main.width() * 0.20).clamp(176., 230.)
            };
            let right_w = if self.timeline.inspector_collapsed {
                0.
            } else {
                (main.width() * 0.25).clamp(220., 310.)
            };
            let left =
                Rect::from_min_max(main.min, egui::pos2(main.left() + left_w, main.bottom()));
            let right =
                Rect::from_min_max(egui::pos2(main.right() - right_w, main.top()), main.max);
            let center = Rect::from_min_max(
                egui::pos2(left.right() + 6., main.top()),
                egui::pos2(right.left() - 6., main.bottom()),
            );
            if left_w > 0. {
                self.timeline_references(ui, left, document, enabled, &mut commands);
            }
            self.timeline_viewer(ui, center, document);
            if right_w > 0. {
                self.timeline_inspector(ui, right, document, enabled, &mut commands);
            }
            self.timeline_tracks(ui, tracks, &document.project, enabled, &mut commands);
        } else {
            chrome::panel(ui, main, false);
            chrome::region(ui, "timeline-empty", main.shrink(24.), |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(36.);
                    ui.heading("Video Timeline");
                    ui.add_space(12.);
                    let blocked = self.timeline_blocked(false);
                    ui.horizontal(|ui| {
                        ui.add_space((ui.available_width() - 70.).max(0.) / 2.);
                        let r = helto::action(ui, Icon::Plus, "New project", true, blocked);
                        if r.clicked() {
                            action = Some(PostAction::New);
                        }
                        let r = helto::action(ui, Icon::Folder, "Open project…", false, blocked);
                        if r.clicked() {
                            action = Some(PostAction::Open);
                        }
                    });
                });
            });
            chrome::panel(ui, tracks, false);
        }
        if self.timeline.needs_reopen || self.timeline.error.is_some() {
            let error = self.timeline.error.clone();
            let recovering = self.timeline.needs_reopen;
            let strip = Rect::from_min_size(
                egui::pos2(bounds.left() + 10., toolbar.bottom() + 4.),
                egui::vec2(bounds.width() - 20., if recovering { 58. } else { 22. }),
            );
            let reveal = !self.prefs.privacy_mode
                || (ui.input(|i| i.focused) && ui.rect_contains_pointer(strip));
            chrome::region(ui, "timeline-error", strip, |ui| {
                if recovering {
                    ui.horizontal_wrapped(|ui|{
                        ui.label("Server session changed. Project and text drafts kept.");
                        let (icon, label) = if self.timeline.session.is_some() { (Icon::Retry, "Reopen project") } else { (Icon::Link, "Reconnect") };
                        let r = helto::action(ui, icon, label, false, self.timeline.busy.then_some("Wait for the current timeline operation.")).on_hover_text("Reconnect the saved project without resuming old jobs. Uncommitted text drafts are kept.");
                        #[cfg(feature="ui-preview")] super::qa::control(ui,"project-reopen".into(),&r);
                        if r.clicked() { self.timeline_reopen(ui.ctx()); }
                    });
                }
                if let Some(error) = &error {
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
                }
            });
        }
        if enabled {
            use crate::app::Command;
            self.command_shortcuts(ui.ctx(), &[Command::TimelineSave]);
            // Delete and Undo belong to the Timeline, never to a floating dialog or
            // text field. The active Timeline remains the shortcut scope after a drag
            // or deletion releases its item. An actually focused field/button still
            // owns its keys, just as Director's interactive targets do.
            let timeline_focused = ui.ctx().memory(|m| {
                m.is_above_modal_layer(ui.layer_id())
                    && m.focused()
                        .is_none_or(|id| self.timeline.keyboard_focus == Some(id))
            });
            if timeline_focused {
                // Redo first: its Ctrl+Shift+Z must not be read as Undo.
                self.command_shortcuts(
                    ui.ctx(),
                    &[
                        Command::TimelineDelete,
                        Command::TimelineRedo,
                        Command::TimelineUndo,
                    ],
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
        self.timeline_project_prompt_window(ui.ctx());
        self.timeline_project_timing_window(ui.ctx());
        self.timeline_generation_window(ui.ctx());
        self.timeline_loras_window(ui.ctx());
        self.timeline_compatibility_window(ui.ctx());
        self.timeline_catalog_window(ui.ctx());
        self.timeline_authoring_window(ui.ctx());
        self.timeline_cast_window(ui.ctx());
        self.timeline_media_picker_window(ui.ctx());
        self.timeline_compiled_window(ui.ctx());
        self.timeline_library_window(ui.ctx());
        self.timeline_reference_cache_window(ui.ctx());
        self.timeline_media_health_window(ui.ctx());
        self.timeline_assembly_window(ui.ctx());
        self.timeline_replace_window(ui.ctx());
    }
    /// Why Timeline actions are unavailable; `needs_project` also requires an open project.
    pub(super) fn timeline_blocked(&self, needs_project: bool) -> Option<&'static str> {
        if self.timeline.session.is_none() {
            Some("Not connected to the video service.")
        } else if self.timeline.needs_reopen {
            Some("Reopen the project first.")
        } else if self.timeline.busy || self.timeline.admitting {
            Some("Wait for the current timeline operation.")
        } else if needs_project && self.timeline.document.is_none() {
            Some("Create or open a project first.")
        } else {
            None
        }
    }
    pub(in crate::app) fn timeline_submission_pending(&self) -> bool {
        self.timeline.pending.is_some()
    }
    pub(in crate::app) fn timeline_command_blocked(
        &self,
        command: crate::app::Command,
    ) -> Option<&'static str> {
        use crate::app::Command;
        self.timeline_blocked(true).or(match command {
            Command::TimelineUndo => self
                .timeline
                .history
                .undo
                .is_empty()
                .then_some("Nothing to undo."),
            Command::TimelineRedo => self
                .timeline
                .history
                .redo
                .is_empty()
                .then_some("Nothing to redo."),
            Command::TimelineDelete => self
                .timeline
                .document
                .as_ref()
                .is_none_or(|d| self.timeline.selection_items(&d.project).is_empty())
                .then_some("Select Sections or audio clips first."),
            _ => None,
        })
    }
    /// Timeline commands apply pending text drafts first, like toolbar edits.
    pub(in crate::app) fn timeline_run_command(
        &mut self,
        ctx: &egui::Context,
        command: crate::app::Command,
    ) {
        use crate::app::Command;
        let Some(d) = self.timeline.document.clone() else {
            return;
        };
        match command {
            Command::TimelineUndo => {
                let edits = self.timeline.history.undo_commands(&d.project);
                self.timeline_edit(ctx, edits, -1);
            }
            Command::TimelineRedo => {
                let edits = self.timeline.history.redo_commands(&d.project);
                self.timeline_edit(ctx, edits, 1);
            }
            Command::TimelineDelete => {
                let mut edits = self.timeline.draft_edits();
                edits.extend(self.timeline.delete_selection());
                self.timeline_edit(ctx, edits, 0);
            }
            Command::TimelineSave => self.timeline_after_drafts(ctx, PostAction::Save),
            Command::TimelineGenerate => self.timeline_after_drafts(ctx, PostAction::Generate),
            _ => {}
        }
    }
    /// Runs `action` once pending text drafts are committed.
    fn timeline_after_drafts(&mut self, ctx: &egui::Context, action: PostAction) {
        let edits = self.timeline.draft_edits();
        if edits.is_empty() {
            self.timeline_action(ctx, action);
        } else {
            self.timeline.after_edit = Some(action);
            self.timeline_edit(ctx, edits, 0);
        }
    }
    pub(in crate::app) fn timeline_generate_blocked(&self) -> Option<String> {
        let reason = self.timeline_blocked(true).or(if !self.video.checked {
            Some("Checking H3 availability.")
        } else if !self.video.available {
            return Some(match &self.video.error {
                Some(error) => format!("H3 is unavailable: {error}"),
                None => "H3 is unavailable. Check the H3 deployment and GPU coordinator, then reconnect.".into(),
            });
        } else if self.timeline.scene().is_none() {
            Some("Select a scene first.")
        } else if self.timeline.scene_active() {
            Some("This scene is already generating.")
        } else {
            None
        });
        reason.map(str::to_owned)
    }
    fn timeline_toolbar_left(
        &mut self,
        ui: &mut egui::Ui,
        document: &Option<Snapshot>,
        enabled: bool,
        commands: &mut Vec<Edit>,
        action: &mut Option<PostAction>,
    ) {
        {
            let _menu = helto::action(
                ui,
                Icon::Menu,
                "Project",
                false,
                self.timeline_blocked(false),
            );
            egui::Popup::menu(&_menu).show(|ui| {
                let health = ui.add_enabled(
                    document.is_some(),
                    egui::Button::new("Check Media / Relink…"),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-media-health".into(), &health);
                if health.clicked() {
                    self.timeline_media_health_open(ui.ctx());
                    ui.close();
                }
                let cache = ui.button("Reference session cache…");
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-reference-cache".into(), &cache);
                if cache.clicked() {
                    self.timeline_reference_cache_open(ui.ctx());
                    ui.close();
                }
                if let Some(d) = document {
                    let r = ui.button("Timeline settings…");
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "project-timing-open".into(), &r);
                    if r.clicked() {
                        self.timeline.project_timing.open(&d.project);
                        ui.close();
                    }
                    ui.separator();
                }
                let r = ui.add_enabled(
                    document.is_some(),
                    egui::Button::new("Generation settings…"),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-generation".into(), &r);
                if r.clicked() {
                    self.timeline_open_generation();
                    ui.close();
                }
                let r = ui.button("Prompt tools…");
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-prompt-tools".into(), &r);
                if r.clicked() {
                    self.timeline.catalog.open = true;
                    ui.close();
                }

                let r = ui.add_enabled(document.is_some(), egui::Button::new("Global prompt…"));
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-global-prompt".into(), &r);
                if r.clicked() {
                    self.timeline.project_prompt_open = true;
                    if self.timeline.global_prompt_options.is_none() {
                        self.timeline.global_prompt_options =
                            Some(imagine_video_contract::creative::GlobalPromptOptions {
                                enabled: !self.timeline.draft_global.is_empty(),
                                position: "Prefix".into(),
                            });
                    }
                    ui.close();
                }
                let response =
                    ui.add_enabled(document.is_some(), egui::Button::new("Project audio…"));
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
                let r = ui.button("New project");
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "project-new".into(), &r);
                if r.clicked() {
                    *action = Some(PostAction::New);
                    ui.close();
                }
                if ui.button("Open project…").clicked() {
                    *action = Some(PostAction::Open);
                    ui.close();
                }
                let r = ui.button("Project & Cast Library…");
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "library-open-window".into(), &r);
                if r.clicked() {
                    *action = Some(PostAction::Library);
                    ui.close();
                }
                ui.add_enabled_ui(document.is_some(), |ui| {
                    if ui.button("Save").clicked() {
                        *action = Some(PostAction::Save);
                        ui.close();
                    }
                    let save_as = ui.button("Save as…").on_hover_text(
                        if self.timeline.private_storage {
                            "Creates a separate private project copy. The original is unchanged."
                        } else {
                            "Choose a folder for a separate project copy."
                        },
                    );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "project-save-as".into(), &save_as);
                    if save_as.clicked() {
                        *action = Some(PostAction::SaveAs);
                        ui.close();
                    }
                    let r = ui.button("Export for older Imagine…").on_hover_text(
                        "Creates a separate compatible copy; the complete original remains active.",
                    );
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, "project-export-compatible".into(), &r);
                    if r.clicked() {
                        *action = Some(PostAction::ExportCompatible);
                        ui.close();
                    }
                    if ui.button("Refresh").clicked() {
                        self.timeline_refresh(ui.ctx());
                        ui.close();
                    }
                });
            });
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "project-menu".into(), &_menu);
        }
        use snapshot::SaveState;
        let saved = document.as_ref().and_then(|d| d.save_state());
        ui.label(
            RichText::new(match saved {
                None => "No project",
                Some(SaveState::SessionOnly) => "Unsaved project",
                Some(SaveState::Saved) => "Saved",
                Some(SaveState::NotSaved) => "Not saved",
                Some(SaveState::Saving) => "Saving…",
            })
            .small()
            .color(if saved == Some(&SaveState::NotSaved) {
                helto::ERROR
            } else {
                helto::DIM
            }),
        );
        ui.separator();
        ui.add_enabled_ui(enabled && document.is_some(), |ui| {
            let add_scene = helto::icon_button(ui, crate::icons::Icon::Scene, "Add Scene", false);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-add-scene".into(), &add_scene);
            if add_scene.clicked() {
                self.timeline_add_scene(ui.ctx());
            }
            let audio_button =
                helto::icon_button(ui, crate::icons::Icon::Audio, "Add Audio Track", false);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "add-audio".into(), &audio_button);
            if audio_button.clicked()
                && let Some(d) = &document
            {
                commands.extend(self.timeline.draft_edits());
                commands.push(self.timeline.add_audio(&d.project, None));
            }
            let attach = helto::icon_button(ui, crate::icons::Icon::Import, "Attach Audio", false);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "attach-audio".into(), &attach);
            if attach.clicked() {
                self.timeline_pick_media(media_picker::Target::TimelineAudio);
            }
            let add = helto::icon_button(ui, crate::icons::Icon::Plus, "Add Section", false);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-add-section".into(), &add);
            if add.clicked()
                && let Some(d) = &document
            {
                match group_commands::prepared(&d.project, self.timeline.draft_edits()) {
                    Ok(current) => {
                        let result = group_commands::add(
                            &current,
                            self.timeline.primary_scene(&current).as_ref(),
                            self.timeline.snap,
                        );
                        if !result.edits.is_empty() {
                            commands.extend(editing::difference(&d.project, &current));
                            commands.extend(result.edits);
                            self.timeline.after_edit =
                                Some(PostAction::SelectGroup(result.sections, result.audio));
                        } else {
                            self.timeline.error =
                                Some("No free space for a Section on this timeline.".into());
                        }
                    }
                    Err(e) => {
                        self.timeline.error = Some(format!("Could not add Section: {}", e.code))
                    }
                }
            }
            self.command_button(ui, crate::app::Command::TimelineUndo);
            self.command_button(ui, crate::app::Command::TimelineRedo);
            self.command_button(ui, crate::app::Command::TimelineSave);
        });
    }

    fn timeline_toolbar_right(
        &mut self,
        ui: &mut egui::Ui,
        document: &Option<Snapshot>,
        enabled: bool,
        _commands: &mut Vec<Edit>,
        action: &mut Option<PostAction>,
    ) {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let run_button = ui
                .add_enabled_ui(
                    enabled && document.is_some() && self.timeline.pending.is_none(),
                    |ui| {
                        helto::icon_button(
                            ui,
                            crate::icons::Icon::Down,
                            "Generation Run options",
                            false,
                        )
                    },
                )
                .inner;
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "run-menu".into(), &run_button);
            egui::Popup::menu(&run_button).show(|ui| {
                for (mode, name, label) in [
                    (
                        imagine_video_contract::runs::Mode::MissingOnly,
                        "run-missing",
                        "Generate missing scenes",
                    ),
                    (
                        imagine_video_contract::runs::Mode::AllSameSeeds,
                        "run-same",
                        "Regenerate all · same seeds",
                    ),
                    (
                        imagine_video_contract::runs::Mode::AllFreshSeeds,
                        "run-fresh",
                        "Regenerate all · fresh seeds",
                    ),
                    (
                        imagine_video_contract::runs::Mode::SelectedWithDependents,
                        "run-dependents",
                        "Selected + dependent scenes",
                    ),
                ] {
                    let ready = !self.timeline.has_active_run()
                        && (mode != imagine_video_contract::runs::Mode::SelectedWithDependents
                            || self.timeline.scene().is_some());
                    let response = ui.add_enabled(ready, egui::Button::new(label));
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, name.into(), &response);
                    let _ = name;
                    if response.clicked() {
                        *action = Some(PostAction::Run(mode));
                        ui.close();
                    }
                }
            });
            let response = self.command_button(ui, crate::app::Command::TimelineGenerate);
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "timeline-generate".into(), &response);
            let _ = response;
            let r = helto::action(
                ui,
                Icon::Review,
                "Review",
                false,
                self.timeline_blocked(true).or(self
                    .timeline
                    .scene()
                    .is_none()
                    .then_some("Select a scene first.")),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "compiled-open".into(), &r);
            if r.clicked() {
                *action = Some(PostAction::Compiled);
            }
            let response = helto::action(
                ui,
                Icon::Montage,
                "Final video…",
                false,
                document
                    .is_none()
                    .then_some("Create or open a project first."),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "assembly-open".into(), &response);
            if response.clicked() {
                *action = Some(PostAction::Assembly);
            }
            let menu = helto::action(ui, Icon::Story, "Story", false, self.timeline_blocked(true));
            egui::Popup::menu(&menu).show(|ui| {
                for (operation, name, label) in [
                    (
                        imagine_video_contract::authoring::Operation::Outline,
                        "story-open",
                        "Story outline…",
                    ),
                    (
                        imagine_video_contract::authoring::Operation::EnhanceScene,
                        "enhance-scene",
                        "Enhance scene…",
                    ),
                    (
                        imagine_video_contract::authoring::Operation::EnhanceSection,
                        "enhance-section",
                        "Enhance section…",
                    ),
                ] {
                    let ready = operation == imagine_video_contract::authoring::Operation::Outline
                        || self.timeline.scene().is_some()
                            && (operation
                                != imagine_video_contract::authoring::Operation::EnhanceSection
                                || self.timeline.section.is_some());
                    let r = ui.add_enabled(ready, egui::Button::new(label));
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, name.into(), &r);
                    let _ = name;
                    if r.clicked() {
                        *action = Some(PostAction::Authoring(operation));
                        ui.close();
                    }
                }
            });
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "story-menu".into(), &menu);
            let _ = menu;
            if self.timeline.busy || self.timeline.admitting {
                ui.spinner();
            }
            if self.timeline.session.is_none()
                && !self.timeline.busy
                && helto::icon_button(ui, Icon::Link, "Reconnect", false).clicked()
            {
                self.timeline.session_checked = false;
                self.timeline_session(ui.ctx());
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cleared_selection_stays_cleared_and_opening_selects_the_first_scene() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.client = None;
        let p = super::super::media_picker::tests::project();
        let raw = json!({"project":p,"persistence":{"state":"saved"}});
        let refresh = |app: &mut ImagineApp, replace: bool| {
            let epoch = app.timeline.epoch;
            app.timeline_message(
                &ctx,
                Event::Document {
                    epoch,
                    replace,
                    before: None,
                    history: 0,
                    result: Ok(raw.clone()),
                },
            );
        };
        refresh(&mut app, true);
        assert!(app.timeline.selected.contains(&p.scenes[0].id));
        app.timeline.selected.clear();
        refresh(&mut app, false);
        assert!(app.timeline.selected.is_empty());
    }

    #[test]
    fn header_reports_an_unreachable_video_service() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.workspace = Workspace::Video;
        let labels = |app: &mut ImagineApp| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
                    )),
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            output.textures_delta.clear();
            output
                .platform_output
                .accesskit_update
                .unwrap()
                .nodes
                .into_iter()
                .filter_map(|(_, n)| n.label().or(n.value()).map(str::to_owned))
                .collect::<Vec<_>>()
        };
        app.timeline.session = Some("own-synthetic-session".into());
        app.timeline.session_checked = true;
        let first = labels(&mut app);
        assert!(first.iter().any(|l| l == "Connected"), "{first:?}");
        app.timeline.session = None;
        assert!(
            labels(&mut app)
                .iter()
                .any(|l| l == "Video service unavailable")
        );
    }

    #[test]
    fn replacing_an_unsaved_project_waits_for_confirmation() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let project = super::super::media_picker::tests::project();
        assert!(!app.timeline_hold_replace(Replace::New));
        for (state, held) in [
            ("saved", false),
            ("session_only", true),
            ("not_saved", true),
        ] {
            app.timeline.replace_pending = None;
            app.timeline.document = Some(Snapshot::with(
                project.clone(),
                json!({"persistence":{"state":state}}),
            ));
            app.workspace = Workspace::Image;
            assert_eq!(app.timeline_hold_replace(Replace::Animate), held, "{state}");
            assert_eq!(app.timeline.replace_pending.is_some(), held);
            assert_eq!(app.workspace == Workspace::Video, held);
        }
    }

    /// Synthetic timeline frames; written only when IMAGINE_UI_PREVIEW_DIR is set.
    #[test]
    fn headless_timeline_with_project() {
        for (width, height, name) in [(1280, 800, "timeline"), (760, 800, "timeline-narrow")] {
            let root = tempfile::tempdir().unwrap();
            let ctx = egui::Context::default();
            helto::install(&ctx);
            let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
            let project = super::super::media_picker::tests::project();
            app.workspace = Workspace::Video;
            app.timeline.session = Some("own-synthetic-session".into());
            app.timeline.session_checked = true;
            app.timeline.document = Some(Snapshot::with(
                project.clone(),
                json!({"persistence":{"state":"saved"}}),
            ));
            app.timeline.selected.insert(project.scenes[0].id.clone());
            app.video.available = true;
            app.video.checked = true;
            let mut output = egui::FullOutput::default();
            for _ in 0..3 {
                output.append(ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width as f32, height as f32),
                        )),
                        focused: true,
                        ..Default::default()
                    },
                    |ui| app.content(ui),
                ));
            }
            assert!(!output.shapes.is_empty());
            if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
                let dir = std::path::PathBuf::from(dir);
                std::fs::create_dir_all(&dir).unwrap();
                crate::app::tests::render(&ctx, output, width, height)
                    .save(dir.join(format!("{name}.png")))
                    .unwrap();
            } else {
                output.textures_delta.clear();
            }
        }
    }
    #[test]
    fn toolbar_actions_do_not_overlap_at_small_and_normal_widths() {
        for width in [600., 760., 1000., 1280.] {
            let root = tempfile::tempdir().unwrap();
            let ctx = egui::Context::default();
            ctx.enable_accesskit();
            let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
            let project = super::super::media_picker::tests::project();
            app.workspace = Workspace::Video;
            app.timeline.session = Some("own-synthetic-session".into());
            app.timeline.session_checked = true;
            app.timeline.document = Some(Snapshot::with(
                project.clone(),
                json!({"persistence":{"state":"saved"}}),
            ));
            app.timeline.selected.insert(project.scenes[0].id.clone());
            app.video.available = true;
            app.video.checked = true;
            let mut nodes = vec![];
            for _ in 0..3 {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(width, 800.),
                        )),
                        ..Default::default()
                    },
                    |ui| app.content(ui),
                );
                output.textures_delta.clear();
                nodes = output.platform_output.accesskit_update.unwrap().nodes;
            }
            let names = [
                "Project",
                "Add Scene",
                "Add Audio Track",
                "Attach Audio",
                "Add Section",
                "Undo",
                "Redo",
                "Save project",
                "Generate",
                "Review",
                "Final video…",
                "Generation Run options",
                "Story",
            ];
            let bounds = names
                .iter()
                .map(|name| {
                    let node = nodes
                        .iter()
                        .find(|(_, n)| n.label() == Some(*name) && n.bounds().is_some())
                        .unwrap_or_else(|| panic!("missing {name} at {width}"));
                    (*name, node.1.bounds().unwrap())
                })
                .collect::<Vec<_>>();
            for (i, (name, a)) in bounds.iter().enumerate() {
                assert!(
                    a.x0 >= 0. && a.x1 <= width as f64,
                    "{name} outside {width}: {a:?}"
                );
                for (other, b) in &bounds[i + 1..] {
                    assert!(
                        !(a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0),
                        "{name}/{other} overlap at {width}"
                    );
                }
            }
        }
    }
}
