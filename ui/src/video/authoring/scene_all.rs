//! Director's Enhance mode for one Scene: every Scene field and Section
//! prompt proposed at once, each row with its original beside the editable
//! proposal, an Include choice, Keep original and Retry, applied together.
use super::*;
use crate::app::workspace as chrome;
use crate::icons::Icon;
use egui::Rect;

/// The row's name: the Scene field, or "Section n".
fn row_name(target: &FieldTarget, scene: Option<&Scene>) -> String {
    match target {
        FieldTarget::Scene { field } => field.label().into(),
        FieldTarget::Section { section_id } => scene
            .and_then(|s| s.sections.iter().position(|v| &v.id == section_id))
            .map_or_else(|| "Section".into(), |i| format!("Section {}", i + 1)),
    }
}

impl ImagineApp {
    pub(in crate::app::timeline) fn timeline_scene_enhance_view(
        &mut self,
        ui: &mut egui::Ui,
        rect: Rect,
    ) {
        let (mut state, document) = self.authoring_begin(ui.ctx());
        let mut actions = Actions::default();
        if state.failure.is_some() {
            state.auto_generate = false;
        }
        if state.auto_generate
            && !state.busy
            && !state.profiles.is_null()
            && !state.providers.is_null()
            && let Some(d) = &document
        {
            state.auto_generate = false;
            actions.call = Some(generate_call(&state, d));
        }
        let scene_id = state
            .draft
            .as_ref()
            .and_then(|d| d.request.scene_id.clone())
            .or_else(|| state.target_scene.clone());
        let scene = document
            .as_ref()
            .and_then(|d| d.project.scene(scene_id.as_ref()?).ok().cloned());
        let number = document.as_ref().and_then(|d| {
            d.project
                .scenes
                .iter()
                .position(|s| Some(&s.id) == scene_id.as_ref())
        });
        let header = Rect::from_min_size(rect.min, egui::vec2(rect.width(), 40.));
        chrome::panel(ui, header, false);
        chrome::region(
            ui,
            "scene-enhance-header",
            header.shrink2(egui::vec2(10., 4.)),
            |ui| {
                ui.horizontal(|ui| {
                    let r = helto::icon_button(ui, Icon::Back, "Back to Timeline", false);
                    control(ui, "scene-enhance-close", &r);
                    actions.close = r.clicked();
                    ui.label(helto::section(&match number {
                        Some(n) => format!("Enhance Scene {}", n + 1),
                        None => "Enhance Scene".into(),
                    }));
                    self.authoring_header(ui, &mut state, &mut actions);
                });
            },
        );
        let body = Rect::from_min_max(egui::pos2(rect.left(), header.bottom() + 6.), rect.max);
        chrome::panel(ui, body, false);
        let privacy = self.prefs.privacy_mode;
        chrome::region(ui, "scene-enhance-body", body.shrink(10.), |ui| {
            let hidden = !helto::revealed(ui, body, privacy);
            helto::private_scope(ui, hidden, |ui| {
                self.authoring_failure(ui, &mut state, &mut actions);
                if state.draft.is_some() {
                    self.authoring_draft_status(ui, &mut state, &document, &mut actions);
                } else {
                    authoring_generate(ui, &mut state, &document, &mut actions);
                }
                ui.separator();
                let busy = state.busy || state.draft.as_ref().is_some_and(|d| d.phase.busy());
                let received = state.received_rows.clone();
                let Some(draft) = state.draft.as_mut() else {
                    return;
                };
                let footer = 44.;
                egui::ScrollArea::vertical()
                    .id_salt("scene-enhance-rows")
                    .max_height((ui.available_height() - footer).max(80.))
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let half = (ui.available_width() - 12.) / 2.;
                        for index in 0..draft.fields.len() {
                            let status = row_status(draft, &received, index);
                            let name = row_name(&draft.fields[index].target, scene.as_ref());
                            ui.push_id(("scene-enhance-row", index), |ui| {
                                ui.group(|ui| {
                                    ui.set_width(ui.available_width());
                                    let row = &mut draft.fields[index];
                                    ui.horizontal(|ui| {
                                        let r = ui.checkbox(&mut row.include, "");
                                        control(ui, &format!("scene-enhance-include-{index}"), &r);
                                        let r = r.on_hover_text("Include when applying");
                                        let _ = r;
                                        ui.strong(&name);
                                        status_pill(ui, status);
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                let r = helto::action(
                                                    ui,
                                                    Icon::Retry,
                                                    &format!("Retry {name}"),
                                                    false,
                                                    busy.then_some(
                                                        "Wait for the current proposal.",
                                                    ),
                                                );
                                                control(
                                                    ui,
                                                    &format!("scene-enhance-retry-{index}"),
                                                    &r,
                                                );
                                                if r.clicked() {
                                                    actions.retry_row = Some(index);
                                                }
                                                let r = helto::action(
                                                    ui,
                                                    Icon::Undo,
                                                    &format!("Keep original {name}"),
                                                    false,
                                                    (busy && row.text.is_none())
                                                        .then_some("Wait for the proposal."),
                                                );
                                                control(
                                                    ui,
                                                    &format!("scene-enhance-keep-{index}"),
                                                    &r,
                                                );
                                                if r.clicked() {
                                                    row.text = Some(row.original.clone());
                                                }
                                            },
                                        );
                                    });
                                    if let Some(failure) = &row.failure {
                                        ui.colored_label(
                                            helto::ERROR,
                                            client::error_message(failure),
                                        );
                                    }
                                    ui.horizontal_top(|ui| {
                                        ui.vertical(|ui| {
                                            ui.set_width(half);
                                            ui.weak("Original");
                                            let mut original = row.original.as_str();
                                            ui.add(
                                                egui::TextEdit::multiline(&mut original)
                                                    .desired_rows(3)
                                                    .desired_width(half),
                                            );
                                        });
                                        ui.vertical(|ui| {
                                            ui.set_width(half);
                                            ui.label(
                                                egui::RichText::new("Proposal")
                                                    .color(helto::ACCENT),
                                            );
                                            let mut text = row.text.clone().unwrap_or_default();
                                            let r = ui.add_enabled(
                                                row.text.is_some(),
                                                egui::TextEdit::multiline(&mut text)
                                                    .desired_rows(3)
                                                    .desired_width(half),
                                            );
                                            control(
                                                ui,
                                                &format!("scene-enhance-proposal-{index}"),
                                                &r,
                                            );
                                            if r.changed() {
                                                row.text = Some(text);
                                            }
                                        });
                                    });
                                });
                            });
                        }
                    });
                let included = draft
                    .fields
                    .iter()
                    .filter(|f| f.include && f.text.as_ref().is_some_and(|t| !t.trim().is_empty()))
                    .count();
                let stale = authoring_stale(draft, &document);
                ui.horizontal(|ui| {
                    let r = helto::action(
                        ui,
                        Icon::Check,
                        "Apply enhancements",
                        true,
                        if busy {
                            Some("Wait for the current proposal.")
                        } else if stale {
                            Some("The project changed. Create a new proposal.")
                        } else if included == 0 {
                            Some("Include at least one proposal.")
                        } else {
                            None
                        },
                    );
                    control(ui, "scene-enhance-apply", &r);
                    if r.clicked() {
                        actions.call = Some(apply_call(draft));
                    }
                    ui.weak(format!("{included} of {} included", draft.fields.len()));
                    let r = helto::icon_button(ui, Icon::Close, "Discard proposal", false);
                    control(ui, "scene-enhance-discard", &r);
                    if r.clicked() {
                        actions.new_proposal = true;
                        actions.close = true;
                    }
                });
            });
        });
        if let Some(index) = actions.retry_row.take()
            && let Some(draft) = &state.draft
        {
            actions.call = Some(regenerate_call(draft, index));
        }
        self.authoring_finish(ui.ctx(), state, actions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rows_show_original_beside_proposal_keep_original_and_apply_included() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let project = super::super::super::media_picker::tests::project();
        let scene = project.scenes[0].clone();
        let mut fields = FieldProposal::rows(&project, &scene);
        for (i, row) in fields.iter_mut().enumerate() {
            row.text = Some(format!("Own proposal {i}"));
        }
        let draft: Draft = serde_json::from_value(json!({
            "schema_version":1,"id":"own-draft","project_id":project.id,
            "request":{"schema_version":1,"expected_project_revision":project.project_revision,
                "provider_profile_id":"proxy","provider":"koboldcpp","prompt_profile_id":"normal",
                "operation":"enhance_scene_all","scene_id":scene.id,"duration_seconds":2.0},
            "phase":"review","outline":null,"scenes":[],"text":null,"failure":null,"failed_index":null,
            "prompt_profile_id":"normal","prompt_profile_revision":"1","profile_warning":"",
            "provider_snapshot":{},"fields":fields
        }))
        .unwrap();
        let rows = draft.fields.len();
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.authoring.received_rows =
            draft.fields.iter().map(|f| f.text.clone()).collect();
        app.timeline.authoring.draft = Some(draft);
        app.timeline.authoring.open = true;
        app.timeline.document = Some(Snapshot::with(project, json!({})));
        assert!(app.timeline_scene_enhance_open() && !app.timeline_enhance_open());
        let frame = |app: &mut ImagineApp, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 900.),
                    )),
                    events,
                    focused: true,
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            output.textures_delta.clear();
            output.platform_output.accesskit_update.unwrap().nodes
        };
        let has = |nodes: &[(egui::accesskit::NodeId, egui::accesskit::Node)], text: &str| {
            nodes
                .iter()
                .any(|(_, n)| n.label().or(n.value()).is_some_and(|t| t == text))
        };
        let click = |app: &mut ImagineApp, label: &str| {
            let mut nodes = vec![];
            for _ in 0..2 {
                nodes = frame(app, vec![]);
            }
            let b = nodes
                .iter()
                .find(|(_, n)| n.label().is_some_and(|t| t == label))
                .and_then(|(_, n)| n.bounds())
                .unwrap_or_else(|| panic!("{label}"));
            let pos = egui::pos2(((b.x0 + b.x1) / 2.) as f32, ((b.y0 + b.y1) / 2.) as f32);
            for pressed in [true, false] {
                frame(
                    app,
                    vec![
                        egui::Event::PointerMoved(pos),
                        egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed,
                            modifiers: egui::Modifiers::NONE,
                        },
                    ],
                );
            }
        };
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let mut output = egui::FullOutput::default();
            for _ in 0..4 {
                output.append(ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1280., 900.),
                        )),
                        focused: true,
                        ..Default::default()
                    },
                    |ui| app.content(ui),
                ));
            }
            crate::app::tests::render(&ctx, output, 1280, 900)
                .save(std::path::PathBuf::from(dir).join("scene-enhance.png"))
                .unwrap();
        }
        let mut nodes = vec![];
        for _ in 0..3 {
            nodes = frame(&mut app, vec![]);
        }
        for text in [
            "Enhance Scene 1",
            "Original",
            "Proposal",
            "Description",
            "Own proposal 0",
            "Apply enhancements",
            &format!("{rows} of {rows} included"),
        ] {
            assert!(has(&nodes, text), "{text}");
        }
        click(&mut app, "Keep original Summary");
        let d = app.timeline.authoring.draft.as_ref().unwrap();
        assert_eq!(d.fields[1].text.as_ref(), Some(&d.fields[1].original));
        app.timeline.authoring.draft.as_mut().unwrap().fields[2].include = false;
        frame(&mut app, vec![]);
        let nodes = frame(&mut app, vec![]);
        // Excluded rows and empty proposals are not applied.
        let d = app.timeline.authoring.draft.as_ref().unwrap();
        let included = d
            .fields
            .iter()
            .filter(|f| f.include && f.text.as_ref().is_some_and(|t| !t.trim().is_empty()))
            .count();
        assert!(included < rows && !d.fields[2].include);
        assert!(has(&nodes, &format!("{included} of {rows} included")));
        click(&mut app, "Apply enhancements");
        assert!(app.timeline.authoring.busy, "the apply request was sent");
    }
}
