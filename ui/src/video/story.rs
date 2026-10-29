//! The Story view: brief, outline and scene source side by side, as in the
//! prototype's Story screen. Each column follows Hide previews on its own.
use super::authoring::*;
use super::*;
use crate::app::workspace as chrome;
use egui::Rect;

impl ImagineApp {
    pub(super) fn timeline_story_view(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let (mut state, document) = self.authoring_begin(ui.ctx());
        let mut actions = Actions::default();
        let header = Rect::from_min_size(rect.min, egui::vec2(rect.width(), 40.));
        chrome::panel(ui, header, false);
        chrome::region(
            ui,
            "story-header",
            header.shrink2(egui::vec2(10., 4.)),
            |ui| {
                ui.horizontal(|ui| {
                    let r =
                        helto::icon_button(ui, crate::icons::Icon::Back, "Back to Timeline", false);
                    control(ui, "story-close", &r);
                    actions.close = r.clicked();
                    ui.label(helto::section("Story"));
                    stage_strip(ui, state.draft.as_ref());
                    ui.separator();
                    self.authoring_header(ui, &mut state, &mut actions);
                });
            },
        );
        let body = Rect::from_min_max(egui::pos2(rect.left(), header.bottom() + 6.), rect.max);
        let brief_w = (body.width() * 0.3).clamp(240., 380.);
        let editor_w = (body.width() * 0.34).clamp(260., 440.);
        let brief = Rect::from_min_size(body.min, egui::vec2(brief_w, body.height()));
        let editor = Rect::from_min_max(egui::pos2(body.right() - editor_w, body.top()), body.max);
        let outline = Rect::from_min_max(
            egui::pos2(brief.right() + 6., body.top()),
            egui::pos2(editor.left() - 6., body.bottom()),
        );
        let privacy = self.prefs.privacy_mode;
        let column =
            |ui: &mut egui::Ui, id: &str, rect: Rect, add: &mut dyn FnMut(&mut egui::Ui)| {
                chrome::panel(ui, rect, false);
                let _canvas = helto::panel_canvas(ui, id, rect);
                chrome::region(ui, id, rect.shrink(10.), |ui| {
                    let hidden = !helto::shown(ui, privacy);
                    egui::ScrollArea::vertical()
                        .id_salt((id, "scroll"))
                        .auto_shrink([false, false])
                        .show(ui, |ui| helto::private_scope(ui, hidden, |ui| add(ui)));
                });
            };
        column(ui, "story-brief", brief, &mut |ui| {
            ui.label(helto::section("Story brief"));
            ui.separator();
            if let Some(draft) = &state.draft {
                // The brief the proposal was made from; a new proposal edits a new one.
                ui.label(&draft.request.brief);
                ui.weak(format!(
                    "{:.0} s · {}",
                    draft.request.duration_seconds, draft.prompt_profile_id
                ));
                // The measured budget this outline was started with.
                if let Some(m) = state
                    .measurement
                    .as_ref()
                    .filter(|m| draft.request.budget_fingerprint.as_ref() == Some(&m.fingerprint))
                {
                    ui.weak(format!(
                        "Context: about {} of {} input tokens · {} of {} characters",
                        input_tokens(m),
                        available_tokens(m),
                        m.chars,
                        m.char_limit
                    ));
                }
            } else {
                self.authoring_sources(ui, &mut state, &mut actions);
                ImagineApp::authoring_story_inputs(ui, &mut state, &document, &mut actions);
                authoring_generate(ui, &mut state, &document, &mut actions);
            }
        });
        column(ui, "story-outline", outline, &mut |ui| {
            self.authoring_failure(ui, &mut state, &mut actions);
            let Some(scenes) = state
                .draft
                .as_ref()
                .and_then(|d| d.outline.as_ref())
                .map(|o| o.scenes.len())
            else {
                ui.label(helto::section("Outline"));
                ui.separator();
                if state.draft.is_some() {
                    self.authoring_draft_status(ui, &mut state, &document, &mut actions);
                }
                return;
            };
            let total: f64 = state.draft.as_ref().unwrap().planned_durations.iter().sum();
            ui.horizontal(|ui| {
                ui.label(helto::section("Outline"));
                ui.weak(format!("{scenes} scenes · {total:.0} s"));
            });
            ui.separator();
            self.authoring_draft_status(ui, &mut state, &document, &mut actions);
            let selected = state.selected.min(scenes.saturating_sub(1));
            let received = &state.received;
            let draft = state.draft.as_mut().unwrap();
            for index in 0..scenes {
                let status = scene_status(draft, received, index);
                let source = draft.scenes[index]
                    .as_ref()
                    .unwrap_or(&draft.outline.as_ref().unwrap().scenes[index]);
                let card = ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.strong(format!("{:02}", index + 1));
                        ui.strong(&source.title);
                        if let Some(duration) = draft.planned_durations.get(index) {
                            ui.weak(format!("{duration:.0} s"));
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            status_pill(ui, status);
                        });
                    });
                    ui.label(source.summary.lines().next().unwrap_or(""));
                    ui.weak(if source.boundary_from_previous == "continuous_shot" {
                        "Continuous shot"
                    } else {
                        "Hard cut"
                    });
                });
                let response = card.response.interact(egui::Sense::click());
                if index == selected {
                    ui.painter().rect_stroke(
                        response.rect,
                        4,
                        egui::Stroke::new(1.5, helto::ACCENT),
                        egui::StrokeKind::Inside,
                    );
                }
                control(ui, &format!("story-scene-{index}"), &response);
                if response.clicked() {
                    state.selected = index;
                }
            }
            if outline_editable(draft) {
                let r = ui
                    .add_enabled_ui(scenes < 128, |ui| {
                        helto::action(ui, crate::icons::Icon::Plus, "Add Scene", false, None)
                    })
                    .inner;
                control(ui, "story-scene-add", &r);
                if r.clicked() {
                    state.selected = edit_outline(draft, OutlineEdit::AddScene, selected);
                }
            } else if !draft.phase.busy()
                && draft.phase != imagine_video_contract::authoring::Phase::Applied
                && draft.scenes.iter().any(Option::is_some)
            {
                // Changing the outline discards the storyboard, as in Director.
                if state.confirm_edit_outline {
                    let confirm = helto::confirm(
                        ui,
                        "Edit the outline? The expanded Scenes are discarded.",
                        (crate::icons::Icon::Edit, "Discard storyboard"),
                        "Keep the storyboard",
                        None,
                    );
                    control(ui, "story-edit-outline-confirm", &confirm.yes);
                    if confirm.yes.clicked() {
                        discard_storyboard(draft);
                    }
                    if confirm.yes.clicked() || confirm.no.clicked() {
                        state.confirm_edit_outline = false;
                    }
                } else {
                    let r =
                        helto::action(ui, crate::icons::Icon::Edit, "Edit outline", false, None);
                    control(ui, "story-edit-outline", &r);
                    state.confirm_edit_outline = r.clicked();
                }
            }
            let ready = draft.scenes.iter().filter(|s| s.is_some()).count();
            ui.weak(format!("{ready} / {scenes} scenes expanded"));
            let busy = state.busy || draft.phase.busy();
            let confirm = &mut state.confirm_replace;
            ui.add_enabled_ui(!busy, |ui| {
                authoring_draft_actions(ui, draft, &document, &mut actions, confirm)
            });
        });
        column(ui, "story-scene", editor, &mut |ui| {
            let index = state.selected;
            let busy = state.busy || state.draft.as_ref().is_some_and(|d| d.phase.busy());
            let Some(draft) = state.draft.as_mut().filter(|d| d.outline.is_some()) else {
                ui.label(helto::section("Scene"));
                ui.separator();
                return;
            };
            let index = index.min(draft.scenes.len().saturating_sub(1));
            let status = scene_status(draft, &state.received, index);
            let count = draft.scenes.len();
            let mut edit = None;
            ui.horizontal(|ui| {
                ui.label(helto::section(&format!("Scene {}", index + 1)));
                status_pill(ui, status);
                if status.reviewable() {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let r = helto::action(
                            ui,
                            crate::icons::Icon::Retry,
                            "Regenerate Scene",
                            false,
                            busy.then_some("Wait for the current proposal."),
                        );
                        control(ui, "story-regenerate", &r);
                        if r.clicked() {
                            actions.call = Some(regenerate_call(draft, index));
                        }
                    });
                }
                if outline_editable(draft) {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Right to left: Remove is the outermost.
                        let buttons = [
                            (
                                crate::icons::Icon::Trash,
                                "Remove Scene",
                                count > 1,
                                OutlineEdit::RemoveScene(index),
                            ),
                            (
                                crate::icons::Icon::Down,
                                "Move Scene later",
                                index + 1 < count,
                                OutlineEdit::MoveScene(index, true),
                            ),
                            (
                                crate::icons::Icon::Up,
                                "Move Scene earlier",
                                index > 0,
                                OutlineEdit::MoveScene(index, false),
                            ),
                        ];
                        for (icon, name, enabled, action) in buttons {
                            let r = ui
                                .add_enabled_ui(enabled, |ui| {
                                    helto::icon_button(ui, icon, name, false)
                                })
                                .inner;
                            control(
                                ui,
                                &format!("story-{}", name.to_lowercase().replace(' ', "-")),
                                &r,
                            );
                            if r.clicked() {
                                edit = Some(action);
                            }
                        }
                    });
                }
            });
            if let Some(edit) = edit {
                state.selected = edit_outline(draft, edit, index);
            }
            let index = state.selected.min(draft.scenes.len().saturating_sub(1));
            ui.separator();
            // Finished Scenes stay editable while later ones expand, as in Director.
            ui.add_enabled_ui(!busy || status.reviewable(), |ui| {
                authoring_scene_editor(ui, draft, index)
            });
            if let Some(plan) = draft
                .scenes
                .get(index)
                .and_then(Option::as_ref)
                .and(draft.outline.as_ref())
                .and_then(|o| o.scenes.get(index))
            {
                ui.add_space(6.);
                let header = ui.collapsing("Outline plan", |ui| {
                    ui.strong(&plan.title);
                    ui.label(&plan.summary);
                    for (i, section) in plan.sections.iter().enumerate() {
                        ui.weak(format!("Section {}", i + 1));
                        ui.label(&section.prompt);
                    }
                });
                control(ui, "story-outline-plan", &header.header_response);
            }
        });
        self.authoring_finish(ui.ctx(), state, actions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn draft(project: &VideoProject) -> imagine_video_contract::authoring::Draft {
        let scene = |id: &str, title: &str| {
            json!({"id":id,"title":title,"summary":format!("Own {title} summary"),"description":"",
                "ambient_sound":"","non_diegetic_music":"","weight":1.0,
                "boundary_from_previous":"hard_cut","sections":[]})
        };
        serde_json::from_value(json!({
            "schema_version":1,"id":"own-draft","project_id":project.id,
            "request":{"schema_version":1,"expected_project_revision":project.project_revision,
                "provider_profile_id":"proxy","provider":"koboldcpp","prompt_profile_id":"normal",
                "operation":"outline","brief":"Own brief","duration_seconds":16.0},
            "phase":"review","outline":{"scenes":[scene("a","Own arrival"),scene("b","Own signal")]},
            "scenes":[null,null],"text":null,"failure":null,"failed_index":null,
            "prompt_profile_id":"normal","prompt_profile_revision":"1","profile_warning":"",
            "provider_snapshot":{},"planned_durations":[8.0,8.0]
        }))
        .unwrap()
    }
    #[test]
    fn an_outline_runs_its_inventory_then_starts_only_within_its_measured_budget() {
        use imagine_video_contract::authoring::*;
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let project = super::super::media_picker::tests::project();
        app.timeline.session = Some("own-session".into());
        let mut request = draft(&project).request;
        request.images = vec![StoryImage {
            pair_with: None,
            id: "own_image".into(),
            asset: AssetId("own_asset".into()),
            label: "Source image 1".into(),
            usage: ImageUse::SettingReference,
        }];
        // Story images are analysed first; known notes go to the measurement.
        let (_, _, path, body) = story_call(&project.id, &request);
        assert!(path.ends_with("/authoring"), "{path}");
        assert_eq!(body.as_ref().unwrap()["operation"], "inventory");
        let notes = vec![InventoryNote {
            id: "own_image".into(),
            note: "Own lighthouse note".into(),
        }];
        let mut known = request.clone();
        known.inventory = notes.clone();
        let (_, _, path, body) = story_call(&project.id, &known);
        assert!(path.ends_with("/authoring/measure"), "{path}");
        assert_eq!(
            body.as_ref().unwrap()["inventory"][0]["note"],
            "Own lighthouse note"
        );
        // A finished inventory Draft measures its outline with its notes.
        app.timeline.authoring.pending = Some((project.id.clone(), request.clone()));
        app.timeline.authoring.open = true;
        let mut inventory = serde_json::to_value(draft(&project)).unwrap();
        inventory["request"]["operation"] = json!("inventory");
        inventory["outline"] = Value::Null;
        inventory["scenes"] = json!([]);
        inventory["inventory"] = json!(notes);
        app.timeline_authoring_reply(&ctx, "own-session".into(), Reply::Draft, Ok(inventory));
        let state = &app.timeline.authoring;
        assert!(state.busy, "the measurement was sent");
        assert_eq!(state.pending.as_ref().unwrap().1.inventory, notes);
        assert!(
            app.timeline_story_open(),
            "the Story view stays open during an inventory"
        );
        // A measurement that does not fit stops with the numbers; one that fits starts.
        let measured = |context_tokens: u64| Measurement {
            schema_version: 1,
            operation: Operation::Outline,
            chars: 9000,
            char_limit: 100000,
            text_bytes: 9000,
            message_count: 2,
            context_tokens,
            max_output_tokens: 1000,
            context_reserve_tokens: 8,
            fingerprint: "sha256:own".into(),
        };
        app.timeline.authoring.busy = false;
        app.timeline_authoring_reply(
            &ctx,
            "own-session".into(),
            Reply::Measure,
            Ok(json!(measured(4000))),
        );
        let state = &app.timeline.authoring;
        let failure = state.failure.clone().unwrap();
        assert!(
            failure.contains("about 3515 input tokens") && failure.contains("leaves 2992"),
            "{failure}"
        );
        assert!(state.pending.is_none() && !state.busy);
        app.timeline.authoring.pending = Some((project.id.clone(), known));
        app.timeline_authoring_reply(
            &ctx,
            "own-session".into(),
            Reply::Measure,
            Ok(json!(measured(8192))),
        );
        let state = &app.timeline.authoring;
        assert!(state.failure.is_none() && state.pending.is_none());
        assert!(state.busy, "the outline start was sent");
        assert_eq!(state.measurement.as_ref().unwrap().context_tokens, 8192);
    }
    #[test]
    fn story_view_shows_brief_outline_and_selected_scene_and_follows_privacy() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let project = super::super::media_picker::tests::project();
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.authoring.draft = Some(draft(&project));
        app.timeline.authoring.open = true;
        app.timeline.document = Some(Snapshot::with(project, json!({})));
        let frame = |app: &mut ImagineApp, privacy: bool, events: Vec<egui::Event>| {
            app.prefs.privacy_mode = privacy;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
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
        let has = |nodes: &Vec<(egui::accesskit::NodeId, egui::accesskit::Node)>, text: &str| {
            nodes.iter().any(|(_, n)| {
                n.label()
                    .into_iter()
                    .chain(n.value())
                    .any(|t| t.contains(text))
            })
        };
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            app.prefs.privacy_mode = false;
            let mut output = egui::FullOutput::default();
            for _ in 0..2 {
                output.append(ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1280., 800.),
                        )),
                        focused: true,
                        ..Default::default()
                    },
                    |ui| app.content(ui),
                ));
            }
            crate::app::tests::render(&ctx, output, 1280, 800)
                .save(std::path::PathBuf::from(dir).join("story.png"))
                .unwrap();
        }
        let mut nodes = vec![];
        for _ in 0..3 {
            nodes = frame(&mut app, false, vec![]);
        }
        assert!(app.timeline_story_open());
        for text in [
            "Back to Timeline",
            "Own brief",
            "Own arrival",
            "Own signal",
            "Scene 1",
            "Apply reviewed proposal",
            "2 scenes · 16 s",
        ] {
            assert!(has(&nodes, text), "{text}");
        }
        let card = nodes
            .iter()
            .find(|(_, n)| n.label().or(n.value()).is_some_and(|t| t == "Own signal"))
            .and_then(|(_, n)| n.bounds())
            .unwrap();
        let pos = egui::pos2(card.x0 as f32 + 2., ((card.y0 + card.y1) / 2.) as f32);
        for pressed in [true, false] {
            frame(
                &mut app,
                false,
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
        assert_eq!(app.timeline.authoring.selected, 1);
        let hidden = frame(&mut app, true, vec![egui::Event::PointerGone]);
        assert!(!has(&hidden, "Own arrival") && !has(&hidden, "Own brief"));
        // Generate again keeps this draft in history; Discard also leaves the view.
        let click = |app: &mut ImagineApp, label: &str| {
            let nodes = frame(app, false, vec![]);
            let bounds = nodes
                .iter()
                .find(|(_, n)| n.label().is_some_and(|t| t == label))
                .and_then(|(_, n)| n.bounds())
                .unwrap_or_else(|| panic!("{label}"));
            let pos = egui::pos2(
                ((bounds.x0 + bounds.x1) / 2.) as f32,
                ((bounds.y0 + bounds.y1) / 2.) as f32,
            );
            for pressed in [true, false] {
                frame(
                    app,
                    false,
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
        click(&mut app, "Generate again");
        assert!(app.timeline.authoring.draft.is_none() && app.timeline.authoring.busy);
        assert_eq!(app.timeline.authoring.history.len(), 1);
        // The request reaches the unreachable fixture endpoint and fails.
        let start = std::time::Instant::now();
        while app.timeline.authoring.busy && start.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            frame(&mut app, false, vec![]);
        }
        assert!(app.timeline.authoring.failure.is_some());
        app.timeline.authoring.draft = app.timeline.authoring.history.pop();
        click(&mut app, "Discard proposal");
        assert!(app.timeline.authoring.draft.is_none());
        assert_eq!(app.timeline.authoring.history.len(), 1);
        assert!(!app.timeline_story_open());
    }

    #[test]
    fn the_outline_gains_moves_and_loses_scenes_and_an_expanded_one_asks_first() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let project = super::super::media_picker::tests::project();
        let mut d = draft(&project);
        d.phase = imagine_video_contract::authoring::Phase::Outline;
        for scene in &mut d.outline.as_mut().unwrap().scenes {
            scene
                .sections
                .push(imagine_video_contract::authoring::StorySection {
                    id: format!("{}-1", scene.id),
                    summary: String::new(),
                    prompt: "Own beat".into(),
                    weight: 1.,
                    image_ids: vec![],
                    first_frame_id: String::new(),
                    last_frame_id: String::new(),
                    dialogue_intent: None,
                });
        }
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.authoring.received = d.scenes.clone();
        app.timeline.authoring.draft = Some(d);
        app.timeline.authoring.open = true;
        app.timeline.document = Some(Snapshot::with(project, json!({})));
        let frame = |app: &mut ImagineApp, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
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
        // The Story was not asked for music, so its field is hidden.
        let nodes = frame(&mut app, vec![]);
        let music = |nodes: &[(egui::accesskit::NodeId, egui::accesskit::Node)]| {
            nodes
                .iter()
                .any(|(_, n)| n.label().or(n.value()) == Some("Audience-only music"))
        };
        assert!(!music(&nodes));
        app.timeline
            .authoring
            .draft
            .as_mut()
            .unwrap()
            .request
            .add_music = true;
        frame(&mut app, vec![]);
        let nodes = frame(&mut app, vec![]);
        assert!(music(&nodes));
        let titles = |app: &ImagineApp| {
            let d = app.timeline.authoring.draft.as_ref().unwrap();
            d.outline
                .as_ref()
                .unwrap()
                .scenes
                .iter()
                .map(|s| s.title.clone())
                .collect::<Vec<_>>()
        };
        click(&mut app, "Add Scene");
        assert_eq!(titles(&app), ["Own arrival", "Own signal", "Scene 3"]);
        assert_eq!(app.timeline.authoring.selected, 2);
        let d = app.timeline.authoring.draft.as_ref().unwrap();
        assert_eq!(d.planned_durations.len(), 3);
        // The new Scene's empty Section blocks Expand until it has a beat.
        assert!(outline_blocker(d).is_some());
        click(&mut app, "Move Scene earlier");
        assert_eq!(titles(&app), ["Own arrival", "Scene 3", "Own signal"]);
        click(&mut app, "Remove Scene");
        assert_eq!(titles(&app), ["Own arrival", "Own signal"]);
        // Once a Scene is expanded, the outline changes only after confirming.
        let d = app.timeline.authoring.draft.as_mut().unwrap();
        d.scenes[0] = Some(d.outline.as_ref().unwrap().scenes[0].clone());
        click(&mut app, "Edit outline");
        assert!(app.timeline.authoring.confirm_edit_outline);
        click(&mut app, "Keep the storyboard");
        assert!(app.timeline.authoring.draft.as_ref().unwrap().scenes[0].is_some());
        click(&mut app, "Edit outline");
        click(&mut app, "Discard storyboard");
        let d = app.timeline.authoring.draft.as_ref().unwrap();
        assert!(d.scenes.iter().all(Option::is_none));
        assert!(outline_editable(d));
    }

    #[test]
    fn a_replacing_storyboard_shows_scene_status_and_asks_before_apply() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let project = super::super::media_picker::tests::project();
        let mut d = draft(&project);
        d.request.destination = imagine_video_contract::authoring::Destination::Replace;
        d.phase = imagine_video_contract::authoring::Phase::Review;
        d.scenes = d
            .outline
            .as_ref()
            .unwrap()
            .scenes
            .iter()
            .cloned()
            .map(Some)
            .collect();
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.authoring.received = d.scenes.clone();
        app.timeline.authoring.draft = Some(d);
        app.timeline.authoring.open = true;
        app.timeline.document = Some(Snapshot::with(project, json!({})));
        let frame = |app: &mut ImagineApp, events: Vec<egui::Event>| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 800.),
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
        let mut nodes = vec![];
        for _ in 0..3 {
            nodes = frame(&mut app, vec![]);
        }
        for text in [
            "3 Storyboard",
            "1 Story brief ✓",
            "Ready",
            "Ready to apply",
            "Regenerate Scene",
        ] {
            assert!(has(&nodes, text), "{text}");
        }
        // An expanded Scene's cut stays editable in review.
        app.timeline.authoring.selected = 1;
        let mut nodes = vec![];
        for _ in 0..2 {
            nodes = frame(&mut app, vec![]);
        }
        let cut = nodes
            .iter()
            .find(|(_, n)| {
                n.value()
                    .or(n.label())
                    .is_some_and(|t| t == "Continuous shot" || t == "Hard cut")
            })
            .map(|(_, n)| n)
            .expect("cut choice");
        assert!(!cut.is_disabled());
        let d = app.timeline.authoring.draft.as_ref().unwrap();
        let (_, _, path, body) = super::super::authoring::regenerate_call(d, 1);
        assert!(path.ends_with("/review"));
        assert_eq!(body.unwrap()["then_regenerate"], 1);
        click(&mut app, "Apply reviewed proposal");
        assert!(app.timeline.authoring.confirm_replace);
        assert!(!app.timeline.authoring.busy, "nothing is applied yet");
        click(&mut app, "Keep the timeline");
        assert!(!app.timeline.authoring.confirm_replace);
        assert!(!app.timeline.authoring.busy);
        click(&mut app, "Apply reviewed proposal");
        click(&mut app, "Replace timeline");
        assert!(!app.timeline.authoring.confirm_replace);
        assert!(app.timeline.authoring.busy, "the apply request was sent");
    }
}
