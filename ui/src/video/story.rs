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
                chrome::region(ui, id, rect.shrink(10.), |ui| {
                    let hidden = !helto::revealed(ui, rect, privacy);
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
            let draft = state.draft.as_mut().unwrap();
            for index in 0..scenes {
                let expanded = draft.scenes[index].is_some();
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
                    });
                    ui.label(source.summary.lines().next().unwrap_or(""));
                    ui.weak(format!(
                        "{}{}",
                        if source.boundary_from_previous == "continuous_shot" {
                            "Continuous shot"
                        } else {
                            "Hard cut"
                        },
                        if expanded { " · expanded" } else { "" }
                    ));
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
            let ready = draft.scenes.iter().filter(|s| s.is_some()).count();
            ui.weak(format!("{ready} / {scenes} scenes expanded"));
            let busy = state.busy || draft.phase.busy();
            ui.add_enabled_ui(!busy, |ui| {
                authoring_draft_actions(ui, draft, &document, &mut actions)
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
            ui.label(helto::section(&format!("Scene {}", index + 1)));
            ui.separator();
            ui.add_enabled_ui(!busy, |ui| authoring_scene_editor(ui, draft, index));
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
}
