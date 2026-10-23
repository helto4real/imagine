//! Scene field and Section prompt enhancement, shown under the field it
//! improves: it starts at once, then offers the proposal with Use, Generate
//! again and Discard. The field above stays the original for comparison.
use super::*;
use crate::icons::Icon;

/// Whether the open enhancement, or the one about to start, is for this field.
fn shows(state: &State, scene: &SceneId, section: Option<&SectionId>, field: SceneField) -> bool {
    let (operation, target, target_section, target_field) = state.draft.as_ref().map_or(
        (
            state.operation,
            state.target_scene.as_ref(),
            state.target_section.as_ref(),
            state.scene_field,
        ),
        |d| {
            (
                d.request.operation,
                d.request.scene_id.as_ref(),
                d.request.section_id.as_ref(),
                d.request.scene_field,
            )
        },
    );
    target == Some(scene)
        && match section {
            Some(id) => operation == Operation::EnhanceSection && target_section == Some(id),
            None => {
                operation == Operation::EnhanceScene && target_field.unwrap_or_default() == field
            }
        }
}

impl ImagineApp {
    /// The enhancement card under a Scene field, or under a Section prompt
    /// when `section` is given; nothing when the enhancement is elsewhere.
    /// `original` is the field's text box, kept in view above the card.
    pub(in crate::app::timeline) fn timeline_inline_enhance(
        &mut self,
        ui: &mut egui::Ui,
        scene: &SceneId,
        section: Option<&SectionId>,
        field: SceneField,
        original: egui::Rect,
    ) {
        let state = &self.timeline.authoring;
        if !state.open || !shows(state, scene, section, field) {
            return;
        }
        let (mut state, document) = self.authoring_begin(ui.ctx());
        state.shown_in_pass = Some(ui.ctx().cumulative_pass_nr());
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
        let privacy = self.prefs.privacy_mode;
        let frame = egui::Frame::new()
            .fill(helto::BG)
            .stroke(egui::Stroke::new(1., helto::ACCENT_BORDER))
            .corner_radius(5.)
            .inner_margin(6.)
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                let reveal = helto::reveal_in_viewport(ui, privacy);
                helto::private_scope(ui, !reveal, |ui| {
                    self.authoring_inline_body(ui, &mut state, &document, &mut actions)
                });
            });
        // The inspector's columns are short: bring a new card or proposal into
        // view with the field and its label above it. A first frame has no
        // content size yet, so ask until it is in view.
        let label = ui.spacing().interact_size.y + ui.spacing().item_spacing.y;
        let wanted = frame
            .response
            .rect
            .union(original.with_min_y(original.top() - label));
        let shown = state.draft.as_ref().map(|d| d.id.clone());
        if state.scrolled_to.as_ref() != Some(&shown) {
            let view = ui.clip_rect();
            // A view too short for both keeps the card itself in view.
            let wanted = if wanted.height() > view.height() {
                frame.response.rect
            } else {
                wanted
            };
            if view.contains_rect(wanted) {
                state.scrolled_to = Some(shown);
            } else {
                ui.scroll_to_rect(wanted, Some(egui::Align::TOP));
            }
        }
        ui.add_space(4.);
        self.authoring_finish(ui.ctx(), state, actions);
    }

    fn authoring_inline_body(
        &mut self,
        ui: &mut egui::Ui,
        state: &mut State,
        document: &Option<Snapshot>,
        actions: &mut Actions,
    ) {
        let busy = state.busy || state.draft.as_ref().is_some_and(|d| d.phase.busy());
        let starting = state.auto_generate && state.draft.is_none();
        // One status line, then the proposal and one row of actions, so the
        // card fits the docked inspector beside the field it improves.
        ui.horizontal(|ui| {
            let (glyph, _) = ui.allocate_exact_size(egui::vec2(14., 14.), egui::Sense::hover());
            crate::icons::paint(ui.painter(), glyph, Icon::Enhance, helto::ACCENT);
            ui.label(
                egui::RichText::new(if busy || starting {
                    "Enhancing…"
                } else if state.draft.is_some() {
                    "Proposal"
                } else {
                    "Enhance"
                })
                .small()
                .strong()
                .color(helto::ACCENT_TEXT),
            );
            if let Some(d) = &state.draft
                && let Some(model) = d.provider_snapshot["model"].as_str()
            {
                ui.label(egui::RichText::new(model).small().color(helto::DIM));
            }
            if busy || starting {
                ui.spinner();
                if let Some(d) = &state.draft
                    && state.progress["draft_id"] == d.id
                    && let Some(elapsed) = state.progress["elapsed_seconds"].as_f64()
                {
                    ui.label(
                        egui::RichText::new(format!("{elapsed:.1} s"))
                            .small()
                            .color(helto::DIM),
                    );
                }
            }
        });
        self.authoring_failure(ui, state, actions);
        if let Some(draft) = &state.draft {
            if !draft.profile_warning.is_empty() {
                ui.label(&draft.profile_warning);
            }
            for warning in &draft.warnings {
                ui.colored_label(helto::WARN, warning);
            }
            if let Some(failure) = &draft.failure {
                ui.colored_label(helto::ERROR, failure);
            }
            if authoring_stale(draft, document) && draft.phase != Phase::Applied {
                ui.colored_label(
                    helto::ERROR,
                    "The project changed since this proposal. Generate again to use it.",
                );
            }
        }
        let privacy = self.prefs.privacy_mode;
        if let Some(draft) = state.draft.as_mut()
            && let Some(text) = &mut draft.text
        {
            let rows = text.lines().count().clamp(2, 8);
            let r = helto::textarea(ui, text, rows, "story-proposal-text", privacy, true);
            control(ui, "story-proposal-text", &r);
        }
        ui.horizontal(|ui| {
            let wait = busy.then_some("Wait for the current proposal.");
            if let Some(d) = state.draft.as_ref().filter(|d| d.phase.busy()) {
                let r = helto::icon_button(ui, Icon::Stop, "Cancel", false);
                control(ui, "story-cancel", &r);
                if r.clicked() {
                    actions.call = Some((
                        Reply::Draft,
                        true,
                        format!("/v1/video/authoring/{}/cancel", d.id),
                        Some(json!({})),
                    ));
                }
            } else if let Some(draft) = &state.draft {
                let r = helto::action(
                    ui,
                    Icon::Check,
                    "Use proposal",
                    true,
                    wait.or(if authoring_stale(draft, document) {
                        Some("The project changed. Generate again first.")
                    } else {
                        (draft.phase != Phase::Review || draft.text.is_none())
                            .then_some("Wait for a proposal.")
                    }),
                );
                control(ui, "story-apply", &r);
                if r.clicked() {
                    actions.call = Some(apply_call(draft));
                }
                let r = helto::action(ui, Icon::Trash, "Discard proposal", false, wait);
                control(ui, "story-discard", &r);
                if r.clicked() {
                    actions.new_proposal = true;
                    actions.close = true;
                }
            }
            // The instruction steers the next proposal; three buttons follow it.
            let gap = ui.spacing().item_spacing.x;
            let width = (ui.available_width() - 3. * (helto::ICON_BUTTON + gap)).max(80.);
            let r = ui.add_enabled(
                !busy,
                egui::TextEdit::singleline(&mut state.instruction)
                    .hint_text("Instruction (optional)")
                    .desired_width(width),
            );
            control(ui, "story-instruction", &r);
            let submit = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let again = state.draft.is_some();
            let r = helto::action(
                ui,
                if again { Icon::Retry } else { Icon::Generate },
                if again {
                    "Generate again"
                } else {
                    "Generate enhancement"
                },
                !again,
                if document.is_none() {
                    Some("Create or open a project first.")
                } else {
                    wait.or(starting.then_some("Starting the enhancement."))
                },
            );
            control(
                ui,
                if again {
                    "story-again"
                } else {
                    "story-generate"
                },
                &r,
            );
            if (r.clicked() || submit && !busy && !starting)
                && let Some(d) = document
            {
                if let Some(draft) = &state.draft {
                    // The same field against the current project; the old
                    // proposal stays under Recent proposals.
                    state.operation = draft.request.operation;
                    state.scene_field = draft.request.scene_field;
                    state.target_scene = draft.request.scene_id.clone();
                    state.target_section = draft.request.section_id.clone();
                    actions.new_proposal = true;
                }
                actions.call = Some(generate_call(state, d));
            }
            let r = helto::icon_button(
                ui,
                Icon::Settings,
                "Enhancement options",
                state.show_options,
            );
            control(ui, "story-options", &r);
            if r.clicked() {
                state.show_options = !state.show_options;
            }
            // Closing keeps the proposal; Enhance on the field reopens it.
            let r = helto::icon_button(ui, Icon::Close, "Close enhancement", false);
            control(ui, "story-close", &r);
            actions.close |= r.clicked();
        });
        if state.show_options {
            ui.separator();
            self.authoring_header(ui, state, actions);
            self.authoring_sources(ui, state, actions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Rect;
    type Nodes = Vec<(egui::accesskit::NodeId, egui::accesskit::Node)>;
    fn frame(ctx: &egui::Context, app: &mut ImagineApp, events: Vec<egui::Event>) -> Nodes {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1600., 1000.),
                )),
                events,
                focused: true,
                ..Default::default()
            },
            |ui| app.content(ui),
        );
        output.textures_delta.clear();
        output.platform_output.accesskit_update.unwrap().nodes
    }
    fn find<'a>(nodes: &'a Nodes, label: &str) -> Option<&'a egui::accesskit::Node> {
        nodes
            .iter()
            .map(|(_, n)| n)
            .find(|n| n.label().is_some_and(|t| t == label))
    }
    fn top(nodes: &Nodes, label: &str) -> f64 {
        find(nodes, label)
            .and_then(|n| n.bounds())
            .unwrap_or_else(|| panic!("{label}"))
            .y0
    }
    fn click(ctx: &egui::Context, app: &mut ImagineApp, nodes: &Nodes, label: &str) {
        let b = find(nodes, label)
            .and_then(|n| n.bounds())
            .unwrap_or_else(|| panic!("{label}"));
        let pos = egui::pos2(((b.x0 + b.x1) / 2.) as f32, ((b.y0 + b.y1) / 2.) as f32);
        for pressed in [true, false] {
            frame(
                ctx,
                app,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
    }
    fn app(ctx: &egui::Context, root: &std::path::Path) -> ImagineApp {
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(ctx, root.into());
        // Test frames carry no time, so scrolling must not animate.
        ctx.all_styles_mut(|s| s.scroll_animation = egui::style::ScrollAnimation::none());
        app.client = None;
        let project = super::super::super::media_picker::tests::project();
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.selected.insert(project.scenes[0].id.clone());
        app.timeline.document = Some(Snapshot::with(project, json!({})));
        app
    }
    fn proposal(project: &VideoProject, field: SceneField) -> Draft {
        serde_json::from_value(json!({
            "schema_version":1,"id":"own-enhance","project_id":project.id,
            "request":{"schema_version":1,"expected_project_revision":project.project_revision,
                "provider_profile_id":"proxy","provider":"koboldcpp","prompt_profile_id":"normal",
                "operation":"enhance_scene","scene_field":field,"scene_id":"scene",
                "brief":"","duration_seconds":10.0},
            "phase":"review","outline":null,"scenes":[],"text":"Own better description",
            "failure":null,"failed_index":null,"prompt_profile_id":"normal",
            "prompt_profile_revision":"1","profile_warning":"","provider_snapshot":{}
        }))
        .unwrap()
    }

    #[test]
    fn enhancing_a_field_opens_its_card_under_that_field_without_a_window() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = app(&ctx, root.path());
        let mut nodes = vec![];
        for _ in 0..3 {
            nodes = frame(&ctx, &mut app, vec![]);
        }
        assert!(find(&nodes, "Close enhancement").is_none());
        click(&ctx, &mut app, &nodes, "Enhance Description");
        for _ in 0..2 {
            nodes = frame(&ctx, &mut app, vec![]);
        }
        let state = &app.timeline.authoring;
        assert!(state.open);
        assert_eq!(state.operation, Operation::EnhanceScene);
        assert_eq!(state.scene_field, Some(SceneField::Description));
        // Between Description and the Summary under it, not in a window.
        let card = top(&nodes, "Close enhancement");
        assert!(card > top(&nodes, "Enhance Description"));
        assert!(card < top(&nodes, "Enhance Summary"));
        assert!(find(&nodes, "Story & Enhance").is_none());
        assert!(find(&nodes, "Generate enhancement").is_some());
    }

    #[test]
    fn a_proposal_sits_under_its_field_and_discard_closes_it() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = app(&ctx, root.path());
        let project = app.timeline.document.as_ref().unwrap().project.clone();
        let state = &mut app.timeline.authoring;
        state.open = true;
        state.operation = Operation::EnhanceScene;
        state.target_scene = Some(project.scenes[0].id.clone());
        state.scene_field = Some(SceneField::Description);
        state.draft = Some(proposal(&project, SceneField::Description));
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let mut output = egui::FullOutput::default();
            for _ in 0..4 {
                output.append(ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(1600., 1000.),
                        )),
                        focused: true,
                        ..Default::default()
                    },
                    |ui| app.content(ui),
                ));
            }
            crate::app::tests::render(&ctx, output, 1600, 1000)
                .save(std::path::PathBuf::from(dir).join("enhance-inline.png"))
                .unwrap();
        }
        let mut nodes = vec![];
        for _ in 0..3 {
            nodes = frame(&ctx, &mut app, vec![]);
        }
        let used = find(&nodes, "Use proposal").expect("Use proposal");
        assert!(!used.is_disabled());
        let card = top(&nodes, "Use proposal");
        assert!(card > top(&nodes, "Enhance Description"));
        assert!(card < top(&nodes, "Enhance Summary"));
        assert!(find(&nodes, "Generate again").is_some());
        assert!(nodes.iter().any(|(_, n)| {
            n.value()
                .is_some_and(|v| v.contains("Own better description"))
        }));
        click(&ctx, &mut app, &nodes, "Discard proposal");
        nodes = frame(&ctx, &mut app, vec![]);
        assert!(!app.timeline.authoring.open);
        assert!(app.timeline.authoring.draft.is_none());
        assert_eq!(app.timeline.authoring.history.len(), 1);
        assert!(find(&nodes, "Use proposal").is_none());
    }
}
