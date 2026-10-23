use super::*;

struct UiHarness {
    ctx: egui::Context,
    app: ImagineApp,
    root: tempfile::TempDir,
    nodes: Vec<(egui::accesskit::NodeId, egui::accesskit::Node)>,
}
impl UiHarness {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        ctx.global_style_mut(|style| style.animation_time = 0.);
        let mut harness = Self {
            ctx,
            app,
            root,
            nodes: Vec::new(),
        };
        harness.app.enhancement_message(
            &harness.ctx,
            Event::Profiles {
                result: Ok(vec![Profile::default()]),
                select: None,
                deleted: None,
            },
        );
        harness
            .app
            .prefs
            .model_groups
            .entry(crate::model_catalog::QWEN_GROUP.into())
            .or_default()
            .enhancement_profile_id = Some("proxy".into());
        for _ in 0..3 {
            harness.frame(vec![], false);
        }
        harness
    }
    fn frame(&mut self, events: Vec<egui::Event>, settings: bool) -> egui::FullOutput {
        let mut output = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    if settings {
                        egui::vec2(480., 1400.)
                    } else {
                        egui::vec2(1280., 800.)
                    },
                )),
                events,
                ..Default::default()
            },
            |ui| {
                if settings {
                    self.app.enhancement_settings(ui);
                } else {
                    self.app.content(ui);
                }
            },
        );
        self.nodes = output
            .platform_output
            .accesskit_update
            .as_ref()
            .unwrap()
            .nodes
            .clone();
        output.textures_delta.clear();
        output
    }
    fn node(&self, label: &str) -> &egui::accesskit::Node {
        self.nodes
            .iter()
            .map(|(_, n)| n)
            .filter(|n| {
                n.label() == Some(label)
                    || (n.role() == egui::accesskit::Role::ComboBox && n.value() == Some(label))
                    || n.labelled_by().iter().any(|id| {
                        self.nodes.iter().any(|(other, l)| {
                            id == other && (l.label() == Some(label) || l.value() == Some(label))
                        })
                    })
            })
            .min_by_key(|n| match n.role() {
                egui::accesskit::Role::MultilineTextInput | egui::accesskit::Role::TextInput => 0,
                egui::accesskit::Role::Button => 1,
                _ => 2,
            })
            .unwrap_or_else(|| panic!("Missing {label}"))
    }
    fn click(&mut self, label: &str, settings: bool) {
        let bounds = self.node(label).bounds().unwrap();
        let pos = egui::pos2(
            (bounds.x0 + bounds.x1) as f32 / 2.,
            (bounds.y0 + bounds.y1) as f32 / 2.,
        );
        for pressed in [true, false] {
            self.frame(
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
                settings,
            );
        }
        self.frame(vec![], settings);
    }
    fn pending(&mut self, target: Target) -> String {
        let id = uuid::Uuid::new_v4().to_string();
        self.app.enhancement.active = Some(Run {
            id: id.clone(),
            target,
            original: self.app.enhancement_text(target).clone(),
            context: self.app.enhancement_context(target),
            widget: self.app.enhancement.fields[&target],
            cancel: None,
        });
        self.frame(vec![], false);
        id
    }
    fn finish(&mut self, id: String, result: Result<String, Failure>) {
        self.app
            .enhancement_message(&self.ctx, Event::Finished { id, result });
        self.frame(vec![], false);
        self.frame(vec![], false);
        self.frame(vec![], false);
    }
}

#[test]
fn enhancement_replaces_the_origin_field_as_one_undo_step_without_touching_generation() {
    let mut h = UiHarness::new();
    let target = Target {
        workspace: Workspace::Image,
        field: Field::Prompt,
    };
    let original = h.app.image.prompt.clone();
    let id = h.pending(target);
    h.finish(id, Ok("SYNTHETIC improved prompt".into()));
    assert_eq!(h.app.image.prompt, "SYNTHETIC improved prompt");
    assert!(h.ctx.text_edit_focused());
    h.frame(
        vec![egui::Event::Key {
            key: egui::Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers {
                ctrl: true,
                command: true,
                ..egui::Modifiers::NONE
            },
        }],
        false,
    );
    assert_eq!(h.app.image.prompt, original);
    assert!(!h.app.image.actual_pixels);
    assert!(!h.app.image.working);
    assert!(h.app.image.pending.is_none());

    let id = h.pending(target);
    h.finish(id, Ok("SYNTHETIC second improvement".into()));
    let widget = h.app.enhancement.fields[&target];
    let state = egui::TextEdit::load_state(&h.ctx, widget).unwrap();
    assert!(!state.undoer().has_redo(&(
        state.cursor.char_range().unwrap(),
        h.app.image.prompt.clone()
    )));
    h.frame(
        vec![egui::Event::Key {
            key: egui::Key::Z,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers {
                ctrl: true,
                command: true,
                ..egui::Modifiers::NONE
            },
        }],
        false,
    );
    assert_eq!(h.app.image.prompt, original);

    h.app.workspace = Workspace::CharacterSheet;
    h.app.sheet.character_description = "SYNTHETIC character".into();
    h.frame(vec![], false);
    let target = Target {
        workspace: Workspace::CharacterSheet,
        field: Field::Character,
    };
    let id = h.pending(target);
    h.app.workspace = Workspace::Image;
    h.finish(id, Ok("SYNTHETIC improved character".into()));
    assert_eq!(
        h.app.sheet.character_description,
        "SYNTHETIC improved character"
    );
    assert_eq!(h.app.image.prompt, original);
}

#[test]
fn edited_prompts_require_explicit_use_and_errors_cancellation_and_context_changes_preserve_text() {
    let mut h = UiHarness::new();
    let target = Target {
        workspace: Workspace::Image,
        field: Field::Prompt,
    };
    let id = h.pending(target);
    h.app.image.prompt = "SYNTHETIC user's newer draft".into();
    h.finish(id, Ok("SYNTHETIC suggested prompt".into()));
    assert_eq!(h.app.image.prompt, "SYNTHETIC user's newer draft");
    h.click("Use", false);
    assert_eq!(h.app.image.prompt, "SYNTHETIC suggested prompt");
    let id = h.pending(target);
    h.app.image.prompt = "SYNTHETIC another edit".into();
    h.finish(id, Ok("SYNTHETIC discarded suggestion".into()));
    h.click("Discard", false);
    assert_eq!(h.app.image.prompt, "SYNTHETIC another edit");

    let id = h.pending(target);
    h.finish(
        id,
        Err(Failure {
            message: "SYNTHETIC provider error".into(),
            retryable: false,
        }),
    );
    assert_eq!(h.app.image.prompt, "SYNTHETIC another edit");
    assert!(h.app.enhancement.error.is_some());
    let id = h.pending(target);
    h.click("Cancel enhancement", false);
    h.finish(id, Ok("SYNTHETIC late result".into()));
    assert_eq!(h.app.image.prompt, "SYNTHETIC another edit");
    let id = h.pending(target);
    h.app.image.references.clear();
    h.finish(id, Ok("SYNTHETIC wrong context".into()));
    assert_eq!(h.app.image.prompt, "SYNTHETIC another edit");
    assert!(
        h.app
            .enhancement
            .error
            .as_ref()
            .unwrap()
            .1
            .contains("references changed")
    );
}

#[test]
fn profile_editor_edits_required_system_prompt_and_new_copied_profiles_keep_instructions() {
    let mut h = UiHarness::new();
    for _ in 0..3 {
        h.frame(vec![], true);
    }
    h.click("Prompt enhancement", true);
    assert!(
        !h.nodes
            .iter()
            .any(|(_, node)| { matches!(node.label(), Some("API key" | "Clear saved key")) })
    );
    let system = h.node("System prompt").bounds().unwrap();
    assert!(system.width() > 300. && system.x1 < 480., "{system:?}");
    h.click("System prompt", true);
    let system_prompt =
        "SYNTHETIC editable system prompt: preserve åäö and all reference bindings.\n".repeat(4096);
    assert!(system_prompt.len() > 65536);
    h.frame(
        vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers {
                    ctrl: true,
                    command: true,
                    ..egui::Modifiers::NONE
                },
            },
            egui::Event::Paste(system_prompt.clone()),
        ],
        true,
    );
    assert_eq!(
        h.app.enhancement.draft.as_ref().unwrap().system_prompt,
        system_prompt
    );
    h.app.prefs.privacy_mode = true;
    let output = h.frame(vec![egui::Event::PointerGone], true);
    fn visible_system_text(shape: &egui::Shape) -> bool {
        match shape {
            egui::Shape::Text(text) => {
                text.opacity_factor > 0.
                    && text
                        .galley
                        .text()
                        .contains("SYNTHETIC editable system prompt")
            }
            egui::Shape::Vec(shapes) => shapes.iter().any(visible_system_text),
            _ => false,
        }
    }
    assert!(
        !output
            .shapes
            .iter()
            .any(|shape| visible_system_text(&shape.shape))
    );
    h.app.prefs.privacy_mode = false;
    assert!(!h.node("Save profile").is_disabled());
    h.click("Duplicate enhancement profile", true);
    assert_eq!(
        h.app.enhancement.draft.as_ref().unwrap().system_prompt,
        system_prompt
    );
    assert_ne!(h.app.enhancement.draft.as_ref().unwrap().id, "proxy");
    h.app
        .enhancement
        .draft
        .as_mut()
        .unwrap()
        .system_prompt
        .clear();
    h.frame(vec![], true);
    assert!(h.node("Save profile").is_disabled());
    h.click("New enhancement profile", true);
    // The unsaved copy is only replaced after confirmation.
    assert!(
        h.app
            .enhancement
            .draft
            .as_ref()
            .unwrap()
            .system_prompt
            .is_empty()
    );
    h.click("Discard changes", true);
    assert!(
        !h.app
            .enhancement
            .draft
            .as_ref()
            .unwrap()
            .system_prompt
            .is_empty()
    );

    h.app.save_preferences();
    let restored = Preferences::load(h.root.path()).unwrap();
    assert_eq!(
        restored.model_groups[crate::model_catalog::QWEN_GROUP]
            .enhancement_profile_id
            .as_deref(),
        Some("proxy")
    );
    let mut old = serde_json::to_value(&restored).unwrap();
    old["model_groups"][crate::model_catalog::QWEN_GROUP]
        .as_object_mut()
        .unwrap()
        .remove("enhancement_profile_id");
    let migrated: Preferences = serde_json::from_value(old).unwrap();
    assert!(
        migrated.model_groups[crate::model_catalog::QWEN_GROUP]
            .enhancement_profile_id
            .is_none()
    );
}

#[tokio::test]
async fn enhance_button_uses_saved_profile_through_the_real_http_client() {
    use axum::{
        Json, Router,
        routing::{get, post},
    };
    use serde_json::json;
    let mut h = UiHarness::new();
    let original = h.app.image.prompt.clone();
    let provider = Router::new()
        .route("/v1/models", get(|| async { Json(json!({"data":[{"id":"SYNTHETIC discovered model"}]})) }))
        .route("/v1/chat/completions", post(move |Json(body): Json<Value>| {
        let original = original.clone();
        async move {
            assert_eq!(body["messages"][0]["content"], "SYNTHETIC UI system instructions");
            assert_eq!(body["messages"][1]["content"], original);
            assert_eq!(body["model"], "SYNTHETIC discovered model");
            assert_eq!(body["top_p"], 1.0);
            assert_eq!(body["stop"], json!(["<end>"]));
            Json(json!({"choices":[{"finish_reason":"stop","message":{"content":"SYNTHETIC HTTP improvement"}}]}))
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let provider_task = tokio::spawn(async move {
        axum::serve(listener, provider).await.unwrap();
    });
    let service =
        imagine_server::enhancement::Service::open(h.root.path().join("enhancement.json")).unwrap();
    service
        .save(imagine_server::enhancement_types::ProfileUpdate {
            profile: imagine_server::enhancement_types::Profile {
                base_url,
                system_prompt: "SYNTHETIC UI system instructions".into(),
                ..Default::default()
            },
        })
        .unwrap();
    let profiles =
        serde_json::from_value(serde_json::to_value(service.profiles()).unwrap()).unwrap();
    h.app.enhancement_message(
        &h.ctx,
        Event::Profiles {
            result: Ok(profiles),
            select: None,
            deleted: None,
        },
    );
    let router = imagine_server::enhancement::routes(service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    h.app.client = Some(
        Client::connect(&Connection {
            url: format!("http://{}", listener.local_addr().unwrap()),
        })
        .unwrap(),
    );
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    h.frame(vec![], true);
    h.click("Prompt enhancement", true);
    h.click("Load models", true);
    tokio::time::timeout(Duration::from_secs(3), async {
        while h.app.enhancement.models_pending.is_some() {
            tokio::time::sleep(Duration::from_millis(5)).await;
            h.app.receive(&h.ctx.clone());
        }
    })
    .await
    .unwrap();
    h.frame(vec![], true);
    assert_eq!(h.app.enhancement.models, vec!["SYNTHETIC discovered model"]);
    h.click("Choose model", true);
    h.click("SYNTHETIC discovered model", true);
    assert_eq!(
        h.app.enhancement.draft.as_ref().unwrap().model,
        "SYNTHETIC discovered model"
    );
    h.click("Top P", true);
    h.click("Advanced", true);
    h.click("Extra parameters (JSON)", true);
    h.frame(
        vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers {
                    ctrl: true,
                    command: true,
                    ..egui::Modifiers::NONE
                },
            },
            egui::Event::Paste(r#"{"stop":["<end>"]}"#.into()),
        ],
        true,
    );
    assert!(h.app.enhancement.extra_error.is_none());
    h.click("Advanced", true);
    h.click("System prompt", true);
    h.frame(
        vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers {
                    ctrl: true,
                    command: true,
                    ..egui::Modifiers::NONE
                },
            },
            egui::Event::Text("SYNTHETIC UI system instructions".into()),
        ],
        true,
    );
    h.click("Save profile", true);
    tokio::time::timeout(Duration::from_secs(3), async {
        while h.app.enhancement.saving {
            tokio::time::sleep(Duration::from_millis(5)).await;
            h.frame(vec![], false);
        }
    })
    .await
    .unwrap();
    assert!(h.app.enhancement.settings_error.is_none());
    assert_eq!(
        h.app.enhancement.profiles[0].system_prompt,
        "SYNTHETIC UI system instructions"
    );
    for _ in 0..3 {
        h.frame(vec![], false);
    }
    h.click("Enhance prompt", false);
    assert!(h.app.enhancement.active.is_some());
    tokio::time::timeout(Duration::from_secs(3), async {
        while h.app.image.prompt != "SYNTHETIC HTTP improvement" {
            tokio::time::sleep(Duration::from_millis(5)).await;
            h.frame(vec![], false);
        }
    })
    .await
    .unwrap();
    assert!(h.app.enhancement.active.is_none());
    assert!(h.app.image.pending.is_none());
    server.abort();
    provider_task.abort();
}

#[test]
fn model_results_are_ignored_after_connection_changes_and_json_cannot_override_controls() {
    let mut h = UiHarness::new();
    let lookup = h.app.enhancement_models_request().unwrap();
    h.app.enhancement.models_pending = Some(lookup.clone());
    h.app.enhancement.draft.as_mut().unwrap().base_url = "http://127.0.0.1:1/v1".into();
    h.app.enhancement_message(
        &h.ctx,
        Event::Models {
            lookup,
            result: Ok(vec!["wrong server".into()]),
        },
    );
    assert!(h.app.enhancement.models_for.is_none());
    assert!(h.app.enhancement.models.is_empty());
    for text in ["[]", "{", r#"{"messages":[]}"#, r#"{"top_p":0.9}"#] {
        assert!(parse_extra_parameters(text).is_err());
    }
    assert_eq!(
        parse_extra_parameters(
            r#"{"stop":["<end>"],"chat_template_kwargs":{"enable_thinking":false}}"#
        )
        .unwrap()["stop"],
        serde_json::json!(["<end>"])
    );
}

#[test]
fn deleting_or_leaving_an_edited_profile_asks_first() {
    let mut h = UiHarness::new();
    for _ in 0..3 {
        h.frame(vec![], true);
    }
    h.click("Prompt enhancement", true);
    let original = h.app.enhancement.draft.clone().unwrap();
    h.click("Delete enhancement profile", true);
    assert!(!h.app.enhancement.saving);
    h.click("Keep profile", true);
    assert_eq!(h.app.enhancement.draft.as_ref(), Some(&original));
    h.app.enhancement.draft.as_mut().unwrap().name = "Own unsaved name".into();
    h.click("New enhancement profile", true);
    assert_eq!(
        h.app.enhancement.draft.as_ref().unwrap().name,
        "Own unsaved name"
    );
    h.click("Keep profile", true);
    assert_eq!(
        h.app.enhancement.draft.as_ref().unwrap().name,
        "Own unsaved name"
    );
    h.click("New enhancement profile", true);
    h.click("Discard changes", true);
    assert_eq!(
        h.app.enhancement.draft.as_ref().unwrap().name,
        "New profile"
    );
}

#[test]
fn a_proposal_cannot_be_used_after_its_context_changes_and_keeps_its_text() {
    let mut h = UiHarness::new();
    let target = Target {
        workspace: Workspace::Image,
        field: Field::Prompt,
    };
    let id = h.pending(target);
    h.app.image.prompt = "SYNTHETIC newer draft".into();
    h.finish(id, Ok("SYNTHETIC proposal".into()));
    h.app.image.references.clear();
    h.frame(vec![], false);
    h.frame(vec![], false);
    assert!(h.node("Use").is_disabled());
    assert!(h.app.enhancement.proposal.is_some());
    h.click("Discard", false);
    assert_eq!(h.app.image.prompt, "SYNTHETIC newer draft");
}

#[test]
fn a_proposal_shows_under_its_field_and_retry_enhances_the_current_text() {
    let mut h = UiHarness::new();
    let target = Target {
        workspace: Workspace::Image,
        field: Field::Prompt,
    };
    let id = h.pending(target);
    h.app.image.prompt = "SYNTHETIC newer draft".into();
    h.finish(id, Ok("SYNTHETIC proposal".into()));
    let field = h.node("Prompt").bounds().unwrap();
    let proposal = h.node("Use").bounds().unwrap();
    assert!(proposal.y0 > field.y1, "{proposal:?} below {field:?}");
    assert!(h.node("Enhance prompt").is_disabled());
    h.click("Retry", false);
    assert!(h.app.enhancement.proposal.is_none());
    assert!(
        h.app
            .enhancement
            .active
            .as_ref()
            .is_some_and(|run| run.original == "SYNTHETIC newer draft")
            || h.app.enhancement.error.is_some()
    );
    assert_eq!(h.app.image.prompt, "SYNTHETIC newer draft");
    h.app.client = None;
    h.app.enhancement.active = None;
    let id = h.pending(target);
    h.app.image.prompt = "SYNTHETIC offline draft".into();
    h.finish(id, Ok("SYNTHETIC proposal".into()));
    assert!(h.node("Retry").is_disabled());
}
