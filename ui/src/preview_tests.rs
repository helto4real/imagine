//! Synthetic, headless layout evidence. No window, desktop capture or GPU.
use super::*;
use std::collections::HashMap;

#[path = "editing_tests.rs"]
mod editing_tests;

#[path = "shortcut_tests.rs"]
mod shortcut_tests;

#[path = "queue_tests.rs"]
mod queue_tests;

struct Desk {
    ctx: egui::Context,
    app: ImagineApp,
    nodes: Vec<(egui::accesskit::NodeId, egui::accesskit::Node)>,
    _root: tempfile::TempDir,
}
impl Desk {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let mut desk = Self {
            ctx,
            app,
            nodes: Vec::new(),
            _root: root,
        };
        for _ in 0..3 {
            desk.frame(Vec::new());
        }
        desk
    }
    fn frame(&mut self, events: Vec<egui::Event>) {
        let mut output = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280., 800.),
                )),
                events,
                ..Default::default()
            },
            |ui| self.app.content(ui),
        );
        output.textures_delta.clear();
        self.nodes = output.platform_output.accesskit_update.unwrap().nodes;
    }
    fn control(&self, name: &str) -> &egui::accesskit::Node {
        self.nodes
            .iter()
            .map(|(_, node)| node)
            .filter(|node| {
                node.label() == Some(name)
                    || node.labelled_by().iter().any(|label_id| {
                        self.nodes.iter().any(|(id, label)| {
                            id == label_id
                                && (label.label() == Some(name) || label.value() == Some(name))
                        })
                    })
            })
            .min_by(|a, b| a.bounds().unwrap().y0.total_cmp(&b.bounds().unwrap().y0))
            .unwrap_or_else(|| panic!("Missing control: {name}"))
    }
    /// Opens the prompt panel's Advanced section and scrolls it into view.
    fn open_advanced(&mut self) {
        self.click("Advanced");
        let panel = self.control("Advanced").bounds().unwrap();
        let pos = egui::pos2(((panel.x0 + panel.x1) / 2.) as f32, panel.y0 as f32);
        for _ in 0..3 {
            self.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0., -1000.),
                    phase: egui::TouchPhase::Move,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        self.frame(vec![egui::Event::PointerGone]);
    }
    fn click(&mut self, name: &str) {
        let rect = self.control(name).bounds().unwrap();
        let pos = egui::pos2(
            (rect.x0 + rect.x1) as f32 / 2.,
            (rect.y0 + rect.y1) as f32 / 2.,
        );
        for pressed in [true, false] {
            self.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        self.frame(Vec::new());
    }
}

#[test]
fn prompt_height_can_be_dragged_without_changing_width_or_text() {
    let mut desk = Desk::new();
    let text = desk.app.image.prompt.clone();
    let before = desk.control("Prompt").bounds().unwrap();
    let ratio_y = desk.control("Ratio").bounds().unwrap().y0;
    let start = egui::pos2(before.x1 as f32 - 3., before.y1 as f32 - 3.);
    desk.frame(vec![egui::Event::PointerMoved(start)]);
    desk.frame(vec![egui::Event::PointerButton {
        pos: start,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    }]);
    let end = start + egui::vec2(0., 90.);
    desk.frame(vec![egui::Event::PointerMoved(end)]);
    desk.frame(Vec::new());
    desk.frame(vec![egui::Event::PointerButton {
        pos: end,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    desk.frame(Vec::new());
    let after = desk.control("Prompt").bounds().unwrap();
    assert!(
        after.height() > before.height() + 70.,
        "{before:?} -> {after:?}"
    );
    assert!((after.width() - before.width()).abs() < 1.);
    assert!(desk.control("Ratio").bounds().unwrap().y0 > ratio_y + 70.);
    assert_eq!(desk.app.image.prompt, text);
    let resized_ratio_y = desk.control("Ratio").bounds().unwrap().y0;
    desk.app.image.prompt = "Synthetic long prompt line\n".repeat(80);
    desk.frame(Vec::new());
    desk.frame(Vec::new());
    assert!((desk.control("Ratio").bounds().unwrap().y0 - resized_ratio_y).abs() < 1.);
}

#[test]
fn reference_desk_tabs_settings_and_zoom_are_interactive() {
    let mut desk = Desk::new();
    let prompt = desk.app.image.prompt.clone();
    desk.click("Video");
    assert!(desk.app.workspace == Workspace::Video);
    desk.click("Image");
    assert!(desk.app.workspace == Workspace::Image);
    assert_eq!(desk.app.image.prompt, prompt);
    assert_eq!(desk.app.image.references.len(), 2);
    desk.click("Actual pixels");
    assert!(desk.app.image.actual_pixels);
    desk.click("Pan image");
    assert!(!desk.app.image.pan_enabled);
    desk.click("Fit image");
    assert!(!desk.app.image.actual_pixels);
    assert_eq!(desk.app.image.image_pan, egui::Vec2::ZERO);
    desk.click("Pan image");
    assert!(desk.app.image.actual_pixels && desk.app.image.pan_enabled);
    desk.click("Hide previews");
    assert!(desk.app.prefs.privacy_mode);
    assert!(Preferences::load(desk._root.path()).unwrap().privacy_mode);
    desk.click("Settings");
    assert!(desk.app.settings_open);
    desk.click("Close settings");
    assert!(!desk.app.settings_open);
    desk.click("Settings");
    desk.frame(vec![egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    assert!(!desk.app.settings_open);
}

#[test]
fn reference_order_and_pending_request_controls_are_preserved() {
    let mut desk = Desk::new();
    desk.click("Move reference down");
    assert_eq!(desk.app.image.references[0].id, "synthetic-reference-2");
    desk.click("Remove reference");
    assert_eq!(desk.app.image.references.len(), 1);
    assert_eq!(desk.app.image.references[0].id, "synthetic-reference-1");
    desk.app.image.pending = Some(Submission {
        upscale_source: None,
        request: serde_json::json!({}),
        key: "existing-key".into(),
        folder: desk._root.path().into(),
        job: Some("existing-job".into()),
    });
    desk.app.image.error = Some("Synthetic recoverable error".into());
    desk.frame(Vec::new());
    assert!(desk.control("Add to queue").is_disabled());
    assert!(!desk.control("Retry").is_disabled());
    assert!(desk.control("Remove reference").is_disabled());
    desk.click("Remove reference");
    assert_eq!(desk.app.image.references.len(), 1);
    desk.app.image.working = true;
    desk.frame(Vec::new());
    assert!(
        !desk
            .nodes
            .iter()
            .any(|(_, node)| node.label() == Some("Retry"))
    );
    assert!(desk.control("Upscale").is_disabled());
    assert!(!desk.control("Cancel generation").is_disabled());
    desk.app.image.cancelling = true;
    desk.frame(Vec::new());
    assert!(desk.control("Cancel generation").is_disabled());
}

#[test]
fn upscale_uses_result_id_and_captures_target_ratio_for_retry() {
    for random in [true, false] {
        let mut desk = Desk::new();
        assert!(!desk.control("Upscale").is_disabled());
        assert!(
            !desk
                .nodes
                .iter()
                .any(|(_, n)| n.label() == Some("Generate again"))
        );
        desk.app.prefs.generation.random_seed = random;
        desk.app.prefs.generation.seed = 42;
        desk.app.prefs.generation.aspect_ratio = "3:4".into();
        desk.app.prefs.upscale_target = crate::settings::UpscaleTarget::K3;
        desk.app.image.prompt = "Unrelated draft".into();
        let submission = desk.app.upscale_submission().unwrap();
        assert_eq!(submission.request["profile"], "image_upscale");
        assert_eq!(
            submission.request["inputs"]["reference_images"],
            serde_json::json!(["asset_synthetic"])
        );
        assert_eq!(submission.request["max_side"], 3072);
        assert_eq!(submission.request["aspect_ratio"], "3:4");
        assert_eq!(
            submission.request["seed"],
            if random {
                Value::Null
            } else {
                serde_json::json!(42)
            }
        );
        assert!(submission.request.get("prompt").is_none());
        assert!(submission.request.get("sampling").is_none());
        assert_ne!(submission.key, desk.app.upscale_submission().unwrap().key);
        assert_eq!(
            submission.upscale_source.as_ref(),
            Some(&desk.app.image.selected_result().unwrap().file)
        );
        let source_file = submission.upscale_source.clone();
        let captured = submission.request.clone();
        let key = submission.key.clone();
        desk.app.image.pending = Some(Submission {
            job: Some("job_upscale".into()),
            ..submission
        });
        desk.app.prefs.upscale_target = crate::settings::UpscaleTarget::K4;
        desk.app.prefs.generation.aspect_ratio = "1:1".into();
        desk.app
            .tx
            .send(Message::Finished(
                Workspace::Image,
                Err(Failure {
                    message: "Save failed".into(),
                    retryable: true,
                }),
            ))
            .unwrap();
        desk.frame(vec![]);
        assert!(desk.control("Upscale").is_disabled());
        assert!(!desk.control("Retry").is_disabled());
        let pending = desk.app.image.pending.as_ref().unwrap();
        assert_eq!(pending.request, captured);
        assert_eq!(pending.upscale_source, source_file);
        assert_eq!(pending.key, key);
        assert_eq!(pending.job.as_deref(), Some("job_upscale"));
        desk.app
            .tx
            .send(Message::Finished(
                Workspace::Image,
                Err(Failure {
                    message: "Cancelled".into(),
                    retryable: false,
                }),
            ))
            .unwrap();
        desk.frame(vec![]);
        assert!(!desk.control("Upscale").is_disabled());
        desk.app.image.result = None;
        desk.frame(vec![]);
        assert!(desk.control("Upscale").is_disabled());
        assert!(desk.app.upscale_submission().is_err());
    }
}

#[test]
fn editing_next_prompt_and_settings_does_not_change_the_running_job() {
    fn replace(desk: &mut Desk, bounds: egui::accesskit::Rect, value: &str) {
        let pos = egui::pos2(
            (bounds.x0 + bounds.x1) as f32 / 2.,
            (bounds.y0 + bounds.y1) as f32 / 2.,
        );
        for pressed in [true, false] {
            desk.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                },
            ]);
        }
        desk.frame(vec![
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
            egui::Event::Text(value.into()),
        ]);
    }
    let mut desk = Desk::new();
    let request = serde_json::json!({"prompt": "Running prompt", "negative_prompt": "Running negative", "sampling": {"steps": 40}});
    let original_folder = desk.app.prefs.output_folder.clone();
    desk.app.image.pending = Some(Submission {
        upscale_source: None,
        request: request.clone(),
        key: "running-key".into(),
        job: Some("running-job".into()),
        folder: original_folder.clone(),
    });
    desk.app.image.working = true;
    desk.frame(Vec::new());
    for name in ["Prompt", "Ratio", "Max side", "Seed"] {
        assert!(!desk.control(name).is_disabled(), "{name}");
    }
    assert!(desk.control("Add to queue").is_disabled());
    assert!(desk.control("Upscale").is_disabled());
    let prompt = desk.control("Prompt").bounds().unwrap();
    replace(&mut desk, prompt, "Next prompt");
    assert_eq!(desk.app.image.prompt, "Next prompt");
    desk.ctx.global_style_mut(|style| style.animation_time = 0.);
    desk.click("Negative prompt");
    // The lower of the two multiline fields is the expanded negative prompt.
    let negative = desk
        .nodes
        .iter()
        .map(|(_, node)| node)
        .filter(|node| node.role() == egui::accesskit::Role::MultilineTextInput)
        .max_by(|a, b| a.bounds().unwrap().y0.total_cmp(&b.bounds().unwrap().y0))
        .unwrap();
    assert!(!negative.is_disabled());
    let negative = negative.bounds().unwrap();
    replace(&mut desk, negative, "Next negative prompt");
    assert_eq!(desk.app.prefs.negative_prompt, "Next negative prompt");
    // Typing is saved once the field is left, not on every keystroke.
    assert_ne!(
        Preferences::load(desk._root.path())
            .unwrap()
            .negative_prompt,
        "Next negative prompt"
    );
    desk.frame(vec![egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    assert_eq!(
        Preferences::load(desk._root.path())
            .unwrap()
            .negative_prompt,
        "Next negative prompt"
    );
    desk.click("Settings");
    assert!(!desk.control("Choose output folder").is_disabled());
    desk.click("Close settings");
    // Sampling settings live in the prompt panel's Advanced section.
    desk.open_advanced();
    for name in ["Steps", "CFG", "Reference size"] {
        assert!(!desk.control(name).is_disabled(), "{name}");
    }
    let steps = desk.control("Steps").bounds().unwrap();
    replace(&mut desk, steps, "24");
    desk.frame(vec![egui::Event::Key {
        key: egui::Key::Enter,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    assert_eq!(desk.app.prefs.generation.steps, 24);
    let reference_size = desk.control("Reference size").bounds().unwrap();
    replace(&mut desk, reference_size, "2.5");
    desk.frame(vec![egui::Event::Key {
        key: egui::Key::Enter,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    assert_eq!(
        desk.app.prefs.generation.reference_max_pixels_per_image,
        2_500_000
    );
    assert_eq!(
        Preferences::load(desk._root.path())
            .unwrap()
            .generation
            .reference_max_pixels_per_image,
        2_500_000
    );
    let new_folder = desk._root.path().join("next-output");
    desk.app
        .tx
        .send(Message::Folder(Some(new_folder.clone())))
        .unwrap();
    desk.app.receive(&desk.ctx);
    let pending = desk.app.image.pending.as_ref().unwrap();
    assert_eq!(pending.request, request);
    assert_eq!(pending.folder, original_folder);
    assert_eq!(pending.key, "running-key");
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::Image,
            Ok(Completed {
                assets: vec![client::CompletedAsset {
                    asset_id: "asset_synthetic".into(),
                    role: OutputRole::Image,
                    file: original_folder.join("synthetic.png"),
                    preview: Preview {
                        width: 1,
                        height: 1,
                        pixels: vec![0, 0, 0, 255],
                    },
                }],
                seed: Some(42),
            }),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(desk.app.image.prompt, "Next prompt");
    assert_eq!(desk.app.prefs.negative_prompt, "Next negative prompt");
    assert_eq!(desk.app.prefs.generation.steps, 24);
    assert_eq!(desk.app.prefs.output_folder, new_folder);
    assert!(desk.app.image.pending.is_none());
    assert_eq!(
        desk.app.prefs.generation.reference_max_pixels_per_image,
        2_500_000
    );
}

#[test]
fn settings_controls_save_and_report_write_failures() {
    let mut desk = Desk::new();
    desk.open_advanced();
    desk.click("Steps");
    desk.frame(vec![
        egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::CTRL,
        },
        egui::Event::Text("24".into()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    assert_eq!(desk.app.prefs.generation.steps, 24);
    assert_eq!(
        Preferences::load(desk._root.path())
            .unwrap()
            .generation
            .steps,
        24
    );
    let saved = desk._root.path().join(".imagine/ui-settings.json");
    std::fs::remove_file(&saved).unwrap();
    std::fs::create_dir(&saved).unwrap();
    desk.click("Hide previews");
    assert!(desk.app.settings_error.is_some());
    std::fs::remove_dir(&saved).unwrap();
    desk.click("Hide previews");
    assert!(desk.app.settings_error.is_none());
    assert!(!Preferences::load(desk._root.path()).unwrap().privacy_mode);
}

#[test]
fn headless_reference_desk_and_settings() {
    let root = tempfile::tempdir().unwrap();
    for (width, height, name) in [
        (1800, 850, "reference-desk"),
        (640, 1000, "reference-desk-compact"),
    ] {
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let mut combined = egui::FullOutput::default();
        for _ in 0..3 {
            combined.append(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    ..Default::default()
                },
                |ui| {
                    if width > 1000 {
                        ui.scope_builder(
                            egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(
                                egui::pos2(8., 8.),
                                egui::vec2(1280., 800.),
                            )),
                            |ui| app.content(ui),
                        );
                        ui.scope_builder(
                            egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(
                                egui::pos2(1310., 8.),
                                egui::vec2(480., 680.),
                            )),
                            |ui| app.settings_content(ui, false),
                        );
                    } else {
                        app.content(ui);
                    }
                },
            ));
        }
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            render(&ctx, combined, width, height)
                .save(dir.join(format!("{name}.png")))
                .unwrap();
        } else {
            combined.textures_delta.clear();
        }
    }
}

#[test]
fn actual_pixels_respects_display_scale_clipping_and_privacy() {
    for pixels_per_point in [1., 2.] {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(pixels_per_point);
        let image = texture(
            &ctx,
            &Preview {
                width: 800,
                height: 600,
                pixels: [255, 0, 0, 255].repeat(800 * 600),
            },
            "zoom-test",
        );
        for (actual, hidden) in [(false, false), (true, false), (true, true)] {
            let mut pan = egui::vec2(10_000., -10_000.);
            let mut output = ctx.run_ui(
                egui::RawInput {
                    focused: false,
                    ..Default::default()
                },
                |ui| {
                    image_view(
                        ui,
                        &image,
                        egui::vec2(200., 150.),
                        hidden,
                        actual,
                        true,
                        &mut pan,
                    );
                },
            );
            output.textures_delta.clear();
            let meshes = ctx.tessellate(output.shapes, output.pixels_per_point);
            let rendered = meshes.iter().find_map(|item| match &item.primitive {
                egui::epaint::Primitive::Mesh(mesh) if mesh.texture_id == image.id() => {
                    Some((item.clip_rect, mesh))
                }
                _ => None,
            });
            if hidden {
                assert!(rendered.is_none());
            } else {
                let (clip, mesh) = rendered.unwrap();
                let bounds = egui::Rect::from_points(
                    &mesh.vertices.iter().map(|v| v.pos).collect::<Vec<_>>(),
                );
                let expected = if actual {
                    800. / pixels_per_point
                } else {
                    200.
                };
                assert!((bounds.width() - expected).abs() < 0.01);
                assert!(clip.width() <= 200. && clip.height() <= 150.);
                if actual {
                    assert_eq!(
                        pan,
                        egui::vec2(
                            (expected - 200.) / 2.,
                            -(600. / pixels_per_point - 150.) / 2.
                        )
                    );
                }
            }
        }
    }
}

#[test]
fn privacy_off_shows_image_without_hover_or_focus_and_can_be_reenabled() {
    let ctx = egui::Context::default();
    let image = texture(
        &ctx,
        &Preview {
            width: 2,
            height: 2,
            pixels: [255, 0, 0, 255].repeat(4),
        },
        "privacy-toggle-test",
    );
    for privacy_mode in [false, true, false] {
        let mut output = ctx.run_ui(
            egui::RawInput {
                focused: false,
                events: vec![egui::Event::PointerGone],
                ..Default::default()
            },
            |ui| {
                private_image(ui, &image, egui::vec2(64., 48.), privacy_mode);
            },
        );
        output.textures_delta.clear();
        let draws_image = ctx.tessellate(output.shapes, output.pixels_per_point).iter().any(|mesh| {
            matches!(&mesh.primitive, egui::epaint::Primitive::Mesh(mesh) if mesh.texture_id == image.id())
        });
        assert_eq!(draws_image, !privacy_mode);
    }
}

#[test]
fn preferences_survive_folder_change_and_app_restart() {
    let root = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    let connection = Connection {
        url: "http://127.0.0.1:8080".into(),
    };
    let mut app = ImagineApp::empty(
        root.path().into(),
        connection.clone(),
        Preferences::load(root.path()).unwrap(),
    );
    app.prefs.privacy_mode = false;
    app.prefs.negative_prompt = "  artifacts\nextra fingrar  ".into();
    app.prefs.generation = crate::settings::GenerationSettings {
        steps: 25,
        cfg_scale: 3.5,
        sampler: Some("euler".into()),
        scheduler: Some("simple".into()),
        random_seed: false,
        seed: 0,
        aspect_ratio: "9:16".into(),
        max_side: 1024,
        reference_max_pixels_per_image: 2_000_000,
    };
    app.save_preferences();
    let folder = root.path().join("selected-images");
    app.tx.send(Message::Folder(Some(folder.clone()))).unwrap();
    app.receive(&ctx);
    assert!(app.image.error.is_none());
    let restarted = ImagineApp::empty(
        root.path().into(),
        connection,
        Preferences::load(root.path()).unwrap(),
    );
    assert_eq!(restarted.prefs, app.prefs);
    assert_eq!(restarted.prefs.output_folder, folder);
    assert!(!restarted.prefs.privacy_mode);
    assert_eq!(
        restarted.prefs.negative_prompt,
        "  artifacts\nextra fingrar  "
    );
}

#[test]
fn privacy_reveals_only_hovered_images_in_a_focused_window() {
    for size in [egui::vec2(64., 48.), egui::vec2(320., 180.)] {
        let ctx = egui::Context::default();
        let sample = Preview {
            width: 2,
            height: 2,
            pixels: [255, 0, 0, 255].repeat(4),
        };
        let image = texture(&ctx, &sample, "private-test");
        let other = texture(&ctx, &sample, "other-private-test");
        let inside = egui::pos2(24., 24.);
        let outside = egui::pos2(600., 400.);
        for (name, events, focused, clipped, expected) in [
            ("initial", vec![], true, false, false),
            (
                "enter",
                vec![egui::Event::PointerMoved(inside)],
                true,
                false,
                true,
            ),
            (
                "leave",
                vec![egui::Event::PointerMoved(outside)],
                true,
                false,
                false,
            ),
            (
                "re-enter",
                vec![egui::Event::PointerMoved(inside)],
                true,
                false,
                true,
            ),
            (
                "leave window",
                vec![egui::Event::PointerGone],
                true,
                false,
                false,
            ),
            (
                "return",
                vec![egui::Event::PointerMoved(inside)],
                true,
                false,
                true,
            ),
            ("lose focus", vec![], false, false, false),
            ("regain focus", vec![], true, false, true),
            ("clipped area", vec![], true, true, false),
        ] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800., 600.),
                    )),
                    focused,
                    events,
                    ..Default::default()
                },
                |ui| {
                    if clipped {
                        ui.set_clip_rect(egui::Rect::from_min_max(
                            egui::pos2(100., 100.),
                            egui::pos2(800., 600.),
                        ));
                    }
                    private_image(ui, &image, size, true);
                    private_image(ui, &other, size, true);
                },
            );
            output.textures_delta.clear();
            let meshes = ctx.tessellate(output.shapes, output.pixels_per_point);
            let draws_texture = |id| {
                meshes.iter().any(|mesh| {
                    matches!(&mesh.primitive, egui::epaint::Primitive::Mesh(mesh) if mesh.texture_id == id)
                })
            };
            assert_eq!(draws_texture(image.id()), expected, "{name} at {size:?}");
            assert!(!draws_texture(other.id()), "neighbor revealed: {name}");
        }
    }
}

#[test]
fn headless_layout_and_recoverable_job_state() {
    let root = tempfile::tempdir().unwrap();
    for (width, height, name) in [
        (1180, 760, "desktop"),
        (640, 900, "compact"),
        (1180, 760, "progress"),
        (640, 900, "progress-compact"),
    ] {
        let ctx = egui::Context::default();
        helto::install(&ctx);
        let mut app = ImagineApp::empty(
            root.path().into(),
            Connection {
                url: "http://127.0.0.1:8080".into(),
            },
            Preferences {
                output_folder: PathBuf::from("/Pictures/Imagine"),
                ..Preferences::load(root.path()).unwrap()
            },
        );
        app.client = Some(Client::connect(&app.connection).unwrap());
        app.image.prompt="A quiet Scandinavian harbor at sunrise, warm light on the waterfront, realistic photography.".into();
        app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
        let sample = Preview {
            width: 160,
            height: 90,
            pixels: (0..160 * 90)
                .flat_map(|i| {
                    let y = i / 160;
                    [60 + y as u8, 70 + y as u8, 92 + y as u8, 255]
                })
                .collect(),
        };
        app.image.references.push(ReferenceView {
            number: 0,
            id: "synthetic-reference".into(),
            name: "reference.png".into(),
            texture: texture(&ctx, &sample, "synthetic-reference"),
        });
        app.image.result = Some(Results {
            assets: vec![ResultView {
                asset_id: "asset_synthetic".into(),
                role: OutputRole::Image,
                texture: texture(&ctx, &sample, "synthetic-result"),
                file: PathBuf::from("/Pictures/Imagine/imagine-example.png"),
            }],
            seed: Some(42),
        });
        app.image.status = "Saved".into();
        if name.starts_with("progress") {
            app.image.working = true;
            app.image.status = "Sampling".into();
            app.image.sampling = Some(client::SamplingProgress {
                completed: 12,
                total: 40,
                remaining_seconds: 25.,
                updated_at_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            });
        }
        let mut combined = egui::FullOutput::default();
        for _ in 0..3 {
            let frame = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            combined.append(frame);
        }
        assert!(!combined.shapes.is_empty());
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            render(&ctx, combined, width, height)
                .save(dir.join(format!("{name}.png")))
                .unwrap();
        } else {
            combined.textures_delta.clear();
        }
        // A retrieval/save failure keeps the original job and idempotency key.
        app.image.pending = Some(Submission {
            upscale_source: None,
            request: serde_json::json!({}),
            key: "original-key".into(),
            folder: root.path().into(),
            job: Some("job_original".into()),
        });
        app.image.working = true;
        app.tx
            .send(Message::Finished(
                Workspace::Image,
                Err(Failure {
                    message: "disk full".into(),
                    retryable: true,
                }),
            ))
            .unwrap();
        app.receive(&ctx);
        assert!(!app.image.working);
        assert!(app.image.sampling.is_none());
        assert_eq!(
            app.image.pending.as_ref().unwrap().job.as_deref(),
            Some("job_original")
        );
        app.tx
            .send(Message::Finished(
                Workspace::Image,
                Err(Failure {
                    message: "cancelled".into(),
                    retryable: false,
                }),
            ))
            .unwrap();
        app.receive(&ctx);
        assert!(app.image.pending.is_none());
    }
}

// Minimal CPU rasterization of egui's own triangle output for inspection.
// This is not a substitute for native compositor/font-renderer acceptance.
pub(super) fn render(
    ctx: &egui::Context,
    mut output: egui::FullOutput,
    width: u32,
    height: u32,
) -> image::RgbaImage {
    let mut textures: HashMap<egui::TextureId, egui::ColorImage> = HashMap::new();
    for (id, deltas) in std::mem::take(&mut output.textures_delta.set) {
        for delta in deltas {
            let egui::ImageData::Color(data) = delta.image;
            if let Some([x, y]) = delta.pos {
                let target = textures.get_mut(&id).unwrap();
                for row in 0..data.size[1] {
                    for col in 0..data.size[0] {
                        target.pixels[(y + row) * target.size[0] + x + col] =
                            data.pixels[row * data.size[0] + col];
                    }
                }
            } else {
                textures.insert(id, (*data).clone());
            }
        }
    }
    let mut image =
        image::RgbaImage::from_pixel(width, height, image::Rgba(helto::SURFACE.to_array()));
    for primitive in ctx.tessellate(output.shapes, output.pixels_per_point) {
        let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
            continue;
        };
        let tex = &textures[&mesh.texture_id];
        for tri in mesh.indices.chunks_exact(3) {
            let v = [
                mesh.vertices[tri[0] as usize],
                mesh.vertices[tri[1] as usize],
                mesh.vertices[tri[2] as usize],
            ];
            let cross = |a: egui::Vec2, b: egui::Vec2| a.x * b.y - a.y * b.x;
            let area = cross(v[1].pos - v[0].pos, v[2].pos - v[0].pos);
            if area.abs() < 0.0001 {
                continue;
            }
            let bounds = egui::Rect::from_points(&v.map(|v| v.pos))
                .intersect(primitive.clip_rect)
                .intersect(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width as f32, height as f32),
                ));
            for y in
                bounds.min.y.ceil().max(0.) as u32..bounds.max.y.ceil().min(height as f32) as u32
            {
                for x in
                    bounds.min.x.ceil().max(0.) as u32..bounds.max.x.ceil().min(width as f32) as u32
                {
                    let p = egui::pos2(x as f32 + 0.5, y as f32 + 0.5);
                    let w = [
                        cross(v[1].pos - p, v[2].pos - p) / area,
                        cross(v[2].pos - p, v[0].pos - p) / area,
                        cross(v[0].pos - p, v[1].pos - p) / area,
                    ];
                    if w.iter().any(|v| *v < -0.0001) {
                        continue;
                    }
                    let uv = v[0].uv.to_vec2() * w[0]
                        + v[1].uv.to_vec2() * w[1]
                        + v[2].uv.to_vec2() * w[2];
                    let tx = (uv.x * tex.size[0] as f32 - 0.5).clamp(0., (tex.size[0] - 1) as f32);
                    let ty = (uv.y * tex.size[1] as f32 - 0.5).clamp(0., (tex.size[1] - 1) as f32);
                    let ix = tx.floor() as usize;
                    let iy = ty.floor() as usize;
                    let fx = tx - ix as f32;
                    let fy = ty - iy as f32;
                    let samples = [
                        (ix, iy, (1. - fx) * (1. - fy)),
                        ((ix + 1).min(tex.size[0] - 1), iy, fx * (1. - fy)),
                        (ix, (iy + 1).min(tex.size[1] - 1), (1. - fx) * fy),
                        (
                            (ix + 1).min(tex.size[0] - 1),
                            (iy + 1).min(tex.size[1] - 1),
                            fx * fy,
                        ),
                    ];
                    let mut color = [0.; 4];
                    for (channel, value) in color.iter_mut().enumerate() {
                        let pixel: f32 = samples
                            .iter()
                            .map(|(x, y, k)| {
                                tex.pixels[y * tex.size[0] + x].to_array()[channel] as f32 * k
                            })
                            .sum();
                        let vertex: f32 = (0..3)
                            .map(|i| v[i].color.to_array()[channel] as f32 * w[i])
                            .sum();
                        *value = pixel * vertex / 255.;
                    }
                    let dst = image.get_pixel_mut(x, y);
                    let alpha = color[3] / 255.;
                    for channel in 0..3 {
                        dst[channel] = (color[channel] + dst[channel] as f32 * (1. - alpha))
                            .clamp(0., 255.) as u8;
                    }
                    dst[3] = 255;
                }
            }
        }
    }
    output.textures_delta.clear();
    image
}

fn sheet_result(folder: &std::path::Path) -> Completed {
    Completed {
        assets: [
            (OutputRole::Sheet, 1536, 864),
            (OutputRole::Portrait, 704, 864),
            (OutputRole::Views, 832, 864),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (role, width, height))| client::CompletedAsset {
            asset_id: format!("asset_{}", role.name()),
            role,
            file: folder.join(format!("{}.png", role.name())),
            preview: Preview {
                width,
                height,
                pixels: [90 + i as u8 * 30, 100, 130, 255].repeat(width * height),
            },
        })
        .collect(),
        seed: Some(42),
    }
}

#[test]
fn sheet_tab_preserves_image_draft_and_routes_async_results_to_origin() {
    let mut desk = Desk::new();
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    let prompt = desk.app.image.prompt.clone();
    let image_path = desk.app.image.selected_result().unwrap().file.clone();
    let settings = desk.app.prefs.generation.clone();
    desk.click("Character sheets");
    assert_eq!(desk.app.workspace, Workspace::CharacterSheet);
    assert!(desk.control("Generate").is_disabled());
    assert!(desk.control("Ratio").is_disabled());
    desk.click("Width");
    desk.frame(vec![
        egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::CTRL,
        },
        egui::Event::Text("2048".into()),
        egui::Event::Key {
            key: egui::Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    assert_eq!(desk.app.prefs.sheet_width, 2048);
    assert_eq!(
        Preferences::load(desk._root.path()).unwrap().sheet_width,
        2048
    );
    assert_eq!(desk.app.prefs.generation, settings);
    desk.app.sheet.character_description = "SYNTHETIC CHARACTER".into();
    desk.app.sheet.clothing_description = "SYNTHETIC CLOTHING".into();
    desk.frame(vec![]);
    assert!(!desk.control("Generate").is_disabled());
    desk.app.sheet.pending = Some(Submission {
        upscale_source: None,
        request: serde_json::json!({"operation":"character_sheet"}),
        key: "sheet-key".into(),
        folder: desk._root.path().into(),
        job: None,
    });
    desk.app.sheet.working = true;
    desk.click("Image");
    // Each tab has its own desk: a running character sheet leaves Image usable.
    assert!(!desk.control("Add to queue").is_disabled());
    desk.app
        .tx
        .send(Message::Progress(
            Workspace::CharacterSheet,
            Progress::Submitted("sheet-job".into()),
        ))
        .unwrap();
    desk.app
        .tx
        .send(Message::Progress(
            Workspace::CharacterSheet,
            Progress::Status {
                label: "Views · Sampling".into(),
                sampling: None,
            },
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(
        desk.app.sheet.pending.as_ref().unwrap().job.as_deref(),
        Some("sheet-job")
    );
    assert_eq!(desk.app.sheet.status, "Views · Sampling");
    assert_eq!(desk.app.image.status, "Saved");
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::CharacterSheet,
            Ok(sheet_result(desk._root.path())),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(desk.app.image.selected_result().unwrap().file, image_path);
    assert_eq!(desk.app.image.prompt, prompt);
    assert_eq!(desk.app.prefs.generation, settings);
    assert_eq!(desk.app.sheet.status, "Saved");
    desk.click("Character sheets");
    assert_eq!(desk.app.sheet.result.as_ref().unwrap().assets.len(), 3);
    assert_eq!(
        desk.app.sheet.selected_result().unwrap().role,
        OutputRole::Sheet
    );
    desk.click("Actual pixels");
    desk.app.sheet.image_pan = egui::vec2(20., 20.);
    desk.click("Portrait");
    assert_eq!(
        desk.app.sheet.selected_result().unwrap().role,
        OutputRole::Portrait
    );
    assert!(!desk.app.sheet.actual_pixels);
    assert_eq!(desk.app.sheet.image_pan, egui::Vec2::ZERO);
    desk.click("Views");
    assert_eq!(
        desk.app.sheet.selected_result().unwrap().role,
        OutputRole::Views
    );
    let again = desk.app.upscale_submission().unwrap();
    assert_eq!(again.request["profile"], "image_upscale");
    assert_ne!(again.key, "sheet-key");
    assert!(again.job.is_none());
    desk.click("Image");
    assert_eq!(desk.app.image.references.len(), 2);
    assert_eq!(desk.app.image.prompt, prompt);
}

#[test]
fn sheet_upload_limits_retry_cancel_and_capability_guard() {
    let mut desk = Desk::new();
    desk.click("Character sheets");
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    desk.app.sheet.uploading = true;
    desk.click("Image");
    desk.app
        .tx
        .send(Message::Uploaded(
            Workspace::CharacterSheet,
            Ok(Reference {
                id: "sheet-ref".into(),
                name: "synthetic.png".into(),
                preview: Preview {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0, 255],
                },
            }),
        ))
        .unwrap();
    desk.app
        .tx
        .send(Message::UploadsFinished(Workspace::CharacterSheet))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(desk.app.image.references.len(), 2);
    assert_eq!(desk.app.sheet.references.len(), 1);
    assert!(!desk.app.sheet.uploading);
    desk.click("Character sheets");
    assert!(
        !desk.control("Generate").is_disabled(),
        "reference alone is sufficient"
    );
    desk.app
        .tx
        .send(Message::Picked(
            Workspace::CharacterSheet,
            vec![PathBuf::from("unused"); 8],
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert!(desk.app.sheet.error.as_deref().unwrap().contains("Up to 8"));
    assert_eq!(desk.app.sheet.references.len(), 1);
    desk.app.sheet.pending = Some(Submission {
        upscale_source: None,
        request: serde_json::json!({"operation":"character_sheet"}),
        key: "retry-sheet".into(),
        folder: desk._root.path().into(),
        job: Some("sheet-job".into()),
    });
    desk.app.sheet.working = true;
    desk.frame(vec![]);
    assert!(!desk.control("Cancel generation").is_disabled());
    desk.click("Image");
    assert!(desk.control("Cancel generation").is_disabled());
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::CharacterSheet,
            Err(Failure {
                message: "Save failed".into(),
                retryable: true,
            }),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(
        desk.app.sheet.pending.as_ref().unwrap().job.as_deref(),
        Some("sheet-job")
    );
    assert!(desk.app.image.pending.is_none());
    desk.click("Character sheets");
    assert!(!desk.control("Retry").is_disabled());
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::CharacterSheet,
            Err(Failure {
                message: "Generation cancelled.".into(),
                retryable: false,
            }),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert!(desk.app.sheet.pending.is_none());
    desk.app.metadata = Some(serde_json::json!({"operations":["text_to_image","image_edit"]}));
    desk.frame(vec![]);
    assert!(desk.control("Generate").is_disabled());
}

#[test]
fn headless_character_sheet_layout_and_privacy() {
    for (name, width, height, role, hidden, progress) in [
        ("sheet-desktop", 1280, 800, OutputRole::Sheet, false, false),
        (
            "sheet-portrait",
            1280,
            800,
            OutputRole::Portrait,
            false,
            false,
        ),
        ("sheet-views", 1280, 800, OutputRole::Views, false, false),
        ("sheet-narrow", 600, 1000, OutputRole::Sheet, false, false),
        (
            "sheet-stacked-tools",
            920,
            800,
            OutputRole::Sheet,
            false,
            false,
        ),
        ("sheet-progress", 1280, 800, OutputRole::Sheet, false, true),
        ("sheet-hidden", 1280, 800, OutputRole::Sheet, true, false),
    ] {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.workspace = Workspace::CharacterSheet;
        app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
        app.prefs.privacy_mode = hidden;
        app.sheet.character_description = "SYNTHETIC CHARACTER".into();
        app.sheet.clothing_description = "SYNTHETIC CLOTHING".into();
        app.tx
            .send(Message::Finished(
                Workspace::CharacterSheet,
                Ok(sheet_result(root.path())),
            ))
            .unwrap();
        app.receive(&ctx);
        app.sheet.select_result(role);
        let sample = Preview {
            width: 32,
            height: 32,
            pixels: [140, 120, 100, 255].repeat(32 * 32),
        };
        app.sheet.references.push(ReferenceView {
            number: 0,
            id: "ref".into(),
            name: "synthetic-reference".into(),
            texture: texture(&ctx, &sample, "ref"),
        });
        if progress {
            app.sheet.working = true;
            app.sheet.status = "Views · Sampling".into();
            app.sheet.sampling = Some(client::SamplingProgress {
                completed: 12,
                total: 40,
                remaining_seconds: 20.,
                updated_at_ms: 1,
            });
        }
        let mut output = egui::FullOutput::default();
        for _ in 0..3 {
            output.append(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    focused: true,
                    ..Default::default()
                },
                |ui| app.content(ui),
            ));
        }
        let nodes = &output
            .platform_output
            .accesskit_update
            .as_ref()
            .unwrap()
            .nodes;
        let bounds = |label| {
            nodes
                .iter()
                .find_map(|(_, n)| (n.label() == Some(label)).then(|| n.bounds()).flatten())
                .unwrap()
        };
        assert!(
            bounds("Views").x1 < bounds("Fit image").x0
                || bounds("Views").y1 <= bounds("Fit image").y0
        );
        if hidden {
            let private_ids: Vec<_> = app
                .sheet
                .result
                .as_ref()
                .unwrap()
                .assets
                .iter()
                .map(|a| a.texture.id())
                .chain(app.sheet.references.iter().map(|r| r.texture.id()))
                .collect();
            let shapes = output.shapes.clone();
            assert!(!ctx.tessellate(shapes,output.pixels_per_point).iter().any(|p|matches!(&p.primitive,egui::epaint::Primitive::Mesh(m) if private_ids.contains(&m.texture_id))));
            fn visible_text(shape: &egui::Shape) -> bool {
                match shape {
                    egui::Shape::Text(t) => {
                        t.opacity_factor > 0. && t.galley.text().contains("SYNTHETIC")
                    }
                    egui::Shape::Vec(v) => v.iter().any(visible_text),
                    _ => false,
                }
            }
            assert!(!output.shapes.iter().any(|s| visible_text(&s.shape)));
        }
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            render(&ctx, output, width, height)
                .save(dir.join(format!("{name}.png")))
                .unwrap();
        } else {
            output.textures_delta.clear();
        }
    }
}

#[test]
fn upscale_target_settings_persist_without_changing_generation_size() {
    let mut desk = Desk::new();
    let generation = desk.app.prefs.generation.clone();
    desk.click("Settings");
    desk.click("Upscale target");
    desk.click("3K");
    assert_eq!(
        desk.app.prefs.upscale_target,
        crate::settings::UpscaleTarget::K3
    );
    assert_eq!(
        Preferences::load(desk._root.path()).unwrap().upscale_target,
        crate::settings::UpscaleTarget::K3
    );
    assert_eq!(desk.app.prefs.generation, generation);
    desk.click("Upscale target");
    desk.click("4K");
    assert_eq!(
        Preferences::load(desk._root.path()).unwrap().upscale_target,
        crate::settings::UpscaleTarget::K4
    );
}

#[test]
fn sheet_upscale_replaces_only_whole_sheet_even_when_viewing_portrait() {
    let mut desk = Desk::new();
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::CharacterSheet,
            Ok(sheet_result(desk._root.path())),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    desk.click("Character sheets");
    desk.click("Portrait");
    desk.app.prefs.generation.aspect_ratio = "3:4".into();
    desk.app.prefs.upscale_target = crate::settings::UpscaleTarget::K4;
    let submission = desk.app.upscale_submission().unwrap();
    assert_eq!(
        submission.request["inputs"]["reference_images"],
        serde_json::json!(["asset_character_sheet"])
    );
    assert_eq!(submission.request["aspect_ratio"], "16:9");
    assert_eq!(submission.request["max_side"], 4096);
    let original: Vec<_> = desk
        .app
        .sheet
        .result
        .as_ref()
        .unwrap()
        .assets
        .iter()
        .map(|a| (a.role, a.asset_id.clone(), a.file.clone()))
        .collect();
    let image_id = desk.app.image.selected_result().unwrap().asset_id.clone();
    desk.app.sheet.pending = Some(submission);
    desk.app.sheet.working = true;
    desk.click("Image");
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::CharacterSheet,
            Ok(Completed {
                assets: vec![client::CompletedAsset {
                    role: OutputRole::Image,
                    asset_id: "asset_upscaled_sheet".into(),
                    file: desk._root.path().join("upscaled.png"),
                    preview: Preview {
                        width: 2,
                        height: 1,
                        pixels: [0, 0, 0, 255].repeat(2),
                    },
                }],
                seed: Some(99),
            }),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    let result = desk.app.sheet.result.as_ref().unwrap();
    assert_eq!(result.assets.len(), 3);
    for (role, id, path) in original {
        let asset = result.assets.iter().find(|a| a.role == role).unwrap();
        if role == OutputRole::Sheet {
            assert_eq!(asset.asset_id, "asset_upscaled_sheet");
            assert_ne!(asset.file, path);
        } else {
            assert_eq!(asset.asset_id, id);
            assert_eq!(asset.file, path);
        }
    }
    assert_eq!(desk.app.image.selected_result().unwrap().asset_id, image_id);
    assert_eq!(desk.app.sheet.selected_role, OutputRole::Sheet);
    assert!(desk.app.sheet.pending.is_none());
    desk.click("Character sheets");
    assert_eq!(
        desk.app.upscale_submission().unwrap().request["inputs"]["reference_images"],
        serde_json::json!(["asset_upscaled_sheet"])
    );
}

#[path = "model_ui_tests.rs"]
mod model_ui_tests;

/// Icon sheet for visual review; written only when IMAGINE_UI_PREVIEW_DIR is set.
#[test]
fn headless_icon_sheet() {
    use crate::icons::Icon::*;
    let icons = [
        Image,
        Save,
        Edit,
        Enhance,
        Duplicate,
        Copy,
        Insert,
        Lora,
        CharacterSheet,
        Video,
        Audio,
        Settings,
        Eye,
        EyeOff,
        Plus,
        Up,
        Down,
        Trash,
        Fit,
        Actual,
        Hand,
        Generate,
        Upscale,
        Retry,
        Stop,
        Folder,
        Close,
        Check,
        Undo,
        Redo,
        Compare,
        Swap,
        Unlink,
        Animate,
        Play,
        Pause,
        Story,
        Montage,
        Review,
        Link,
        Menu,
        Person,
        Library,
        Import,
        Export,
        Download,
        Back,
        Lock,
        Info,
        Scene,
        PanelRight,
        PanelBottom,
    ];
    let ctx = egui::Context::default();
    helto::install(&ctx);
    let (width, height) = (1100, 60 + 84 * icons.len().div_ceil(4) as u32);
    let mut combined = egui::FullOutput::default();
    for _ in 0..2 {
        combined.append(ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(width as f32, height as f32),
                )),
                ..Default::default()
            },
            |ui| {
                egui::Grid::new("icons").spacing([12., 10.]).show(ui, |ui| {
                    for (index, icon) in icons.iter().enumerate() {
                        let name = format!("{icon:?}");
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(72., 72.), egui::Sense::hover());
                        crate::icons::paint(ui.painter(), rect, *icon, helto::TEXT);
                        helto::icon_button(ui, *icon, &name, index % 9 == 0);
                        ui.label(name);
                        if index % 4 == 3 {
                            ui.end_row();
                        }
                    }
                });
            },
        ));
    }
    if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        render(&ctx, combined, width, height)
            .save(dir.join("icons.png"))
            .unwrap();
    } else {
        combined.textures_delta.clear();
    }
}

#[test]
fn pane_dividers_resize_save_once_settled_and_reset_on_double_click() {
    let mut desk = Desk::new();
    let prompt_width = |desk: &Desk| desk.control("Prompt").bounds().unwrap().width();
    let before = prompt_width(&desk);
    let handle = desk.control("Resize prompt panel").bounds().unwrap();
    let start = egui::pos2(
        (handle.x0 + handle.x1) as f32 / 2.,
        (handle.y0 + handle.y1) as f32 / 2.,
    );
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    desk.frame(vec![egui::Event::PointerMoved(start), button(start, true)]);
    for step in 1..=4 {
        desk.frame(vec![egui::Event::PointerMoved(
            start - egui::vec2(15. * step as f32, 0.),
        )]);
    }
    // Still dragging: the size is live but not yet saved.
    let dragged = f64::from(desk.app.prefs.layout.prompt.unwrap());
    assert!(dragged > before + 40., "{before} -> {dragged}");
    assert!(
        Preferences::load(desk._root.path())
            .unwrap()
            .layout
            .prompt
            .is_none()
    );
    let end = start - egui::vec2(60., 0.);
    desk.frame(vec![button(end, false)]);
    desk.frame(Vec::new());
    assert!(prompt_width(&desk) > before + 40.);
    assert_eq!(
        Preferences::load(desk._root.path()).unwrap().layout.prompt,
        desk.app.prefs.layout.prompt
    );
    // Double-click restores the default width.
    let handle = desk.control("Resize prompt panel").bounds().unwrap();
    let pos = egui::pos2(
        (handle.x0 + handle.x1) as f32 / 2.,
        (handle.y0 + handle.y1) as f32 / 2.,
    );
    for _ in 0..2 {
        desk.frame(vec![egui::Event::PointerMoved(pos), button(pos, true)]);
        desk.frame(vec![button(pos, false)]);
    }
    desk.frame(Vec::new());
    assert!(desk.app.prefs.layout.prompt.is_none());
    assert!((prompt_width(&desk) - before).abs() < 1.);
}

#[test]
fn a_busy_image_desk_leaves_the_sheet_usable_but_blocks_editing_its_result() {
    let mut desk = Desk::new();
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    desk.app.image.working = true;
    assert_eq!(
        desk.app.edit_image_blocked(),
        Some("Wait for the current generation.")
    );
    desk.click("Character sheets");
    desk.app.sheet.character_description = "SYNTHETIC CHARACTER".into();
    desk.frame(vec![]);
    assert!(!desk.control("Generate").is_disabled());
}

#[test]
fn references_reorder_by_dragging_a_picture_onto_another_card() {
    let mut desk = Desk::new();
    desk.app.prefs.privacy_mode = false;
    desk.frame(vec![]);
    let names: Vec<String> = desk
        .app
        .image
        .references
        .iter()
        .map(|r| r.id.clone())
        .collect();
    assert!(names.len() >= 2);
    let card = |desk: &Desk, number: usize| {
        let node = desk
            .nodes
            .iter()
            .filter(|(_, n)| n.label() == Some("Move reference up"))
            .nth(number)
            .unwrap()
            .1
            .bounds()
            .unwrap();
        egui::pos2(node.x0 as f32 - 60., ((node.y0 + node.y1) / 2.) as f32)
    };
    let (from, to) = (card(&desk, 1), card(&desk, 0));
    let button = |pos, pressed| egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    };
    desk.frame(vec![egui::Event::PointerMoved(from), button(from, true)]);
    for step in 1..=6 {
        desk.frame(vec![egui::Event::PointerMoved(
            from + (to - from) * step as f32 / 6.,
        )]);
    }
    desk.frame(vec![button(to, false)]);
    desk.frame(vec![]);
    let after: Vec<String> = desk
        .app
        .image
        .references
        .iter()
        .map(|r| r.id.clone())
        .collect();
    assert_eq!(after[0], names[1], "{names:?} -> {after:?}");
    assert_eq!(after[1], names[0]);
}

#[test]
fn image_files_dropped_on_the_window_upload_as_references() {
    #[derive(Debug)]
    struct Own(PathBuf);
    impl egui::DroppedFile for Own {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
        fn bytes(&self) -> Result<Vec<u8>, String> {
            Err("not read by the UI".into())
        }
    }
    let mut desk = Desk::new();
    desk.app.image.references.clear();
    let mut output = desk.ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280., 800.),
            )),
            dropped_files: vec![
                std::sync::Arc::new(Own(desk._root.path().join("own.png"))),
                std::sync::Arc::new(Own(desk._root.path().join("notes.txt"))),
            ],
            ..Default::default()
        },
        |ui| desk.app.content(ui),
    );
    output.textures_delta.clear();
    assert!(desk.app.image.uploading);
}
