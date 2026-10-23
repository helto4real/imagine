use super::*;
use crate::model_catalog::{LibrarySnapshot, QWEN_GROUP};
use crate::settings::{LoraSetting, ModelGroupSettings};
use serde_json::json;
#[path = "../../shared/model_catalog_fixture.rs"]
mod fixture;
fn catalog(root: &std::path::Path) -> LibrarySnapshot {
    fixture::models(&root.join("models"));
    let lib = imagine_server::model_library::Library::load(
        root.join("models"),
        root.join("library.json"),
    )
    .unwrap();
    serde_json::from_value(json!(lib.snapshot())).unwrap()
}
fn install(desk: &mut Desk) {
    let library = catalog(desk._root.path());
    desk.app.accept_library(library);
}
fn selected_lora(desk: &Desk) -> LoraSetting {
    let file = desk
        .app
        .models
        .catalog
        .as_ref()
        .unwrap()
        .files
        .iter()
        .find(|f| f.path == "loras/nested/style.safetensors")
        .unwrap();
    LoraSetting {
        id: file.id.clone(),
        name: file.path.clone(),
        strength: 0.65,
        enabled: true,
    }
}
#[test]
fn model_and_lora_settings_survive_restart_empty_lists_and_group_switches() {
    let mut desk = Desk::new();
    install(&mut desk);
    let lora = selected_lora(&desk);
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras = vec![lora.clone()];
    desk.app.prefs.model_groups.insert(
        "future-group".into(),
        ModelGroupSettings {
            variant: "future".into(),
            loras: vec![],
            ..Default::default()
        },
    );
    desk.app.save_preferences();
    let prefs = Preferences::load(desk._root.path()).unwrap();
    assert_eq!(prefs.model_groups[QWEN_GROUP].loras, vec![lora]);
    desk.app.prefs = prefs;
    assert_eq!(desk.app.model_selection().unwrap().loras.len(), 1);
    desk.app.prefs.model_group = "future-group".into();
    assert!(desk.app.model_selection().is_err());
    desk.app.prefs.model_group = QWEN_GROUP.into();
    assert_eq!(desk.app.model_selection().unwrap().loras.len(), 1);
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras
        .clear();
    desk.app.save_preferences();
    desk.app.prefs = Preferences::load(desk._root.path()).unwrap();
    desk.app.accept_library(catalog(desk._root.path()));
    assert!(desk.app.model_selection().unwrap().loras.is_empty());
}

#[test]
fn escape_cancels_component_search_before_closing_settings() {
    let mut desk = Desk::new();
    install(&mut desk);
    desk.frame(vec![]);
    desk.click("Settings");
    let bounds = desk
        .nodes
        .iter()
        .map(|(_, node)| node)
        .find(|node| {
            node.role() == egui::accesskit::Role::TextInput
                && node.labelled_by().iter().any(|label_id| {
                    desk.nodes.iter().any(|(id, label)| {
                        id == label_id
                            && (label.label() == Some("Text encoder")
                                || label.value() == Some("Text encoder"))
                    })
                })
        })
        .unwrap()
        .bounds()
        .unwrap();
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
    assert!(egui::Popup::is_any_open(&desk.ctx));
    let escape = egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    desk.frame(vec![escape.clone()]);
    assert!(desk.app.settings_open);
    assert!(!egui::Popup::is_any_open(&desk.ctx));
    desk.frame(vec![escape]);
    assert!(!desk.app.settings_open);
}
#[test]
fn variants_keep_loras_and_missing_files_do_not_silently_change_selection() {
    let mut desk = Desk::new();
    install(&mut desk);
    let lora = selected_lora(&desk);
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras = vec![lora.clone()];
    let turbo = desk
        .app
        .models
        .catalog
        .as_ref()
        .unwrap()
        .models
        .iter()
        .find(|m| m["model"] == "qwen-image-2.1-turbo")
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    desk.app.prefs.generation.sampler = Some("euler".into());
    desk.app.prefs.generation.scheduler = Some("beta".into());
    desk.app.select_model(turbo.clone());
    assert_eq!(desk.app.prefs.generation.steps, 8);
    assert_eq!(desk.app.prefs.generation.cfg_scale, 1.);
    assert!(desk.app.prefs.generation.sampler.is_none());
    assert!(desk.app.prefs.generation.scheduler.is_none());
    assert_eq!(desk.app.model_selection().unwrap().loras[0].id, lora.id);
    desk.app.workspace = Workspace::CharacterSheet;
    desk.app.sheet.character_description = "Synthetic".into();
    assert!(!desk.app.can_generate());
    desk.app.workspace = Workspace::Image;
    let mut library = desk.app.models.catalog.clone().unwrap();
    library.files.retain(|f| f.id != lora.id);
    desk.app.accept_library(library.clone());
    assert!(desk.app.model_selection().is_err());
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras[0]
        .enabled = false;
    assert!(desk.app.model_selection().unwrap().loras.is_empty());
    library.models.retain(|m| m["id"] != turbo);
    desk.app.accept_library(library);
    assert_eq!(desk.app.prefs.model_groups[QWEN_GROUP].variant, turbo);
    assert!(desk.app.model_selection().is_err());
}

#[test]
fn model_picker_roundtrip_keeps_cold_and_switched_requests_identical() {
    fn choose(desk: &mut Desk, name: &str) {
        let current = desk.app.selected_model_name();
        let bounds = desk
            .nodes
            .iter()
            .map(|(_, n)| n)
            .find(|n| {
                n.role() == egui::accesskit::Role::ComboBox && n.value() == Some(current.as_str())
            })
            .unwrap()
            .bounds()
            .unwrap();
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
        desk.frame(vec![]);
        desk.click(name);
    }
    fn request(desk: &Desk) -> Value {
        desk.app
            .apply_model_selection(
                client::make_request(
                    "Synthetic model-switch check",
                    "",
                    &[],
                    desk.app.metadata.as_ref().unwrap(),
                    &desk.app.prefs.generation,
                )
                .unwrap(),
            )
            .unwrap()
    }
    let mut cold = Desk::new();
    let mut library = catalog(cold._root.path());
    library
        .models
        .retain(|m| m["model"] == "qwen-image-2.1-turbo");
    cold.app.accept_library(library);
    cold.app.prefs.generation.random_seed = false;
    let turbo = request(&cold);
    assert_eq!(turbo["sampling"]["sampler"], "euler");
    assert_eq!(turbo["sampling"]["steps"], 8);
    assert_eq!(turbo["sampling"]["cfg_scale"], 1.);

    let mut desk = Desk::new();
    install(&mut desk);
    desk.app.prefs.generation.random_seed = false;
    desk.frame(vec![]);
    let normal = request(&desk);
    for _ in 0..2 {
        choose(&mut desk, "Qwen Image 2.1 · Turbo");
        assert_eq!(request(&desk), turbo);
        choose(&mut desk, "Qwen Image 2.1 · INT8");
        assert_eq!(request(&desk), normal);
    }
}

#[test]
fn sampling_widgets_persist_and_reach_the_server_without_overriding_turbo_sigmas() {
    let mut desk = Desk::new();
    install(&mut desk);
    desk.click("Settings");
    desk.click("Sampler");
    desk.click("euler");
    desk.click("Scheduler");
    desk.click("simple");
    desk.app.prefs = Preferences::load(desk._root.path()).unwrap();
    assert_eq!(desk.app.prefs.generation.sampler.as_deref(), Some("euler"));
    assert_eq!(
        desk.app.prefs.generation.scheduler.as_deref(),
        Some("simple")
    );
    // A catalog refresh preserves saved choices; changing models restores defaults.
    desk.app.accept_library(catalog(desk._root.path()));
    let request = desk
        .app
        .apply_model_selection(
            client::make_request(
                "Synthetic",
                "",
                &[],
                desk.app.metadata.as_ref().unwrap(),
                &desk.app.prefs.generation,
            )
            .unwrap(),
        )
        .unwrap();
    let library = imagine_server::model_library::Library::load(
        desk._root.path().join("models"),
        desk._root.path().join("library.json"),
    )
    .unwrap();
    let (resolved, _) = library.prepare(request).unwrap();
    let typed: imagine_server::contract::JobRequest = serde_json::from_value(resolved).unwrap();
    typed.validate().unwrap();
    assert_eq!(typed.sampling.sampler, "euler");
    assert_eq!(typed.sampling.scheduler, "simple");
    assert!(typed.sampling.custom_sigmas.is_empty());

    let turbo = desk
        .app
        .models
        .catalog
        .as_ref()
        .unwrap()
        .models
        .iter()
        .find(|m| m["model"] == "qwen-image-2.1-turbo")
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    desk.app.select_model(turbo);
    desk.frame(vec![]);
    desk.frame(vec![]);
    assert!(!desk.control("Sampler").is_disabled());
    assert!(desk.control("Scheduler").is_disabled());
    desk.click("Sampler");
    desk.click("euler");
    let meta = desk.app.metadata.as_ref().unwrap();
    let request =
        client::make_request("Synthetic", "", &[], meta, &desk.app.prefs.generation).unwrap();
    assert_eq!(request["sampling"]["sampler"], "euler");
    assert_eq!(request["sampling"]["scheduler"], "simple");
    assert_eq!(
        request["sampling"]["custom_sigmas"],
        meta["defaults"]["sampling"]["custom_sigmas"]
    );
    assert_eq!(request["sampling"]["steps"], 8);
    assert_eq!(request["sampling"]["cfg_scale"], 1.);
}
#[test]
fn lora_button_is_available_in_generation_and_editing_and_pending_requests_are_frozen() {
    let mut desk = Desk::new();
    install(&mut desk);
    desk.frame(vec![]);
    assert!(!desk.control("LoRAs").is_disabled());
    let lora = selected_lora(&desk);
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras = vec![lora];
    let request = desk
        .app
        .apply_model_selection(
            client::make_request(
                "Synthetic",
                "",
                &[],
                desk.app.metadata.as_ref().unwrap(),
                &desk.app.prefs.generation,
            )
            .unwrap(),
        )
        .unwrap();
    desk.app.image.pending = Some(Submission {
        request: request.clone(),
        key: "frozen".into(),
        folder: desk._root.path().into(),
        job: Some("queued".into()),
        upscale_source: None,
    });
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras
        .clear();
    assert_eq!(desk.app.image.pending.as_ref().unwrap().request, request);
    desk.frame(vec![]);
    assert!(desk.control("LoRAs").is_disabled());
    desk.app.image.pending = None;
    desk.frame(vec![]);
    desk.click("Edit image");
    assert_eq!(desk.app.workspace, Workspace::ImageEdit);
    assert!(!desk.control("LoRAs").is_disabled());
    desk.app.edit.desk.prompt = "Synthetic edit".into();
    let edit = desk
        .app
        .apply_model_selection(desk.app.edit_request().unwrap())
        .unwrap();
    assert_eq!(edit["model_selection"]["loras"], json!([]));
    assert_eq!(edit["operation"], "image_edit");
}
#[test]
fn headless_model_library_and_lora_layouts() {
    for (width, height, name) in [
        (1280, 800, "models-desktop"),
        (640, 900, "models-compact"),
        (520, 420, "loras"),
        (420, 300, "loras-compact"),
        (480, 680, "model-settings"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.accept_library(catalog(root.path()));
        let file = app
            .models
            .catalog
            .as_ref()
            .unwrap()
            .files
            .iter()
            .find(|f| f.path == "loras/nested/style.safetensors")
            .unwrap()
            .clone();
        app.prefs.model_groups.get_mut(QWEN_GROUP).unwrap().loras = vec![
            LoraSetting {
                id: file.id,
                name: file.path,
                strength: 0.8,
                enabled: true,
            },
            LoraSetting {
                id: "missing".into(),
                name: "loras/optional-detail.safetensors".into(),
                strength: 0.45,
                enabled: false,
            },
        ];
        let mut output = egui::FullOutput::default();
        for _ in 0..3 {
            output.append(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, height as f32),
                    )),
                    ..Default::default()
                },
                |ui| {
                    if name.starts_with("loras") {
                        app.loras_content(ui);
                    } else if name == "model-settings" {
                        app.settings_content(ui, false);
                    } else {
                        app.content(ui);
                    }
                },
            ));
        }
        assert!(!output.shapes.is_empty());
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
fn model_settings_keep_controls_visible_with_long_paths_and_many_scan_warnings() {
    let root = tempfile::tempdir().unwrap();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
    let mut library = catalog(root.path());
    library.settings.root =
        "/synthetic/shared/storage/very-long-model-library-path/ComfyUI/models".into();
    library.warnings = (0..40).map(|i| format!("Model header does not match the supported component: diffusion_models/synthetic-{i}.safetensors")).collect();
    let mut unsupported = library
        .files
        .iter()
        .find(|f| f.kind == "loras")
        .unwrap()
        .clone();
    unsupported.id = "unsupported".into();
    unsupported.supported = false;
    unsupported.kind = "checkpoints".into();
    unsupported.path = "checkpoints/unsupported.safetensors".into();
    library.files.push(unsupported);
    app.accept_library(library);
    let mut output = egui::FullOutput::default();
    for _ in 0..3 {
        output.append(ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(360., 680.),
                )),
                ..Default::default()
            },
            |ui| app.settings_content(ui, false),
        ));
    }
    let nodes = &output
        .platform_output
        .accesskit_update
        .as_ref()
        .unwrap()
        .nodes;
    for label in [
        "Choose models folder",
        "Refresh models",
        "Text encoder",
        "VAE",
        "Generation",
        "Scan warnings (40)",
        "Unsupported files (1)",
    ] {
        let node = nodes
            .iter()
            .map(|(_, node)| node)
            .find(|n| n.label() == Some(label) || n.value() == Some(label))
            .unwrap_or_else(|| panic!("Missing {label}"));
        let bounds = node.bounds().unwrap();
        assert!(bounds.x0 >= 20. && bounds.x1 <= 340., "{label}: {bounds:?}");
        assert!(bounds.y0 >= 80. && bounds.y1 < 400., "{label}: {bounds:?}");
    }
    assert!(!nodes.iter().any(|(_, n)| {
        n.value()
            .or(n.label())
            .is_some_and(|s| s.starts_with("Model header does not match"))
    }));
    if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        render(&ctx, output, 360, 680)
            .save(dir.join("model-settings-warnings.png"))
            .unwrap();
    } else {
        output.textures_delta.clear();
    }
}

#[test]
fn settings_scrollbar_leaves_the_privacy_toggle_visible_and_clickable() {
    for width in [360, 480] {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.global_style_mut(|style| style.animation_time = 0.);
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.accept_library(catalog(root.path()));
        let frame = |app: &mut ImagineApp, events| {
            ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(width as f32, 680.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| app.settings_content(ui, false),
            )
        };
        let mut output = egui::FullOutput::default();
        for _ in 0..3 {
            output.append(frame(&mut app, vec![]));
        }
        let nodes = &output
            .platform_output
            .accesskit_update
            .as_ref()
            .unwrap()
            .nodes;
        let scrollbar = nodes
            .iter()
            .map(|(_, n)| n)
            .find(|n| n.role() == egui::accesskit::Role::ScrollBar)
            .unwrap()
            .bounds()
            .unwrap();
        let mut first: Option<(f64, f64)> = None;
        let mut text_left: Option<f32> = None;
        for (name, value) in [
            ("Steps", "40"),
            ("CFG", "2.5"),
            ("Sampler", "er_sde"),
            ("Scheduler", "beta"),
            ("Reference size", "1.0"),
            ("Upscale target", "2K"),
        ] {
            let bounds = nodes
                .iter()
                .map(|(_, n)| n)
                .find(|n| {
                    n.labelled_by().iter().any(|label_id| {
                        nodes.iter().any(|(id, label)| {
                            label_id == id
                                && (label.label() == Some(name) || label.value() == Some(name))
                        })
                    })
                })
                .unwrap_or_else(|| panic!("Missing {name}"))
                .bounds()
                .unwrap();
            assert!(
                bounds.x1 + 16. <= scrollbar.x0,
                "{name}: {bounds:?}, {scrollbar:?}"
            );
            assert!((bounds.height() - 32.).abs() < 1., "{name}: {bounds:?}");
            if let Some((left, right)) = first {
                assert!(
                    (bounds.x0 - left).abs() < 1. && (bounds.x1 - right).abs() < 1.,
                    "{name}: {bounds:?}"
                );
            } else {
                first = Some((bounds.x0, bounds.x1));
            }
            let text = output
                .shapes
                .iter()
                .find_map(|shape| match &shape.shape {
                    egui::Shape::Text(t) if t.galley.text().trim() == value => Some(t),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("Missing rendered value: {value}"));
            if let Some(left) = text_left {
                assert!((text.pos.x - left).abs() < 1., "{name}: {:?}", text.pos);
            } else {
                text_left = Some(text.pos.x);
            }
            assert!(
                (f64::from(text.pos.y + text.galley.size().y / 2.) - (bounds.y0 + bounds.y1) / 2.)
                    .abs()
                    < 1.,
                "{name}: text {:?}, {bounds:?}",
                text.pos
            );
        }
        if width == 480
            && let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR")
        {
            std::fs::create_dir_all(&dir).unwrap();
            render(&ctx, output.clone(), width, 680)
                .save(PathBuf::from(dir).join("settings-sampling.png"))
                .unwrap();
        }
        for _ in 0..3 {
            output.append(frame(
                &mut app,
                vec![
                    egui::Event::PointerMoved(egui::pos2(200., 450.)),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Point,
                        delta: egui::vec2(0., -1000.),
                        phase: egui::TouchPhase::Move,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            ));
        }
        let nodes = &output
            .platform_output
            .accesskit_update
            .as_ref()
            .unwrap()
            .nodes;
        let toggle = nodes
            .iter()
            .map(|(_, n)| n)
            .find(|n| {
                n.role() == egui::accesskit::Role::CheckBox && n.label() == Some("Hide previews")
            })
            .unwrap()
            .bounds()
            .unwrap();
        let scrollbar = nodes
            .iter()
            .map(|(_, n)| n)
            .find(|n| n.role() == egui::accesskit::Role::ScrollBar)
            .unwrap()
            .bounds()
            .unwrap();
        let folder = nodes
            .iter()
            .map(|(_, n)| n)
            .find(|n| n.label() == Some("Choose output folder"))
            .unwrap()
            .bounds()
            .unwrap();
        let right = first.unwrap().1;
        assert!(
            (toggle.x1 - right).abs() < 1. && (folder.x1 - right).abs() < 1.,
            "Mismatched right edges: {toggle:?}, {folder:?}, {right}"
        );
        assert!(
            toggle.x1 + 16. <= scrollbar.x0,
            "{width}: {toggle:?} overlaps {scrollbar:?}"
        );
        assert!(toggle.y0 >= 80. && toggle.y1 <= 632., "{toggle:?}");
        let pos = egui::pos2(toggle.x1 as f32 - 1., (toggle.y0 + toggle.y1) as f32 / 2.);
        let before = app.prefs.privacy_mode;
        for pressed in [true, false] {
            output.append(frame(
                &mut app,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers::NONE,
                    },
                ],
            ));
        }
        assert_eq!(app.prefs.privacy_mode, !before);
        assert_eq!(
            Preferences::load(root.path()).unwrap().privacy_mode,
            !before
        );
        if width == 480
            && let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR")
        {
            render(&ctx, output, width, 680)
                .save(PathBuf::from(dir).join("settings-privacy-scrolled.png"))
                .unwrap();
        } else {
            output.textures_delta.clear();
        }
    }
}

#[test]
fn lora_dialog_add_remove_and_close_persist_through_real_widgets() {
    let mut desk = Desk::new();
    install(&mut desk);
    desk.frame(vec![]);
    desk.click("LoRAs");
    assert!(desk.app.models.open);
    desk.click("Add loras/nested/style.safetensors");
    assert_eq!(
        Preferences::load(desk._root.path()).unwrap().model_groups[QWEN_GROUP]
            .loras
            .len(),
        1
    );
    desk.click("Remove loras/nested/style.safetensors");
    assert!(
        Preferences::load(desk._root.path()).unwrap().model_groups[QWEN_GROUP]
            .loras
            .is_empty()
    );
    desk.click("Close LoRAs");
    assert!(!desk.app.models.open);
}

#[test]
fn lora_picker_lists_unknown_and_other_metadata_without_assignment() {
    let mut desk = Desk::new();
    let long_path =
        "loras/styles/portraits/deeply_nested/a-very-long-user-chosen-adapter-filename.safetensors";
    let path = desk._root.path().join("models").join(long_path);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b"synthetic uninspected contents").unwrap();
    install(&mut desk);
    desk.frame(vec![]);
    desk.click("LoRAs");
    desk.click("Add loras/other.safetensors");
    desk.click("Add loras/unknown.safetensors");
    desk.click("Add LoRA");
    desk.frame(vec![egui::Event::Text("deeply_nested".into())]);
    desk.frame(vec![]);
    let add = format!("Add {long_path}");
    assert!(
        desk.control(&add).bounds().unwrap().x1 <= desk.control("Close LoRAs").bounds().unwrap().x1
    );
    desk.click(&add);
    let selection = desk.app.model_selection().unwrap();
    assert_eq!(selection.loras.len(), 3);
    let prefs = Preferences::load(desk._root.path()).unwrap();
    assert_eq!(prefs.model_groups[QWEN_GROUP].loras.len(), 3);
    assert_eq!(
        prefs.model_groups[QWEN_GROUP].loras[0].name,
        "loras/other.safetensors"
    );
}

#[test]
fn lora_search_filters_each_keystroke_across_words_and_path_segments() {
    let mut desk = Desk::new();
    for path in [
        "loras/search-tests/qwen/somemoreword-style.safetensors",
        "loras/search-tests/qwen/landscape.safetensors",
        "loras/search-tests/flux/somemoreword-style.safetensors",
    ] {
        fixture::write(&desk._root.path().join("models"), path, json!({}));
    }
    let mut library = catalog(desk._root.path());
    library
        .files
        .retain(|f| f.kind != "loras" || f.path.contains("search-tests"));
    desk.app.accept_library(library);
    desk.frame(vec![]);
    desk.click("LoRAs");
    desk.click("Add LoRA");
    let results = |desk: &Desk| -> Vec<String> {
        let mut paths: Vec<_> = desk
            .nodes
            .iter()
            .filter_map(|(_, node)| node.label())
            .filter(|label| label.starts_with("Add loras/"))
            .map(str::to_owned)
            .collect();
        paths.sort();
        paths
    };
    assert_eq!(results(&desk).len(), 3);
    desk.frame(vec![egui::Event::Text("qwen".into())]);
    assert_eq!(results(&desk).len(), 2);
    desk.frame(vec![egui::Event::Text(" somemoreword".into())]);
    assert_eq!(
        results(&desk),
        vec!["Add loras/search-tests/qwen/somemoreword-style.safetensors"]
    );
    for (query, count) in [("MOREWORD   QWEN", 1), ("qwen nonexistent", 0), ("  ", 3)] {
        desk.frame(vec![
            egui::Event::Key {
                key: egui::Key::A,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers {
                    ctrl: true,
                    command: true,
                    ..Default::default()
                },
            },
            egui::Event::Text(query.into()),
        ]);
        assert_eq!(results(&desk).len(), count, "{query}");
    }
    assert!(desk.app.model_selection().unwrap().loras.is_empty());
}

#[test]
fn moss_model_switch_uses_its_group_defaults_and_preserves_qwen_loras() {
    let mut desk = Desk::new();
    install(&mut desk);
    let qwen = desk.app.prefs.model_groups[QWEN_GROUP].variant.clone();
    let lora = selected_lora(&desk);
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras = vec![lora.clone()];
    let mut moss = imagine_server::model_library::model_metadata(imagine_server::contract::MOSS);
    moss["id"] = json!("synthetic-moss-id");
    moss["model"] = json!(imagine_server::contract::MOSS);
    moss["name"] = json!("MOSS Image · SFT INT8");
    moss["group"] = json!(crate::model_catalog::MOSS_GROUP);
    moss["available"] = json!(true);
    desk.app.models.catalog.as_mut().unwrap().models.push(moss);
    desk.app.models.open = true;
    desk.app.select_model("synthetic-moss-id".into());
    assert_eq!(desk.app.prefs.model_group, crate::model_catalog::MOSS_GROUP);
    assert_eq!(desk.app.prefs.generation.steps, 50);
    assert_eq!(desk.app.prefs.generation.cfg_scale, 3.);
    assert!(!desk.app.models.open);
    assert!(desk.app.model_selection().unwrap().loras.is_empty());

    desk.app.select_model(qwen);
    assert_eq!(desk.app.prefs.model_group, QWEN_GROUP);
    assert_eq!(desk.app.prefs.model_groups[QWEN_GROUP].loras, vec![lora]);
}

#[test]
fn bf16_variant_switch_preserves_qwen_loras_and_request_precision() {
    let mut desk = Desk::new();
    install(&mut desk);
    let lora = selected_lora(&desk);
    desk.app
        .prefs
        .model_groups
        .get_mut(QWEN_GROUP)
        .unwrap()
        .loras = vec![lora.clone()];
    fixture::write(
        &desk._root.path().join("models"),
        "diffusion_models/nested/qwen_image_2.1_bf16.safetensors",
        json!({"txt_in.text_norm.weight":fixture::tensor(&[4096]),"img_in.weight":fixture::tensor(&[4096,64])}),
    );
    let library = catalog(desk._root.path());
    let bf16 = library
        .models
        .iter()
        .find(|m| m["model"] == imagine_server::contract::BF16)
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    desk.app.accept_library(library);
    desk.app.select_model(bf16);
    assert_eq!(desk.app.prefs.model_group, QWEN_GROUP);
    assert_eq!(desk.app.prefs.model_groups[QWEN_GROUP].loras, vec![lora]);
    let meta = desk.app.metadata.as_ref().unwrap();
    let mut request = client::make_request(
        "Synthetic BF16 control",
        "",
        &[],
        meta,
        &desk.app.prefs.generation,
    )
    .unwrap();
    assert_eq!(request["model"], imagine_server::contract::BF16);
    assert_eq!(request["sampling"]["steps"], 40);
    assert_eq!(request["sampling"]["cfg_scale"], 2.5);
    request["model_selection"] = json!(desk.app.model_selection().unwrap());
    let lib = imagine_server::model_library::Library::load(
        desk._root.path().join("models"),
        desk._root.path().join("synthetic-bf16-library.json"),
    )
    .unwrap();
    let (resolved, _) = lib.prepare(request).unwrap();
    let typed: imagine_server::contract::JobRequest = serde_json::from_value(resolved).unwrap();
    typed.validate().unwrap();
    assert!(
        typed
            .model_options
            .generation(&typed.model)
            .unwrap()
            .diffusion_model
            .ends_with("qwen_image_2.1_bf16.safetensors")
    );
}
