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
    desk.app.select_model(turbo.clone());
    assert_eq!(desk.app.prefs.generation.steps, 8);
    assert_eq!(desk.app.prefs.generation.cfg_scale, 1.);
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
        (360, 680, "model-settings"),
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
