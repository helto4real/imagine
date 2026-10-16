//! Synthetic editing fixtures; no native window, live services or model.
use super::*;

fn upload(desk: &mut Desk, id: &str) {
    desk.app
        .tx
        .send(Message::Uploaded(
            Workspace::ImageEdit,
            Ok(Reference {
                id: id.into(),
                name: format!("{id}.png"),
                preview: Preview {
                    width: 32,
                    height: 32,
                    pixels: vec![128; 32 * 32 * 4],
                },
            }),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
}
fn completed(desk: &Desk, id: &str) -> Completed {
    Completed {
        assets: vec![client::CompletedAsset {
            asset_id: id.into(),
            role: OutputRole::Image,
            file: desk._root.path().join(format!("{id}.png")),
            preview: Preview {
                width: 512,
                height: 288,
                pixels: vec![128; 512 * 288 * 4],
            },
        }],
        seed: Some(7),
    }
}
fn enter(desk: &mut Desk) {
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    desk.click("Edit image");
    assert_eq!(desk.app.workspace, Workspace::ImageEdit);
}

#[test]
fn edit_ui_preserves_generation_draft_and_routes_uploads_across_tabs() {
    let mut desk = Desk::new();
    let prompt = desk.app.image.prompt.clone();
    enter(&mut desk);
    assert!(desk.app.edit.desk.prompt.is_empty());
    assert!(desk.app.edit.desk.references.is_empty());
    assert!(desk.control("Create version").is_disabled());
    upload(&mut desk, "a");
    upload(&mut desk, "b");
    desk.frame(vec![]);
    desk.click("Remove reference 1");
    desk.click("Character sheets");
    upload(&mut desk, "c");
    assert!(desk.app.sheet.references.is_empty());
    assert_eq!(desk.app.image.references.len(), 2);
    desk.click("Image");
    assert_eq!(desk.app.workspace, Workspace::ImageEdit);
    assert_eq!(
        desk.app
            .edit
            .desk
            .references
            .iter()
            .map(|r| r.number)
            .collect::<Vec<_>>(),
        [2, 3]
    );
    desk.app.edit.desk.prompt = "Use Reference 3".into();
    let request = desk.app.edit_request().unwrap();
    assert_eq!(
        request["inputs"]["reference_images"],
        serde_json::json!(["asset_synthetic", "b", "c"])
    );
    assert!(
        request["prompt"]
            .as_str()
            .unwrap()
            .contains("Reference 3 / referens 3 = <image3>")
    );
    desk.app.edit.desk.prompt = "Use Reference 1".into();
    assert!(
        desk.app
            .edit_request()
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    desk.click("Finish editing");
    assert_eq!(desk.app.workspace, Workspace::Image);
    assert_eq!(desk.app.image.prompt, prompt);
    desk.click("Edit image");
    assert_eq!(desk.app.edit.desk.references.len(), 2);
    assert_eq!(desk.app.edit.desk.prompt, "Use Reference 1");
}

#[test]
fn edit_versions_keep_original_and_selected_version_becomes_source() {
    let mut desk = Desk::new();
    enter(&mut desk);
    let original = desk.app.edit.desk.selected_result().unwrap().file.clone();
    desk.app.edit.desk.prompt = "Adjust lighting".into();
    desk.click("Character sheets");
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::ImageEdit,
            Ok(completed(&desk, "v1")),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert!(desk.app.sheet.result.is_none());
    assert_eq!(desk.app.image.selected_result().unwrap().file, original);
    desk.click("Image");
    assert_eq!(desk.app.edit.versions.len(), 2);
    assert_eq!(desk.app.edit.versions[0].result.assets[0].file, original);
    assert_eq!(
        desk.app.edit_request().unwrap()["inputs"]["reference_images"][0],
        "v1"
    );
    desk.app.select_edit_version(0);
    assert_eq!(
        desk.app.edit_request().unwrap()["inputs"]["reference_images"][0],
        "asset_synthetic"
    );
    desk.app.select_edit_version(1);
    desk.frame(vec![]);
    desk.click("Finish editing");
    assert_eq!(desk.app.image.selected_result().unwrap().asset_id, "v1");
    desk.click("Edit image");
    assert_eq!(desk.app.edit.versions.len(), 2);
    desk.click("Finish editing");
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::Image,
            Ok(completed(&desk, "fresh")),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert!(desk.app.edit.versions.is_empty());
    desk.app.start_editing();
    assert_eq!(desk.app.edit.versions[0].result.assets[0].asset_id, "fresh");
}

#[test]
fn edit_comparison_follows_source_when_versions_branch() {
    let mut desk = Desk::new();
    enter(&mut desk);
    assert!(desk.app.edit_comparison_source().is_none());
    for id in ["v1", "v2"] {
        desk.app
            .tx
            .send(Message::Finished(
                Workspace::ImageEdit,
                Ok(completed(&desk, id)),
            ))
            .unwrap();
        desk.app.receive(&desk.ctx);
    }
    assert_eq!(desk.app.edit_comparison_source().unwrap().asset_id, "v1");
    desk.app.select_edit_version(0);
    assert!(desk.app.edit_comparison_source().is_none());
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::ImageEdit,
            Ok(completed(&desk, "branch")),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(
        desk.app.edit_comparison_source().unwrap().asset_id,
        "asset_synthetic"
    );
    desk.app.select_edit_version(2);
    assert_eq!(desk.app.edit_comparison_source().unwrap().asset_id, "v1");
    desk.app.workspace = Workspace::CharacterSheet;
    assert!(desk.app.edit_comparison_source().is_none());
    desk.app.workspace = Workspace::ImageEdit;
    desk.app.finish_editing();
    assert!(desk.app.edit_comparison_source().is_none());
}

#[test]
fn comparison_clips_both_images_at_pointer_and_respects_focus_privacy_and_zoom() {
    // Procedural solid pixels distinguish the two sides, including exported CPU evidence.
    for (name, pointer, private, focused, actual, scale, expected_split) in [
        (
            "split",
            Some(egui::pos2(220., 220.)),
            false,
            true,
            false,
            1.,
            true,
        ),
        (
            "split-right",
            Some(egui::pos2(480., 220.)),
            false,
            true,
            false,
            1.,
            true,
        ),
        (
            "split-private",
            Some(egui::pos2(220., 220.)),
            true,
            true,
            false,
            1.,
            true,
        ),
        (
            "split-pan",
            Some(egui::pos2(220., 220.)),
            false,
            true,
            true,
            1.,
            true,
        ),
        (
            "split-hidpi",
            Some(egui::pos2(220., 220.)),
            false,
            true,
            true,
            2.,
            true,
        ),
        ("split-left-image", None, false, true, false, 1., false),
        (
            "split-letterbox",
            Some(egui::pos2(220., 45.)),
            false,
            true,
            false,
            1.,
            false,
        ),
        (
            "split-unfocused",
            Some(egui::pos2(220., 220.)),
            false,
            false,
            false,
            1.,
            false,
        ),
        ("split-hidden", None, true, true, false, 1., false),
        (
            "split-hidden-unfocused",
            Some(egui::pos2(220., 220.)),
            true,
            false,
            false,
            1.,
            false,
        ),
    ] {
        let ctx = egui::Context::default();
        helto::install(&ctx);
        ctx.set_pixels_per_point(scale);
        let before = texture(
            &ctx,
            &Preview {
                width: 1024,
                height: 576,
                pixels: [180, 70, 50, 255].repeat(1024 * 576),
            },
            "before",
        );
        let after = texture(
            &ctx,
            &Preview {
                width: 1024,
                height: 576,
                pixels: [40, 110, 180, 255].repeat(1024 * 576),
            },
            "after",
        );
        let mut combined = egui::FullOutput::default();
        let mut pan = egui::vec2(40., -30.);
        let rect = egui::Rect::from_min_size(egui::pos2(20., 40.), egui::vec2(600., 360.));
        for _ in 0..3 {
            combined.append(ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(640., 440.),
                    )),
                    focused,
                    events: vec![
                            pointer
                                .map(egui::Event::PointerMoved)
                                .unwrap_or(egui::Event::PointerGone),
                        ],
                    ..Default::default()
                },
                |ui| {
                    ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                        image_view_layers(
                            ui,
                            (&after, Some(&before)),
                            rect.size(),
                            private,
                            actual,
                            true,
                            &mut pan,
                        );
                    });
                },
            ));
        }
        let primitives = ctx.tessellate(combined.shapes.clone(), combined.pixels_per_point);
        let images = primitives
            .iter()
            .filter_map(|p| match &p.primitive {
                egui::epaint::Primitive::Mesh(m)
                    if m.texture_id == before.id() || m.texture_id == after.id() =>
                {
                    Some((p.clip_rect, m))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        let old = images
            .iter()
            .find(|(_, mesh)| mesh.texture_id == before.id());
        let new = images
            .iter()
            .find(|(_, mesh)| mesh.texture_id == after.id());
        assert_eq!(old.is_some(), expected_split, "{name}");
        assert_eq!(
            new.is_some(),
            !private || (focused && pointer.is_some()),
            "{name}"
        );
        if expected_split {
            let (left, old) = old.unwrap();
            let (right, new) = new.unwrap();
            let x = pointer.unwrap().x;
            assert_eq!(left.right(), x, "{name}");
            assert_eq!(right.left(), x, "{name}");
            assert_eq!(
                old.vertices
                    .iter()
                    .map(|v| (v.pos, v.uv))
                    .collect::<Vec<_>>(),
                new.vertices
                    .iter()
                    .map(|v| (v.pos, v.uv))
                    .collect::<Vec<_>>(),
                "Both images must share zoom/pan geometry: {name}"
            );
            assert!(left.left() >= rect.left() && right.right() <= rect.right());
        }
        if name == "split"
            && let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR")
        {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            let image = render(&ctx, combined, 640, 440);
            assert_eq!(image.get_pixel(100, 220).0, [180, 70, 50, 255]);
            assert_eq!(image.get_pixel(400, 220).0, [40, 110, 180, 255]);
            image.save(dir.join("edit-comparison.png")).unwrap();
        } else {
            combined.textures_delta.clear();
        }
    }
}

#[test]
fn edit_retry_freezes_request_and_cancellation_keeps_draft_and_versions() {
    let mut desk = Desk::new();
    enter(&mut desk);
    upload(&mut desk, "ref_a");
    desk.app.edit.desk.prompt = "Use Reference 1".into();
    let request = desk.app.edit_request().unwrap();
    desk.app.edit.desk.pending = Some(Submission {
        upscale_source: None,
        request: request.clone(),
        key: "edit-key".into(),
        job: Some("edit-job".into()),
        folder: desk._root.path().into(),
    });
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::ImageEdit,
            Err(Failure {
                message: "Synthetic save error".into(),
                retryable: true,
            }),
        ))
        .unwrap();
    desk.frame(vec![]);
    for control in [
        "Create version",
        "Add references",
        "Remove reference 1",
        "Finish editing",
    ] {
        assert!(desk.control(control).is_disabled(), "{control}");
    }
    assert!(!desk.control("Retry").is_disabled());
    desk.app.edit.desk.prompt = "Next draft".into();
    desk.app.prefs.generation.steps = 12;
    desk.app.select_edit_version(0);
    let pending = desk.app.edit.desk.pending.as_ref().unwrap();
    assert_eq!(pending.request, request);
    assert_eq!(pending.key, "edit-key");
    assert_eq!(pending.job.as_deref(), Some("edit-job"));
    desk.app.edit.desk.working = true;
    desk.frame(vec![]);
    assert!(!desk.control("Cancel generation").is_disabled());
    desk.app
        .tx
        .send(Message::Cancelled(
            Workspace::ImageEdit,
            "edit-job".into(),
            Err(Failure {
                message: "Synthetic cancellation failure".into(),
                retryable: true,
            }),
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert!(desk.app.edit.desk.pending.is_some());
    desk.app
        .tx
        .send(Message::Finished(
            Workspace::ImageEdit,
            Err(Failure {
                message: "Generation cancelled.".into(),
                retryable: false,
            }),
        ))
        .unwrap();
    desk.frame(vec![]);
    assert!(desk.app.edit.desk.pending.is_none());
    assert_eq!(desk.app.edit.desk.prompt, "Next draft");
    assert_eq!(desk.app.edit.desk.references.len(), 1);
    assert_eq!(desk.app.edit.versions.len(), 1);
    assert!(!desk.control("Finish editing").is_disabled());
}

#[test]
fn edit_upload_limit_and_failure_preserve_existing_references() {
    let mut desk = Desk::new();
    enter(&mut desk);
    for n in 1..=9 {
        upload(&mut desk, &format!("ref_{n}"));
    }
    desk.app.edit.desk.uploading = true;
    desk.app
        .tx
        .send(Message::Picked(
            Workspace::ImageEdit,
            vec![desk._root.path().join("never-opened.png")],
        ))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert!(!desk.app.edit.desk.uploading);
    assert!(desk.app.edit.desk.error.as_ref().unwrap().contains("9"));
    desk.frame(vec![]);
    assert!(desk.control("Add references").is_disabled());
    desk.app
        .tx
        .send(Message::Uploaded(
            Workspace::ImageEdit,
            Err(Failure {
                message: "Synthetic upload error".into(),
                retryable: false,
            }),
        ))
        .unwrap();
    desk.app
        .tx
        .send(Message::UploadsFinished(Workspace::ImageEdit))
        .unwrap();
    desk.app.receive(&desk.ctx);
    assert_eq!(desk.app.edit.desk.references.len(), 9);
    assert_eq!(desk.app.edit.next_reference_number, 10);
}

#[test]
fn headless_edit_layout_and_privacy() {
    let root = tempfile::tempdir().unwrap();
    for (width, height, private, name) in [
        (1280, 800, false, "edit-desktop"),
        (640, 1000, false, "edit-compact"),
        (1280, 800, true, "edit-private"),
    ] {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.start_editing();
        app.edit.desk.prompt =
            "Use Reference 1 for the lighting and Reference 3 for the composition.".into();
        let source = app.edit.desk.selected_result().unwrap().texture.clone();
        for number in [1, 3] {
            app.edit.desk.references.push(ReferenceView {
                number,
                id: format!("ref-{number}"),
                name: "synthetic.png".into(),
                texture: source.clone(),
            });
        }
        app.prefs.privacy_mode = private;
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
                |ui| app.content(ui),
            ));
        }
        let nodes = &combined
            .platform_output
            .accesskit_update
            .as_ref()
            .unwrap()
            .nodes;
        for name in [
            "Add references",
            "Remove reference 1",
            "Remove reference 3",
            "Create version",
            "Finish editing",
        ] {
            let node = nodes
                .iter()
                .map(|(_, n)| n)
                .find(|n| n.label() == Some(name))
                .unwrap_or_else(|| panic!("Missing {name}"));
            let bounds = node.bounds().unwrap();
            assert!(
                bounds.x0 >= 0. && bounds.x1 <= width as f64,
                "{name}: {bounds:?}"
            );
            assert!(
                bounds.y0 >= 0. && bounds.y1 <= height as f64,
                "{name}: {bounds:?}"
            );
        }
        let meshes = ctx.tessellate(combined.shapes.clone(), combined.pixels_per_point);
        let visible = meshes.iter().any(|p| matches!(&p.primitive, egui::epaint::Primitive::Mesh(m) if m.texture_id == source.id()));
        assert_eq!(visible, !private, "{name}");
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
