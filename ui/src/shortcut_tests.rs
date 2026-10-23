//! Headless keyboard events against synthetic UI state; no live generation.
use super::*;

fn key(key: egui::Key, modifiers: egui::Modifiers, repeat: bool) -> egui::Event {
    egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat,
        modifiers,
    }
}

fn ctrl(desk: &mut Desk, key_code: egui::Key) {
    press(desk, key_code, egui::Modifiers::CTRL);
}

fn press(desk: &mut Desk, key_code: egui::Key, modifiers: egui::Modifiers) {
    desk.frame(vec![key(key_code, modifiers, false)]);
    desk.frame(vec![egui::Event::Key {
        key: key_code,
        physical_key: None,
        pressed: false,
        repeat: false,
        modifiers,
    }]);
}

#[test]
fn shortcuts_toggle_zoom_and_persist_privacy_without_taking_button_focus() {
    let mut desk = Desk::new();
    desk.click("Fit image");
    press(&mut desk, egui::Key::Z, egui::Modifiers::NONE);
    assert!(desk.app.image.actual_pixels);
    desk.app.image.image_pan = egui::vec2(40., 20.);
    press(&mut desk, egui::Key::Z, egui::Modifiers::NONE);
    assert!(!desk.app.image.actual_pixels);
    assert_eq!(desk.app.image.image_pan, egui::Vec2::ZERO);

    for hidden in [true, false] {
        ctrl(&mut desk, egui::Key::Space);
        assert_eq!(desk.app.prefs.privacy_mode, hidden);
        assert_eq!(
            Preferences::load(desk._root.path()).unwrap().privacy_mode,
            hidden
        );
    }
    desk.frame(vec![key(egui::Key::Z, egui::Modifiers::NONE, false)]);
    assert!(desk.app.image.actual_pixels);
    desk.frame(vec![key(egui::Key::Z, egui::Modifiers::NONE, true)]);
    assert!(desk.app.image.actual_pixels);
    desk.frame(vec![egui::Event::Key {
        key: egui::Key::Z,
        physical_key: None,
        pressed: false,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]);
    press(&mut desk, egui::Key::Z, egui::Modifiers::NONE);
    ctrl(&mut desk, egui::Key::Z);
    assert!(!desk.app.image.actual_pixels);
    desk.frame(vec![key(
        egui::Key::Z,
        egui::Modifiers::CTRL | egui::Modifiers::SHIFT,
        false,
    )]);
    assert!(!desk.app.image.actual_pixels);
    desk.frame(vec![key(egui::Key::Space, egui::Modifiers::CTRL, false)]);
    assert!(desk.app.prefs.privacy_mode);
    desk.frame(vec![key(egui::Key::Space, egui::Modifiers::CTRL, true)]);
    assert!(desk.app.prefs.privacy_mode);
}

#[test]
fn shortcuts_focus_the_active_prompt_and_leave_text_editing_alone() {
    let mut desk = Desk::new();
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    desk.app.prefs.generation.steps = 0; // Any generation would fail synchronously.
    ctrl(&mut desk, egui::Key::P);
    assert!(desk.ctx.text_edit_focused());
    let prompt_id = desk.ctx.memory(|memory| memory.focused()).unwrap();
    let before = desk.app.image.prompt.clone();
    desk.frame(vec![egui::Event::Text(" keyboard input".into())]);
    assert_ne!(desk.app.image.prompt, before);
    for key_code in [egui::Key::Z, egui::Key::Space, egui::Key::Enter] {
        ctrl(&mut desk, key_code);
    }
    press(&mut desk, egui::Key::Z, egui::Modifiers::NONE);
    desk.frame(vec![egui::Event::Text("z".into())]);
    assert!(desk.app.image.prompt.ends_with('z'));
    assert!(!desk.app.image.actual_pixels);
    assert!(!desk.app.prefs.privacy_mode);
    assert!(desk.app.image.error.is_none());
    assert!(desk.app.image.pending.is_none());
    assert_eq!(desk.ctx.memory(|memory| memory.focused()), Some(prompt_id));

    desk.click("Character sheets");
    ctrl(&mut desk, egui::Key::P);
    assert!(desk.ctx.text_edit_focused());
    desk.frame(vec![egui::Event::Text("Synthetic character".into())]);
    assert_eq!(desk.app.sheet.character_description, "Synthetic character");
    desk.click("Clothing & accessories");
    assert!(desk.ctx.text_edit_focused());
    let clothing_id = desk.ctx.memory(|memory| memory.focused());
    ctrl(&mut desk, egui::Key::P);
    assert_eq!(desk.ctx.memory(|memory| memory.focused()), clothing_id);
    desk.frame(vec![egui::Event::Text("Synthetic clothing".into())]);
    assert_eq!(desk.app.sheet.clothing_description, "Synthetic clothing");

    desk.click("Image");
    desk.click("Edit image");
    ctrl(&mut desk, egui::Key::P);
    assert!(desk.ctx.text_edit_focused());
    desk.frame(vec![egui::Event::Text("Synthetic edit".into())]);
    assert_eq!(desk.app.edit.desk.prompt, "Synthetic edit");
}

#[test]
fn shortcuts_reveal_prompt_until_pointer_moves_then_restore_hover_privacy() {
    fn visible_prompt(shape: &egui::Shape) -> bool {
        match shape {
            egui::Shape::Text(text) => {
                text.opacity_factor > 0. && text.galley.text().contains("SYNTHETIC KEYBOARD")
            }
            egui::Shape::Vec(shapes) => shapes.iter().any(visible_prompt),
            _ => false,
        }
    }
    fn frame(desk: &mut Desk, events: Vec<egui::Event>, focused: bool) -> bool {
        let mut output = desk.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1280., 800.),
                )),
                events,
                focused,
                ..Default::default()
            },
            |ui| desk.app.content(ui),
        );
        output.textures_delta.clear();
        output
            .shapes
            .iter()
            .any(|shape| visible_prompt(&shape.shape))
    }
    for workspace in [
        Workspace::Image,
        Workspace::ImageEdit,
        Workspace::CharacterSheet,
    ] {
        let mut desk = Desk::new();
        desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
        match workspace {
            Workspace::ImageEdit => desk.click("Edit image"),
            Workspace::CharacterSheet => desk.click("Character sheets"),
            _ => (),
        }
        let field = if workspace == Workspace::CharacterSheet {
            desk.app.sheet.character_description = "SYNTHETIC KEYBOARD character".into();
            "Character"
        } else {
            desk.app.desk_mut().prompt = "SYNTHETIC KEYBOARD prompt".into();
            "Prompt"
        };
        let bounds = desk.control(field).bounds().unwrap();
        let inside = egui::pos2(
            (bounds.x0 + bounds.x1) as f32 / 2.,
            (bounds.y0 + bounds.y1) as f32 / 2.,
        );
        let outside = egui::pos2(20., 700.);
        desk.app.prefs.privacy_mode = true;
        desk.app.save_preferences();
        assert!(!frame(
            &mut desk,
            vec![egui::Event::PointerMoved(outside)],
            true
        ));
        ctrl(&mut desk, egui::Key::P);
        assert!(frame(&mut desk, vec![], true), "{workspace:?}");
        assert!(frame(
            &mut desk,
            vec![egui::Event::Text(" typed".into())],
            true
        ));
        assert!(!frame(
            &mut desk,
            vec![egui::Event::MouseMoved(egui::vec2(1., 0.))],
            true,
        ));
        assert!(!frame(&mut desk, vec![], true));
        assert!(frame(
            &mut desk,
            vec![egui::Event::PointerMoved(inside)],
            true
        ));
        assert!(!frame(
            &mut desk,
            vec![egui::Event::PointerMoved(outside)],
            true
        ));

        desk.frame(vec![key(egui::Key::Escape, egui::Modifiers::NONE, false)]);
        ctrl(&mut desk, egui::Key::P);
        assert!(frame(&mut desk, vec![], true));
        assert!(!frame(&mut desk, vec![], false));
        assert!(!frame(&mut desk, vec![], true));
        assert!(desk.app.prefs.privacy_mode);
        assert!(Preferences::load(desk._root.path()).unwrap().privacy_mode);
    }
}

#[test]
fn shortcuts_generate_uses_button_validation_and_respects_disabled_states() {
    let mut desk = Desk::new();
    desk.app.metadata = Some(imagine_server::contract::model(client::MODEL).unwrap());
    desk.app.prefs.generation.steps = 0;
    desk.click("Add to queue");
    let error = desk.app.image.error.take().expect("button validation");
    ctrl(&mut desk, egui::Key::Enter);
    assert_eq!(desk.app.image.error.take().as_deref(), Some(error.as_str()));
    assert!(desk.app.image.pending.is_none());

    for blocked in [
        "busy",
        "connecting",
        "folder",
        "disconnected",
        "empty",
        "pending",
    ] {
        let client = desk.app.client.clone();
        let prompt = desk.app.image.prompt.clone();
        match blocked {
            "busy" => desk.app.image.working = true,
            "connecting" => desk.app.connecting = true,
            "folder" => desk.app.picking_folder = true,
            "disconnected" => desk.app.client = None,
            "empty" => desk.app.image.prompt.clear(),
            "pending" => {
                desk.app.image.pending = Some(Submission {
                    upscale_source: None,
                    request: serde_json::json!({}),
                    key: "synthetic-pending".into(),
                    folder: desk._root.path().into(),
                    job: None,
                })
            }
            _ => unreachable!(),
        }
        ctrl(&mut desk, egui::Key::Enter);
        assert!(desk.app.image.error.is_none(), "{blocked}");
        desk.app.image.working = false;
        desk.app.connecting = false;
        desk.app.picking_folder = false;
        desk.app.client = client;
        desk.app.image.prompt = prompt;
        desk.app.image.pending = None;
    }
    desk.click("Video");
    ctrl(&mut desk, egui::Key::Enter);
    press(&mut desk, egui::Key::Z, egui::Modifiers::NONE);
    ctrl(&mut desk, egui::Key::P);
    assert!(desk.app.image.error.is_none());
    assert!(!desk.app.image.actual_pixels);
    assert!(!desk.ctx.text_edit_focused());
}
