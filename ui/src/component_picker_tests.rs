//! Synthetic input only; no desktop, model weights or runtime data.
use super::*;

struct Picker {
    ctx: egui::Context,
    files: Vec<ModelFile>,
    kind: &'static str,
    selected: String,
    nodes: Vec<egui::accesskit::Node>,
}
impl Picker {
    fn new(kind: &'static str) -> Self {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        ctx.global_style_mut(|s| s.animation_time = 0.);
        let files = [
            ("old", "original.safetensors", true, Some("qwen-image-2.1")),
            ("a", "portraits/qwen-custom.GGUF", false, None),
            ("b", "portraits/qwen-other.pt", false, Some("other-family")),
            ("c", "landscapes/flux.bin", false, None),
        ].into_iter().map(|(id,path,supported,group)| serde_json::from_value(json!({
            "id":id,"path":format!("{kind}/{path}"),"kind":kind,"supported":supported,"group":group,"model":null
        })).unwrap()).collect();
        let mut picker = Self {
            ctx,
            files,
            kind,
            selected: "old".into(),
            nodes: Vec::new(),
        };
        for _ in 0..3 {
            picker.frame(vec![]);
        }
        picker
    }
    fn label(&self) -> &'static str {
        if self.kind == "vae" {
            "VAE"
        } else {
            "Text encoder"
        }
    }
    fn frame(&mut self, events: Vec<egui::Event>) {
        let label = self.label();
        let mut output = self.ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(480., 680.),
                )),
                events,
                ..Default::default()
            },
            |ui| component_picker(ui, label, &self.files, self.kind, &mut self.selected),
        );
        output.textures_delta.clear();
        self.nodes = output
            .platform_output
            .accesskit_update
            .unwrap()
            .nodes
            .into_iter()
            .map(|(_, node)| node)
            .collect();
    }
    fn click_bounds(&mut self, bounds: egui::accesskit::Rect) {
        let pos = egui::pos2(
            (bounds.x0 + bounds.x1) as f32 / 2.,
            (bounds.y0 + bounds.y1) as f32 / 2.,
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
        self.frame(vec![]);
    }
    fn open(&mut self) {
        let bounds = self
            .nodes
            .iter()
            .find(|node| node.role() == egui::accesskit::Role::TextInput)
            .unwrap()
            .bounds()
            .unwrap();
        self.click_bounds(bounds);
        self.frame(vec![]);
    }
    fn key(&mut self, key: egui::Key) {
        self.frame(vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        self.frame(vec![]);
    }
    fn results(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter_map(|node| node.label())
            .filter(|label| label.starts_with(self.kind))
            .collect()
    }
}

#[test]
fn both_pickers_search_all_paths_and_commit_only_explicit_keyboard_or_mouse_choices() {
    for kind in ["text_encoders", "vae"] {
        let mut h = Picker::new(kind);
        h.open();
        assert_eq!(h.results().len(), 4);
        h.frame(vec![egui::Event::Text("QWEN   PORTRAITS".into())]);
        assert_eq!(h.results().len(), 2);
        assert_eq!(h.selected, "old");
        h.key(egui::Key::ArrowDown);
        h.key(egui::Key::Enter);
        assert_eq!(h.selected, "b");
        assert!(h.results().is_empty());
        h.open();
        h.frame(vec![egui::Event::Text("nonexistent".into())]);
        assert!(
            h.nodes
                .iter()
                .any(|node| node.label() == Some("No matching files")
                    || node.value() == Some("No matching files"))
        );
        h.key(egui::Key::Enter);
        assert_eq!(h.selected, "b");
        h.key(egui::Key::Escape);
        assert_eq!(h.selected, "b");
        assert_eq!(
            h.nodes
                .iter()
                .find(|n| n.role() == egui::accesskit::Role::TextInput)
                .unwrap()
                .value(),
            Some(format!("{kind}/portraits/qwen-other.pt").as_str())
        );
        h.open();
        h.frame(vec![egui::Event::Text("landscapes".into())]);
        let bounds = h
            .nodes
            .iter()
            .find(|node| node.label() == Some(format!("{kind}/landscapes/flux.bin").as_str()))
            .unwrap()
            .bounds()
            .unwrap();
        h.click_bounds(bounds);
        assert_eq!(h.selected, "c");
    }
}

#[test]
fn focus_opens_the_picker_and_empty_search_restores_all_files_without_changing_selection() {
    let mut h = Picker::new("vae");
    h.key(egui::Key::Tab);
    assert_eq!(h.results().len(), 4);
    h.frame(vec![egui::Event::Text("portraits".into())]);
    assert_eq!(h.results().len(), 2);
    h.frame(vec![egui::Event::Key {
        key: egui::Key::A,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers {
            ctrl: true,
            command: true,
            ..Default::default()
        },
    }]);
    h.key(egui::Key::Backspace);
    assert_eq!(h.results().len(), 4);
    h.key(egui::Key::Escape);
    assert_eq!(h.selected, "old");
}
