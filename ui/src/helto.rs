//! Native adaptation of the canonical Helto tokens.css/components.css.
use eframe::egui::{self, Color32, Stroke};

pub const BG: Color32 = Color32::from_rgb(0x18, 0x18, 0x25);
pub const SURFACE: Color32 = Color32::from_rgb(0x1e, 0x1e, 0x2e);
pub const FIELD: Color32 = Color32::from_rgb(0x31, 0x32, 0x44);
pub const RAISED: Color32 = Color32::from_rgb(0x45, 0x47, 0x5a);
pub const HOVER: Color32 = Color32::from_rgb(0x58, 0x5b, 0x70);
pub const BORDER_HOVER: Color32 = Color32::from_rgb(0x6c, 0x70, 0x86);
pub const MEDIA: Color32 = Color32::from_rgb(0x11, 0x11, 0x1b);
pub const TEXT: Color32 = Color32::from_rgb(0xcd, 0xd6, 0xf4);
pub const DIM: Color32 = Color32::from_rgb(0xa6, 0xad, 0xc8);
pub const ACCENT: Color32 = Color32::from_rgb(0xfa, 0xb3, 0x87);
pub const ACCENT_TEXT: Color32 = Color32::from_rgb(0xfd, 0xdc, 0xc4);
pub const ACCENT_BORDER: Color32 = Color32::from_rgb(0x93, 0x66, 0x4a);
pub const ACCENT_TOP: Color32 = Color32::from_rgb(0x4f, 0x3a, 0x2a);
pub const ACCENT_BOTTOM: Color32 = Color32::from_rgb(0x3d, 0x2d, 0x20);
pub const ACCENT_HOVER_TOP: Color32 = Color32::from_rgb(0x5d, 0x45, 0x31);
pub const ACCENT_HOVER_BOTTOM: Color32 = Color32::from_rgb(0x49, 0x36, 0x26);
pub const FOCUS: Color32 = Color32::from_rgb(0x89, 0xb4, 0xfa);
pub const ERROR: Color32 = Color32::from_rgb(0xf3, 0x8b, 0xa8);
pub const OK: Color32 = Color32::from_rgb(0xa6, 0xe3, 0xa1);

pub fn install(ctx: &egui::Context) {
    let mut style = egui::Style {
        visuals: egui::Visuals::dark(),
        ..Default::default()
    };
    style.visuals.override_text_color = Some(TEXT);
    style.visuals.weak_text_color = Some(DIM);
    style.visuals.panel_fill = SURFACE;
    style.visuals.window_fill = SURFACE;
    style.visuals.extreme_bg_color = BG;
    style.visuals.text_edit_bg_color = Some(FIELD);
    style.visuals.selection.bg_fill = ACCENT_TOP;
    style.visuals.selection.stroke = Stroke::new(1., ACCENT);
    style.visuals.error_fg_color = ERROR;
    style.visuals.hyperlink_color = ACCENT;
    for visual in [
        &mut style.visuals.widgets.noninteractive,
        &mut style.visuals.widgets.inactive,
        &mut style.visuals.widgets.hovered,
        &mut style.visuals.widgets.active,
        &mut style.visuals.widgets.open,
    ] {
        visual.bg_fill = FIELD;
        visual.weak_bg_fill = FIELD;
        visual.bg_stroke = Stroke::new(1., RAISED);
        visual.fg_stroke = Stroke::new(1., TEXT);
        visual.corner_radius = 5.into();
        visual.expansion = 0.;
    }
    style.visuals.widgets.hovered.bg_fill = HOVER;
    style.visuals.widgets.hovered.bg_stroke.color = BORDER_HOVER;
    style.visuals.widgets.active.bg_fill = ACCENT_TOP;
    style.visuals.widgets.active.fg_stroke.color = ACCENT;
    style.spacing.item_spacing = egui::vec2(6., 8.);
    style.spacing.button_padding = egui::vec2(8., 4.);
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(12.));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(12.));
    style
        .text_styles
        .insert(egui::TextStyle::Small, egui::FontId::proportional(11.));
    style
        .text_styles
        .insert(egui::TextStyle::Heading, egui::FontId::proportional(18.));
    ctx.set_global_style(style);
    // Prefer the host's system sans; retain bundled glyph fallbacks.
    if let Ok(output) = std::process::Command::new("fc-match")
        .args(["-f", "%{file}", "sans-serif"])
        .output()
        && output.status.success()
        && let Ok(path) = String::from_utf8(output.stdout)
        && let Ok(data) = std::fs::read(path.trim())
    {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "system-sans".into(),
            egui::FontData::from_owned(data).into(),
        );
        fonts
            .families
            .get_mut(&egui::FontFamily::Proportional)
            .unwrap()
            .insert(0, "system-sans".into());
        ctx.set_fonts(fonts);
    }
}
pub fn button(ui: &mut egui::Ui, label: &str, primary: bool) -> egui::Response {
    let font = egui::TextStyle::Button.resolve(ui.style());
    let galley = ui.painter().layout_no_wrap(
        label.to_owned(),
        font,
        if primary { ACCENT_TEXT } else { TEXT },
    );
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2((galley.size().x + 16.).max(28.), 24.),
        egui::Sense::click(),
    );
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
    });
    if ui.is_rect_visible(rect) {
        let (top, bottom, border) = if primary {
            if response.hovered() {
                (ACCENT_HOVER_TOP, ACCENT_HOVER_BOTTOM, ACCENT_BORDER)
            } else {
                (ACCENT_TOP, ACCENT_BOTTOM, ACCENT_BORDER)
            }
        } else if response.hovered() {
            (HOVER, RAISED, BORDER_HOVER)
        } else {
            (RAISED, FIELD, RAISED)
        };
        let alpha = if ui.is_enabled() { 1. } else { 0.4 };
        let painted = if response.is_pointer_button_down_on() {
            rect.translate(egui::vec2(0., 1.))
        } else {
            rect
        };
        gradient(
            ui.painter(),
            painted,
            5.,
            top.gamma_multiply(alpha),
            bottom.gamma_multiply(alpha),
        );
        ui.painter().rect_stroke(
            painted,
            5,
            Stroke::new(1., border.gamma_multiply(alpha)),
            egui::StrokeKind::Inside,
        );
        if response.has_focus() {
            ui.painter().rect_stroke(
                rect.expand(2.),
                6,
                Stroke::new(2., FOCUS),
                egui::StrokeKind::Outside,
            );
        }
        ui.painter().galley(
            painted.center() - galley.size() / 2.,
            galley,
            if primary { ACCENT_TEXT } else { TEXT },
        );
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}
pub fn gradient(
    painter: &egui::Painter,
    rect: egui::Rect,
    radius: f32,
    top: Color32,
    bottom: Color32,
) {
    let mut mesh = egui::Mesh::default();
    let color = |y: f32| {
        let t = ((y - rect.top()) / rect.height()).clamp(0., 1.);
        let a = egui::Rgba::from(top);
        let b = egui::Rgba::from(bottom);
        Color32::from(a * (1. - t) + b * t)
    };
    mesh.colored_vertex(rect.center(), color(rect.center().y));
    for (center, start) in [
        (egui::pos2(rect.right() - radius, rect.top() + radius), -90.),
        (
            egui::pos2(rect.right() - radius, rect.bottom() - radius),
            0.,
        ),
        (
            egui::pos2(rect.left() + radius, rect.bottom() - radius),
            90.,
        ),
        (egui::pos2(rect.left() + radius, rect.top() + radius), 180.),
    ] {
        for i in 0..=6 {
            let angle = (start + i as f32 * 15.).to_radians();
            let p = center + radius * egui::vec2(angle.cos(), angle.sin());
            mesh.colored_vertex(p, color(p.y));
        }
    }
    let n = mesh.vertices.len() as u32 - 1;
    for i in 1..=n {
        mesh.add_triangle(0, i, if i == n { 1 } else { i + 1 });
    }
    painter.add(mesh);
}
pub fn textarea(ui: &mut egui::Ui, text: &mut String, rows: usize, id: &str) -> egui::Response {
    ui.scope(|ui| {
        ui.visuals_mut().selection.stroke = Stroke::new(1., FOCUS);
        let id = ui.make_persistent_id(id);
        let previous_selection =
            egui::TextEdit::load_state(ui.ctx(), id).and_then(|state| state.cursor.char_range());
        let mut output = egui::TextEdit::multiline(text)
            .id(id)
            .desired_rows(rows)
            .desired_width(f32::INFINITY)
            .margin(egui::vec2(9., 7.))
            .show(ui);
        // egui 0.36 moves the caret on any pointer press, including right-click.
        // Preserve the selection that the context-menu action should operate on.
        if output.response.hovered()
            && ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Secondary))
            && previous_selection.is_some_and(|range| !range.is_empty())
        {
            output.state.cursor.set_char_range(previous_selection);
            output.state.clone().store(ui.ctx(), output.response.id);
            ui.ctx().request_repaint();
        }
        let selected = output
            .state
            .cursor
            .char_range()
            .is_some_and(|range| !range.is_empty());
        output.response.context_menu(|ui| {
            for (label, shortcut, command, enabled) in [
                ("Cut", "Ctrl+X", egui::ViewportCommand::RequestCut, selected),
                (
                    "Copy",
                    "Ctrl+C",
                    egui::ViewportCommand::RequestCopy,
                    selected,
                ),
                ("Paste", "Ctrl+V", egui::ViewportCommand::RequestPaste, true),
            ] {
                if ui
                    .add_enabled(enabled, egui::Button::new(label).shortcut_text(shortcut))
                    .clicked()
                {
                    output.response.request_focus();
                    ui.ctx().send_viewport_cmd(command);
                    ui.close();
                }
            }
        });
        output.response.response
    })
    .inner
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn text_fields_preserve_clipboard_events_and_unicode_selection() {
        for name in ["prompt", "negative-prompt"] {
            let ctx = egui::Context::default();
            let mut text = String::new();
            let mut id = None;
            let mut rect = egui::Rect::NOTHING;
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let response = textarea(ui, &mut text, 3, name);
                id = Some(response.id);
                rect = response.rect;
            });
            output.textures_delta.clear();
            let id = id.unwrap();
            ctx.memory_mut(|m| m.request_focus(id));
            let sample = "Synthetic clipboard: åäö\nsecond line";
            let mut output = ctx.run_ui(
                egui::RawInput {
                    events: vec![egui::Event::Paste(sample.into())],
                    ..Default::default()
                },
                |ui| {
                    textarea(ui, &mut text, 3, name);
                },
            );
            output.textures_delta.clear();
            assert_eq!(text, sample);
            let mut state = egui::TextEdit::load_state(&ctx, id).unwrap();
            state
                .cursor
                .set_char_range(Some(egui::text::CCursorRange::two(
                    egui::text::CCursor::new(0),
                    egui::text::CCursor::new(text.chars().count()),
                )));
            state.store(&ctx, id);
            // Opening the menu must not collapse the selected Unicode text.
            let pos = rect.center();
            for pressed in [true, false] {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        events: vec![
                            egui::Event::PointerMoved(pos),
                            egui::Event::PointerButton {
                                pos,
                                button: egui::PointerButton::Secondary,
                                pressed,
                                modifiers: egui::Modifiers::NONE,
                            },
                        ],
                        ..Default::default()
                    },
                    |ui| {
                        textarea(ui, &mut text, 3, name);
                    },
                );
                output.textures_delta.clear();
                let range = egui::TextEdit::load_state(&ctx, id)
                    .unwrap()
                    .cursor
                    .char_range()
                    .unwrap();
                assert_eq!(range.slice_str(&text), sample);
            }
            egui::Popup::close_all(&ctx);
            ctx.memory_mut(|m| m.request_focus(id));
            for event in [egui::Event::Copy, egui::Event::Cut] {
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        events: vec![event],
                        ..Default::default()
                    },
                    |ui| {
                        textarea(ui, &mut text, 3, name);
                    },
                );
                output.textures_delta.clear();
                assert!(output.platform_output.commands.iter().any(|command| matches!(command, egui::OutputCommand::CopyText(value) if value == sample)));
            }
            assert!(text.is_empty());
        }
    }
    #[test]
    fn primary_button_keyboard_focus_and_disabled_state() {
        let ctx = egui::Context::default();
        install(&ctx);
        let mut id = None;
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let response = button(ui, "Generate", true);
            id = Some(response.id);
        });
        output.textures_delta.clear();
        ctx.memory_mut(|memory| memory.request_focus(id.unwrap()));
        let mut focused = false;
        let mut activated = false;
        let mut output = ctx.run_ui(
            egui::RawInput {
                events: vec![egui::Event::Key {
                    key: egui::Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |ui| {
                let response = button(ui, "Generate", true);
                focused = response.has_focus();
                activated = response.clicked();
            },
        );
        output.textures_delta.clear();
        assert!(focused);
        assert!(activated);
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let r = ui
                .add_enabled_ui(false, |ui| button(ui, "Generate", true))
                .inner;
            assert!(!r.enabled());
            assert!(!r.clicked());
        });
        output.textures_delta.clear();
    }
}
