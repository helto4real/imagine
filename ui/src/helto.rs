//! Native adaptation of the canonical Helto tokens.css/components.css.
use crate::icons::{self, Icon};
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
    style.spacing.interact_size.y = 32.;
    style.spacing.combo_height = 260.;
    style.visuals.window_corner_radius = 10.into();
    style.visuals.window_stroke = Stroke::new(1., RAISED);
    style
        .text_styles
        .insert(egui::TextStyle::Monospace, egui::FontId::monospace(13.));
    style
        .text_styles
        .insert(egui::TextStyle::Body, egui::FontId::proportional(14.));
    style
        .text_styles
        .insert(egui::TextStyle::Button, egui::FontId::proportional(14.));
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
pub(crate) fn icon_button(
    ui: &mut egui::Ui,
    icon: Icon,
    label: &str,
    active: bool,
) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(32., 32.), egui::Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), active, label)
    });
    if ui.is_rect_visible(rect) {
        let (top, bottom, border) = if active {
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
        let painted = if response.is_pointer_button_down_on() {
            rect.translate(egui::vec2(0., 1.))
        } else {
            rect
        };
        gradient(ui.painter(), painted, 5., top, bottom);
        ui.painter().rect_stroke(
            painted,
            5,
            Stroke::new(1., border),
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
        let color = if active {
            ACCENT_TEXT
        } else if matches!(icon, Icon::Trash) {
            ERROR
        } else {
            TEXT
        };
        icons::paint(ui.painter(), painted.shrink(7.), icon, color);
    }
    response
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(label)
}

pub fn select_button(ui: &mut egui::Ui, label: &str, active: bool) -> egui::Response {
    let galley = ui.painter().layout_no_wrap(
        label.into(),
        egui::FontId::proportional(14.),
        if active { ACCENT_TEXT } else { TEXT },
    );
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(galley.size().x + 20., 32.), egui::Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::SelectableLabel,
            ui.is_enabled(),
            active,
            label,
        )
    });
    let (top, bottom, border) = match (active, response.hovered()) {
        (true, true) => (ACCENT_HOVER_TOP, ACCENT_HOVER_BOTTOM, ACCENT_BORDER),
        (true, false) => (ACCENT_TOP, ACCENT_BOTTOM, ACCENT_BORDER),
        (false, true) => (HOVER, RAISED, BORDER_HOVER),
        (false, false) => (RAISED, FIELD, RAISED),
    };
    gradient(ui.painter(), rect, 5., top, bottom);
    ui.painter()
        .rect_stroke(rect, 5, Stroke::new(1., border), egui::StrokeKind::Inside);
    ui.painter()
        .galley(rect.center() - galley.size() * 0.5, galley, TEXT);
    if response.has_focus() {
        ui.painter().rect_stroke(
            rect.expand(2.),
            6,
            Stroke::new(2., FOCUS),
            egui::StrokeKind::Outside,
        );
    }
    response.on_hover_cursor(egui::CursorIcon::PointingHand)
}

pub fn toggle(ui: &mut egui::Ui, value: &mut bool, label: &str) -> egui::Response {
    let (rect, mut response) = ui.allocate_exact_size(egui::vec2(36., 20.), egui::Sense::click());
    if response.clicked() {
        *value = !*value;
        response.mark_changed();
    }
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *value, label)
    });
    let fill = if *value { ACCENT_BOTTOM } else { FIELD };
    ui.painter().rect_filled(rect, 10, fill);
    ui.painter().rect_stroke(
        rect,
        10,
        Stroke::new(1., if *value { ACCENT_BORDER } else { RAISED }),
        egui::StrokeKind::Inside,
    );
    let x = if *value {
        rect.right() - 10.
    } else {
        rect.left() + 10.
    };
    ui.painter().circle_filled(
        egui::pos2(x, rect.center().y),
        7.,
        if *value { ACCENT } else { TEXT },
    );
    if response.has_focus() {
        ui.painter().rect_stroke(
            rect.expand(2.),
            10,
            Stroke::new(2., FOCUS),
            egui::StrokeKind::Outside,
        );
    }
    response.on_hover_text(label)
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
pub fn focus_and_reveal_textarea(response: &egui::Response) {
    response.request_focus();
    response.ctx.data_mut(|data| {
        data.insert_temp(response.id.with("keyboard-reveal"), true);
    });
    response.ctx.request_repaint();
}

pub fn textarea(
    ui: &mut egui::Ui,
    text: &mut String,
    rows: usize,
    id: &str,
    privacy_mode: bool,
    resizable: bool,
) -> egui::Response {
    let width = ui.available_width();
    let height = ui.text_style_height(&egui::TextStyle::Monospace) * rows as f32 + 14.;
    egui::Resize::default()
        .id_salt((id, "size"))
        .default_size(egui::vec2(width, height))
        .min_size(egui::vec2(width, if resizable { 60. } else { height }))
        .max_size(egui::vec2(width, if resizable { 800. } else { height }))
        .resizable([false, resizable])
        .with_stroke(false)
        .show(ui, |ui| {
            let rect = ui.available_rect_before_wrap();
            let hidden = privacy_mode
                && !(ui.rect_contains_pointer(rect)
                    && ui.input(|input| {
                        input.focused
                            && input
                                .pointer
                                .hover_pos()
                                .is_some_and(|pos| rect.intersect(ui.clip_rect()).contains(pos))
                    }));
            ui.painter().rect_filled(rect, 5, FIELD);
            let response = egui::ScrollArea::vertical()
                .id_salt((id, "text-scroll"))
                .auto_shrink([false, false])
                .max_height(rect.height())
                .show(ui, |ui| {
                    textarea_content(ui, text, id, hidden, rect.height())
                })
                .inner;
            ui.painter().rect_stroke(
                rect,
                5,
                Stroke::new(1., if response.has_focus() { FOCUS } else { RAISED }),
                egui::StrokeKind::Inside,
            );
            response
        })
}

fn textarea_content(
    ui: &mut egui::Ui,
    text: &mut String,
    id: &str,
    hidden: bool,
    height: f32,
) -> egui::Response {
    ui.scope(|ui| {
        ui.visuals_mut().selection.stroke = Stroke::new(1., FOCUS);
        let id = ui.make_persistent_id(id);
        let reset_reveal = !ui.memory(|memory| memory.has_focus(id))
            || ui.input(|input| {
                !input.focused
                    || input.events.iter().any(|event| {
                        matches!(
                            event,
                            egui::Event::PointerMoved(_)
                                | egui::Event::MouseMoved(_)
                                | egui::Event::PointerGone
                        )
                    })
            });
        let keyboard_reveal = ui.data_mut(|data| {
            let reveal_id = id.with("keyboard-reveal");
            if reset_reveal {
                data.remove::<bool>(reveal_id);
            }
            data.get_temp::<bool>(reveal_id).unwrap_or(false)
        });
        let previous_selection =
            egui::TextEdit::load_state(ui.ctx(), id).and_then(|state| state.cursor.char_range());
        let mut output = ui
            .scope(|ui| {
                // Suppress glyphs, selection and caret together, while keeping editing active.
                // Keep the clipboard menu outside this scope so it remains visible.
                if hidden && !keyboard_reveal {
                    ui.set_opacity(0.);
                }
                egui::TextEdit::multiline(text)
                    .id(id)
                    .desired_rows(1)
                    .frame(egui::Frame::NONE.inner_margin(egui::Margin::symmetric(9, 7)))
                    .min_size(egui::vec2(0., height))
                    .desired_width(f32::INFINITY)
                    .font(egui::TextStyle::Monospace)
                    .margin(egui::vec2(9., 7.))
                    .show(ui)
            })
            .inner;
        // Opacity hides pixels only; concealed text must also leave accessibility values.
        if hidden && !keyboard_reveal {
            redact_text_accessibility(ui.ctx(), output.response.id);
        }
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

fn redact_text_accessibility(ctx: &egui::Context, id: egui::Id) {
    let children = ctx
        .accesskit_node_builder(id, |node| {
            node.clear_value();
            node.clear_text_selection();
            node.children().to_vec()
        })
        .unwrap_or_default();
    for child in children {
        // These are IDs emitted by egui itself, preserving its nonzero high-entropy bits.
        let id = unsafe { egui::Id::from_high_entropy_bits(child.0) };
        redact_text_accessibility(ctx, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_text_hides_on_leave_even_with_focus_and_selection() {
        fn contains_text(shape: &egui::Shape) -> bool {
            match shape {
                egui::Shape::Text(text) => text.opacity_factor > 0.,
                egui::Shape::Vec(shapes) => shapes.iter().any(contains_text),
                _ => false,
            }
        }
        for name in ["prompt", "negative-prompt"] {
            let ctx = egui::Context::default();
            let mut text = "Synthetic private prompt: åäö".to_owned();
            let inside = egui::pos2(24., 24.);
            let outside = egui::pos2(500., 400.);
            for (events, focused, privacy, visible) in [
                (vec![], true, true, false),
                (vec![egui::Event::PointerMoved(inside)], true, true, true),
                (vec![egui::Event::PointerMoved(outside)], true, true, false),
                (vec![egui::Event::PointerMoved(inside)], true, true, true),
                (vec![], false, true, false),
                (vec![], true, true, true),
                (vec![egui::Event::PointerGone], true, true, false),
                (vec![], false, false, true),
            ] {
                let mut response = None;
                let mut output = ctx.run_ui(
                    egui::RawInput {
                        screen_rect: Some(egui::Rect::from_min_size(
                            egui::Pos2::ZERO,
                            egui::vec2(800., 600.),
                        )),
                        events,
                        focused,
                        ..Default::default()
                    },
                    |ui| {
                        ui.set_width(300.);
                        response = Some(textarea(ui, &mut text, 8, name, privacy, true));
                    },
                );
                output.textures_delta.clear();
                assert_eq!(
                    output
                        .shapes
                        .iter()
                        .any(|shape| contains_text(&shape.shape)),
                    visible,
                    "{name}: focused={focused}, privacy={privacy}"
                );
                let response = response.unwrap();
                ctx.memory_mut(|memory| memory.request_focus(response.id));
                let mut state = egui::TextEdit::load_state(&ctx, response.id).unwrap();
                state
                    .cursor
                    .set_char_range(Some(egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(text.chars().count()),
                    )));
                state.store(&ctx, response.id);
                assert_eq!(text, "Synthetic private prompt: åäö");
            }
        }
    }

    #[test]
    fn text_fields_preserve_clipboard_events_and_unicode_selection() {
        for (name, privacy) in [
            ("prompt", false),
            ("prompt", true),
            ("negative-prompt", false),
            ("negative-prompt", true),
        ] {
            let ctx = egui::Context::default();
            let mut text = String::new();
            let mut id = None;
            let mut rect = egui::Rect::NOTHING;
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let response = textarea(ui, &mut text, 3, name, privacy, false);
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
                    textarea(ui, &mut text, 3, name, privacy, false);
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
                        textarea(ui, &mut text, 3, name, privacy, false);
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
                        events: vec![egui::Event::PointerGone, event],
                        ..Default::default()
                    },
                    |ui| {
                        textarea(ui, &mut text, 3, name, privacy, false);
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
            let response = icon_button(ui, Icon::Generate, "Generate", true);
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
                let response = icon_button(ui, Icon::Generate, "Generate", true);
                focused = response.has_focus();
                activated = response.clicked();
            },
        );
        output.textures_delta.clear();
        assert!(focused);
        assert!(activated);
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            let r = ui
                .add_enabled_ui(false, |ui| {
                    icon_button(ui, Icon::Generate, "Generate", true)
                })
                .inner;
            assert!(!r.enabled());
            assert!(!r.clicked());
        });
        output.textures_delta.clear();
    }
}
