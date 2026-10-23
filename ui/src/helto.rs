//! Native adaptation of the canonical Helto tokens.css/components.css.
use crate::icons::{self, Icon};
use eframe::egui::{self, Color32, Stroke};

/// The one Hide previews rule: private content shows only while the pointer
/// is over its visible part of `rect` in a focused window.
pub fn revealed(ui: &egui::Ui, rect: egui::Rect, privacy_mode: bool) -> bool {
    let visible = rect.intersect(ui.clip_rect());
    // The live pointer position also covers a pointer that left the window.
    !privacy_mode
        || ui.rect_contains_pointer(visible)
            && ui.input(|i| i.focused && i.pointer.hover_pos().is_some_and(|p| visible.contains(p)))
}
pub fn reveal_in_viewport(ui: &egui::Ui, privacy_mode: bool) -> bool {
    // Content coordinates move with scrolling; hover belongs to the visible viewport.
    revealed(ui, ui.clip_rect(), privacy_mode)
}
/// A section title inside a panel or window.
pub fn section(title: &str) -> egui::RichText {
    egui::RichText::new(title).strong().size(14.).color(TEXT)
}
/// Error text; details can name files, so they follow Hide previews.
pub fn error_text(error: &str, revealed: bool) -> egui::RichText {
    egui::RichText::new(if revealed { error } else { "Hidden error" }).color(ERROR)
}
/// The two actions of a consequential question.
pub(crate) struct Confirm {
    pub yes: egui::Response,
    pub no: egui::Response,
}
/// Asks `question`, offering `yes` (blocked with a reason when given) or `no`.
pub(crate) fn confirm(
    ui: &mut egui::Ui,
    question: &str,
    yes: (Icon, &str),
    no: &str,
    blocked: Option<&str>,
) -> Confirm {
    ui.label(question);
    ui.horizontal(|ui| Confirm {
        yes: action(ui, yes.0, yes.1, false, blocked),
        no: icon_button(ui, Icon::Close, no, false),
    })
    .inner
}

const PRIVACY_MODE: &str = "helto-privacy-mode";
/// Records Hide previews for this frame; private scopes default to hidden mode.
pub fn set_privacy_mode(ctx: &egui::Context, on: bool) {
    ctx.data_mut(|data| data.insert_temp(egui::Id::new(PRIVACY_MODE), on));
}
fn privacy_mode(ctx: &egui::Context) -> bool {
    ctx.data(|data| data.get_temp(egui::Id::new(PRIVACY_MODE)))
        .unwrap_or(true)
}

/// Conceal a private window body while preserving its layout and scroll extent.
/// Controls stay usable, as on the main desk. The text field being typed in
/// stays revealed while it keeps focus in a focused window.
pub(crate) fn private_scope<R>(
    ui: &mut egui::Ui,
    hidden: bool,
    body: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    ui.scope(|ui| {
        let layer = ui.layer_id();
        let start = ui.ctx().graphics_mut(|g| g.entry(layer).next_idx());
        let result = body(ui);
        let typed = typed_field(ui);
        if hidden {
            let keep = typed.map(|(_, rect)| rect.expand(1.));
            ui.ctx().graphics_mut(|g| {
                let list = g.entry(layer);
                for index in start.0..list.next_idx().0 {
                    list.mutate_shape(egui::layers::ShapeIdx(index), |clipped| {
                        let bounds = clipped
                            .shape
                            .visual_bounding_rect()
                            .intersect(clipped.clip_rect);
                        if !keep.is_some_and(|keep| keep.contains_rect(bounds)) {
                            clipped.shape = egui::Shape::Noop;
                        }
                    });
                }
            });
            redact_private_accessibility(ui.ctx(), ui.unique_id(), typed.map(|(id, _)| id));
        }
        result
    })
    .inner
}
/// The focused text field inside this scope once the user has typed in it.
fn typed_field(ui: &egui::Ui) -> Option<(egui::Id, egui::Rect)> {
    let ctx = ui.ctx();
    let id = ctx.memory(|memory| memory.focused())?;
    let response = ctx.read_response(id)?;
    let editing = ctx.text_edit_focused()
        && ctx.input(|input| input.focused)
        && response.layer_id == ui.layer_id()
        && ui.min_rect().intersects(response.rect);
    let private = privacy_mode(ctx);
    let typed = editing && ctx.input(|input| input.events.iter().any(edits_text));
    let reveal_id = id.with("scope-typing-reveal");
    ctx.data_mut(|data| {
        if !editing || !private {
            data.remove::<bool>(reveal_id);
        } else if typed {
            data.insert_temp(reveal_id, true);
        }
        data.get_temp::<bool>(reveal_id)
    })
    .map(|_| (id, response.rect))
}
fn redact_private_accessibility(ctx: &egui::Context, id: egui::Id, keep: Option<egui::Id>) {
    if Some(id) == keep {
        return;
    }
    let children = ctx
        .accesskit_node_builder(id, |node| {
            node.clear_label();
            node.clear_description();
            node.clear_value();
            node.clear_text_selection();
            node.children().to_vec()
        })
        .unwrap_or_default();
    for child in children {
        let id = unsafe { egui::Id::from_high_entropy_bits(child.0) };
        redact_private_accessibility(ctx, id, keep);
    }
}

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
pub const WARN: Color32 = Color32::from_rgb(0xf9, 0xe2, 0xaf);

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
/// Icon-only button; its label is the accessible name and the hover text,
/// also while disabled.
pub(crate) fn icon_button(
    ui: &mut egui::Ui,
    icon: Icon,
    label: &str,
    active: bool,
) -> egui::Response {
    paint_icon_button(ui, icon, label, active)
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(label)
        .on_disabled_hover_text(label)
}

/// Icon-only action that is disabled while `blocked` names a reason; the
/// hover text then gives the name and that reason.
pub(crate) fn action(
    ui: &mut egui::Ui,
    icon: Icon,
    label: &str,
    primary: bool,
    blocked: Option<&str>,
) -> egui::Response {
    let response = ui
        .add_enabled_ui(blocked.is_none(), |ui| {
            paint_icon_button(ui, icon, label, primary)
        })
        .inner
        .on_hover_cursor(egui::CursorIcon::PointingHand)
        .on_hover_text(label);
    match blocked {
        Some(reason) => response.on_disabled_hover_text(format!("{label}\n{reason}")),
        None => response,
    }
}

fn paint_icon_button(ui: &mut egui::Ui, icon: Icon, label: &str, active: bool) -> egui::Response {
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
}

/// A pane divider drawn in the gap `handle`. Dragging moves it along the
/// x axis for a vertical bar, else the y axis; double-click asks to reset.
pub(crate) enum Divider {
    Moved(f32),
    Reset,
}
pub(crate) fn divider(
    ui: &mut egui::Ui,
    handle: egui::Rect,
    vertical_bar: bool,
    label: &str,
) -> Option<Divider> {
    let response = ui.interact(
        handle,
        ui.id().with(("divider", label)),
        egui::Sense::click_and_drag(),
    );
    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Other, true, label));
    let active = response.hovered() || response.dragged();
    if active {
        ui.ctx().set_cursor_icon(if vertical_bar {
            egui::CursorIcon::ResizeHorizontal
        } else {
            egui::CursorIcon::ResizeVertical
        });
        let line = if vertical_bar {
            egui::Rect::from_center_size(handle.center(), egui::vec2(2., handle.height() - 16.))
        } else {
            egui::Rect::from_center_size(handle.center(), egui::vec2(handle.width() - 16., 2.))
        };
        ui.painter().rect_filled(
            line,
            1,
            if response.dragged() {
                ACCENT
            } else {
                BORDER_HOVER
            },
        );
    }
    if response.double_clicked() {
        return Some(Divider::Reset);
    }
    let delta = response.drag_delta();
    let delta = if vertical_bar { delta.x } else { delta.y };
    (response.dragged() && delta != 0.).then_some(Divider::Moved(delta))
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
/// Typing, pasting or deleting in the focused field; focus or caret movement alone is not.
fn edits_text(event: &egui::Event) -> bool {
    matches!(
        event,
        egui::Event::Text(_) | egui::Event::Paste(_) | egui::Event::Cut | egui::Event::Ime(_)
    ) || matches!(
        event,
        egui::Event::Key {
            key: egui::Key::Backspace | egui::Key::Delete | egui::Key::Enter,
            pressed: true,
            ..
        }
    )
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
            let hidden = !revealed(ui, rect, privacy_mode);
            ui.painter().rect_filled(rect, 5, FIELD);
            let mut response = egui::ScrollArea::vertical()
                .id_salt((id, "text-scroll"))
                .auto_shrink([false, false])
                .max_height(rect.height())
                .show(ui, |ui| {
                    textarea_content(ui, text, id, privacy_mode, hidden, rect.height())
                })
                .inner;
            // A field's geometry is its frame, not the potentially much taller
            // text content inside its own scroll area. Keep the text widget ID
            // and editing/focus state for completion, clipboard and accessibility.
            response.rect = rect;
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
    private: bool,
    hidden: bool,
    height: f32,
) -> egui::Response {
    ui.scope(|ui| {
        ui.visuals_mut().selection.stroke = Stroke::new(1., FOCUS);
        let id = ui.make_persistent_id(id);
        // In hidden mode the field being typed in stays revealed while it keeps
        // focus in a focused window. Checked before painting so the first
        // keystroke is not concealed; typing before Hide previews was on is not.
        let editing = ui.memory(|memory| memory.has_focus(id)) && ui.input(|input| input.focused);
        let typed = editing && ui.input(|input| input.events.iter().any(edits_text));
        let typing_reveal = ui.data_mut(|data| {
            let reveal_id = id.with("typing-reveal");
            if !editing || !private {
                data.remove::<bool>(reveal_id);
            } else if typed {
                data.insert_temp(reveal_id, true);
            }
            data.get_temp::<bool>(reveal_id).unwrap_or(false)
        });
        #[cfg(feature = "ui-preview")]
        {
            let focused = ui.input(|i| i.focused);
            let frame = ui.ctx().cumulative_frame_nr();
            ui.data_mut(|d| {
                let key = egui::Id::new("qa-text-privacy");
                let mut values = d
                    .get_temp::<std::collections::BTreeMap<String, (u64, bool, bool, bool)>>(key)
                    .unwrap_or_default();
                values.insert(format!("{id:?}"), (frame, hidden, typing_reveal, focused));
                d.insert_temp(key, values);
            });
        }
        let previous_selection =
            egui::TextEdit::load_state(ui.ctx(), id).and_then(|state| state.cursor.char_range());
        let mut output = ui
            .scope(|ui| {
                // Suppress glyphs, selection and caret together, while keeping editing active.
                // Keep the clipboard menu outside this scope so it remains visible.
                if hidden && !typing_reveal {
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
        if hidden && !typing_reveal {
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
    fn blocked_icon_actions_keep_their_name_and_report_disabled() {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut clicked = false;
        let mut output = ctx.run_ui(Default::default(), |ui| {
            clicked |=
                action(ui, Icon::Generate, "Own action", true, Some("Own reason.")).clicked();
            icon_button(ui, Icon::Save, "Own ready action", false);
        });
        output.textures_delta.clear();
        let nodes = output.platform_output.accesskit_update.unwrap().nodes;
        let node = |label| {
            nodes
                .iter()
                .find(|(_, n)| n.label() == Some(label))
                .unwrap_or_else(|| panic!("missing {label}"))
                .1
                .clone()
        };
        assert!(node("Own action").is_disabled());
        assert!(!node("Own ready action").is_disabled());
        assert!(!clicked);
    }

    #[test]
    fn multiline_field_geometry_tracks_the_frame_without_truncating_long_text() {
        let ctx = egui::Context::default();
        let mut heights = vec![];
        for original in [
            "Own short prompt".to_owned(),
            "Own wrapped long prompt. ".repeat(200),
        ] {
            let mut text = original.clone();
            let mut rect = egui::Rect::NOTHING;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(500., 400.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    ui.set_width(240.);
                    rect = textarea(ui, &mut text, 3, "Own field geometry", false, false).rect;
                },
            );
            output.textures_delta.clear();
            assert_eq!(text, original);
            assert!(rect.height() < 100., "field response grew to {:?}", rect);
            heights.push(rect.height());
        }
        assert!((heights[0] - heights[1]).abs() < 1.);
    }
    #[test]
    fn concealed_window_body_preserves_extent_and_redacts_accessibility() {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut text = "Own private field".to_owned();
        let mut sizes = vec![];
        for hidden in [false, false, true, false, true] {
            let mut size = egui::Vec2::ZERO;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800., 700.),
                    )),
                    focused: !hidden,
                    ..Default::default()
                },
                |ui| {
                    ui.set_width(300.);
                    size = private_scope(ui, hidden, |ui| {
                        ui.group(|ui| {
                            ui.label("Own private name");
                            ui.text_edit_multiline(&mut text);
                        })
                    })
                    .response
                    .rect
                    .size();
                },
            );
            output.textures_delta.clear();
            if hidden {
                assert!(
                    !output
                        .platform_output
                        .accesskit_update
                        .unwrap()
                        .nodes
                        .iter()
                        .any(|(_, n)| n
                            .value()
                            .into_iter()
                            .chain(n.label())
                            .any(|t| t.contains("Own private")))
                );
            }
            sizes.push(size);
        }
        assert!(sizes[1..].windows(2).all(|v| v[0] == v[1]), "{sizes:?}");
        assert_eq!(text, "Own private field");
    }
    #[test]
    fn concealed_window_reveals_only_the_field_being_typed_in() {
        fn painted(shapes: &[egui::epaint::ClippedShape], needle: &str) -> bool {
            fn find(shape: &egui::Shape, needle: &str) -> bool {
                match shape {
                    egui::Shape::Text(text) => text.galley.text().contains(needle),
                    egui::Shape::Vec(shapes) => shapes.iter().any(|s| find(s, needle)),
                    _ => false,
                }
            }
            shapes.iter().any(|s| find(&s.shape, needle))
        }
        let ctx = egui::Context::default();
        let mut text = "Own typed field".to_owned();
        let mut field = None;
        let mut frame = |events: Vec<egui::Event>, focused: bool| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800., 700.),
                    )),
                    events,
                    focused,
                    ..Default::default()
                },
                |ui| {
                    private_scope(ui, true, |ui| {
                        ui.label("Own private name");
                        let response = ui.text_edit_singleline(&mut text);
                        if field.is_none() {
                            response.request_focus();
                            field = Some(response.id);
                        }
                    });
                },
            );
            output.textures_delta.clear();
            (
                painted(&output.shapes, "Own typed"),
                painted(&output.shapes, "Own private name"),
            )
        };
        assert_eq!(frame(vec![], true), (false, false));
        // Focus alone reveals nothing; typing reveals only the field.
        assert_eq!(frame(vec![], true), (false, false));
        assert_eq!(
            frame(vec![egui::Event::Text("!".into())], true),
            (true, false)
        );
        assert_eq!(frame(vec![], true), (true, false));
        assert_eq!(frame(vec![], false), (false, false));
        assert_eq!(frame(vec![], true), (false, false));
        assert_eq!(text, "Own typed field!");
    }
    #[test]
    fn confirm_offers_named_actions_and_blocks_with_a_reason() {
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut output = ctx.run_ui(Default::default(), |ui| {
            confirm(
                ui,
                "Own question?",
                (Icon::Trash, "Own yes"),
                "Own no",
                Some("Own reason."),
            );
        });
        output.textures_delta.clear();
        let nodes = output.platform_output.accesskit_update.unwrap().nodes;
        let node = |label| {
            nodes
                .iter()
                .find(|(_, n)| n.label() == Some(label))
                .unwrap()
                .1
                .clone()
        };
        assert!(node("Own yes").is_disabled());
        assert!(!node("Own no").is_disabled());
    }
    #[test]
    fn viewport_privacy_still_hides_on_leave_and_lost_focus() {
        for (privacy, focused, inside, expected) in [
            (true, true, true, true),
            (true, true, false, false),
            (true, false, true, false),
            (false, false, false, true),
        ] {
            let ctx = egui::Context::default();
            let mut actual = false;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(900., 700.),
                    )),
                    focused,
                    events: vec![egui::Event::PointerMoved(if inside {
                        egui::pos2(100., 100.)
                    } else {
                        egui::pos2(950., 750.)
                    })],
                    ..Default::default()
                },
                |ui| {
                    actual = super::reveal_in_viewport(ui, privacy);
                },
            );
            output.textures_delta.clear();
            assert_eq!(
                actual, expected,
                "privacy={privacy}, focused={focused}, inside={inside}"
            );
        }
    }
    #[test]
    fn moving_pointer_inside_scrolled_viewport_does_not_hide_or_reset_content() {
        let ctx = egui::Context::default();
        let mut viewport = egui::Rect::NOTHING;
        let mut shown = false;
        let mut top = 0.;
        let mut frame = |pos: egui::Pos2, wheel: f32| {
            let mut events = vec![egui::Event::PointerMoved(pos)];
            if wheel != 0. {
                events.push(egui::Event::MouseWheel {
                    phase: egui::TouchPhase::Move,
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0., wheel),
                    modifiers: egui::Modifiers::NONE,
                });
            }
            shown = false;
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(900., 700.),
                    )),
                    focused: true,
                    events,
                    ..Default::default()
                },
                |ui| {
                    egui::Window::new("Synthetic references")
                        .fixed_pos(egui::pos2(30., 30.))
                        .fixed_size(egui::vec2(500., 320.))
                        .vscroll(true)
                        .show(ui.ctx(), |ui| {
                            viewport = ui.clip_rect();
                            if !super::reveal_in_viewport(ui, true) {
                                ui.label("Hidden");
                                return;
                            }
                            shown = true;
                            top = ui.cursor().min.y;
                            for i in 0..80 {
                                ui.checkbox(&mut false, format!("Synthetic image {i}"));
                            }
                        });
                },
            );
            output.textures_delta.clear();
            (shown, top, viewport)
        };
        for _ in 0..4 {
            frame(egui::pos2(150., 100.), 0.);
        }
        let (_, initial, rect) = frame(egui::pos2(150., 100.), 0.);
        let position = egui::pos2(rect.center().x, rect.bottom() - 20.);
        frame(position, -750.);
        let mut last = 0.;
        for _ in 0..60 {
            let (visible, y, _) = frame(position, 0.);
            assert!(
                visible,
                "Scrolled content hid while the pointer was inside its viewport"
            );
            last = y;
        }
        assert!(
            last < initial - 200.,
            "Test did not scroll far enough: {initial} -> {last}"
        );
        for x in [rect.left() + 20., rect.center().x, rect.right() - 20.] {
            let (visible, y, _) = frame(egui::pos2(x, position.y), 0.);
            assert!(
                visible,
                "Moving the pointer inside the dialog hid its content"
            );
            assert!(
                (y - last).abs() < 2.,
                "Pointer motion reset scroll: {last} -> {y}"
            );
        }
    }
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
