//! Cursor-aware reference completion. Catalogs and provider proposals use separate sources.
use super::*;
use imagine_video_contract::creative::Creative;
use imagine_video_contract::prompt_catalog as pc;

#[derive(Clone)]
struct Completion {
    id: String,
    label: String,
    tag: String,
    description: String,
    kind: String,
    catalog: Option<pc::Completion>,
}
#[derive(Clone, Default)]
struct Menu {
    widget: Option<egui::Id>,
    range: Option<(usize, usize)>,
    query: String,
    items: Vec<Completion>,
    selected: usize,
    dismissed: bool,
    popup: Option<egui::Rect>,
    context: Option<pc::Context>,
    /// The keyboard moved the selection; scroll it into view.
    reveal: bool,
}
impl Completion {
    fn row(&self) -> String {
        if self.catalog.is_some() {
            format!("{} · {}", self.label, self.kind)
        } else {
            format!("{} · {} · {}", self.tag, self.label, self.kind)
        }
    }
}
/// Popup content width that fits the longest row on one line: at least the
/// field's width, at most the window's. Rows wrap only when the window is narrower.
fn popup_width(ui: &egui::Ui, items: &[Completion], field: f32) -> f32 {
    let style = ui.style();
    let font = egui::TextStyle::Button.resolve(style);
    let widest = ui.fonts_mut(|f| {
        items
            .iter()
            .map(|item| {
                f.layout_no_wrap(item.row(), font.clone(), egui::Color32::PLACEHOLDER)
                    .size()
                    .x
            })
            .fold(0., f32::max)
    });
    let frame = style.spacing.menu_margin.sum().x + 2.;
    let row = widest + 2. * style.spacing.button_padding.x + style.spacing.scroll.allocated_width();
    row.max(field.max(220.) - frame)
        .min(ui.ctx().content_rect().width() - 16. - frame)
}

pub(super) fn reference_tag(alias: &str) -> String {
    format!(
        "@{}",
        imagine_video_contract::creative::reference_alias(alias)
    )
}
fn js_space(c: char) -> bool {
    matches!(c,'\u{0009}'..='\u{000d}' | ' ' | '\u{00a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}' | '\u{feff}')
}
fn context(text: &str, caret: usize) -> Option<(usize, String)> {
    let chars = text.chars().take(caret).collect::<Vec<_>>();
    let start = chars.iter().rposition(|c| *c == '@')?;
    if start > 0 && !js_space(chars[start - 1]) && !"([{,".contains(chars[start - 1]) {
        return None;
    }
    let query = chars[start + 1..].iter().collect::<String>();
    (!query.chars().any(js_space)).then_some((start, query))
}
fn completions(creative: &Creative, query: &str) -> Vec<Completion> {
    if creative.visual_references_enabled == Some(false) {
        return vec![];
    }
    let needle = query.trim_matches(js_space).to_lowercase();
    creative
        .references
        .iter()
        .filter(|r| r.enabled)
        .filter(|r| {
            [
                &r.name,
                &r.alias,
                &r.description,
                &r.kind,
                &reference_tag(&r.alias),
            ]
            .iter()
            .any(|s| s.to_lowercase().contains(&needle))
        })
        .map(|r| Completion {
            id: r.id.clone(),
            label: if r.name.is_empty() {
                r.alias.clone()
            } else {
                r.name.clone()
            },
            tag: reference_tag(&r.alias),
            description: r.description.clone(),
            kind: r.kind.clone(),
            catalog: None,
        })
        .take(50)
        .collect()
}
fn insert(text: &mut String, start: usize, end: usize, tag: &str) -> usize {
    let before = text.chars().take(start).collect::<String>();
    let after = text.chars().skip(end).collect::<String>();
    let needs_space = after.chars().next().is_none_or(|c| !js_space(c));
    let caret = start + tag.chars().count() + usize::from(needs_space || after.starts_with(' '));
    *text = format!("{before}{tag}{}{after}", if needs_space { " " } else { "" });
    caret
}
#[derive(Clone, Default)]
struct Snippets {
    sessions: Vec<pc::Session>,
    text: String,
}
pub(super) fn cursor(ui: &egui::Ui, widget: egui::Id, start: usize, end: usize) {
    if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), widget) {
        state
            .cursor
            .set_char_range(Some(egui::text::CCursorRange::two(
                egui::text::CCursor::new(start),
                egui::text::CCursor::new(end),
            )));
        state.store(ui.ctx(), widget);
    }
}
fn expansion(
    ui: &egui::Ui,
    widget: egui::Id,
    text: &mut String,
    start: usize,
    end: usize,
    expanded: pc::Expansion,
) {
    let before = text.clone();
    let prefix = text.chars().take(start).collect::<String>();
    let suffix = text.chars().skip(end).collect::<String>();
    *text = format!("{prefix}{}{suffix}", expanded.text);
    let key = widget.with("catalog-snippets");
    let mut state = ui.data_mut(|d| d.get_temp::<Snippets>(key).unwrap_or_default());
    if !state.sessions.iter_mut().all(|s| s.track(&before, text)) {
        state.sessions.clear();
    }
    let session = pc::Session::new(expanded.slots, start);
    let selected = session.selection();
    if !session.slots.is_empty() {
        state.sessions.push(session);
    }
    state.text = text.clone();
    ui.data_mut(|d| d.insert_temp(key, state));
    if let Some(o) = selected {
        cursor(ui, widget, o.start, o.end);
    } else {
        let caret = start + expanded.text.chars().count();
        cursor(ui, widget, caret, caret);
    }
    ui.memory_mut(|m| m.request_focus(widget));
}
fn apply(ui: &egui::Ui, widget: egui::Id, text: &mut String, menu: &Menu, index: usize) -> bool {
    let (Some((start, end)), Some(item)) = (menu.range, menu.items.get(index)) else {
        return false;
    };
    if let Some(expected) = &menu.context {
        if pc::context(text, end, end).as_ref() != Some(expected) {
            return false;
        }
        let entry = item.catalog.as_ref().unwrap();
        if let Some(choice) = &entry.choice {
            expansion(
                ui,
                widget,
                text,
                start,
                end,
                choice.expand(&std::collections::BTreeMap::new()),
            );
        } else {
            let prefix = text.chars().take(start).collect::<String>();
            let after = text.chars().skip(end).collect::<String>();
            let tag = format!("#{}.", entry.path.join("."));
            *text = format!("{prefix}{tag}{after}");
            let caret = start + tag.chars().count();
            cursor(ui, widget, caret, caret);
        }
    } else {
        if context(text, end) != Some((start, menu.query.clone())) {
            return false;
        }
        let caret = insert(text, start, end, &item.tag);
        cursor(ui, widget, caret, caret);
    }
    ui.memory_mut(|m| m.request_focus(widget));
    true
}
fn move_snippet(ui: &egui::Ui, widget: egui::Id, text: &mut String, back: bool) -> bool {
    let key = widget.with("catalog-snippets");
    let mut state = ui.data_mut(|d| d.get_temp::<Snippets>(key).unwrap_or_default());
    if state.sessions.is_empty() {
        return false;
    }
    let mut selection = None;
    let mut handled = true;
    while let Some(mut current) = state.sessions.pop() {
        let before = text.clone();
        current.commit(text);
        if !state.sessions.iter_mut().all(|p| p.track(&before, text)) {
            state.sessions.clear();
        }
        if back {
            if current.active > 0 {
                current.active -= 1;
                selection = current.selection();
                state.sessions.push(current);
            } else if let Some(parent) = state.sessions.last() {
                selection = parent.selection();
            } else {
                handled = false;
            }
            break;
        } else if current.active + 1 < current.slots.len() {
            current.active += 1;
            selection = current.selection();
            state.sessions.push(current);
            break;
        }
    }
    state.text = text.clone();
    ui.data_mut(|d| d.insert_temp(key, state));
    if let Some(o) = selection {
        cursor(ui, widget, o.start, o.end);
    } else {
        let end = text.chars().count();
        cursor(ui, widget, end, end);
    }
    if !handled {
        ui.memory_mut(|m| m.move_focus(egui::FocusDirection::Previous));
    }
    handled
}
pub(super) struct Source<'a> {
    pub creative: &'a Creative,
    pub scope: egui::Id,
}
impl<'a> Source<'a> {
    pub fn project(creative: &'a Creative, project: &ProjectId) -> Self {
        Self {
            creative,
            scope: egui::Id::new(&project.0),
        }
    }
    pub fn scene(creative: &'a Creative, project: &ProjectId, scene: &SceneId) -> Self {
        Self {
            creative,
            scope: egui::Id::new((&project.0, &scene.0)),
        }
    }
    pub fn section(creative: &'a Creative, project: &ProjectId, section: &SectionId) -> Self {
        Self {
            creative,
            scope: egui::Id::new((&project.0, &section.0)),
        }
    }
}

pub(super) fn textarea(
    ui: &mut egui::Ui,
    text: &mut String,
    rows: usize,
    id: &str,
    privacy: bool,
    resizable: bool,
    source: Source<'_>,
) -> egui::Response {
    ui.push_id(source.scope, |ui| {
        textarea_inner(ui, text, rows, id, privacy, resizable, source.creative)
    })
    .inner
}
fn textarea_inner(
    ui: &mut egui::Ui,
    text: &mut String,
    rows: usize,
    id: &str,
    privacy: bool,
    resizable: bool,
    creative: &Creative,
) -> egui::Response {
    let key = ui.make_persistent_id((id, "reference-completion"));
    let mut menu = ui.data_mut(|d| d.get_temp::<Menu>(key).unwrap_or_default());
    let mut changed = false;
    if let Some(widget) = menu.widget {
        if let Some(insertion) = super::catalog::pending(ui.ctx(), widget) {
            if *text == insertion.target.text {
                expansion(
                    ui,
                    widget,
                    text,
                    insertion.target.start,
                    insertion.target.end,
                    insertion.expansion,
                );
                changed = true;
                menu.items.clear();
                menu.range = None;
            } else {
                super::catalog::insertion_error(ui.ctx());
            }
        }
        let snippets = ui
            .data(|d| d.get_temp::<Snippets>(widget.with("catalog-snippets")))
            .unwrap_or_default();
        if ui.memory(|m| m.has_focus(widget)) && !snippets.sessions.is_empty() {
            let back = ui.input_mut(|i| i.consume_key(egui::Modifiers::SHIFT, egui::Key::Tab));
            let forward =
                !back && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Tab));
            if back || forward {
                let before = text.clone();
                move_snippet(ui, widget, text, back);
                changed |= before != *text;
                menu.dismissed = true;
            }
        }
    }

    if let Some(widget) = menu.widget
        && ui.memory(|m| m.has_focus(widget))
        && !menu.items.is_empty()
        && !menu.dismissed
    {
        let mut pick = None;
        ui.input_mut(|input| {
            let modifiers = egui::Modifiers::NONE;
            let before = menu.selected;
            if input.consume_key(modifiers, egui::Key::ArrowDown) {
                menu.selected = (menu.selected + 1) % menu.items.len();
            }
            if input.consume_key(modifiers, egui::Key::ArrowUp) {
                menu.selected = (menu.selected + menu.items.len() - 1) % menu.items.len();
            }
            if input.consume_key(modifiers, egui::Key::Home) {
                menu.selected = 0;
            }
            if input.consume_key(modifiers, egui::Key::End) {
                menu.selected = menu.items.len() - 1;
            }
            menu.reveal |= menu.selected != before;
            if input.consume_key(modifiers, egui::Key::Enter) {
                pick = Some(menu.selected);
            }
            if input.consume_key(modifiers, egui::Key::Escape) {
                menu.dismissed = true;
            }
            if input.key_pressed(egui::Key::Tab) {
                menu.dismissed = true;
            }
            input.events.retain(|event| {
                if matches!(event,egui::Event::Text(value) if value==".")
                    && menu.items[menu.selected]
                        .catalog
                        .as_ref()
                        .is_some_and(|c| c.choice.is_none())
                {
                    pick = Some(menu.selected);
                    return false;
                }

                if let egui::Event::Text(value) = event
                    && value.len() == 1
                    && let Some(digit) = value.chars().next().and_then(|c| c.to_digit(10))
                    && digit > 0
                    && menu.context.is_none()
                    && (digit as usize) <= menu.items.len()
                {
                    pick = Some(digit as usize - 1);
                    return false;
                }
                true
            });
        });
        if let Some(index) = pick {
            changed = apply(ui, widget, text, &menu, index);
            menu.dismissed = menu.items[index]
                .catalog
                .as_ref()
                .is_none_or(|c| c.choice.is_some());
        }
    }
    let mut response = helto::textarea(ui, text, rows, id, privacy, resizable);
    menu.widget = Some(response.id);
    let range =
        egui::TextEdit::load_state(ui.ctx(), response.id).and_then(|s| s.cursor.char_range());
    let pointer_in_popup = menu.popup.is_some_and(|rect| {
        ui.input(|i| i.focused && i.pointer.hover_pos().is_some_and(|p| rect.contains(p)))
    });
    let caret = range
        .filter(|r| r.is_empty())
        .map(|r| usize::from(r.primary.index));
    if let Some(caret) = caret.filter(|_| response.has_focus() || pointer_in_popup) {
        let reference = context(text, caret);
        let catalog_context = pc::context(text, caret, caret);
        if let Some((start, query)) = reference {
            if menu.range != Some((start, caret)) || menu.query != query || response.changed() {
                menu.dismissed = false;
                menu.selected = 0;
            }
            menu.range = Some((start, caret));
            menu.query = query;
            menu.context = None;
            menu.items = completions(creative, &menu.query);
        } else if let Some(context) = catalog_context {
            if menu.context.as_ref() != Some(&context) || response.changed() {
                menu.dismissed = false;
                menu.selected = 0;
            }
            menu.items = super::catalog::resolved(ui.ctx())
                .map(|catalog| catalog.completions(&context))
                .unwrap_or_default()
                .into_iter()
                .map(|c| Completion {
                    id: format!("catalog-{}", c.path.join(".")),
                    label: c.label.clone(),
                    tag: String::new(),
                    description: c.detail.clone(),
                    kind: if c.choice.is_some() {
                        "Phrase"
                    } else {
                        "Category"
                    }
                    .into(),
                    catalog: Some(c),
                })
                .collect();
            menu.range = Some((context.start, context.end));
            menu.query = String::new();
            menu.context = Some(context);
        } else {
            menu.items.clear();
            menu.range = None;
            menu.context = None;
        }
        menu.selected = menu.selected.min(menu.items.len().saturating_sub(1));
    } else {
        menu.items.clear();
        menu.range = None;
        menu.context = None;
        menu.popup = None;
    }
    if changed && menu.context.is_none() {
        menu.dismissed = true;
    }
    if !menu.dismissed && !menu.items.is_empty() && helto::shown(ui, privacy) {
        let mut pick = None;
        let width = popup_width(ui, &menu.items, response.rect.width());
        let reveal = std::mem::take(&mut menu.reveal);
        if let Some(popup) = egui::Popup::from_response(&response)
            .id(key.with("popup"))
            .show(|ui| {
                // The area keeps last frame's size; set this frame's width.
                ui.set_width(width);
                egui::ScrollArea::vertical()
                    .max_height(220.)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for (i, item) in menu.items.iter().enumerate() {
                            let r = ui
                                .push_id(&item.id, |ui| {
                                    ui.add(
                                        egui::Button::selectable(i == menu.selected, item.row())
                                            .wrap_mode(egui::TextWrapMode::Wrap)
                                            .min_size(egui::vec2(ui.available_width(), 0.)),
                                    )
                                })
                                .inner;
                            #[cfg(feature = "ui-preview")]
                            super::qa::control(ui, format!("reference-completion-{}", item.id), &r);
                            if reveal && i == menu.selected {
                                r.scroll_to_me(None);
                            }
                            let r = r.on_hover_text(&item.description);
                            if r.clicked() {
                                pick = Some(i);
                            }
                        }
                    });
            })
        {
            menu.popup = Some(popup.response.rect);
        }
        if let Some(index) = pick {
            changed |= apply(ui, response.id, text, &menu, index);
            menu.dismissed = menu.items[index]
                .catalog
                .as_ref()
                .is_none_or(|c| c.choice.is_some());
        }
    }
    let snippet_key = response.id.with("catalog-snippets");
    let mut snippets = ui.data_mut(|d| d.get_temp::<Snippets>(snippet_key).unwrap_or_default());
    if !response.has_focus() {
        snippets.sessions.clear();
    }
    if !snippets.sessions.is_empty() {
        let old = snippets.text.clone();
        if !snippets
            .sessions
            .iter_mut()
            .all(|session| session.track(&old, text))
        {
            snippets.sessions.clear();
        }
        if let Some(range) =
            egui::TextEdit::load_state(ui.ctx(), response.id).and_then(|s| s.cursor.char_range())
        {
            let a: usize = range.primary.index.into();
            let b: usize = range.secondary.index.into();
            if response.clicked() {
                if let Some(index) = snippets
                    .sessions
                    .iter_mut()
                    .rposition(|s| s.activate(a.min(b), a.max(b)))
                {
                    snippets.sessions.truncate(index + 1);
                } else {
                    snippets.sessions.clear();
                }
            } else if let Some(session) = snippets.sessions.last_mut() {
                session.activate(a.min(b), a.max(b));
            }
        }
        snippets.text = text.clone();
    }
    let snippet_active = !snippets.sessions.is_empty();
    ui.data_mut(|d| d.insert_temp(snippet_key, snippets));
    if response.has_focus() && (snippet_active || !menu.dismissed && !menu.items.is_empty()) {
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                response.id,
                egui::EventFilter {
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: !menu.dismissed && !menu.items.is_empty(),
                    tab: snippet_active,
                },
            )
        });
    }
    let frame = ui.ctx().cumulative_frame_nr();
    ui.data_mut(|d| {
        d.insert_temp(
            response.id.with("prompt-live-target"),
            (frame, text.clone()),
        )
    });
    if response.has_focus()
        && let Some(range) =
            egui::TextEdit::load_state(ui.ctx(), response.id).and_then(|s| s.cursor.char_range())
    {
        let a: usize = range.primary.index.into();
        let b: usize = range.secondary.index.into();
        super::catalog::active(
            ui.ctx(),
            super::catalog::Target {
                widget: response.id,
                text: text.clone(),
                start: a.min(b),
                end: a.max(b),
            },
        );
    }
    if changed {
        response.mark_changed();
        ui.ctx().request_repaint();
    }
    ui.data_mut(|d| d.insert_temp(key, menu));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separate_prompt_fields_conceal_together_without_geometry_change() {
        fn visible(shape: &egui::Shape) -> bool {
            match shape {
                egui::Shape::Text(t) => t.opacity_factor > 0.,
                egui::Shape::Vec(v) => v.iter().any(visible),
                _ => false,
            }
        }
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let creative = Creative::default();
        let mut description = "Own synthetic description".to_owned();
        let mut section = "Own synthetic Section".to_owned();
        let mut bounds = Vec::new();
        for (privacy, focused, expected) in [
            (false, true, true),
            (true, false, false),
            (false, true, true),
            (true, false, false),
        ] {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800., 700.),
                    )),
                    focused,
                    ..Default::default()
                },
                |ui| {
                    ui.set_width(300.);
                    let a = textarea(
                        ui,
                        &mut description,
                        3,
                        "Own description",
                        privacy,
                        true,
                        Source {
                            creative: &creative,
                            scope: egui::Id::new("own-test"),
                        },
                    );
                    let b = textarea(
                        ui,
                        &mut section,
                        3,
                        "Own Section",
                        privacy,
                        true,
                        Source {
                            creative: &creative,
                            scope: egui::Id::new("own-test"),
                        },
                    );
                    bounds.push((a.rect.size(), b.rect.size()));
                },
            );
            output.textures_delta.clear();
            assert_eq!(output.shapes.iter().any(|s| visible(&s.shape)), expected);
            if !expected {
                assert!(
                    !output
                        .platform_output
                        .accesskit_update
                        .unwrap()
                        .nodes
                        .iter()
                        .any(|(_, n)| n.value().is_some_and(|s| s.contains("Own synthetic")))
                );
            }
        }
        assert!(bounds[1..].windows(2).all(|v| v[0] == v[1]), "{bounds:?}");
    }
    #[test]
    fn completion_popup_fits_long_rows_beyond_a_narrow_field() {
        fn texts<'a>(shape: &'a egui::Shape, out: &mut Vec<&'a egui::epaint::TextShape>) {
            match shape {
                egui::Shape::Text(t) => out.push(t),
                egui::Shape::Vec(v) => v.iter().for_each(|s| texts(s, out)),
                _ => {}
            }
        }
        let mut p = super::super::media_picker::tests::project();
        p.creative.references[0].alias = "keeper".into();
        p.creative.references[0].name =
            "Own keeper of the far northern lighthouse at the end of the long pier".into();
        let ctx = egui::Context::default();
        let mut text = "@".to_owned();
        let mut field = egui::Rect::NOTHING;
        let mut shapes = Vec::new();
        for frame in 0..4 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000., 700.),
                )),
                focused: true,
                ..Default::default()
            };
            let mut output = ctx.run_ui(input, |ui| {
                ui.set_width(300.);
                let r = textarea(
                    ui,
                    &mut text,
                    3,
                    "Own prompt",
                    false,
                    true,
                    Source {
                        creative: &p.creative,
                        scope: egui::Id::new("own-test"),
                    },
                );
                if frame == 0 {
                    r.request_focus();
                    let mut state = egui::TextEdit::load_state(ui.ctx(), r.id).unwrap_or_default();
                    state
                        .cursor
                        .set_char_range(Some(egui::text::CCursorRange::one(
                            egui::text::CCursor::new(1),
                        )));
                    state.store(ui.ctx(), r.id);
                }
                field = r.rect;
            });
            output.textures_delta.clear();
            shapes = output.shapes;
        }
        let row = shapes
            .iter()
            .find_map(|clipped| {
                let mut found = Vec::new();
                texts(&clipped.shape, &mut found);
                found
                    .into_iter()
                    .find(|t| t.galley.text().contains("far northern"))
                    .map(|t| (clipped.clip_rect, t))
            })
            .expect("the completion row is shown");
        let (clip, shape) = row;
        let rect = shape.visual_bounding_rect();
        assert_eq!(shape.galley.rows.len(), 1, "the row wraps");
        assert!(rect.width() > field.width(), "{rect:?} {field:?}");
        assert!(clip.contains_rect(rect), "clipped: {rect:?} in {clip:?}");
    }
    #[test]
    fn completion_boundaries_and_unicode_caret() {
        for (text, caret, expected) in [
            ("@AL", 3, Some((0, "AL"))),
            ("å🙂 (@ca", 7, Some((4, "ca"))),
            ("email@ca", 8, None),
            ("@@ca", 4, None),
            ("(@ca words", 10, None),
            ("x,@", 3, Some((2, ""))),
            ("x:@ca", 5, None),
        ] {
            assert_eq!(
                context(text, caret),
                expected.map(|(i, s)| (i, s.into())),
                "{text}"
            );
        }
        for (after, expected, caret) in [
            ("!", "å🙂 @alice !", 10),
            (" rest", "å🙂 @alice rest", 10),
            ("\nnext", "å🙂 @alice\nnext", 9),
        ] {
            let mut text = format!("å🙂 @al{after}");
            assert_eq!(insert(&mut text, 3, 6, "@alice"), caret);
            assert_eq!(text, expected);
        }
    }
    #[test]
    fn tags_and_active_catalog_follow_reference_source() {
        assert_eq!(reference_tag(" Alice A! "), "@alice_a");
        assert_eq!(reference_tag("123"), "@ref1");
        let mut p = super::super::media_picker::tests::project();
        let r = &mut p.creative.references[0];
        r.alias = "alice".into();
        r.description = "own blue coat".into();
        assert_eq!(completions(&p.creative, "BLUE")[0].tag, "@alice");
        p.creative.references[0].name = "Own keeper".into();
        p.creative.references[0].alias = " Alice A! ".into();
        let candidate = &completions(&p.creative, "KEEPER")[0];
        assert_eq!(candidate.tag, "@alice_a");
        assert_eq!(candidate.label, "Own keeper");
        assert_eq!(
            imagine_video_contract::creative::selected_references(
                &p.creative,
                &BTreeSet::from(["alice_a".into()])
            )[0]
            .id,
            p.creative.references[0].id
        );
        p.creative.references[0].enabled = false;
        assert!(completions(&p.creative, "").is_empty());
        p.creative.references[0].enabled = true;
        p.creative.visual_references_enabled = Some(false);
        assert!(completions(&p.creative, "").is_empty());
    }
}
