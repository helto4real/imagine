//! Visual catalog editing writes canonical schema 1.1 and preserves named links.
use super::*;
fn namespace() -> Value {
    json!({"key":"new","label":"New namespace","description":"New prompt catalog","choices":[choice()]})
}
fn choice() -> Value {
    json!({"key":"phrase","aliases":[],"label":"New phrase","append":"none","segments":[{"text":"New phrase"}]})
}
pub(super) fn starter() -> String {
    imagine_video_contract::prompt_catalog::source_text(
        &json!({"schema_version":"1.1","mode":"merge","namespaces":[namespace()]}),
    )
    .unwrap()
}
fn at<'a>(value: &'a mut Value, path: &[(String, usize)]) -> Option<&'a mut Value> {
    let mut value = value;
    for (field, index) in path {
        value = value.get_mut(field)?.get_mut(*index)?;
    }
    Some(value)
}
fn field(ui: &mut egui::Ui, value: &mut Value, key: &str, label: &str) -> bool {
    ui.label(label);
    let mut text = value[key].as_str().unwrap_or("").to_owned();
    let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(f32::INFINITY));
    control(ui, &format!("catalog-form-{key}"), &r);
    if r.changed() {
        if key == "description" && text.is_empty() {
            value.as_object_mut().unwrap().remove(key);
        } else {
            value[key] = json!(text);
        }
        true
    } else {
        false
    }
}
fn tree(
    ui: &mut egui::Ui,
    node: &Value,
    path: Vec<(String, usize)>,
    selected: &mut Vec<(String, usize)>,
) {
    let name = node["label"]
        .as_str()
        .filter(|s| !s.is_empty())
        .or_else(|| node["key"].as_str())
        .unwrap_or("Untitled");
    let r = ui.selectable_label(*selected == path, name);
    control(
        ui,
        &format!("catalog-form-select-{}", node["key"].as_str().unwrap_or("")),
        &r,
    );
    if r.clicked() {
        *selected = path.clone();
    }
    ui.indent(egui::Id::new(&path), |ui| {
        for field in ["groups", "choices"] {
            for (i, next) in node[field].as_array().into_iter().flatten().enumerate() {
                let mut p = path.clone();
                p.push((field.into(), i));
                tree(ui, next, p, selected);
            }
        }
    });
}
pub(super) fn show(ui: &mut egui::Ui, state: &mut State, privacy: bool) {
    if state.form.is_none() {
        match imagine_video_contract::prompt_catalog::source_value(&state.draft) {
            Ok(mut v) => {
                canonical_form(&mut v);
                state.form = Some(v);
            }
            Err(e) => {
                ui.colored_label(helto::ERROR, e);
                return;
            }
        }
    }
    let value = state.form.as_mut().unwrap();
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        ui.weak("Schema 1.1 · Form edits save canonical YAML");
        {
            let (key, label, options) = ("mode", "Mode", ["replace", "merge"]);
            ui.label(label);
            let current = value[key].as_str().unwrap_or("");
            let mut selected = current.to_owned();
            egui::ComboBox::from_id_salt(key)
                .selected_text(&selected)
                .show_ui(ui, |ui| {
                    for option in options {
                        ui.selectable_value(&mut selected, option.into(), option);
                    }
                });
            if selected != current {
                value[key] = json!(selected);
                changed = true;
            }
        }
        let r = helto::icon_button(ui, crate::icons::Icon::Plus, "Add Namespace", false);
        control(ui, "catalog-form-add-namespace", &r);
        if r.clicked() {
            if !value["namespaces"].is_array() {
                value["namespaces"] = json!([]);
            }
            let nodes = value["namespaces"].as_array_mut().unwrap();
            nodes.push(namespace());
            state.form_path = vec![("namespaces".into(), nodes.len() - 1)];
            changed = true;
        }
    });
    let snapshot = value.clone();
    egui::ScrollArea::vertical()
        .id_salt("catalog-form-tree")
        .max_height(130.)
        .show(ui, |ui| {
            for (i, n) in snapshot["namespaces"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                tree(ui, n, vec![("namespaces".into(), i)], &mut state.form_path);
            }
        });
    if state.form_path.is_empty()
        && value["namespaces"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
    {
        state.form_path = vec![("namespaces".into(), 0)];
    }
    let version = value["schema_version"].as_str().unwrap_or("1.1").to_owned();
    let composer_key = ui.make_persistent_id((
        &state.folder,
        &state.selected,
        &state.form_path,
        "composer-raw",
    ));
    let mut remove = false;
    let mut duplicate = false;
    if let Some(node) = at(value, &state.form_path) {
        ui.separator();
        changed |= field(ui, node, "key", "Key");
        changed |= field(ui, node, "label", "Label");
        if node.get("segments").is_none() || version == "1.1" {
            changed |= field(ui, node, "description", "Description");
        }
        if node.get("segments").is_some() {
            let r =
                helto::icon_button(ui, crate::icons::Icon::Duplicate, "Duplicate Phrase", false);
            control(ui, "catalog-duplicate-phrase", &r);
            duplicate = r.clicked();
            ui.label("Aliases · comma separated");
            let mut aliases = node["aliases"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            if ui.text_edit_singleline(&mut aliases).changed() {
                node["aliases"] = json!(
                    aliases
                        .split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                );
                changed = true;
            }
            let old = node["append"].as_str().unwrap_or("none");
            let mut append = old.to_owned();
            ui.horizontal(|ui| {
                ui.label("Append");
                for v in ["none", "space"] {
                    ui.selectable_value(&mut append, v.into(), v);
                }
            });
            if append != old {
                node["append"] = json!(append);
                changed = true;
            }
            ui.label("Phrase Composer");
            changed |= composer(ui, node, composer_key, privacy);
            if version == "1.1" {
                if !node["placeholders"].is_object() {
                    node["placeholders"] = json!({});
                }
                ui.collapsing("Named placeholders", |ui| {
                    let definitions = node["placeholders"].as_object_mut().unwrap();
                    let mut renamed = None;
                    let mut deleted = None;
                    for (id, d) in definitions.iter_mut() {
                        ui.push_id(id, |ui| {
                            ui.group(|ui| {
                                let mut key = id.clone();
                                ui.label("ID");
                                if ui.text_edit_singleline(&mut key).changed() {
                                    renamed = Some((id.clone(), key));
                                }
                                for (k, l) in [
                                    ("label", "Label"),
                                    ("initial_text", "Initial text"),
                                    ("hint", "Hint"),
                                ] {
                                    changed |= field(ui, d, k, l);
                                }
                                if helto::icon_button(
                                    ui,
                                    crate::icons::Icon::Trash,
                                    "Remove Placeholder",
                                    false,
                                )
                                .clicked()
                                {
                                    deleted = Some(id.clone());
                                }
                            });
                        });
                    }
                    if let Some((old, new)) = renamed
                        && old != new
                        && !definitions.contains_key(&new)
                    {
                        let d = definitions.remove(&old).unwrap();
                        definitions.insert(new.clone(), d);
                        for s in node["segments"].as_array_mut().unwrap() {
                            if s["placeholder_ref"] == old {
                                s["placeholder_ref"] = json!(new);
                            }
                        }
                        changed = true;
                    }
                    if let Some(id) = deleted {
                        node["placeholders"].as_object_mut().unwrap().remove(&id);
                        changed = true;
                    }
                });
            }
            let mut movement = None;
            let mut deletion = None;
            let refs = node["placeholders"]
                .as_object()
                .map(|v| v.keys().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            let segments = node["segments"].as_array_mut().unwrap();
            for (i, segment) in segments.iter_mut().enumerate() {
                ui.push_id(i, |ui| {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.strong(format!("Segment {}", i + 1));
                            for (icon, label, delta) in [
                                (crate::icons::Icon::Up, "Move Segment Up", -1),
                                (crate::icons::Icon::Down, "Move Segment Down", 1),
                            ] {
                                if helto::icon_button(ui, icon, label, false).clicked() {
                                    movement = Some((i, delta));
                                }
                            }
                            if helto::icon_button(
                                ui,
                                crate::icons::Icon::Trash,
                                "Remove Segment",
                                false,
                            )
                            .clicked()
                            {
                                deletion = Some(i);
                            }
                        });
                        if segment.get("text").is_some() {
                            changed |= field(ui, segment, "text", "Text");
                        } else if segment.get("placeholder_ref").is_some() {
                            let mut id =
                                segment["placeholder_ref"].as_str().unwrap_or("").to_owned();
                            let old = id.clone();
                            egui::ComboBox::from_id_salt("ref")
                                .selected_text(&id)
                                .show_ui(ui, |ui| {
                                    for key in &refs {
                                        ui.selectable_value(&mut id, key.clone(), key);
                                    }
                                });
                            if id != old {
                                segment["placeholder_ref"] = json!(id);
                                changed = true;
                            }
                        } else if let Some(p) = segment.get_mut("placeholder") {
                            for (k, l) in [
                                ("id", "ID"),
                                ("label", "Label"),
                                ("initial_text", "Initial text"),
                                ("hint", "Hint"),
                            ] {
                                changed |= field(ui, p, k, l);
                            }
                        }
                    });
                });
            }
            if let Some(i) = deletion {
                segments.remove(i);
                changed = true;
            }
            if let Some((i, delta)) = movement {
                let j = i as isize + delta;
                if j >= 0 && (j as usize) < segments.len() {
                    segments.swap(i, j as usize);
                    changed = true;
                }
            }
            ui.horizontal(|ui| {
                if helto::icon_button(ui, crate::icons::Icon::Plus, "Add Text Segment", false)
                    .clicked()
                {
                    segments.push(json!({"text":"Text"}));
                    changed = true;
                }
            });
            if helto::icon_button(
                ui,
                crate::icons::Icon::CharacterSheet,
                "Add Placeholder Segment",
                false,
            )
            .clicked()
            {
                let mut n = 1;
                let id = loop {
                    let id = format!("slot{n}");
                    if !refs.contains(&id)
                        && !node["segments"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .any(|s| s["placeholder"]["id"] == id)
                    {
                        break id;
                    }
                    n += 1;
                };
                if version == "1.1" {
                    node["placeholders"][&id] = json!({"label":"Placeholder","initial_text":"value","hint":"Replace this value"});
                    node["segments"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"placeholder_ref":id}));
                } else {
                    node["segments"].as_array_mut().unwrap().push(json!({"placeholder":{"id":id,"label":"Placeholder","initial_text":"value","hint":"Replace this value"}}));
                }
                changed = true;
            }
        } else {
            ui.horizontal(|ui| {
                for (name, label, entry) in [
                    ("choices", "Add Choice", choice()),
                    (
                        "groups",
                        "Add Group",
                        json!({"key":"group","label":"New group","choices":[choice()]}),
                    ),
                ] {
                    if ui
                        .push_id(name, |ui| {
                            helto::icon_button(ui, crate::icons::Icon::Plus, label, false)
                        })
                        .inner
                        .clicked()
                    {
                        if !node[name].is_array() {
                            node[name] = json!([]);
                        }
                        node[name].as_array_mut().unwrap().push(entry);
                        changed = true;
                    }
                }
            });
        }
        remove = helto::icon_button(ui, crate::icons::Icon::Trash, "Remove Catalog Entry", false)
            .clicked();
    }
    if duplicate {
        let mut parent = state.form_path.clone();
        if let Some((field, index)) = parent.pop()
            && field == "choices"
            && let Some(container) = at(value, &parent)
            && let Some(i) =
                imagine_video_contract::prompt_catalog::composer::duplicate(container, index)
        {
            parent.push(("choices".into(), i));
            state.form_path = parent;
            changed = true;
        }
    }
    if remove
        && let Some((key, index)) = state.form_path.pop()
        && let Some(parent) = at(value, &state.form_path)
            .and_then(|p| p.get_mut(key))
            .and_then(Value::as_array_mut)
    {
        parent.remove(index);
        changed = true;
    }
    if changed {
        match imagine_video_contract::prompt_catalog::source_text(value) {
            Ok(t) => {
                state.draft = t;
                state.validation = None;
            }
            Err(e) => state.error = Some(e),
        }
    }
}

#[derive(Clone)]
struct ComposerDraft {
    segments: Value,
    text: String,
    known: Vec<String>,
    suggestions: Vec<String>,
    context: Option<(usize, usize)>,
    selected: usize,
    widget: Option<egui::Id>,
}

fn composer(ui: &mut egui::Ui, node: &mut Value, key: egui::Id, privacy: bool) -> bool {
    use imagine_video_contract::prompt_catalog::composer as pc;
    let mut draft = ui
        .data_mut(|d| d.get_temp::<ComposerDraft>(key))
        .filter(|d| d.segments == node["segments"])
        .unwrap_or_else(|| ComposerDraft {
            segments: node["segments"].clone(),
            text: pc::text(node),
            known: node["placeholders"]
                .as_object()
                .map(|o| o.keys().cloned().collect())
                .unwrap_or_default(),
            suggestions: vec![],
            context: None,
            selected: 0,
            widget: None,
        });
    fn insert(ui: &egui::Ui, d: &mut ComposerDraft, index: usize) -> bool {
        let (Some((start, end)), Some(id), Some(widget)) =
            (d.context, d.suggestions.get(index), d.widget)
        else {
            return false;
        };
        let before = d.text.chars().take(start).collect::<String>();
        let after = d.text.chars().skip(end).collect::<String>();
        let caret = start + 1 + id.chars().count();
        d.text = format!("{before}@{id}{after}");
        super::super::prompt_tools::cursor(ui, widget, caret, caret);
        ui.memory_mut(|m| m.request_focus(widget));
        d.suggestions.clear();
        d.context = None;
        true
    }
    let mut changed = false;
    if draft
        .widget
        .is_some_and(|id| ui.memory(|m| m.has_focus(id)))
        && !draft.suggestions.is_empty()
    {
        let mut choose = false;
        ui.input_mut(|i| {
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) {
                draft.selected = (draft.selected + 1) % draft.suggestions.len();
            }
            if i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) {
                draft.selected =
                    (draft.selected + draft.suggestions.len() - 1) % draft.suggestions.len();
            }
            choose = i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
                || i.consume_key(egui::Modifiers::NONE, egui::Key::Tab);
            if i.consume_key(egui::Modifiers::NONE, egui::Key::Escape) {
                draft.suggestions.clear();
                draft.context = None;
            }
        });
        if choose {
            let selected = draft.selected;
            changed |= insert(ui, &mut draft, selected);
        }
    }
    let response = ui
        .push_id(key, |ui| {
            helto::textarea(
                ui,
                &mut draft.text,
                4,
                "Type prompt text; @identity creates an editable placeholder, @@ is a literal @",
                privacy,
                true,
            )
        })
        .inner;
    draft.widget = Some(response.id);
    control(ui, "catalog-form-composer", &response);
    if response.changed() || response.clicked() {
        let range =
            egui::TextEdit::load_state(ui.ctx(), response.id).and_then(|s| s.cursor.char_range());
        if let Some(range) = range
            && range.is_empty()
        {
            let caret: usize = range.primary.index.into();
            if let Some((start, query)) = pc::completion(&draft.text, caret) {
                draft.context = Some((start, caret));
                draft.suggestions = draft
                    .known
                    .iter()
                    .filter(|id| query.is_empty() || id.starts_with(&query))
                    .take(30)
                    .cloned()
                    .collect();
                draft.selected = 0;
            } else {
                draft.suggestions.clear();
                draft.context = None;
            }
        } else {
            draft.suggestions.clear();
            draft.context = None;
        }
    }
    changed |= response.changed();
    if !draft.suggestions.is_empty() && response.has_focus() && helto::shown(ui, privacy) {
        let mut pick = None;
        egui::Popup::from_response(&response)
            .id(key.with("identities"))
            .show(|ui| {
                for (i, id) in draft.suggestions.iter().enumerate() {
                    let r = ui.selectable_label(i == draft.selected, format!("@{id}"));
                    control(ui, &format!("composer-identity-{id}"), &r);
                    if r.clicked() {
                        pick = Some(i);
                    }
                }
            });
        if let Some(i) = pick {
            changed |= insert(ui, &mut draft, i);
        }
    }
    if response.has_focus() {
        let active = !draft.suggestions.is_empty();
        ui.memory_mut(|m| {
            m.set_focus_lock_filter(
                response.id,
                egui::EventFilter {
                    horizontal_arrows: true,
                    vertical_arrows: true,
                    escape: active,
                    tab: active,
                },
            )
        });
    } else {
        draft.suggestions.clear();
        draft.context = None;
    }
    if changed {
        let parsed = pc::parse(&draft.text, &node["placeholders"]);
        node["placeholders"] = parsed["placeholders"].clone();
        node["segments"] = parsed["segments"].clone();
        draft.known = node["placeholders"]
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
    }
    draft.segments = node["segments"].clone();
    ui.data_mut(|d| d.insert_temp(key, draft));
    changed
}

/// The form edits Director's normalized editor document.
pub(super) fn canonical_form(value: &mut Value) {
    *value = imagine_video_contract::prompt_catalog::editor::normalize(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn form_roundtrip_keeps_absent_arrays_absent_and_inline_slots_loadable() {
        let mut source = imagine_video_contract::prompt_catalog::source_value(&starter()).unwrap();
        canonical_form(&mut source);
        let text = imagine_video_contract::prompt_catalog::source_text(&source).unwrap();
        assert!(!text.contains("null"));
        imagine_video_contract::prompt_catalog::load(&BTreeMap::from([("own.yaml".into(), text)]))
            .unwrap();
        source["schema_version"] = json!("1.0");
        source["namespaces"][0]["choices"][0]["segments"] = json!([{"placeholder":{"id":"who","label":"Person","initial_text":"@keeper","hint":"Choose person"}}]);
        canonical_form(&mut source);
        let text = imagine_video_contract::prompt_catalog::source_text(&source).unwrap();
        let c = imagine_video_contract::prompt_catalog::load(&BTreeMap::from([(
            "own.yaml".into(),
            text,
        )]))
        .unwrap();
        assert_eq!(
            c.namespaces[0].choices[0].expand(&BTreeMap::new()).text,
            "@keeper"
        );
    }
}
