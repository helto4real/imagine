//! Director's catalog editor document (comfyui-helto-director
//! `web/timeline/prompt_catalog_editor.js`): its normalization, canonical
//! schema 1.1 YAML, and linked `@identity` placeholder merging on Form saves.
use serde_json::{Map, Value, json};

const DEFAULT_HINT: &str = "Free text or an optional Director @reference";

/// JavaScript's `String(value ?? "")`.
fn js_string(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| match v {
                Value::Null => String::new(),
                other => js_string(Some(other)),
            })
            .collect::<Vec<_>>()
            .join(","),
        Some(Value::Object(_)) => "[object Object]".into(),
    }
}
/// `String(value ?? fallback)`
fn or(value: Option<&Value>, fallback: String) -> String {
    match value {
        None | Some(Value::Null) => fallback,
        other => js_string(other),
    }
}
fn object(value: Option<&Value>) -> Map<String, Value> {
    match value {
        Some(Value::Object(map)) => map.clone(),
        _ => Map::new(),
    }
}
fn array(value: Option<&Value>) -> Vec<Value> {
    match value {
        Some(Value::Array(items)) => items.clone(),
        _ => vec![],
    }
}
fn identifier(value: &str) -> bool {
    !value.is_empty()
        && value.split(['-', '_']).all(|part| {
            !part.is_empty()
                && part
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        })
}
/// `normalizeComposerIdentity`
fn identity(value: Option<&Value>) -> String {
    let normalized = js_string(value).to_lowercase();
    if identifier(&normalized) {
        normalized
    } else {
        String::new()
    }
}
/// `labelFromIdentifier`
fn label_from(id: &str) -> String {
    id.split(['-', '_'])
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().chain(chars).collect::<String>())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}
fn placeholder(source: &Map<String, Value>, id: &str) -> Value {
    json!({
        "label": or(source.get("label"), label_from(id)),
        "initial_text": or(source.get("initial_text"), format!("@{id}")),
        "hint": or(source.get("hint"), DEFAULT_HINT.into()),
    })
}
/// `normalizeEditorChoice`
fn choice(value: &Value) -> Value {
    let source = object(Some(value));
    let mut placeholders = Map::new();
    let mut segments = vec![];
    for segment in array(source.get("segments")) {
        let segment = object(Some(&segment));
        if let Some(Value::String(text)) = segment.get("text") {
            segments.push(json!({"text": text}));
            continue;
        }
        if let Some(Value::String(id)) = segment.get("placeholder_ref") {
            segments.push(json!({"placeholder_ref": id}));
            continue;
        }
        let inline = object(segment.get("placeholder"));
        let id = identity(inline.get("id"));
        if id.is_empty() {
            continue;
        }
        if !placeholders.contains_key(&id) {
            placeholders.insert(id.clone(), placeholder(&inline, &id));
        }
        segments.push(json!({"placeholder_ref": id}));
    }
    for (raw, definition) in object(source.get("placeholders")) {
        let id = identity(Some(&Value::String(raw)));
        if id.is_empty() {
            continue;
        }
        placeholders.insert(id.clone(), placeholder(&object(Some(&definition)), &id));
    }
    json!({
        "key": js_string(source.get("key")),
        "aliases": array(source.get("aliases")).iter().map(|a| js_string(Some(a))).collect::<Vec<_>>(),
        "label": js_string(source.get("label")),
        "description": js_string(source.get("description")),
        "append": if source.get("append") == Some(&json!("space")) { "space" } else { "none" },
        "placeholders": placeholders,
        "segments": segments,
    })
}
/// `normalizeContainer`
fn container(value: &Value) -> Value {
    let source = object(Some(value));
    json!({
        "key": js_string(source.get("key")),
        "label": js_string(source.get("label")),
        "description": js_string(source.get("description")),
        "choices": array(source.get("choices")).iter().map(choice).collect::<Vec<_>>(),
        "groups": array(source.get("groups")).iter().map(container).collect::<Vec<_>>(),
    })
}
/// `normalizePromptCatalogEditorDocument`
pub fn normalize(value: &Value) -> Value {
    let source = object(Some(value));
    json!({
        "schema_version": "1.1",
        "mode": if source.get("mode") == Some(&json!("replace")) { "replace" } else { "merge" },
        "namespaces": array(source.get("namespaces")).iter().map(container).collect::<Vec<_>>(),
    })
}
/// `yamlString`: a JSON string literal, which YAML reads back unchanged.
fn quoted(value: &Value) -> String {
    serde_json::to_string(&js_string(Some(value))).expect("string")
}
fn text(value: &Value, key: &str) -> String {
    js_string(value.get(key))
}
fn serialize_choice(lines: &mut Vec<String>, value: &Value, depth: usize) {
    let c = choice(value);
    let prefix = "  ".repeat(depth);
    lines.push(format!("{prefix}- key: {}", text(&c, "key")));
    let aliases = c["aliases"]
        .as_array()
        .unwrap()
        .iter()
        .map(quoted)
        .collect::<Vec<_>>();
    lines.push(format!("{prefix}  aliases: [{}]", aliases.join(", ")));
    lines.push(format!("{prefix}  label: {}", quoted(&c["label"])));
    if !text(&c, "description").is_empty() {
        lines.push(format!(
            "{prefix}  description: {}",
            quoted(&c["description"])
        ));
    }
    lines.push(format!("{prefix}  append: {}", text(&c, "append")));
    let segments = c["segments"].as_array().unwrap();
    let mut referenced: Vec<String> = vec![];
    for id in segments
        .iter()
        .filter_map(|s| s["placeholder_ref"].as_str())
    {
        if !id.is_empty() && !referenced.iter().any(|r| r == id) {
            referenced.push(id.into());
        }
    }
    if !referenced.is_empty() {
        lines.push(format!("{prefix}  placeholders:"));
        for id in &referenced {
            let p = c["placeholders"]
                .get(id)
                .cloned()
                .unwrap_or_else(|| placeholder(&Map::new(), id));
            lines.push(format!("{prefix}    {id}:"));
            lines.push(format!("{prefix}      label: {}", quoted(&p["label"])));
            lines.push(format!(
                "{prefix}      initial_text: {}",
                quoted(&p["initial_text"])
            ));
            lines.push(format!("{prefix}      hint: {}", quoted(&p["hint"])));
        }
    }
    if segments.is_empty() {
        lines.push(format!("{prefix}  segments: []"));
        return;
    }
    lines.push(format!("{prefix}  segments:"));
    for segment in segments {
        match segment.get("text") {
            Some(t @ Value::String(_)) => lines.push(format!("{prefix}    - text: {}", quoted(t))),
            _ => lines.push(format!(
                "{prefix}    - placeholder_ref: {}",
                text(segment, "placeholder_ref")
            )),
        }
    }
}
fn serialize_container(lines: &mut Vec<String>, c: &Value, depth: usize) {
    let prefix = "  ".repeat(depth);
    lines.push(format!("{prefix}- key: {}", text(c, "key")));
    if !text(c, "label").is_empty() {
        lines.push(format!("{prefix}  label: {}", quoted(&c["label"])));
    }
    if !text(c, "description").is_empty() {
        lines.push(format!(
            "{prefix}  description: {}",
            quoted(&c["description"])
        ));
    }
    let choices = c["choices"].as_array().unwrap();
    if !choices.is_empty() {
        lines.push(format!("{prefix}  choices:"));
        for choice in choices {
            serialize_choice(lines, choice, depth + 2);
        }
    }
    let groups = c["groups"].as_array().unwrap();
    if !groups.is_empty() {
        lines.push(format!("{prefix}  groups:"));
        for group in groups {
            serialize_container(lines, group, depth + 2);
        }
    }
}
/// `serializePromptCatalogDocument`
pub fn serialize(value: &Value) -> String {
    let document = normalize(value);
    let mut lines = vec![
        format!("schema_version: {}", quoted(&json!("1.1"))),
        format!("mode: {}", text(&document, "mode")),
        "namespaces:".into(),
    ];
    for namespace in document["namespaces"].as_array().unwrap() {
        serialize_container(&mut lines, namespace, 1);
    }
    format!("{}\n", lines.join("\n"))
}
/// `linkedIdentityPlaceholderGroups` of a normalized choice: placeholder ids
/// whose initial text is the same `@identity`, by first reference.
fn linked(choice: &Value) -> Vec<(String, Vec<String>)> {
    let mut groups: Vec<(String, Vec<String>)> = vec![];
    let mut seen: Vec<String> = vec![];
    for segment in choice["segments"].as_array().into_iter().flatten() {
        let id = js_string(segment.get("placeholder_ref"));
        if id.is_empty() || seen.contains(&id) {
            continue;
        }
        seen.push(id.clone());
        let initial = js_string(
            choice["placeholders"]
                .get(&id)
                .and_then(|p| p.get("initial_text")),
        );
        let name = initial.strip_prefix('@').filter(|s| {
            s.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
                && s.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        });
        let identity = name.map_or_else(String::new, |n| identity(Some(&json!(n))));
        if identity.is_empty() {
            continue;
        }
        match groups.iter_mut().find(|(k, _)| *k == identity) {
            Some((_, ids)) => ids.push(id),
            None => groups.push((identity, vec![id])),
        }
    }
    groups.retain(|(_, ids)| ids.len() > 1);
    groups
}
fn visit(container: &mut Value, f: &mut dyn FnMut(&mut Value)) {
    for c in container["choices"].as_array_mut().into_iter().flatten() {
        f(c);
    }
    for g in container["groups"].as_array_mut().into_iter().flatten() {
        visit(g, f);
    }
}
/// `mergeLinkedIdentityPlaceholders` on a normalized document; returns how
/// many duplicate definitions were removed.
pub fn merge_linked_identities(document: &mut Value) -> usize {
    let mut count = 0;
    for namespace in document["namespaces"].as_array_mut().into_iter().flatten() {
        visit(namespace, &mut |choice| {
            for (identity, ids) in linked(choice) {
                let canonical = ids
                    .iter()
                    .find(|id| **id == identity)
                    .unwrap_or(&ids[0])
                    .clone();
                let duplicates: Vec<&String> = ids.iter().filter(|id| **id != canonical).collect();
                for segment in choice["segments"].as_array_mut().into_iter().flatten() {
                    if segment["placeholder_ref"]
                        .as_str()
                        .is_some_and(|r| duplicates.iter().any(|d| *d == r))
                    {
                        segment["placeholder_ref"] = json!(canonical);
                    }
                }
                if let Some(map) = choice["placeholders"].as_object_mut() {
                    for id in &duplicates {
                        map.remove(*id);
                    }
                }
                count += duplicates.len();
            }
        });
    }
    count
}
/// A Form save: the normalized document with linked identities merged.
pub fn form_save(value: &Value) -> String {
    let mut document = normalize(value);
    merge_linked_identities(&mut document);
    serialize(&document)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_yaml_quotes_text_and_links_identities_on_form_save() {
        let document = json!({"schema_version":"1.0","mode":"replace","namespaces":[{"key":"own","label":"on","description":"12:30",
            "choices":[{"key":"meet","aliases":["greet"],"label":"Meeting","append":"space","segments":[
                {"text":"yes "},{"placeholder":{"id":"Who","label":"Person","initial_text":"@Keeper","hint":"H"}},
                {"text":" meets "},{"placeholder_ref":"other"}],
                "placeholders":{"other":{"label":"Other","initial_text":"@keeper","hint":"H"}}}]}]});
        let text = serialize(&document);
        assert_eq!(
            text,
            "schema_version: \"1.1\"\nmode: replace\nnamespaces:\n  - key: own\n    label: \"on\"\n    description: \"12:30\"\n    choices:\n      - key: meet\n        aliases: [\"greet\"]\n        label: \"Meeting\"\n        append: space\n        placeholders:\n          who:\n            label: \"Person\"\n            initial_text: \"@Keeper\"\n            hint: \"H\"\n          other:\n            label: \"Other\"\n            initial_text: \"@keeper\"\n            hint: \"H\"\n        segments:\n          - text: \"yes \"\n          - placeholder_ref: who\n          - text: \" meets \"\n          - placeholder_ref: other\n"
        );
        let saved = form_save(&document);
        assert!(saved.contains("          - placeholder_ref: who\n          - text: \" meets \"\n          - placeholder_ref: who\n"));
        assert!(!saved.contains("other:"));
    }
}
