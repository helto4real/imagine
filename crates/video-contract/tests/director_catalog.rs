//! Imagine's prompt catalogs against answers recorded from Director itself
//! (comfyui-helto-director): its validator's verdict and error details, the
//! merged catalog, frontend completions and expansions, and the editor's
//! canonical YAML. Every case is synthetic.
use imagine_video_contract::prompt_catalog::{self as pc, Container, Segment};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const CASES: &str = include_str!("fixtures/director-catalog-cases.json");

fn normalized(nodes: &[Container]) -> Value {
    Value::Array(
        nodes
            .iter()
            .map(|n| {
                json!({"key":n.key,"label":n.label,"description":n.description,"groups":normalized(&n.groups),
                "choices":n.choices.iter().map(|c| json!({"key":c.key,"label":c.label,"description":c.description,
                    "aliases":c.aliases,"append":c.append,"segments":c.segments.iter().map(|s| match s {
                        Segment::Text { text } => json!({"text": text}),
                        Segment::Placeholder { placeholder } => json!({"placeholder": placeholder}),
                    }).collect::<Vec<_>>()})).collect::<Vec<_>>()})
            })
            .collect(),
    )
}
/// Director's `public_details` of an issue.
fn details(issue: &pc::Issue) -> Value {
    let mut d = json!({"code": issue.code});
    if let Some(source) = issue.safe_source() {
        d["source"] = json!(source);
    }
    if let Some(path) = issue.safe_path() {
        d["path"] = json!(path);
    }
    if let Some(line) = issue.line {
        d["line"] = json!(line);
    }
    if let Some(column) = issue.column {
        d["column"] = json!(column);
    }
    d
}
#[test]
fn every_recorded_director_answer_matches() {
    let fixture: Value = serde_json::from_str(CASES).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() > 100);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let sources: BTreeMap<String, String> =
            serde_json::from_value(case["sources"].clone()).unwrap();
        let expected = &case["expected"];
        match pc::load(&sources) {
            Err(issue) => {
                assert_eq!(
                    expected["ok"], false,
                    "{name}: Director accepts; Imagine: {issue}"
                );
                assert_eq!(details(&issue), expected["details"], "{name}");
                assert_eq!(issue.to_string(), expected["summary"], "{name}");
            }
            Ok(catalog) => {
                assert_eq!(
                    expected["ok"], true,
                    "{name}: Director rejects with {}",
                    expected["summary"]
                );
                assert_eq!(
                    normalized(&catalog.namespaces),
                    expected["namespaces"],
                    "{name}"
                );
                let queries = case["queries"].as_array().unwrap();
                for (text, wanted) in queries.iter().zip(expected["queries"].as_array().unwrap()) {
                    let text = text.as_str().unwrap();
                    let caret = text.chars().count();
                    let items = pc::context(text, caret, caret)
                        .map(|c| catalog.completions(&c))
                        .unwrap_or_default()
                        .iter()
                        .map(|e| {
                            json!({"path":e.path,"label":e.label,"detail":e.detail,
                            "expansion":e.choice.as_ref().map(|c| c.expand(&BTreeMap::new()))})
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(
                        json!({"text": text, "items": items}),
                        *wanted,
                        "{name}: {text}"
                    );
                }
                let serialized = &expected["serialized"];
                let source = serialized_source(&catalog);
                assert_eq!(
                    pc::source_text(&source).unwrap(),
                    serialized["text"],
                    "{name}"
                );
                assert_eq!(pc::editor::form_save(&source), serialized["form"], "{name}");
            }
        }
    }
}
/// The merged catalog as Director's validator returns it, which its editor serialized.
fn serialized_source(catalog: &pc::Catalog) -> Value {
    fn nodes(ns: &[Container]) -> Value {
        Value::Array(
            ns.iter()
                .map(|n| {
                    let mut v = json!({"key": n.key, "label": n.label});
                    if !n.description.is_empty() {
                        v["description"] = json!(n.description);
                    }
                    if !n.choices.is_empty() {
                        v["choices"] = Value::Array(
                            n.choices
                                .iter()
                                .map(|c| {
                                    let mut x = json!({"key":c.key,"aliases":c.aliases,"label":c.label,"append":c.append,
                                    "segments":c.segments.iter().map(|s| match s {
                                        Segment::Text { text } => json!({"text": text}),
                                        Segment::Placeholder { placeholder } => json!({"placeholder": placeholder}),
                                    }).collect::<Vec<_>>()});
                                    if !c.description.is_empty() {
                                        x["description"] = json!(c.description);
                                    }
                                    x
                                })
                                .collect(),
                        );
                    }
                    if !n.groups.is_empty() {
                        v["groups"] = nodes(&n.groups);
                    }
                    v
                })
                .collect(),
        )
    }
    json!({"schema_version": catalog.schema_version, "namespaces": nodes(&catalog.namespaces)})
}
