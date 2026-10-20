use imagine_inference::{
    native::Error,
    patching::{
        options::{GeneralMapping, Options},
        relationships::{ModelRelationships, RelationshipCloneBindings},
    },
};
use serde_json::{Value, json};
use std::{cell::Cell, cell::RefCell, rc::Rc};

struct Token(String);
fn token_label(value: &Options) -> &str {
    let Options::Object(value) = value else {
        panic!("own foreign object")
    };
    &value.downcast_ref::<Token>().unwrap().0
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
struct Bindings {
    mapping_keys: Option<Rc<super::own_mapping_keys::OwnMappingKeys>>,
    source: ModelRelationships,
    target: ModelRelationships,
    pool: RefCell<Vec<(String, Options)>>,
    events: Vec<Value>,
    position: Cell<usize>,
}
impl Bindings {
    fn token(&self, label: &str) -> Options {
        let mut pool = self.pool.borrow_mut();
        if let Some((_, value)) = pool.iter().find(|(k, _)| k == label) {
            return value.clone();
        }
        let value = Options::Object(Rc::new(Token(label.into())));
        pool.push((label.into(), value.clone()));
        value
    }
    fn decode(&self, wire: &Value) -> Options {
        match wire[0].as_str().unwrap() {
            "object" => self.token(wire[1].as_str().unwrap()),
            "dict" if self.mapping_keys.is_some() => {
                let out = self.mapping_keys.as_ref().unwrap().empty();
                for pair in wire[1].as_array().unwrap() {
                    out.set(pair[0].as_str().unwrap(), self.decode(&pair[1]))
                        .unwrap();
                }
                out
            }
            "dict" => Options::dictionary(
                wire[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| (v[0].as_str().unwrap().into(), self.decode(&v[1])))
                    .collect(),
            ),
            kind => {
                let values = wire[1]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| self.decode(v))
                    .collect();
                if kind == "tuple" {
                    Options::Tuple(Rc::new(values))
                } else {
                    assert_eq!(kind, "list");
                    list(values)
                }
            }
        }
    }
    fn apply(&self, action: &Value) {
        let owner = if action[1] == "source" {
            &self.source
        } else {
            &self.target
        };
        let attachments = action[2] == "attachments";
        if action[0] == "root" {
            let value = self.decode(&action[3]);
            if attachments {
                owner.replace_attachments(value);
            } else {
                owner.replace_additional_models(value);
            }
            return;
        }
        let root = if attachments {
            owner.attachments().unwrap()
        } else {
            owner.additional_models().unwrap()
        };
        let key = action[3].as_str().unwrap();
        match action[0].as_str().unwrap() {
            "set" => root.set(key, self.decode(&action[4])).unwrap(),
            "remove" => drop(root.pop(key).unwrap()),
            op => {
                if let Some(Options::List(group)) = root.get(key).unwrap() {
                    if op == "append" {
                        group.borrow_mut().push(self.decode(&action[4]));
                    } else {
                        assert_eq!(op, "clear");
                        group.borrow_mut().clear();
                    }
                }
            }
        }
    }
    fn emit(&self, op: &str, label: &str) -> Result<Options, Error> {
        let pos = self.position.get();
        let event = self.events.get(pos).expect("extra foreign operation");
        self.position.set(pos + 1);
        assert_eq!(event["op"], op, "foreign operation {pos}");
        assert_eq!(event["label"], label, "foreign object {pos}");
        assert_eq!(snapshot(self), event["state"], "foreign entry {pos}");
        if let Some(action) = event.get("action") {
            self.apply(action);
        }
        if let Some(error) = event.get("error") {
            let class = match error[0].as_str().unwrap() {
                "AttributeError" => "AttributeError",
                "ValueError" => "ValueError",
                _ => unreachable!(),
            };
            return Err(Error::ReferenceException {
                class,
                message: error[1].as_str().unwrap().into(),
            });
        }
        Ok(self.token(event["result"].as_str().unwrap()))
    }
}
impl RelationshipCloneBindings for Bindings {
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self
            .mapping_keys
            .as_ref()
            .map_or_else(|| Options::dictionary(vec![]), |keys| keys.empty()))
    }
    fn clone_attribute(&self, attachment: &Options) -> Result<Options, Error> {
        self.emit("attribute", token_label(attachment))
    }
    fn call_clone_attribute(&self, method: &Options) -> Result<Options, Error> {
        self.emit(
            "attachment-call",
            token_label(method).strip_prefix("method:").unwrap(),
        )
    }
    fn clone_model(&self, model: &Options) -> Result<Options, Error> {
        self.emit("model-clone", token_label(model))
    }
}
fn encode(value: &Options) -> Value {
    if let Some(mapping) = GeneralMapping::get(value) {
        return json!([
            "dict",
            mapping
                .text_items()
                .unwrap()
                .iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>()
        ]);
    }
    match value {
        Options::Object(_) => json!(["object", token_label(value)]),
        Options::Dict(dict) => json!([
            "dict",
            dict.borrow()
                .iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>()
        ]),
        Options::List(values) => {
            json!([
                "list",
                values.borrow().iter().map(encode).collect::<Vec<_>>()
            ])
        }
        Options::Tuple(values) => json!(["tuple", values.iter().map(encode).collect::<Vec<_>>()]),
        _ => unreachable!(),
    }
}
fn walk(value: &Options, path: String, seen: &mut Vec<(Options, String)>, links: &mut Vec<Value>) {
    let previous = seen.iter().find(|(v, _)| value.same_object(v));
    links.push(json!([path, previous.map(|(_, path)| path)]));
    if previous.is_some() {
        return;
    }
    seen.push((value.clone(), path.clone()));
    let children = match value {
        _ if GeneralMapping::get(value).is_some() => {
            GeneralMapping::get(value).unwrap().text_items().unwrap()
        }
        Options::Dict(dict) => dict.borrow().to_vec(),
        Options::List(values) => values
            .borrow()
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect(),
        Options::Tuple(values) => values
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect(),
        _ => vec![],
    };
    for (key, child) in children {
        walk(&child, format!("{path}/{key}"), seen, links);
    }
}
fn snapshot(bindings: &Bindings) -> Value {
    let roots = [
        bindings.source.attachments().unwrap(),
        bindings.source.additional_models().unwrap(),
        bindings.target.attachments().unwrap(),
        bindings.target.additional_models().unwrap(),
    ];
    let mut seen = vec![];
    let mut links = vec![];
    for (i, value) in roots.iter().enumerate() {
        walk(value, i.to_string(), &mut seen, &mut links);
    }
    json!({"roots":roots.iter().map(encode).collect::<Vec<_>>(),"links":links})
}
#[test]
fn original_clone_relationship_blocks_preserve_descriptor_reaccess_live_maps_lists_root_changes_aliases_and_partial_failures()
 {
    run(false);
    run(true);
}
fn run(general: bool) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-relationship-clone.json"
    ))
    .unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 140);
    for (index, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            mapping_keys: general.then(super::own_mapping_keys::OwnMappingKeys::seeded),
            source: ModelRelationships::new(),
            target: ModelRelationships::new(),
            pool: RefCell::new(vec![]),
            events: case["events"].as_array().unwrap().clone(),
            position: Cell::new(0),
        };
        if let Some(keys) = &bindings.mapping_keys {
            for owner in [&bindings.source, &bindings.target] {
                owner.replace_attachments(keys.empty());
                owner.replace_additional_models(keys.empty());
            }
        }
        for (key, label) in [("a", "a"), ("b", "plain"), ("again", "a")] {
            bindings
                .source
                .set_attachments(key, bindings.token(label))
                .unwrap();
        }
        let group = vec![
            bindings.token("m0"),
            bindings.token("m0"),
            bindings.token("m1"),
        ];
        let group = if case["layout"] == "tuple" {
            Options::Tuple(Rc::new(group))
        } else {
            list(group)
        };
        for key in ["g", "shared"] {
            bindings
                .source
                .set_additional_models(key, group.clone())
                .unwrap();
        }
        bindings
            .target
            .set_attachments("old", bindings.token("old"))
            .unwrap();
        bindings
            .target
            .set_additional_models("old", list(vec![bindings.token("old")]))
            .unwrap();
        assert_eq!(snapshot(&bindings), case["before"], "initial {index}");
        let result = bindings
            .source
            .clone_members_into(&bindings.target, &bindings);
        let outcome = match result {
            Ok(()) => json!({"ok":true}),
            Err(Error::ReferenceException { class, message }) => {
                json!({"error":class,"message":message})
            }
            Err(error) => panic!("unbound case {index}: {error:?}"),
        };
        let expected = if case.get("ok").is_some() {
            json!({"ok":true})
        } else {
            json!({"error":case["error"],"message":case["message"]})
        };
        assert_eq!(outcome, expected, "outcome {index}");
        assert_eq!(snapshot(&bindings), case["after"], "final {index}");
        assert_eq!(
            bindings.position.get(),
            bindings.events.len(),
            "events {index}"
        );
    }
}

struct ReleaseObject {
    label: &'static str,
    probe: bool,
    events: Rc<RefCell<Vec<String>>>,
}
impl Drop for ReleaseObject {
    fn drop(&mut self) {
        self.events.borrow_mut().push(format!(
            "drop-{}:{}",
            if self.probe { "probe" } else { "attachment" },
            self.label
        ));
    }
}
struct ReleaseBindings {
    source: ModelRelationships,
    events: Rc<RefCell<Vec<String>>>,
}
impl ReleaseBindings {
    fn object(&self, label: &'static str, probe: bool) -> Options {
        Options::Object(Rc::new(ReleaseObject {
            label,
            probe,
            events: self.events.clone(),
        }))
    }
}
fn release_object(value: &Options) -> &ReleaseObject {
    let Options::Object(value) = value else {
        unreachable!()
    };
    value.downcast_ref::<ReleaseObject>().unwrap()
}
impl RelationshipCloneBindings for ReleaseBindings {
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(Options::dictionary(vec![]))
    }
    fn clone_attribute(&self, attachment: &Options) -> Result<Options, Error> {
        let object = release_object(attachment);
        self.events
            .borrow_mut()
            .push(format!("get:{}", object.label));
        if object.label == "a" {
            self.source.set_attachments("a", self.object("b", false))?;
        }
        Ok(self.object(object.label, true))
    }
    fn call_clone_attribute(&self, method: &Options) -> Result<Options, Error> {
        self.events
            .borrow_mut()
            .push(format!("call:{}", release_object(method).label));
        Ok(Options::Object(Rc::new(())))
    }
    fn clone_model(&self, _: &Options) -> Result<Options, Error> {
        unreachable!()
    }
}
#[test]
fn original_hasattr_releases_temporary_descriptor_before_last_receiver_after_reentrant_root_replacement()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-relationship-clone.json"
    ))
    .unwrap();
    let bindings = ReleaseBindings {
        source: ModelRelationships::new(),
        events: Rc::new(RefCell::new(vec![])),
    };
    bindings
        .source
        .set_attachments("a", bindings.object("a", false))
        .unwrap();
    let target = ModelRelationships::new();
    bindings
        .source
        .clone_members_into(&target, &bindings)
        .unwrap();
    assert_eq!(json!(*bindings.events.borrow()), fixture["release_events"]);
}
