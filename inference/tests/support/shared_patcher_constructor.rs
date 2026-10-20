use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        attributes::AttributeStore,
        model_patcher::{Initialization, InitializationBindings, PatcherState},
        options::{GeneralMapping, Options},
    },
};
use serde_json::{Value, json};
use std::{cell::Cell, cell::RefCell, rc::Rc};

enum Foreign {
    Model(usize),
    Token(String),
    EmptySet,
}
fn foreign(value: &Options) -> &Foreign {
    let Options::Object(value) = value else {
        panic!("own foreign boundary")
    };
    value.downcast_ref::<Foreign>().unwrap()
}
fn model_index(value: &Options) -> usize {
    let Foreign::Model(index) = foreign(value) else {
        panic!("own model")
    };
    *index
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
        Options::Object(_) => match foreign(value) {
            Foreign::Model(index) => json!(["model", format!("m{index}")]),
            Foreign::Token(label) => json!(["object", label]),
            Foreign::EmptySet => json!(["set", []]),
        },
        Options::Dict(dict) => json!([
            "dict",
            dict.borrow()
                .iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>()
        ]),
        Options::List(list) => {
            json!(["list", list.borrow().iter().map(encode).collect::<Vec<_>>()])
        }
        Options::Tuple(list) => json!(["tuple", list.iter().map(encode).collect::<Vec<_>>()]),
        Options::Atom(value) => json!([
            "atom",
            match value {
                V::None => Value::Null,
                V::Bool(value) => json!(value),
                V::Int(value) => json!(value),
                V::Text(value) => json!(value),
                _ => unreachable!(),
            }
        ]),
    }
}
fn walk(value: &Options, name: String, seen: &mut Vec<(Options, String)>, links: &mut Vec<Value>) {
    if matches!(value, Options::Atom(_)) {
        return;
    }
    let previous = seen.iter().find(|(v, _)| v.same_object(value));
    links.push(json!([name, previous.map(|(_, name)| name)]));
    if previous.is_some() {
        return;
    }
    seen.push((value.clone(), name.clone()));
    let children = match value {
        _ if GeneralMapping::get(value).is_some() => {
            GeneralMapping::get(value).unwrap().text_items().unwrap()
        }
        Options::Dict(dict) => dict.borrow().to_vec(),
        Options::List(list) => list
            .borrow()
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect(),
        Options::Tuple(list) => list
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect(),
        _ => vec![],
    };
    for (key, value) in children {
        walk(&value, format!("{name}/{key}"), seen, links);
    }
}
struct Bindings {
    view: AttributeStore,
    mapping_keys: Option<Rc<super::own_mapping_keys::OwnMappingKeys>>,
    state: PatcherState,
    models: Vec<Options>,
    model_fields: Vec<Options>,
    tokens: RefCell<Vec<(String, Options)>>,
    events: Vec<Value>,
    position: Cell<usize>,
}
impl Bindings {
    fn token(&self, label: &str) -> Options {
        let mut tokens = self.tokens.borrow_mut();
        if let Some((_, value)) = tokens.iter().find(|(key, _)| key == label) {
            return value.clone();
        }
        let value = Options::Object(Rc::new(Foreign::Token(label.into())));
        tokens.push((label.into(), value.clone()));
        value
    }
    fn decode(&self, value: &Value) -> Options {
        match value[0].as_str().unwrap() {
            "atom" => Options::from_value(match &value[1] {
                Value::Null => V::None,
                Value::Bool(v) => V::Bool(*v),
                Value::String(v) => V::Text(v.clone()),
                Value::Number(v) => V::Int(v.as_i64().unwrap()),
                _ => unreachable!(),
            }),
            "object" => self.token(value[1].as_str().unwrap()),
            _ => unreachable!(),
        }
    }
    fn snapshot(&self) -> Value {
        let fields = self.state.fields.entries();
        let view = self.view.entries();
        assert_eq!(
            fields
                .iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>(),
            view.iter()
                .map(|(k, v)| json!([k, encode(v)]))
                .collect::<Vec<_>>(),
            "live native field-store view"
        );
        for ((_, a), (_, b)) in fields.iter().zip(&view) {
            if !matches!(a, Options::Atom(_)) {
                assert!(a.same_object(b), "field views must not copy source owners");
            }
        }
        let mut roots = vec![Options::dictionary(fields.clone())];
        roots.extend(self.model_fields.clone());
        let mut seen = vec![];
        let mut links = vec![];
        for (i, root) in roots.iter().enumerate() {
            walk(root, i.to_string(), &mut seen, &mut links);
        }
        json!({
            "owner":fields.iter().map(|(k,v)|json!([k,encode(v)])).collect::<Vec<_>>(),
            "models":self.model_fields.iter().map(|root|{
                let Options::Dict(root)=root else {unreachable!()};
                root.borrow().iter().map(|(k,v)|json!([k,encode(v)])).collect::<Vec<_>>()
            }).collect::<Vec<_>>(),
            "links":links
        })
    }
    fn action(&self, action: &Value) {
        match action[0].as_str().unwrap() {
            "replace-model" => self.state.fields.set("model", self.models[1].clone()),
            "delete-model" => {
                let _ = self.state.fields.delete("model");
            }
            "replace-attachments" => self.state.fields.set(
                "attachments",
                Options::dictionary(vec![("own".into(), self.token("attachment"))]),
            ),
            "delete-attachments" => {
                let _ = self.state.fields.delete("attachments");
            }
            "set-existing-memory" => self.model_fields[0]
                .set("model_loaded_weight_memory", self.token("existing-memory"))
                .unwrap(),
            "delete-device" => drop(self.model_fields[0].pop("device").unwrap()),
            "replace-device" => self.model_fields[0]
                .set("device", Options::Atom(V::None))
                .unwrap(),
            _ => unreachable!(),
        }
    }
    fn event(&self, op: &str, inputs: Vec<(&str, Value)>) -> Result<Value, Error> {
        let index = self.position.get();
        let event = self.events.get(index).expect("extra constructor operation");
        self.position.set(index + 1);
        assert_eq!(event["op"], op, "operation {index}");
        for (key, value) in inputs {
            assert_eq!(event[key], value, "input {key} at {index}");
        }
        assert_eq!(self.snapshot(), event["state"], "entry state {index}");
        if let Some(action) = event.get("action") {
            self.action(action);
        }
        if let Some(error) = event.get("error")
            && error[1] == "own constructor boundary failure"
        {
            let class = match error[0].as_str().unwrap() {
                "ValueError" => "ValueError",
                "AttributeError" => "AttributeError",
                _ => unreachable!(),
            };
            return Err(Error::ReferenceException {
                class,
                message: error[1].as_str().unwrap().into(),
            });
        }
        Ok(event.clone())
    }
}
impl InitializationBindings for Bindings {
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self
            .mapping_keys
            .as_ref()
            .map_or_else(|| Options::dictionary(vec![]), |keys| keys.empty()))
    }
    fn model_attribute(&self, model: &Options, name: &str) -> Result<Options, Error> {
        let index = model_index(model);
        let event = self.event(
            "get",
            vec![("model", json!(format!("m{index}"))), ("name", json!(name))],
        )?;
        let value = self.model_fields[index].get(name)?;
        if let Some(value) = value {
            assert_eq!(encode(&value), event["result"]);
            Ok(value)
        } else {
            assert_eq!(event["error"][1], "own missing model attribute");
            Err(Error::ReferenceException {
                class: "AttributeError",
                message: "own missing model attribute".into(),
            })
        }
    }
    fn set_model_attribute(
        &self,
        model: &Options,
        name: &str,
        value: Options,
    ) -> Result<(), Error> {
        let index = model_index(model);
        self.event(
            "set",
            vec![
                ("model", json!(format!("m{index}"))),
                ("name", json!(name)),
                ("value", encode(&value)),
            ],
        )?;
        self.model_fields[index].set(name, value)
    }
    fn debug(&self, message: &str) -> Result<(), Error> {
        self.event("debug", vec![("message", json!(message))])?;
        Ok(())
    }
    fn uuid4(&self) -> Result<Options, Error> {
        let event = self.event("uuid", vec![])?;
        Ok(self.decode(&event["result"]))
    }
    fn empty_pinned_set(&self) -> Result<Options, Error> {
        self.event("pinned-set", vec![])?;
        Ok(Options::Object(Rc::new(Foreign::EmptySet)))
    }
    fn max_speed_hook_mode(&self) -> Result<Options, Error> {
        self.event("hook-mode", vec![])?;
        Ok(self.token("maxspeed"))
    }
}
fn outcome(result: Result<Options, Error>) -> Value {
    match result {
        Ok(value) => json!({"output":encode(&value)}),
        Err(Error::ReferenceException { class, message }) => {
            json!({"error":class,"message":message})
        }
        Err(error) => panic!("unexpected native constructor gap: {error:?}"),
    }
}
#[test]
fn original_whole_patcher_constructor_preserves_partial_fields_attribute_reaccess_factories_model_changes_and_shared_relationship_slots()
 {
    run(false);
    run(true);
}
fn run(general: bool) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-patcher-constructor.json"
    ))
    .unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["fixture_tensors"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 255);
    for (case_index, case) in cases.iter().enumerate() {
        let state = PatcherState::uninitialized("ModelPatcher");
        let view = state.fields.clone();
        let bindings = Bindings {
            view,
            mapping_keys: general.then(super::own_mapping_keys::OwnMappingKeys::seeded),
            state,
            models: (0..2)
                .map(|index| Options::Object(Rc::new(Foreign::Model(index))))
                .collect(),
            model_fields: (0..2).map(|_| Options::dictionary(vec![])).collect(),
            tokens: RefCell::new(vec![]),
            events: case["events"].as_array().unwrap().clone(),
            position: Cell::new(0),
        };
        for (index, fields) in case["before"]["models"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            for pair in fields.as_array().unwrap() {
                bindings.model_fields[index]
                    .set(pair[0].as_str().unwrap(), bindings.decode(&pair[1]))
                    .unwrap();
            }
        }
        assert_eq!(bindings.snapshot(), case["before"], "initial {case_index}");
        let arguments = &case["arguments"];
        let args = Initialization {
            model: bindings.models[0].clone(),
            load_device: bindings.decode(&arguments["load_device"]),
            offload_device: bindings.decode(&arguments["offload_device"]),
            size: bindings.decode(&arguments["size"]),
            weight_inplace_update: bindings.decode(&arguments["weight_inplace_update"]),
            fast_disk: bindings.decode(&arguments["fast_disk"]),
        };
        let result = bindings.state.initialize(&args, &bindings);
        let result = outcome(result.map(|_| Options::Atom(V::None)));
        let expected = if case.get("ok").is_some() {
            json!({"output":["atom",null]})
        } else {
            json!({"error":case["error"],"message":case["message"]})
        };
        assert_eq!(result, expected, "constructor {case_index}");
        assert_eq!(bindings.snapshot(), case["after"], "final {case_index}");
        assert_eq!(bindings.position.get(), bindings.events.len());
        for step in case["post"].as_array().unwrap() {
            let state = &bindings.state;
            let result = (|| {
                match step["action"].as_str().unwrap() {
                    "get-attachment" => return state.relationships.get_attachment("own"),
                    "set-attachment" => state
                        .relationships
                        .set_attachments("own", bindings.token("attachment"))?,
                    "alias-additional" => bindings
                        .view
                        .set("additional_models", state.fields.get("attachments")?),
                    "get-additional" => {
                        return state.relationships.get_additional_models_with_key("own");
                    }
                    "delete-attachments" => bindings.view.delete("attachments")?,
                    "replace-attachments" => {
                        state
                            .relationships
                            .replace_attachments(Options::dictionary(vec![(
                                "again".into(),
                                bindings.token("again"),
                            )]))
                    }
                    "delete-additional" => state.fields.delete("additional_models")?,
                    "replace-additional" => state
                        .relationships
                        .replace_additional_models(Options::dictionary(vec![])),
                    _ => unreachable!(),
                }
                Ok(Options::Atom(V::None))
            })();
            let result = outcome(result);
            let expected = if step.get("output").is_some() {
                json!({"output":step["output"]})
            } else {
                json!({"error":step["error"],"message":step["message"]})
            };
            assert_eq!(result, expected, "post {case_index}: {}", step["action"]);
            assert_eq!(bindings.snapshot(), step["state"]);
        }
    }
}
