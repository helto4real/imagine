use super::binding_tests::reference_error;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        options::Options as O,
        relationships::{
            AdditionalModelBindings, AdditionalModelLocals, ModelRelationships,
            get_nested_additional_models,
        },
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../fixtures/shared-model-relationships.json"
    ))
    .unwrap()
}
struct Node {
    index: usize,
    state: ModelRelationships,
}
fn node(v: &O) -> &Node {
    let O::Object(v) = v else {
        panic!("own model owner")
    };
    v.downcast_ref().unwrap()
}
fn list(v: Vec<O>) -> O {
    O::List(Rc::new(RefCell::new(v)))
}
fn decode(v: &Value, models: &[O]) -> O {
    if let Some(ty) = v.get("type").and_then(Value::as_str) {
        let values = v["items"].as_array().unwrap();
        return match ty {
            "dict" => O::dictionary(
                values
                    .iter()
                    .map(|v| (v[0].as_str().unwrap().into(), decode(&v[1], models)))
                    .collect(),
            ),
            "list" => list(values.iter().map(|v| decode(v, models)).collect()),
            "tuple" => O::Tuple(Rc::new(values.iter().map(|v| decode(v, models)).collect())),
            _ => panic!("own type"),
        };
    }
    if !models.is_empty()
        && let Some(label) = v.as_str()
        && let Some(index) = label
            .strip_prefix('m')
            .and_then(|v| v.parse::<usize>().ok())
    {
        return models[index].clone();
    }
    O::Atom(match v {
        Value::Null => V::None,
        Value::String(v) => V::Text(v.clone()),
        Value::Bool(v) => V::Bool(*v),
        Value::Number(v) => V::Int(v.as_i64().unwrap()),
        _ => panic!("own scalar"),
    })
}
fn encode(v: &O) -> Value {
    match v {
        O::Object(_) => json!(format!("m{}", node(v).index)),
        O::Dict(v) => {
            json!({"type":"dict","items":v.borrow().iter().map(|(k,v)|json!([k,encode(v)])).collect::<Vec<_>>()})
        }
        O::List(v) => {
            json!({"type":"list","items":v.borrow().iter().map(encode).collect::<Vec<_>>()})
        }
        O::Tuple(v) => json!({"type":"tuple","items":v.iter().map(encode).collect::<Vec<_>>()}),
        O::Atom(V::None) => Value::Null,
        O::Atom(V::Text(v)) => json!(v),
        O::Atom(V::Int(v)) => json!(v),
        O::Atom(V::Bool(v)) => json!(v),
        _ => panic!("own scalar"),
    }
}
struct Bindings {
    models: Vec<O>,
    hashes: Vec<i64>,
    mode: String,
    calls: Cell<usize>,
    events: RefCell<Vec<Value>>,
    current: RefCell<Value>,
    previous: RefCell<Option<O>>,
    depth: Cell<usize>,
    finished: RefCell<Value>,
}
impl AdditionalModelBindings for Bindings {
    fn additional_models(&self, model: &O) -> Result<O, Error> {
        self.calls.set(self.calls.get() + 1);
        let call = self.calls.get();
        self.events
            .borrow_mut()
            .push(json!({"tag":"models","model":encode(model),"locals":*self.current.borrow()}));
        if self.mode == "error-root" && call == 1 || self.mode == "error-child" && call == 2 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own model failure".into(),
            });
        }
        if call == 2 {
            match self.mode.as_str() {
                "append-previous" => {
                    let previous = self.previous.borrow().clone().unwrap();
                    let O::List(v) = previous else { unreachable!() };
                    v.borrow_mut().push(self.models[4].clone());
                }
                "clear-previous" => {
                    let previous = self.previous.borrow().clone().unwrap();
                    let O::List(v) = previous else { unreachable!() };
                    v.borrow_mut().clear();
                }
                "replace-root" => node(&self.models[0])
                    .state
                    .replace_additional_models(O::dictionary(vec![])),
                _ => {}
            }
        }
        let out = node(model).state.get_additional_models()?;
        if call == 2 && self.mode == "append-returned" {
            let O::List(v) = &out else { unreachable!() };
            v.borrow_mut().push(self.models[4].clone());
        }
        Ok(out)
    }
    fn model_hash(&self, model: &O) -> Result<i64, Error> {
        self.events
            .borrow_mut()
            .push(json!({"tag":"hash","model":encode(model)}));
        Ok(self.hashes[node(model).index])
    }
    fn same_model(&self, a: &O, b: &O) -> bool {
        a.same_object(b)
    }
    fn enter_model_recursion(&self) -> Result<(), Error> {
        self.depth.set(self.depth.get() + 1);
        Ok(())
    }
    fn leave_model_recursion(&self) {
        self.depth.set(self.depth.get() - 1);
    }
    fn source_locals(&self, l: &AdditionalModelLocals, phase: &str) -> Result<(), Error> {
        if phase == "root" || phase == "model" {
            *self.current.borrow_mut() = json!({"all":l.all_models.as_ref().map(encode),"cache":if l.all_models.is_none(){Value::Null}else{json!(l.cached_models().map(encode).collect::<Vec<_>>())},"frames":l.frames.iter().map(|f|json!({"previous":encode(&f.previous),"next":encode(&f.next),"model":f.model.as_ref().map(encode),"candidates":f.candidates.as_ref().map(encode),"candidate":f.candidate.as_ref().map(encode)})).collect::<Vec<_>>()});
            *self.previous.borrow_mut() = l.frames.last().map(|f| f.previous.clone());
        }
        if phase == "return" {
            *self.finished.borrow_mut() = json!({"all":l.all_models.as_ref().map(encode),"cache":l.cached_models().map(encode).collect::<Vec<_>>(),"same_all":l.result.as_ref().unwrap().same_object(l.all_models.as_ref().unwrap())});
            assert!(l.frames.is_empty());
        }
        Ok(())
    }
}
#[test]
fn original_nested_model_dependencies_preserve_cycles_first_level_duplicates_hash_calls_live_lists_frames_and_failures()
 {
    let f = fixture();
    assert_eq!(f["trained_model_weights"], 0);
    assert_eq!(f["fixture_tensors"], 0);
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 258);
    for (id, case) in cases.iter().enumerate() {
        let models = (0..case["hashes"].as_array().unwrap().len())
            .map(|i| {
                O::Object(Rc::new(Node {
                    index: i,
                    state: ModelRelationships::new(),
                }))
            })
            .collect::<Vec<_>>();
        for (i, model) in models.iter().enumerate() {
            node(model)
                .state
                .replace_additional_models(decode(&case["before"][i], &models));
        }
        let mode = case["params"]["mode"].as_str().unwrap();
        if mode == "duplicate-groups" {
            let group = node(&models[0])
                .state
                .get_additional_models_with_key("0")
                .unwrap();
            node(&models[0])
                .state
                .set_additional_models("again", group)
                .unwrap();
        }
        let bindings = Bindings {
            models: models.clone(),
            hashes: case["hashes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect(),
            mode: mode.into(),
            calls: Cell::new(0),
            events: RefCell::new(vec![]),
            current: RefCell::new(Value::Null),
            previous: RefCell::new(None),
            depth: Cell::new(0),
            finished: RefCell::new(json!({})),
        };
        match get_nested_additional_models(&models[0], &bindings) {
            Ok(result) => {
                assert!(case.get("error").is_none(), "{id} expected {case}");
                assert_eq!(
                    encode(&result),
                    case["output"],
                    "{id} {} original model order",
                    case["params"]
                );
            }
            Err(e) => {
                assert!(case.get("error").is_some(), "{id} unexpected {e}");
                reference_error(e, case);
            }
        }
        assert_eq!(bindings.depth.get(), 0, "{id} recursion bookkeeping");
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "{id} {} exact calls and source frame state",
            case["params"]
        );
        assert_eq!(
            *bindings.finished.borrow(),
            case["finished"],
            "{id} return alias"
        );
        assert_eq!(
            json!(
                models
                    .iter()
                    .map(|m| encode(&node(m).state.additional_models().unwrap()))
                    .collect::<Vec<_>>()
            ),
            case["after"],
            "{id} source owner mutations"
        );
        // Own fixture teardown only. A general model owner's cycle collection
        // and destructor/resource policy remain a separate Stage 3 obligation.
        for model in &models {
            node(model)
                .state
                .replace_additional_models(O::dictionary(vec![]));
        }
    }
}
fn history_snapshot(state: &ModelRelationships, held: &[(String, O)]) -> Value {
    let roots = [
        ("attachments".into(), state.attachments().unwrap()),
        (
            "additional_models".into(),
            state.additional_models().unwrap(),
        ),
    ];
    let mut values = roots.to_vec();
    values.extend_from_slice(held);
    for (name, root) in &roots {
        let O::Dict(v) = root else { unreachable!() };
        values.extend(
            v.borrow()
                .iter()
                .map(|(k, v)| (format!("{name}:{k}"), v.clone())),
        );
    }
    let mut seen = Vec::<(String, O)>::new();
    let mut links = vec![];
    for (name, value) in values {
        if !matches!(value, O::Dict(_) | O::List(_) | O::Tuple(_)) {
            continue;
        }
        links.push(json!([
            name,
            seen.iter()
                .find(|(_, v)| v.same_object(&value))
                .map(|(n, _)| n)
        ]));
        seen.push((name, value));
    }
    json!({"attachments":encode(&state.attachments().unwrap()),"additional_models":encode(&state.additional_models().unwrap()),"held":held.iter().map(|(k,v)|json!([k,encode(v)])).collect::<Vec<_>>(),"links":links})
}
#[test]
fn original_model_attachments_and_additional_groups_preserve_aliases_root_replacement_fresh_defaults_and_sequence_errors()
 {
    let f = fixture();
    assert_eq!(f["histories"].as_array().unwrap().len(), 10);
    for case in f["histories"].as_array().unwrap() {
        let state = ModelRelationships::new();
        let mut held = vec![];
        let value = decode(&case["value"], &[]);
        let attachments = case["field"] == "attachments";
        assert_eq!(history_snapshot(&state, &held), case["before"]);
        for event in case["events"].as_array().unwrap() {
            let a = &event["action"];
            let key = a["key"].as_str().unwrap_or("");
            let result = (|| -> Result<(), Error> {
                match a["op"].as_str().unwrap() {
                    "get" => held.push((
                        a["hold"].as_str().unwrap().into(),
                        if attachments {
                            state.get_attachment(key)?
                        } else {
                            state.get_additional_models_with_key(key)?
                        },
                    )),
                    "set" => {
                        if attachments {
                            state.set_attachments(key, value.clone())?
                        } else {
                            state.set_additional_models(key, value.clone())?
                        }
                    }
                    "remove" => {
                        if attachments {
                            state.remove_attachments(key)?
                        } else {
                            state.remove_additional_models(key)?
                        }
                    }
                    "flatten" => held.push(("flatten".into(), state.get_additional_models()?)),
                    "hold-root" => held.push((
                        a["hold"].as_str().unwrap().into(),
                        if attachments {
                            state.attachments().unwrap()
                        } else {
                            state.additional_models().unwrap()
                        },
                    )),
                    "replace-root" => {
                        if attachments {
                            state.replace_attachments(O::dictionary(vec![]))
                        } else {
                            state.replace_additional_models(O::dictionary(vec![]))
                        }
                    }
                    _ => panic!("own action"),
                }
                Ok(())
            })();
            match result {
                Ok(()) => assert_eq!(event["ok"], true),
                Err(e) => reference_error(e, event),
            }
            assert_eq!(
                history_snapshot(&state, &held),
                event["state"],
                "{} {} {a}",
                case["field"],
                case["layout"]
            );
        }
    }
}
