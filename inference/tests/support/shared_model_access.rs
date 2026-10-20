use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        model_access::{self, AccessBindings},
        model_patcher::PatcherState,
        options::{GeneralMapping, Options, SourceText},
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
enum Object {
    Model(String),
    Token(String),
    Scalar(String, String),
    Size(i64),
    Other,
    Callable(Options),
}
fn object(value: &Options) -> &Object {
    let Options::Object(v) = value else {
        panic!("own object")
    };
    v.downcast_ref().unwrap()
}
fn label(value: &Options) -> &str {
    match object(value) {
        Object::Model(v) | Object::Token(v) | Object::Scalar(v, _) => v,
        _ => panic!("own label"),
    }
}
struct Bindings<'a> {
    case: &'a Value,
    state: PatcherState,
    other: RefCell<Option<Options>>,
    keys: Rc<OwnMappingKeys>,
    objects: RefCell<Vec<(String, Options)>>,
    events: RefCell<Vec<Value>>,
}
impl Bindings<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own model-access failure".into(),
            });
        }
        Ok(())
    }
    fn token(&self, label: &str) -> Options {
        if let Some((_, v)) = self.objects.borrow().iter().find(|(k, _)| k == label) {
            return v.clone();
        }
        let kind = if label == "first" || label == "second" || label.ends_with(":child") {
            Object::Model(label.into())
        } else {
            Object::Token(label.into())
        };
        let value = Options::Object(Rc::new(kind));
        self.objects
            .borrow_mut()
            .push((label.into(), value.clone()));
        value
    }
    fn decode(&self, value: &Value) -> Options {
        match value {
            Value::Null => Options::Atom(V::None),
            Value::Bool(v) => Options::Atom(V::Bool(*v)),
            Value::String(v) => SourceText::from_utf8(v),
            Value::Object(v) if v.contains_key("object") => {
                self.token(v["object"].as_str().unwrap())
            }
            Value::Object(v) if v.contains_key("size") => {
                Options::Object(Rc::new(Object::Size(v["size"].as_i64().unwrap())))
            }
            Value::Object(v) if v.contains_key("scalar") => {
                Options::Object(Rc::new(Object::Scalar(
                    v["scalar"].as_str().unwrap().into(),
                    v["group"].as_str().unwrap().into(),
                )))
            }
            Value::Object(v) if v.contains_key("dict") => {
                let map = self.keys.empty();
                for pair in v["dict"].as_array().unwrap() {
                    map.set_item(self.decode(&pair[0]), self.decode(&pair[1]))
                        .unwrap();
                }
                map
            }
            Value::Object(v) if v.contains_key("list") => Options::List(Rc::new(RefCell::new(
                v["list"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| self.decode(v))
                    .collect(),
            ))),
            Value::Object(v) if v.contains_key("tuple") => Options::Tuple(Rc::new(
                v["tuple"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| self.decode(v))
                    .collect(),
            )),
            _ => panic!("own wire {value}"),
        }
    }
    fn snap(&self, value: &Options) -> Value {
        if let Some(text) = SourceText::get(value) {
            return json!(text.utf8().unwrap());
        }
        if let Some(map) = GeneralMapping::get(value) {
            return json!({"dict":map.dictionary.entries().iter().map(|(k,v)|json!([self.snap(k),self.snap(v)])).collect::<Vec<_>>()});
        }
        match value {
            Options::Atom(V::None) => Value::Null,
            Options::Atom(V::Bool(v)) => json!(v),
            Options::List(v) => {
                json!({"list":v.borrow().iter().map(|v|self.snap(v)).collect::<Vec<_>>()})
            }
            Options::Tuple(v) => json!({"tuple":v.iter().map(|v|self.snap(v)).collect::<Vec<_>>()}),
            Options::Object(_) => match object(value) {
                Object::Model(v) | Object::Token(v) => json!({"object":v}),
                Object::Size(v) => json!({"size":v}),
                Object::Scalar(label, group) => json!({"scalar":label,"group":group}),
                _ => panic!("own result"),
            },
            _ => panic!("own result"),
        }
    }
    fn compare(&self, left: &Options, right: &Options, negate: bool) -> Result<bool, Error> {
        self.emit(json!([
            if negate { "not-equal" } else { "equal" },
            label(left),
            label(right)
        ]))?;
        let (Object::Scalar(_, a), Object::Scalar(_, b)) = (object(left), object(right)) else {
            panic!("own scalar comparison")
        };
        Ok(if negate { a != b } else { a == b })
    }
}
impl AccessBindings for Bindings<'_> {
    fn name(&self, value: &str) -> Result<Options, Error> {
        Ok(SourceText::from_utf8(value))
    }
    fn attribute(&self, value: &Options, name: &Options) -> Result<Options, Error> {
        let name = SourceText::get(name).unwrap().utf8().unwrap();
        match object(value) {
            Object::Other => {
                self.emit(json!(["other-field", name]))?;
                let other = self.other.borrow().as_ref().unwrap().get(&name)?;
                let Some(other) = other else {
                    return Err(Error::ReferenceException {
                        class: "AttributeError",
                        message: format!("own other field missing: {name}"),
                    });
                };
                if self.case["setting"] == "other-model-rebind" && name == "model" {
                    self.state.fields.set("model", self.token("second"));
                }
                if self.case["setting"] == "callbacks-rebind" && name == "callbacks" {
                    self.state.fields.set("callbacks", self.keys.empty());
                }
                Ok(other)
            }
            Object::Model(label) => {
                self.emit(json!(["attribute", label, name]))?;
                match name.as_str() {
                    "get_dtype" => {
                        if self.case["setting"] == "dtype-absent" {
                            return Err(Error::ReferenceException {
                                class: "AttributeError",
                                message: "own absent dtype".into(),
                            });
                        }
                        if self.case["setting"] == "dtype-rebind" && label == "first" {
                            self.state.fields.set("model", self.token("second"));
                        }
                        Ok(Options::Object(Rc::new(Object::Callable(value.clone()))))
                    }
                    "model_loaded_weight_memory" => Ok(self.token(&format!("{label}:loaded"))),
                    "lowvram_patch_counter" => Ok(self.token(&format!("{label}:counter"))),
                    "child" => Ok(self.token(&format!("{label}:child"))),
                    "value" => Ok(self.token(&format!("{label}:value"))),
                    _ => Err(Error::ReferenceException {
                        class: "AttributeError",
                        message: format!("own model attribute missing: {name}"),
                    }),
                }
            }
            _ => panic!("own attribute receiver"),
        }
    }
    fn split_dots(&self, name: &Options) -> Result<Options, Error> {
        let Some(text) = SourceText::get(name) else {
            return Err(Error::ReferenceException {
                class: "AttributeError",
                message: format!(
                    "'{}' object has no attribute 'split'",
                    match name {
                        Options::Tuple(_) => "tuple",
                        Options::Atom(V::None) => "NoneType",
                        _ => panic!("own invalid attribute path"),
                    }
                ),
            });
        };
        Ok(Options::List(Rc::new(RefCell::new(
            text.utf8()
                .unwrap()
                .split('.')
                .map(SourceText::from_utf8)
                .collect(),
        ))))
    }
    fn positive(&self, size: &Options) -> Result<bool, Error> {
        let Object::Size(v) = object(size) else {
            panic!("own size")
        };
        self.emit(json!(["positive", v]))?;
        if self.case["setting"] == "positive-rebind" {
            self.state.fields.set("size", self.token("rebound-size"));
        }
        Ok(*v > 0)
    }
    fn module_size(&self, model: &Options) -> Result<Options, Error> {
        self.emit(json!(["module-size", label(model)]))?;
        Ok(Options::Object(Rc::new(Object::Size(97))))
    }
    fn call_get_dtype(&self, method: &Options) -> Result<Options, Error> {
        let Object::Callable(model) = object(method) else {
            panic!("own bound method")
        };
        self.emit(json!(["call-dtype", label(model)]))?;
        Ok(self.token(&format!("{}:dtype", label(model))))
    }
    fn equal(&self, left: &Options, right: &Options) -> Result<bool, Error> {
        self.compare(left, right, false)
    }
    fn not_equal(&self, left: &Options, right: &Options) -> Result<bool, Error> {
        self.compare(left, right, true)
    }
    fn length(&self, value: &Options) -> Result<usize, Error> {
        if let Some(mapping) = GeneralMapping::get(value) {
            return Ok(mapping.dictionary.len());
        }
        match value {
            Options::List(v) => Ok(v.borrow().len()),
            Options::Tuple(v) => Ok(v.len()),
            _ => panic!("own sized value"),
        }
    }
    fn warning(&self, message: &str) -> Result<(), Error> {
        self.emit(json!(["warning", message]))
    }
}
#[test]
fn original_model_access_preserves_patch_priority_cached_size_dtype_reaccess_clone_checks_key_view_equality_and_source_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-model-access.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 280);
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let bindings = Bindings {
            case,
            state: PatcherState::uninitialized("ModelPatcher"),
            other: RefCell::new(None),
            keys: OwnMappingKeys::seeded(),
            objects: RefCell::new(vec![]),
            events: RefCell::new(vec![]),
        };
        for pair in case["initial"]["dict"].as_array().unwrap() {
            bindings
                .state
                .fields
                .set(pair[0].as_str().unwrap(), bindings.decode(&pair[1]));
        }
        *bindings.other.borrow_mut() = Some(bindings.decode(&case["other_initial"]));
        let other = Options::Object(Rc::new(Object::Other));
        let state = &bindings.state;
        let out = (|| match case["mode"].as_str().unwrap() {
            "model_size" => model_access::model_size(state, &bindings),
            "loaded_size" => model_access::loaded_size(state, &bindings),
            "lowvram_patch_counter" => model_access::lowvram_patch_counter(state, &bindings),
            "get_clone_model_override" => model_access::get_clone_model_override(state),
            "get_model_object" => {
                model_access::get_model_object(state, &bindings.decode(&case["query"]), &bindings)
            }
            "model_dtype" => model_access::model_dtype(state, &bindings),
            "is_clone" => Ok(Options::Atom(V::Bool(model_access::is_clone(
                state, &other, &bindings,
            )?))),
            "clone_has_same_weights" => model_access::clone_has_same_weights(
                state,
                &other,
                case["allow_multigpu"].as_bool().unwrap(),
                &bindings,
            ),
            _ => panic!("own mode"),
        })();
        match out {
            Ok(value) => {
                assert!(
                    case["error"].is_null(),
                    "source error case {i}: {}",
                    case["error"]
                );
                assert_eq!(bindings.snap(&value), case["output"], "output {i}");
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound model access {i}: {error:?}"),
        }
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {i} {} {}",
            case["mode"],
            case["setting"]
        );
        assert_eq!(
            json!({"dict":state.fields.entries().iter().map(|(k,v)|json!([k,bindings.snap(v)])).collect::<Vec<_>>()}),
            case["fields"],
            "fields {i}"
        );
    }
}
