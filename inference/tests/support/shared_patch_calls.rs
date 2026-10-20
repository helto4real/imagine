use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        model_patch_calls::{
            PatchCallBindings, model_patches_call_function, model_patches_models, model_patches_to,
        },
        model_patcher::PatcherState,
        options::{GeneralMapping, Options, SourceText},
    },
    sampling::cast_options::cast_to_load_options,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
};

enum Object {
    Patch(u64),
    Bound(Options, String),
    Partial,
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own patch object")
    };
    value.downcast_ref::<Object>().expect("own patch object")
}
fn text(value: &Options) -> String {
    if let Some(value) = SourceText::get(value) {
        return value.utf8().unwrap();
    }
    let Options::Atom(V::Text(value)) = value else {
        panic!("own text")
    };
    value.clone()
}
fn failure(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
struct Bindings {
    state: PatcherState,
    keys: Rc<OwnMappingKeys>,
    patches: RefCell<BTreeMap<u64, Options>>,
    events: RefCell<Vec<Value>>,
    action: String,
    failure: Option<usize>,
    fired: Cell<bool>,
    group: RefCell<Option<Options>>,
    groups: RefCell<Option<Options>>,
    transformer: RefCell<Option<Options>>,
}
impl Bindings {
    fn patch(&self, index: u64) -> Options {
        self.patches
            .borrow_mut()
            .entry(index)
            .or_insert_with(|| Options::Object(Rc::new(Object::Patch(index))))
            .clone()
    }
    fn parse(&self, value: &Value) -> Options {
        if let Some(index) = value.get("patch") {
            return self.patch(index.as_u64().unwrap());
        }
        if let Some(entries) = value.get("dict") {
            let result = self.keys.empty();
            for pair in entries.as_array().unwrap() {
                result
                    .set_item(self.parse(&pair[0]), self.parse(&pair[1]))
                    .unwrap();
            }
            return result;
        }
        for (tag, tuple) in [("list", false), ("tuple", true)] {
            if let Some(values) = value.get(tag) {
                let items = values
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| self.parse(v))
                    .collect();
                return if tuple {
                    Options::Tuple(Rc::new(items))
                } else {
                    Options::List(Rc::new(RefCell::new(items)))
                };
            }
        }
        match value {
            Value::Null => Options::Atom(V::None),
            Value::String(value) => self.name(value),
            Value::Number(value) => Options::Atom(V::Int(value.as_i64().unwrap())),
            _ => panic!("own scalar"),
        }
    }
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(failure("ValueError", "own patch-call failure"));
        }
        Ok(())
    }
    fn mutate(&self, stage: &str, index: u64) {
        let wanted = if self.action.starts_with("call-") {
            "call"
        } else {
            "attribute"
        };
        if self.fired.get() || index != 0 || stage != wanted {
            return;
        }
        self.fired.set(true);
        match self.action.trim_start_matches("call-") {
            "replace-item" => {
                let Options::List(values) = self.group.borrow().as_ref().unwrap().clone() else {
                    unreachable!()
                };
                values.borrow_mut()[0] = self.patch(6);
            }
            "clear-list" | "append-list" => {
                let Options::List(values) = self.group.borrow().as_ref().unwrap().clone() else {
                    unreachable!()
                };
                if self.action.ends_with("clear-list") {
                    values.borrow_mut().clear();
                } else {
                    values.borrow_mut().push(self.patch(6));
                }
            }
            "clear-groups" => self.groups.borrow().as_ref().unwrap().clear_dict().unwrap(),
            "remove-replacements" => {
                self.transformer
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .pop("patches_replace")
                    .unwrap();
            }
            "replace-options" => {
                let options = self.keys.empty();
                options
                    .set("model_function_wrapper", self.patch(7))
                    .unwrap();
                self.state.fields.set("model_options", options);
            }
            "replace-wrapper" => {
                self.state
                    .fields
                    .get("model_options")
                    .unwrap()
                    .set("model_function_wrapper", self.patch(7))
                    .unwrap();
            }
            "none" | "bad-models" | "partial-models" => {}
            _ => panic!("own mutation"),
        }
    }
    fn snap(value: &Options) -> Value {
        if let Some(mapping) = GeneralMapping::get(value) {
            return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([Self::snap(k),Self::snap(v)])).collect::<Vec<_>>()});
        }
        if SourceText::get(value).is_some() {
            return json!(text(value));
        }
        match value {
            Options::Atom(V::None) => Value::Null,
            Options::Atom(V::Int(value)) => json!(value),
            Options::List(values) => {
                json!({"list":values.borrow().iter().map(Self::snap).collect::<Vec<_>>()})
            }
            Options::Tuple(values) => {
                json!({"tuple":values.iter().map(Self::snap).collect::<Vec<_>>()})
            }
            Options::Object(_) => match object(value) {
                Object::Patch(index) => json!({"patch":index}),
                _ => panic!("own serialized patch"),
            },
            _ => panic!("own snapshot"),
        }
    }
}
impl PatchCallBindings for Bindings {
    fn name(&self, name: &str) -> Options {
        SourceText::from_utf8(name)
    }
    fn contains(&self, container: &Options, key: &str) -> Result<bool, Error> {
        Ok(container.get(key)?.is_some())
    }
    fn item(&self, container: &Options, key: &Options) -> Result<Options, Error> {
        if let Options::List(values) = container {
            let Options::Atom(V::Int(index)) = key else {
                panic!("own index")
            };
            return values
                .borrow()
                .get(*index as usize)
                .cloned()
                .ok_or_else(|| failure("IndexError", "list index out of range"));
        }
        container.item(key)
    }
    fn set_item(&self, container: &Options, key: &Options, value: Options) -> Result<(), Error> {
        if let Options::List(values) = container {
            let Options::Atom(V::Int(index)) = key else {
                panic!("own index")
            };
            let mut values = values.borrow_mut();
            let target = values
                .get_mut(*index as usize)
                .ok_or_else(|| failure("IndexError", "list assignment index out of range"))?;
            *target = value;
            return Ok(());
        }
        container.set_item(key.clone(), value)
    }
    fn length(&self, container: &Options) -> Result<usize, Error> {
        let Options::List(values) = container else {
            panic!("own patch list")
        };
        Ok(values.borrow().len())
    }
    fn attribute(&self, receiver: &Options, name: &Options) -> Result<Options, Error> {
        let Object::Patch(index) = object(receiver) else {
            panic!("own receiver")
        };
        let name = text(name);
        self.emit(json!(["attribute", index, name]))?;
        self.mutate("attribute", *index);
        if *index == 1 {
            return Err(failure("AttributeError", "own absent method"));
        }
        Ok(Options::Object(Rc::new(Object::Bound(
            receiver.clone(),
            name,
        ))))
    }
    fn call(
        &self,
        callable: &Options,
        positional: &[Options],
        keywords: Option<&Options>,
    ) -> Result<Options, Error> {
        let Object::Bound(receiver, name) = object(callable) else {
            panic!("own method")
        };
        let Object::Patch(index) = object(receiver) else {
            unreachable!()
        };
        let args = positional
            .iter()
            .map(|v| json!(text(v)))
            .collect::<Vec<_>>();
        let mut kwargs = serde_json::Map::new();
        if let Some(keywords) = keywords {
            for (key, value) in GeneralMapping::get(keywords).unwrap().dictionary.entries() {
                kwargs.insert(text(&key), Self::snap(&value));
            }
        }
        self.emit(json!(["call", index, name, args, kwargs]))?;
        self.mutate("call", *index);
        if name == "to" {
            return Ok(self.patch(100 + index));
        }
        if name == "models" {
            if self.action == "bad-models" && *index == 0 {
                return Ok(Options::Atom(V::Int(3)));
            }
            if self.action == "partial-models" && *index == 0 {
                return Ok(Options::Object(Rc::new(Object::Partial)));
            }
            return Ok(Options::List(Rc::new(RefCell::new(vec![
                self.name(&format!("m:{index}")),
                self.name("shared"),
            ]))));
        }
        Ok(self.patch(200 + index))
    }
    fn extend(&self, list: &Options, iterable: &Options) -> Result<(), Error> {
        let Options::List(target) = list else {
            panic!("own result list")
        };
        if let Options::List(values) = iterable {
            target.borrow_mut().extend(values.borrow().iter().cloned());
            return Ok(());
        }
        if matches!(iterable, Options::Atom(V::Int(3))) {
            return Err(failure("TypeError", "'int' object is not iterable"));
        }
        assert!(matches!(object(iterable), Object::Partial));
        target.borrow_mut().push(self.name("m:0"));
        Err(failure("ValueError", "own partial models"))
    }
}
#[test]
fn original_patch_calls_preserve_live_groups_descriptor_reaccess_partial_mutations_and_errors() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-patch-calls.json")).unwrap();
    run_fixture(&fixture, 122);
}
#[test]
fn original_cast_options_preserve_single_hasattr_multiple_replacements_live_registries_partial_casts_and_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-cast-options.json")).unwrap();
    run_fixture(&fixture, 128);
}
fn run_fixture(fixture: &Value, count: usize) {
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), count);
    for (index, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            state: PatcherState::uninitialized("OwnPatcher"),
            keys: OwnMappingKeys::seeded(),
            patches: RefCell::new(BTreeMap::new()),
            events: RefCell::new(vec![]),
            action: case["action"].as_str().unwrap().into(),
            failure: case["failure"].as_u64().map(|v| v as usize),
            fired: Cell::new(false),
            group: RefCell::new(None),
            groups: RefCell::new(None),
            transformer: RefCell::new(None),
        };
        let options = bindings.parse(&case["initial"]);
        let transformer = options.get("transformer_options").unwrap().unwrap();
        let mode = case["mode"].as_str().unwrap();
        if mode.starts_with("cast-") {
            // Source deliberately exposes this same transformer through both
            // keys; reconstruct that own fixture alias after JSON decoding.
            options.set("to_load_options", transformer.clone()).unwrap();
        }
        let groups = transformer.get("patches").unwrap().unwrap();
        *bindings.group.borrow_mut() = groups.get("a").unwrap();
        *bindings.groups.borrow_mut() = Some(groups);
        *bindings.transformer.borrow_mut() = Some(transformer.clone());
        bindings.state.fields.set("model_options", options);
        let result = match mode {
            mode if mode.starts_with("cast-") => {
                let device = if ["cast-both", "cast-device"].contains(&mode) {
                    bindings.name("own-device")
                } else {
                    Options::Atom(V::None)
                };
                let dtype = if ["cast-both", "cast-dtype"].contains(&mode) {
                    bindings.name("own-dtype")
                } else {
                    Options::Atom(V::None)
                };
                cast_to_load_options(
                    &bindings.state.fields.get("model_options").unwrap(),
                    &device,
                    &dtype,
                    &bindings,
                )
                .map(|()| Options::Atom(V::None))
            }
            "to" => model_patches_to(&bindings.state, &bindings.name("own-device"), &bindings)
                .map(|()| Options::Atom(V::None)),
            "models" => model_patches_models(&bindings.state, &bindings),
            _ => {
                let keywords = bindings.keys.empty();
                keywords
                    .set("own_argument", Options::Atom(V::Int(9)))
                    .unwrap();
                model_patches_call_function(
                    &bindings.state,
                    &bindings.name(mode),
                    &keywords,
                    &bindings,
                )
                .map(|()| Options::Atom(V::None))
            }
        };
        let (result, error) = match result {
            Ok(result) => (Bindings::snap(&result), Value::Null),
            Err(Error::ReferenceException { class, message }) => {
                (Value::Null, json!([class, message]))
            }
            Err(error) => panic!("native protocol {index}: {error:?}"),
        };
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(result, case["result"], "result {index}");
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(
            Bindings::snap(&bindings.state.fields.get("model_options").unwrap()),
            case["options"],
            "options {index}"
        );
        assert_eq!(
            Bindings::snap(&transformer),
            case["captured_transformer"],
            "captured {index}"
        );
    }
}
