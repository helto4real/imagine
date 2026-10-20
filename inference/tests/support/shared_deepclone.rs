use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        model_deepclone::{DeepCloneBindings, deepclone_multigpu},
        model_patcher::PatcherState,
        options::{GeneralMapping, Options, SourceText},
    },
};
use serde_json::{Value, json};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

enum Object {
    Owner(String, PatcherState),
    Model(String),
    Child(String, RefCell<Options>),
    Token(String),
    Set,
    Factory,
    Register(Options),
    Callback(String),
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own object")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn state(value: &Options) -> &PatcherState {
    let Object::Owner(_, state) = object(value) else {
        panic!("own owner")
    };
    state
}
fn label(value: &Options) -> &str {
    match object(value) {
        Object::Owner(label, _)
        | Object::Model(label)
        | Object::Child(label, _)
        | Object::Token(label) => label,
        _ => panic!("own labelled object"),
    }
}
fn text(value: &Options) -> String {
    SourceText::get(value).unwrap().utf8().unwrap()
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn tuple(values: Vec<Options>) -> Options {
    Options::Tuple(Rc::new(values))
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn failure(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
struct Bindings<'a> {
    settings: &'a Value,
    action: &'a str,
    failure: Option<usize>,
    keys: Rc<OwnMappingKeys>,
    owners: RefCell<Vec<Options>>,
    pool: RefCell<BTreeMap<String, Options>>,
    children: RefCell<Vec<Options>>,
    events: RefCell<Vec<Value>>,
}
impl Bindings<'_> {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.failure == Some(events.len() - 1) {
            return Err(failure("ValueError", "own deepclone failure"));
        }
        Ok(())
    }
    fn token(&self, name: &str) -> Options {
        self.pool
            .borrow_mut()
            .entry(name.into())
            .or_insert_with(|| Options::Object(Rc::new(Object::Token(name.into()))))
            .clone()
    }
    fn owner(&self, name: &str) -> Options {
        let result = Options::Object(Rc::new(Object::Owner(
            name.into(),
            PatcherState::uninitialized("Owner"),
        )));
        let fields = &state(&result).fields;
        fields.set(
            "model",
            Options::Object(Rc::new(Object::Model(format!("model:{name}")))),
        );
        fields.set("load_device", SourceText::from_utf8("old-device"));
        fields.set("hook_backup", self.token("stale-hook-backup"));
        fields.set("additional_models", self.keys.empty());
        self.owners.borrow_mut().push(result.clone());
        result
    }
    fn target(&self) -> Options {
        self.owners
            .borrow()
            .iter()
            .find(|v| label(v) == "target")
            .unwrap()
            .clone()
    }
    fn snap(value: &Options) -> Value {
        if SourceText::get(value).is_some() {
            return json!(text(value));
        }
        if let Some(mapping) = GeneralMapping::get(value) {
            return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([text(k),Self::snap(v)])).collect::<Vec<_>>()});
        }
        match value {
            Options::Atom(V::None) => Value::Null,
            Options::List(values) => {
                json!({"list":values.borrow().iter().map(Self::snap).collect::<Vec<_>>()})
            }
            Options::Tuple(values) => {
                json!({"tuple":values.iter().map(Self::snap).collect::<Vec<_>>()})
            }
            Options::Object(_) if matches!(object(value), Object::Set) => json!({"set":[]}),
            Options::Object(_) => json!({"object":label(value)}),
            _ => panic!("own snapshot"),
        }
    }
    fn owner_snap(&self, owner: &Options) -> Value {
        let fields = &state(owner).fields;
        let mut result = json!({"label":label(owner),"model":label(&fields.get("model").unwrap()),
            "load_device":text(&fields.get("load_device").unwrap()),
            "hook_backup":Self::snap(&fields.get("hook_backup").unwrap()),
            "additional_models":Self::snap(&fields.get("additional_models").unwrap())});
        if let Ok(backups) = fields.get("own_backups") {
            result["backups"] = Self::snap(&backups);
        }
        result
    }
}
impl DeepCloneBindings for Bindings<'_> {
    fn state<'a>(&self, owner: &'a Options) -> Result<&'a PatcherState, Error> {
        Ok(state(owner))
    }
    fn model_class_name(&self, model: &Options) -> Result<String, Error> {
        assert!(matches!(object(model), Object::Model(_)));
        Ok("Model".into())
    }
    fn format_device(&self, device: &Options) -> Result<String, Error> {
        Ok(text(device))
    }
    fn info(&self, message: &str) -> Result<(), Error> {
        self.emit(json!(["info", message]))
    }
    fn truth(&self, value: &Options) -> Result<bool, Error> {
        if matches!(value, Options::Atom(V::None)) {
            return Ok(false);
        }
        Ok(!text(value).is_empty())
    }
    fn unload_model_and_clones(&self, owner: &Options) -> Result<(), Error> {
        self.emit(json!(["unload", label(owner)]))
    }
    fn factory(&self, callable: &Options, arguments: &Options) -> Result<Options, Error> {
        assert!(matches!(object(callable), Object::Factory));
        let Options::Tuple(arguments) = arguments else {
            panic!("own factory args")
        };
        self.emit(json!([
            "factory",
            arguments.iter().map(text).collect::<Vec<_>>(),
            {}
        ]))?;
        let fresh = self.owner("fresh");
        Ok(if self.settings["wrapped"] == true {
            list(vec![self.token("unselected"), fresh])
        } else {
            fresh
        })
    }
    fn index(&self, value: &Options, key: &Options) -> Result<Options, Error> {
        if let Options::Atom(V::Int(index)) = key {
            return match value {
                Options::Tuple(values) => Ok(values[*index as usize].clone()),
                Options::List(values) => values
                    .borrow()
                    .get(*index as usize)
                    .cloned()
                    .ok_or_else(|| failure("IndexError", "list index out of range")),
                _ => panic!("own sequence"),
            };
        }
        value.item(key)
    }
    fn length(&self, value: &Options) -> Result<usize, Error> {
        Ok(match value {
            Options::Tuple(values) => values.len(),
            Options::List(values) => values.borrow().len(),
            _ => panic!("own sequence"),
        })
    }
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.keys.empty())
    }
    fn empty_pinned_set(&self) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Object::Set)))
    }
    fn clone_with_override(
        &self,
        owner: &Options,
        model_override: &Options,
    ) -> Result<Options, Error> {
        let Options::Tuple(values) = model_override else {
            panic!("own override")
        };
        let Options::Tuple(backups) = &values[1] else {
            panic!("own backups")
        };
        self.emit(json!([
            "clone",
            label(owner),
            label(&values[0]),
            backups.len()
        ]))?;
        let target = self.owner("target");
        state(&target).fields.set("model", values[0].clone());
        state(&target).fields.set("own_backups", values[1].clone());
        let groups = self.keys.empty();
        for (key, group) in GeneralMapping::get(&state(owner).fields.get("additional_models")?)
            .unwrap()
            .dictionary
            .entries()
        {
            let Options::List(values) = group else {
                panic!("own group")
            };
            groups.set_item(key, list(values.borrow().clone()))?;
        }
        state(&target).fields.set("additional_models", groups);
        Ok(target)
    }
    fn attribute(&self, owner: &Options, name: &str) -> Result<Options, Error> {
        if name == "register_load_device" {
            self.emit(json!(["attribute", label(owner), name]))?;
            if self.settings["register"] != true {
                return Err(failure("AttributeError", "own absent register"));
            }
            return Ok(Options::Object(Rc::new(Object::Register(owner.clone()))));
        }
        if name == "clone_base_uuid" {
            let Object::Child(_, uuid) = object(owner) else {
                panic!("own child")
            };
            return Ok(uuid.borrow().clone());
        }
        assert_eq!(name, "model");
        state(owner).fields.get(name)
    }
    fn register_load_device(&self, callable: &Options, device: &Options) -> Result<(), Error> {
        let Object::Register(owner) = object(callable) else {
            panic!("own method")
        };
        self.emit(json!(["register", label(owner), text(device)]))?;
        if self.action == "register-append" {
            let Options::List(values) = state(owner)
                .fields
                .get("additional_models")?
                .get("a")?
                .unwrap()
            else {
                panic!("own list")
            };
            values.borrow_mut().push(self.children.borrow()[1].clone());
        }
        Ok(())
    }
    fn cache_contains(&self, cache: &Options, key: &Options) -> Result<bool, Error> {
        let mapping = GeneralMapping::get(cache).unwrap();
        Ok(mapping
            .dictionary
            .get(key, mapping.bindings.as_ref())?
            .is_some())
    }
    fn deepclone_child(
        &self,
        child: &Options,
        device: &Options,
        cache: &Options,
    ) -> Result<Options, Error> {
        let keys = GeneralMapping::get(cache)
            .unwrap()
            .dictionary
            .entries()
            .iter()
            .map(|(k, _)| text(k))
            .collect::<Vec<_>>();
        let device = if matches!(device, Options::Atom(V::None)) {
            Value::Null
        } else {
            json!(text(device))
        };
        self.emit(json!(["child", label(child), device, keys]))?;
        let target = self.target();
        let groups = state(&target).fields.get("additional_models")?;
        match self.action {
            "child-clear-list" | "child-append-list" => {
                let Options::List(values) = groups.get("a")?.unwrap() else {
                    panic!("own list")
                };
                if self.action == "child-clear-list" {
                    values.borrow_mut().clear();
                } else {
                    values.borrow_mut().push(self.children.borrow()[1].clone());
                }
            }
            "child-clear-groups" => groups.clear_dict()?,
            "child-change-uuid" => {
                let Object::Child(_, uuid) = object(child) else {
                    panic!("own child")
                };
                *uuid.borrow_mut() = SourceText::from_utf8("changed-uuid");
            }
            _ => {}
        }
        Ok(self.token(&format!("deep:{}", label(child))))
    }
    fn set_index(&self, container: &Options, key: &Options, value: Options) -> Result<(), Error> {
        let Options::List(values) = container else {
            panic!("own list")
        };
        let Options::Atom(V::Int(index)) = key else {
            panic!("own index")
        };
        let mut values = values.borrow_mut();
        let target = values
            .get_mut(*index as usize)
            .ok_or_else(|| failure("IndexError", "list assignment index out of range"))?;
        *target = value;
        Ok(())
    }
    fn callbacks(&self, owner: &Options) -> Result<Options, Error> {
        self.emit(json!(["callbacks", label(owner), "on_deepclone_multigpu"]))?;
        Ok(list(
            ["first", "last"]
                .iter()
                .map(|v| Options::Object(Rc::new(Object::Callback((*v).into()))))
                .collect(),
        ))
    }
    fn callback(&self, callback: &Options, owner: &Options, target: &Options) -> Result<(), Error> {
        let Object::Callback(name) = object(callback) else {
            panic!("own callback")
        };
        self.emit(json!(["callback", name, label(owner), label(target)]))?;
        if self.action == "callback-device" && name == "last" {
            state(target)
                .fields
                .set("load_device", SourceText::from_utf8("callback-device"));
        }
        Ok(())
    }
}
#[test]
fn original_deepclone_preserves_pristine_factory_calls_backups_device_registration_postorder_cache_aliases_and_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-deepclone.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 69);
    for (index, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            settings: &case["settings"],
            action: case["action"].as_str().unwrap(),
            failure: case["failure"].as_u64().map(|v| v as usize),
            keys: OwnMappingKeys::seeded(),
            owners: RefCell::new(vec![]),
            pool: RefCell::new(BTreeMap::new()),
            children: RefCell::new(vec![]),
            events: RefCell::new(vec![]),
        };
        let source = bindings.owner("source");
        let children = [
            "u:0",
            if bindings.settings["alias_uuid"] == true {
                "u:0"
            } else {
                "u:1"
            },
            "u:excluded",
        ]
        .iter()
        .enumerate()
        .map(|(i, uuid)| {
            Options::Object(Rc::new(Object::Child(
                if i == 2 {
                    "excluded".into()
                } else {
                    format!("child:{i}")
                },
                RefCell::new(SourceText::from_utf8(uuid)),
            )))
        })
        .collect::<Vec<_>>();
        let groups = bindings.keys.empty();
        groups
            .set("multigpu", list(vec![children[2].clone()]))
            .unwrap();
        groups
            .set(
                "a",
                list(vec![
                    children[0].clone(),
                    children[1].clone(),
                    children[0].clone(),
                ]),
            )
            .unwrap();
        groups.set("b", list(vec![children[1].clone()])).unwrap();
        *bindings.children.borrow_mut() = children;
        state(&source).fields.set("additional_models", groups);
        let cached = if bindings.settings["missing_factory"] == true {
            none()
        } else {
            let mut cached = vec![
                Options::Object(Rc::new(Object::Factory)),
                tuple(vec![SourceText::from_utf8("own-factory")]),
            ];
            if bindings.settings["wrapped"] == true {
                cached.push(Options::Atom(V::Int(1)));
            }
            tuple(cached)
        };
        state(&source).fields.set("cached_patcher_init", cached);
        let cache = if bindings.settings["preseed"] == true {
            let cache = bindings.keys.empty();
            cache.set("u:0", bindings.token("preseed")).unwrap();
            cache
        } else {
            none()
        };
        let device = bindings.settings["device"]
            .as_str()
            .map_or_else(none, SourceText::from_utf8);
        let result = deepclone_multigpu(&source, &device, &cache, &bindings);
        let (result, error) = match result {
            Ok(result) => (json!(label(&result)), Value::Null),
            Err(Error::ReferenceException { class, message }) => {
                (Value::Null, json!([class, message]))
            }
            Err(error) => panic!("native deepclone {index}: {error:?}"),
        };
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(result, case["result"], "result {index}");
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(
            json!(
                bindings
                    .owners
                    .borrow()
                    .iter()
                    .map(|v| bindings.owner_snap(v))
                    .collect::<Vec<_>>()
            ),
            case["owners"],
            "owners {index}"
        );
        assert_eq!(Bindings::snap(&cache), case["cache"], "cache {index}");
    }
}
