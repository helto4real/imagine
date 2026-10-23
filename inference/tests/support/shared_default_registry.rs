use imagine_inference::{
    checkpoint::{
        builtin_keys::BuiltinKeys,
        default_registry::{self, Bindings},
    },
    native::Error,
    patching::options::{GeneralDictionary, GeneralMapping, Options, SourceText},
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};
struct Owner(String);
struct Field(Vec<u32>);
struct Sequence(String);
struct Iter {
    group: String,
    index: Cell<usize>,
}
struct World<'a> {
    case: &'a Value,
    pool: BTreeMap<String, Options>,
    events: Vec<Value>,
    reads: BTreeMap<(String, String), usize>,
    keys: Rc<BuiltinKeys>,
    constants: BTreeMap<String, Options>,
}
fn label(value: &Options) -> &str {
    let Options::Object(value) = value else {
        panic!("owner")
    };
    &value.downcast_ref::<Owner>().unwrap().0
}
fn points(value: &Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect()
}
impl World<'_> {
    fn emit(&mut self, event: Value) -> Result<(), Error> {
        self.events.push(event);
        if self.case["fail"] == json!(self.events.len()) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: format!("own registry failure {}", self.events.len()),
            });
        }
        Ok(())
    }
    fn owner(&self, name: &str) -> Options {
        self.pool
            .get(name)
            .unwrap_or_else(|| panic!("source never created {name}"))
            .clone()
    }
    fn encode(&self, value: &Options) -> Value {
        if let Some(mapping) = GeneralMapping::get(value) {
            json!(
                mapping
                    .dictionary
                    .entries()
                    .iter()
                    .map(|(k, v)| json!([SourceText::get(k).unwrap().codepoints(), label(v)]))
                    .collect::<Vec<_>>()
            )
        } else {
            json!(label(value))
        }
    }
}
impl Bindings for World<'_> {
    fn load(&mut self, path: &str) -> Result<Options, Error> {
        let parts = path.split('.').collect::<Vec<_>>();
        for end in 2..=parts.len() {
            self.emit(json!(["get", parts[..end].join(".")]))?;
        }
        if matches!(path, "torch._tensor_classes" | "torch._storage_classes") {
            return Ok(Options::Object(Rc::new(Sequence(path.into()))));
        }
        Ok(self.owner(match path {
            "torch.storage.TypedStorage" => "typed",
            "torch.storage.UntypedStorage" => "untyped",
            _ => path,
        }))
    }
    fn attribute(&mut self, value: &Options, name: &str) -> Result<Options, Error> {
        let owner = label(value).to_owned();
        self.emit(json!(["attr", owner, name]))?;
        let count = self.reads.entry((owner.clone(), name.into())).or_default();
        *count += 1;
        let key = if name == "__module__" {
            "module"
        } else if *count > 1 {
            "name_later"
        } else {
            "name"
        };
        Ok(Options::Object(Rc::new(Field(points(
            &self.case["objects"][owner][key],
        )))))
    }
    fn call(&mut self, function: &Options, args: &[Options]) -> Result<Options, Error> {
        let texts = args
            .iter()
            .map(|value| {
                let Options::Object(value) = value else {
                    panic!("field")
                };
                let f = value.downcast_ref::<Field>().unwrap();
                f.0.iter()
                    .map(|v| char::from_u32(*v).unwrap())
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        let function = label(function);
        self.emit(json!(["call", function, texts]))?;
        if function == "torch.serialization.StorageType" {
            return Ok(self.owner(&format!("wrapped:{}", texts[0])));
        }
        Ok(Options::Object(Rc::new(Sequence(function.into()))))
    }
    fn iter(&mut self, value: &Options) -> Result<Options, Error> {
        let Options::Object(value) = value else {
            panic!("sequence")
        };
        let sequence = value.downcast_ref::<Sequence>().unwrap();
        self.emit(json!(["iter", sequence.0]))?;
        Ok(Options::Object(Rc::new(Iter {
            group: sequence.0.clone(),
            index: Cell::new(0),
        })))
    }
    fn next(&mut self, iterator: &Options) -> Result<Option<Options>, Error> {
        let Options::Object(value) = iterator else {
            panic!("iterator")
        };
        let value = value.downcast_ref::<Iter>().unwrap();
        let index = value.index.get();
        self.emit(json!(["next", value.group, index]))?;
        let values = self.case["groups"][&value.group].as_array().unwrap();
        if index == values.len() {
            return Ok(None);
        }
        value.index.set(index + 1);
        Ok(Some(self.owner(values[index].as_str().unwrap())))
    }
    fn string(&mut self, value: &Options) -> Result<Options, Error> {
        let owner = label(value);
        self.emit(json!(["str", owner]))?;
        SourceText::from_codepoints(points(&self.case["objects"][owner]["text"]))
    }
    fn format(&mut self, value: &Options) -> Result<Options, Error> {
        let Options::Object(value) = value else {
            panic!("field")
        };
        let f = value.downcast_ref::<Field>().unwrap();
        self.emit(json!(["format", f.0]))?;
        SourceText::from_codepoints(f.0.clone())
    }
    fn equal(&mut self, left: &Options, right: &Options) -> Result<bool, Error> {
        self.emit(json!(["equal", label(left), label(right)]))?;
        Ok(false)
    }
    fn literal_map(&mut self, entries: Vec<(&'static str, Options)>) -> Result<Options, Error> {
        let map = GeneralMapping::wrap(
            GeneralDictionary::presized(entries.len())?,
            self.keys.clone(),
        );
        for (key, value) in entries {
            let key = self.literal_key(key)?;
            map.set_item(key, value)?;
        }
        Ok(map)
    }
    fn literal_key(&mut self, text: &'static str) -> Result<Options, Error> {
        Ok(self
            .constants
            .entry(text.into())
            .or_insert_with(|| SourceText::from_utf8(text))
            .clone())
    }
    fn set(&mut self, dictionary: &Options, key: Options, value: Options) -> Result<(), Error> {
        dictionary.set_item(key, value)
    }
}
#[test]
fn original_complete_default_registry_body_preserves_eager_values_iterators_late_storage_name_reads_collisions_surrogates_partial_maps_and_named_exception_roots()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-default-registry.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 144);
    for (index, c) in cases.iter().enumerate() {
        let pool = c["objects"]
            .as_object()
            .unwrap()
            .keys()
            .map(|name| (name.clone(), Options::Object(Rc::new(Owner(name.clone())))))
            .collect();
        let mut world = World {
            case: c,
            pool,
            events: vec![],
            reads: BTreeMap::new(),
            keys: Rc::new(BuiltinKeys {
                hash_state: HashState::from_seed(0),
                decimal_digit_limit: Rc::new(Cell::new(4300)),
            }),
            constants: BTreeMap::new(),
        };
        match default_registry::build(&mut world) {
            Ok(value) => {
                assert!(c["error"].is_null(), "case {index}");
                assert_eq!(world.encode(&value), c["output"], "case {index}");
            }
            Err(error) => {
                let Error::ReferenceException { class, .. } = error.root() else {
                    panic!("case {index}: {error:?}")
                };
                assert_eq!(
                    json!({"type":class,"message":error.python_message().unwrap()}),
                    c["error"],
                    "case {index}"
                );
                let exception = error.into_exception();
                let frames = exception.frames();
                assert_eq!(frames.len(), 1);
                assert_eq!(frames[0].function, "_get_allowed_globals");
                let locals = frames[0].locals();
                let mut observed = serde_json::Map::new();
                for name in locals.names() {
                    let root = locals.get(&name).unwrap();
                    let value = root.downcast_ref::<Options>().unwrap();
                    observed.insert(name, world.encode(value));
                }
                assert_eq!(Value::Object(observed), c["locals"], "case {index}");
            }
        }
        assert_eq!(json!(world.events), c["events"], "case {index}");
    }
}
