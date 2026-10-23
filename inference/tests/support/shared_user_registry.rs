use imagine_inference::{
    checkpoint::{
        builtin_keys::BuiltinKeys,
        default_registry,
        user_registry::{self, Bindings},
    },
    native::Error,
    patching::options::{GeneralDictionary, GeneralMapping, Options, SourceNumber, SourceText},
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{cell::Cell, rc::Rc};
struct Owner(&'static str);
struct Alias;
struct Marked;
struct Field(Vec<u32>);
struct Substring;
struct Ty(&'static str);
struct IteratorOwner(Cell<usize>);
struct World<'a> {
    case: &'a Value,
    events: Vec<Value>,
    owners: [Options; 3],
    alias: Options,
    name: Options,
    reads: [usize; 2],
    keys: Rc<BuiltinKeys>,
}
fn gap() -> Error {
    Error::BindingGap {
        backend: "user_registry_fixture".into(),
        operation: "outside this admitted caller protocol".into(),
    }
}
fn label(value: &Options) -> &str {
    let Options::Object(value) = value else {
        panic!("owner")
    };
    value.downcast_ref::<Owner>().unwrap().0
}
impl World<'_> {
    fn mode(&self) -> &str {
        self.case["mode"].as_str().unwrap()
    }
    fn emit(&mut self, event: Value) -> Result<(), Error> {
        self.events.push(event);
        if self.case["fail"] == json!(self.events.len()) {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("own user registry failure {}", self.events.len()),
            });
        }
        Ok(())
    }
    fn owner(&self, name: &str) -> Options {
        self.owners[match name {
            "a" => 0,
            "b" => 1,
            "c" => 2,
            _ => panic!("owner"),
        }]
        .clone()
    }
    fn encode(&self, value: &Options) -> Value {
        if let Some(text) = SourceText::get(value) {
            return json!({"kind":"str","points":text.codepoints()});
        }
        if SourceNumber::get(value).is_some() {
            let imagine_inference::execution_value::Value::Int(n) = value.snapshot().unwrap()
            else {
                unreachable!()
            };
            return json!({"kind":"int","value":n});
        }
        if let Some(map) = GeneralMapping::get(value) {
            return json!({"kind":"dict","entries":map.dictionary.entries().iter().map(|(key,value)|json!([self.encode(key),self.encode(value)])).collect::<Vec<_>>()});
        }
        if let Options::List(values) = value {
            return json!({"kind":"list","items":values.borrow().iter().map(|v|self.encode(v)).collect::<Vec<_>>()});
        }
        let Options::Object(value) = value else {
            panic!("own fixture object")
        };
        if let Some(owner) = value.downcast_ref::<Owner>() {
            json!({"kind":"owner","label":owner.0})
        } else if value.is::<Alias>() {
            json!({"kind":"alias"})
        } else if let Some(field) = value.downcast_ref::<Field>() {
            json!({"kind":"field","points":field.0})
        } else {
            panic!("unexpected root")
        }
    }
}
impl default_registry::Bindings for World<'_> {
    fn load(&mut self, path: &str) -> Result<Options, Error> {
        assert_eq!(path, "_marked_safe_globals_set");
        Ok(Options::Object(Rc::new(Marked)))
    }
    fn attribute(&mut self, value: &Options, name: &str) -> Result<Options, Error> {
        let owner = label(value);
        self.emit(json!(["attr", owner, name]))?;
        let text = if name == "__module__" {
            "own"
        } else if self.mode() == "collision" {
            "same"
        } else {
            owner
        };
        Ok(Options::Object(Rc::new(Field(
            text.chars().map(u32::from).collect(),
        ))))
    }
    fn call(&mut self, _: &Options, _: &[Options]) -> Result<Options, Error> {
        Err(gap())
    }
    fn iter(&mut self, _: &Options) -> Result<Options, Error> {
        self.emit(json!(["iter"]))?;
        Ok(Options::Object(Rc::new(IteratorOwner(Cell::new(0)))))
    }
    fn next(&mut self, iterator: &Options) -> Result<Option<Options>, Error> {
        let Options::Object(iterator) = iterator else {
            panic!("iter")
        };
        let index = iterator.downcast_ref::<IteratorOwner>().unwrap();
        let i = index.0.get();
        self.emit(json!(["next", i]))?;
        if i == 3 {
            return Ok(None);
        }
        index.0.set(i + 1);
        let i = if self.mode() == "reverse" { 2 - i } else { i };
        Ok(Some(match i {
            0 => self.owner("a"),
            1 => self.alias.clone(),
            2 => self.owner("c"),
            _ => unreachable!(),
        }))
    }
    fn string(&mut self, _: &Options) -> Result<Options, Error> {
        Err(gap())
    }
    fn format(&mut self, value: &Options) -> Result<Options, Error> {
        let Options::Object(value) = value else {
            return Err(gap());
        };
        if let Some(field) = value.downcast_ref::<Field>() {
            self.emit(json!(["format", field.0]))?;
            SourceText::from_codepoints(field.0.clone())
        } else if let Some(ty) = value.downcast_ref::<Ty>() {
            Ok(SourceText::from_utf8(ty.0))
        } else {
            Err(gap())
        }
    }
    fn equal(&mut self, _: &Options, _: &Options) -> Result<bool, Error> {
        Err(gap())
    }
    fn literal_map(&mut self, entries: Vec<(&'static str, Options)>) -> Result<Options, Error> {
        assert!(entries.is_empty());
        Ok(GeneralMapping::wrap(
            GeneralDictionary::new(),
            self.keys.clone(),
        ))
    }
    fn literal_key(&mut self, _: &'static str) -> Result<Options, Error> {
        Err(gap())
    }
    fn set(&mut self, map: &Options, key: Options, value: Options) -> Result<(), Error> {
        map.set_item(key, value)
    }
}
impl Bindings for World<'_> {
    fn is_tuple(&mut self, value: &Options) -> Result<bool, Error> {
        Ok(value.same_object(&self.alias))
    }
    fn length(&mut self, _: &Options) -> Result<usize, Error> {
        self.reads[0] += 1;
        self.emit(json!(["len", self.reads[0]]))?;
        Ok(match self.mode() {
            "length-changes" => {
                if self.reads[0] == 1 {
                    1
                } else {
                    2
                }
            }
            "length-three" => 3,
            _ => 2,
        })
    }
    fn tuple_item(&mut self, _: &Options, index: usize) -> Result<Options, Error> {
        assert_eq!(index, 1);
        self.reads[1] += 1;
        self.emit(json!(["item", index, self.reads[1]]))?;
        Ok(match self.mode() {
            "wrong-type" => SourceNumber::from_i64(42),
            "string-subclass" => Options::Object(Rc::new(Substring)),
            "item-changes" => {
                if self.reads[1] == 1 {
                    Options::Object(Rc::new(Substring))
                } else {
                    SourceNumber::from_i64(42)
                }
            }
            _ => self.name.clone(),
        })
    }
    fn is_exact_string(&mut self, value: &Options) -> Result<bool, Error> {
        Ok(SourceText::get(value).is_some())
    }
    fn type_object(&mut self, value: &Options) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Ty(
            if SourceNumber::get(value).is_some() {
                "<class 'int'>"
            } else {
                "<class '__main__.run.<locals>.OwnString'>"
            },
        ))))
    }
    fn unpack_pair(&mut self, _: &Options) -> Result<(Options, Options), Error> {
        self.emit(json!(["unpack"]))?;
        match self.mode() {
            "unpack-short" => {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "not enough values to unpack (expected 2, got 1)".into(),
                });
            }
            "unpack-long" => {
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: "too many values to unpack (expected 2)".into(),
                });
            }
            _ => (),
        };
        Ok((
            self.owner(if self.mode() == "function-changes" {
                "a"
            } else {
                "b"
            }),
            match self.mode() {
                "name-number" => SourceNumber::from_i64(42),
                "name-list" => Options::List(Default::default()),
                _ => self.name.clone(),
            },
        ))
    }
}
#[test]
fn original_user_registry_body_preserves_uncached_live_iteration_tuple_subclasses_repeated_len_index_unpacked_names_partial_maps_and_exception_roots()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-user-registry.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 171);
    for (index, c) in cases.iter().enumerate() {
        let mut world = World {
            case: c,
            events: vec![],
            owners: [
                Options::Object(Rc::new(Owner("a"))),
                Options::Object(Rc::new(Owner("b"))),
                Options::Object(Rc::new(Owner("c"))),
            ],
            alias: Options::Object(Rc::new(Alias)),
            name: SourceText::from_utf8("own.alias"),
            reads: [0; 2],
            keys: Rc::new(BuiltinKeys {
                hash_state: HashState::from_seed(0),
                decimal_digit_limit: Rc::new(Cell::new(4300)),
            }),
        };
        match user_registry::build(&mut world) {
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
                assert_eq!(frames[0].function, "_get_user_allowed_globals");
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
