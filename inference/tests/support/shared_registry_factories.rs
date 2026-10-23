use imagine_inference::{
    checkpoint::{
        builtin_keys::BuiltinKeys,
        dtype_objects::SourceDtype,
        registry_cache::RegistryCache,
        registry_sets::{self, Bindings},
    },
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    python_hash::HashState,
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Token(usize);
struct World<'a> {
    case: &'a Value,
    values: Vec<Options>,
    events: Vec<Value>,
    counts: [usize; 2],
}
fn error(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
impl World<'_> {
    fn event(&mut self, kind: &str, value: Value) -> Result<(), Error> {
        let index = usize::from(kind == "hash");
        self.counts[index] += 1;
        self.events.push(json!([kind, value]));
        if self.case["fail"] == json!([kind, self.counts[index]]) {
            return Err(error(format!("own {kind} failure {}", self.counts[index])));
        }
        Ok(())
    }
}
impl Bindings for World<'_> {
    fn load(&mut self, path: &str) -> Result<Options, Error> {
        let parts = path.split('.').collect::<Vec<_>>();
        for end in 2..=parts.len() {
            self.event("get", json!(parts[..end].join(".")))?;
        }
        let index = self.case["paths"]
            .as_array()
            .unwrap()
            .iter()
            .position(|p| p == path)
            .unwrap();
        Ok(self.values[self.case["ids"][index].as_u64().unwrap() as usize].clone())
    }
    fn hash(&mut self, value: &Options) -> Result<i64, Error> {
        let Options::Object(v) = value else {
            panic!("token")
        };
        let index = v.downcast_ref::<Token>().unwrap().0;
        self.event("hash", json!(index))?;
        let hash = self.case["hashes"][index].as_i64().unwrap();
        Ok(if hash == -1 { -2 } else { hash })
    }
}
#[test]
fn original_registry_identity_literals_preserve_eager_getters_hash_failure_prefixes_collisions_aliases_and_exact_controlled_set_order()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-registry-factories.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 76);
    for (index, c) in cases.iter().enumerate() {
        let paths = match c["factory"].as_str().unwrap() {
            "new_dtypes" => registry_sets::NEW_DTYPES,
            "rebuild_functions" => registry_sets::REBUILD_FUNCTIONS,
            _ => panic!("factory"),
        };
        assert_eq!(json!(paths), c["paths"]);
        let mut world = World {
            case: c,
            values: (0..paths.len())
                .map(|i| Options::Object(Rc::new(Token(i))))
                .collect(),
            events: vec![],
            counts: [0; 2],
        };
        match registry_sets::literal_from_paths(paths, &mut world) {
            Ok(set) => {
                assert!(c["error"].is_null(), "case {index}");
                assert_eq!(
                    json!(
                        set.iter()
                            .map(|value| {
                                let Options::Object(v) = value else {
                                    unreachable!()
                                };
                                v.downcast_ref::<Token>().unwrap().0
                            })
                            .collect::<Vec<_>>()
                    ),
                    c["order"],
                    "case {index}"
                );
                assert_eq!(set.capacity(), c["capacity"].as_u64().unwrap() as usize);
            }
            Err(error) => {
                let Error::ReferenceException { class, .. } = error.root() else {
                    panic!("{error:?}")
                };
                assert_eq!(
                    json!({"type":class,"message":error.python_message().unwrap()}),
                    c["error"],
                    "case {index}"
                );
            }
        }
        assert_eq!(json!(world.events), c["events"], "case {index}");
    }
    struct Actual;
    impl Bindings for Actual {
        fn load(&mut self, path: &str) -> Result<Options, Error> {
            Ok(SourceDtype::lookup(path).unwrap())
        }
        fn hash(&mut self, value: &Options) -> Result<i64, Error> {
            Ok(SourceDtype::get(value).unwrap().hash())
        }
    }
    let actual = registry_sets::literal_from_paths(registry_sets::NEW_DTYPES, &mut Actual).unwrap();
    assert_eq!(actual.iter().count(), 15);
    let keys = BuiltinKeys {
        hash_state: HashState::from_seed(0),
        decimal_digit_limit: Rc::new(Cell::new(4300)),
    };
    for value in actual.iter() {
        assert!(value.same_object(
            &SourceDtype::lookup(&SourceDtype::get(value).unwrap().representation()).unwrap()
        ));
        assert_eq!(
            imagine_inference::patching::options::DictionaryKeyBindings::hash(&keys, value)
                .unwrap(),
            SourceDtype::get(value).unwrap().hash()
        );
    }
}
#[derive(Default)]
struct Factory {
    count: Cell<usize>,
    events: RefCell<Vec<Value>>,
    roots: RefCell<Vec<(String, Options)>>,
}
fn call(cache: &RegistryCache, mode: &str, state: &Factory) -> Result<Options, Error> {
    cache.get_or_build(|| {
        let number = state.count.get() + 1;
        state.count.set(number);
        state.events.borrow_mut().push(json!(["build", number]));
        if number == 1 {
            if mode == "retry" {
                return Err(error("own factory failure"));
            }
            if matches!(mode, "recursive" | "recursive-error" | "clear-after-inner") {
                let inner = call(cache, mode, state)?;
                state.roots.borrow_mut().push(("inner".into(), inner));
                if mode == "recursive-error" {
                    return Err(error("own outer failure"));
                }
            }
            if matches!(mode, "clear-in-body" | "clear-after-inner") {
                cache.clear();
                state.events.borrow_mut().push(json!(["clear"]));
            }
        }
        Ok(Options::dictionary(vec![(
            "label".into(),
            Options::Atom(V::Text(if number == 2 { "inner" } else { "outer" }.into())),
        )]))
    })
}
fn snapshot(value: &Options) -> Value {
    let V::Dict(values) = value.snapshot().unwrap() else {
        panic!("dict")
    };
    let mut out = serde_json::Map::new();
    for (key, value) in values {
        out.insert(
            key,
            match value {
                V::Text(s) => json!(s),
                V::Int(n) => json!(n),
                _ => panic!("fixture"),
            },
        );
    }
    Value::Object(out)
}
#[test]
fn original_no_argument_registry_cache_preserves_mutable_aliases_errors_recursive_inner_winner_outer_return_clear_and_cache_statistics()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-registry-factories.json"
    ))
    .unwrap();
    let cases = fixture["cache_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for c in cases {
        let cache = RegistryCache::default();
        let state = Factory::default();
        let mode = c["mode"].as_str().unwrap();
        for turn in 0..3 {
            let result = match call(&cache, mode, &state) {
                Ok(value) => {
                    state
                        .roots
                        .borrow_mut()
                        .push((format!("turn{turn}"), value.clone()));
                    if turn == 0 {
                        value.set("mutated", Options::Atom(V::Int(1))).unwrap();
                    }
                    json!({"value":snapshot(&value)})
                }
                Err(error) => {
                    json!({"error":{"type":"ValueError","message":error.python_message().unwrap()}})
                }
            };
            assert_eq!(result, c["turns"][turn]["result"], "{mode} turn {turn}");
            assert_eq!(
                json!(cache.info()),
                c["turns"][turn]["info"],
                "{mode} turn {turn}"
            );
        }
        assert_eq!(json!(*state.events.borrow()), c["events"], "{mode}");
        let roots = state.roots.borrow();
        let mut identity = serde_json::Map::new();
        for (name, value) in roots.iter() {
            identity.insert(
                name.clone(),
                json!(roots.iter().find(|(_, v)| v.same_object(value)).unwrap().0),
            );
        }
        assert_eq!(Value::Object(identity), c["identity"], "{mode}");
    }
}
