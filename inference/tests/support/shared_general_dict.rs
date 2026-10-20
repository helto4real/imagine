use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{DictionaryKeyBindings, GeneralDictionary, Options},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};

enum Token {
    Key(usize),
    Value(String),
}
fn key_index(key: &Options) -> usize {
    let Options::Object(key) = key else {
        panic!("own key object")
    };
    let Token::Key(index) = key.downcast_ref::<Token>().unwrap() else {
        panic!("own key object")
    };
    *index
}
struct Bindings<'a> {
    fixture: &'a Value,
    keys: Vec<Options>,
    values: RefCell<Vec<(String, Options)>>,
    trace: RefCell<Vec<Value>>,
}
impl Bindings<'_> {
    fn decode_value(&self, wire: &Value) -> Options {
        match wire[0].as_str().unwrap() {
            "none" => Options::Atom(V::None),
            "key" => self.keys[wire[1].as_u64().unwrap() as usize].clone(),
            "value" => {
                let label = wire[1].as_str().unwrap();
                let mut values = self.values.borrow_mut();
                if let Some((_, value)) = values.iter().find(|(k, _)| k == label) {
                    return value.clone();
                }
                let value = Options::Object(Rc::new(Token::Value(label.into())));
                values.push((label.into(), value.clone()));
                value
            }
            _ => unreachable!(),
        }
    }
    fn encode(&self, value: Option<&Options>) -> Value {
        let Some(value) = value else {
            return json!(["missing"]);
        };
        let Options::Object(object) = value else {
            assert!(matches!(value, Options::Atom(V::None)));
            return json!(["none"]);
        };
        match object.downcast_ref::<Token>().unwrap() {
            Token::Key(index) => {
                assert!(value.same_object(&self.keys[*index]));
                json!(["key", index])
            }
            Token::Value(label) => {
                let values = self.values.borrow();
                assert!(value.same_object(&values.iter().find(|(k, _)| k == label).unwrap().1));
                json!(["value", label])
            }
        }
    }
}
impl DictionaryKeyBindings for Bindings<'_> {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        let index = key_index(key);
        if self.fixture["definitions"][index][0] == "key" {
            self.trace.borrow_mut().push(json!(["hash", index]));
        }
        let hash = &self.fixture["hashes"][index];
        if let Some(value) = hash.get("value") {
            Ok(value.as_i64().unwrap())
        } else {
            Err(Error::ReferenceException {
                class: "TypeError",
                message: hash["message"].as_str().unwrap().into(),
            })
        }
    }
    fn identity_token(&self, key: &Options) -> u64 {
        let Options::Object(object) = key else {
            unreachable!()
        };
        Rc::as_ptr(object) as *const () as usize as u64
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        let (a, b) = (key_index(a), key_index(b));
        let comparison = self.fixture["comparisons"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["left"] == a && v["right"] == b)
            .unwrap_or_else(|| panic!("unexpected native comparison {a} / {b}"));
        self.trace
            .borrow_mut()
            .extend(comparison["trace"].as_array().unwrap().clone());
        if let Some(value) = comparison.get("value") {
            Ok(value.as_bool().unwrap())
        } else {
            Err(Error::ReferenceException {
                class: "ValueError",
                message: comparison["message"].as_str().unwrap().into(),
            })
        }
    }
    fn exact_unicode(&self, key: &Options) -> bool {
        self.fixture["definitions"][key_index(key)][0] == "str"
    }
}
fn snapshot(maps: &[(String, GeneralDictionary)], bindings: &Bindings<'_>) -> Value {
    json!(
        maps.iter()
            .map(|(name, map)| {
                let layout = map.layout();
                json!([
                    name,
                    {
                        "capacity":layout.capacity,"used":layout.used,"usable":layout.usable,
                        "entry_count":layout.entry_count,"unicode":layout.unicode,"indices":layout.indices,
                        "entries":map.entries().iter().map(|(key,value)|json!([key_index(key),bindings.encode(Some(value))])).collect::<Vec<_>>()
                    }
                ])
            })
            .collect::<Vec<_>>()
    )
}
#[test]
fn original_general_dictionary_preserves_mixed_keys_collisions_rich_comparisons_layout_holes_compacting_copies_and_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-general-dict.json")).unwrap();
    replay(&fixture);
}
#[test]
fn original_setdefault_preserves_single_hash_single_lookup_existing_aliases_compaction_layout_and_errors()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-general-setdefault.json"
    ))
    .unwrap();
    replay(&fixture);
}
fn replay(fixture: &Value) {
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 72);
    for (case_index, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            fixture,
            keys: (0..fixture["definitions"].as_array().unwrap().len())
                .map(|index| Options::Object(Rc::new(Token::Key(index))))
                .collect(),
            values: RefCell::new(vec![]),
            trace: RefCell::new(vec![]),
        };
        let mut maps = vec![("root".into(), GeneralDictionary::new())];
        for (step_index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let action = &step["action"];
            let root = maps[0].1.clone();
            let result = (|| {
                let key = action
                    .get("key")
                    .map(|v| &bindings.keys[v.as_u64().unwrap() as usize]);
                let mut value = None;
                let mut boolean = None;
                match action["op"].as_str().unwrap() {
                    "set" => root.set(
                        key.unwrap().clone(),
                        bindings.decode_value(&action["value"]),
                        &bindings,
                    )?,
                    "setdefault" => {
                        value = Some(root.setdefault(
                            key.unwrap(),
                            &bindings.decode_value(&action["value"]),
                            &bindings,
                        )?)
                    }
                    "get" => value = root.get(key.unwrap(), &bindings)?,
                    "contains" => boolean = Some(root.contains(key.unwrap(), &bindings)?),
                    "pop" => value = root.pop(key.unwrap(), &bindings)?,
                    "clear" => root.clear(),
                    "copy" => maps.push((
                        action["output"].as_str().unwrap().into(),
                        root.shallow_copy(&bindings)?,
                    )),
                    _ => unreachable!(),
                }
                Ok(if let Some(boolean) = boolean {
                    json!({"output":["bool",boolean]})
                } else {
                    json!({"output":bindings.encode(value.as_ref())})
                })
            })();
            let result = match result {
                Ok(value) => value,
                Err(Error::ReferenceException { class, message }) => {
                    json!({"error":class,"message":message})
                }
                Err(error) => panic!("unbound native dictionary: {error:?}"),
            };
            let expected = if step.get("output").is_some() {
                json!({"output":step["output"]})
            } else {
                json!({"error":step["error"],"message":step["message"]})
            };
            assert_eq!(result, expected, "outcome {case_index}/{step_index}");
            assert_eq!(
                json!(std::mem::take(&mut *bindings.trace.borrow_mut())),
                step["trace"],
                "protocols {case_index}/{step_index}"
            );
            assert_eq!(
                snapshot(&maps, &bindings),
                step["state"],
                "state {case_index}/{step_index}"
            );
        }
    }
}
