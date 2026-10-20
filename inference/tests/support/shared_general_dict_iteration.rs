use imagine_inference::{
    native::Error,
    patching::options::{
        DictionaryKeyBindings, GeneralDictionary, GeneralDictionaryIterator, Options,
    },
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, rc::Rc, rc::Weak};

enum Object {
    Key(String),
    Token(usize),
}
fn object(value: &Options) -> &Object {
    let Options::Object(value) = value else {
        panic!("own object expected")
    };
    value.downcast_ref::<Object>().unwrap()
}
fn key_name(value: &Options) -> &str {
    let Object::Key(name) = object(value) else {
        panic!("own key")
    };
    name
}
fn value_label(value: &Options) -> usize {
    let Object::Token(label) = object(value) else {
        panic!("own value")
    };
    *label
}
struct Bindings(BTreeMap<String, (i64, Options)>);
impl Bindings {
    fn key(&self, name: &str) -> Options {
        self.0[name].1.clone()
    }
}
impl DictionaryKeyBindings for Bindings {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        Ok(self.0[key_name(key)].0)
    }
    fn identity_token(&self, value: &Options) -> u64 {
        let Options::Object(value) = value else {
            unreachable!()
        };
        Rc::as_ptr(value) as *const () as usize as u64
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        Ok(key_name(a) == key_name(b))
    }
    fn exact_unicode(&self, _: &Options) -> bool {
        false
    }
}
fn token(refs: &mut Vec<Weak<Object>>) -> Options {
    let value = Rc::new(Object::Token(refs.len()));
    refs.push(Rc::downgrade(&value));
    Options::Object(value)
}
fn encode(value: &Options, kind: &str) -> Value {
    match kind {
        "keys" => json!(key_name(value)),
        "values" => json!(value_label(value)),
        _ => {
            let Options::Tuple(pair) = value else {
                unreachable!()
            };
            json!([key_name(&pair[0]), value_label(&pair[1])])
        }
    }
}
fn snapshot(
    dictionary: &Option<GeneralDictionary>,
    iterator: &Option<GeneralDictionaryIterator>,
    held: &[(String, Options)],
    refs: &[Weak<Object>],
    kind: &str,
) -> Value {
    let dictionary = dictionary.as_ref().map(|dictionary| {
        dictionary
            .entries()
            .iter()
            .map(|(key, value)| json!([key_name(key), value_label(value)]))
            .collect::<Vec<_>>()
    });
    let held = held
        .iter()
        .map(|(name, value)| {
            let encoded = encode(value, kind);
            if kind == "items" {
                json!([name, encoded[0], encoded[1]])
            } else {
                json!([name, encoded])
            }
        })
        .collect::<Vec<_>>();
    json!({
        "dictionary":dictionary,"held":held,
        "live":refs.iter().map(|v|v.upgrade().is_some()).collect::<Vec<_>>(),
        "hint":iterator.as_ref().map(GeneralDictionaryIterator::length_hint)
    })
}
fn run(source: &str) {
    let fixture: Value = serde_json::from_str(source).unwrap();
    let kind = fixture["kind"].as_str().unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    let bindings = Bindings(
        fixture["key_hashes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let name = pair[0].as_str().unwrap();
                (
                    name.into(),
                    (
                        pair[1].as_i64().unwrap(),
                        Options::Object(Rc::new(Object::Key(name.into()))),
                    ),
                )
            })
            .collect(),
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 864);
    for (case_index, case) in cases.iter().enumerate() {
        let params = &case["params"];
        let count = params["count"].as_u64().unwrap() as usize;
        let mut refs = vec![];
        let mut dictionary = if params["style"] == "presized" {
            GeneralDictionary::presized(count).unwrap()
        } else {
            GeneralDictionary::new()
        };
        for index in 0..count {
            dictionary
                .set(
                    bindings.key(&format!("k{index}")),
                    token(&mut refs),
                    &bindings,
                )
                .unwrap();
        }
        for key in params["removed"].as_array().unwrap() {
            drop(
                dictionary
                    .pop(&bindings.key(key.as_str().unwrap()), &bindings)
                    .unwrap(),
            );
        }
        if params["copy"].as_bool().unwrap() {
            dictionary = dictionary.shallow_copy(&bindings).unwrap();
        }
        let mut iterator = Some(match kind {
            "keys" => dictionary.keys_iterator(),
            "values" => dictionary.values_iterator(),
            _ => dictionary.items_iterator(),
        });
        let mut dictionary = Some(dictionary);
        let mut held: Vec<(String, Options)> = vec![];
        assert_eq!(
            snapshot(&dictionary, &iterator, &held, &refs, kind),
            case["initial"]
        );
        for (step_index, step) in case["events"].as_array().unwrap().iter().enumerate() {
            let action = &step["action"];
            let note = format!("{kind} {case_index}/{step_index}: {action}");
            match action["op"].as_str().unwrap() {
                "next" => match iterator.as_ref().unwrap().next() {
                    Ok(Some(value)) => {
                        assert_eq!(encode(&value, kind), step["output"], "{note}");
                        assert_eq!(
                            json!(
                                held.iter()
                                    .map(|(name, old)| json!([name, old.same_object(&value)]))
                                    .collect::<Vec<_>>()
                            ),
                            step["same_held"],
                            "{note}"
                        );
                        if let Some(name) = action["keep"].as_str() {
                            if let Some((_, old)) = held.iter_mut().find(|(key, _)| key == name) {
                                *old = value;
                            } else {
                                held.push((name.into(), value));
                            }
                        }
                    }
                    Ok(None) => {
                        assert_eq!(step["error"], "StopIteration", "{note}");
                        assert_eq!(step["message"], "");
                    }
                    Err(Error::ReferenceException { class, message }) => {
                        assert_eq!(step["error"], class, "{note}");
                        assert_eq!(step["message"], message, "{note}");
                    }
                    Err(error) => panic!("{note}: {error:?}"),
                },
                "set" => dictionary
                    .as_ref()
                    .unwrap()
                    .set(
                        bindings.key(action["key"].as_str().unwrap()),
                        token(&mut refs),
                        &bindings,
                    )
                    .unwrap(),
                "pop" => drop(
                    dictionary
                        .as_ref()
                        .unwrap()
                        .pop(&bindings.key(action["key"].as_str().unwrap()), &bindings)
                        .unwrap(),
                ),
                "clear" => dictionary.as_ref().unwrap().clear(),
                "copy" => {
                    dictionary = Some(
                        dictionary
                            .as_ref()
                            .unwrap()
                            .shallow_copy(&bindings)
                            .unwrap(),
                    )
                }
                "drop-dict" => dictionary = None,
                "drop-iterator" => iterator = None,
                "drop-held" => {
                    if let Some(index) = held
                        .iter()
                        .position(|(name, _)| Some(name.as_str()) == action["key"].as_str())
                    {
                        held.remove(index);
                    }
                }
                _ => unreachable!(),
            }
            assert_eq!(
                snapshot(&dictionary, &iterator, &held, &refs, kind),
                step["post"],
                "{note}"
            );
        }
        assert!(refs.iter().all(|value| value.upgrade().is_none()));
    }
}
#[test]
fn original_general_dictionary_items_preserve_none_tuple_keys_mutations_resize_cached_tuple_identity_and_value_lifetimes()
 {
    run(include_str!(
        "../../fixtures/shared-general-dict-items.json"
    ));
}
#[test]
fn original_general_dictionary_keys_preserve_native_key_identity_mutations_sticky_errors_and_value_release()
 {
    run(include_str!("../../fixtures/shared-general-dict-keys.json"));
}
#[test]
fn original_general_dictionary_values_preserve_mutations_exhaustion_errors_and_uncached_owners() {
    run(include_str!(
        "../../fixtures/shared-general-dict-values.json"
    ));
}
