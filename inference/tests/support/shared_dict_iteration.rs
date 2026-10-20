use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{Dictionary, ItemsIterator, KeysIterator, Options, ValuesIterator},
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};
struct Token {
    label: usize,
}
fn token(refs: &mut Vec<Weak<Token>>) -> Options {
    let t = Rc::new(Token { label: refs.len() });
    refs.push(Rc::downgrade(&t));
    Options::Object(t)
}
fn label(value: &Options) -> usize {
    let Options::Object(t) = value else {
        panic!("own token expected")
    };
    t.downcast_ref::<Token>().unwrap().label
}
fn pair(value: &Options) -> Value {
    if matches!(value, Options::Object(_)) {
        return json!(label(value));
    }
    if let Options::Atom(V::Text(key)) = value {
        return json!(key);
    }
    let Options::Tuple(v) = value else {
        panic!("source items tuple expected")
    };
    let V::Text(key) = v[0].snapshot().unwrap() else {
        panic!("own string key")
    };
    json!([key, label(&v[1])])
}
enum Traversal {
    Items(ItemsIterator),
    Keys(KeysIterator),
    Values(ValuesIterator),
}
impl Traversal {
    fn length_hint(&self) -> usize {
        match self {
            Self::Items(v) => v.length_hint(),
            Self::Keys(v) => v.length_hint(),
            Self::Values(v) => v.length_hint(),
        }
    }
    fn next(&self) -> Result<Option<Options>, Error> {
        match self {
            Self::Items(v) => v.next_item(),
            Self::Keys(v) => v.next_key().map(|v| v.map(|v| Options::Atom(V::Text(v)))),
            Self::Values(v) => v.next_value(),
        }
    }
}
fn snapshot(
    d: &Option<Options>,
    it: &Option<Traversal>,
    held: &[(String, Options)],
    refs: &[Weak<Token>],
) -> Value {
    let dictionary = d.as_ref().map(|d| {
        let Options::Dict(v) = d else {
            panic!("own dict")
        };
        v.borrow()
            .iter()
            .map(|(k, v)| json!([k, label(v)]))
            .collect::<Vec<_>>()
    });
    let held = held
        .iter()
        .map(|(k, v)| {
            let p = pair(v);
            if p.is_string() || p.is_number() {
                json!([k, p])
            } else {
                json!([k, p[0], p[1]])
            }
        })
        .collect::<Vec<_>>();
    json!({"dictionary":dictionary,"held":held,"live":refs.iter().map(|v|v.upgrade().is_some()).collect::<Vec<_>>(),"hint":it.as_ref().map(Traversal::length_hint)})
}
#[test]
fn source_live_dictionary_items_preserve_mutations_resize_copy_tuple_identity_cached_owners_and_sticky_errors()
 {
    run_cases(
        include_str!("../../fixtures/shared-dict-iteration.json"),
        TraversalKind::Items,
    );
}
#[test]
fn source_live_dictionary_keys_preserve_mutations_resize_copy_exhaustion_and_value_owners() {
    run_cases(
        include_str!("../../fixtures/shared-dict-keys.json"),
        TraversalKind::Keys,
    );
}
#[test]
fn source_live_dictionary_values_preserve_mutations_resize_copy_exhaustion_aliases_and_uncached_value_owners()
 {
    run_cases(
        include_str!("../../fixtures/shared-dict-values.json"),
        TraversalKind::Values,
    );
}
#[derive(Clone, Copy)]
enum TraversalKind {
    Items,
    Keys,
    Values,
}
fn run_cases(text: &str, kind: TraversalKind) {
    let fixture: Value = serde_json::from_str(text).unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["torch_python_imported"], false);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 864);
    for (index, c) in cases.iter().enumerate() {
        let p = &c["params"];
        let note = format!("dictionary history {index} {p}");
        let mut refs = vec![];
        let pairs = (0..p["count"].as_u64().unwrap())
            .map(|i| (format!("k{i}"), token(&mut refs)))
            .collect();
        let mut dictionary = if p["style"] == "presized" {
            Options::Dict(Rc::new(RefCell::new(Dictionary::presized(pairs))))
        } else {
            Options::dictionary(pairs)
        };
        for key in p["removed"].as_array().unwrap() {
            drop(dictionary.pop(key.as_str().unwrap()).unwrap());
        }
        if p["copy"].as_bool().unwrap() {
            dictionary = dictionary.shallow_copy_dict().unwrap();
        }
        let mut it = Some(match kind {
            TraversalKind::Keys => Traversal::Keys(dictionary.keys_iterator().unwrap()),
            TraversalKind::Items => Traversal::Items(dictionary.items_iterator().unwrap()),
            TraversalKind::Values => Traversal::Values(dictionary.values_iterator().unwrap()),
        });
        let mut d = Some(dictionary);
        let mut held: Vec<(String, Options)> = vec![];
        assert_eq!(snapshot(&d, &it, &held, &refs), c["initial"], "{note}");
        for (event_index, event) in c["events"].as_array().unwrap().iter().enumerate() {
            let a = &event["action"];
            let note = format!("{note} action {event_index} {a}");
            match a["op"].as_str().unwrap() {
                "next" => match it.as_ref().unwrap().next() {
                    Ok(Some(value)) => {
                        assert_eq!(pair(&value), event["output"], "{note}");
                        let aliases = if matches!(kind, TraversalKind::Keys) {
                            vec![]
                        } else {
                            held.iter()
                                .map(|(k, v)| json!([k, v.same_object(&value)]))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(json!(aliases), event["same_held"], "{note}");
                        if let Some(key) = a["keep"].as_str() {
                            if let Some((_, old)) = held.iter_mut().find(|(k, _)| k == key) {
                                *old = value;
                            } else {
                                held.push((key.into(), value));
                            }
                        }
                    }
                    Ok(None) => {
                        assert_eq!(event["error"], "StopIteration", "{note}");
                        assert_eq!(event["message"], "");
                    }
                    Err(Error::ReferenceException { class, message }) => {
                        assert_eq!(event["error"], class, "{note}");
                        assert_eq!(event["message"], message, "{note}");
                    }
                    Err(e) => panic!("{note}: {e:?}"),
                },
                "set" => d
                    .as_ref()
                    .unwrap()
                    .set(a["key"].as_str().unwrap(), token(&mut refs))
                    .unwrap(),
                "pop" => drop(d.as_ref().unwrap().pop(a["key"].as_str().unwrap()).unwrap()),
                "clear" => d.as_ref().unwrap().clear_dict().unwrap(),
                "copy" => d = Some(d.as_ref().unwrap().shallow_copy_dict().unwrap()),
                "drop-dict" => d = None,
                "drop-iterator" => it = None,
                "drop-held" => {
                    if let Some(i) = held
                        .iter()
                        .position(|(k, _)| Some(k.as_str()) == a["key"].as_str())
                    {
                        held.remove(i);
                    }
                }
                _ => panic!("{note}"),
            }
            assert_eq!(snapshot(&d, &it, &held, &refs), event["post"], "{note}");
        }
        assert!(
            refs.iter().all(|v| v.upgrade().is_none()),
            "{note}: leaked own token"
        );
    }
}

struct Finalizer {
    run: Box<dyn Fn()>,
}
#[derive(Default)]
struct ValuesReentrant {
    iterator: Option<Rc<ValuesIterator>>,
    events: Vec<Value>,
}
#[test]
fn dictionary_value_iterator_releases_its_dictionary_before_reentrant_finalizers_at_exhaustion_and_drop()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dict-values.json")).unwrap();
    assert_eq!(fixture["reentrant"].as_array().unwrap().len(), 2);
    for case in fixture["reentrant"].as_array().unwrap() {
        let state = Rc::new(RefCell::new(ValuesReentrant::default()));
        let weak = Rc::downgrade(&state);
        let dictionary = Options::dictionary(vec![(
            "one".into(),
            Options::Object(Rc::new(Finalizer {
                run: Box::new(move || {
                    let state = weak.upgrade().unwrap();
                    let iterator = state.borrow().iterator.clone();
                    state.borrow_mut().events.push(json!({"tag":"finalizer", "iterator_none":iterator.is_none(), "hint":iterator.as_ref().map(|v| v.length_hint())}));
                    if let Some(iterator) = iterator {
                        assert!(iterator.next_value().unwrap().is_none());
                        state
                            .borrow_mut()
                            .events
                            .push(json!({"tag":"nested", "error":"StopIteration", "message":""}));
                    }
                }),
            })),
        )]);
        let iterator = Rc::new(dictionary.values_iterator().unwrap());
        *state.borrow_mut() = ValuesReentrant {
            iterator: Some(iterator.clone()),
            events: vec![],
        };
        drop(iterator.next_value().unwrap());
        drop(dictionary);
        state.borrow_mut().events.push(json!({"tag":"before"}));
        if case["mode"] == "exhaust" {
            assert!(iterator.next_value().unwrap().is_none());
            state
                .borrow_mut()
                .events
                .push(json!({"tag":"outer", "error":"StopIteration", "message":""}));
        } else {
            drop(iterator);
            let old = state.borrow_mut().iterator.take();
            drop(old);
        }
        assert_eq!(json!(state.borrow().events), case["events"]);
    }
}
impl Drop for Finalizer {
    fn drop(&mut self) {
        (self.run)();
    }
}
#[derive(Default)]
struct Reentrant {
    dictionary: Option<Options>,
    iterator: Option<Rc<ItemsIterator>>,
    nested: Option<Options>,
    events: Vec<Value>,
}
fn strings(d: &Options) -> Value {
    let V::Dict(v) = d.snapshot().unwrap() else {
        panic!("own string dict")
    };
    json!(
        v.into_iter()
            .map(|(k, v)| {
                let V::Text(v) = v else {
                    panic!("replacement not committed before finalizer")
                };
                json!([k, v])
            })
            .collect::<Vec<_>>()
    )
}
fn string_pair(value: &Options) -> Value {
    let V::Tuple(v) = value.snapshot().unwrap() else {
        panic!("own tuple")
    };
    json!(
        v.into_iter()
            .map(|v| {
                let V::Text(v) = v else { panic!("own string") };
                v
            })
            .collect::<Vec<_>>()
    )
}
#[test]
fn dictionary_replacement_clear_and_cached_items_release_borrows_before_reentrant_source_finalizers()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dict-iteration.json")).unwrap();
    assert_eq!(f["reentrant"].as_array().unwrap().len(), 3);
    for case in f["reentrant"].as_array().unwrap() {
        let mode = case["mode"].as_str().unwrap();
        let state = Rc::new(RefCell::new(Reentrant::default()));
        let d = Options::dictionary(vec![]);
        state.borrow_mut().dictionary = Some(d.clone());
        let captured = state.clone();
        let cached = mode == "cached-item";
        d.set(
            "a",
            Options::Object(Rc::new(Finalizer {
                run: Box::new(move || {
                    if cached {
                        let it = captured.borrow().iterator.as_ref().unwrap().clone();
                        captured
                            .borrow_mut()
                            .events
                            .push(json!(["before", it.length_hint()]));
                        let value = it.next_item().unwrap().unwrap();
                        let event = json!(["nested", string_pair(&value), it.length_hint()]);
                        let mut state = captured.borrow_mut();
                        state.nested = Some(value);
                        state.events.push(event);
                    } else {
                        let d = captured.borrow().dictionary.as_ref().unwrap().clone();
                        let before = strings(&d);
                        captured.borrow_mut().events.push(json!(["before", before]));
                        d.set("from-finalizer", Options::Atom(V::Text("value".into())))
                            .unwrap();
                        let after = strings(&d);
                        captured.borrow_mut().events.push(json!(["after", after]));
                    }
                }),
            })),
        )
        .unwrap();
        let mut outer = None;
        match mode {
            "set" => d
                .set("a", Options::Atom(V::Text("replacement".into())))
                .unwrap(),
            "clear" => d.clear_dict().unwrap(),
            _ => {
                d.set("b", Options::Atom(V::Text("bee".into()))).unwrap();
                d.set("c", Options::Atom(V::Text("see".into()))).unwrap();
                let it = Rc::new(d.items_iterator().unwrap());
                state.borrow_mut().iterator = Some(it.clone());
                drop(it.next_item().unwrap().unwrap());
                d.set("a", Options::Atom(V::Text("replacement".into())))
                    .unwrap();
                outer = it.next_item().unwrap();
                let nested = state.borrow().nested.clone().unwrap();
                assert_eq!(string_pair(outer.as_ref().unwrap()), case["outer"]);
                assert_eq!(string_pair(&nested), case["nested"]);
                assert_eq!(
                    outer.as_ref().unwrap().same_object(&nested),
                    case["same_tuple"].as_bool().unwrap()
                );
                assert_eq!(json!(it.length_hint()), case["hint"]);
            }
        }
        assert_eq!(json!(state.borrow().events), case["events"]);
        assert_eq!(strings(&d), case["dictionary"]);
        drop(outer);
        *state.borrow_mut() = Reentrant::default();
    }
}
