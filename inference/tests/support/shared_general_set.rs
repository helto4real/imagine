use imagine_inference::{
    native::Error,
    patching::options::{
        DictionaryKeyBindings, GeneralSet, GeneralSetIterator, MappingKeyBindings, Options,
        SourceText, WeakGeneralSet,
    },
    source_key_error::{KeyError, KeyRepresentation},
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Key(usize);
fn index(value: &Options) -> usize {
    let Options::Object(value) = value else {
        panic!("key")
    };
    value.downcast_ref::<Key>().unwrap().0
}
struct State {
    case: Value,
    events: RefCell<Vec<Value>>,
    keys: RefCell<Vec<Option<Options>>>,
    target: RefCell<Option<WeakGeneralSet>>,
    active: Cell<bool>,
    done: Cell<bool>,
    faults: Cell<bool>,
}
#[derive(Clone)]
struct Keys(Rc<State>);
impl State {
    fn key(&self, index: usize) -> Options {
        self.keys.borrow()[index].as_ref().unwrap().clone()
    }
    fn emit(&self, event: Value) -> Result<(), Error> {
        let callback = matches!(event[0].as_str(), Some("hash" | "equal"));
        self.events.borrow_mut().push(event);
        let n = self.events.borrow().len();
        if callback && self.faults.get() && self.case["fail"] == json!(n) {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: format!("own key callback failure {n}"),
            });
        }
        Ok(())
    }
    fn trigger(&self, kind: &str, left: usize, right: Option<usize>) -> Result<bool, Error> {
        let action = &self.case["action"];
        if !self.active.get() || self.done.get() || action.is_null() || action["trigger"] != kind {
            return Ok(false);
        }
        let count = self.case["count"].as_u64().unwrap() as usize;
        if kind == "equal" && right != Some(count + 1) || kind == "hash" && left != count + 1 {
            return Ok(false);
        }
        self.done.set(true);
        let verb = action["kind"].as_str().unwrap();
        self.emit(json!(["action", verb]))?;
        let set = self.target.borrow().as_ref().unwrap().upgrade().unwrap();
        match verb {
            "clear" => set.clear(),
            "remove" => {
                set.discard_key(&self.key(left))?;
            }
            "grow" => {
                for index in count + 10..count + 50 {
                    set.add(&self.key(index))?;
                }
            }
            "fill-dummy" => set.add(&self.key(count + 2))?,
            _ => panic!("action"),
        }
        Ok(action["return_true"].as_bool().unwrap_or(false))
    }
}
impl DictionaryKeyBindings for Keys {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        let i = index(key);
        self.0.emit(json!(["hash", i]))?;
        self.0.trigger("hash", i, None)?;
        let hash = self.0.case["hashes"][i].as_i64().unwrap();
        Ok(if hash == -1 { -2 } else { hash })
    }
    fn identity_token(&self, key: &Options) -> u64 {
        let Options::Object(key) = key else {
            panic!("key")
        };
        Rc::as_ptr(key) as *const () as usize as u64
    }
    fn equal(&self, left: &Options, right: &Options) -> Result<bool, Error> {
        let (a, b) = (index(left), index(right));
        self.0.emit(json!(["equal", a, b]))?;
        let changed = self.0.trigger("equal", a, Some(b))?;
        Ok(changed || self.0.case["groups"][a] == self.0.case["groups"][b])
    }
    fn exact_unicode(&self, _: &Options) -> bool {
        false
    }
}
impl MappingKeyBindings for Keys {
    fn text_key(&self, key: &str) -> Result<Options, Error> {
        Ok(SourceText::from_utf8(key))
    }
    fn text_contents(&self, _: &Options) -> Result<Option<String>, Error> {
        Ok(None)
    }
    fn missing_key(&self, key: &Options) -> Result<Error, Error> {
        Ok(Error::KeyError(KeyError::new(
            key.clone(),
            Rc::new(self.clone()),
        )))
    }
}
impl KeyRepresentation for Keys {
    fn repr(&self, key: &Options) -> Result<String, Error> {
        let i = index(key);
        self.0.emit(json!(["repr", i]))?;
        Ok(format!("<own key {i}>"))
    }
}
fn perform(
    set: &GeneralSet,
    state: &State,
    iterator: &mut Option<GeneralSetIterator>,
    op: &Value,
) -> Result<Value, Error> {
    let parts = op.as_array().unwrap();
    let key = (parts.len() > 1).then(|| state.key(parts[1].as_u64().unwrap() as usize));
    match parts[0].as_str().unwrap() {
        "add" => {
            set.add(key.as_ref().unwrap())?;
        }
        "discard" => {
            set.discard_key(key.as_ref().unwrap())?;
        }
        "remove" => set.remove_key(key.as_ref().unwrap())?,
        "contains" => return Ok(json!(set.contains_key(key.as_ref().unwrap())?)),
        "iter" => *iterator = Some(set.iterator()),
        "clear" => set.clear(),
        "next" => {
            return iterator
                .as_ref()
                .unwrap()
                .next()?
                .map(|key| json!(index(&key)))
                .ok_or_else(|| Error::ReferenceException {
                    class: "StopIteration",
                    message: String::new(),
                });
        }
        _ => panic!("operation"),
    }
    Ok(Value::Null)
}
#[test]
fn original_general_set_key_kernels_preserve_fallible_callbacks_reentrant_positive_add_restart_rules_dummies_resizes_owners_and_sticky_iterators()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-general-set.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 96);
    for (case_index, c) in cases.iter().enumerate() {
        let state = Rc::new(State {
            case: c.clone(),
            events: RefCell::new(vec![]),
            keys: RefCell::new(
                (0..c["hashes"].as_array().unwrap().len())
                    .map(|i| Some(Options::Object(Rc::new(Key(i)))))
                    .collect(),
            ),
            target: RefCell::new(None),
            active: Cell::new(false),
            done: Cell::new(false),
            faults: Cell::new(true),
        });
        let keys = Rc::new(Keys(state.clone()));
        let set = GeneralSet::new(keys.clone());
        *state.target.borrow_mut() = Some(set.downgrade());
        let mut iterator = None;
        let steps = c["steps"].as_array().unwrap();
        for (step, expected) in steps.iter().enumerate() {
            state
                .active
                .set(!c["action"].is_null() && step + 1 == steps.len());
            let output = perform(&set, &state, &mut iterator, &expected["op"]);
            let (result, error) = match output {
                Ok(value) => (value, Value::Null),
                Err(error) => {
                    let class = if error.is_reference("KeyError") {
                        "KeyError"
                    } else {
                        let Error::ReferenceException { class, .. } = error.root() else {
                            panic!("case {case_index} step {step}: {error:?}")
                        };
                        *class
                    };
                    (
                        Value::Null,
                        json!({"type":class,"message":error.python_message().unwrap()}),
                    )
                }
            };
            assert_eq!(result, expected["result"], "case {case_index} step {step}");
            assert_eq!(error, expected["error"], "case {case_index} step {step}");
            assert_eq!(
                json!(
                    set.entries()
                        .iter()
                        .map(|(key, _)| index(key))
                        .collect::<Vec<_>>()
                ),
                expected["order"],
                "case {case_index} step {step}"
            );
            assert_eq!(
                set.len(),
                expected["length"].as_u64().unwrap() as usize,
                "case {case_index} step {step}"
            );
            assert_eq!(
                set.capacity(),
                expected["capacity"].as_u64().unwrap() as usize,
                "case {case_index} step {step}"
            );
            assert_eq!(
                json!(iterator.as_ref().map(GeneralSetIterator::length_hint)),
                expected["hint"],
                "case {case_index} step {step}"
            );
            assert_eq!(
                state.events.borrow().len(),
                expected["events"].as_u64().unwrap() as usize,
                "case {case_index} step {step}"
            );
        }
        assert_eq!(
            json!(*state.events.borrow()),
            c["events"],
            "case {case_index}"
        );
        if let Some(expected) = c["helper_alive_after_release"].as_bool() {
            let i = c["count"].as_u64().unwrap() as usize + 2;
            let key = state.key(i);
            let Options::Object(owner) = &key else {
                unreachable!()
            };
            let weak = Rc::downgrade(owner);
            drop(key);
            drop(state.keys.borrow_mut()[i].take());
            set.clear();
            assert_eq!(
                weak.upgrade().is_some(),
                expected,
                "source overwritten-dummy reference effect"
            );
        }
        let value = set.clone().into_options();
        assert!(GeneralSet::get(&value).unwrap().same_object(&set));
        assert!(matches!(value.snapshot(), Err(Error::BindingGap { .. })));
        for (i, args) in [
            vec![value.clone()],
            vec![
                imagine_inference::patching::options::SourceNumber::from_i64(1),
                value.clone(),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let error = match imagine_inference::checkpoint::builtin_scalar_ops::complex(&args) {
                Err(error) => error,
                Ok(_) => panic!("set cannot be numeric"),
            };
            assert_eq!(
                json!({"type":"TypeError","message":error.python_message().unwrap()}),
                fixture["complex_errors"][i]
            );
            assert!(error.is_reference("TypeError"));
        }
    }
}

fn set_view(set: &GeneralSet) -> Value {
    json!({"order":set.entries().iter().map(|(key,_)|index(key)).collect::<Vec<_>>(),"length":set.len(),"capacity":set.capacity()})
}
#[test]
fn original_set_copy_merge_union_intersection_difference_preserve_cached_hashes_holes_resize_thresholds_aliases_and_partial_error_prefixes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-set-operators.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 549);
    run_operator_cases(cases);
}
fn run_operator_cases(cases: &[Value]) {
    for (case_index, c) in cases.iter().enumerate() {
        let state = Rc::new(State {
            case: c.clone(),
            events: RefCell::new(vec![]),
            keys: RefCell::new(
                (0..c["hashes"].as_array().unwrap().len())
                    .map(|i| Some(Options::Object(Rc::new(Key(i)))))
                    .collect(),
            ),
            target: RefCell::new(None),
            active: Cell::new(false),
            done: Cell::new(false),
            faults: Cell::new(false),
        });
        let bindings = Rc::new(Keys(state.clone()));
        let left = GeneralSet::new(bindings.clone());
        let mut right = GeneralSet::new(bindings);
        for i in c["left_ids"].as_array().unwrap() {
            left.add(&state.key(i.as_u64().unwrap() as usize)).unwrap();
        }
        for i in c["right_ids"].as_array().unwrap() {
            right.add(&state.key(i.as_u64().unwrap() as usize)).unwrap();
        }
        for i in c["removed"].as_array().unwrap() {
            left.discard_key(&state.key(i.as_u64().unwrap() as usize))
                .unwrap();
        }
        let operation = c["operation"].as_str().unwrap();
        if operation.ends_with("-self") {
            right = left.clone();
        }
        state.events.borrow_mut().clear();
        state.faults.set(true);
        *state.target.borrow_mut() = Some(left.downgrade());
        let mut answer = Value::Null;
        let result: Result<Option<GeneralSet>, Error> = match operation.trim_end_matches("-self") {
            "copy" => left.copy_set().map(Some),
            "merge" => left.merge_set(&right).map(|()| None),
            "union" => left.union_set(&right).map(Some),
            "intersection" => left.intersection_set(&right).map(Some),
            "difference" => left.difference_set(&right).map(Some),
            "difference-update" => left.difference_update_set(&right).map(|()| None),
            "symmetric-difference" => left.symmetric_difference_set(&right).map(Some),
            "symmetric-update" => left.symmetric_difference_update_set(&right).map(|()| None),
            "intersection-update" => left.intersection_update_set(&right).map(|()| None),
            "subset" => left.is_subset_set(&right).map(|value| {
                answer = json!(value);
                None
            }),
            "superset" => right.is_subset_set(&left).map(|value| {
                answer = json!(value);
                None
            }),
            "disjoint" => left.is_disjoint_set(&right).map(|value| {
                answer = json!(value);
                None
            }),
            _ => panic!("operator"),
        };
        match result {
            Ok(value) => {
                assert!(c["error"].is_null(), "case {case_index}");
                assert_eq!(
                    value.as_ref().map_or(Value::Null, set_view),
                    c["result"],
                    "case {case_index}"
                );
                assert_eq!(
                    value.as_ref().is_some_and(|v| v.same_object(&left)),
                    c["result_is_left"].as_bool().unwrap()
                );
            }
            Err(error) => {
                let Error::ReferenceException { class, .. } = error.root() else {
                    panic!("case {case_index}: {error:?}")
                };
                assert_eq!(
                    json!({"type":class,"message":error.python_message().unwrap()}),
                    c["error"],
                    "case {case_index}"
                );
            }
        }
        assert_eq!(
            answer,
            c.get("answer").cloned().unwrap_or(Value::Null),
            "answer {case_index}"
        );
        assert_eq!(set_view(&left), c["left"], "case {case_index} left");
        assert_eq!(set_view(&right), c["right"], "case {case_index} right");
        assert_eq!(
            json!(*state.events.borrow()),
            c["events"],
            "case {case_index}"
        );
    }
}
#[test]
fn original_remaining_exact_set_operators_predicates_pop_cursor_clear_and_body_swap_preserve_order_errors_and_partial_mutations()
 {
    use imagine_inference::{
        checkpoint::builtin_keys::BuiltinKeys, patching::options::SourceNumber,
        python_hash::HashState,
    };
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-set-remaining.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 543);
    run_operator_cases(cases);
    for c in fixture["pop_cases"].as_array().unwrap() {
        let keys = Rc::new(BuiltinKeys {
            hash_state: HashState::from_seed(0),
            decimal_digit_limit: Rc::new(Cell::new(4300)),
        });
        let set = GeneralSet::new(keys.clone());
        for v in c["values"].as_array().unwrap() {
            set.add(&SourceNumber::from_i64(v.as_i64().unwrap()))
                .unwrap();
        }
        for v in c["first"].as_array().unwrap() {
            let key = set.pop().unwrap();
            assert_eq!(
                SourceNumber::get(&key)
                    .unwrap()
                    .repr_with_limit(4300)
                    .unwrap(),
                v.to_string()
            );
        }
        match c["phase"].as_str().unwrap() {
            "plain" => {}
            "clear" => {
                set.clear();
                for v in c["values"].as_array().unwrap() {
                    set.add(&SourceNumber::from_i64(v.as_i64().unwrap()))
                        .unwrap();
                }
            }
            "intersection" => {
                let other = GeneralSet::new(keys);
                for v in c["values"].as_array().unwrap().iter().step_by(2) {
                    other
                        .add(&SourceNumber::from_i64(v.as_i64().unwrap()))
                        .unwrap();
                }
                set.intersection_update_set(&other).unwrap();
            }
            _ => panic!("phase"),
        }
        for v in c["remaining"].as_array().unwrap() {
            let key = set.pop().unwrap();
            assert_eq!(
                SourceNumber::get(&key)
                    .unwrap()
                    .repr_with_limit(4300)
                    .unwrap(),
                v.to_string()
            );
        }
        assert!(set.is_empty());
        let error = match set.pop() {
            Err(error) => error,
            Ok(_) => panic!("empty"),
        };
        let Error::KeyError(e) = &error else {
            panic!("KeyError")
        };
        let Options::Tuple(args) = e.arguments() else {
            panic!("args")
        };
        assert_eq!(
            SourceText::get(&args[0]).unwrap().utf8().unwrap(),
            c["error"]["argument"].as_str().unwrap()
        );
        assert_eq!(
            json!({"type":"KeyError","message":error.python_message().unwrap(),"argument":SourceText::get(&args[0]).unwrap().utf8().unwrap()}),
            c["error"]
        );
    }
}
