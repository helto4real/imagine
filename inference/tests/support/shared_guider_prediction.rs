use super::own_mapping_keys::OwnMappingKeys;
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        extensions::{Original, Wrapper, WrapperExecutor},
        options::{GeneralMapping, Options, SourceText},
    },
    sampling::{
        guider::GuiderState,
        guider_prediction::{
            PredictionArguments, PredictionBindings, PredictionWrappers, SamplingArguments,
            outer_predict_noise, predict_noise,
        },
    },
};
use serde_json::{Value, json};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::{Rc, Weak},
};
struct Token(String);
fn token(value: &str) -> Options {
    Options::Object(Rc::new(Token(value.into())))
}
fn none() -> Options {
    Options::Atom(V::None)
}
type Names = Rc<RefCell<BTreeMap<usize, String>>>;
fn snap(value: &Options, names: &Names) -> Value {
    if let Some(value) = SourceText::get(value) {
        return json!(value.utf8().unwrap());
    }
    if let Some(mapping) = GeneralMapping::get(value) {
        return json!({"dict":mapping.dictionary.entries().iter().map(|(k,v)|json!([snap(k,names),snap(v,names)])).collect::<Vec<_>>()});
    }
    match value {
        Options::Atom(V::None) => Value::Null,
        Options::Atom(V::Text(v)) => json!(v),
        Options::Atom(V::Int(v)) => json!(v),
        Options::Atom(V::Float(v)) => json!(v),
        Options::List(values) => {
            json!({"list":values.borrow().iter().map(|v|snap(v,names)).collect::<Vec<_>>()})
        }
        Options::Object(value) => {
            if let Some(wrapper) = value.downcast_ref::<Wrapper<Options>>() {
                return json!({"function":names.borrow()[&(Rc::as_ptr(wrapper) as *const () as usize)]});
            }
            json!({"object":value.downcast_ref::<Token>().unwrap().0})
        }
        _ => panic!("own snapshot"),
    }
}
struct Backend {
    state: Rc<GuiderState>,
    events: Rc<RefCell<Vec<Value>>>,
    names: Names,
    keys: Rc<OwnMappingKeys>,
    failure: Option<usize>,
    calls: Cell<usize>,
}
fn emit(
    events: &Rc<RefCell<Vec<Value>>>,
    failure: Option<usize>,
    event: Value,
) -> Result<(), Error> {
    let mut events = events.borrow_mut();
    events.push(event);
    if failure == Some(events.len() - 1) {
        return Err(Error::ReferenceException {
            class: "ValueError",
            message: "own prediction failure".into(),
        });
    }
    Ok(())
}
impl Backend {
    fn mapping(&self, key: &str, value: Options) -> Options {
        let result = self.keys.empty();
        result.set(key, value).unwrap();
        result
    }
    fn wrap(&self, name: &str, wrapper: Wrapper<Options>) -> Options {
        self.names
            .borrow_mut()
            .insert(Rc::as_ptr(&wrapper) as *const () as usize, name.into());
        Options::Object(Rc::new(wrapper))
    }
}
impl PredictionBindings for Backend {
    fn sampling_function(&self, args: SamplingArguments<'_>) -> Result<Options, Error> {
        let values = [
            args.model,
            args.x,
            args.timestep,
            args.negative,
            args.positive,
            args.cfg,
            args.model_options,
            args.seed,
        ];
        let mut event = vec![json!("sampling")];
        event.extend(values.iter().map(|v| snap(v, &self.names)));
        emit(&self.events, self.failure, json!(event))?;
        let result = token(&format!("sample:{}", self.calls.get()));
        self.calls.set(self.calls.get() + 1);
        Ok(result)
    }
}
struct WrapperBindings(Rc<Backend>);
impl PredictionWrappers for WrapperBindings {
    fn original(&self) -> Result<Original<Options>, Error> {
        let backend = self.0.clone();
        Ok(Rc::new(move |args, kwargs| {
            // Own cases keep the original four explicit arguments. A real
            // method producer must also supply its original default/binder.
            assert!(kwargs.is_empty());
            assert_eq!(args.len(), 4);
            predict_noise(
                &backend.state,
                &PredictionArguments {
                    x: args[0].clone(),
                    timestep: args[1].clone(),
                    model_options: args[2].clone(),
                    seed: args[3].clone(),
                },
                backend.as_ref(),
            )
        }))
    }
    fn canonical_guider(&self) -> Rc<dyn Any> {
        self.0.state.clone()
    }
}
struct Lease(Rc<RefCell<Vec<Value>>>);
impl Drop for Lease {
    fn drop(&mut self) {
        self.0.borrow_mut().push(json!(["lease-drop"]));
    }
}
#[test]
fn original_prediction_preserves_live_state_explicit_call_options_wrapper_short_circuit_repeated_calls_and_removed_wrapper_lifetimes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guider-prediction.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 118);
    for (index, case) in cases.iter().enumerate() {
        let backend = Rc::new(Backend {
            state: Rc::new(GuiderState::default()),
            events: Rc::new(RefCell::new(vec![])),
            names: Rc::new(RefCell::new(BTreeMap::new())),
            keys: OwnMappingKeys::seeded(),
            failure: case["failure"].as_u64().map(|v| v as usize),
            calls: Cell::new(0),
        });
        backend.state.fields.set("inner_model", token("model"));
        backend
            .state
            .fields
            .set("cfg", Options::Atom(V::Float(2.5)));
        let conditions = match case["conditions"].as_str().unwrap() {
            "bad" => none(),
            "empty" => backend.keys.empty(),
            "both" => {
                let values = backend.mapping("negative", token("negative"));
                values.set("positive", token("positive")).unwrap();
                values
            }
            _ => panic!("own conditions"),
        };
        backend.state.fields.set("conds", conditions);
        let plan = case["plan"].as_str().unwrap();
        let mut group = vec![];
        if plan != "none" {
            let plan = plan.to_owned();
            let events = backend.events.clone();
            let keys = backend.keys.clone();
            let failure = backend.failure;
            let first: Wrapper<Options> = Rc::new(move |executor, args, kwargs| {
                let state = executor
                    .class_obj
                    .as_ref()
                    .unwrap()
                    .clone()
                    .downcast::<GuiderState>()
                    .unwrap();
                emit(&events, failure, json!(["wrapper", "first", executor.idx]))?;
                if ["short", "skip-bad"].contains(&plan.as_str()) {
                    return Ok(token("short"));
                }
                if plan == "remove-tail" {
                    let options = state.fields.get("model_options")?;
                    let registry = options
                        .get("transformer_options")?
                        .unwrap()
                        .get("wrappers")?
                        .unwrap()
                        .get("predict_noise")?
                        .unwrap();
                    let Options::List(values) = registry.item(&none())? else {
                        panic!("own registry list")
                    };
                    values.borrow_mut().clear();
                    executor.wrappers.borrow_mut().truncate(1);
                    emit(&events, failure, json!(["cleared-tail"]))?;
                    return Ok(token("short"));
                }
                if plan == "twice" {
                    return Ok(Options::List(Rc::new(RefCell::new(vec![
                        executor.next(args, kwargs)?,
                        executor.next(args, kwargs)?,
                    ]))));
                }
                if plan == "rebind" {
                    state.fields.set("inner_model", token("changed-model"));
                    state.fields.set("cfg", Options::Atom(V::Float(9.)));
                    let conditions = keys.empty();
                    conditions.set("positive", token("changed-positive"))?;
                    state.fields.set("conds", conditions);
                    let options = keys.empty();
                    options.set("marker", Options::Atom(V::Text("changed-state".into())))?;
                    state.fields.set("model_options", options);
                }
                if plan == "change-args" {
                    let options = keys.empty();
                    options.set("marker", Options::Atom(V::Text("forwarded-options".into())))?;
                    return executor.next(
                        &[
                            token("forwarded-x"),
                            token("forwarded-timestep"),
                            options,
                            Options::Atom(V::Int(41)),
                        ],
                        kwargs,
                    );
                }
                executor.next(args, kwargs)
            });
            group.push(backend.wrap("first", first));
        }
        if ["two", "rebind"].contains(&plan) {
            let events = backend.events.clone();
            let failure = backend.failure;
            group.push(backend.wrap(
                "second",
                Rc::new(move |executor, args, kwargs| {
                    emit(&events, failure, json!(["wrapper", "second", executor.idx]))?;
                    executor.next(args, kwargs)
                }),
            ));
        }
        if ["skip-bad", "bad-tail"].contains(&plan) {
            group.push(Options::Atom(V::Int(3)));
        }
        if plan == "remove-tail" {
            let lease = Lease(backend.events.clone());
            group.push(backend.wrap(
                "tail",
                Rc::new(move |executor, args, kwargs| {
                    let _keep = &lease;
                    executor.next(args, kwargs)
                }),
            ));
        }
        let registry = backend.keys.empty();
        registry
            .set_item(none(), Options::List(Rc::new(RefCell::new(group))))
            .unwrap();
        let options = backend.mapping("marker", Options::Atom(V::Text("state-options".into())));
        options
            .set(
                "transformer_options",
                backend.mapping("wrappers", backend.mapping("predict_noise", registry)),
            )
            .unwrap();
        backend.state.fields.set("model_options", options.clone());
        let call_options = if case["call_none"] == true {
            none()
        } else {
            backend.mapping("marker", Options::Atom(V::Text("call-options".into())))
        };
        let arguments = PredictionArguments {
            x: token("x"),
            timestep: token("timestep"),
            model_options: call_options.clone(),
            seed: Options::Atom(V::Int(17)),
        };
        let result = if case["direct"] == true {
            predict_noise(&backend.state, &arguments, backend.as_ref())
        } else {
            outer_predict_noise(&backend.state, arguments, &WrapperBindings(backend.clone()))
        };
        let (result, error) = match result {
            Ok(value) => (snap(&value, &backend.names), Value::Null),
            Err(Error::ReferenceException { class, message }) => {
                (Value::Null, json!([class, message]))
            }
            Err(error) => panic!("native prediction {index}: {error:?}"),
        };
        assert_eq!(
            json!(*backend.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(error, case["error"], "error {index}");
        assert_eq!(result, case["result"], "result {index}");
        for (field, name) in [
            ("inner_model", "model"),
            ("conds", "conds"),
            ("cfg", "cfg"),
            ("model_options", "options"),
        ] {
            assert_eq!(
                snap(&backend.state.fields.get(field).unwrap(), &backend.names),
                case[name],
                "{field} {index}"
            );
        }
        assert_eq!(
            snap(&options, &backend.names),
            case["original_options"],
            "original options {index}"
        );
        assert_eq!(
            snap(&call_options, &backend.names),
            case["call_options"],
            "call options {index}"
        );
    }
}
#[test]
fn copied_wrapper_executor_releases_parent_list_borrow_before_reentrant_parent_mutation() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guider-prediction.json")).unwrap();
    let events = Rc::new(RefCell::new(Vec::<String>::new()));
    type Parent = Rc<RefCell<Option<Weak<WrapperExecutor<Options>>>>>;
    let parent: Parent = Rc::new(RefCell::new(None));
    let original: Original<Options> = {
        let events = events.clone();
        Rc::new(move |_, _| {
            events.borrow_mut().push("original".into());
            Ok(Options::Atom(V::Int(41)))
        })
    };
    let first: Wrapper<Options> = {
        let events = events.clone();
        Rc::new(move |executor, args, kwargs| {
            events.borrow_mut().push("first".into());
            let result = executor.next(args, kwargs)?;
            events.borrow_mut().push("first-exit".into());
            Ok(result)
        })
    };
    let second: Wrapper<Options> = {
        let events = events.clone();
        let parent = parent.clone();
        Rc::new(move |executor, args, kwargs| {
            parent
                .borrow()
                .as_ref()
                .unwrap()
                .upgrade()
                .unwrap()
                .wrappers
                .borrow_mut()
                .clear();
            events.borrow_mut().push("second".into());
            executor.next(args, kwargs)
        })
    };
    let root = Rc::new(WrapperExecutor::new(original, None, &[first, second], 0));
    *parent.borrow_mut() = Some(Rc::downgrade(&root));
    let result = root.execute(&[], &vec![]).unwrap();
    assert_eq!(
        json!(*events.borrow()),
        fixture["parent_mutation"]["events"]
    );
    let Options::Atom(V::Int(result)) = result else {
        panic!("own result")
    };
    assert_eq!(json!(result), fixture["parent_mutation"]["result"]);
    assert_eq!(
        json!(root.wrappers.borrow().len()),
        fixture["parent_mutation"]["remaining"]
    );
}
