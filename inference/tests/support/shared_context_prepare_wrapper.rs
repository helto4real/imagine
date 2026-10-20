use super::binding_tests::{check, reference_error, restore};
use imagine_inference::{
    context_windows::{self as c, Handler},
    execution_value::Value as V,
    patching::{
        extensions::{Arguments, Original, WrapperExecutor},
        options::Options as O,
    },
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
use tch::Device;
fn list(v: Vec<O>) -> O {
    O::List(Rc::new(RefCell::new(v)))
}
fn text(v: &str) -> O {
    O::Atom(V::Text(v.into()))
}
#[test]
fn whole_context_preparation_wrapper_preserves_shape_clamps_packed_exceptions_forwarding_aliases_and_argument_errors()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-prepare-wrapper.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1344);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let note = format!("context preparation wrapper {index} {p}");
        let shape = O::from_value(restore(&case["input"], Device::Cpu));
        let model = list(vec![text("model")]);
        let conds = list(vec![text("conds")]);
        let token = list(vec![text("return")]);
        let h = Rc::new(Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat));
        h.settings.borrow_mut().dim = p["dim"].as_i64().unwrap();
        h.settings.borrow_mut().length = p["length"].as_i64().unwrap();
        let options = p["options"].as_str().unwrap();
        let opts = O::dictionary(match options {
            "empty" => vec![],
            "none-handler" => vec![("context_handler".into(), O::Atom(V::None))],
            "bad-handler" => vec![("context_handler".into(), O::Atom(V::Int(3)))],
            _ => vec![("context_handler".into(), O::Object(h))],
        });
        let mut args = vec![model.clone(), shape.clone(), conds.clone()];
        let mut kwargs = vec![
            (
                "model_options".into(),
                match options {
                    "none-options" => O::Atom(V::None),
                    "bad-options" => O::Atom(V::Int(3)),
                    _ => opts.clone(),
                },
            ),
            ("own".into(), text("extra")),
        ];
        if options == "missing" {
            kwargs.remove(0);
        }
        let named = vec![
            ("model".into(), model.clone()),
            ("noise_shape".into(), shape.clone()),
            ("conds".into(), conds.clone()),
        ];
        match p["call"].as_str().unwrap() {
            "keywords" => {
                args.clear();
                kwargs.extend(named);
            }
            "mixed" => {
                args.truncate(1);
                kwargs.extend(named.into_iter().skip(1));
            }
            "extra" => args.extend([text("extra"), O::Atom(V::Int(7))]),
            "duplicate" => kwargs.push(("model".into(), model.clone())),
            "receiver" => kwargs.push(("executor".into(), O::Atom(V::Int(3)))),
            "missing-all" => args.clear(),
            "missing-shape" => {
                args.truncate(1);
                kwargs.push(("conds".into(), conds.clone()));
            }
            "missing-conds" => args.truncate(2),
            _ => {}
        }
        let events = case["events"].as_array().unwrap().clone();
        let calls = Rc::new(Cell::new(0));
        let body: Original<O> = {
            let calls = calls.clone();
            let model = model.clone();
            let conds = conds.clone();
            let shape = shape.clone();
            let opts = opts.clone();
            let token = token.clone();
            let note = note.clone();
            Rc::new(move |args, kwargs: &Arguments<O>| {
                let event = &events[calls.get()];
                calls.set(calls.get() + 1);
                check(&args[1].snapshot()?, &event["shape"], Device::Cpu, &note);
                // Native primitive values have no exposed host-language object
                // identity. Keep that source observation in the fixture; only
                // the independently represented container aliases are checked.
                if !matches!(shape, O::Atom(_)) {
                    assert_eq!(args[1].same_object(&shape), event["same_shape"], "{note}");
                }
                assert_eq!(args[0].same_object(&model), event["same_model"], "{note}");
                assert_eq!(args[2].same_object(&conds), event["same_conds"], "{note}");
                check(
                    &V::Tuple(
                        args[3..]
                            .iter()
                            .map(O::snapshot)
                            .collect::<Result<_, _>>()?,
                    ),
                    &event["extra"],
                    Device::Cpu,
                    &note,
                );
                assert_eq!(
                    json!(kwargs.iter().map(|(k, _)| k).collect::<Vec<_>>()),
                    event["keys"],
                    "{note}"
                );
                assert_eq!(
                    kwargs
                        .iter()
                        .find(|(k, _)| k == "model_options")
                        .is_some_and(|(_, v)| v.same_object(&opts)),
                    event["same_options"],
                    "{note}"
                );
                assert!(
                    matches!(kwargs.iter().find(|(k,_)|k=="own").unwrap().1.snapshot()?,V::Text(v)if v==event["own"].as_str().unwrap())
                );
                Ok(token.clone())
            })
        };
        let result = WrapperExecutor::new(body, None, &[c::prepare_sampling_wrapper()], 0)
            .execute(&args, &kwargs);
        match result {
            Ok(v) => {
                assert_eq!(case["same_return"], true, "{note}");
                assert!(v.same_object(&token), "{note}");
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note} {e}");
                reference_error(e, case);
            }
        }
        assert_eq!(
            calls.get(),
            case["events"].as_array().unwrap().len(),
            "{note}"
        );
        check(
            &shape.snapshot().unwrap(),
            &case["post"],
            Device::Cpu,
            &note,
        );
    }
}
