use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
    shared_context_state_tests::l,
};
use imagine_inference::{
    conditioning::{Condition, Kind},
    context_windows::{self as c, BuiltinConditionAttributes, Handler, WindowStateModel},
    execution::Runtime,
    execution_value::Value as V,
    native::Error,
    patching::options::Options as O,
};
use serde_json::{Value, json};
use std::rc::Rc;
fn constant(value: O) -> O {
    O::Object(Rc::new(Condition::from_options(Kind::Constant, value)))
}
fn list(v: Vec<O>) -> O {
    O::List(l(v))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
struct Model;
impl WindowStateModel for Model {
    fn temporal_downscale_ratio(&self) -> Result<i64, Error> {
        Ok(2)
    }
}
#[test]
fn whole_context_decision_preserves_guide_frame_counts_thresholds_diagnostic_mutations_and_errors()
{
    let _rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-should-use.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1920);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let note = format!("context decision {index} {p}");
        let dev = device(p["device"].as_str().unwrap());
        let x = restore(&case["input"], dev);
        let h = Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat);
        {
            let mut s = h.settings.borrow_mut();
            s.dim = 2;
            s.length = p["length"].as_i64().unwrap();
            s.overlap = p["overlap"].as_i64().unwrap();
            s.condition_retain = l(if p["retains"] == true {
                vec![0, 1]
            } else {
                vec![]
            });
            s.latent_retain = l(if p["retains"] == true {
                vec![2]
            } else {
                vec![]
            });
        }
        let model_conds = dict(vec![]);
        if p["multi"] == true {
            model_conds
                .set(
                    "latent_shapes",
                    constant(list(vec![
                        O::Tuple(Rc::new(
                            vec![1, 1, 12, 1, 2]
                                .into_iter()
                                .map(|n| O::Atom(V::Int(n)))
                                .collect(),
                        )),
                        O::Tuple(Rc::new(
                            vec![1, 1, 6, 1, 1]
                                .into_iter()
                                .map(|n| O::Atom(V::Int(n)))
                                .collect(),
                        )),
                    ])),
                )
                .unwrap();
        }
        if p["guides"] != "none" {
            let count = p["guides"].as_str().unwrap().parse().unwrap();
            model_conds
                .set(
                    "guide_attention_entries",
                    constant(list(vec![dict(vec![
                        (
                            "latent_shape",
                            O::Tuple(Rc::new(
                                vec![count, 1, 1]
                                    .into_iter()
                                    .map(|n| O::Atom(V::Int(n)))
                                    .collect(),
                            )),
                        ),
                        ("pre_filter_count", O::Atom(V::Int(1))),
                    ])])),
                )
                .unwrap();
        }
        let conditions = list(vec![list(vec![dict(vec![("model_conds", model_conds)])])]);
        let mut infos = vec![];
        let result = h.should_use_context(
            &x,
            &conditions,
            &BuiltinConditionAttributes,
            &Model,
            &mut |message| {
                infos.push(message.to_string());
                let mode = p["mode"].as_str().unwrap();
                if mode == "mutate" && infos.len() == 1 {
                    let mut s = h.settings.borrow_mut();
                    s.length += 5;
                    s.overlap += 1;
                    *s.condition_retain.borrow_mut() = vec![3];
                    *s.latent_retain.borrow_mut() = vec![4];
                }
                if mode == "error-first" && infos.len() == 1
                    || mode == "error-second" && infos.len() == 2
                {
                    return Err(Error::ReferenceException {
                        class: "ValueError",
                        message: "own context info failure".into(),
                    });
                }
                Ok(())
            },
        );
        match result {
            Ok(v) => {
                assert!(!case["error"].is_string(), "{note}");
                check(&V::Bool(v), &case["output"], dev, &note);
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
            }
        }
        assert_eq!(json!(infos), case["infos"], "{note}");
        let s = h.settings.borrow();
        assert_eq!(
            json!({"length":s.length,"overlap":s.overlap,"cond_retain":*s.condition_retain.borrow(),"latent_retain":*s.latent_retain.borrow()}),
            case["post"],
            "{note}"
        );
    }
}
