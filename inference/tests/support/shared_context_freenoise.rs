use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    conditioning::{Condition, Kind},
    context_windows::{self as c, BuiltinConditionAttributes, Handler},
    execution::Runtime,
    execution_value::Value as V,
    native,
    patching::options::Options as O,
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Device;
fn list(v: Vec<O>) -> O {
    O::List(Rc::new(RefCell::new(v)))
}
fn dict(v: Vec<(&str, O)>) -> O {
    O::dictionary(v.into_iter().map(|(k, v)| (k.into(), v)).collect())
}
fn cond(v: O) -> O {
    O::Object(Rc::new(Condition::from_options(Kind::Constant, v)))
}
#[test]
fn handler_freenoise_preserves_guide_suffixes_multimodal_scaling_aliases_partial_errors_and_private_rng()
 {
    let _rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-freenoise.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1544);
    assert_eq!(fixture["trained_model_weights"], 0);
    for (index, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = device(p["device"].as_str().unwrap());
        let note = format!("handler FreeNoise {index} {p}");
        let noise = restore(&case["initial"], dev);
        let source_shapes = list(
            case["shapes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    let values = v
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| O::Atom(V::Int(v.as_i64().unwrap())))
                        .collect::<Vec<_>>();
                    if p["shape_kind"] == "tuple" {
                        O::Tuple(Rc::new(values))
                    } else {
                        list(values)
                    }
                })
                .collect(),
        );
        let model = dict(vec![]);
        if p["mode"] != "single" {
            model.set("latent_shapes", cond(source_shapes)).unwrap();
        }
        let guides = p["guides"].as_str().unwrap();
        if guides != "none" {
            model
                .set(
                    "guide_attention_entries",
                    cond(list(vec![dict(vec![
                        (
                            "latent_shape",
                            O::Tuple(Rc::new(vec![
                                O::Atom(V::Int(guides.parse().unwrap())),
                                O::Atom(V::Int(1)),
                                O::Atom(V::Int(1)),
                            ])),
                        ),
                        ("pre_filter_count", O::Atom(V::Int(1))),
                    ])])),
                )
                .unwrap();
        }
        let later = if guides == "none" {
            list(vec![])
        } else {
            list(vec![dict(vec![(
                "latent_shape",
                O::Tuple(Rc::new(vec![
                    O::Atom(V::Int(99)),
                    O::Atom(V::Int(1)),
                    O::Atom(V::Int(1)),
                ])),
            )])])
        };
        let conditions = list(vec![
            O::Atom(V::None),
            list(vec![
                dict(vec![]),
                dict(vec![("model_conds", model)]),
                dict(vec![(
                    "model_conds",
                    dict(vec![("guide_attention_entries", cond(later))]),
                )]),
            ]),
        ]);
        let h = Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat);
        {
            let mut s = h.settings.borrow_mut();
            s.dim = 2;
            s.length = p["length"].as_i64().unwrap();
            s.overlap = p["overlap"].as_i64().unwrap();
        }
        let cpu = native::global_rng_state(Device::Cpu).unwrap();
        let gpu = native::global_rng_state(Device::Cuda(0)).unwrap();
        let mut infos = vec![];
        let result = h.apply_freenoise(
            &_rt.native,
            &noise,
            &conditions,
            p["seed"].as_u64().unwrap(),
            &BuiltinConditionAttributes,
            &mut |message| {
                infos.push(message.to_string());
                if p["info_mode"] == "mutate" && infos.len() == 1 {
                    h.settings.borrow_mut().length += 1;
                }
                Ok(())
            },
        );
        assert!(
            cpu.equal(&native::global_rng_state(Device::Cpu).unwrap()),
            "{note}"
        );
        assert!(
            gpu.equal(&native::global_rng_state(Device::Cuda(0)).unwrap()),
            "{note}"
        );
        assert_eq!(case["global_rng_unchanged"], true);
        match result {
            Ok(out) => {
                assert!(!case["error"].is_string(), "{note}");
                check(&out, &case["output"], dev, &note);
                assert_eq!(
                    out.same_object(&noise),
                    case["owners"]["same_object"],
                    "{note}"
                );
                assert_eq!(
                    native::storage_info(&out.as_tensor().unwrap()).unwrap().0
                        == native::storage_info(&noise.as_tensor().unwrap()).unwrap().0,
                    case["owners"]["same_storage"],
                    "{note}"
                );
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
            }
        }
        check(&noise, &case["post"], dev, &note);
        assert_eq!(json!(infos), case["infos"], "{note}");
        assert_eq!(json!(h.settings.borrow().length), case["length"], "{note}");
    }
}
