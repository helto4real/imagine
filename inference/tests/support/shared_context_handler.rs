use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
    shared_context_state_tests::{data, l, snapshot},
};
use imagine_inference::{
    context_windows::{self as c, CombineFrame, Handler, IndexWindow},
    execution::Runtime,
    execution_value::Value as V,
    native::{self, Error},
};
use serde_json::{Value, json};
use std::rc::Rc;
use tch::{Device, Kind, Tensor};
fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
struct Dtype(imagine_inference::native::Kitchen, Kind);
impl Drop for Dtype {
    fn drop(&mut self) {
        self.0.set_default_dtype(self.1).unwrap();
    }
}
#[test]
fn original_context_handler_step_windows_and_accumulation_preserve_callbacks_partial_state_and_weight_owners()
 {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    let _default = Dtype(rt.native.clone(), rt.native.default_dtype().unwrap());
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-context-handler.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4458);
    for (i, case) in cases.iter().enumerate() {
        let p = &case["params"];
        let dev = p["device"].as_str().map(device).unwrap_or(Device::Cpu);
        rt.native
            .set_default_dtype(p["default"].as_str().map(kind).unwrap_or(Kind::Float))
            .unwrap();
        let note = format!("context handler {i} {} {p}", case["name"]);
        let out: Result<V, Error> = match case["name"].as_str().unwrap() {
            "step" => {
                let h = Handler::new(c::Schedule::StaticStandard, c::Fuse::Flat);
                h.step.set(p["initial"].as_u64().unwrap());
                let r = h.set_step(&restore(&case["time"], dev), &restore(&case["sigmas"], dev));
                assert_eq!(json!(h.step.get()), case["post"], "{note}");
                r.map(|_| V::None)
            }
            "windows" => {
                let h = Handler::new(
                    c::Schedule::from_name(p["schedule"].as_str().unwrap()).unwrap(),
                    c::Fuse::Flat,
                );
                {
                    let mut s = h.settings.borrow_mut();
                    s.length = p["length"].as_i64().unwrap();
                    s.overlap = p["overlap"].as_i64().unwrap();
                    s.stride = 3;
                    s.dim = p["dim"].as_i64().unwrap();
                }
                h.step.set(p["step"].as_u64().unwrap());
                let mut warnings = vec![];
                let r = h.get_context_windows(&restore(&case["input"], dev), &mut |message| {
                    warnings.push(json!({"category":"RuntimeWarning","message":message}));
                    Ok(())
                });
                assert_eq!(json!(warnings), case["warnings"], "{note}");
                r.map(|v| V::List(v.iter().map(snapshot).collect()))
            }
            "combine" => {
                let dtype = kind(p["dtype"].as_str().unwrap());
                let dim = p["dim"].as_i64().unwrap();
                let fuse = c::Fuse::from_name(p["fuse"].as_str().unwrap()).unwrap();
                let h = Handler::new(c::Schedule::StaticStandard, fuse);
                {
                    let mut s = h.settings.borrow_mut();
                    s.length = 4;
                    s.overlap = 2;
                    s.dim = dim;
                }
                let indices = ints(&p["indices"]);
                let w = IndexWindow::new(
                    l(if indices.is_empty() {
                        vec![0]
                    } else {
                        indices.clone()
                    }),
                    dim,
                    8,
                    None,
                    2,
                )
                .unwrap();
                if indices.is_empty() {
                    w.borrow().indices.borrow_mut().clear();
                }
                let x = restore(&case["input"], dev);
                let full = x.as_tensor().unwrap();
                let mut out_shape = full.size();
                out_shape[dim.max(0) as usize] = indices.len() as i64;
                let outputs = l((0..2)
                    .map(|i| data(&out_shape, dev, dtype, i * 5 + 2))
                    .collect());
                let accum = l((0..2)
                    .map(|_| V::tensor(full.f_zeros_like().unwrap()))
                    .collect());
                let count_shape = c::shape_for_dim(&full, dim).unwrap();
                let default = rt.native.default_dtype().unwrap();
                let counts = l((0..2)
                    .map(|_| {
                        V::tensor(if matches!(fuse, c::Fuse::Relative) {
                            Tensor::ones(&count_shape[..], (default, dev))
                        } else {
                            Tensor::zeros(&count_shape[..], (default, dev))
                        })
                    })
                    .collect());
                let biases = l((0..2)
                    .map(|_| {
                        l(vec![
                            0.;
                            if dim < 0 {
                                full.size()[full.dim() - 1]
                            } else {
                                full.size()[dim as usize]
                            } as usize
                        ])
                    })
                    .collect());
                let frame = CombineFrame {
                    input: &x,
                    outputs: &outputs,
                    window: &w,
                    accum: &accum,
                    counts: &counts,
                    biases: &biases,
                };
                let mut calls = 0;
                let r = h.combine(&rt, &frame, &mut |handler, frame, locals| {
                    let event = &case["events"][calls];
                    calls += 1;
                    assert_eq!(event["same_accum"], Rc::ptr_eq(frame.accum, &accum));
                    assert_eq!(event["same_counts"], Rc::ptr_eq(frame.counts, &counts));
                    assert_eq!(event["same_biases"], Rc::ptr_eq(frame.biases, &biases));
                    assert_eq!(event["same_outputs"], Rc::ptr_eq(frame.outputs, &outputs));
                    let expected = event["locals"].as_object().unwrap();
                    assert_eq!(locals.tensors().count(), expected.len(), "{note}");
                    for (name, tensor) in locals.tensors() {
                        check(
                            &V::tensor(tensor.shallow_clone()),
                            &expected[name],
                            dev,
                            &format!("{note} local {name}"),
                        );
                    }
                    match p["mode"].as_str().unwrap() {
                        "mutate" => {
                            let mut t = accum.borrow()[0].as_tensor()?;
                            let _ = t.f_add_scalar_(0.25)?;
                            biases.borrow()[0].borrow_mut()[0] += 0.5;
                            let copy = native::clone_tensor(&counts.borrow()[0].as_tensor()?)?;
                            counts.borrow_mut()[0] = V::tensor(copy);
                            outputs.borrow_mut().pop();
                            handler.settings.borrow_mut().length += 1;
                            Ok(())
                        }
                        "error" => Err(Error::ReferenceException {
                            class: "ValueError",
                            message: "own combine callback failure".into(),
                        }),
                        _ => Ok(()),
                    }
                });
                assert_eq!(calls, case["events"].as_array().unwrap().len(), "{note}");
                let post = V::Tuple(vec![
                    V::List(accum.borrow().clone()),
                    V::List(counts.borrow().clone()),
                    V::List(
                        biases
                            .borrow()
                            .iter()
                            .map(|v| V::List(v.borrow().iter().map(|v| V::Float(*v)).collect()))
                            .collect(),
                    ),
                    V::List(outputs.borrow().clone()),
                    V::Int(h.settings.borrow().length),
                ]);
                check(&post, &case["post"], dev, &note);
                r.map(|_| V::None)
            }
            _ => panic!("{note}"),
        };
        match out {
            Ok(v) => {
                assert!(!case["error"].is_string(), "{note}: expected {case}");
                check(&v, &case["output"], dev, &note);
            }
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: unexpected {e}");
                reference_error(e, case);
            }
        }
    }
}
