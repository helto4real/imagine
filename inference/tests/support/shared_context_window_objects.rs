use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    context_windows::{IndexWindow, WindowHandle},
    execution_value::Value as V,
    native::{self, Error},
};
use serde_json::Value;
use std::{cell::RefCell, rc::Rc};
use tch::Device;
fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn list(v: Vec<i64>) -> V {
    V::List(v.into_iter().map(V::Int).collect())
}
fn metadata(w: &WindowHandle) -> V {
    let w = w.borrow();
    V::Tuple(vec![
        list(w.indices.borrow().clone()),
        V::Int(w.context_length as i64),
        V::Int(w.overlap),
        V::Int(w.dim),
        V::Int(w.total_frames),
        V::Float(w.center_ratio),
        list(w.guide_frames_indices.borrow().clone()),
        V::List(vec![]),
        list(w.guide_kf_local_positions.borrow().clone()),
        list(w.guide_downscale_factors.borrow().clone()),
    ])
}
fn outcome(v: Result<V, Error>, expected: &Value, dev: Device, note: &str) {
    match v {
        Ok(v) => {
            assert!(
                !expected["error"].is_string(),
                "{note}: expected {expected}"
            );
            check(&v, &expected["output"], dev, note);
        }
        Err(e) => {
            assert!(expected["error"].is_string(), "{note}: unexpected {e}");
            reference_error(e, expected);
        }
    }
}
#[test]
fn original_index_windows_preserve_mutable_containers_tensor_indexing_retention_and_modality_owners()
 {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-context-window-objects.json"
    ))
    .unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1441);
    for (i, c) in cases.iter().enumerate() {
        let p = &c["params"];
        let dev = p["device"].as_str().map(device).unwrap_or(Device::Cpu);
        let note = format!("window object {i} {} {p}", c["name"]);
        match c["name"].as_str().unwrap() {
            "construct" => {
                let values = Rc::new(RefCell::new(ints(&p["indices"])));
                let out = IndexWindow::new(
                    values.clone(),
                    p["dim"].as_i64().unwrap(),
                    p["frames"].as_i64().unwrap(),
                    None,
                    p["overlap"].as_i64().unwrap(),
                )
                .and_then(|w| {
                    assert!(Rc::ptr_eq(&values, &w.borrow().indices));
                    let before = metadata(&w);
                    values.borrow_mut().push(5);
                    let regions = [-3, 0, 1, 3, 10]
                        .into_iter()
                        .map(|n| w.borrow().region_index(n).map(V::Int))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(V::Tuple(vec![before, metadata(&w), V::List(regions)]))
                });
                outcome(out, c, dev, &note);
            }
            "get" | "add" => {
                let full = restore(&c["initial"], dev);
                let w = IndexWindow::new(
                    Rc::new(RefCell::new(vec![0])),
                    p["dim"].as_i64().unwrap(),
                    8,
                    None,
                    0,
                )
                .unwrap();
                *w.borrow().indices.borrow_mut() = ints(&p["indices"]);
                w.borrow_mut().causal_anchor = p["anchor"].as_i64();
                let out = if c["name"] == "get" {
                    w.borrow().get_tensor(
                        &full,
                        p["out_device"].as_str().map(device),
                        p["override"].as_i64(),
                        &ints(&p["retain"]),
                    )
                } else {
                    w.borrow().add_window(
                        &full,
                        &restore(&c["to_add"], dev),
                        p["override"].as_i64(),
                    )
                };
                if let Ok(v) = &out {
                    assert_eq!(
                        v.same_object(&full),
                        c["owners"]["same_object"].as_bool().unwrap(),
                        "{note}"
                    );
                    assert_eq!(
                        native::storage_info(&v.as_tensor().unwrap()).unwrap().0
                            == native::storage_info(&full.as_tensor().unwrap()).unwrap().0,
                        c["owners"]["same_storage"].as_bool().unwrap(),
                        "{note}"
                    );
                }
                outcome(out, c, dev, &note);
                check(&full, &c["post"], dev, &note);
            }
            "modality" => {
                let m = (p["mapping"] != "none").then(|| Rc::new(RefCell::new(vec![])));
                let w = IndexWindow::new(Rc::new(RefCell::new(vec![0, 2])), 0, 8, m.clone(), 0)
                    .unwrap();
                let child =
                    IndexWindow::new(Rc::new(RefCell::new(vec![1])), 1, 8, None, 0).unwrap();
                if p["mapping"] == "child" {
                    m.as_ref().unwrap().borrow_mut().push((1, child.clone()));
                }
                let index = p["index"].as_i64().unwrap();
                let lookup = || {
                    IndexWindow::modality(&w, index).map(|v| {
                        V::Tuple(vec![
                            V::Bool(Rc::ptr_eq(&v, &w)),
                            V::Bool(Rc::ptr_eq(&v, &child)),
                        ])
                    })
                };
                outcome(lookup(), &c["before"], dev, &note);
                if let Some(m) = m {
                    let mut m = m.borrow_mut();
                    if let Some(entry) = m.iter_mut().find(|(k, _)| *k == index) {
                        entry.1 = child.clone();
                    } else {
                        m.push((index, child.clone()));
                    }
                }
                outcome(lookup(), &c["after"], dev, &note);
            }
            "guide-owners" => {
                let w = IndexWindow::new(Rc::new(RefCell::new(vec![0, 1])), 0, 8, None, 0).unwrap();
                let old = w.borrow().guide_frames_indices.clone();
                old.borrow_mut().push(4);
                let first = old.borrow().clone();
                w.borrow_mut().guide_frames_indices = Rc::new(RefCell::new(vec![7]));
                old.borrow_mut().push(9);
                outcome(
                    Ok(V::Tuple(vec![
                        list(first),
                        list(old.borrow().clone()),
                        list(w.borrow().guide_frames_indices.borrow().clone()),
                        V::Bool(Rc::ptr_eq(&old, &w.borrow().guide_frames_indices)),
                    ])),
                    c,
                    dev,
                    &note,
                );
            }
            _ => panic!("{note}"),
        }
    }
}
