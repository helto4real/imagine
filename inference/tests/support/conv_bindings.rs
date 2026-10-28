use super::binding_tests::{check, reference_error, restore as restore_plain};
use super::*;
use imagine_inference::{
    execution::Runtime,
    execution_value::{Kwargs, Value as V},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn restore(v: &Value, dev: Device) -> V {
    if v["type"] == "tensor" && !v["storage"].is_null() {
        let base = restore_plain(&v["storage"], dev).as_tensor().unwrap();
        V::tensor(base.as_strided(
            ints(&v["shape"]),
            ints(&v["strides"]),
            Some(v["storage_offset"].as_i64().unwrap()),
        ))
    } else {
        restore_plain(v, dev)
    }
}
fn leaves(v: &V) -> Vec<V> {
    match v {
        V::Tensor(_) => vec![v.clone()],
        V::Tuple(v) => v.iter().flat_map(leaves).collect(),
        _ => vec![],
    }
}
fn arguments(c: &Value, dev: Device) -> (Vec<V>, Kwargs, Option<V>) {
    let base = (!c["shared"].is_null()).then(|| restore(&c["shared"]["base"], dev));
    // Arguments with the same view of the base are one object, as `out=x` is.
    let made = std::cell::RefCell::new(Vec::<(Value, V)>::new());
    let make = |key: &str, v: &Value| {
        let view = &c["shared"]["views"][key];
        if let Some((_, same)) = made
            .borrow()
            .iter()
            .find(|(w, _)| !view.is_null() && w == view)
        {
            return same.clone();
        }
        let value = if view.is_null() {
            restore(v, dev)
        } else {
            let raw = base
                .as_ref()
                .unwrap()
                .as_tensor()
                .unwrap()
                .view_dtype(kind(view["dtype"].as_str().unwrap()));
            let t = raw.as_strided(
                ints(&view["shape"]),
                ints(&view["strides"]),
                Some(view["offset"].as_i64().unwrap()),
            );
            V::tensor(if v["requires_grad"].as_bool().unwrap_or(false) {
                t.set_requires_grad(true)
            } else {
                t
            })
        };
        if !view.is_null() {
            made.borrow_mut().push((view.clone(), value.clone()));
        }
        value
    };
    let k = c["kwargs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let key = p[0].as_str().unwrap();
            (key.into(), make(key, &p[1]))
        })
        .collect();
    let a = c["args"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, v)| make(c["selection_kwargs"][i][0].as_str().unwrap(), v))
        .collect();
    (a, k, base)
}
fn run_category(key: &str) {
    let mut rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()[key].as_array().unwrap().iter().enumerate() {
        trace.lock().unwrap().clear();
        let launches = Arc::new(Mutex::new(Vec::<String>::new()));
        let captured = launches.clone();
        rt.native = rt
            .native
            .clone()
            .with_launch_guard(Some(Arc::new(move |name: &str| {
                captured.lock().unwrap().push(name.into());
                Ok(())
            })));
        let dev = device(c["device"].as_str().unwrap());
        let (a, k, base) = arguments(c, dev);
        let names = c["selection_kwargs"].as_array().unwrap();
        let selection: Kwargs = if a.is_empty() {
            k.clone()
        } else {
            a.iter()
                .zip(names)
                .map(|(v, p)| (p[0].as_str().unwrap().into(), v.clone()))
                .collect()
        };
        let meta = selection
            .iter()
            .map(|(key, v)| (key.clone(), v.metadata(&rt.native).unwrap()))
            .collect::<BTreeMap<_, _>>();
        let out = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                c["backend"].as_str(),
                &meta,
            )
            .and_then(|f| unsafe { f.call(&rt, &a, &k) });
        if c["error"].is_string() {
            reference_error(
                out.err()
                    .unwrap_or_else(|| panic!("Expected original rejection {key}:{i}")),
                c,
            );
        } else {
            let out = out.unwrap_or_else(|e| {
                panic!(
                    "Conv/norm {key}:{i} {} {} {}: {e}",
                    c["backend"], c["operation"], c["label"]
                )
            });
            check(&out, &c["output"], dev, &format!("Conv/norm {key}:{i}"));
            let inputs: Vec<_> = a
                .iter()
                .chain(k.iter().map(|(_, v)| v))
                .flat_map(leaves)
                .collect();
            let output = leaves(&out);
            let aliases: Vec<Vec<_>> = output
                .iter()
                .map(|t| {
                    inputs
                        .iter()
                        .map(|x| {
                            t.as_tensor().unwrap().data_ptr() == x.as_tensor().unwrap().data_ptr()
                        })
                        .collect()
                })
                .collect();
            let ids: Vec<Vec<_>> = output
                .iter()
                .map(|t| inputs.iter().map(|x| t.same_object(x)).collect())
                .collect();
            assert_eq!(
                serde_json::json!(aliases),
                c["aliases"],
                "Conv/norm alias {i}"
            );
            assert_eq!(
                serde_json::json!(ids),
                c["identities"],
                "Conv/norm identity {i}"
            );
        }
        for (v, e) in a.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, dev, &format!("Conv/norm post args {i}"));
        }
        for ((name, v), e) in k.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(name, e[0].as_str().unwrap());
            check(v, &e[1], dev, &format!("Conv/norm post kwargs {i}:{name}"));
        }
        if let Some(base) = &base {
            check(
                base,
                &c["post_base"],
                dev,
                &format!("Conv/norm entire shared allocation {i}"),
            );
        }
        assert_eq!(
            serde_json::json!(*launches.lock().unwrap()),
            c["native_calls"],
            "original native route {key}:{i}"
        );
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "Conv/norm backend trace {i}"
        );
    }
}
#[test]
fn conv_bindings_preserve_native_fallback_layout_outbuffer_and_errors() {
    run_category("conv_bindings");
}
#[test]
fn group_norm_bindings_preserve_native_fallback_padding_affine_outbuffer_and_errors() {
    run_category("group_norm_bindings");
}
