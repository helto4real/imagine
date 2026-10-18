use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution::Runtime,
    execution_value::{Kwargs, Value as V},
    native::Error,
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
fn leaves(v: &V) -> Vec<V> {
    match v {
        V::Tensor(_) => vec![v.clone()],
        V::Tuple(v) => v.iter().flat_map(leaves).collect(),
        _ => vec![],
    }
}
fn arguments(c: &Value, dev: Device) -> (Vec<V>, Kwargs, Option<V>) {
    let base = (!c["shared"].is_null()).then(|| restore(&c["shared"]["base"], dev));
    let make = |key: &str, v: &Value| {
        let view = &c["shared"]["views"][key];
        if view.is_null() {
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
        }
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
        assert_eq!(c["native_guard_entered"], false);
        let guarded = c["guard_required"].as_bool().unwrap();
        rt.native = rt.native.clone().with_launch_guard(guarded.then(|| {
            Arc::new(|name: &str| {
                Err(Error::Native(format!(
                    "Unexpected native launch after expected host rejection: {name}"
                )))
            }) as imagine_inference::native::LaunchGuard
        }));
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
        if let Some(base) = &base {
            let root = base.as_tensor().unwrap();
            let base_address = root.data_ptr() as u64;
            for (name, expected) in c["shared"]["storage"].as_object().unwrap() {
                let t = selection
                    .iter()
                    .find(|(key, _)| key == name)
                    .unwrap()
                    .1
                    .as_tensor()
                    .unwrap();
                let info = rt.native.tensor_storage(&t).unwrap();
                assert_eq!(
                    info.base == base_address,
                    expected["same_base"].as_bool().unwrap()
                );
                assert_eq!(info.offset, expected["offset"].as_i64().unwrap());
                assert_eq!(
                    info.element_bytes,
                    expected["element_bytes"].as_i64().unwrap()
                );
            }
        }
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
                    "RoPE {key}:{i} {} {} {}: {e}",
                    c["backend"], c["operation"], c["label"]
                )
            });
            check(&out, &c["output"], dev, &format!("RoPE {key}:{i}"));
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
            assert_eq!(serde_json::json!(aliases), c["aliases"], "RoPE alias {i}");
            assert_eq!(serde_json::json!(ids), c["identities"], "RoPE identity {i}");
        }
        for (v, e) in a.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, dev, &format!("RoPE post args {i}"));
        }
        for ((name, v), e) in k.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(name, e[0].as_str().unwrap());
            check(v, &e[1], dev, &format!("RoPE post kwargs {i}:{name}"));
        }
        if let Some(base) = &base {
            check(
                base,
                &c["post_base"],
                dev,
                &format!("RoPE entire shared allocation {i}"),
            );
        }
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "RoPE backend trace {i}"
        );
    }
}
#[test]
fn rope_bindings_preserve_original_eager_cuda_math_inplace_grad_flags_and_backend_errors() {
    run_category("rope_bindings");
}
#[test]
fn rope_bindings_preserve_shared_storage_bounds_qkv_disjointness_and_prelaunch_rejections() {
    run_category("rope_shared_bindings");
}
