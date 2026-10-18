use super::binding_tests::{check, kwargs, reference_error, restore};
use super::*;
use imagine_inference::{execution::Runtime, execution_value::Value as V};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
fn leaves(v: &V) -> Vec<V> {
    match v {
        V::Tensor(_) => vec![v.clone()],
        V::Tuple(v) | V::List(v) => v.iter().flat_map(leaves).collect(),
        _ => vec![],
    }
}
#[test]
fn backend_bindings_execute_original_registered_functions_with_exact_outputs_mutations_and_aliases()
{
    let mut rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()["backend_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        trace.lock().unwrap().clear();
        rt.disable_cutlass_int8 = c["disable_cutlass"].as_bool().unwrap();
        let dev = device(c["device"].as_str().unwrap());
        let a: Vec<_> = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect();
        let k = kwargs(&c["kwargs"], dev);
        let selection = kwargs(&c["selection_kwargs"], dev);
        let meta = selection
            .iter()
            .map(|(key, v)| (key.clone(), v.metadata(&rt.native).unwrap()))
            .collect::<BTreeMap<_, _>>();
        let resolved = rt.backends.resolve(
            c["operation"].as_str().unwrap(),
            c["backend"].as_str(),
            &meta,
        );
        let (handle, out) = match resolved {
            Ok(f) => {
                let out = unsafe { f.call(&rt, &a, &k) };
                (Some(f), out)
            }
            Err(e) => (None, Err(e)),
        };
        if c["error"].is_string() {
            reference_error(
                out.err()
                    .unwrap_or_else(|| panic!("Expected rejection at {i}")),
                c,
            );
        } else {
            let out = out.unwrap_or_else(|e| {
                panic!(
                    "Backend binding {i} {} {}: {e}",
                    c["backend"], c["operation"]
                )
            });
            check(
                &out,
                &c["output"],
                dev,
                &format!("backend binding {i} {} {}", c["backend"], c["operation"]),
            );
            let inputs: Vec<_> = a
                .iter()
                .chain(k.iter().map(|(_, v)| v))
                .flat_map(leaves)
                .collect();
            let outputs = leaves(&out);
            let aliases: Vec<Vec<_>> = outputs
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
            let identities: Vec<Vec<_>> = outputs
                .iter()
                .map(|t| inputs.iter().map(|x| t.same_object(x)).collect())
                .collect();
            assert_eq!(serde_json::json!(aliases), c["aliases"], "alias case {i}");
            assert_eq!(
                serde_json::json!(identities),
                c["identities"],
                "identity case {i}"
            );
            if let Some(caches) = c["natural_caches"].as_object() {
                for (key, expected) in caches {
                    let v = k
                        .iter()
                        .find(|(name, _)| name == key)
                        .map(|(_, v)| v)
                        .unwrap_or_else(|| {
                            &a[c["selection_kwargs"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .position(|p| p[0].as_str() == Some(key.as_str()))
                                .unwrap()]
                        });
                    let cached = v.natural_lora_cache();
                    if expected.is_null() {
                        assert!(cached.is_none(), "Unexpected natural LoRA cache {i}:{key}");
                    } else {
                        check(
                            &V::tensor(cached.unwrap()),
                            expected,
                            dev,
                            &format!("natural LoRA cache {i}"),
                        );
                    }
                }
            }
            if !c["repeat_output"].is_null() {
                let up = k
                    .iter()
                    .find(|(key, _)| key == "lora_up")
                    .unwrap()
                    .1
                    .clone();
                let before = up.natural_lora_cache().unwrap();
                let again = unsafe { handle.as_ref().unwrap().call(&rt, &a, &k) }.unwrap();
                check(
                    &again,
                    &c["repeat_output"],
                    dev,
                    &format!("repeat cached SVDQuant {i}"),
                );
                assert_eq!(
                    before.data_ptr() == up.natural_lora_cache().unwrap().data_ptr(),
                    c["repeat_cache_reused"].as_bool().unwrap()
                );
            }
        }
        for (v, e) in a.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, dev, &format!("post arg {i}"));
        }
        for ((key, v), e) in k.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(key, e[0].as_str().unwrap());
            check(v, &e[1], dev, &format!("post kw {i} {key}"));
        }
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "trace case {i}"
        );
    }
}

#[test]
fn backend_bindings_runtime_chain_and_automatic_selection_never_substitute() {
    let mut rt = Runtime::frozen_partial(kitchen()).unwrap();
    let mut observed_gaps = std::collections::BTreeSet::new();
    for (i, c) in fixtures()["backend_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        rt.disable_cutlass_int8 = c["disable_cutlass"].as_bool().unwrap();
        let dev = device(c["device"].as_str().unwrap());
        let k = kwargs(&c["selection_kwargs"], dev);
        let op = c["operation"].as_str().unwrap();
        let meta = k
            .iter()
            .map(|(name, v)| (name.clone(), v.metadata(&rt.native).unwrap()))
            .collect();
        match rt.backends.resolve(op, None, &meta) {
            Ok(f) => assert_eq!(
                f.backend,
                c["automatic_backend"].as_str().unwrap(),
                "automatic case {i}"
            ),
            Err(imagine_inference::native::Error::BindingGap { backend, operation }) => {
                assert_eq!(
                    backend,
                    c["automatic_backend"].as_str().unwrap(),
                    "automatic gap {i}"
                );
                assert_eq!(operation, op);
                observed_gaps.insert((backend, operation));
            }
            Err(imagine_inference::native::Error::Selection(e)) => {
                assert_eq!(e.exception_class(), c["automatic_error"].as_str().unwrap());
                assert_eq!(e.to_string(), c["automatic_message"].as_str().unwrap());
            }
            Err(e) => panic!("Unexpected selection error {i}: {e}"),
        }
        // A capable explicit backend and scoped override select the same body.
        // Original selection failures are covered by the explicit-call test;
        // their override is allowed to fall back in the reference policy.
        if c["backend_calls"].as_array().unwrap().is_empty() {
            continue;
        }
        let _guard = rt
            .backends
            .policy
            .use_backend(c["backend"].as_str().unwrap())
            .unwrap();
        let out = unsafe { rt.call_kitchen(op, &k) };
        if c["error"].is_string() {
            reference_error(out.err().unwrap(), c);
        } else {
            check(
                &out.unwrap_or_else(|e| panic!("Runtime chain {i} {op}: {e}")),
                &c["output"],
                dev,
                &format!("runtime chain {i}"),
            );
        }
    }
    for c in fixtures()["backend_binding_selections"].as_array().unwrap() {
        assert_eq!(c["kernel_executed"], false);
        let k = kwargs(&c["kwargs"], device(c["device"].as_str().unwrap()));
        let meta = k
            .iter()
            .map(|(name, v)| (name.clone(), v.metadata(&rt.native).unwrap()))
            .collect();
        let Err(imagine_inference::native::Error::BindingGap { backend, operation }) = rt
            .backends
            .resolve(c["operation"].as_str().unwrap(), None, &meta)
        else {
            panic!("Missing body must neither execute nor fall through");
        };
        assert_eq!(backend, c["backend"].as_str().unwrap());
        assert_eq!(operation, c["operation"].as_str().unwrap());
        observed_gaps.insert((backend, operation));
    }
    let gaps = rt.backends.gaps();
    assert_eq!(gaps.len(), 28);
    assert!(
        !observed_gaps.is_empty(),
        "Matrix must exercise original automatic selection of an unbound backend"
    );
    eprintln!(
        "BACKEND_BINDING_GAPS={}",
        serde_json::to_string(&gaps).unwrap()
    );
}
