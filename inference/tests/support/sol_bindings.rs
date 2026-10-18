use super::binding_tests::{check, kwargs, restore};
use super::*;
use imagine_inference::{execution::Runtime, execution_value::Value as V};
use std::sync::{Arc, Mutex};

#[test]
fn original_eager_sol_attn_approved_scatter_and_exact_downstream() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let cases = fixtures()["sol_eager_bindings"].as_array().unwrap().clone();
    assert_eq!(cases.len(), 6);
    for (i, c) in cases.into_iter().enumerate() {
        let dev = Device::Cuda(0);
        let k = kwargs(&c["kwargs"], dev);
        let controlled = c["controlled"].as_bool().unwrap();
        assert_eq!(controlled, i != 0);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let events = observed.clone();
        let guard = c["guards"][0].clone();
        let canonical = c["canonical"].clone();
        rt.observe_sol_scatter(Some(Arc::new(
            move |before, dim, out, index, src, stream| {
                assert_eq!(dim, 2);
                assert_eq!(stream, 0);
                assert_eq!(out.device(), Device::Cuda(0));
                let mut events = events.lock().unwrap();
                if before {
                    assert!(events.is_empty());
                    check(
                        &V::tensor(out.shallow_clone()),
                        &guard["pre"],
                        dev,
                        "scatter pre",
                    );
                    check(
                        &V::tensor(index.shallow_clone()),
                        &guard["index"],
                        dev,
                        "scatter index",
                    );
                    check(
                        &V::tensor(src.shallow_clone()),
                        &guard["src"],
                        dev,
                        "scatter src",
                    );
                } else {
                    assert_eq!(*events, vec![true]);
                    assert_eq!(
                        out.size(),
                        canonical["shape"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_i64().unwrap())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        out.stride(),
                        canonical["strides"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_i64().unwrap())
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(out.kind(), Kind::Float);
                    assert!(out.isfinite().all().int64_value(&[]) != 0);
                    assert!(
                        out.f_ne(0)?.f_any()?.int64_value(&[]) != 0,
                        "original scatter must run before replacement"
                    );
                    if controlled {
                        out.f_copy_(&restore(&canonical, dev).as_tensor()?)?;
                        check(
                            &V::tensor(out.shallow_clone()),
                            &canonical,
                            dev,
                            "canonical post scatter",
                        );
                    }
                }
                events.push(before);
                Ok(())
            },
        )));
        let meta = k
            .iter()
            .map(|(key, v)| (key.clone(), v.metadata(&rt.native).unwrap()))
            .collect();
        let resolved = rt
            .backends
            .resolve("sol_attn", Some("eager"), &meta)
            .unwrap();
        assert_eq!(
            rt.backends
                .resolve("sol_attn", None, &meta)
                .unwrap()
                .backend,
            "eager"
        );
        let out = unsafe { resolved.call(&rt, &[], &k) }.unwrap();
        check(&out, &c["output"], dev, &format!("Sol case {i}"));
        assert_eq!(*observed.lock().unwrap(), vec![true, false]);
        for ((key, v), expected) in k.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(key, expected[0].as_str().unwrap());
            check(v, &expected[1], dev, "unchanged Sol inputs");
            if let V::Tensor(_) = v {
                assert_ne!(
                    out.as_tensor().unwrap().data_ptr(),
                    v.as_tensor().unwrap().data_ptr()
                );
            }
        }
        rt.observe_sol_scatter(None);
        if !controlled {
            let ordinary = unsafe { resolved.call(&rt, &[], &k) }.unwrap();
            check(
                &ordinary,
                &c["output"],
                dev,
                "ordinary observer-free Sol control",
            );
        }
    }
}

#[test]
fn original_eager_sol_attn_ordinary_block_lengths_masks_dtype_and_coarse_branches() {
    use super::binding_tests::reference_error;
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let data = fixtures();
    let cases = data["sol_eager_ordinary_bindings"].as_array().unwrap();
    assert_eq!(cases.len(), 45);
    for (i, c) in cases.iter().enumerate() {
        let k: Vec<(String, V)> = c["kwargs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                let dev = entry[1]["device"]
                    .as_str()
                    .map(device)
                    .unwrap_or(Device::Cpu);
                (entry[0].as_str().unwrap().into(), restore(&entry[1], dev))
            })
            .collect::<Vec<_>>();
        let meta = k
            .iter()
            .map(|(n, v)| (n.clone(), v.metadata(&rt.native).unwrap()))
            .collect();
        let direct = c["direct"] == true;
        let empty = std::collections::BTreeMap::new();
        let f = rt
            .backends
            .resolve(
                "sol_attn",
                Some("eager"),
                if direct { &empty } else { &meta },
            )
            .unwrap();
        let out = unsafe { f.call(&rt, &[], &k) };
        let out = if c["error"].is_string() {
            reference_error(out.err().unwrap(), c);
            None
        } else {
            let out = out.unwrap();
            check(
                &out,
                &c["output"],
                Device::Cpu,
                &format!("ordinary Sol {i}"),
            );
            Some(out)
        };
        for ((n, v), expected) in k.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(n, expected[0].as_str().unwrap());
            let dev = expected[1]["device"]
                .as_str()
                .map(device)
                .unwrap_or(Device::Cpu);
            check(v, &expected[1], dev, "ordinary Sol unchanged input");
            if let (Some(out), V::Tensor(_)) = (&out, v) {
                assert_ne!(
                    out.as_tensor().unwrap().data_ptr(),
                    v.as_tensor().unwrap().data_ptr()
                );
            }
        }
        if direct {
            reference_error(
                rt.backends
                    .resolve("sol_attn", Some("eager"), &meta)
                    .err()
                    .unwrap(),
                &serde_json::json!({"error":c["validated_error"],"message":c["validated_message"]}),
            );
        } else {
            assert_eq!(
                rt.backends
                    .resolve("sol_attn", None, &meta)
                    .unwrap()
                    .backend,
                c["automatic_backend"].as_str().unwrap()
            );
        }
        assert_eq!(c["controlled"], false);
        assert_eq!(c["observer_installed"], false);
        assert_eq!(c["canonical_copy_applied"], false);
    }
}
