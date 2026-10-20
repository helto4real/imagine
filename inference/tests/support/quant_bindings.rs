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
fn run_category(category: &str) {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()[category].as_array().unwrap().iter().enumerate() {
        trace.lock().unwrap().clear();
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
        let out = resolved.and_then(|f| unsafe { f.call(&rt, &a, &k) });
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
            for input in &inputs {
                assert!(
                    input.natural_lora_cache().is_none(),
                    "eager must not create CUDA LoRA cache"
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
        let actual = if c["direct"] == true {
            k.clone()
        } else {
            selection.clone()
        };
        let metadata = actual
            .iter()
            .map(|(key, v)| (key.clone(), v.metadata(&rt.native).unwrap()))
            .collect();
        match rt
            .backends
            .resolve(c["operation"].as_str().unwrap(), None, &metadata)
        {
            Ok(f) => assert_eq!(f.backend, c["automatic_backend"].as_str().unwrap()),
            Err(imagine_inference::native::Error::BindingGap { backend, .. }) => {
                assert_eq!(backend, c["automatic_backend"].as_str().unwrap())
            }
            Err(e) => reference_error(
                e,
                &serde_json::json!({"error":c["automatic_error"],"message":c["automatic_message"]}),
            ),
        }
    }
}

#[test]
fn eager_quant_bindings_preserve_fp4_bits_svdquant_layout_lora_padding_and_errors() {
    run_category("quant_backend_bindings");
}
#[test]
fn scaled_mm_bindings_preserve_original_operator_hardware_rejections_and_no_substitution() {
    run_category("scaled_mm_backend_bindings");
}

#[test]
fn eager_convrot_w4a4_linear_preserves_original_rotation_int4_math_and_validation() {
    run_category("w4a4_linear_bindings");
}

#[test]
fn na3d_bindings_preserve_original_tiling_mask_sdpa_and_fused_cuda_paths() {
    // The retained original corpus was captured with one CPU thread. This
    // matters for the unchanged large CPU SDPA tile; other synthetic model
    // tests may have initialized OpenMP with a different count beforehand.
    struct Threads(i32);
    impl Drop for Threads {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _threads = Threads(tch::get_num_threads());
    let reference: Value = serde_json::from_str(include_str!(
        "../../fixtures/kitchen-0238-na3d-threads.json"
    ))
    .unwrap();
    assert!(
        reference["cases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["threads"] == 1 && c["baseline_exact"] == true)
    );
    tch::set_num_threads(1);
    run_category("na3d_backend_bindings");
}

#[test]
fn w4a8_bindings_preserve_original_grid_decode_rotation_correction_and_linear_paths() {
    run_category("w4a8_backend_bindings");
}
#[test]
fn cuda_sol_bindings_preserve_original_threshold_workspace_coarse_and_native_paths() {
    run_category("sol_cuda_bindings");
}
