//! Shared differential runner for source-defined JIT boundaries and wrappers.
use super::{bits, fixtures, kind, kitchen};
use imagine_inference::{autotuning::INT8_CONFIGS, triton_compiler::Compiler};
use std::{path::Path, sync::Arc};
use tch::Device;
pub(super) fn compare(family: &str) {
    use super::binding_tests::{check, kwargs, reference_error, restore};
    use imagine_inference::{autotuning::Int8ArgumentsState, execution::Runtime, native::Error};
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn state(compiler: &Compiler, expected: &serde_json::Value) {
        if expected.is_null() {
            return;
        }
        for (per_row, e) in [false, true].into_iter().zip(expected.as_array().unwrap()) {
            let s = compiler.int8_autotuning_state(per_row);
            assert_eq!(
                match s.arguments {
                    Int8ArgumentsState::Absent => "absent",
                    Int8ArgumentsState::EmptyMapping => "empty",
                    Int8ArgumentsState::None => "none",
                },
                e["arguments"].as_str().unwrap()
            );
            assert_eq!(s.cache_entries as u64, e["cache_entries"].as_u64().unwrap());
            assert_eq!(
                s.best_config.is_some(),
                e["best_config_present"].as_bool().unwrap()
            );
            assert_eq!(
                s.last_timings.is_some(),
                e["timings_present"].as_bool().unwrap()
            );
            assert_eq!(
                s.bench_time.is_some(),
                e["bench_time_present"].as_bool().unwrap()
            );
            if let Some(index) = s.best_config {
                assert!(index < INT8_CONFIGS.len());
            }
            if let Some(timings) = s.last_timings {
                assert_eq!(timings.len() as u64, e["timing_count"].as_u64().unwrap());
                assert!(timings.iter().flatten().all(|x| *x >= 0.));
            }
            if let Some(time) = s.bench_time {
                assert!(time >= 0.);
            }
        }
    }
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native, compiler.clone()).unwrap();
    assert!(
        rt.backends.gaps().is_empty(),
        "Every frozen registered body is connected"
    );
    let operation = match family {
        "int8" => "int8_linear",
        "w4a8" => "w4a8_int8_linear",
        "na" => "na3d",
        _ => panic!("Known original wrapper family"),
    };
    let selected = rt
        .backends
        .resolve(operation, Some("triton"), &BTreeMap::new())
        .unwrap();
    for (field, boundary) in [
        (format!("triton_{family}_boundaries"), true),
        (format!("triton_{family}_bindings"), false),
    ] {
        for (i, c) in fixtures()[&field].as_array().unwrap().iter().enumerate() {
            eprintln!("{family} {field}/{i}: {}", c["label"]);
            state(&compiler, &c["state_before"]);
            let args = c["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, Device::Cuda(0)))
                .collect::<Vec<_>>();
            let kw = kwargs(&c["kwargs"], Device::Cuda(0));
            let seen = Arc::new(AtomicUsize::new(0));
            if boundary {
                let expected = c.clone();
                let inputs = args.clone();
                let seen = seen.clone();
                compiler.set_pre_run_guard(Some(Arc::new(move |op, values, kw| {
                    if op == "rowwise"
                        || (op == "w4a8_unpack" && expected["launch_operation"] != "w4a8_unpack")
                    {
                        return Ok(());
                    }
                    assert_eq!(op, expected["launch_operation"].as_str().unwrap());
                    let check_value = |name: &str,
                                       v: &imagine_inference::execution_value::Value,
                                       e: &serde_json::Value|
                     -> Result<(), Error> {
                        if let Some(meta) = e.get("tensor_metadata") {
                            let t = v.as_tensor()?;
                            assert_eq!(serde_json::json!(t.size()), meta["shape"], "{name}");
                            assert_eq!(serde_json::json!(t.stride()), meta["strides"], "{name}");
                            assert_eq!(t.kind(), kind(meta["dtype"].as_str().unwrap()), "{name}");
                            assert_eq!(t.device(), super::device(meta["device"].as_str().unwrap()));
                            assert_eq!(t.requires_grad(), meta["requires_grad"].as_bool().unwrap());
                            if e["payload"].is_array() {
                                assert_eq!(serde_json::json!(bits(&t)), e["payload"], "{name}");
                            }
                            assert_eq!(
                                serde_json::json!(
                                    inputs.iter().map(|x| v.same_object(x)).collect::<Vec<_>>()
                                ),
                                e["input_identities"],
                                "{name}"
                            );
                            assert_eq!(
                                serde_json::json!(
                                    inputs
                                        .iter()
                                        .map(|x| x
                                            .as_tensor()
                                            .is_ok_and(|x| x.data_ptr() == t.data_ptr()))
                                        .collect::<Vec<_>>()
                                ),
                                e["input_aliases"],
                                "{name}"
                            );
                        } else {
                            check(v, e, Device::Cuda(0), name);
                        }
                        Ok(())
                    };
                    let positional = expected
                        .get("launch")
                        .and_then(serde_json::Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    assert_eq!(values.len(), positional.len());
                    for (i, (v, e)) in values.iter().zip(&positional).enumerate() {
                        check_value(&format!("positional {i}"), v, e)?;
                    }
                    let launch = expected["launch_kwargs"].as_array().unwrap();
                    assert_eq!(kw.len(), launch.len());
                    for ((name, v), e) in kw.iter().zip(launch) {
                        assert_eq!(name, e[0].as_str().unwrap());
                        check_value(name, v, &e[1])?;
                    }
                    seen.fetch_add(1, Ordering::SeqCst);
                    Err(Error::ReferenceException {
                        class: "OwnBoundaryReached",
                        message: expected["message"].as_str().unwrap().into(),
                    })
                })));
            }
            for _ in 0..if c.get("output").is_some() { 2 } else { 1 } {
                let result = unsafe { selected.call(&rt, &args, &kw) };
                if c.get("error").is_some() {
                    reference_error(result.err().expect("Original INT8 failure"), c);
                } else {
                    let out = result.unwrap();
                    check(
                        &out,
                        &c["output"],
                        Device::Cuda(0),
                        c["label"].as_str().unwrap(),
                    );
                    let t = out.as_tensor().unwrap();
                    let inputs = args
                        .iter()
                        .filter(|v| v.as_tensor().is_ok())
                        .collect::<Vec<_>>();
                    assert_eq!(
                        serde_json::json!([inputs
                            .iter()
                            .map(|v| v.as_tensor().unwrap().data_ptr() == t.data_ptr())
                            .collect::<Vec<_>>()]),
                        c["aliases"]
                    );
                    assert_eq!(
                        serde_json::json!([inputs
                            .iter()
                            .map(|v| out.same_object(v))
                            .collect::<Vec<_>>()]),
                        c["identities"]
                    );
                }
            }
            compiler.set_pre_run_guard(None);
            assert_eq!(
                seen.load(Ordering::SeqCst),
                usize::from(boundary && c.get("launch_kwargs").is_some())
            );
            state(&compiler, &c["state_after"]);
            for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
                check(v, e, Device::Cuda(0), "unchanged INT8 input");
            }
            for ((k, v), e) in kw.iter().zip(c["post_kwargs"].as_array().unwrap()) {
                assert_eq!(k, e[0].as_str().unwrap());
                check(v, &e[1], Device::Cuda(0), "unchanged INT8 keyword input");
            }
        }
    }
}
