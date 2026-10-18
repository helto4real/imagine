use super::binding_tests::{check, kwargs, reference_error, restore};
use super::*;
use imagine_inference::{execution::Runtime, execution_value::Value as V};
use std::sync::{Arc, Mutex};

#[test]
fn layout_preparation_requantization_export_decode_and_dispatch_match_original() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()["layout_completion"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        trace.lock().unwrap().clear();
        let dev = device(c["device"].as_str().unwrap());
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect::<Vec<_>>();
        let kwargs = kwargs(&c["kwargs"], dev);
        let guard = c["override"]
            .as_str()
            .map(|name| rt.backends.policy.use_backend(name).unwrap());
        let previous_default = rt.native.default_dtype().unwrap();
        if let Some(default) = c["default_dtype"].as_str() {
            rt.native.set_default_dtype(kind(default)).unwrap();
        }
        let action = c["action"].as_str().unwrap();
        let out = (|| -> Result<V, imagine_inference::native::Error> {
            Ok(match action {
                "set_default_dtype" => {
                    let V::DType(kind) = args[0] else {
                        unreachable!()
                    };
                    rt.native.set_default_dtype(kind)?;
                    V::DType(rt.native.default_dtype()?)
                }
                "from_float" => V::quantized(unsafe {
                    rt.from_float(
                        &args[0].as_tensor()?,
                        c["layout"].as_str().unwrap(),
                        &kwargs,
                    )?
                }),
                "requantize" => V::quantized(unsafe {
                    rt.requantize_from_float(
                        &args[0].as_quantized()?,
                        &args[1].as_tensor()?,
                        &kwargs,
                    )?
                }),
                "requantize_kwargs" => V::Dict(rt.requantize_kwargs(&args[0].as_quantized()?)?),
                "decode" => V::tensor(unsafe { rt.decode(&args[0].as_quantized()?)? }),
                "state_dict" => V::Dict(
                    rt.state_dict(&args[0].as_quantized()?, "layer.weight")?
                        .into_iter()
                        .map(|(k, t)| (k, V::tensor(t)))
                        .collect(),
                ),
                "op" => unsafe { rt.dispatch(c["operation"].as_str().unwrap(), &args, &kwargs)? },
                _ => panic!("Unknown own layout fixture action"),
            })
        })();
        let post_default = rt.native.default_dtype().unwrap();
        rt.native.set_default_dtype(previous_default).unwrap();
        if let Some(expected) = c["post_default_dtype"].as_str() {
            assert_eq!(post_default, kind(expected), "global dtype after case {i}");
        }
        drop(guard);
        let note = format!("layout {i} {} {} {}", c["layout"], action, c["label"]);
        if c["error"].is_string() {
            match out {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected original rejection: {note}"),
            }
        } else {
            let out = out.unwrap_or_else(|e| panic!("{note}: {e}"));
            check(&out, &c["output"], dev, &note);
            if let Some(same) = c["same_object"].as_bool() {
                assert_eq!(out.same_object(&args[0]), same, "{note} object identity");
                let output = out.as_quantized().unwrap();
                let input = args[0].as_quantized().unwrap();
                assert_eq!(
                    output.payload.data_ptr() == input.payload.data_ptr(),
                    c["payload_alias"].as_bool().unwrap(),
                    "{note} payload alias"
                );
                for (name, expected) in c["parameter_aliases"].as_object().unwrap() {
                    assert_eq!(
                        output.params.tensors[name].data_ptr()
                            == input.params.tensors[name].data_ptr(),
                        expected.as_bool().unwrap(),
                        "{note} {name} alias"
                    );
                }
            }
        }
        for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, dev, &format!("{note} post argument"));
        }
        for ((n, v), e) in kwargs.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(n, e[0].as_str().unwrap());
            check(v, &e[1], dev, &format!("{note} post keyword"));
        }
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "{note} backend trace"
        );
    }
}

#[test]
fn approved_raw_cuda_export_preserves_plain_storage_and_rejects_nonempty_wrappers() {
    use imagine_inference::{layouts, native::Error};
    let _native = kitchen();
    for dev in [Device::Cpu, Device::Cuda(0)] {
        let plain = Tensor::arange(6, (Kind::Half, dev)).reshape([2, 3]);
        let value = V::tensor(plain.shallow_clone());
        let exported = value.as_cuda_export().unwrap();
        assert_eq!(exported.data_ptr(), plain.data_ptr());
        assert_eq!(exported.size(), plain.size());
        assert_eq!(exported.stride(), plain.stride());
        for shape in [vec![2, 3], vec![]] {
            let q = layouts::int8_parts(
                Tensor::zeros(shape.as_slice(), (Kind::Int8, dev)),
                Tensor::from(1.0f32).to_device(dev),
                Kind::Half,
                shape,
                false,
                false,
                256,
            );
            match V::quantized(q).as_cuda_export() {
                Err(Error::ReferenceException { class, message }) => {
                    assert_eq!(class, "TypeError");
                    assert_eq!(
                        message,
                        "Cannot export a non-empty QuantizedTensor with null DLPack data to a CUDA kernel."
                    );
                }
                _ => panic!("Nonempty wrappers must stop before the raw CUDA boundary"),
            }
        }
        let q = layouts::int8_parts(
            Tensor::empty([2, 0], (Kind::Int8, dev)),
            Tensor::from(1.0f32).to_device(dev),
            Kind::Half,
            vec![2, 0],
            false,
            false,
            256,
        );
        let empty = V::quantized(q).as_cuda_export().unwrap();
        assert_eq!(empty.size(), [2, 0]);
        assert_eq!(empty.kind(), Kind::Half);
        assert_eq!(empty.device(), dev);
        assert_eq!(empty.numel(), 0);
    }
}

#[test]
fn svd_layout_shared_quantization_grouping_and_fusion_match_original() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()["svd_layout_grouping"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        trace.lock().unwrap().clear();
        let dev = device(c["device"].as_str().unwrap());
        let weights = c["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| restore(w, dev))
            .collect::<Vec<_>>();
        for share in c["shared_fields"].as_array().unwrap() {
            let i = share[0].as_u64().unwrap() as usize;
            let name = share[1].as_str().unwrap();
            let j = share[2].as_u64().unwrap() as usize;
            let source = share[3].as_str().unwrap();
            let t = weights[j].as_quantized().unwrap().params.tensors[source].shallow_clone();
            if let V::Quantized(q) = &weights[i] {
                q.lock().unwrap().params.tensors.insert(name.into(), t);
            }
        }
        let input = restore(&c["input"], dev).as_tensor().unwrap();
        let biases = c["biases"]
            .as_array()
            .map(|b| b.iter().map(|v| restore(v, dev)).collect::<Vec<_>>());
        let validate = c["validate"].as_bool().unwrap();
        let trust = c["trust"].as_bool().unwrap();
        let guard = rt
            .backends
            .policy
            .use_backend(c["override"].as_str().unwrap())
            .unwrap();
        let note = format!("SVD grouping {i} {} {}", c["mode"], c["action"]);
        let result = (|| -> Result<V, imagine_inference::native::Error> {
            match c["action"].as_str().unwrap() {
                "can_share" => Ok(V::Bool(
                    rt.svdquant_can_share_quant(&weights, validate, trust)?,
                )),
                "group" => Ok(V::Tuple(unsafe {
                    rt.svdquant_grouped_linear(
                        &input,
                        &weights,
                        biases.as_deref(),
                        validate,
                        trust,
                    )?
                })),
                action => {
                    let (w, b, sizes) = rt.svdquant_fuse_linear_weights(
                        &weights,
                        biases.as_deref(),
                        validate,
                        trust,
                    )?;
                    if action == "fuse" {
                        let q = w.as_quantized()?;
                        let base = weights[0].as_quantized()?;
                        for (n, expected) in c["shared_output_fields"].as_object().unwrap() {
                            assert_eq!(
                                q.params.tensors[n].data_ptr() == base.params.tensors[n].data_ptr(),
                                expected.as_bool().unwrap(),
                                "{note} {n} alias"
                            );
                        }
                        Ok(V::Tuple(vec![
                            w,
                            b,
                            V::Tuple(sizes.into_iter().map(V::Int).collect()),
                        ]))
                    } else {
                        let output =
                            unsafe { rt.svdquant_fused_grouped_linear(&input, &w, &b, &sizes)? };
                        if c["split_shared_storage"].is_boolean() {
                            let a = output[0].as_tensor()?;
                            let b = output[1].as_tensor()?;
                            // Split views retain the fused row stride and adjacent feature offsets.
                            assert_eq!(a.stride(), b.stride());
                            assert_eq!(
                                (b.data_ptr() as usize) - (a.data_ptr() as usize),
                                sizes[0] as usize * a.kind().elt_size_in_bytes()
                            );
                            assert!(c["split_shared_storage"].as_bool().unwrap());
                        }
                        Ok(V::Tuple(output))
                    }
                }
            }
        })();
        drop(guard);
        if c["error"].is_string() {
            match result {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected rejection: {note}"),
            }
        } else {
            check(
                &result.unwrap_or_else(|e| panic!("{note}: {e}")),
                &c["output"],
                dev,
                &note,
            );
        }
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "{note} backend trace"
        );
    }
}

#[test]
fn svd_layout_parameter_attributes_follow_original_tensor_identity_and_clone_lifetime() {
    struct RestoreEnv(Vec<(&'static str, Option<std::ffi::OsString>)>);
    impl Drop for RestoreEnv {
        fn drop(&mut self) {
            for (n, v) in &self.0 {
                unsafe {
                    if let Some(v) = v {
                        std::env::set_var(n, v)
                    } else {
                        std::env::remove_var(n)
                    }
                }
            }
        }
    }
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let keys = [
        "COMFY_KITCHEN_SVDQUANT_FUSE_LORA_UP",
        "COMFY_KITCHEN_SVDQUANT_DIRECT_CUDA",
    ];
    let _env = RestoreEnv(keys.iter().map(|n| (*n, std::env::var_os(n))).collect());
    for (i, c) in fixtures()["svd_layout_tensor_attributes"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        unsafe {
            std::env::set_var(keys[0], "0");
            std::env::set_var(
                keys[1],
                if c["direct"].as_bool().unwrap() {
                    "1"
                } else {
                    "0"
                },
            );
        }
        let dev = device(c["device"].as_str().unwrap());
        let q = restore(&c["weight"], dev);
        let input = restore(&c["input"], dev);
        if c["shared_scale_up"].as_bool().unwrap_or(false) {
            let V::Quantized(q) = &q else { unreachable!() };
            let mut q = q.lock().unwrap();
            let up = q.params.tensors["proj_up"].shallow_clone();
            q.params.tensors.insert("scale".into(), up);
        }
        let root = q
            .as_quantized()
            .unwrap()
            .tensor_field_value(&rt.native, "proj_up")
            .unwrap();
        let mut previous = None::<Tensor>;
        let _backend = rt.backends.policy.use_backend("cuda").unwrap();
        for step in c["steps"].as_array().unwrap() {
            let action = step["action"].as_str().unwrap();
            let note = format!("SVD attributes {i} {action}");
            if action == "mutate" {
                let V::Quantized(p) = &q else { unreachable!() };
                let _ = p
                    .lock()
                    .unwrap()
                    .params
                    .tensors
                    .get_mut("proj_up")
                    .unwrap()
                    .f_add_scalar_(0.25)
                    .unwrap();
            }
            let weight = match action {
                "transpose" => unsafe {
                    rt.dispatch("aten::t.default", std::slice::from_ref(&q), &vec![])
                        .unwrap()
                },
                "clone" => unsafe {
                    rt.dispatch("aten::clone.default", std::slice::from_ref(&q), &vec![])
                        .unwrap()
                },
                "detach" => unsafe {
                    rt.dispatch("aten::detach.default", std::slice::from_ref(&q), &vec![])
                        .unwrap()
                },
                "contiguous" => unsafe {
                    rt.dispatch(
                        "aten::contiguous.default",
                        std::slice::from_ref(&q),
                        &vec![],
                    )
                    .unwrap()
                },
                "dtype" => unsafe {
                    rt.dispatch(
                        "aten::to.dtype_layout",
                        std::slice::from_ref(&q),
                        &vec![("dtype".into(), V::DType(Kind::Float))],
                    )
                    .unwrap()
                },
                "cpu" => unsafe {
                    rt.dispatch(
                        "aten::to.dtype_layout",
                        std::slice::from_ref(&q),
                        &vec![("device".into(), V::Device(Device::Cpu))],
                    )
                    .unwrap()
                },
                _ => q.clone(),
            };
            check(&weight, &step["weight"], dev, &note);
            let field = weight
                .as_quantized()
                .unwrap()
                .tensor_field_value(&rt.native, "proj_up")
                .unwrap();
            assert_eq!(
                field.same_object(&root),
                step["root_up_object"].as_bool().unwrap(),
                "{note} field object"
            );
            let scale = weight
                .as_quantized()
                .unwrap()
                .tensor_field_value(&rt.native, "scale")
                .unwrap();
            assert_eq!(
                scale.same_object(&field),
                step["scale_up_object"].as_bool().unwrap(),
                "{note} field aliases"
            );
            let before = field.natural_lora_cache();
            check(
                &before
                    .as_ref()
                    .map_or(V::None, |t| V::tensor(t.shallow_clone())),
                &step["cache_before"],
                dev,
                &note,
            );
            assert_eq!(
                before
                    .as_ref()
                    .zip(previous.as_ref())
                    .is_some_and(|(a, b)| a.data_ptr() == b.data_ptr()),
                step["same_cache_before"].as_bool().unwrap(),
                "{note} root cache"
            );
            if action != "cpu" {
                let op = if action == "transpose" {
                    "aten::mm.default"
                } else {
                    "aten::linear.default"
                };
                let out = unsafe { rt.dispatch(op, &[input.clone(), weight], &vec![]).unwrap() };
                check(&out, &step["output"], dev, &note);
                let after = field.natural_lora_cache();
                check(
                    &after
                        .as_ref()
                        .map_or(V::None, |t| V::tensor(t.shallow_clone())),
                    &step["cache_after"],
                    dev,
                    &note,
                );
                assert_eq!(
                    after
                        .as_ref()
                        .zip(before.as_ref())
                        .is_some_and(|(a, b)| a.data_ptr() == b.data_ptr()),
                    step["same_cache_after"].as_bool().unwrap(),
                    "{note} warm cache"
                );
                if action == "first" {
                    previous = after;
                }
            }
        }
    }
}
