use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution_value::Value as V,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, Element, Fp8Body, Fp8Specialization},
};
use std::{collections::BTreeMap, sync::Arc};
fn element(k: Kind) -> Element {
    match k {
        Kind::Float => Element::F32,
        Kind::Half => Element::F16,
        Kind::BFloat16 => Element::BF16,
        Kind::Float8e4m3fn => Element::E4M3,
        Kind::Float8e5m2 => Element::E5M2,
        _ => panic!("Unexpected source element"),
    }
}
#[test]
fn runtime_rust_frontend_compiles_cold_reuses_warm_and_restores_original_disk_cache_for_all_fp8_types()
 {
    let cases = &fixtures()["triton_driver"];
    let profile = std::env::var("IMAGINE_TRITON_COMPILER_PROFILE")
        .expect("Verified compiler configuration required");
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let cache = own.join("cache");
    let work = own.join("work");
    let native = kitchen();
    let stream = native.current_stream(0).unwrap();
    let compiler =
        unsafe { Compiler::open(Path::new(&profile), &cache, &work, native.clone()).unwrap() };
    let mut remembered: BTreeMap<String, Arc<imagine_inference::triton_compiler::CompiledKernel>> =
        BTreeMap::new();
    let mut specs = vec![];
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        let cpu = c["operation"] == "cpu_pointer";
        let input = restore(&c["input"], Device::Cuda(0)).as_tensor().unwrap();
        let scale = if cpu {
            Tensor::from_slice(&[0.7f32]).to_device(Device::Cuda(0))
        } else {
            restore(&c["scale"], Device::Cuda(0)).as_tensor().unwrap()
        };
        let outkind = if cpu {
            Kind::Float8e4m3fn
        } else {
            kind(c["output"]["dtype"].as_str().unwrap())
        };
        let out = Tensor::empty(input.size(), (outkind, input.device()));
        let n = input.numel();
        let s = Fp8Specialization {
            body: if c["operation"] == "dequantize" {
                Fp8Body::Dequantize
            } else {
                Fp8Body::Quantize
            },
            input: element(input.kind()),
            output: element(outkind),
            scale: element(scale.kind()),
            block_size: 128,
            pointer_divisibility_16: [&input, &out, &scale]
                .map(|t| (t.data_ptr() as usize).is_multiple_of(16)),
            count: Count::I32 {
                divisible_by_16: n.is_multiple_of(16),
            },
            source_path: compiler.source_path(),
        };
        let compiled = unsafe {
            compiler
                .fp8(0, &s)
                .unwrap_or_else(|e| panic!("Cold compiler case {i}: {e}"))
        };
        if let Some(previous) = remembered.get(&compiled.digest) {
            assert!(
                Arc::ptr_eq(previous, &compiled),
                "Source JIT memory reuse {i}"
            );
        } else {
            assert!(!compiled.disk_cache_hit);
            specs.push(s);
            remembered.insert(compiled.digest.clone(), compiled.clone());
        }
        let mut args = [&input, &out, &scale]
            .map(|t| Scalar::TensorPointer(t.data_ptr() as u64))
            .into_iter()
            .collect::<Vec<_>>();
        if !matches!(s.body, Fp8Body::Dequantize) {
            args.push(Scalar::F32(if cpu {
                448.
            } else {
                c["limit"].as_f64().unwrap() as f32
            }));
        }
        args.extend([
            Scalar::I32(n.try_into().unwrap()),
            Scalar::Pointer(0),
            Scalar::Pointer(0),
        ]);
        let result = unsafe {
            compiled.run(
                [n.div_ceil(128).try_into().unwrap(), 1, 1],
                stream,
                &mut args,
            )
        };
        if cpu {
            match result {
                Err(e) => reference_error(e, c),
                Ok(()) => panic!("Expected source pointer rejection"),
            }
        } else {
            result.unwrap_or_else(|e| panic!("Runtime compiler case {i}: {e}"));
            check(
                &V::tensor(out),
                &c["output"],
                Device::Cuda(0),
                &format!("Runtime JIT {i}"),
            );
            let resources = compiled.resources().unwrap();
            assert_eq!(
                serde_json::json!({"registers":resources[0],"spills":resources[1],"max_threads":resources[2]}),
                c["resources"]
            );
        }
    }
    assert_eq!(remembered.len(), 24);
    // A new original JIT function set is cold in memory but reuses disk files.
    let restored = unsafe { Compiler::open(Path::new(&profile), &cache, &work, native).unwrap() };
    for s in specs {
        let kernel = unsafe { restored.fp8(0, &s).unwrap() };
        assert!(kernel.disk_cache_hit);
        let previous = &remembered[&kernel.digest];
        assert_eq!(kernel.metadata, previous.metadata);
        assert_eq!(kernel.assembly, previous.assembly);
        assert!(
            kernel.resources().is_none(),
            "Source driver handles remain lazy on disk restore"
        );
    }
    assert_eq!(
        std::fs::read_dir(&work).unwrap().count(),
        0,
        "Successful compiler temporaries are removed"
    );
}

#[test]
fn runtime_compiler_preserves_original_cold_and_warm_module_failures_resource_errors_and_handle_state()
 {
    let cases = &fixtures()["triton_runtime_failures"];
    let profile = std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap();
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let cache = own.join("cache");
    let work = own.join("work");
    let native = kitchen();
    let stream = native.current_stream(0).unwrap();
    let compiler =
        unsafe { Compiler::open(Path::new(&profile), &cache, &work, native.clone()).unwrap() };
    let s = Fp8Specialization {
        body: Fp8Body::Quantize,
        input: Element::F32,
        output: Element::E4M3,
        scale: Element::F32,
        block_size: 128,
        pointer_divisibility_16: [true; 3],
        count: Count::I32 {
            divisible_by_16: true,
        },
        source_path: compiler.source_path(),
    };
    let base = unsafe { compiler.fp8(0, &s).unwrap() };
    assert!(base.resources().is_none());
    let cached_path = |suffix: &str| {
        let (_, path) = base
            .metadata_group
            .iter()
            .find(|(key, _)| key.to_utf8().unwrap().ends_with(suffix))
            .unwrap();
        let imagine_inference::triton_file_cache::JsonPath::Text(path) = path else {
            unreachable!()
        };
        std::path::PathBuf::from(path.to_utf8().unwrap())
    };
    let metadata_path = cached_path(".json");
    let image_path = cached_path(".cubin");
    let valid_image = base
        .assembly
        .iter()
        .find(|(ext, _)| ext == "cubin")
        .unwrap()
        .1
        .clone();
    let input = Tensor::zeros([64], (Kind::Float, Device::Cuda(0)));
    let out = Tensor::empty([64], (Kind::Float8e4m3fn, Device::Cuda(0)));
    let scale = Tensor::ones([], (Kind::Float, Device::Cuda(0)));
    for c in cases.as_array().unwrap() {
        std::fs::write(&metadata_path, serde_json::to_vec(&c["metadata"]).unwrap()).unwrap();
        let image = c["image_hex"]
            .as_str()
            .map(|s| {
                (0..s.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| valid_image.clone());
        std::fs::write(&image_path, image).unwrap();
        let fresh =
            unsafe { Compiler::open(Path::new(&profile), &cache, &work, native.clone()).unwrap() };
        let kernel = unsafe { fresh.fp8(0, &s).unwrap() };
        assert!(kernel.disk_cache_hit);
        for step in c["steps"].as_array().unwrap() {
            let mut args = [&input, &out, &scale]
                .map(|t| Scalar::TensorPointer(t.data_ptr() as u64))
                .into_iter()
                .collect::<Vec<_>>();
            args.extend([
                Scalar::F32(448.),
                Scalar::I32(64),
                Scalar::Pointer(0),
                Scalar::Pointer(0),
            ]);
            match unsafe { kernel.run([1, 1, 1], stream, &mut args) } {
                Err(e) => reference_error(e, step),
                Ok(()) => panic!(
                    "Expected original safe load/resource rejection {}",
                    c["label"]
                ),
            }
            let (module, function) = kernel.driver_handle_presence();
            assert_eq!(module, step["module_present"].as_bool().unwrap());
            assert_eq!(function, step["function_present"].as_bool().unwrap());
            assert_eq!(serde_json::json!(kernel.resources()), step["resources"]);
        }
    }
}

#[test]
fn registered_triton_fp8_wrappers_preserve_shapes_views_defaults_argument_errors_and_all_block_thresholds()
 {
    use sha2::{Digest, Sha256};
    let cases = &fixtures()["triton_fp8_bindings"];
    let profile = std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap();
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let rt = imagine_inference::execution::Runtime::frozen_partial(kitchen()).unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&profile),
            &own.join("cache"),
            &own.join("work"),
            rt.native.clone(),
        )
        .unwrap()
    });
    rt.bind_triton_fp8(compiler);
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        let kwargs = super::binding_tests::kwargs(&c["kwargs"], Device::Cuda(0));
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let result = unsafe { selected.call(&rt, &args, &kwargs) };
        if c["error"].is_string() {
            match result {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected wrapper error {i} {}", c["label"]),
            }
        } else {
            check(
                &result.unwrap_or_else(|e| panic!("Wrapper {i} {}: {e}", c["label"])),
                &c["output"],
                Device::Cuda(0),
                &format!("FP8 wrapper {i} {}", c["label"]),
            );
        }
    }
    for c in fixtures()["triton_fp8_jit_thresholds"].as_array().unwrap() {
        let n = c["n"].as_i64().unwrap();
        let input = Tensor::arange(n, (Kind::Float, Device::Cuda(0)))
            .f_remainder(97)
            .unwrap()
            .f_mul_scalar(0.125)
            .unwrap()
            .f_sub_scalar(4.)
            .unwrap()
            .to_kind(kind(c["input_dtype"].as_str().unwrap()));
        let scale = Tensor::from_slice(&[0.7f32]).to_device(Device::Cuda(0));
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let result = unsafe {
            selected
                .call(
                    &rt,
                    &[
                        V::tensor(input),
                        V::tensor(scale),
                        V::DType(kind(c["output_dtype"].as_str().unwrap())),
                    ],
                    &vec![],
                )
                .unwrap()
        }
        .as_tensor()
        .unwrap();
        assert_eq!(serde_json::json!(result.size()), c["shape"]);
        assert_eq!(serde_json::json!(result.stride()), c["strides"]);
        let actual = Sha256::digest(bits(&result))
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            actual, c["sha256"],
            "Complete byte digest at original block boundary {} {n}",
            c["operation"]
        );
    }
}

#[test]
fn rowwise_specialization_keys_preserve_original_types_alignment_count_and_both_constants() {
    use imagine_inference::{
        triton_frontend::rowwise::Specialization, triton_specialization::rowwise_ast_digest,
    };
    let function_key = fixtures()["triton_rowwise_function_cache_key"]
        .as_str()
        .unwrap();
    for c in fixtures()["triton_rowwise_specialization"]
        .as_array()
        .unwrap()
    {
        let count = match c["count_type"].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: c["count_div16"].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: c["count_div16"].as_bool().unwrap(),
            },
            _ => unreachable!(),
        };
        let s = Specialization {
            input: match c["input"].as_str().unwrap() {
                "fp32" => Element::F32,
                "fp16" => Element::F16,
                "bf16" => Element::BF16,
                _ => unreachable!(),
            },
            input_dtype_code: c["input_dtype_code"].as_u64().unwrap() as u8,
            block_size: c["block_size"].as_u64().unwrap() as u32,
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            count,
            source_path: c["source_path"].as_str().unwrap(),
            standard_source_path: c["standard_source_path"].as_str().unwrap(),
        };
        assert_eq!(
            rowwise_ast_digest(&s, function_key),
            c["ast_digest"].as_str().unwrap()
        );
    }
}

#[test]
fn registered_rowwise_triton_bindings_preserve_original_quantization_rotation_seed_strides_and_errors()
 {
    let profile = std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap();
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let rt = imagine_inference::execution::Runtime::frozen_partial(kitchen()).unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&profile),
            &own.join("cache"),
            &own.join("work"),
            rt.native.clone(),
        )
        .unwrap()
    });
    rt.bind_triton_fp8(compiler.clone());
    rt.bind_triton_rowwise(compiler);
    assert_eq!(rt.backends.gaps().len(), 24);
    for (i, c) in fixtures()["triton_rowwise_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        let kwargs = super::binding_tests::kwargs(&c["kwargs"], Device::Cuda(0));
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let result = unsafe { selected.call(&rt, &args, &kwargs) };
        if c["error"].is_string() {
            match result {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected original rowwise error {i} {}", c["label"]),
            }
        } else {
            check(
                &result.unwrap_or_else(|e| panic!("Rowwise wrapper {i} {}: {e}", c["label"])),
                &c["output"],
                Device::Cuda(0),
                &format!("Rowwise {i} {}", c["label"]),
            );
        }
    }
}
