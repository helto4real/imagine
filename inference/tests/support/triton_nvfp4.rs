use super::binding_tests::{check, kwargs, reference_error, restore};
use super::{bits, fixtures, kind, kitchen};
use imagine_inference::{
    execution::Runtime,
    native::Error,
    triton_compiler::Compiler,
    triton_frontend::{Count, Element, nvfp4},
    triton_specialization::{dequant_nvfp4_ast_digest, nvfp4_ast_digest},
};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use tch::{Device, Kind};
fn count(c: &serde_json::Value, i: usize) -> Count {
    match c["scalar_types"][i].as_str().unwrap() {
        "one" => Count::One,
        "i32" => Count::I32 {
            divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
        },
        "i64" => Count::I64 {
            divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
        },
        _ => panic!("Original scalar type"),
    }
}
fn element(v: &serde_json::Value) -> Element {
    match v.as_str().unwrap() {
        "fp32" => Element::F32,
        "fp16" => Element::F16,
        "bf16" => Element::BF16,
        _ => panic!("Original floating signature"),
    }
}
#[test]
fn original_nvfp4_specialization_keys_preserve_both_nibble_orders_scalar_types_and_constexpr_positions()
 {
    for c in fixtures()["triton_nvfp4_specialization"]
        .as_array()
        .unwrap()
    {
        let s = nvfp4::Specialization {
            input: element(&c["input"]),
            global_scale: Element::F32,
            block_size: c["block_size"].as_u64().unwrap() as u32,
            blocks_per_program: c["blocks_per_program"].as_u64().unwrap() as u32,
            hi_first: c["hi_first"].as_bool().unwrap(),
            scalars: std::array::from_fn(|i| count(c, i)),
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: "",
            standard_source_path: "",
        };
        assert_eq!(
            nvfp4_ast_digest(
                &s,
                fixtures()["triton_nvfp4_function_cache_key"]
                    .as_str()
                    .unwrap()
            ),
            c["ast_digest"].as_str().unwrap(),
            "{c}"
        );
    }
    for c in fixtures()["triton_dequant_nvfp4_specialization"]
        .as_array()
        .unwrap()
    {
        let s = nvfp4::dequantize::Specialization {
            output: element(&c["input"]),
            global_scale: Element::F32,
            block_scale: Element::E4M3,
            block_size: c["block_size"].as_u64().unwrap() as u32,
            tile_size: 128,
            hi_first: c["hi_first"].as_bool().unwrap(),
            scalars: std::array::from_fn(|i| count(c, i)),
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: "",
        };
        assert_eq!(
            dequant_nvfp4_ast_digest(
                &s,
                fixtures()["triton_dequant_nvfp4_function_cache_key"]
                    .as_str()
                    .unwrap()
            ),
            c["ast_digest"].as_str().unwrap(),
            "{c}"
        );
    }
}
#[test]
fn registered_nvfp4_wrappers_preserve_original_pre_run_layout_dtype_aliases_strides_and_early_errors()
 {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            rt.native.clone(),
        )
        .unwrap()
    });
    rt.bind_triton_nvfp4(compiler.clone());
    for (i, c) in fixtures()["triton_nvfp4_boundaries"]
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
        let kw = kwargs(&c["kwargs"], Device::Cuda(0));
        let expected = c.clone();
        let inputs = args.clone();
        let seen = Arc::new(AtomicUsize::new(0));
        let calls = seen.clone();
        let native = rt.native.clone();
        compiler.set_pre_run_guard(Some(Arc::new(move |op, values, kw| {
            assert_eq!(
                op,
                if expected["operation"] == "quantize_nvfp4" {
                    "nvfp4"
                } else {
                    "dequant_nvfp4"
                }
            );
            let launch = expected["launch"].as_array().unwrap();
            assert_eq!(values.len(), launch.len());
            for (v, e) in values.iter().zip(launch) {
                if let Some(meta) = e.get("tensor_metadata") {
                    let t = v.as_tensor()?;
                    assert_eq!(serde_json::json!(t.size()), meta["shape"]);
                    assert_eq!(serde_json::json!(t.stride()), meta["strides"]);
                    assert_eq!(t.kind(), kind(meta["dtype"].as_str().unwrap()));
                    assert_eq!(t.device(), super::device(meta["device"].as_str().unwrap()));
                    if e["payload"].is_array() {
                        assert_eq!(serde_json::json!(bits(&t)), e["payload"]);
                    }
                    let identities = inputs.iter().map(|x| v.same_object(x)).collect::<Vec<_>>();
                    let aliases = inputs
                        .iter()
                        .map(|x| x.as_tensor().is_ok_and(|x| x.data_ptr() == t.data_ptr()))
                        .collect::<Vec<_>>();
                    assert_eq!(serde_json::json!(identities), e["input_identities"]);
                    assert_eq!(serde_json::json!(aliases), e["input_aliases"]);
                    assert_eq!(
                        native.tensor_type(&t)?.scalar_type,
                        match t.kind() {
                            Kind::Uint8 => 0,
                            Kind::Float => 6,
                            Kind::Half => 5,
                            Kind::BFloat16 => 15,
                            Kind::Float8e4m3fn => 24,
                            _ => panic!("Original NVFP4 boundary dtype"),
                        }
                    );
                } else {
                    check(v, e, Device::Cuda(0), "Original NVFP4 scalar argument");
                }
            }
            assert_eq!(
                kw.len(),
                expected["launch_kwargs"].as_array().unwrap().len()
            );
            for ((name, value), e) in kw.iter().zip(expected["launch_kwargs"].as_array().unwrap()) {
                assert_eq!(name, e[0].as_str().unwrap());
                check(
                    value,
                    &e[1],
                    Device::Cuda(0),
                    "Original NVFP4 compile-time argument",
                );
            }
            calls.fetch_add(1, Ordering::SeqCst);
            Err(Error::ReferenceException {
                class: "OwnBoundaryReached",
                message: "own synthetic pre-run boundary reached".into(),
            })
        })));
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let result = unsafe { selected.call(&rt, &args, &kw) };
        compiler.set_pre_run_guard(None);
        reference_error(
            result
                .err()
                .expect("Original pre-run boundary or early error"),
            c,
        );
        assert_eq!(
            seen.load(Ordering::SeqCst),
            usize::from(c.get("launch").is_some()),
            "{i}"
        );
        for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, Device::Cuda(0), "NVFP4 unchanged input");
        }
    }
}

#[test]
fn registered_nvfp4_wrappers_preserve_original_early_errors_generated_ptx_and_repeated_assembler_failures()
 {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            rt.native.clone(),
        )
        .unwrap()
    });
    rt.bind_triton_nvfp4(compiler);
    for c in fixtures()["triton_nvfp4_bindings"].as_array().unwrap() {
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        let kw = kwargs(&c["kwargs"], Device::Cuda(0));
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let mut previous_path = None;
        for _ in 0..if c["error"] == "PTXASError" { 2 } else { 1 } {
            let failure = unsafe { selected.call(&rt, &args, &kw) }
                .err()
                .expect("Original NVFP4 rejection");
            if c["error"] == "PTXASError" {
                let Error::ReferenceException { class, message } = failure else {
                    panic!("Original PTXAS error identity")
                };
                assert_eq!(class, "PTXASError");
                let command = message.split_once("Repro command: ").unwrap().1.trim();
                let parts = command.split_whitespace().collect::<Vec<_>>();
                let input = parts[parts.len() - 3];
                let output = parts[parts.len() - 1];
                assert_eq!(parts[parts.len() - 2], "-o");
                assert!(Path::new(input).starts_with(own.join("work")));
                assert!(Path::new(input).is_file());
                assert!(!Path::new(output).exists());
                assert_eq!(
                    std::fs::read_to_string(input).unwrap(),
                    c["compiler_ptx"][0].as_str().unwrap(),
                    "Original complete wrapper-generated PTX {}",
                    c["label"]
                );
                // Work-file names are parameters of the paired environments.
                // Bind both concrete names; compare every other message byte,
                // including flags, native diagnostics, line numbers and paths.
                let original = c["message"].as_str().unwrap();
                let original_parts = original
                    .split_once("Repro command: ")
                    .unwrap()
                    .1
                    .split_whitespace()
                    .collect::<Vec<_>>();
                let original_input = original_parts[original_parts.len() - 3];
                let original_output = original_parts[original_parts.len() - 1];
                let expected = original
                    .replace(original_output, output)
                    .replace(original_input, input);
                assert_eq!(message, expected);
                assert_ne!(
                    previous_path.as_deref(),
                    Some(input),
                    "Failed compilations must retry instead of publishing a kernel"
                );
                previous_path = Some(input.to_string());
            } else {
                reference_error(failure, c);
            }
        }
        for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            check(
                v,
                e,
                Device::Cuda(0),
                "NVFP4 rejection preserves original inputs",
            );
        }
    }
    for entry in std::fs::read_dir(own.join("cache")).unwrap() {
        for file in std::fs::read_dir(entry.unwrap().path()).unwrap() {
            let path = file.unwrap().path();
            assert!(
                !matches!(
                    path.extension().and_then(|x| x.to_str()),
                    Some("json" | "cubin")
                ),
                "Failed compilation must not publish metadata/group/cubin {path:?}"
            );
        }
    }
}

#[test]
fn original_nvfp4_literal_ast_keys_preserve_none_numeric_container_and_tensor_identity_text() {
    let external = std::env::var("IMAGINE_NVFP4_LITERAL_FIXTURE")
        .ok()
        .map(|p| {
            serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p).unwrap()).unwrap()
        });
    let cases = external
        .as_ref()
        .map(|v| &v["literal_cases"])
        .unwrap_or_else(|| &fixtures()["triton_nvfp4_literal_ast"])
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), 78);
    let profile: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap())
            .unwrap(),
    )
    .unwrap();
    for c in cases {
        let function = &profile["functions"][c["family"].as_str().unwrap()];
        let names = function["argument_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        let index = names.iter().position(|n| *n == "hi_first").unwrap();
        let literal = c["constants"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v[0] == serde_json::json!([index]))
            .unwrap()[1]
            .as_str()
            .unwrap();
        let count = Count::I32 {
            divisible_by_16: false,
        };
        let ast = if c["family"] == "nvfp4" {
            let s = nvfp4::Specialization {
                input: element(&c["element"]),
                global_scale: Element::F32,
                block_size: 16,
                blocks_per_program: 1,
                hi_first: false,
                scalars: [count; 7],
                pointer_divisibility_16: [true; 4],
                source_path: "",
                standard_source_path: "",
            };
            imagine_inference::triton_specialization::nvfp4_literal_ast_digest(
                &s,
                c["function_cache_key"].as_str().unwrap(),
                &names,
                literal,
            )
        } else {
            let s = nvfp4::dequantize::Specialization {
                output: element(&c["element"]),
                block_scale: Element::E4M3,
                global_scale: Element::F32,
                block_size: 16,
                tile_size: 128,
                hi_first: false,
                scalars: [count; 4],
                pointer_divisibility_16: [true; 4],
                source_path: "",
            };
            imagine_inference::triton_specialization::dequant_nvfp4_literal_ast_digest(
                &s,
                c["function_cache_key"].as_str().unwrap(),
                &names,
                literal,
            )
        };
        assert_eq!(
            ast,
            c["ast_digest"].as_str().unwrap(),
            "{} {}",
            c["family"],
            c["index"]
        );
    }
}
