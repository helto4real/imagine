use super::{fixtures, kind, kitchen};
use imagine_inference::{execution::Runtime, native::Error, triton_compiler::Compiler};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::Device;
#[test]
fn original_dynamic_arguments_preserve_constexpr_types_scalar_keys_and_none_decode_paths() {
    compare("dynamic_arguments", 56)
}
#[test]
fn original_dynamic_seeds_preserve_comparison_generator_native_binding_and_rope_state() {
    compare("dynamic_seed_state", 136)
}
#[test]
fn original_na_host_arithmetic_preserves_negative_windows_tensor_sweeps_and_cold_literal_errors() {
    compare("dynamic_na_host", 39)
}
#[test]
fn original_nvfp4_dynamic_literals_preserve_three_distinct_backends_and_native_bool_errors() {
    compare("dynamic_nvfp4_literals", 90)
}
#[test]
fn original_uint64_stochastic_seeds_preserve_enablement_and_complete_unsigned_bits() {
    compare("uint64_seeds", 138)
}
#[test]
fn original_int8_rms_epsilon_preserves_optional_defaults_scalar_tensors_grad_and_parser_errors() {
    compare("int8_rms_epsilon", 174)
}
#[test]
fn original_weight_seed_values_preserve_delayed_integer_conversion_branch_enablement_and_native_diagnostics()
 {
    compare("seed_types", 450)
}
#[test]
fn original_eager_na_values_preserve_tensor_geometry_dtype_scalar_scale_grad_and_source_errors() {
    compare("na_values", 560)
}
#[test]
fn original_constexpr_state_preserves_signed_zero_warm_aliases_cold_failures_and_hash_before_repr()
{
    compare("constexpr_state", 18)
}
#[test]
fn original_cuda_na_values_preserve_indexing_conversion_order_native_ranges_and_outputs() {
    compare("cuda_na_values", 68)
}
#[test]
fn original_input_activation_values_preserve_lazy_weights_fused_hashing_and_scalar_error_order() {
    compare("input_activation_values", 196)
}
fn compare(category: &str, expected: usize) {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let _dtype = super::triton_default_dtype_tests::DefaultDtype::preserve(native.clone());
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let loads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed = loads.clone();
    compiler
        .hooks
        .replace(imagine_inference::triton_hooks::Hooks {
            load_start: Some(Arc::new(move |_| {
                observed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })),
            ..Default::default()
        });
    let rt = Runtime::frozen(native.clone(), compiler.clone()).unwrap();
    let external = match category {
        "dynamic_nvfp4_literals" => std::env::var("IMAGINE_DYNAMIC_NVFP4_FIXTURE").ok(),
        "uint64_seeds" => std::env::var("IMAGINE_UINT64_SEEDS_FIXTURE").ok(),
        "int8_rms_epsilon" => std::env::var("IMAGINE_INT8_RMS_EPSILON_FIXTURE").ok(),
        "input_activation_values" => std::env::var("IMAGINE_INPUT_ACTIVATION_VALUES_FIXTURE").ok(),
        "seed_types" => std::env::var("IMAGINE_SEED_TYPES_FIXTURE").ok(),
        "na_values" => std::env::var("IMAGINE_NA_VALUES_FIXTURE").ok(),
        "constexpr_state" => std::env::var("IMAGINE_CONSTEXPR_STATE_FIXTURE").ok(),
        "cuda_na_values" => std::env::var("IMAGINE_CUDA_NA_VALUES_FIXTURE").ok(),
        _ => None,
    }
    .map(|p| {
        serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p).unwrap()).unwrap()
    });
    let cases = external.as_ref().unwrap_or_else(|| fixtures())[category]
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), expected);
    for c in cases {
        eprintln!("Dynamic argument: {}", c["label"]);
        if c["clear_jit"].as_bool().unwrap_or(false) {
            compiler.clear_specializations();
        }
        native
            .set_default_dtype(
                c.get("default_dtype")
                    .map(|v| kind(v.as_str().unwrap()))
                    .unwrap_or(tch::Kind::Float),
            )
            .unwrap();
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| super::binding_tests::restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        let f = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                c["backend"].as_str(),
                &BTreeMap::new(),
            )
            .unwrap();
        let before = loads.load(std::sync::atomic::Ordering::Relaxed);
        for _ in 0..2 {
            let kwargs = super::binding_tests::kwargs(&c["kwargs"], Device::Cuda(0));
            let result = unsafe { f.call(&rt, &args, &kwargs) };
            if c["error"] == "PTXASError" {
                let Error::ReferenceException { class, message } =
                    result.err().expect("Original SM89 assembler rejection")
                else {
                    panic!("Original error identity")
                };
                assert_eq!(class, "PTXASError");
                let parts = message
                    .split_once("Repro command: ")
                    .unwrap()
                    .1
                    .split_whitespace()
                    .collect::<Vec<_>>();
                let input = parts[parts.len() - 3];
                let output = parts[parts.len() - 1];
                assert!(Path::new(input).starts_with(own.join("work")));
                assert_eq!(
                    std::fs::read_to_string(input).unwrap(),
                    c["compiler_ptx"].as_str().unwrap(),
                    "Complete original PTX {}",
                    c["label"]
                );
                let original = c["message"].as_str().unwrap();
                let before = original
                    .split_once("Repro command: ")
                    .unwrap()
                    .1
                    .split_whitespace()
                    .collect::<Vec<_>>();
                assert_eq!(
                    message,
                    original
                        .replace(before[before.len() - 1], output)
                        .replace(before[before.len() - 3], input)
                );
            } else if c.get("error").is_some() {
                super::binding_tests::reference_error(
                    result
                        .err()
                        .unwrap_or_else(|| panic!("Original rejection {}", c["label"])),
                    c,
                );
            } else {
                super::binding_tests::check(
                    &result.unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                    &c["output"],
                    Device::Cuda(0),
                    c["label"].as_str().unwrap(),
                );
            }
        }
        if category == "uint64_seeds" {
            use imagine_inference::execution_value::Value as Source;
            fn alternate(v: &mut Source) {
                match v {
                    Source::Int(n) if *n >= 0 => *v = Source::UInt(*n as u64),
                    Source::List(v) | Source::Tuple(v) => v.iter_mut().for_each(alternate),
                    Source::Dict(v) => v.iter_mut().for_each(|(_, v)| alternate(v)),
                    _ => {}
                }
            }
            let mut alternate_args = args.clone();
            alternate_args.iter_mut().for_each(alternate);
            let mut kwargs = super::binding_tests::kwargs(&c["kwargs"], Device::Cuda(0));
            kwargs.iter_mut().for_each(|(_, v)| alternate(v));
            for _ in 0..2 {
                let result = unsafe { f.call(&rt, &alternate_args, &kwargs) };
                if c.get("error").is_some() {
                    super::binding_tests::reference_error(
                        result.err().expect("Original unsigned alias rejection"),
                        c,
                    );
                } else {
                    super::binding_tests::check(
                        &result.unwrap_or_else(|e| {
                            panic!("Original unsigned alias {}: {e}", c["label"])
                        }),
                        &c["output"],
                        Device::Cuda(0),
                        "Signed-compatible UInt representation retains the original Python int result",
                    );
                }
            }
        }
        if let Some(expected) = c["expected_new_loads"].as_u64() {
            assert_eq!(
                loads.load(std::sync::atomic::Ordering::Relaxed) - before,
                expected as usize,
                "Original cache-load state {}",
                c["label"]
            );
        }
        if let Some(post) = c.get("post_args").and_then(serde_json::Value::as_array) {
            for (input, expected) in args.iter().zip(post) {
                super::binding_tests::check(
                    input,
                    expected,
                    Device::Cuda(0),
                    "Original dynamic arguments preserve inputs",
                );
            }
        }
        for source in c
            .get("jit_specializations")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flatten()
        {
            assert!(
                own.join("cache")
                    .join(
                        imagine_inference::triton_specialization::cache_directory(
                            source["compiler_digest"].as_str().unwrap()
                        )
                        .unwrap()
                    )
                    .is_dir(),
                "Exact original cold AST/compiler key {}: {}",
                c["label"],
                source["compiler_digest"]
            );
        }
    }
}
