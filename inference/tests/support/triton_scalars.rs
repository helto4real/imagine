use super::{fixtures, kitchen};
use imagine_inference::{
    execution::Runtime, triton_compiler::Compiler, triton_specialization::cache_directory,
};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::Device;
#[test]
fn original_numeric_and_structured_epsilon_wrappers_preserve_outputs_errors_and_cold_cache_keys() {
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
    let rt = Runtime::frozen(native, compiler).unwrap();
    let cases = fixtures()["triton_scalar_bindings"].as_array().unwrap();
    assert_eq!(cases.len(), 180);
    assert_eq!(
        cases.iter().filter(|c| c.get("output").is_some()).count(),
        144
    );
    for c in cases {
        eprintln!("Scalar wrapper: {}", c["label"]);
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
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        for _ in 0..if c.get("output").is_some() { 2 } else { 1 } {
            let result = unsafe { f.call(&rt, &args, &vec![]) };
            if c.get("error").is_some() {
                super::binding_tests::reference_error(
                    result.err().expect("Original defined epsilon failure"),
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
        if let Some(original) = c.get("jit_specialization") {
            assert!(
                own.join("cache")
                    .join(cache_directory(original["compiler_digest"].as_str().unwrap()).unwrap())
                    .is_dir(),
                "Actual original cold compiler key {}",
                c["label"]
            );
        }
        for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            super::binding_tests::check(v, e, Device::Cuda(0), "Unchanged epsilon input");
        }
    }
}

pub(super) fn epsilon(
    c: &serde_json::Value,
) -> imagine_inference::triton_frontend::scalar::ScalarType {
    use imagine_inference::triton_frontend::scalar::ScalarType;
    match c
        .get("epsilon_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("fp32")
    {
        "fp32" => ScalarType::F32,
        "i32" => ScalarType::I32 {
            divisible_by_16: c["epsilon_div16"].as_bool().unwrap_or(false),
        },
        "i64" => ScalarType::I64 {
            divisible_by_16: c["epsilon_div16"].as_bool().unwrap_or(false),
        },
        "u64" => ScalarType::U64 {
            divisible_by_16: c["epsilon_div16"].as_bool().unwrap_or(false),
        },
        "one" => ScalarType::One,
        "u1" => ScalarType::Bool,
        _ => panic!("Observed original runtime scalar type"),
    }
}
