use super::{binding_tests::restore, fixtures, kitchen};
use imagine_inference::backend_contract;
use std::collections::BTreeMap;
use tch::Device;
#[test]
fn original_structured_values_preserve_hashability_lazy_diagnostics_and_na_tensor_validation() {
    compare("structured_metadata", 14592);
}
#[test]
fn original_sol_values_preserve_range_iteration_python_comparisons_truthiness_and_diagnostics() {
    compare("sol_structured", 756);
}
#[test]
fn original_default_tensor_representations_preserve_scalar_vector_nd_complex_summarization_and_autograd()
 {
    let cpu = std::env::var_os("IMAGINE_TENSOR_REPR_CPU_ONLY").is_some();
    if cpu {
        assert!(std::env::var_os("IMAGINE_STRUCTURED_METADATA_FIXTURE").is_some());
    }
    compare("tensor_repr", if cpu { 500 } else { 1000 });
}
pub(super) fn compare(category: &str, expected: usize) {
    let external = std::env::var("IMAGINE_STRUCTURED_METADATA_FIXTURE")
        .ok()
        .map(|p| {
            serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p).unwrap()).unwrap()
        });
    let cases = external.as_ref().unwrap_or_else(|| fixtures())[category]
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), expected);
    let native = kitchen();
    let tables = ["eager", "triton", "cuda", "cuda_lt"]
        .map(|b| {
            (
                b,
                backend_contract::constraints(
                    if b == "cuda_lt" { "cuda" } else { b },
                    b == "cuda_lt",
                )
                .unwrap(),
            )
        })
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let _dtype = super::triton_default_dtype_tests::DefaultDtype::preserve(native.clone());
    for c in cases {
        if let Some(dtype) = c["default_dtype"].as_str() {
            native.set_default_dtype(super::kind(dtype)).unwrap();
        }
        let args = c["kwargs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                (
                    pair[0].as_str().unwrap().to_owned(),
                    restore(&pair[1], Device::Cpu).metadata(&native).unwrap(),
                )
            })
            .collect();
        let result = tables[c["backend"].as_str().unwrap()][c["operation"].as_str().unwrap()]
            .validate(&args, Some((8, 9)));
        let expected = &c["expected"];
        if expected["success"] == true {
            assert_eq!(result, Ok(()), "{}", c["label"]);
        } else {
            let actual = result.unwrap_err();
            assert_eq!(
                actual.reason,
                expected["reason"].as_str().unwrap(),
                "{}",
                c["label"]
            );
            assert_eq!(
                actual.raised.as_deref(),
                expected["raised"].as_str(),
                "{}",
                c["label"]
            );
            if actual.raised.is_none() {
                assert_eq!(
                    actual.parameter,
                    expected["parameter"].as_str().unwrap(),
                    "{}",
                    c["label"]
                );
            }
        }
    }
}
