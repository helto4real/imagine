use super::{
    binding_tests::{check, reference_error, restore},
    kind, kitchen,
};
use imagine_inference::{
    checkpoint::detection, execution::Runtime, execution_value::Value as V,
    patching::options::Options,
};
use serde_json::Value;
use std::{
    cell::RefCell,
    ffi::{CStr, c_void},
    rc::Rc,
};
use tch::{Device, Tensor};
unsafe extern "C" {
    fn at_imagine_test_meta_empty(dims: *const i64, count: i64, dtype: i64) -> *mut c_void;
}
pub(super) fn meta(shape: &[i64], kind: tch::Kind) -> Tensor {
    // This symbol is compiled only by the existing dev-dependency feature.
    // No model-sized CPU/GPU allocation exists anywhere in this constructor.
    let prototype = Tensor::empty([0], (kind, Device::Cpu));
    let dtype = unsafe { torch_sys::at_scalar_type(prototype.as_ptr().cast_mut()) };
    let raw =
        unsafe { at_imagine_test_meta_empty(shape.as_ptr(), shape.len() as i64, i64::from(dtype)) };
    let error = unsafe { torch_sys::get_and_reset_last_err() };
    if !error.is_null() {
        let message = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe {
            libc::free(error.cast());
        }
        panic!("Synthetic meta constructor: {message}");
    }
    assert!(!raw.is_null());
    unsafe { Tensor::from_ptr(raw.cast()) }
}
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
#[test]
fn whole_model_detector_matches_original_shape_only_checkpoints_and_tiny_learned_std_fields() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _guard = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(1);
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-model-detection.json")).unwrap();
    assert_eq!(fixture["trained_model_weights"], 0);
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 328);
    assert_eq!(
        fixture["observed_success_return_lines"]
            .as_array()
            .unwrap()
            .len(),
        52
    );
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        assert!(case["real_data_bytes"].as_u64().unwrap() <= 8192);
        let weights = case["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                let specification = &entry[1];
                let value = if specification["meta"] == true {
                    let shape = specification["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_i64().unwrap())
                        .collect::<Vec<_>>();
                    V::tensor(meta(&shape, kind(specification["kind"].as_str().unwrap())))
                } else {
                    restore(&specification["value"], Device::Cpu)
                };
                (entry[0].as_str().unwrap().into(), Options::Atom(value))
            })
            .collect();
        let state = Options::Dict(Rc::new(RefCell::new(weights)));
        let metadata = Options::from_value(restore(&case["metadata"], Device::Cpu));
        let result = detection::detect_unet_config(
            &rt,
            &state,
            case["prefix"].as_str().unwrap(),
            Some(&metadata),
        );
        let note = format!(
            "detector {index} {} line {}",
            case["label"], case["target_if_line"]
        );
        if case["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected source error {note}")),
                case,
            );
        } else {
            check(
                &result
                    .unwrap_or_else(|e| panic!("{note}: {e}"))
                    .snapshot()
                    .unwrap(),
                &case["output"],
                Device::Cpu,
                &note,
            );
        }
    }
}
