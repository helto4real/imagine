use imagine_inference::{execution_value::Value as V, native, patching};
use serde_json::Value;
fn without_approved_stack(message: &str) -> (&str, &str) {
    let (prefix, stack) = message
        .split_once("\nException raised from ")
        .expect("retained raw C++ stack");
    assert!(prefix.contains("Overflow when unpacking long long"));
    assert!(stack.contains("frame #0:"), "actual native frames required");
    assert!(
        stack.ends_with('"'),
        "preserve original wrapper closing quote"
    );
    (prefix, &stack[stack.len() - 1..])
}
#[test]
fn original_padding_raw_shape_values_preserve_eager_comparisons_truth_dimension_guards_size_parser_errors_and_approved_overflow_stacks()
 {
    let _library = super::kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-pad-values.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 584);
    for case in fixture["cases"].as_array().unwrap() {
        let device = super::device(case["weight"]["device"].as_str().unwrap());
        let weight = super::binding_tests::restore(&case["weight"], device)
            .as_tensor()
            .unwrap();
        let shape = super::binding_tests::restore(&case["shape"], device);
        for expected in case["results"].as_array().unwrap() {
            let result = patching::pad_to_value_shape(&weight, &shape);
            if expected["error"].is_string() {
                let error = result.expect_err("source failed");
                if expected["message"]
                    .as_str()
                    .unwrap()
                    .contains("Overflow when unpacking long long\nException raised from ")
                {
                    // D-05-LORA-SHAPE-OVERFLOW-BACKTRACE: only this operation,
                    // overflow, error class and embedded segment are admitted.
                    let native::Error::ReferenceException { class, message } = error.root() else {
                        panic!("expected typed source TypeError: {error:?}")
                    };
                    assert_eq!(*class, "TypeError");
                    assert_eq!(
                        without_approved_stack(message),
                        without_approved_stack(expected["message"].as_str().unwrap()),
                        "{case}"
                    );
                } else {
                    super::binding_tests::reference_error(error, expected);
                }
            } else {
                let output = result.unwrap_or_else(|error| panic!("{}: {error:?}", case["label"]));
                assert_eq!(
                    output.as_ptr() == weight.as_ptr(),
                    expected["same"].as_bool().unwrap()
                );
                assert_eq!(
                    native::storage_info(&output).unwrap().0
                        == native::storage_info(&weight).unwrap().0,
                    expected["storage_alias"].as_bool().unwrap()
                );
                super::binding_tests::check(
                    &V::tensor(output),
                    &expected["output"],
                    device,
                    case["label"].as_str().unwrap(),
                );
            }
            super::binding_tests::check(
                &V::tensor(weight.shallow_clone()),
                &case["weight"],
                device,
                case["label"].as_str().unwrap(),
            );
        }
    }
}
