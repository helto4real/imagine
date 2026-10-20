use imagine_inference::{checkpoint, native::Error};
use serde_json::{Value, json};
use std::{path::Path, rc::Rc};
#[test]
fn original_safetensors_header_unicode_json_syntax_extra_data_integer_limit_errors_preserve_wrapping_causes_context_identity_and_valid_ordered_entries()
 {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-checkpoint-headers.json"
    ))
    .unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 33);
    for case in fixture["cases"].as_array().unwrap() {
        let bytes = super::exact_hex(&case["bytes"]);
        let result = checkpoint::parse_header(&bytes, Path::new("{file}"));
        if case["error"].is_string() {
            let exception = result.expect_err("source header failure").into_exception();
            super::binding_tests::reference_error(Error::Retained(exception.clone()), case);
            assert_eq!(
                exception.suppress_context(),
                case["suppress_context"].as_bool().unwrap(),
                "{case}"
            );
            let cause = exception.cause();
            assert_eq!(cause.is_some(), case["cause"].is_object(), "{case}");
            if let Some(cause) = &cause {
                super::binding_tests::reference_error(
                    Error::Retained(cause.clone()),
                    &case["cause"],
                );
            }
            let same = match (exception.context(), cause) {
                (Some(context), Some(cause)) => Rc::ptr_eq(&context, &cause),
                _ => false,
            };
            assert_eq!(same, case["context_is_cause"].as_bool().unwrap(), "{case}");
        } else {
            assert_eq!(
                json!(result.unwrap_or_else(|error| panic!("{case}: {error:?}"))),
                case["items"],
                "{case}"
            );
        }
    }
}
