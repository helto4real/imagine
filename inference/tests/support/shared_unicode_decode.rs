use imagine_inference::{patching::options::SourceText, python_unicode_codec};
use serde_json::{Value, json};
#[test]
fn original_cpython_utf8_strict_surrogatepass_and_ascii_preserve_codepoints_and_complete_error_messages_at_all_byte_boundaries()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-unicode-decode.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 57897);
    for case in fixture["cases"].as_array().unwrap() {
        let bytes = super::exact_hex(&case["bytes"]);
        let result = if case["encoding"] == "ascii" {
            python_unicode_codec::ascii(&bytes)
        } else {
            python_unicode_codec::utf8(&bytes, case["mode"] == "surrogatepass")
        };
        if case["error"].is_string() {
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("expected original decode failure: {case}"),
            };
            super::binding_tests::reference_error(error, case);
        } else {
            let value = result.unwrap_or_else(|error| panic!("{case}: {error:?}"));
            assert_eq!(
                json!(SourceText::get(&value).unwrap().codepoints()),
                case["points"],
                "{case}"
            );
        }
    }
}
