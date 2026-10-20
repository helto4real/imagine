use imagine_inference::{
    execution_value::Value as V,
    patching::options::{Options, SourceText},
    python_hash::HashState,
};
use serde_json::{Value, json};

#[test]
fn original_unicode_hashes_preserve_code_unit_width_lone_surrogates_empty_singleton_and_native_aliases()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-unicode.json")).unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["hash_algorithm"], "siphash13");
    assert_eq!(fixture["trained_model_weights"], 0);
    for run in fixture["runs"].as_array().unwrap() {
        let state = HashState::from_seed(run["seed"].as_u64().unwrap() as u32);
        for case in run["cases"].as_array().unwrap() {
            let points = case["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            let value = SourceText::from_codepoints(points.clone()).unwrap();
            let text = SourceText::get(&value).unwrap();
            assert_eq!(text.codepoints(), points);
            assert_eq!(
                state.hash_unicode(text.codepoints()),
                case["hash"].as_i64().unwrap()
            );
            assert_eq!(text.utf8(), case["utf8"].as_str().map(str::to_owned));
            assert!(value.same_object(&value.clone()));
            let other = SourceText::from_codepoints(points.clone()).unwrap();
            assert_eq!(value.same_object(&other), points.is_empty());
            assert_eq!(
                SourceText::get(&value).unwrap().identity_token()
                    == SourceText::get(&other).unwrap().identity_token(),
                points.is_empty()
            );
            if let Some(text) = case["utf8"].as_str() {
                assert_eq!(state.hash_text(text), case["hash"].as_i64().unwrap());
                assert!(matches!(value.snapshot().unwrap(),V::Text(v) if v==text));
            } else {
                assert!(value.snapshot().is_err());
            }
        }
    }
    assert!(SourceText::from_codepoints(vec![0x110000]).is_err());
    let empty = SourceText::from_utf8("");
    assert!(empty.same_object(&SourceText::from_codepoints(vec![]).unwrap()));
    assert!(!empty.same_object(&Options::Atom(V::None)));
}
