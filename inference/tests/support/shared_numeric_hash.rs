use imagine_inference::python_hash;
use serde_json::Value;
#[test]
fn original_numeric_hashes_preserve_arbitrary_width_signed_integers_all_float_exponents_and_complex_owner_nan_hashes()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-numeric-hash.json")).unwrap();
    assert_eq!(fixture["integer_bytes"].as_array().unwrap().len(), 764);
    assert_eq!(fixture["floats"].as_array().unwrap().len(), 20480);
    assert_eq!(fixture["complex"].as_array().unwrap().len(), 1168);
    for row in fixture["integer_bytes"].as_array().unwrap() {
        assert_eq!(
            python_hash::hash_signed_integer_bytes(&super::exact_hex(&row["bytes"])),
            row["hash"].as_i64().unwrap(),
            "{row}"
        );
    }
    let number = |v: &Value| f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap());
    for row in fixture["floats"].as_array().unwrap() {
        assert_eq!(
            python_hash::hash_float(number(&row["bits"]), row["owner_hash"].as_i64().unwrap()),
            row["hash"].as_i64().unwrap(),
            "{row}"
        );
    }
    for row in fixture["complex"].as_array().unwrap() {
        assert_eq!(
            python_hash::hash_complex(
                number(&row["real"]),
                number(&row["imaginary"]),
                row["owner_hash"].as_i64().unwrap()
            ),
            row["hash"].as_i64().unwrap(),
            "{row}"
        );
    }
}
