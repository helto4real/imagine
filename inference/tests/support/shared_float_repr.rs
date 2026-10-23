use imagine_inference::patching::options::SourceNumber;
use serde_json::Value;
#[test]
fn original_float_repr_preserves_shortest_nearest_even_decimal_ties_and_asymmetric_power_of_two_roundtrips()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-float-repr.json")).unwrap();
    let cases = f["cases"].as_array().unwrap();
    assert!(cases.len() > 100_000);
    for c in cases {
        let bits = u64::from_str_radix(c[0].as_str().unwrap(), 16).unwrap();
        let value = SourceNumber::float(f64::from_bits(bits));
        assert_eq!(
            SourceNumber::get(&value)
                .unwrap()
                .repr_with_limit(4300)
                .unwrap(),
            c[1].as_str().unwrap(),
            "float bits {bits:016x}"
        );
    }
}
