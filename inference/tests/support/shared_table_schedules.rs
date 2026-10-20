use super::{
    binding_tests::{check, reference_error},
    kitchen,
};
use imagine_inference::{execution_value::Value as V, sampling::table_schedules as s};
use serde_json::Value;
use tch::Device;
#[test]
fn ays_gits_tables_rounding_and_numpy_float64_interpolation_match_original() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-table-schedules.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let y = if c["name"] == "ays" {
            s::align_your_steps(
                c["key"].as_str().unwrap(),
                c["steps"].as_i64().unwrap(),
                c["denoise"].as_f64().unwrap(),
            )
        } else {
            s::gits(
                c["key"].as_f64().unwrap(),
                c["steps"].as_i64().unwrap(),
                c["denoise"].as_f64().unwrap(),
            )
        };
        if c["error"].is_string() {
            reference_error(
                y.err()
                    .unwrap_or_else(|| panic!("Expected source error {c}")),
                c,
            );
        } else {
            check(
                &V::tensor(y.unwrap()),
                &c["output"],
                Device::Cpu,
                &format!("{c}"),
            );
        }
    }
    for c in fixture["interpolations"].as_array().unwrap() {
        let values = if let Some(bits) = c["values_bits"].as_array() {
            bits.iter()
                .map(|v| f64::from_bits(v.as_u64().unwrap()))
                .collect::<Vec<_>>()
        } else {
            c["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>()
        };
        let y = s::loglinear(&values, c["length"].as_i64().unwrap());
        if c["error"].is_string() {
            reference_error(
                y.err()
                    .unwrap_or_else(|| panic!("Expected source error {c}")),
                c,
            );
        } else {
            let y = y.unwrap();
            let bytes = super::bits(&y);
            let actual = bytes
                .chunks_exact(8)
                .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                serde_json::json!(actual),
                c["bits"],
                "interpolation bits {c}"
            );
            check(
                &V::tensor(y),
                &c["output"],
                Device::Cpu,
                &format!("Float64 interpolation {c}"),
            );
        }
    }
}
