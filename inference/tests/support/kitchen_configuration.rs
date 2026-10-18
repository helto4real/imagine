use super::{exact_hex, fixtures, kitchen};
use imagine_inference::{execution::Runtime, native::Error};
use std::ffi::OsString;
use std::os::unix::ffi::OsStringExt;
const NAMES: [&str; 4] = [
    "COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N",
    "COMFY_KITCHEN_DISABLE_CUTLASS",
    "COMFY_KITCHEN_W4A8_CHUNKED",
    "COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK",
];
struct Environment(Vec<Option<OsString>>);
impl Drop for Environment {
    fn drop(&mut self) {
        for (name, value) in NAMES.iter().zip(&self.0) {
            unsafe {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }
}
#[test]
fn original_configuration_parsing_preserves_python_digits_unbounded_consumed_bounds_flags_and_import_failures()
 {
    let _preserve = Environment(NAMES.iter().map(std::env::var_os).collect());
    let native = kitchen();
    let external = std::env::var("IMAGINE_KITCHEN_CONFIGURATION_FIXTURE")
        .ok()
        .map(|p| {
            serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p).unwrap()).unwrap()
        });
    let cases = external.as_ref().unwrap_or_else(|| fixtures())["kitchen_configuration"]
        .as_array()
        .unwrap();
    assert_eq!(cases.len(), 34);
    for c in cases {
        let value = c["raw_hex"]
            .as_str()
            .map(|_| OsString::from_vec(exact_hex(&c["raw_hex"])));
        for name in NAMES {
            unsafe {
                match &value {
                    Some(v) => std::env::set_var(name, v),
                    None => std::env::remove_var(name),
                }
            }
        }
        match Runtime::frozen_partial(native.clone()) {
            Ok(rt) => {
                assert!(
                    c.get("error").is_none(),
                    "Expected source import error: {c}"
                );
                let bound = rt.int4_weight_chunk_n.unwrap();
                assert_eq!(bound, c["bound"].as_i64().unwrap());
                assert_eq!(
                    rt.disable_cutlass_int8,
                    c["disable_cutlass"].as_bool().unwrap()
                );
                assert_eq!(rt.w4a8_chunked, c["w4_chunked"].as_bool().unwrap());
                assert_eq!(
                    rt.force_int4_int8_fallback,
                    c["force_fallback"].as_bool().unwrap()
                );
                for (row, expected) in [0, 1, 16, 4096, 1_i64 << 31, i64::MAX]
                    .iter()
                    .zip(c["consumed"].as_array().unwrap())
                {
                    assert_eq!((*row).min(bound), expected.as_i64().unwrap());
                }
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(class, c["error"].as_str().unwrap());
                assert_eq!(message, c["message"].as_str().unwrap());
            }
            Err(e) => panic!("Original configuration exception identity: {e}"),
        }
    }
}
