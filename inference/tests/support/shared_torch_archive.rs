use imagine_inference::{
    checkpoint::{storage_rebuild, torch_archive::Reader},
    native::{self, Error},
};
use serde_json::{Value, json};
use std::path::Path;
fn exact<T: std::fmt::Debug>(
    result: Result<T, Error>,
    expected: &Value,
    value: impl FnOnce(T) -> Value,
) {
    if expected["error"].is_string() {
        let error = result.expect_err("original archive failure");
        assert!(
            error.is_reference(expected["error"].as_str().unwrap()),
            "{error:?}: {expected}"
        );
        assert_eq!(error.to_string(), expected["message"].as_str().unwrap());
    } else {
        assert_eq!(value(result.unwrap()), expected["value"]);
    }
}
#[test]
fn original_torch_zip_reader_preserves_records_order_bytes_offsets_storage_size_errors_and_independent_allocation_lifetime()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-torch-archive.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 11);
    let path = std::env::temp_dir().join(format!(
        "imagine-stage3-own-archive-{}.pt",
        std::process::id()
    ));
    struct Cleanup<'a>(&'a Path);
    impl Drop for Cleanup<'_> {
        fn drop(&mut self) {
            std::fs::remove_file(self.0).unwrap();
        }
    }
    for case in fixture["cases"].as_array().unwrap() {
        std::fs::write(&path, super::exact_hex(&case["archive_hex"])).unwrap();
        let _cleanup = Cleanup(&path);
        let result = Reader::open(&path);
        if case["error"].is_string() {
            let error = result.err().expect("original archive open failure");
            assert!(
                error.is_reference(case["error"].as_str().unwrap()),
                "{error:?}: {case}"
            );
            assert_eq!(error.to_string(), case["message"].as_str().unwrap());
            continue;
        }
        let reader = result.unwrap_or_else(|error| panic!("{case}: {error:?}"));
        assert_eq!(json!(reader.records().unwrap()), case["records"]);
        for query in case["queries"].as_array().unwrap() {
            let key = query["key"].as_str().unwrap();
            exact(reader.has_record(key), &query["has"], |v| json!(v));
            exact(reader.record(key), &query["data"], |v| {
                json!(v.iter().map(|b| format!("{b:02x}")).collect::<String>())
            });
            exact(reader.record_offset(key), &query["offset"], |v| json!(v));
        }
        for item in case["storages"].as_array().unwrap() {
            let result = reader.storage(
                item["key"].as_str().unwrap(),
                item["count"].as_u64().unwrap() as usize,
                item["code"].as_i64().unwrap() as i32,
            );
            if item["error"].is_string() {
                exact(result, item, |_| Value::Null);
                continue;
            }
            let tensor = result.unwrap();
            let second = reader
                .storage(
                    item["key"].as_str().unwrap(),
                    item["count"].as_u64().unwrap() as usize,
                    item["code"].as_i64().unwrap() as i32,
                )
                .unwrap();
            assert_eq!(
                native::storage_info(&tensor).unwrap().0
                    == native::storage_info(&second).unwrap().0,
                item["repeated_storage_alias"].as_bool().unwrap()
            );
            drop(second);
            assert_eq!(
                imagine_inference::checkpoint::torch_archive::storage_resizable(&tensor).unwrap(),
                item["resizable"].as_bool().unwrap()
            );
            assert_eq!(json!(tensor.size()), item["shape"]);
            assert_eq!(json!(tensor.stride()), item["stride"]);
            let nbytes = native::storage_info(&tensor).unwrap().1;
            assert_eq!(json!(nbytes), item["bytes"]);
            let view = storage_rebuild::tensor(&tensor, 0, 0, &[nbytes as i64], &[1]).unwrap();
            assert_eq!(super::bits(&view), super::exact_hex(&item["payload_hex"]));
            let weak = native::WeakStorage::new(&tensor).unwrap();
            drop(view);
            drop(tensor);
            assert_eq!(
                weak.expired(),
                item["expired_after_output_release"].as_bool().unwrap()
            );
        }
        let bytes = reader.record("data/0").unwrap();
        let held = reader.storage("data/0", bytes.len(), 0).unwrap();
        drop(reader);
        let nbytes = native::storage_info(&held).unwrap().1;
        let view = storage_rebuild::tensor(&held, 0, 0, &[nbytes as i64], &[1]).unwrap();
        assert_eq!(
            super::bits(&view),
            super::exact_hex(&case["held_payload_hex"])
        );
    }
}
