use imagine_inference::{
    checkpoint::storage_rebuild,
    native::{self, Error},
};
use serde_json::Value;
use tch::{Device, Kind, Tensor};
fn output_bytes(value: &Tensor) -> Result<Vec<u8>, Error> {
    if value.numel() == 0 {
        return Ok(vec![]);
    }
    let cpu = value
        .f_detach()?
        .f_contiguous()?
        .f_reshape([-1])?
        .f_view_dtype(Kind::Uint8)?
        .f_to_device(Device::Cpu)?;
    let mut bytes = vec![0; cpu.numel()];
    let n = bytes.len();
    cpu.f_copy_data_u8(&mut bytes, n)?;
    Ok(bytes)
}
#[test]
fn original_storage_tensor_rebuild_preserves_twenty_one_dtypes_views_offsets_strides_storage_aliases_payloads_native_errors_and_physical_storage_release()
 {
    let library = super::kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-storage-rebuild.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 504);
    for case in fixture["cases"].as_array().unwrap() {
        let bytes: Vec<u8> = (0..256)
            .map(|i| (i % case["modulus"].as_u64().unwrap()) as u8)
            .collect();
        let source =
            Tensor::from_slice(&bytes).to_device(super::device(case["device"].as_str().unwrap()));
        let dimensions = |key: &str| {
            case[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
        };
        let result = storage_rebuild::tensor(
            &source,
            case["code"].as_i64().unwrap() as i32,
            case["offset"].as_i64().unwrap(),
            &dimensions("shape"),
            &dimensions("stride"),
        );
        if case["error"].is_string() {
            super::binding_tests::reference_error(
                result.expect_err("original reconstruction failure"),
                case,
            );
            assert_eq!(
                super::bits(&source),
                bytes,
                "source buffer unchanged on error: {case}"
            );
            continue;
        }
        let output = result.unwrap_or_else(|error| panic!("{case}: {error:?}"));
        assert_eq!(serde_json::json!(output.size()), case["output_shape"]);
        assert_eq!(serde_json::json!(output.stride()), case["output_stride"]);
        assert_eq!(
            output.requires_grad(),
            case["requires_grad"].as_bool().unwrap()
        );
        assert_eq!(
            library.tensor_type(&output).unwrap().scalar_type,
            case["code"].as_i64().unwrap()
        );
        let storage = library.tensor_storage(&output).unwrap();
        assert_eq!(storage.offset, case["output_offset"].as_i64().unwrap());
        assert_eq!(
            storage.element_bytes,
            case["element_bytes"].as_i64().unwrap()
        );
        assert_eq!(
            native::storage_info(&output).unwrap().0 == native::storage_info(&source).unwrap().0,
            case["storage_alias"].as_bool().unwrap()
        );
        assert_eq!(native::storage_info(&output).unwrap().1, 256);
        let observed = output_bytes(&output);
        if case["observation_error"].is_object() {
            super::binding_tests::reference_error(
                observed.expect_err("same unsupported byte observer"),
                &case["observation_error"],
            );
        } else {
            assert_eq!(
                observed.unwrap(),
                super::exact_hex(&case["payload_hex"]),
                "{case}"
            );
        }
        assert_eq!(
            super::bits(&source),
            bytes,
            "source buffer unchanged on success"
        );
        let weak = native::WeakStorage::new(&output).unwrap();
        drop(source);
        assert_eq!(
            !weak.expired(),
            case["alive_after_source_release"].as_bool().unwrap()
        );
        drop(output);
        assert_eq!(
            weak.expired(),
            case["expired_after_output_release"].as_bool().unwrap()
        );
    }
}
