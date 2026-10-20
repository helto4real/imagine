use imagine_inference::{
    checkpoint::{
        Aimdo,
        component::{self, StorageAnnotations},
    },
    execution::Runtime,
    execution_value::Value as V,
    native,
    patching::options::{Options, SourceText},
};
use serde_json::{Value, json};
use std::{path::PathBuf, rc::Rc, sync::Arc};
fn names(value: &Options) -> Vec<String> {
    let Options::Dict(values) = value else {
        panic!("native component dictionary")
    };
    values
        .borrow()
        .iter()
        .map(|(key, _)| key.to_owned())
        .collect()
}
fn bytes(value: &Options) -> Vec<u8> {
    super::bits(&value.snapshot().unwrap().as_tensor().unwrap())
}
#[test]
fn selected_av_component_metadata_preserves_original_precedence_order_custom_operations_and_all_weight_owners_without_allocating_weights()
 {
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-quant-metadata.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 12);
    for case in fixture["cases"].as_array().unwrap() {
        let weights = Options::dictionary(
            case["names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| {
                    let name = name.as_str().unwrap();
                    let value = if case["initial_quant"][name].is_string() {
                        Options::Atom(V::tensor(tch::Tensor::from_slice(&super::exact_hex(
                            &case["initial_quant"][name],
                        ))))
                    } else {
                        Options::Object(Rc::new(()))
                    };
                    (name.to_owned(), value)
                })
                .collect(),
        );
        let metadata = Options::dictionary(
            case["metadata_keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(|name| {
                    let name = name.as_str().unwrap();
                    let value = if name == "_quantization_metadata" {
                        SourceText::from_utf8(case["quantization_metadata"].as_str().unwrap())
                    } else {
                        Options::Object(Rc::new(()))
                    };
                    (name.to_owned(), value)
                })
                .collect(),
        );
        let old = names(&weights)
            .iter()
            .map(|key| (key.clone(), weights.get(key).unwrap().unwrap()))
            .collect::<Vec<_>>();
        let custom = match case["custom_operations"].as_str().unwrap() {
            "none" => Options::Atom(V::None),
            "false" => Options::Atom(V::Bool(false)),
            "opaque" => Options::Object(Rc::new(())),
            _ => unreachable!(),
        };
        let output =
            unsafe { component::normalize(&rt, weights.clone(), metadata.clone(), &custom) }
                .unwrap();
        assert_eq!(
            output.weights.same_object(&weights),
            case["root_same"].as_bool().unwrap()
        );
        assert_eq!(
            output.metadata.same_object(&metadata),
            case["metadata_same"].as_bool().unwrap()
        );
        assert_eq!(json!(names(&output.weights)), case["output_keys"]);
        for (key, initial) in old {
            let value = output.weights.get(&key).unwrap().unwrap();
            let changed = case["changed_quant"]
                .as_array()
                .unwrap()
                .iter()
                .any(|name| name == &key);
            assert_eq!(
                value.same_object(&initial),
                !changed,
                "{key}: {}",
                case["custom_operations"]
            );
            if case["output_quant"][&key].is_string() {
                assert_eq!(bytes(&value), super::exact_hex(&case["output_quant"][&key]));
            }
        }
        for key in names(&metadata) {
            assert!(
                output
                    .metadata
                    .get(&key)
                    .unwrap()
                    .unwrap()
                    .same_object(&metadata.get(&key).unwrap().unwrap())
            );
        }
    }
}
#[test]
fn native_storage_source_annotations_follow_views_reassignment_empty_storage_and_physical_release_without_owning_data()
 {
    let _native = super::kitchen();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/native-av-quant-metadata.json")).unwrap();
    for case in fixture["storage_annotations"].as_array().unwrap() {
        let context = StorageAnnotations::default();
        let shape = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect::<Vec<_>>();
        let base = tch::Tensor::zeros(shape.as_slice(), (tch::Kind::Float, tch::Device::Cpu));
        let view = base.view([-1]);
        let same = base.to_kind(tch::Kind::Float);
        let clone = base.copy();
        let cast = base.to_kind(tch::Kind::Int64);
        let path_a = Arc::new(PathBuf::from(case["path_a"].as_str().unwrap()));
        let path_b = Arc::new(PathBuf::from(case["path_b"].as_str().unwrap()));
        context.annotate(&base, path_a.clone()).unwrap();
        context.annotate(&view, path_a).unwrap();
        assert_eq!(
            Arc::ptr_eq(
                &context.source_path(&base).unwrap().unwrap(),
                &context.source_path(&view).unwrap().unwrap()
            ),
            case["same_path_object"].as_bool().unwrap()
        );
        for (key, value) in [
            ("base", &base),
            ("view", &view),
            ("same", &same),
            ("clone", &clone),
            ("cast", &cast),
        ] {
            assert_eq!(
                json!(
                    context
                        .source_path(value)
                        .unwrap()
                        .map(|p| p.to_string_lossy().into_owned())
                ),
                case["before"][key]
            );
        }
        context.annotate(&view, path_b).unwrap();
        for (key, value) in [
            ("base", &base),
            ("view", &view),
            ("same", &same),
            ("clone", &clone),
            ("cast", &cast),
        ] {
            assert_eq!(
                json!(
                    context
                        .source_path(value)
                        .unwrap()
                        .map(|p| p.to_string_lossy().into_owned())
                ),
                case["after"][key]
            );
        }
        let weak = native::WeakStorage::new(&base).unwrap();
        drop(base);
        drop(same);
        assert_eq!(!weak.expired(), case["alive_with_view"].as_bool().unwrap());
        drop(view);
        assert_eq!(
            weak.expired(),
            case["expired_after_release"].as_bool().unwrap()
        );
    }
}
#[test]
fn tiny_aimdo_component_keeps_ordered_metadata_normalizes_legacy_markers_and_releases_mapped_storage()
 {
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    let aimdo = unsafe {
        Aimdo::load(std::path::Path::new(
            &std::env::var("IMAGINE_AIMDO_LIBRARY").unwrap(),
        ))
        .unwrap()
    };
    let path = std::env::temp_dir().join(format!(
        "imagine-own-av-component-{}.safetensors",
        std::process::id()
    ));
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }
    let _cleanup = Cleanup(path.clone());
    let old = br#"{"format":"int8_tensorwise","per_row":true}"#;
    let legacy = r#"{"layers":{"layer":{"format":"int8_tensorwise"}}}"#;
    let header = format!(
        r#"{{"__metadata__":{{"z":"own","_quantization_metadata":{},"a":"own"}},"layer.weight":{{"dtype":"I8","shape":[2,2],"data_offsets":[0,4]}},"layer.comfy_quant":{{"dtype":"U8","shape":[{}],"data_offsets":[4,{}]}}}}"#,
        serde_json::to_string(legacy).unwrap(),
        old.len(),
        4 + old.len()
    );
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend([1u8, 254, 3, 252]);
    file.extend(old);
    std::fs::write(&path, file).unwrap();
    let annotations = StorageAnnotations::default();
    let output =
        unsafe { component::load_aimdo(&rt, &aimdo, &path, &Options::Atom(V::None), &annotations) }
            .unwrap();
    assert_eq!(
        names(&output.metadata),
        ["z", "_quantization_metadata", "a"]
    );
    let first = output.metadata.get("z").unwrap().unwrap();
    assert!(SourceText::get(&first).is_some());
    assert!(first.same_object(&output.metadata.get("z").unwrap().unwrap()));
    let weight = output
        .weights
        .get("layer.weight")
        .unwrap()
        .unwrap()
        .snapshot()
        .unwrap()
        .as_tensor()
        .unwrap();
    assert_eq!(super::bits(&weight), [1u8, 254, 3, 252]);
    let quant = output.weights.get("layer.comfy_quant").unwrap().unwrap();
    assert_eq!(bytes(&quant), br#"{"format": "int8_tensorwise"}"#);
    assert!(
        annotations
            .source_path(&quant.snapshot().unwrap().as_tensor().unwrap())
            .unwrap()
            .is_none()
    );
    let canonical = std::fs::canonicalize(&path).unwrap();
    assert_eq!(
        annotations.source_path(&weight).unwrap().unwrap().as_ref(),
        &canonical
    );
    let alias = weight.view([4]);
    let weak = native::WeakStorage::new(&weight).unwrap();
    drop(output);
    drop(weight);
    assert!(!weak.expired());
    assert_eq!(
        annotations.source_path(&alias).unwrap().unwrap().as_ref(),
        &canonical
    );
    drop(alias);
    assert!(weak.expired());
}
