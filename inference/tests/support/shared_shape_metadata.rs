use imagine_inference::{execution_value::Value as V, patching as p};
use serde_json::Value;
use tch::{Device, Kind, Tensor};
#[test]
fn original_patch_shape_metadata_preserves_tuple_conversion_tensor_views_raw_values_offset_skips_diff_order_size_aliases_and_equality_errors()
 {
    let _library = super::kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-shape-metadata.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 2336);
    for case in fixture["cases"].as_array().unwrap() {
        let label = case["label"].as_str().unwrap();
        let device = if label.starts_with("cuda:") {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let initial: Vec<i64> = case["initial"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let initial_value = p::size_metadata(&initial);
        let reshape = super::binding_tests::restore(&case["reshape"], device);
        let mut lora = p::Patch::new(
            p::PatchValue::Adapter(p::Adapter {
                name: "lora".into(),
                weights: vec![V::None, V::None, V::None, V::None, V::None, reshape],
            }),
            1.,
        );
        let diff = || {
            p::Patch::new(
                p::PatchValue::Diff {
                    weight: V::tensor(Tensor::zeros([9, 8], (Kind::Float, Device::Cpu))),
                    pad: true,
                },
                1.,
            )
        };
        let patches = match case["mode"].as_str().unwrap() {
            "lora" => vec![lora],
            "offset" => {
                lora.offset = Some([0, 0, 1]);
                vec![lora]
            }
            "diff-first" => vec![diff(), lora],
            "diff-last" => vec![lora, diff()],
            _ => panic!("unknown source scenario"),
        };
        let result = p::calculate_shape_metadata(&patches, initial_value.clone());
        if case["error"].is_string() {
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("expected source conversion failure: {label}"),
            };
            super::binding_tests::reference_error(error, case);
            continue;
        }
        let output = result.unwrap_or_else(|error| panic!("{label}: {error:?}"));
        assert_eq!(
            output.same_object(&initial_value),
            case["initial_alias"].as_bool().unwrap(),
            "{label}"
        );
        let values = match &output {
            V::Tuple(values) => {
                assert_eq!(case["container"], "tuple", "{label}");
                values.clone()
            }
            V::NamedTuple(values) => {
                assert_eq!(case["container"], "torch.Size", "{label}");
                assert_eq!(values.class.name, "Size");
                values.fields.iter().map(|(_, v)| v.clone()).collect()
            }
            _ => panic!("original must return a tuple or Size"),
        };
        super::binding_tests::check(&V::Tuple(values), &case["output"], device, label);
        let comparison = p::shape_metadata_matches(&output, &initial);
        if case["comparison_error"].is_object() {
            super::binding_tests::reference_error(
                comparison.expect_err("source comparison failure"),
                &case["comparison_error"],
            );
        } else {
            assert_eq!(
                comparison.unwrap_or_else(|error| panic!("{label}: {error:?}")),
                case["matches_initial"].as_bool().unwrap(),
                "{label}"
            );
        }
    }
}
