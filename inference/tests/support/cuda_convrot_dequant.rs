use super::{binding_tests::check, kind, kitchen};
use imagine_inference::{convrot, execution_value::Value as V};
use serde_json::Value;
use tch::{Device, Kind, Tensor};
#[test]
fn cuda_original_convrot_wide_dequantization_matches() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/cuda-convrot-dequant.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let width = case["width"].as_i64().unwrap();
        let q = Tensor::arange(4 * width, (Kind::Int64, Device::Cuda(0)))
            .f_remainder(255)
            .unwrap()
            .f_sub_scalar(127)
            .unwrap()
            .f_to_kind(Kind::Int8)
            .unwrap()
            .f_reshape([4, width])
            .unwrap();
        let scale = Tensor::arange_start(1, 5, (Kind::Float, Device::Cuda(0)))
            .f_mul_scalar(0.03125)
            .unwrap();
        let output = unsafe {
            convrot::dequantize_weight(
                &native,
                &q,
                &scale,
                case["group"].as_i64().unwrap(),
                kind(case["dtype"].as_str().unwrap()),
            )
        }
        .unwrap();
        check(
            &V::tensor(output),
            &case["output"],
            Device::Cuda(0),
            "wide ConvRot dequantization",
        );
    }
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 24);
}
