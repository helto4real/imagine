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

#[test]
#[ignore = "Requires an explicitly reserved CUDA worker; injects an own pin refusal"]
fn cuda_pin_refusal_discard_does_not_poison_next_convrot_launch() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    cuda_original_convrot_wide_dequantization_matches();
    let own_host = Tensor::ones([1], (Kind::Uint8, Device::Cpu));
    // CUDA rejects a zero-byte registration before accessing the owned pointer.
    assert!(
        !unsafe { native.host_registration_raw(own_host.data_ptr() as usize, 0, true) }.unwrap()
    );
    imagine_inference::memory::exceptions::discard_native_cuda_async_error(
        &native,
        Device::Cuda(0),
    )
    .unwrap();
    cuda_original_convrot_wide_dequantization_matches();
    let [allocated, reserved, peak_allocated, peak_reserved] =
        native.cuda_allocator_stats(0).unwrap();
    assert!(peak_allocated >= allocated && peak_allocated > 0);
    assert!(peak_reserved >= reserved && peak_reserved >= peak_allocated);
}
