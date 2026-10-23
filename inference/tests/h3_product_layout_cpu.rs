//! Isolated H3 CPU reference entry point; no Runtime, model loader or CUDA calls.
use serde_json::Value;
use tch::{Device, Kind, Tensor};
#[path = "support/h3_layout.rs"]
mod layout_reference;
fn exact_hex(value: &Value) -> Vec<u8> {
    let s = value.as_str().unwrap();
    assert_eq!(s.len() % 2, 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn bits(tensor: &Tensor) -> Vec<u8> {
    assert_eq!(tensor.device(), Device::Cpu);
    let cpu = tensor
        .detach()
        .contiguous()
        .reshape([-1])
        .view_dtype(Kind::Uint8);
    let mut bytes = vec![0; cpu.numel()];
    let len = bytes.len();
    cpu.copy_data_u8(&mut bytes, len);
    bytes
}

#[test]
fn vae_wrapper_preserves_black_gray_white_and_does_not_change_vision_input() {
    let source = Tensor::from_slice(&[0f32, 0.5, 1.]).reshape([1, 1, 1, 3]);
    let normalized = imagine_inference::h3_product::image::vae_input(&source, Kind::Float).unwrap();
    let mut got = vec![0f32; 3];
    normalized.copy_data(&mut got, 3);
    assert_eq!(got, [-1., 0., 1.]);
    let restored = (normalized + 1.) * 0.5;
    let mut values = vec![0f32; 3];
    restored.copy_data(&mut values, 3);
    assert_eq!(values, [0., 0.5, 1.]);
    source.copy_data(&mut values, 3);
    assert_eq!(values, [0., 0.5, 1.]);
}

#[test]
fn vae_wrapper_normalizes_in_float_before_casting_to_selected_encoder_dtype() {
    let source = Tensor::from_slice(&[0.5001f32, 0.123456, 0.876543]).reshape([1, 1, 1, 3]);
    let original = bits(&source);
    let encoded = imagine_inference::h3_product::image::vae_input(&source, Kind::Half).unwrap();
    let expected = (source.permute([0, 3, 1, 2]) * 2. - 1.).to_kind(Kind::Half);
    let premature = source.permute([0, 3, 1, 2]).to_kind(Kind::Half) * 2. - 1.;
    assert_eq!(encoded.kind(), Kind::Half);
    assert_eq!(encoded.size(), [1, 3, 1, 1]);
    assert_eq!(bits(&encoded), bits(&expected));
    assert_ne!(bits(&encoded), bits(&premature));
    assert_eq!(bits(&source), original);
}
