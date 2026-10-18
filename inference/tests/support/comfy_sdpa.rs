use super::{device, kind, kitchen};
use imagine_inference::native::{Error, Kitchen, SdpaOptions};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tch::{Device, Kind, Tensor};
struct Reset(Kitchen, [i64; 10], [i64; 3]);
impl Drop for Reset {
    fn drop(&mut self) {
        self.0.set_sdpa_state(self.1).unwrap();
        self.0.set_comfy_math_state(self.2).unwrap();
    }
}
fn stamp(t: &Tensor, expected: &Value, note: &str) {
    let bytes = t
        .f_to_device(Device::Cpu)
        .unwrap()
        .f_contiguous()
        .unwrap()
        .f_reshape([-1])
        .unwrap()
        .f_view_dtype(Kind::Uint8)
        .unwrap();
    let mut data = vec![0; bytes.numel()];
    let len = data.len();
    bytes.f_copy_data_u8(&mut data, len).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&data)),
        expected["sha256"].as_str().unwrap(),
        "{note}/exact bytes"
    );
    assert_eq!(
        serde_json::json!(t.size()),
        expected["shape"],
        "{note}/shape"
    );
    assert_eq!(
        serde_json::json!(t.stride()),
        expected["strides"],
        "{note}/strides"
    );
    assert_eq!(t.kind(), kind(expected["dtype"].as_str().unwrap()));
    assert_eq!(t.device(), device(expected["device"].as_str().unwrap()));
    assert_eq!(
        t.requires_grad(),
        expected["requires_grad"].as_bool().unwrap()
    );
}
fn data(length: i64, heads: i64, dim: i64, offset: i64, device: Device, kind: Kind) -> Tensor {
    Tensor::f_arange(length * heads * dim, (Kind::Float, device))
        .unwrap()
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .f_reshape([1, length, heads, dim])
        .unwrap()
        .f_to_kind(kind)
        .unwrap()
        .f_transpose(1, 2)
        .unwrap()
}
#[test]
fn qwen_image21_original_comfy_sdpa_small_large_gqa_and_backend_state_match() {
    let native = kitchen();
    let _reset = Reset(
        native.clone(),
        native.sdpa_state().unwrap(),
        native.comfy_math_state().unwrap(),
    );
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/comfy-sdpa.json")).unwrap();
    for (index, c) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let mut math = native.comfy_math_state().unwrap();
        math[0] = i64::from(c["reduction_math"].as_bool().unwrap());
        native.set_comfy_math_state(math).unwrap();
        let dev = device(c["device"].as_str().unwrap());
        let dt = kind(c["dtype"].as_str().unwrap());
        let length = c["length"].as_i64().unwrap();
        let heads = c["heads"].as_i64().unwrap();
        let kv = c["kv_heads"].as_i64().unwrap();
        let dim = c["head_dim"].as_i64().unwrap();
        let q = data(length, heads, dim, 3, dev, dt);
        let k = data(length, kv, dim, 7, dev, dt);
        let v = data(length, kv, dim, 11, dev, dt);
        for (t, e) in [&q, &k, &v]
            .into_iter()
            .zip(c["inputs"].as_array().unwrap())
        {
            stamp(t, e, &format!("{index}/input"));
        }
        let mask = match c["mask"].as_str().unwrap() {
            "none" => None,
            name => {
                let mask = Tensor::f_ones([length, length], (Kind::Bool, dev))
                    .unwrap()
                    .f_tril(0)
                    .unwrap();
                Some(if name == "bool" {
                    mask
                } else {
                    Tensor::f_zeros([1, 1, length, length], (Kind::Float, dev))
                        .unwrap()
                        .f_masked_fill_(&mask.f_logical_not().unwrap(), f64::NEG_INFINITY)
                        .unwrap()
                })
            }
        };
        if let Some(mask) = &mask {
            stamp(mask, &c["mask_stamp"], &format!("{index}/mask"));
        }
        let initial = c["initial"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        native.set_sdpa_state(initial).unwrap();
        let output = native.comfy_sdpa(
            &q,
            &k,
            &v,
            SdpaOptions {
                mask: mask.as_ref(),
                scale: None,
                enable_gqa: heads != kv,
                priority_wrapper: c["priority_wrapper"].as_bool().unwrap(),
            },
        );
        if let Some(expected) = c["error"].as_str() {
            match output
                .err()
                .unwrap_or_else(|| panic!("{index}: expected {expected}"))
            {
                Error::ReferenceException { class, message } => {
                    assert_eq!(class, expected, "{index}");
                    assert_eq!(message, c["message"].as_str().unwrap(), "{index}");
                }
                other => panic!("{index}: {other:?}"),
            }
        } else {
            stamp(
                &output.unwrap_or_else(|e| panic!("{index}: {e}")),
                &c["output"],
                &format!("{index}/output"),
            );
        }
        assert_eq!(
            serde_json::json!(native.sdpa_state().unwrap()),
            c["post_state"],
            "{index}/finally state"
        );
    }
}
