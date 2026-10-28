//! Depth Anything V2 against the locked transformers source: a bounded synthetic
//! model on CPU and CUDA and DPTImageProcessor on synthetic images.
use super::{
    binding_tests::{check, restore},
    qwen_text_encoder_tests::data,
};
use imagine_inference::{
    execution_value::Value as V,
    models::{depth_anything as d, qwen_image21::Pillow},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
use tch::{Device, Kind, Tensor};
fn fixture() -> Value {
    let path = std::env::var("IMAGINE_DEPTH_ANYTHING_FIXTURE").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/reference-20261027/depth-anything.json"
        )
        .into()
    });
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}
/// The reference's integer test image `[height, width, 3]`.
pub(super) fn image(height: i64, width: i64, offset: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            out.push((x * 255 / (width - 1).max(1)) as u8);
            out.push((y * 255 / (height - 1).max(1)) as u8);
            out.push((((x + y) * 7 + offset) % 256) as u8);
        }
    }
    out
}
pub(super) fn bytes(t: &Tensor) -> Vec<u8> {
    let t = t.contiguous().to_device(Device::Cpu);
    let mut out = vec![0u8; t.numel() * t.kind().elt_size_in_bytes()];
    t.f_copy_data_u8(&mut out, t.numel()).unwrap();
    out
}
#[test]
fn depth_anything_synthetic_model_matches_transformers() {
    // CUDA models need the native library, which loads torch's CUDA backend.
    let _native = super::kitchen();
    let _threads = super::moss_dit_tests::Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let _grad = tch::no_grad_guard();
    let f = fixture();
    let mut count = 0;
    for model in f["models"].as_array().unwrap() {
        let dev = if model["device"] == "cpu" {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let weights = model["factories"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, f)| {
                let shape = f["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap())
                    .collect::<Vec<_>>();
                let mut x = data(&shape, dev, Kind::Float, f["offset"].as_i64().unwrap())
                    .f_mul_scalar(0.125)
                    .unwrap();
                if f["plus_one"] == true {
                    x = x.f_add_scalar(1.).unwrap();
                }
                (name.clone(), x)
            })
            .collect::<BTreeMap<_, _>>();
        let attention = match model["attention"].as_str().unwrap() {
            "sdpa" => d::DinoAttention::Sdpa,
            _ => d::DinoAttention::Eager,
        };
        let net = d::DepthAnything::new(
            d::DepthAnythingConfig {
                hidden: 32,
                heads: 2,
                layers: 4,
                patch: 14,
                out_indices: vec![1, 2, 3, 4],
                neck: vec![8, 16, 24, 32],
                reassemble: vec![
                    d::Reassemble::Up(4),
                    d::Reassemble::Up(2),
                    d::Reassemble::Same,
                    d::Reassemble::Down,
                ],
                fusion: 16,
                head_hidden: 8,
                eps: 1e-6,
                attention,
            },
            weights,
        );
        for call in model["calls"].as_array().unwrap() {
            let x = restore(&call["input"], dev).as_tensor().unwrap();
            let out = net.forward(&x).unwrap();
            check(
                &V::tensor(out),
                &call["output"],
                dev,
                &format!("{}/{:?}", model["device"], x.size()),
            );
            count += 1;
        }
    }
    assert!(count >= 3);
}
#[test]
fn depth_anything_processor_matches_dpt_image_processor() {
    let pillow =
        unsafe { Pillow::open(Path::new(&std::env::var("IMAGINE_PILLOW_LIBRARY").unwrap())) }
            .unwrap();
    let f = fixture();
    for case in f["processed"].as_array().unwrap() {
        let (h, w, o) = (
            case["height"].as_i64().unwrap(),
            case["width"].as_i64().unwrap(),
            case["offset"].as_i64().unwrap(),
        );
        let pixels = d::depth_pixels(&pillow, &image(h, w, o), h, w).unwrap();
        let expected = &case["pixel_values"];
        assert_eq!(
            serde_json::json!(pixels.size()),
            expected["shape"],
            "{h}x{w}"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes(&pixels))),
            expected["payload_sha256"].as_str().unwrap(),
            "{h}x{w}"
        );
    }
}
