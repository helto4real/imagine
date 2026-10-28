//! Mask2Former against the locked transformers source: a bounded synthetic model on
//! CPU and CUDA, Mask2FormerImageProcessor on synthetic images and panoptic segments
//! from crafted logits.
use super::{
    binding_tests::{check, restore},
    depth_anything_tests::{bytes, image},
    qwen_text_encoder_tests::data,
};
use imagine_inference::{
    execution_value::Value as V,
    models::{mask2former as m, qwen_image21::Pillow},
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
use tch::{Device, Kind, Tensor};
fn fixture() -> Value {
    let path = std::env::var("IMAGINE_MASK2FORMER_FIXTURE").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/reference-20261027/mask2former.json"
        )
        .into()
    });
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}
#[test]
fn mask2former_synthetic_model_matches_transformers() {
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
        let net = m::Mask2Former::new(
            m::Mask2FormerConfig {
                embed: 16,
                depths: vec![2, 2, 2, 2],
                swin_heads: vec![1, 2, 4, 8],
                window: 4,
                patch: 4,
                feature: 32,
                encoder_layers: 2,
                heads: 4,
                decoder_layers: 4,
            },
            weights,
        );
        for call in model["calls"].as_array().unwrap() {
            let x = restore(&call["input"], dev).as_tensor().unwrap();
            let out = net.forward(&x).unwrap();
            let note = format!("{}/{:?}", model["device"], x.size());
            check(
                &V::tensor(out.class_logits),
                &call["class_logits"],
                dev,
                &format!("{note} classes"),
            );
            check(
                &V::tensor(out.mask_logits),
                &call["mask_logits"],
                dev,
                &format!("{note} masks"),
            );
            count += 1;
        }
    }
    assert!(count >= 3);
}
#[test]
fn mask2former_processor_matches_image_processor() {
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
        let pixels = m::segmentation_pixels(&pillow, &image(h, w, o), h, w).unwrap();
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
/// Segments as the reference lists them, the score as float32 bits.
pub(super) fn segments(p: &m::Panoptic) -> Value {
    p.segments
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "label": s.label,
                "fused": s.fused,
                "score_bits": s.score.to_bits(),
            })
        })
        .collect()
}
#[test]
fn mask2former_panoptic_segments_match_reference() {
    let (queries, labels) = (12, 6);
    let class = data(&[queries, labels + 1], Device::Cpu, Kind::Float, 5)
        .f_mul_scalar(4)
        .unwrap();
    for (q, label) in [0, 1, 4, 4, 2, 5, 6, 3, 5, 1, 4, 0].into_iter().enumerate() {
        let boost = if q == 7 { 1. } else { 6. };
        let _ = class.get(q as i64).get(label).f_add_scalar_(boost).unwrap();
    }
    let yy = Tensor::arange(10, (Kind::Float, Device::Cpu)).view([10, 1]);
    let xx = Tensor::arange(14, (Kind::Float, Device::Cpu)).view([1, 14]);
    let rows = [(0, 4), (4, 7), (7, 10)];
    let cols = [(0, 4), (4, 7), (7, 11), (11, 14)];
    let masks = (0..queries)
        .map(|q| {
            let ((y0, y1), (x0, x1)) = if q == 11 {
                ((0, 5), (0, 14))
            } else {
                (rows[(q / 4) as usize], cols[(q % 4) as usize])
            };
            let inside = yy
                .ge(y0)
                .logical_and(&yy.lt(y1))
                .logical_and(&xx.ge(x0))
                .logical_and(&xx.lt(x1))
                .to_kind(Kind::Float);
            let wave = ((&xx * 0.37) * (q % 4 + 1) + &yy * 0.53 + q).sin() * 0.5;
            inside * if q == 11 { 2. } else { 10. } - 5. + wave
        })
        .collect::<Vec<_>>();
    let masks = Tensor::stack(&masks, 0);
    let f = fixture();
    for case in f["crafted"].as_array().unwrap() {
        let (h, w) = (
            case["height"].as_i64().unwrap(),
            case["width"].as_i64().unwrap(),
        );
        let p = m::panoptic(
            &class,
            &masks,
            h,
            w,
            case["threshold"].as_f64().unwrap(),
            case["overlap"].as_f64().unwrap(),
            |label| label >= 3,
        )
        .unwrap();
        let note = format!("{h}x{w}");
        check(
            &V::tensor(p.segmentation.shallow_clone()),
            &case["segmentation"],
            Device::Cpu,
            &note,
        );
        assert_eq!(segments(&p), case["segments"], "{note}");
    }
}
#[test]
fn mask2former_region_colors_mark_unassigned_black() {
    let ids = Tensor::from_slice(&[0i32, 1, 2, 3, 0, 2]).view([2, 3]);
    let colors = m::region_colors(&ids, &[[10, 20, 30], [40, 50, 60]]).unwrap();
    assert_eq!(
        bytes(&colors),
        [
            0, 0, 0, 10, 20, 30, 40, 50, 60, 10, 20, 30, 0, 0, 0, 40, 50, 60
        ]
    );
}
