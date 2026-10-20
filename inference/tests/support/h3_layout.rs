use imagine_inference::models::minimax_h3::layout as h3;
use serde_json::{Value, json};
use tch::{Kind, Tensor};

fn kind(v: &Value) -> Kind {
    match v.as_str().unwrap() {
        "float32" => Kind::Float,
        "float64" => Kind::Double,
        "bfloat16" => Kind::BFloat16,
        "int64" => Kind::Int64,
        "bool" => Kind::Bool,
        other => panic!("Unexpected dtype {other}"),
    }
}

fn input(v: &Value) -> Tensor {
    let shape: Vec<i64> = serde_json::from_value(v["shape"].clone()).unwrap();
    Tensor::f_from_data_size(&super::exact_hex(&v["bytes"]), &shape, kind(&v["dtype"])).unwrap()
}

fn check(t: &Tensor, v: &Value) {
    assert_eq!(json!(t.size()), v["shape"]);
    assert_eq!(t.kind(), kind(&v["dtype"]));
    assert_eq!(super::bits(t), super::exact_hex(&v["bytes"]));
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/h3-layout.json")).unwrap()
}

#[test]
fn h3_packed_layout_matches_original_references_keyframes_time_grids_and_update_masks() {
    let f = fixture();
    let cases = f["layouts"].as_array().unwrap();
    assert_eq!(cases.len(), 84);
    for (i, case) in cases.iter().enumerate() {
        let shape = serde_json::from_value(case["shape"].clone()).unwrap();
        let keys: Vec<h3::Keyframe> = serde_json::from_value(case["keyframes"].clone()).unwrap();
        let refs: Vec<h3::ReferenceBlock> = serde_json::from_value(case["refs"].clone()).unwrap();
        let layout = h3::PackedLayout::new(shape, &keys, &refs).unwrap();
        assert_eq!(json!(layout.seq_len), case["seq_len"], "case {i}");
        assert_eq!(json!(layout.signature), case["signature"]);
        assert_eq!(
            json!(
                layout
                    .segments
                    .iter()
                    .map(|s| (s.start, s.stop, s.kind.name()))
                    .collect::<Vec<_>>()
            ),
            case["segments"]
        );
        for (name, tensor) in [
            ("position_ids", &layout.position_ids),
            ("img_pos", &layout.img_pos),
            ("img_update", &layout.img_update),
            ("audio_pos", &layout.audio_pos),
            ("audio_update", &layout.audio_update),
        ] {
            check(tensor, &case["tensors"][name]);
        }
    }
}

#[test]
fn h3_latent_patch_rows_and_stereo_audio_order_match_original() {
    let f = fixture();
    let cases = f["packing"].as_array().unwrap();
    assert_eq!(cases.len(), 18);
    for case in cases {
        let x = input(&case["input"]);
        let (rows, output) = if case["kind"] == "video" {
            let rows = h3::patchify_video(&x, [1, 2, 2]).unwrap();
            let s = x.size();
            let out =
                h3::unpatchify_video(&rows, [s[2], s[3] / 2, s[4] / 2], s[1], [1, 2, 2]).unwrap();
            (rows, out)
        } else {
            let rows = h3::pack_audio(&x).unwrap();
            let out = h3::unpack_audio(&rows, 2).unwrap();
            (rows, out)
        };
        check(&rows, &case["rows"]);
        check(&output, &case["output"]);
    }
}

#[test]
fn h3_denoise_mask_pooling_preserves_original_replicate_padding_and_generation_threshold() {
    let f = fixture();
    let cases = f["masks"].as_array().unwrap();
    assert_eq!(cases.len(), 36);
    for case in cases {
        let [t, h, w]: [i64; 3] = serde_json::from_value(case["target"].clone()).unwrap();
        let x = input(&case["input"]);
        let out = h3::mask_row_values(&x, t, h, w).unwrap();
        if case["output"].is_null() {
            assert!(out.is_none());
        } else {
            check(&out.unwrap(), &case["output"]);
        }
        check(&x, &case["input"]);
    }
}
