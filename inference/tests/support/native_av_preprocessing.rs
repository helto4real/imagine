use imagine_inference::{
    execution_value::Value as V, models::minimax_h3::preprocessing as h3, native,
};
use serde_json::Value;
use tch::{Device, Kind, Tensor};
fn kind(name: &str) -> Kind {
    match name {
        "float32" => Kind::Float,
        "float64" => Kind::Double,
        "bfloat16" => Kind::BFloat16,
        "int64" => Kind::Int64,
        _ => panic!("unexpected fixture dtype"),
    }
}
fn input(v: &Value) -> Tensor {
    let shape = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect::<Vec<_>>();
    Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, Device::Cpu))
        .reshape(&shape)
        .g_div_scalar(127)
        .to_kind(kind(v["dtype"].as_str().unwrap()))
}
fn check(t: &Tensor, v: &Value) {
    assert_eq!(serde_json::json!(t.size()), v["shape"]);
    assert_eq!(t.kind(), kind(v["dtype"].as_str().unwrap()));
    assert_eq!(super::bits(t), super::exact_hex(&v["bytes"]));
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/native-av-preprocessing.json")).unwrap()
}
#[test]
fn h3_video_block_matches_original_two_frame_patches_and_default_dtype() {
    let native = super::kitchen();
    let old = native.default_dtype().unwrap();
    struct Restore<'a>(&'a native::Kitchen, Kind);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            self.0.set_default_dtype(self.1).unwrap();
        }
    }
    let _restore = Restore(&native, old);
    let fixture = fixture();
    let cases = fixture["video"].as_array().unwrap();
    assert_eq!(cases.len(), 24);
    for case in cases {
        native
            .set_default_dtype(kind(case["default_dtype"].as_str().unwrap()))
            .unwrap();
        let x = input(&case["input"]);
        check(&x, &case["input"]);
        let (out, grid) = h3::video_block(&x, &native).unwrap();
        check(&out, &case["output"]);
        check(&grid, &case["grid"]);
    }
}
#[test]
fn h3_empty_streams_preserve_original_five_token_frame_cycle_and_independent_video_storage() {
    let f = fixture();
    let cases = f["empty"].as_array().unwrap();
    assert_eq!(cases.len(), 39);
    for case in cases {
        let x = input(&case["input"]);
        let out = h3::empty_streams(&x).unwrap();
        for (tensor, expected) in out.iter().zip(case["outputs"].as_array().unwrap()) {
            check(&tensor.as_tensor().unwrap(), expected);
        }
        assert_eq!(
            native::storage_info(&out[0].as_tensor().unwrap())
                .unwrap()
                .0
                == native::storage_info(&x).unwrap().0,
            case["video_alias"].as_bool().unwrap()
        );
    }
}
#[test]
fn h3_packed_audio_scale_keeps_identity_shortcut_clone_slice_and_original_dtype_errors() {
    let f = fixture();
    let cases = f["scale"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for case in cases {
        let x = V::tensor(input(&case["input"]));
        let out = h3::scale_packed_audio(
            &x,
            &[2, case["cut"].as_i64().unwrap()],
            case["scale"].as_f64().unwrap(),
            &super::kitchen(),
        );
        if case["error"].is_object() {
            let error = out.err().expect("original in-place dtype error");
            assert_eq!(case["error"]["type"], "RuntimeError");
            match error {
                native::Error::ReferenceException { class, message } => {
                    assert_eq!(class, case["error"]["type"].as_str().unwrap());
                    assert_eq!(message, case["error"]["message"].as_str().unwrap());
                }
                error => panic!("{error:?}"),
            }
        } else {
            let out = out.unwrap();
            check(&out.as_tensor().unwrap(), &case["output"]);
            assert_eq!(out.same_object(&x), case["same"].as_bool().unwrap());
            assert_eq!(
                native::storage_info(&out.as_tensor().unwrap()).unwrap().0
                    == native::storage_info(&x.as_tensor().unwrap()).unwrap().0,
                case["same"].as_bool().unwrap()
            );
        }
        check(&x.as_tensor().unwrap(), &case["input_after"]);
    }
}
#[test]
fn h3_image_tags_match_original_widened_negative_empty_and_extreme_slices() {
    let f = fixture();
    let cases = f["tags"].as_array().unwrap();
    assert_eq!(cases.len(), 40);
    for case in cases {
        let spans = case["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| (v[0].as_i64().unwrap(), v[1].as_i64().unwrap()))
            .collect::<Vec<_>>();
        check(
            &h3::token_tags(case["length"].as_i64().unwrap(), &spans).unwrap(),
            &case["output"],
        );
    }
}
