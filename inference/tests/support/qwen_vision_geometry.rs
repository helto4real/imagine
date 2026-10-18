use super::{binding_tests::check, device, kind};
use imagine_inference::{execution_value::Value as V, models::qwen_image21 as q};
use serde_json::{Value, json};
use tch::{Device, Kind, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-vision-geometry.json"
    ))
    .unwrap()
}
fn data(shape: &[i64], dev: Device, kind: Kind, offset: i64) -> Tensor {
    Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev))
        .g_add_scalar(offset)
        .remainder(29)
        .g_mul_scalar(0.03125)
        .g_sub_scalar(0.40625)
        .reshape(shape)
        .to_kind(kind)
}
fn numbers(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap())
        .collect()
}
fn optional(t: Option<Tensor>) -> V {
    t.map(V::tensor).unwrap_or(V::None)
}
#[test]
fn qwen_image21_original_vision_patching_resize_normalization_and_first_image_match() {
    let _native = super::kitchen();
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let f = fixture();
    let cases = f["patches"].as_array().unwrap();
    assert_eq!(cases.len(), 42);
    for c in cases {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["dtype"].as_str().unwrap());
        let mut options = q::VisionPreprocess::default();
        let o = &c["options"];
        if let Some(n) = o["max_pixels"].as_i64() {
            options.max_pixels = n;
        }
        if let Some(n) = o["patch_size"].as_i64() {
            options.patch_size = n;
            options.mean = [0.48145466, 0.4578275, 0.40821073];
            options.std = [0.26862954, 0.26130258, 0.27577711];
        }
        options.bicubic = o["interpolation"] == "bicubic";
        let x = data(&numbers(&c["shape"]), dev, dtype, 3);
        let (patches, grid) = options.patches(&x).unwrap();
        let note = format!("patch {:?} {dtype:?} {dev:?}", c["shape"]);
        check(
            &V::Tuple(vec![V::tensor(patches), V::tensor(grid)]),
            &c["output"],
            dev,
            &note,
        );
        check(
            &V::tensor(x),
            &c["input_after"],
            dev,
            "input remains unchanged",
        );
    }
}
#[test]
fn qwen_image21_original_multimodal_positions_sorted_deepstack_and_mask_match() {
    let _native = super::kitchen();
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let f = fixture();
    let cases = f["positions"].as_array().unwrap();
    assert_eq!(cases.len(), 10);
    for c in cases {
        let dev = device(c["device"].as_str().unwrap());
        let seq = c["seq"].as_i64().unwrap();
        let info = c["specs"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let size = e[1].as_i64().unwrap();
                q::ImageEmbeddingInfo {
                    index: e[0].as_i64().unwrap(),
                    size,
                    grid: vec![numbers(&e[2]).try_into().unwrap()],
                    deepstack: (0..3)
                        .map(|d| data(&[size, 8], dev, Kind::Float, 7 + i as i64 + d))
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        check(
            &optional(q::image_position_ids(&info, seq, dev).unwrap()),
            &c["raw"],
            dev,
            "original incoming-order MRoPE",
        );
        check(
            &optional(q::image_position_ids(&info, seq, dev).unwrap()),
            &c["legacy"],
            dev,
            "legacy tensor grid",
        );
        let built = q::build_image_inputs(&info, seq, dev).unwrap();
        check(
            &V::Tuple(vec![
                optional(built.position_ids),
                optional(built.visual_mask),
                built
                    .deepstack
                    .map(|v| V::List(v.into_iter().map(V::tensor).collect()))
                    .unwrap_or(V::None),
            ]),
            &c["built"],
            dev,
            "sorted DeepStack and mask; incoming-order MRoPE",
        );
    }
}
#[test]
fn qwen_image21_original_image_spans_system_filter_keep_vision_and_slots_match() {
    let _native = super::kitchen();
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let f = fixture();
    let cases = f["filters"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for c in cases {
        let dev = device(c["device"].as_str().unwrap());
        let seq = c["seq"].as_i64().unwrap();
        let tokens = c["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                t.as_i64()
                    .map(q::PromptToken::Id)
                    .unwrap_or(q::PromptToken::Image(0))
            })
            .collect::<Vec<_>>();
        let spans = c["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s[0].as_i64().unwrap(), s[1].as_i64().unwrap()))
            .collect::<Vec<_>>();
        let x = data(&[1, seq, 8], dev, Kind::Float, 11);
        let mask = Tensor::ones([1, seq], (Kind::Int64, dev));
        if c["masked"] == true {
            let _ = mask.get(0).get(seq - 1).fill_(0);
        }
        let out = q::filter_image_conditioning(
            &x,
            &mask,
            &tokens,
            &spans,
            c["keep_vision"].as_bool().unwrap(),
        )
        .unwrap();
        let mut extra = vec![];
        if let Some(mask) = out.attention_mask {
            extra.push(("attention_mask".into(), V::tensor(mask)));
        }
        if !out.image_slots.is_empty() {
            extra.push((
                "image_slots".into(),
                V::List(out.image_slots.into_iter().map(V::Int).collect()),
            ));
        }
        check(
            &V::Tuple(vec![
                V::tensor(out.context),
                optional(out.pooled),
                V::Dict(extra),
            ]),
            &c["output"],
            dev,
            &format!("filter {}", json!(c)),
        );
    }
}
