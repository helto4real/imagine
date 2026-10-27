//! MOSS-Image color/depth control adapters against ComfyUI 426f5eea: the adapter
//! conversion, its loading through the MOSS key map and shape-changing set patches.
use super::{
    kitchen,
    qwen_lora_tests::{assert_patch, run_math, value},
};
use imagine_inference::{models::moss_image as m, patching as p};
use serde_json::{Value, json};
use tch::Device;
fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../fixtures/reference-20261027/moss-control-lora.json"
    ))
    .unwrap()
}
#[test]
fn moss_control_set_patches_replace_the_weight_shape_like_426f5eea() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let f = fixture();
    assert_eq!(f["math"].as_array().unwrap().len(), 16);
    run_math(&native, &f);
    for c in f["math"].as_array().unwrap() {
        let dev = if c["weight"]["device"].as_str().unwrap().starts_with("cuda") {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let patches = c["patches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                p::Patch::new(
                    super::qwen_lora_tests::patch_value(&v["value"], dev),
                    v["strength"].as_f64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let shape = c["weight"]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_i64().unwrap())
            .collect();
        assert_eq!(
            json!(p::calculate_shape_from_shape(&patches, shape).unwrap()),
            c["shape"],
            "{}",
            c["label"]
        );
    }
}
#[test]
fn moss_control_adapters_convert_and_load_through_the_moss_key_map_like_426f5eea() {
    for c in fixture()["loading"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        let state = c["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), value(v, Device::Cpu)))
            .collect();
        let converted = m::convert_moss_control_lora(&state);
        assert_eq!(
            json!(converted.keys().collect::<Vec<_>>()),
            c["converted"],
            "{label}"
        );
        let weights = c["weights"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| {
                let shape = w[1].as_array().unwrap().iter();
                (
                    w[0].as_str().unwrap().to_owned(),
                    shape.map(|d| d.as_i64().unwrap()).collect(),
                )
            })
            .collect::<Vec<_>>();
        let loaded = p::load_lora(&converted, &m::moss_key_map(&weights)).unwrap();
        let expected = c["patches"].as_array().unwrap();
        let mut actual = loaded
            .patches
            .iter()
            .map(|(target, patch)| {
                let key = match target.offset {
                    None => json!(target.key),
                    Some(offset) => json!([target.key, offset]),
                };
                (key, patch)
            })
            .collect::<Vec<_>>();
        let mut expected = expected
            .iter()
            .map(|e| (e[0].clone(), &e[1]))
            .collect::<Vec<_>>();
        actual.sort_by_key(|(k, _)| k.to_string());
        expected.sort_by_key(|(k, _)| k.to_string());
        assert_eq!(
            actual.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            expected.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            "{label}"
        );
        for ((_, patch), (_, e)) in actual.iter().zip(&expected) {
            assert_patch(patch, e);
        }
    }
}
