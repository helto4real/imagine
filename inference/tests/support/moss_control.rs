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
fn control_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../fixtures/reference-20261027/moss-control.json"
    ))
    .unwrap()
}
fn optional(v: &Value, dev: Device) -> Option<tch::Tensor> {
    (!v.is_null()).then(|| super::binding_tests::restore(v, dev).as_tensor().unwrap())
}
fn compare(actual: Option<tch::Tensor>, expected: &Value, label: &str) {
    match actual {
        None => assert!(expected.is_null(), "{label}: expected an output"),
        Some(t) => super::binding_tests::check(
            &imagine_inference::execution_value::Value::tensor(t),
            expected,
            Device::Cpu,
            label,
        ),
    }
}
#[test]
fn moss_control_latents_concat_conds_and_timestep_spans_match_426f5eea() {
    let _grad = tch::no_grad_guard();
    let f = control_fixture();
    for c in f["latents"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        let (color, depth) = (
            optional(&c["color"], Device::Cpu),
            optional(&c["depth"], Device::Cpu),
        );
        let control = m::moss_control_latent(
            color.as_ref(),
            depth.as_ref(),
            c["strength"].as_f64().unwrap(),
        )
        .unwrap();
        compare(control, &c["control"], &format!("latent/{label}"));
    }
    for c in f["concats"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        let noise = c["noise"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_i64().unwrap())
            .collect::<Vec<_>>();
        let image = optional(&c["image"], Device::Cpu);
        let out = m::moss_concat_cond(
            &noise,
            c["channels"].as_i64().unwrap(),
            image.as_ref(),
            Device::Cpu,
        )
        .unwrap();
        compare(out, &c["output"], &format!("concat/{label}"));
    }
    let sampling = m::MossSampling::moss().unwrap();
    let sigmas = f["sigmas"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    for c in f["spans"].as_array().unwrap() {
        let (start, end) = (c["start"].as_f64().unwrap(), c["end"].as_f64().unwrap());
        let spans = m::moss_control_spans(start, end, &sampling);
        let expected = c["entries"].as_array().unwrap();
        assert_eq!(spans.len(), expected.len(), "{start}..{end}");
        for (span, e) in spans.iter().zip(expected) {
            assert_eq!(
                span.control,
                e["control"].as_bool().unwrap(),
                "{start}..{end}"
            );
            assert_eq!(
                span.sigma_start,
                e["sigma_start"].as_f64(),
                "{start}..{end}"
            );
            assert_eq!(span.sigma_end, e["sigma_end"].as_f64(), "{start}..{end}");
            let active = sigmas.iter().map(|&s| span.active(s)).collect::<Vec<_>>();
            assert_eq!(json!(active), e["active"], "{start}..{end}");
        }
        // A sampling run admits exactly one span per sigma, or reports the gap.
        let control = m::MossControl::new(
            Some(&tch::Tensor::zeros(
                [1, 4, 2, 2],
                (tch::Kind::Float, Device::Cpu),
            )),
            8,
            &[1, 4, 2, 2],
            Device::Cpu,
            (start, end),
            &sampling,
        )
        .unwrap()
        .unwrap();
        for (i, &sigma) in sigmas.iter().enumerate() {
            let admitted = expected.iter().filter(|e| e["active"][i] == true).count();
            assert_eq!(
                control.concat(sigma).is_ok(),
                admitted == 1,
                "{start}..{end} at {sigma}"
            );
        }
    }
}
#[test]
fn moss_dit_with_a_widened_video_embedder_matches_426f5eea() {
    use super::moss_dit_tests::{Exec, Threads, group_model};
    use imagine_inference::{execution::Runtime, triton_compiler::Compiler};
    use std::{path::Path, sync::Arc};
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    for group in control_fixture()["models"].as_array().unwrap() {
        let label = group["label"].as_str().unwrap();
        let dev = if label.starts_with("cpu/") {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let (mut params, _) = group_model(&rt, group, label, dev);
        params.values.insert(
            "video_embedder.weight".into(),
            super::binding_tests::restore(&group["widened"], dev),
        );
        let c = &group["config"];
        let n = |key: &str| c[key].as_i64().unwrap();
        let model = m::MossDiT::from_parameters(
            m::MossConfig {
                dim: n("dim"),
                ffn_dim: n("ffn_dim"),
                heads: n("num_heads"),
                layers: n("num_layers") as usize,
                text_dim: n("text_dim"),
                latent_channels: n("latent_channels"),
                fused_qkv: false,
            },
            &params,
        )
        .unwrap();
        let args = group["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| super::binding_tests::restore(v, dev).as_tensor().unwrap())
            .collect::<Vec<_>>();
        // BaseModel: cat([x, c_concat]) in the input's dtype, then the model dtype.
        let input = tch::Tensor::cat(&[&args[0], &args[1]], 1).to_kind(tch::Kind::BFloat16);
        let mut exec = Exec {
            training: false,
            patched: false,
            events: vec![],
            operations: 0,
            replacements: vec![],
        };
        let output = unsafe {
            model.forward(
                &rt,
                &m::MossInputs {
                    latent: &input,
                    timestep: &args[2],
                    context: &args[3],
                    references: None,
                    rope: Default::default(),
                },
                &mut exec,
            )
        }
        .unwrap_or_else(|err| panic!("{label}: {err}"));
        super::binding_tests::check(
            &imagine_inference::execution_value::Value::tensor(output),
            &group["output"],
            dev,
            &format!("{label}/widened"),
        );
    }
}
