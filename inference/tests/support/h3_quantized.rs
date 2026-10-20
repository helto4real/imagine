use super::qwen_text_encoder_tests::{Exec, data};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::{self, bind_comfy_layouts},
    execution::Runtime,
    execution_value::Value as V,
    models::{
        minimax_h3::{checkpoint as h3, dit},
        qwen_image21 as q,
    },
    patching::options::{Options, SourceText},
};
use serde_json::Value;
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};
fn tensor(v: &Value) -> Tensor {
    let shape: Vec<i64> = serde_json::from_value(v["shape"].clone()).unwrap();
    Tensor::f_from_data_size(
        &super::exact_hex(&v["bytes"]),
        &shape,
        super::kind(v["dtype"].as_str().unwrap()),
    )
    .unwrap()
}
#[test]
#[ignore = "Requires canonical GPU lease and own tiny quantized original oracle"]
fn h3_tiny_cuda_selected_quantized_language_and_dit_match_original() {
    let path = std::env::var("IMAGINE_H3_GPU_FIXTURE").unwrap();
    let f: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(f["trained_weights"], 0);
    assert_eq!(f["cpu_only"], false);
    assert_eq!(f["kitchen_triton_enabled"], false);
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    bind_comfy_layouts(&rt);
    rt.backends.policy.disable("triton");
    let _grad = tch::no_grad_guard();
    let dev = Device::Cuda(0);
    let mut exec = Exec {
        events: vec![],
        operations: 0,
        attention: Attention {
            method: AttentionMethod::Pytorch,
            priority_wrapper: true,
            attention_precision: None,
            low_precision_attention: true,
            model_device: dev,
        },
    };
    assert_eq!(f["cases"].as_array().unwrap().len(), 2);
    let mut total = 0;
    for entry in f["cases"].as_array().unwrap() {
        let loading = &entry["loading"];
        let role = if entry["role"] == "text" {
            h3::Component::Text
        } else {
            h3::Component::Dit
        };
        let weights = Options::dictionary(
            loading["input"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(n, t)| (n.clone(), Options::Atom(V::tensor(tensor(t)))))
                .collect(),
        );
        let metadata = Options::dictionary(
            loading["metadata"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(n, t)| (n.clone(), SourceText::from_utf8(t.as_str().unwrap())))
                .collect(),
        );
        let normalized = unsafe {
            checkpoint::component::normalize(&rt, weights, metadata, &Options::Atom(V::None))
        }
        .unwrap();
        let V::Dict(values) = normalized.weights.snapshot().unwrap() else {
            panic!("Original ordered tensor state")
        };
        let state = values
            .into_iter()
            .map(|(n, v)| (n, v.as_tensor().unwrap()))
            .collect::<BTreeMap<_, _>>();
        let spec = serde_json::from_value(loading["spec"].clone()).unwrap();
        let loaded = serde_json::from_value(loading["loaded"].clone()).unwrap();
        let mut state = h3::prepare_with_schema(&rt, role, &spec, &loaded, state).unwrap();
        for value in state.values.values_mut() {
            *value = if let Ok(t) = value.as_tensor() {
                V::tensor(t.to_device(dev))
            } else {
                unsafe {
                    rt.op(
                        "aten::to.dtype_layout",
                        std::slice::from_ref(value),
                        &vec![("device".into(), V::Device(dev))],
                    )
                    .unwrap()
                }
            };
        }
        if role == h3::Component::Text {
            let model = q::QwenTextEncoder::from_parameters_with_final_norm(
                q::TextEncoderConfig {
                    vocab_size: 128,
                    hidden_size: 256,
                    intermediate_size: 256,
                    layers: 2,
                    heads: 2,
                    kv_heads: 1,
                    ..Default::default()
                },
                &state,
                false,
            )
            .unwrap();
            for case in entry["forwards"].as_array().unwrap() {
                let ids: Vec<i64> = serde_json::from_value(case["ids"].clone()).unwrap();
                let ids = Tensor::from_slice(&ids).unsqueeze(0).to_device(dev);
                let x = unsafe { model.embed(&rt, &ids, &mut exec) }.unwrap();
                let out = unsafe { model.forward(&rt, &x, Default::default(), &mut exec) }.unwrap();
                super::binding_tests::check(
                    &V::tensor(out.last),
                    &case["output"],
                    dev,
                    "H3 selected INT8 full-precision language",
                );
                total += 1;
            }
        } else {
            let model = dit::Denoiser::from_parameters(
                dit::Config {
                    hidden_size: 256,
                    num_layers: 1,
                    token_refiner_num_layers: 1,
                    num_attention_heads: 2,
                    attention_head_dim: 128,
                    ffn_hidden_size: 256,
                    latents_dim: 24,
                    audio_latents_dim: 32,
                    text_dim: 256,
                    adaln_curve_grid: 5,
                    time_embed_dim: 8,
                    rope_inv_freq_len: 2,
                },
                &state,
            )
            .unwrap();
            for case in entry["forwards"].as_array().unwrap() {
                let v = data(&[1, 24, 2, 2, 4], Device::Cpu, Kind::BFloat16, 0).to_device(dev);
                let a = data(&[1, 32, 2, 3], Device::Cpu, Kind::BFloat16, 0).to_device(dev);
                let context = data(&[1, 3, 256], Device::Cpu, Kind::BFloat16, 0).to_device(dev);
                let t =
                    Tensor::from_slice(&[case["timestep"].as_f64().unwrap() as f32]).to_device(dev);
                let out = unsafe {
                    model.forward(
                        &rt,
                        dit::Forward {
                            video: &v,
                            audio: &a,
                            timestep: &t,
                            context: &context,
                            payload: dit::Payload {
                                seed: 9,
                                audio_scale: 4.,
                                ..Default::default()
                            },
                            denoise_mask: None,
                            audio_denoise_mask: None,
                            shifts: [12., 3.],
                        },
                        &mut exec,
                    )
                }
                .unwrap();
                for (i, t) in out.into_iter().enumerate() {
                    super::binding_tests::check(
                        &V::tensor(t),
                        &case["output"][i],
                        dev,
                        "H3 selected INT8 DiT",
                    );
                }
                total += 1;
            }
        }
    }
    assert_eq!(total, 6);
}
