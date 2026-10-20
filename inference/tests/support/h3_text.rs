use super::{
    binding_tests::check,
    qwen_text_encoder_tests::{Exec, data},
};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::{PreparedParameters, bind_comfy_layouts},
    execution::Runtime,
    execution_value::Value as V,
    models::{minimax_h3::text_encoder as h3, qwen_image21 as q},
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};

#[test]
fn h3_truncated_language_matches_original_fifty_layers_without_final_norm() {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = super::kitchen();
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
    bind_comfy_layouts(&rt);
    let _grad = tch::no_grad_guard();
    let original_threads = tch::get_num_threads();
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _restore = Restore(original_threads);
    tch::set_num_threads(2);
    let f: Value = serde_json::from_str(include_str!("../../fixtures/h3-text.json")).unwrap();
    let cfg = h3::language_config();
    for (name, value) in [
        ("hidden_size", cfg.hidden_size),
        ("intermediate_size", cfg.intermediate_size),
        ("num_hidden_layers", cfg.layers as i64),
        ("num_attention_heads", cfg.heads),
        ("num_key_value_heads", cfg.kv_heads),
        ("head_dim", cfg.head_dim),
        ("vocab_size", cfg.vocab_size),
    ] {
        assert_eq!(json!(value), f["original_config"][name], "{name}");
    }
    assert_eq!(h3::vision_config().out_hidden_size, 5120);
    let mut count = 0;
    for g in f["groups"].as_array().unwrap() {
        let mut state = PreparedParameters::default();
        for (name, p) in g["factories"].as_object().unwrap() {
            let shape: Vec<i64> = serde_json::from_value(p["shape"].clone()).unwrap();
            let mut x = data(
                &shape,
                Device::Cpu,
                Kind::Float,
                p["offset"].as_i64().unwrap(),
            );
            if p["role"] == "norm" {
                x = x.f_mul_scalar(0.25).unwrap().f_add_scalar(1.).unwrap();
            }
            state
                .values
                .insert(name.clone(), V::tensor(x.to_kind(Kind::BFloat16)));
        }
        assert_eq!(
            state.values.contains_key("model.norm.weight"),
            g["final_norm"].as_bool().unwrap()
        );
        let model = q::QwenTextEncoder::from_parameters_with_final_norm(
            q::TextEncoderConfig {
                vocab_size: 128,
                hidden_size: 256,
                intermediate_size: 256,
                layers: g["layers"].as_u64().unwrap() as usize,
                heads: 2,
                kv_heads: 1,
                ..q::TextEncoderConfig::default()
            },
            &state,
            g["final_norm"].as_bool().unwrap(),
        )
        .unwrap();
        let mut exec = Exec {
            events: vec![],
            operations: 0,
            attention: Attention {
                method: AttentionMethod::Pytorch,
                priority_wrapper: true,
                attention_precision: None,
                low_precision_attention: true,
                model_device: Device::Cuda(0),
            },
        };
        for c in g["calls"].as_array().unwrap() {
            exec.events.clear();
            exec.operations = 0;
            let x = data(&[1, 5, 256], Device::Cpu, Kind::Float, 5);
            let subset = c["mode"] == "subset";
            let positions =
                subset.then(|| Tensor::arange(15, (Kind::Int64, Device::Cpu)).reshape([3, 5]));
            let mask = subset
                .then(|| Tensor::from_slice(&[false, true, false, true, false]).reshape([1, 5]));
            let ds = if subset {
                vec![
                    data(&[2, 256], Device::Cpu, Kind::Float, 13),
                    data(&[2, 256], Device::Cpu, Kind::Float, 17),
                ]
            } else {
                vec![]
            };
            let intermediate = match c["mode"].as_str().unwrap() {
                "last" => q::Intermediate::Layer(-1),
                "all" => q::Intermediate::All,
                _ => q::Intermediate::Layers(vec![0, 1, model.config.layers as i64]),
            };
            let out = unsafe {
                model
                    .forward(
                        &rt,
                        &x,
                        q::TextForwardOptions {
                            position_ids: positions.as_ref(),
                            intermediate,
                            final_layer_norm_intermediate: c["final"].as_bool().unwrap(),
                            deepstack: &ds,
                            visual_mask: mask.as_ref(),
                            ..q::TextForwardOptions::default()
                        },
                        &mut exec,
                    )
                    .unwrap()
            };
            check(
                &V::Tuple(vec![
                    V::tensor(out.last),
                    out.intermediate.map(V::tensor).unwrap_or(V::None),
                ]),
                &c["output"],
                Device::Cpu,
                "H3 unnormalized language",
            );
            check(
                &V::tensor(x),
                &c["post_embeds"],
                Device::Cpu,
                "H3 input owners",
            );
            assert_eq!(json!(exec.events), c["events"]);
            assert_eq!(exec.operations, c["operations"].as_u64().unwrap() as usize);
            count += 1;
        }
    }
    assert_eq!(count, 18);
}
