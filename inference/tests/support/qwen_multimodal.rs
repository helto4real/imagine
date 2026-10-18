use super::{
    binding_tests::check, device, kitchen, qwen_text_encoder_tests as text,
    qwen_vision_encoder_tests as vision,
};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::bind_comfy_layouts,
    execution::Runtime,
    execution_value::Value as V,
    models::qwen_image21 as q,
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tch::{Device, Kind};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
#[test]
fn qwen_image21_original_qwen3vl_complete_multimodal_expansion_language_and_filter_match() {
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
    bind_comfy_layouts(&rt);
    rt.backends.policy.disable("triton");
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-multimodal.json")).unwrap();
    let groups = fixture["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 4);
    let mut count = 0;
    for g in groups {
        let dev = device(g["device"].as_str().unwrap());
        let (state, config) = vision::prepare(&rt, g, &g["vision_config"]);
        let visual = q::QwenVisionEncoder::from_parameters(config, &state).unwrap();
        let language =
            q::QwenTextEncoder::from_parameters(text::config(&g["language_config"]), &state)
                .unwrap();
        let images = vec![
            text::data(&[1, 64, 64, 3], dev, Kind::Float, 7).g_add_scalar(0.5),
            text::data(&[1, 64, 96, 3], dev, Kind::Float, 13).g_add_scalar(0.5),
        ];
        let mut exec = text::Exec {
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
            let tokens = c["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| q::WeightedToken {
                    token: if let Some(i) = t.as_i64() {
                        q::PromptToken::Id(i)
                    } else {
                        q::PromptToken::Image(
                            t.as_str()
                                .unwrap()
                                .strip_prefix("image")
                                .unwrap()
                                .parse()
                                .unwrap(),
                        )
                    },
                    weight: 1.,
                    word_id: 0,
                })
                .collect();
            let prompt = q::TokenizedPrompt {
                batches: vec![tokens],
                keep_vision: c["keep_vision"].as_bool().unwrap(),
            };
            let out = unsafe {
                language
                    .encode_prompt_with_images(
                        &rt,
                        &prompt,
                        q::TextConditioningOptions {
                            execution_device: dev,
                            intermediate_device: Device::Cpu,
                        },
                        Some((&visual, &images)),
                        &mut exec,
                    )
                    .unwrap()
            };
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
            let label = format!(
                "{}/{}/{}/keep={}",
                g["device"], g["variant"], c["label"], c["keep_vision"]
            );
            check(
                &V::Tuple(vec![
                    V::tensor(out.context),
                    out.pooled.map(V::tensor).unwrap_or(V::None),
                    V::Dict(extra),
                ]),
                &c["output"],
                dev,
                &label,
            );
            assert_eq!(
                exec.operations,
                c["operations"].as_u64().unwrap() as usize,
                "{label}: original interruption checkpoints"
            );
            assert_eq!(
                serde_json::json!(exec.events),
                c["events"],
                "{label}: prefetch events"
            );
            count += 1;
        }
    }
    assert_eq!(count, 40);
}
