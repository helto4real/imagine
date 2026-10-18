use super::{
    binding_tests::check,
    kitchen,
    moss_text_tests::{make_exec, make_model},
};
use imagine_inference::{
    execution::Runtime,
    execution_value::Value as V,
    models::qwen_image21::{PromptToken, TextConditioningOptions, TokenizedPrompt, WeightedToken},
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use tch::Device;
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0)
    }
}
#[test]
fn moss_original_complete_text_conditioning_final_norm_weights_sections_padding_and_masks_match() {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
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
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-conditioning.json")).unwrap();
    let mut count = 0;
    for group in fixture["groups"].as_array().unwrap() {
        let dev = super::device(group["device"].as_str().unwrap());
        let model = make_model(&rt, group);
        for call in group["calls"].as_array().unwrap() {
            let prompt = TokenizedPrompt {
                batches: call["tokens"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|batch| {
                        batch
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|t| WeightedToken {
                                token: PromptToken::Id(t[0].as_i64().unwrap()),
                                weight: t[1].as_f64().unwrap(),
                                word_id: 0,
                            })
                            .collect()
                    })
                    .collect(),
                keep_vision: false,
            };
            let mut e = make_exec(group["device"].as_str().unwrap());
            let out = unsafe {
                model.encode_prompt(
                    &rt,
                    &prompt,
                    TextConditioningOptions {
                        execution_device: dev,
                        intermediate_device: Device::Cpu,
                    },
                    &mut e,
                )
            }
            .unwrap_or_else(|err| panic!("{dev:?}/{}/{err}", call["label"]));
            let output = V::Tuple(vec![
                V::tensor(out.context),
                V::None,
                V::Dict(vec![(
                    "attention_mask".into(),
                    V::tensor(out.attention_mask.unwrap()),
                )]),
            ]);
            check(
                &output,
                &call["output"],
                Device::Cpu,
                &format!("{dev:?}/int8={}/{}", group["quantized"], call["label"]),
            );
            assert_eq!(json!(e.operations), call["operations"]);
            assert_eq!(json!(e.events), call["events"]);
            count += 1;
        }
    }
    assert_eq!(count, 28);
}
