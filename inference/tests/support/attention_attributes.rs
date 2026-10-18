use super::{binding_tests::restore, kitchen};
use imagine_inference::{
    checkpoint::PreparedParameters, execution::Runtime, models::qwen_image21 as q,
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tch::Device;
#[test]
fn original_comfy_attention_attributes_state_dict_bytes_and_static_memory_ownership_match() {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = Arc::new(
        unsafe {
            Compiler::open(
                Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
                &own.join("cache"),
                &own.join("work"),
                native.clone(),
            )
        }
        .unwrap(),
    );
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/attention-attributes.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut parameters = PreparedParameters::plain(
            [
                ("proj.weight".into(), restore(&case["weight"], Device::Cpu)),
                ("sigmas".into(), restore(&case["sigmas"], Device::Cpu)),
            ]
            .into(),
        );
        if !case["raw"].is_null() {
            parameters
                .insert_attention_config("attn.config".into(), restore(&case["raw"], Device::Cpu))
                .unwrap();
        }
        assert_eq!(
            parameters.attention_state_dict_bytes(),
            case["attribute_bytes"].as_u64().unwrap()
        );
        assert_eq!(
            parameters
                .attention_config("attn.config")
                .unwrap()
                .unwrap_or(Value::Null),
            case["config"]
        );
        let mut owner = q::static_component_from_inventory(
            &rt,
            parameters,
            serde_json::from_value(case["inventory"].clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            owner.model_size(),
            case["inventory"]["model_size"].as_u64().unwrap()
        );
        assert_eq!(owner.modules.len(), 1);
        owner.unload().unwrap();
        let snapshot = owner.execution_parameters();
        assert!(!snapshot.values.contains_key("attn.config"));
        assert_eq!(
            snapshot.attention_state_dict_bytes(),
            case["attribute_bytes"].as_u64().unwrap()
        );
    }
}
