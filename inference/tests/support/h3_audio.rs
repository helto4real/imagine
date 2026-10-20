use super::qwen_text_encoder_tests::Exec;
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value as V,
    models::minimax_h3::audio_vae::{AudioVae, Config},
};
use serde_json::Value;
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
fn h3_complete_audio_vae_matches_original_encoder_causal_projection_resampling_decoder_and_stereo_pcm()
 {
    run_fixture(
        serde_json::from_str(include_str!("../../fixtures/h3-audio.json")).unwrap(),
        Device::Cpu,
    );
}
#[test]
#[ignore = "Requires explicit GPU lease and original tiny CUDA oracle"]
fn h3_tiny_cuda_audio_matches_original_encoder_causal_attention_and_pcm() {
    let path = std::env::var("IMAGINE_H3_GPU_FIXTURE").expect("Explicit own CUDA oracle");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(fixture["cpu_only"], false);
    assert_eq!(fixture["trained_weights"], 0);
    run_fixture(fixture, Device::Cuda(0));
}
fn run_fixture(fixture: Value, dev: Device) {
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    // Original comfy.quant_ops defaults to disabled Kitchen Triton.
    if dev != Device::Cpu {
        assert_eq!(fixture["kitchen_triton_enabled"], false);
        rt.backends.policy.disable("triton");
    }
    let initial_state = if !fixture["cuda_sdpa_initial"].is_null() {
        let saved = rt.native.sdpa_state().unwrap();
        rt.native
            .set_sdpa_state(serde_json::from_value(fixture["cuda_sdpa_initial"].clone()).unwrap())
            .unwrap();
        Some(saved)
    } else {
        None
    };
    let _grad = tch::no_grad_guard();
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _restore = Restore(tch::get_num_threads());
    tch::set_num_threads(2);
    let f = fixture;
    let mut state = PreparedParameters::default();
    for (name, p) in f["factories"].as_object().unwrap() {
        let shape: Vec<i64> = serde_json::from_value(p["shape"].clone()).unwrap();
        let mut x = Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, Device::Cpu))
            .f_add_scalar(p["offset"].as_i64().unwrap())
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_sub_scalar(14.)
            .unwrap()
            .f_mul_scalar(0.00390625)
            .unwrap()
            .f_reshape(&shape)
            .unwrap();
        if p["role"] == "norm" {
            x = x.f_mul_scalar(0.25).unwrap().f_add_scalar(1.).unwrap();
        }
        state
            .values
            .insert(name.clone(), V::tensor(x.to_device(dev)));
    }
    for (name, b) in f["buffers"].as_object().unwrap() {
        state
            .values
            .insert(name.clone(), V::tensor(tensor(b).to_device(dev)));
    }
    let model = AudioVae::new(
        state,
        Config {
            encoder_rates: vec![2, 2],
            attention_heads: 8,
        },
    );
    let mut exec = Exec {
        events: vec![],
        operations: 0,
        attention: Attention {
            method: AttentionMethod::Pytorch,
            priority_wrapper: dev != Device::Cpu,
            attention_precision: None,
            low_precision_attention: true,
            model_device: dev,
        },
    };
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), if dev == Device::Cpu { 8 } else { 9 });
    for case in cases {
        let input = tensor(&case["input"]).to_device(dev);
        let out = unsafe {
            if case["operation"] == "encode" {
                model.encode(&rt, &input, &mut exec)
            } else {
                model.decode(&rt, &input, &mut exec)
            }
        }
        .unwrap();
        super::binding_tests::check(
            &V::tensor(out),
            &case["output"],
            Device::Cpu,
            &format!("H3 audio {} {:?}", case["operation"], input.size()),
        );
        assert_eq!(
            super::bits(&input),
            super::exact_hex(&case["input"]["bytes"])
        );
    }
    if let Some(saved) = initial_state {
        assert_eq!(
            serde_json::json!(rt.native.sdpa_state().unwrap()),
            f["cuda_sdpa_final"]
        );
        rt.native.set_sdpa_state(saved).unwrap();
    }
}
