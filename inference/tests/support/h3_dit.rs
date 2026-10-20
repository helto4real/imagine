use super::qwen_text_encoder_tests::{Exec, data};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::{PreparedParameters, bind_comfy_layouts},
    execution::Runtime,
    execution_value::Value as V,
    models::minimax_h3::{
        dit as h3,
        layout::{Keyframe, ReferenceBlock},
    },
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
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
fn h3_complete_dense_dit_matches_original_basis_refiner_masks_refs_keyframes_and_audio_carry() {
    run_fixture(
        serde_json::from_str(include_str!("../../fixtures/h3-dit.json")).unwrap(),
        Device::Cpu,
    );
}
#[test]
#[ignore = "Requires explicit GPU lease and original tiny CUDA oracle"]
fn h3_tiny_cuda_dit_matches_original_all_dense_forwards() {
    let path = std::env::var("IMAGINE_H3_GPU_FIXTURE").expect("Explicit own CUDA oracle");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(fixture["cpu_only"], false);
    assert_eq!(fixture["trained_weights"], 0);
    run_fixture(fixture, Device::Cuda(0));
}
fn run_fixture(fixture: Value, dev: Device) {
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
    // Original comfy.quant_ops defaults to disabled Kitchen Triton.
    if dev != Device::Cpu {
        assert_eq!(fixture["kitchen_triton_enabled"], false);
        rt.backends.policy.disable("triton");
    }
    bind_comfy_layouts(&rt);
    let _grad = tch::no_grad_guard();
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _restore = Restore(tch::get_num_threads());
    tch::set_num_threads(2);

    let mut state = PreparedParameters::default();
    for (name, p) in fixture["factories"].as_object().unwrap() {
        let value = if p["literal"].is_null() {
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
            x.to_kind(super::kind(p["dtype"].as_str().unwrap()))
        } else {
            tensor(&p["literal"])
        };
        state
            .values
            .insert(name.clone(), V::tensor(value.to_device(dev)));
    }
    let config: h3::Config = serde_json::from_value(fixture["config"].clone()).unwrap();
    let model = h3::Denoiser::from_parameters(config, &state).unwrap();
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
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 42);
    for case in cases {
        let video = tensor(&case["video"]).to_device(dev);
        let audio = tensor(&case["audio"]).to_device(dev);
        let context = tensor(&case["context"]).to_device(dev);
        let timestep =
            Tensor::from_slice(&[case["timestep"].as_f64().unwrap() as f32]).to_device(dev);
        let keyframes: Vec<Keyframe> = serde_json::from_value(case["keyframes"].clone()).unwrap();
        let refs: Vec<ReferenceBlock> = serde_json::from_value(case["refs"].clone()).unwrap();
        let cv = case["video_conditions"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect::<Vec<_>>();
        let ca = case["audio_conditions"]
            .as_array()
            .unwrap()
            .iter()
            .map(tensor)
            .collect::<Vec<_>>();
        let mask =
            (!case["video_mask"].is_null()).then(|| tensor(&case["video_mask"]).to_device(dev));
        let amask =
            (!case["audio_mask"].is_null()).then(|| tensor(&case["audio_mask"]).to_device(dev));
        let tags = Tensor::from_slice(&[1_i64, 0, 1]);
        let output = unsafe {
            model.forward(
                &rt,
                h3::Forward {
                    video: &video,
                    audio: &audio,
                    timestep: &timestep,
                    context: &context,
                    payload: h3::Payload {
                        keyframes: &keyframes,
                        refs: &refs,
                        video_conditions: &cv,
                        audio_conditions: &ca,
                        text_tags: Some(&tags),
                        audio_aug: 0.9,
                        seed: 9,
                        audio_scale: case["audio_scale"].as_f64().unwrap(),
                        ..h3::Payload::default()
                    },
                    denoise_mask: mask.as_ref(),
                    audio_denoise_mask: amask.as_ref(),
                    shifts: [12., 3.],
                },
                &mut exec,
            )
        }
        .unwrap();
        for (i, out) in output.into_iter().enumerate() {
            super::binding_tests::check(
                &V::tensor(out),
                &case["output"][i],
                Device::Cpu,
                &format!(
                    "{}/{}/{} stream{i}",
                    case["dtype"], case["variant"], case["timestep"]
                ),
            );
        }
    }
}
