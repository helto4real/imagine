use super::qwen_text_encoder_tests::{Exec, data};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value as V,
    models::{
        minimax_h3::{
            dit::{self, Payload},
            layout::ReferenceBlock,
            sampling as h3,
            streams::NestedTensor,
        },
        qwen_image21::TextEncoderExecution,
    },
    native::Error,
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path, sync::Arc};
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
struct Trace {
    inner: Exec,
    expected: BTreeMap<String, Value>,
}
impl TextEncoderExecution for Trace {
    fn observe(&mut self, name: &str, t: &Tensor) -> Result<(), Error> {
        if name.starts_with("h3.sampling.") {
            let expected = self
                .expected
                .remove(name)
                .unwrap_or_else(|| panic!("Unexpected {name}"));
            super::binding_tests::check(
                &V::tensor(t.shallow_clone()),
                &expected,
                Device::Cpu,
                name,
            );
        }
        Ok(())
    }
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.inner.check_interrupt()
    }
    fn queue_begin(&mut self, n: usize, d: Device) -> Result<(), Error> {
        self.inner.queue_begin(n, d)
    }
    fn queue_end(&mut self, d: Device) -> Result<(), Error> {
        self.inner.queue_end(d)
    }
    fn block(
        &mut self,
        _: usize,
        core: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        core(self)
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        heads: i64,
        gqa: bool,
    ) -> Result<Tensor, Error> {
        self.inner.attention(rt, q, k, v, mask, heads, gqa)
    }
}
#[test]
fn h3_learned_av_sampling_matches_original_noise_schedule_cfg1_mask_injection_each_euler_step_and_both_streams()
 {
    run_fixture(
        serde_json::from_str(include_str!("../../fixtures/h3-sampling.json")).unwrap(),
        Device::Cpu,
    );
}
#[test]
#[ignore = "Requires explicit GPU lease and original tiny CUDA oracle"]
fn h3_tiny_cuda_sampling_matches_original_complete_av_chain() {
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
    let model = dit::Denoiser::from_parameters(
        serde_json::from_value(f["config"].clone()).unwrap(),
        &state,
    )
    .unwrap();
    let schedule = h3::selected_schedule(&rt).unwrap();
    let sigmas = h3::simple_sigmas(&schedule, 4).unwrap();
    assert_eq!(f["cases"].as_array().unwrap().len(), 5);
    for case in f["cases"].as_array().unwrap() {
        let latent = NestedTensor::new(vec![
            V::tensor(tensor(&case["video"]).to_device(dev)),
            V::tensor(tensor(&case["audio"]).to_device(dev)),
        ]);
        let noise = latent.prepare_noise(42, None).unwrap();
        for (i, t) in noise.unbind().borrow().iter().enumerate() {
            super::binding_tests::check(t, &case["noise"][i], Device::Cpu, "H3 AV noise");
        }
        assert_eq!(
            super::bits(&sigmas),
            super::exact_hex(&case["sigmas"]["bytes"])
        );
        let context = tensor(&case["context"]);
        let tags = Tensor::from_slice(&[1_i64, 0, 1]);
        let refs: Vec<ReferenceBlock> = serde_json::from_value(case["refs"].clone()).unwrap();
        let vc = case["refs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| !r["latent"].is_null())
            .map(|r| tensor(&r["latent"]))
            .collect::<Vec<_>>();
        let ac = case["refs"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|r| !r["audio_latent"].is_null())
            .map(|r| tensor(&r["audio_latent"]))
            .collect::<Vec<_>>();
        let mask = (!case["mask"].is_null()).then(|| tensor(&case["mask"]).to_device(dev));
        if let Some(m) = &mask {
            let (_, shapes) = latent.pack().unwrap();
            for (i, g) in h3::token_grid_masks(m, &shapes)
                .unwrap()
                .into_iter()
                .enumerate()
            {
                super::binding_tests::check(
                    &V::tensor(g),
                    &case["grid"][i],
                    Device::Cpu,
                    "H3 AV pooled grid",
                );
            }
        }
        let expected = case["trace"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|step| {
                ["x", "denoised"].map(|field| {
                    (
                        format!("h3.sampling.{}.{field}", step["i"].as_i64().unwrap()),
                        step[field].clone(),
                    )
                })
            })
            .collect();
        let mut exec = Trace {
            inner: Exec {
                events: vec![],
                operations: 0,
                attention: Attention {
                    method: AttentionMethod::Pytorch,
                    priority_wrapper: dev != Device::Cpu,
                    attention_precision: None,
                    low_precision_attention: true,
                    model_device: dev,
                },
            },
            expected,
        };
        let out = unsafe {
            h3::sample_euler_cfg1(
                &rt,
                &model,
                &schedule,
                h3::EulerRequest {
                    latent: &latent,
                    noise: &noise,
                    sigmas: &sigmas,
                    context: &context,
                    payload: Payload {
                        refs: &refs,
                        video_conditions: &vc,
                        audio_conditions: &ac,
                        text_tags: Some(&tags),
                        seed: 9,
                        ..Payload::default()
                    },
                    denoise_mask: mask.as_ref(),
                    kind: Kind::BFloat16,
                },
                &mut exec,
            )
        }
        .unwrap();
        assert!(exec.expected.is_empty());
        for (i, t) in out.unbind().borrow().iter().enumerate() {
            super::binding_tests::check(
                t,
                &case["output"][i],
                Device::Cpu,
                &format!("H3 complete sampled {} stream{i}", case["variant"]),
            );
        }
    }
}
