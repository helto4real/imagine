use super::qwen_text_encoder_tests::{Exec, data};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::{PreparedParameters, bind_comfy_layouts},
    execution::Runtime,
    execution_value::Value as V,
    models::{
        minimax_h3::{
            text_encoder as h3,
            tokenizer::{Entry, Token},
        },
        qwen_image21 as q,
    },
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct TraceExec<'a> {
    inner: Exec,
    trace: &'a Value,
    block: usize,
}
impl q::TextEncoderExecution for TraceExec<'_> {
    fn observe(&mut self, name: &str, t: &Tensor) -> Result<(), imagine_inference::native::Error> {
        let precise = format!("{}.{name}", self.block);
        if !self.trace[&precise].is_null() {
            if !self.trace[&precise]["debug_payload_hex"].is_null() {
                let actual = super::bits(t);
                let expected = super::exact_hex(&self.trace[&precise]["debug_payload_hex"]);
                if actual != expected {
                    let pairs = actual
                        .chunks_exact(4)
                        .zip(expected.chunks_exact(4))
                        .enumerate()
                        .filter(|(_, (a, b))| a != b)
                        .take(6)
                        .map(|(i, (a, b))| {
                            (
                                i,
                                u32::from_le_bytes(a.try_into().unwrap()),
                                u32::from_le_bytes(b.try_into().unwrap()),
                            )
                        })
                        .collect::<Vec<_>>();
                    eprintln!("{precise} first different f32 words: {pairs:?}");
                }
            }
            super::binding_tests::check(
                &V::tensor(t.shallow_clone()),
                &self.trace[&precise],
                Device::Cpu,
                &precise,
            );
        }
        if !self.trace[name].is_null() {
            super::binding_tests::check(
                &V::tensor(t.shallow_clone()),
                &self.trace[name],
                Device::Cpu,
                name,
            );
        }
        Ok(())
    }
    fn check_interrupt(&mut self) -> Result<(), imagine_inference::native::Error> {
        self.inner.check_interrupt()
    }
    fn queue_begin(&mut self, _: usize, _: Device) -> Result<(), imagine_inference::native::Error> {
        Ok(())
    }
    fn queue_end(&mut self, _: Device) -> Result<(), imagine_inference::native::Error> {
        Ok(())
    }
    fn block(
        &mut self,
        index: usize,
        core: &mut dyn FnMut(&mut Self) -> Result<(), imagine_inference::native::Error>,
    ) -> Result<(), imagine_inference::native::Error> {
        self.block = index;
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
    ) -> Result<Tensor, imagine_inference::native::Error> {
        self.inner.attention(rt, q, k, v, mask, heads, gqa)
    }
}
#[test]
fn h3_complete_conditioning_matches_original_raw_text_images_frame_pairs_deepstack_and_tags() {
    run_fixture(
        serde_json::from_str(include_str!("../../fixtures/h3-conditioning.json")).unwrap(),
        Device::Cpu,
    );
}
#[test]
#[ignore = "Requires explicit GPU lease and original tiny CUDA oracle"]
fn h3_tiny_cuda_conditioning_matches_original_raw_text_vision_deepstack() {
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
    if dev != Device::Cpu {
        rt.backends.observe_selection(Some(Arc::new(|op, backend| {
            if op == "apply_rope_split_half" {
                eprintln!("H3 synthetic text RoPE backend: {backend}");
            }
        })));
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
        state.values.insert(
            name.clone(),
            V::tensor(
                x.to_kind(super::kind(p["dtype"].as_str().unwrap()))
                    .to_device(dev),
            ),
        );
    }
    for (name, v) in fixture["buffers"].as_object().unwrap() {
        let shape: Vec<i64> = serde_json::from_value(v["shape"].clone()).unwrap();
        let t = Tensor::f_from_data_size(
            &super::exact_hex(&v["bytes"]),
            &shape,
            super::kind(v["dtype"].as_str().unwrap()),
        )
        .unwrap();
        state
            .values
            .insert(name.clone(), V::tensor(t.to_device(dev)));
    }
    let language = q::QwenTextEncoder::from_parameters_with_final_norm(
        q::TextEncoderConfig {
            hidden_size: 16,
            intermediate_size: 32,
            layers: 3,
            heads: 2,
            kv_heads: 1,
            ..q::TextEncoderConfig::default()
        },
        &state,
        false,
    )
    .unwrap();
    let vision = q::QwenVisionEncoder::from_parameters(
        q::VisionEncoderConfig {
            hidden_size: 64,
            intermediate_size: 128,
            depth: 3,
            heads: 4,
            out_hidden_size: 16,
            deepstack_indexes: vec![0, 1, 2],
            ..q::VisionEncoderConfig::default()
        },
        &state,
    )
    .unwrap();
    let model = h3::TextEncoder::from_components(language, Some(vision)).unwrap();
    let exec = Exec {
        events: vec![],
        operations: 0,
        attention: Attention {
            method: if dev == Device::Cpu {
                AttentionMethod::Basic
            } else {
                AttentionMethod::Pytorch
            },
            priority_wrapper: dev != Device::Cpu,
            attention_precision: None,
            low_precision_attention: true,
            model_device: dev,
        },
    };
    let mut trace_exec = TraceExec {
        block: 0,
        inner: exec,
        trace: &Value::Null,
    };
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for case in cases {
        trace_exec.trace = &case["trace"];
        let entries = case["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let token = if let Some(id) = t["id"].as_i64() {
                    Token::Id(id)
                } else {
                    let v = &t["vision"];
                    let shape: Vec<i64> = serde_json::from_value(v["shape"].clone()).unwrap();
                    let image = Tensor::f_from_data_size(
                        &super::exact_hex(&v["bytes"]),
                        &shape,
                        super::kind(v["dtype"].as_str().unwrap()),
                    )
                    .unwrap();
                    Token::Vision {
                        data: V::tensor(image),
                        video_block: t["video_block"].as_bool().unwrap(),
                    }
                };
                Entry {
                    token,
                    weight: 1.,
                    word_id: None,
                }
            })
            .collect::<Vec<_>>();
        let output =
            unsafe { model.encode(&rt, &entries, dev, Device::Cpu, &mut trace_exec) }.unwrap();
        super::binding_tests::check(
            &V::tensor(output.context),
            &case["context"],
            Device::Cpu,
            case["label"].as_str().unwrap(),
        );
        assert_eq!(
            super::bits(&output.token_tags),
            super::exact_hex(&case["tags"]["bytes"])
        );
    }
}
