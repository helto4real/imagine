use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    attention::{Attention, AttentionInput, AttentionMethod},
    checkpoint::{LinearPolicy, PreparedParameters, bind_comfy_layouts},
    execution::Runtime,
    execution_value::Value as V,
    models::qwen_image21 as q,
    native::Error,
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
pub(super) struct Exec {
    pub(super) events: Vec<Value>,
    pub(super) operations: usize,
    pub(super) attention: Attention,
}
impl q::TextEncoderExecution for Exec {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        Ok(())
    }
    fn queue_begin(&mut self, _layers: usize, device: Device) -> Result<(), Error> {
        self.events.push(json!(["Queue",if device==Device::Cpu{"cpu"}else{"cuda:0"},{"prefetch_dynamic_vbars":false}]));
        Ok(())
    }
    fn queue_end(&mut self, _device: Device) -> Result<(), Error> {
        self.events.push(json!("QueueEnd"));
        Ok(())
    }
    fn block(
        &mut self,
        index: usize,
        core: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.events.push(json!(format!("Block({index})")));
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
        unsafe {
            self.attention.run(
                rt,
                AttentionInput {
                    q,
                    k,
                    v,
                    heads,
                    mask,
                    skip_reshape: true,
                    skip_output_reshape: false,
                    enable_gqa: gqa,
                    scale: None,
                },
            )
        }
    }
}
pub(super) fn data(shape: &[i64], device: Device, kind: Kind, offset: i64) -> Tensor {
    Tensor::f_arange(shape.iter().product::<i64>(), (Kind::Float, device))
        .unwrap()
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .f_reshape(shape)
        .unwrap()
        .f_to_kind(kind)
        .unwrap()
}
pub(super) fn config(c: &Value) -> q::TextEncoderConfig {
    let n = |k: &str| c[k].as_i64().unwrap();
    q::TextEncoderConfig {
        vocab_size: n("vocab_size"),
        hidden_size: n("hidden_size"),
        intermediate_size: n("intermediate_size"),
        layers: n("layers") as usize,
        heads: n("heads"),
        kv_heads: n("kv_heads"),
        head_dim: n("head_dim"),
        eps: c["eps"].as_f64().unwrap(),
        theta: c["theta"].as_f64().unwrap(),
        rope_dims: c["rope_dims"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap(),
        interleaved_mrope: c["interleaved_mrope"].as_bool().unwrap(),
        merged_qkv: c["merged_qkv"].as_bool().unwrap(),
        merged_mlp: c["merged_mlp"].as_bool().unwrap(),
        qkv_bias: c["qkv_bias"].as_bool().unwrap(),
    }
}
#[test]
fn qwen_image21_original_qwen3vl_language_prefill_conditioning_and_prefinal_norm_match() {
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
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-text-encoder.json"
    ))
    .unwrap();
    let mut complete = 0;
    let mut cores = 0;
    for g in fixture["groups"].as_array().unwrap() {
        if g["triton_disabled"] == true {
            rt.backends.policy.disable("triton");
        } else {
            rt.backends.policy.enable("triton");
        }
        let label = format!(
            "{}/{}/merged={}/layers={}",
            g["device"], g["variant"], g["config"]["merged_mlp"], g["config"]["layers"]
        );
        eprintln!("Original language: {label}");
        let dev = device(g["device"].as_str().unwrap());
        let mut state = PreparedParameters::default();
        for (name, f) in g["factories"].as_object().unwrap() {
            let shape = f["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            let mut weight = data(
                &shape,
                Device::Cpu,
                Kind::Float,
                f["offset"].as_i64().unwrap(),
            );
            if f["role"] == "norm" {
                weight = weight.f_mul_scalar(0.25).unwrap().f_add_scalar(1.).unwrap();
            }
            weight = weight
                .f_to_kind(kind(f["base_dtype"].as_str().unwrap()))
                .unwrap();
            let value = if f["quantized"].as_bool().unwrap() {
                let q = if f["quant_format"] == "float8_e4m3fn" {
                    let scale = Tensor::f_full(
                        [],
                        f["scale"].as_f64().unwrap(),
                        (Kind::Float, Device::Cpu),
                    )
                    .unwrap();
                    let payload = unsafe {
                        rt.call_kitchen(
                            "quantize_per_tensor_fp8",
                            &vec![
                                ("x".into(), V::tensor(weight.shallow_clone())),
                                ("scale".into(), V::tensor(scale.shallow_clone())),
                                ("output_type".into(), V::DType(Kind::Float8e4m3fn)),
                            ],
                        )
                        .unwrap()
                    }
                    .as_tensor()
                    .unwrap();
                    let mut q = imagine_inference::layouts::fp8_parts(
                        payload,
                        scale,
                        weight.kind(),
                        weight.size(),
                    )
                    .unwrap();
                    q.layout = "TensorCoreFP8E4M3Layout".into();
                    q
                } else {
                    unsafe {
                        rt.from_float(
                            &weight,
                            "TensorWiseINT8Layout",
                            &vec![
                                ("is_weight".into(), V::Bool(true)),
                                ("per_channel".into(), V::Bool(true)),
                                ("convrot".into(), V::Bool(true)),
                                ("convrot_groupsize".into(), V::Int(256)),
                            ],
                        )
                        .unwrap()
                    }
                };
                let moved = unsafe {
                    rt.op(
                        "aten::to.dtype_layout",
                        &[V::quantized(q)],
                        &vec![("device".into(), V::Device(dev))],
                    )
                    .unwrap()
                };
                V::quantized(moved.as_quantized().unwrap().detach(&rt.native).unwrap())
            } else {
                V::tensor(weight.f_to_device(dev).unwrap())
            };
            check(
                &value,
                &g["state"][name],
                dev,
                &format!("{label}/{name}/factory"),
            );
            if shape.len() == 2 && name != "model.embed_tokens.weight" {
                state.linear_policies.insert(
                    name.strip_suffix(".weight").unwrap().into(),
                    LinearPolicy {
                        full_precision: true,
                        ..LinearPolicy::default()
                    },
                );
            }
            state.values.insert(name.clone(), value);
        }
        let model = q::QwenTextEncoder::from_parameters(config(&g["config"]), &state).unwrap();
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
            let batches = c["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| {
                    b.as_array()
                        .unwrap()
                        .iter()
                        .map(|t| q::WeightedToken {
                            token: q::PromptToken::Id(t[0].as_i64().unwrap()),
                            weight: t[1].as_f64().unwrap(),
                            word_id: 0,
                        })
                        .collect()
                })
                .collect();
            let prompt = q::TokenizedPrompt {
                batches,
                keep_vision: false,
            };
            let out = unsafe {
                model
                    .encode_prompt(
                        &rt,
                        &prompt,
                        q::TextConditioningOptions {
                            execution_device: dev,
                            intermediate_device: Device::Cpu,
                        },
                        &mut exec,
                    )
                    .unwrap()
            };
            let extra = V::Dict(
                out.attention_mask
                    .into_iter()
                    .map(|mask| ("attention_mask".into(), V::tensor(mask)))
                    .collect(),
            );
            check(
                &V::Tuple(vec![V::tensor(out.context), V::None, extra]),
                &c["output"],
                Device::Cpu,
                &format!("{label}/{}", c["label"]),
            );
            assert_eq!(serde_json::json!(exec.events), c["events"]);
            assert_eq!(exec.operations, c["operations"].as_u64().unwrap() as usize);
            complete += 1;
        }
        for c in g["core_calls"].as_array().unwrap() {
            exec.events.clear();
            exec.operations = 0;
            let x = data(&[1, 7, 256], dev, Kind::Float, 5);
            let positions = if c["position"].as_bool().unwrap() {
                Some(
                    Tensor::f_arange(21, (Kind::Int64, dev))
                        .unwrap()
                        .f_reshape([3, 7])
                        .unwrap(),
                )
            } else {
                None
            };
            let mask = Tensor::f_from_slice(&[1_i64, 1, 1, 1, 1, 0, 0])
                .unwrap()
                .f_unsqueeze(0)
                .unwrap()
                .f_to_device(dev)
                .unwrap();
            let visual = if c["deep"].as_bool().unwrap() {
                Some(
                    Tensor::from_slice(&[false, true, false, true, false, false, false])
                        .unsqueeze(0)
                        .to_device(dev),
                )
            } else {
                None
            };
            let ds = if visual.is_some() {
                vec![
                    data(&[2, 256], dev, Kind::Float, 13),
                    data(&[2, 256], dev, Kind::Float, 17),
                ]
            } else {
                vec![]
            };
            let intermediate = match c["mode"].as_str().unwrap() {
                "last" => q::Intermediate::Layer(-1),
                "all" => q::Intermediate::All,
                _ => q::Intermediate::Layers(vec![0, 1, model.config.layers as i64]),
            };
            let result = unsafe {
                model
                    .forward(
                        &rt,
                        &x,
                        q::TextForwardOptions {
                            attention_mask: Some(&mask),
                            position_ids: positions.as_ref(),
                            intermediate,
                            final_layer_norm_intermediate: false,
                            deepstack: &ds,
                            visual_mask: visual.as_ref(),
                        },
                        &mut exec,
                    )
                    .unwrap()
            };
            check(
                &V::Tuple(vec![
                    V::tensor(result.last),
                    result.intermediate.map(V::tensor).unwrap_or(V::None),
                ]),
                &c["output"],
                dev,
                &format!("{label}/core/{}", c["mode"]),
            );
            check(
                &V::tensor(x),
                &c["post_embeds"],
                dev,
                "original block output aliases embeddings",
            );
            assert_eq!(serde_json::json!(exec.events), c["events"]);
            assert_eq!(exec.operations, c["operations"].as_u64().unwrap() as usize);
            cores += 1;
        }
    }
    assert_eq!((complete, cores), (53, 30));
}
