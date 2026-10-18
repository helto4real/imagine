use super::{
    binding_tests::{check, restore},
    device, kitchen,
    qwen_text_encoder_tests::data,
};
use imagine_inference::{
    attention::{Attention, AttentionInput, AttentionMethod},
    checkpoint::{LinearPolicy, PreparedParameters},
    execution::Runtime,
    execution_value::Value as V,
    models::{moss_image as m, qwen_image21::TextEncoderExecution},
    native::Error,
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0)
    }
}
pub(super) struct Exec {
    device: String,
    pub(super) events: Vec<Value>,
    pub(super) operations: usize,
    attention: Attention,
}
impl TextEncoderExecution for Exec {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        Ok(())
    }
    fn queue_begin(&mut self, _layers: usize, _device: Device) -> Result<(), Error> {
        self.events
            .push(json!(["Queue",self.device,{"prefetch_dynamic_vbars":false}]));
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
pub(super) fn make_model(rt: &Runtime, group: &Value) -> m::Qwen35TextEncoder {
    let dev = device(group["device"].as_str().unwrap());
    let mut p = PreparedParameters::default();
    for (name, f) in group["factories"].as_object().unwrap() {
        let shape = f["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap())
            .collect::<Vec<_>>();
        let mut x = data(&shape, dev, Kind::BFloat16, f["offset"].as_i64().unwrap())
            .f_mul_scalar(0.125)
            .unwrap();
        if f["gated_norm"] == true {
            x = x.f_add_scalar(1.).unwrap()
        }
        let value = if f["quantized"] == true {
            V::quantized(
                unsafe {
                    rt.from_float(
                        &x,
                        "TensorWiseINT8Layout",
                        &vec![
                            ("is_weight".into(), V::Bool(true)),
                            ("per_channel".into(), V::Bool(true)),
                            ("convrot".into(), V::Bool(true)),
                            (
                                "convrot_groupsize".into(),
                                V::Int(if shape.last().unwrap() % 256 == 0 {
                                    256
                                } else {
                                    64
                                }),
                            ),
                            ("stochastic_rounding".into(), V::Int(0)),
                        ],
                    )
                    .unwrap()
                }
                .detach(&rt.native)
                .unwrap(),
            )
        } else {
            V::tensor(x)
        };
        check(&value, &group["state"][name], dev, name);
        if shape.len() == 2 && name != "model.embed_tokens.weight" {
            p.linear_policies.insert(
                name.strip_suffix(".weight").unwrap().into(),
                LinearPolicy {
                    full_precision: true,
                    ..LinearPolicy::default()
                },
            );
        }
        p.values.insert(name.clone(), value);
    }
    let c = &group["config"];
    let n = |name: &str| c[name].as_i64().unwrap();
    m::Qwen35TextEncoder::from_parameters(
        m::Qwen35Config {
            hidden: n("hidden_size"),
            layers: n("num_hidden_layers") as usize,
            heads: n("num_attention_heads"),
            kv_heads: n("num_key_value_heads"),
            head_dim: n("head_dim"),
            rotary_dim: (n("head_dim") as f64 * c["partial_rotary_factor"].as_f64().unwrap())
                as i64,
            key_heads: n("linear_num_key_heads"),
            value_heads: n("linear_num_value_heads"),
            key_dim: n("linear_key_head_dim"),
            value_dim: n("linear_value_head_dim"),
            conv_kernel: n("conv_kernel_size"),
            ..m::Qwen35Config::default()
        },
        &p,
    )
    .unwrap()
}

pub(super) fn make_exec(name: &str) -> Exec {
    Exec {
        device: name.into(),
        events: vec![],
        operations: 0,
        attention: Attention {
            method: AttentionMethod::Pytorch,
            priority_wrapper: true,
            attention_precision: None,
            low_precision_attention: false,
            model_device: device(name),
        },
    }
}

#[test]
fn moss_original_qwen35_hybrid_prefill_full_attention_gqa_int8_bf16_masks_and_input_lifetime_match()
{
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
        serde_json::from_str(include_str!("../../fixtures/moss-text.json")).unwrap();
    let mut count = 0;
    for group in fixture["groups"].as_array().unwrap() {
        let dev = device(group["device"].as_str().unwrap());
        let model = make_model(&rt, group);
        for (i, call) in group["calls"].as_array().unwrap().iter().enumerate() {
            let x = restore(&call["args"][0], dev).as_tensor().unwrap();
            let mask = restore(&call["args"][1], dev);
            let mask = if matches!(mask, V::None) {
                None
            } else {
                Some(mask.as_tensor().unwrap())
            };
            let mut e = make_exec(group["device"].as_str().unwrap());
            let label = format!("{dev:?}/int8={}/{i}", group["quantized"]);
            let out = unsafe { model.forward(&rt, &x, mask.as_ref(), None, &mut e) }
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            check(
                &V::tensor(out),
                &call["output"],
                dev,
                &format!("{label}/output"),
            );
            check(
                &V::tensor(x),
                &call["post_embeds"],
                dev,
                &format!("{label}/post-embeddings"),
            );
            assert_eq!(json!(e.events), call["events"], "{label}/lifecycle");
            assert_eq!(
                json!(e.operations),
                call["operations"],
                "{label}/interrupt-boundaries"
            );
            let input = restore(&call["args"][0], dev).as_tensor().unwrap();
            let mut services = m::MossStaticExecution::bind(
                &rt,
                &PreparedParameters::default(),
                e.attention.clone(),
            )
            .unwrap();
            let out =
                unsafe { model.forward(&rt, &input, mask.as_ref(), None, &mut services) }.unwrap();
            check(
                &V::tensor(out),
                &call["output"],
                dev,
                "bound static Qwen3.5 prefill",
            );
            check(
                &V::tensor(input),
                &call["post_embeds"],
                dev,
                "bound static input lifetime",
            );
            assert_eq!(services.operations, e.operations);
            count += 1;
        }
    }
    assert_eq!(count, 48);
}
