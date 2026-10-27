use super::{
    binding_tests::{check, restore},
    kitchen,
    qwen_text_encoder_tests::data,
};
use imagine_inference::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value as V,
    models::moss_image as m, native::Error, triton_compiler::Compiler,
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

struct Exec {
    training: bool,
    patched: bool,
    events: Vec<String>,
    operations: usize,
    replacements: Vec<Value>,
}
impl m::MossExecution for Exec {
    fn training(&self) -> bool {
        self.training
    }
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        Ok(())
    }
    fn lifecycle(&mut self, event: m::MossLifecycle) -> Result<(), Error> {
        use m::MossLifecycle::*;
        self.events.push(match event {
            TextQueue | Queue => "Queue".into(),
            TextQueueEnd | QueueEnd => "QueueEnd".into(),
            TextBlock(i) => format!("Block(text/{i})"),
            Block(i) => format!("Block({i})"),
            GraphBegin => "GraphBegin".into(),
            GraphEnd => "GraphEnd".into(),
        });
        Ok(())
    }
    fn attention(
        &mut self,
        _rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: i64,
        _module: &str,
    ) -> Result<Tensor, Error> {
        let b = q.size()[0];
        let d = q.size()[2] / heads;
        let reshape = |x: &Tensor| {
            x.f_unsqueeze(3)?
                .f_reshape([b, -1, heads, d])?
                .f_transpose(1, 2)
        };
        let out = Tensor::f_scaled_dot_product_attention(
            &reshape(q)?,
            &reshape(k)?,
            &reshape(v)?,
            None::<&Tensor>,
            0.,
            false,
            None,
            false,
        )?;
        Ok(out.f_transpose(1, 2)?.f_reshape([b, -1, heads * d])?)
    }
    fn replace_block(
        &mut self,
        state: &m::MossState,
        x: &Tensor,
        _vec: &Tensor,
        modulation: &[Tensor],
        pe: &Tensor,
        original: &mut m::MossOriginalBlock<'_, Self>,
    ) -> Result<Option<Tensor>, Error> {
        if !self.patched || state.block_index != 0 {
            return Ok(None);
        }
        self.replacements.push(json!({"total_blocks":state.total_blocks,"block_type":"single","block_index":state.block_index}));
        Ok(Some(
            original(x, modulation, pe, self)?.f_mul_scalar(0.9375)?,
        ))
    }
    fn wrapper(
        &mut self,
        inputs: &m::MossInputs<'_>,
        original: &mut dyn FnMut(&m::MossInputs<'_>, &mut Self) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error> {
        let out = original(inputs, self)?;
        Ok(if self.patched {
            out.f_add_scalar(0.00390625)?
        } else {
            out
        })
    }
}
/// The group's prepared parameters and DiT, from its weight factories.
fn group_model(
    rt: &Runtime,
    group: &Value,
    label: &str,
    dev: Device,
) -> (PreparedParameters, m::MossDiT) {
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
        if f["norm"] == true {
            x = x.f_add_scalar(1.).unwrap()
        }
        let value = if f["quantized"] == true {
            let q = unsafe {
                rt.from_float(
                    &x,
                    "TensorWiseINT8Layout",
                    &vec![
                        ("is_weight".into(), V::Bool(true)),
                        ("per_channel".into(), V::Bool(true)),
                        ("convrot".into(), V::Bool(true)),
                        ("convrot_groupsize".into(), V::Int(64)),
                        ("stochastic_rounding".into(), V::Int(0)),
                    ],
                )
                .unwrap()
            };
            V::quantized(q.detach(&rt.native).unwrap())
        } else {
            V::tensor(x)
        };
        check(
            &value,
            &group["state"][name],
            dev,
            &format!("{label}/{name}/factory"),
        );
        p.values.insert(name.clone(), value);
    }
    let c = &group["config"];
    let n = |key: &str| c[key].as_i64().unwrap();
    let model = m::MossDiT::from_parameters(
        m::MossConfig {
            dim: n("dim"),
            ffn_dim: n("ffn_dim"),
            heads: n("num_heads"),
            layers: n("num_layers") as usize,
            text_dim: n("text_dim"),
            latent_channels: n("latent_channels"),
            fused_qkv: c["fused_qkv"].as_bool().unwrap(),
        },
        &p,
    )
    .unwrap();
    (p, model)
}
#[test]
fn moss_original_complete_dit_int8_bf16_fused_split_training_hooks_and_lifecycle_match() {
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
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
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-dit.json")).unwrap();
    let mut count = 0;
    for group in fixture["models"].as_array().unwrap() {
        let label = group["label"].as_str().unwrap();
        let dev = if label.starts_with("cpu/") {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let (p, model) = group_model(&rt, group, label, dev);
        for (i, call) in group["calls"].as_array().unwrap().iter().enumerate() {
            let args = call["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, dev).as_tensor().unwrap())
                .collect::<Vec<_>>();
            let mut exec = Exec {
                training: call["training"].as_bool().unwrap(),
                patched: call["patched"].as_bool().unwrap(),
                events: vec![],
                operations: 0,
                replacements: vec![],
            };
            let output = unsafe {
                model.forward(
                    &rt,
                    &m::MossInputs {
                        latent: &args[0],
                        timestep: &args[1],
                        context: &args[2],
                        references: None,
                    },
                    &mut exec,
                )
            }
            .unwrap_or_else(|err| panic!("{label}/{i}: {err}"));
            check(
                &V::tensor(output),
                &call["output"],
                dev,
                &format!("{label}/{i}/output"),
            );
            check(
                &V::tensor(args[0].shallow_clone()),
                &call["post_input"],
                dev,
                &format!("{label}/{i}/post-input"),
            );
            assert_eq!(json!(exec.events), call["events"], "{label}/{i}/lifecycle");
            assert_eq!(
                json!(exec.operations),
                call["operations"],
                "{label}/{i}/interrupt-boundaries"
            );
            assert_eq!(
                json!(exec.replacements),
                call["replacements"],
                "{label}/{i}/replacement-state"
            );
            if !exec.training && !exec.patched {
                let args = call["args"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| restore(v, dev).as_tensor().unwrap())
                    .collect::<Vec<_>>();
                let mut services = m::MossStaticExecution::bind(
                    &rt,
                    &p,
                    imagine_inference::attention::Attention {
                        method: imagine_inference::attention::AttentionMethod::Pytorch,
                        priority_wrapper: false,
                        attention_precision: None,
                        low_precision_attention: false,
                        model_device: dev,
                    },
                )
                .unwrap();
                let out = unsafe {
                    model.forward(
                        &rt,
                        &m::MossInputs {
                            latent: &args[0],
                            timestep: &args[1],
                            context: &args[2],
                            references: None,
                        },
                        &mut services,
                    )
                }
                .unwrap();
                check(
                    &V::tensor(out),
                    &call["output"],
                    dev,
                    "bound static MOSS DiT",
                );
                assert_eq!(services.operations, exec.operations);
                services
                    .cancelled
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                assert!(matches!(
                    m::MossExecution::check_interrupt(&mut services),
                    Err(Error::ReferenceException {
                        class: "InterruptProcessingException",
                        ..
                    })
                ));
            }
            count += 1;
        }
    }
    assert_eq!(count, 72);
}
#[test]
fn moss_reference_edits_precompute_keys_values_and_match_426f5eea() {
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
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
    let _grad = tch::no_grad_guard();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/reference-20261027/moss-dit.json"
    ))
    .unwrap();
    let mut count = 0;
    for group in fixture["models"].as_array().unwrap() {
        let label = group["label"].as_str().unwrap();
        let dev = if label.starts_with("cpu/") {
            Device::Cpu
        } else {
            Device::Cuda(0)
        };
        let (_, model) = group_model(&rt, group, label, dev);
        for edit in group["edits"].as_array().unwrap() {
            let method = edit["method"].as_str().unwrap();
            let label = format!(
                "{label}/{method}/{}",
                edit["args"].as_array().unwrap().len() - 3
            );
            let args = edit["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, dev).as_tensor().unwrap())
                .collect::<Vec<_>>();
            let mut exec = Exec {
                training: false,
                patched: false,
                events: vec![],
                operations: 0,
                replacements: vec![],
            };
            let kv = unsafe { model.preprocess_refs(&rt, &args[3..], &mut exec) }
                .unwrap_or_else(|err| panic!("{label}: {err}"));
            check(
                &V::tensor(kv.shallow_clone()),
                &edit["ref_kv"],
                dev,
                &format!("{label}/ref-kv"),
            );
            assert_eq!(
                json!(exec.events),
                edit["kv_events"],
                "{label}/kv-lifecycle"
            );
            let layout = match method {
                "index_timestep_zero_text_start" => m::MossEditLayout::TextStart,
                _ => m::MossEditLayout::ImageFrame,
            };
            let output = unsafe {
                model.forward(
                    &rt,
                    &m::MossInputs {
                        latent: &args[0],
                        timestep: &args[1],
                        context: &args[2],
                        references: Some(m::MossReferences { kv: &kv, layout }),
                    },
                    &mut exec,
                )
            }
            .unwrap_or_else(|err| panic!("{label}: {err}"));
            check(
                &V::tensor(output),
                &edit["output"],
                dev,
                &format!("{label}/output"),
            );
            check(
                &V::tensor(args[0].shallow_clone()),
                &edit["post_input"],
                dev,
                &format!("{label}/post-input"),
            );
            // The original preprocesses inside the forward, so its events include both.
            assert_eq!(json!(exec.events), edit["events"], "{label}/lifecycle");
            count += 1;
        }
    }
    assert_eq!(count, 32);
}
