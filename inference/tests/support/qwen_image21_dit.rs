use super::{
    binding_tests::{check, restore},
    kitchen,
};
use imagine_inference::{
    execution::Runtime, execution_value::Value as V, models::qwen_image21 as q, native::Error,
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct ResidentIo;
struct Ready(Tensor);
impl q::PendingCacheTensor for Ready {
    fn synchronize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn wait(self: Box<Self>, _device: Device) -> Result<Tensor, Error> {
        Ok(self.0)
    }
}
impl q::CacheServices for ResidentIo {
    fn patcher_free_memory(&mut self, _device: Device) -> Result<f64, Error> {
        Ok((1_u64 << 50) as f64)
    }
    fn free_memory(&mut self, _device: Device) -> Result<f64, Error> {
        Ok((1_u64 << 50) as f64)
    }
    fn ensure_pin_budget(&mut self, _bytes: usize, _evict_active: bool) -> Result<bool, Error> {
        Ok(true)
    }
    fn pin(&mut self, _t: &Tensor, _evict_active: bool) -> Result<bool, Error> {
        Ok(false)
    }
    fn unpin(&mut self, _t: &Tensor) -> Result<(), Error> {
        Ok(())
    }
    fn prefetch(
        &mut self,
        t: &Tensor,
        _block: usize,
        kind: Option<Kind>,
        device: Device,
    ) -> Result<Box<dyn q::PendingCacheTensor>, Error> {
        assert_eq!(
            t.device(),
            device,
            "This fixture explicitly exercises the original same-device transfer policy"
        );
        Ok(Box::new(Ready(t.f_to_device_(
            device,
            kind.unwrap_or(t.kind()),
            true,
            false,
        )?)))
    }
    fn clear_staging(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
struct Exec {
    training: bool,
    patches: bool,
    kitchen: bool,
    events: Vec<String>,
    state: Value,
    operations: usize,
    interrupt_at: Option<usize>,
}
impl q::DiTExecution for Exec {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        if self.interrupt_at == Some(self.operations) {
            self.interrupt_at = None;
            return Err(Error::ReferenceException {
                class: "InterruptProcessingException",
                message: String::new(),
            });
        }
        Ok(())
    }
    fn training(&self) -> bool {
        self.training
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: i64,
        mask: Option<&Tensor>,
        _block: usize,
        preferred: Option<&Value>,
    ) -> Result<Tensor, Error> {
        let preferred_int8 = preferred.is_some_and(|p| {
            p.as_array()
                .is_some_and(|a| a.iter().any(|v| v["attention"] == "comfy_kitchen_int8"))
        });
        let kitchen = self.kitchen
            || (!self.patches
                && preferred_int8
                && rt.int8_attention_available(Some(q.device()))?);
        let b = q.size()[0];
        let d = q.size()[2] / heads;
        let shape = |t: &Tensor| {
            t.f_unsqueeze(3)?
                .f_reshape([b, -1, heads, d])?
                .f_transpose(1, 2)
        };
        let q = shape(q)?;
        let k = shape(k)?;
        let v = shape(v)?;
        let mask = mask.map(|m| m.f_unsqueeze(0)?.f_unsqueeze(1)).transpose()?;
        assert!(
            q.numel() < 1024 * 128,
            "The original small-SDPA branch is explicit in this fixture"
        );
        let out = if kitchen {
            unsafe { rt.int8_attention(&q, &k, &v, None, mask.as_ref())? }
        } else {
            Tensor::f_scaled_dot_product_attention(
                &q,
                &k,
                &v,
                mask.as_ref(),
                0.,
                false,
                None,
                false,
            )?
        };
        let out = out.f_transpose(1, 2)?.f_reshape([b, -1, heads * d])?;
        Ok(if self.patches {
            out.f_mul_scalar(0.96875)?
        } else {
            out
        })
    }
    fn forward_state(&mut self, state: &q::ForwardState) -> Result<(), Error> {
        self.state = serde_json::json!({"total_blocks":state.total_blocks,"block_type":state.block_type,"block_index":state.block_index});
        Ok(())
    }
    fn lifecycle(&mut self, event: q::Lifecycle) -> Result<(), Error> {
        self.events.push(format!("{event:?}"));
        Ok(())
    }
    fn post_input(&mut self, x: Tensor, pe: Tensor) -> Result<(Tensor, Tensor), Error> {
        Ok((
            if self.patches {
                x.f_add_scalar(0.03125)?
            } else {
                x
            },
            pe,
        ))
    }
    fn has_attn_patch(&self) -> bool {
        self.patches
    }
    fn hooked(&self) -> bool {
        self.patches
    }
    fn attn_patch(
        &mut self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pe: Tensor,
        _slice: [i64; 2],
        _state: &q::ForwardState,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor), Error> {
        Ok((q.f_mul_scalar(0.875)?, k, v.f_add_scalar(0.015625)?, pe))
    }
    fn single_block(
        &mut self,
        x: Tensor,
        _input: &Tensor,
        _state: &q::ForwardState,
    ) -> Result<Tensor, Error> {
        Ok(if self.patches {
            x.f_add_scalar(0.0078125)?
        } else {
            x
        })
    }
    fn replace_block(
        &mut self,
        state: &q::ForwardState,
        x: &Tensor,
        _temb: &Tensor,
        pe: &Tensor,
        _modulation: &q::Modulation,
        _prefix: i64,
        original: &mut dyn FnMut(Tensor, Tensor, &mut Self) -> Result<Tensor, Error>,
    ) -> Result<Option<Tensor>, Error> {
        if self.patches && state.block_index == 0 {
            Ok(Some(
                original(x.shallow_clone(), pe.shallow_clone(), self)?.f_mul_scalar(0.9375)?,
            ))
        } else {
            Ok(None)
        }
    }
    fn wrapper(
        &mut self,
        input: &q::ForwardInputs<'_>,
        original: &mut dyn FnMut(&q::ForwardInputs<'_>, &mut Self) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error> {
        let out = original(input, self)?;
        Ok(if self.patches {
            out.f_add_scalar(0.00390625)?
        } else {
            out
        })
    }
}
#[test]
fn qwen_image21_original_complete_dit_bf16_int8_turbo_hooks_cache_and_attention_match() {
    complete_dit_fixture(
        include_str!("../../fixtures/qwen-image21-dit.json"),
        209,
        20,
        4,
    );
}
#[test]
fn qwen_image21_original_complete_dit_fp16_fp32_hooks_cache_and_attention_match() {
    complete_dit_fixture(
        include_str!("../../fixtures/qwen-image21-dit-precision.json"),
        82,
        0,
        0,
    );
}
fn complete_dit_fixture(source: &str, forwards: usize, interruptions: usize, control_count: usize) {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let _dtype = super::triton_default_dtype_tests::DefaultDtype::preserve(native.clone());
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
    let fixture: Value = serde_json::from_str(source).unwrap();
    let mut count = 0;
    let mut interrupted = 0;
    for group in fixture["models"].as_array().unwrap() {
        let c = &group["config"];
        let n = |key: &str| c[key].as_i64().unwrap();
        let state = group["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), restore(v, Device::Cpu)))
            .collect::<BTreeMap<_, _>>();
        let config = q::DiTConfig {
            in_channels: n("in_channels"),
            out_channels: n("out_channels"),
            num_layers: n("num_layers") as usize,
            head_dim: n("attention_head_dim"),
            heads: n("num_attention_heads"),
            context_dim: n("context_in_dim"),
            mlp_ratio: n("mlp_ratio"),
            axes: c["axes_dims_rope"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            eps: c["eps"].as_f64().unwrap(),
            fused_mlp: c["fused_mlp"].as_bool().unwrap(),
        };
        let mut model = q::DiT::from_prepared(config, &state).unwrap();
        let mut cache = q::PrefixCache::new(&rt, q::CacheOptions::default(), ResidentIo);
        for call in group["calls"].as_array().unwrap() {
            let label = format!(
                "{}/{}",
                group["label"].as_str().unwrap(),
                call["label"].as_str().unwrap()
            );
            eprintln!("DiT: {label}");
            let args = call["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, Device::Cpu))
                .collect::<Vec<_>>();
            let t = |i: usize| args[i].as_tensor().unwrap();
            let V::List(ref_values) = &args[3] else {
                panic!()
            };
            let refs = ref_values
                .iter()
                .map(|v| v.as_tensor().unwrap())
                .collect::<Vec<_>>();
            let V::List(slot_values) = &args[4] else {
                panic!()
            };
            let slots = slot_values
                .iter()
                .map(|v| if let V::Int(n) = v { *n } else { panic!() })
                .collect::<Vec<_>>();
            let enabled = call["cache_enabled"].as_bool().unwrap();
            if model.prefix_cache_enabled && !enabled {
                model.reset_prefix_cache(false, &mut cache).unwrap();
            }
            model.prefix_cache_enabled = enabled;
            let case = call["label"].as_str().unwrap();
            if case.ends_with("/fill") {
                model.reset_prefix_cache(true, &mut cache).unwrap();
                cache.options.dtype = case.split('/').nth(1).unwrap().into();
                cache.options.device = if matches!(t(0).device(), Device::Cuda(_)) {
                    "gpu"
                } else {
                    "cpu"
                }
                .into();
            }
            let mut exec = Exec {
                training: call["training"].as_bool().unwrap(),
                patches: call["patches"].as_bool().unwrap(),
                kitchen: call["backend"] == "kitchen",
                events: vec![],
                state: Value::Null,
                operations: 0,
                interrupt_at: None,
            };
            let out = unsafe {
                model.forward(
                    &rt,
                    &t(0),
                    &t(1),
                    &t(2),
                    &refs,
                    &slots,
                    Some(&mut cache),
                    &mut exec,
                )
            }
            .unwrap_or_else(|e| panic!("{label}: {e}"));
            check(&V::tensor(out), &call["output"], Device::Cpu, &label);
            assert_eq!(
                exec.operations as u64,
                call["operations"].as_u64().unwrap(),
                "{label}: original per-operation interruption points"
            );
            assert_eq!(
                exec.state, call["state"],
                "{label}: original transformer_options mutation"
            );
            assert_eq!(
                serde_json::json!(exec.events),
                call["events"],
                "{label}: original lifecycle order"
            );
            assert_eq!(
                cache.memory_bytes() as u64,
                call["cache_memory"].as_u64().unwrap(),
                "{label}: stored payload bytes"
            );
            count += 1;
        }
        model.reset_prefix_cache(false, &mut cache).unwrap();
        for case in group["interruptions"].as_array().unwrap() {
            let tensors = case["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, Device::Cpu).as_tensor().unwrap())
                .collect::<Vec<_>>();
            let mut exec = Exec {
                training: false,
                patches: false,
                kitchen: false,
                events: vec![],
                state: Value::Null,
                operations: 0,
                interrupt_at: Some(case["interrupt_at"].as_u64().unwrap() as usize),
            };
            if let Some(phase) = case["cache_phase"].as_str() {
                cache.options = q::CacheOptions::default();
                model.reset_prefix_cache(true, &mut cache).unwrap();
                if phase == "hit" {
                    let mut fill = Exec {
                        training: false,
                        patches: false,
                        kitchen: false,
                        events: vec![],
                        state: Value::Null,
                        operations: 0,
                        interrupt_at: None,
                    };
                    let _ = unsafe {
                        model.forward(
                            &rt,
                            &tensors[0],
                            &tensors[1],
                            &tensors[2],
                            &[],
                            &[],
                            Some(&mut cache),
                            &mut fill,
                        )
                    }
                    .unwrap();
                }
            }
            let result = unsafe {
                model.forward(
                    &rt,
                    &tensors[0],
                    &tensors[1],
                    &tensors[2],
                    &[],
                    &[],
                    Some(&mut cache),
                    &mut exec,
                )
            };
            let Error::ReferenceException { class, message } = result.unwrap_err() else {
                panic!("Expected original interruption")
            };
            assert_eq!(class, case["error"].as_str().unwrap());
            assert_eq!(message, case["message"].as_str().unwrap());
            assert_eq!(exec.operations as u64, case["operations"].as_u64().unwrap());
            assert_eq!(serde_json::json!(exec.events), case["events"]);
            assert!(exec.interrupt_at.is_none());
            assert_eq!(case["flag_after"], false);
            if let Some(expected) = case["cache_memory"].as_u64() {
                assert_eq!(cache.memory_bytes() as u64, expected);
            }
            model.reset_prefix_cache(false, &mut cache).unwrap();
            interrupted += 1;
        }
    }
    let controls = fixture["controls"].as_array().unwrap();
    assert_eq!(controls.len(), control_count);
    for group in controls {
        let state = group["state"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), restore(v, Device::Cpu)))
            .collect::<BTreeMap<_, _>>();
        let config = q::DiTConfig {
            num_layers: 2,
            head_dim: 64,
            heads: 1,
            axes: [16, 24, 24],
            ..Default::default()
        };
        let model = q::FunControl::from_prepared(&config, &state).unwrap();
        let args = group["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cpu))
            .collect::<Vec<_>>();
        let t = |i: usize| args[i].as_tensor().unwrap();
        let mut init_exec = Exec {
            training: false,
            patches: false,
            kitchen: false,
            events: vec![],
            state: Value::Null,
            operations: 0,
            interrupt_at: None,
        };
        let mut stream =
            unsafe { model.init_stream(&rt, &t(0), &t(1), 2, &mut init_exec) }.unwrap();
        let label = group["label"].as_str().unwrap();
        check(
            &V::tensor(stream.shallow_clone()),
            &group["initial"],
            Device::Cpu,
            label,
        );
        let modulation = q::Modulation::from_projection(t(2)).unwrap();
        let segments = vec![
            q::Segment {
                start: 0,
                end: 2,
                mask: Some(Tensor::ones([2, 2], (Kind::Bool, t(0).device())).tril(0)),
            },
            q::Segment {
                start: 2,
                end: 8,
                mask: None,
            },
        ];
        let mut exec = Exec {
            training: false,
            patches: false,
            kitchen: false,
            events: vec![],
            state: Value::Null,
            operations: 0,
            interrupt_at: None,
        };
        for i in 0..2 {
            let mut route = q::AttentionRoute::Segments {
                segments: &segments,
                store: None,
            };
            let step = unsafe {
                model.step(
                    &rt,
                    i,
                    stream,
                    &modulation,
                    &t(3),
                    &mut route,
                    2,
                    &q::ForwardState {
                        total_blocks: 2,
                        block_index: i,
                        block_type: "single",
                    },
                    &mut exec,
                )
            }
            .unwrap();
            stream = step.0;
            check(
                &V::Tuple(vec![V::tensor(stream.shallow_clone()), V::tensor(step.1)]),
                &group["steps"][i],
                Device::Cpu,
                label,
            );
        }
    }
    assert_eq!(count, forwards);
    assert_eq!(interrupted, interruptions);
}

struct PolicyIo {
    free: f64,
    budget: bool,
    trace: Vec<Value>,
}
fn device_name(d: Device) -> &'static str {
    if matches!(d, Device::Cuda(_)) {
        "cuda"
    } else {
        "cpu"
    }
}
impl q::CacheServices for PolicyIo {
    fn patcher_free_memory(&mut self, d: Device) -> Result<f64, Error> {
        self.trace
            .push(serde_json::json!(["patcher", device_name(d)]));
        Ok(self.free)
    }
    fn free_memory(&mut self, d: Device) -> Result<f64, Error> {
        self.trace.push(serde_json::json!(["free", device_name(d)]));
        Ok(self.free)
    }
    fn ensure_pin_budget(&mut self, bytes: usize, evict: bool) -> Result<bool, Error> {
        self.trace
            .push(serde_json::json!(["pin_budget", bytes, evict]));
        Ok(self.budget)
    }
    fn pin(&mut self, _t: &Tensor, _evict: bool) -> Result<bool, Error> {
        panic!("Selection must not pin block storage")
    }
    fn unpin(&mut self, _t: &Tensor) -> Result<(), Error> {
        panic!("A fresh cache has no pinned blocks")
    }
    fn prefetch(
        &mut self,
        _t: &Tensor,
        _block: usize,
        _kind: Option<Kind>,
        _device: Device,
    ) -> Result<Box<dyn q::PendingCacheTensor>, Error> {
        panic!("Selection must not prefetch")
    }
    fn clear_staging(&mut self) -> Result<(), Error> {
        Ok(())
    }
}
#[test]
fn qwen_image21_original_prefix_cache_selection_preserves_memory_thresholds_and_policy() {
    use q::PrefixKv;
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
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-dit.json")).unwrap();
    let cases = fixture["cache_policies"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for c in cases {
        let device = if c["device"] == "cuda" {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        let services = PolicyIo {
            free: c["free"].as_f64().unwrap(),
            budget: c["budget"].as_bool().unwrap(),
            trace: vec![],
        };
        let options = q::CacheOptions {
            device: c["store"].as_str().unwrap().into(),
            dtype: c["dtype"].as_str().unwrap().into(),
        };
        let mut cache = q::PrefixCache::new(&rt, options, services);
        let key = Tensor::from_slice(&[1_f32, 2.])
            .reshape([1, 2])
            .to_device(device);
        let result = cache.select(&key, 4096, 2, device).unwrap();
        assert_eq!(
            match result {
                q::CacheSelection::Disabled => "disabled",
                q::CacheSelection::Fill => "fill",
                q::CacheSelection::Hit => "hit",
            },
            c["result"].as_str().unwrap(),
            "{c}"
        );
        assert_eq!(serde_json::json!(cache.services.trace), c["trace"], "{c}");
        assert_eq!(
            cache.store_device().map(device_name),
            c["allocated_store"].as_str(),
            "{c}"
        );
    }
}
