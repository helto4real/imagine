use super::{
    binding_tests::{check, restore},
    kitchen,
};
use imagine_inference::{
    attention::{Attention, AttentionMethod},
    checkpoint::PreparedParameters,
    execution::Runtime,
    models::qwen_image21 as q,
    triton_compiler::Compiler,
};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::{Device, Kind};
fn sequence(
    v: &imagine_inference::execution_value::Value,
) -> &[imagine_inference::execution_value::Value] {
    match v {
        imagine_inference::execution_value::Value::List(v)
        | imagine_inference::execution_value::Value::Tuple(v) => v,
        _ => panic!("Expected original sequence"),
    }
}
#[test]
fn qwen_image21_original_resident_attention_cache_services_and_cancellation_match() {
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    // Each separately isolated invocation starts before any Torch CUDA tensor:
    // Python-equivalent lazy initialization must precede allocator statistics.
    let cold = native.cuda_memory(0).unwrap();
    assert!(cold[1] >= cold[0]);
    assert!(cold[3] >= cold[2]);
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
    let mut forwards = 0;
    let mut cached = 0;
    for g in fixture["models"].as_array().unwrap() {
        let c = &g["config"];
        let n = |k: &str| c[k].as_i64().unwrap();
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
        let state = PreparedParameters::plain(
            g["state"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), restore(v, Device::Cpu)))
                .collect::<BTreeMap<_, _>>(),
        );
        let mut model = q::DiT::from_parameters(config, &state).unwrap();
        let mut cache = q::PrefixCache::new(
            &rt,
            q::CacheOptions::default(),
            q::ResidentCacheServices {
                native: rt.native.clone(),
            },
        );
        for call in g["calls"].as_array().unwrap() {
            if call["patches"] == true || call["training"] == true || call["backend"] != "pytorch" {
                continue;
            }
            let args = call["args"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| restore(v, Device::Cpu))
                .collect::<Vec<_>>();
            let x = args[0].as_tensor().unwrap();
            let dev = x.device();
            let enabled = call["cache_enabled"].as_bool().unwrap();
            if enabled && dev == Device::Cpu {
                continue;
            }
            if model.prefix_cache_enabled && !enabled {
                model.reset_prefix_cache(false, &mut cache).unwrap();
            }
            let case = call["label"].as_str().unwrap();
            if enabled && case.ends_with("/fill") {
                model.reset_prefix_cache(true, &mut cache).unwrap();
                cache.options.dtype = case.split('/').nth(1).unwrap().into();
                cache.options.device = "gpu".into();
            }
            if enabled && !model.prefix_cache_enabled {
                model.reset_prefix_cache(true, &mut cache).unwrap();
            }
            let refs = sequence(&args[3])
                .iter()
                .map(|v| v.as_tensor().unwrap())
                .collect::<Vec<_>>();
            let slots = sequence(&args[4])
                .iter()
                .map(|v| match v {
                    imagine_inference::execution_value::Value::Int(n) => *n,
                    _ => panic!("Expected original integer slot"),
                })
                .collect::<Vec<_>>();
            let attention = Attention {
                method: AttentionMethod::Pytorch,
                priority_wrapper: true,
                attention_precision: None,
                low_precision_attention: false,
                model_device: dev,
            };
            let mut exec =
                q::ResidentExecution::bind(&rt, &state, model.config.num_layers, attention)
                    .unwrap();
            let out = unsafe {
                model.forward(
                    &rt,
                    &x,
                    &args[1].as_tensor().unwrap(),
                    &args[2].as_tensor().unwrap(),
                    &refs,
                    &slots,
                    Some(&mut cache),
                    &mut exec,
                )
            }
            .unwrap();
            check(
                &imagine_inference::execution_value::Value::tensor(out),
                &call["output"],
                dev,
                &format!("Resident {}/{}", g["label"], call["label"]),
            );
            forwards += 1;
            if enabled {
                cached += 1;
                assert_eq!(
                    cache.memory_bytes() as u64,
                    call["cache_memory"].as_u64().unwrap()
                );
            }
        }
        model.reset_prefix_cache(false, &mut cache).unwrap();
        let attention = Attention {
            method: AttentionMethod::Pytorch,
            priority_wrapper: true,
            attention_precision: None,
            low_precision_attention: false,
            model_device: Device::Cuda(0),
        };
        let mut exec =
            q::ResidentExecution::bind(&rt, &state, model.config.num_layers, attention).unwrap();
        exec.cancelled
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(matches!(
            q::DiTExecution::check_interrupt(&mut exec),
            Err(imagine_inference::native::Error::ReferenceException {
                class: "InterruptProcessingException",
                ..
            })
        ));
    }
    assert!(forwards > 50);
    assert!(cached > 10);
    // The native query exposes precisely the counters used by the source
    // formula; empty-cache preserves live storage and clears inactive blocks.
    let held = tch::Tensor::ones([65536], (Kind::Float, Device::Cuda(0)));
    let before = rt.native.cuda_memory(0).unwrap();
    assert!(before[3] >= before[2]);
    assert!(before[1] >= before[0]);
    rt.native.cuda_soft_empty_cache(0).unwrap();
    assert_eq!(held.sum(Kind::Float).double_value(&[]), 65536.);
    let after = rt.native.cuda_memory(0).unwrap();
    assert!(after[3] >= after[2]);
}
