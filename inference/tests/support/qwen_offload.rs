use imagine_inference::models::qwen_image21 as q;
use serde_json::Value;
#[test]
fn qwen_image21_original_static_offload_placements_unloads_and_budgets_match() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-offload.json")).unwrap();
    assert_eq!(fixture["safety"]["all_parameters_remained_meta"], true);
    assert_eq!(fixture["safety"]["real_allocation_guard_tested"], true);
    assert!(fixture["safety"]["peak_rss_kib"].as_u64().unwrap() < 1024 * 1024);
    let modules = q::bf16_dit_modules().unwrap();
    assert_eq!(serde_json::to_value(&modules).unwrap(), fixture["modules"]);
    for case in fixture["cases"].as_array().unwrap() {
        let initial = q::StaticPlacement::load(
            &modules,
            case["budget"].as_f64().unwrap(),
            case["full_load"].as_bool().unwrap(),
        );
        let verify = |p: &q::StaticPlacement, expected: &Value| {
            assert_eq!(
                serde_json::to_value(&p.resident).unwrap(),
                expected["resident"]
            );
            assert_eq!(p.loaded_bytes, expected["loaded_bytes"].as_u64().unwrap());
            assert_eq!(p.buffer_bytes, expected["buffer_bytes"].as_u64().unwrap());
        };
        verify(&initial, case);
        for unload in case["unloads"].as_array().unwrap() {
            let mut p = initial.clone();
            let freed = p.unload(&modules, unload["request"].as_f64().unwrap());
            assert_eq!(freed, unload["freed"].as_u64().unwrap());
            verify(&p, unload);
        }
        for reload in case["reloads"].as_array().unwrap() {
            verify(
                &initial.reload(&modules, reload["extra"].as_f64().unwrap()),
                reload,
            );
        }
    }
    for case in fixture["budgets"].as_array().unwrap() {
        assert_eq!(
            q::static_load_budget(
                case["free"].as_f64().unwrap(),
                case["loaded"].as_f64().unwrap(),
                case["minimum"].as_f64()
            ),
            case["budget"].as_f64().unwrap()
        );
    }
}
#[test]
fn qwen_image21_original_static_host_registration_copy_and_release_match() {
    use tch::{Device, Kind, Tensor};
    let native = super::kitchen();
    native.cuda_memory(0).unwrap();
    let input = Tensor::arange(1024, (Kind::Float, Device::Cpu));
    assert!(!input.f_is_pinned(Device::Cuda(0)).unwrap());
    assert!(native.host_registration(&input, true).unwrap());
    assert!(input.f_is_pinned(Device::Cuda(0)).unwrap());
    assert!(!native.host_registration(&input, true).unwrap());
    let mut out = native.empty_like(&input, Device::Cuda(0)).unwrap();
    native.copy_tensor(&mut out, &input, true).unwrap();
    native.synchronize_device(0).unwrap();
    assert!(out.to_device(Device::Cpu).equal(&input));
    assert!(native.host_registration(&input, false).unwrap());
    assert!(!input.f_is_pinned(Device::Cuda(0)).unwrap());
}

#[test]
fn qwen_image21_original_bf16_cpu_and_mixed_weights_preserve_cuda_forward_bytes() {
    use super::{
        binding_tests::{check, restore},
        kitchen,
    };
    use imagine_inference::{
        attention::{Attention, AttentionMethod},
        checkpoint::PreparedParameters,
        execution::Runtime,
        execution_value::Value as V,
        triton_compiler::Compiler,
    };
    use std::{path::Path, sync::Arc};
    use tch::Device;
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    native.cuda_memory(0).unwrap();
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
    let mut count = 0;
    for g in fixture["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|g| g["label"].as_str().unwrap().starts_with("cuda/bf16/"))
    {
        for mixed in [false, true] {
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
                    .map(|x| x.as_i64().unwrap())
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
                    .enumerate()
                    .map(|(i, (n, v))| {
                        let t = restore(v, Device::Cpu).as_tensor().unwrap().to_device(
                            if mixed && i % 3 == 0 && !n.ends_with(".comfy_attention.config") {
                                Device::Cuda(0)
                            } else {
                                Device::Cpu
                            },
                        );
                        (n.clone(), V::tensor(t))
                    })
                    .collect(),
            );
            let model = q::DiT::from_parameters(config, &state).unwrap();
            for call in g["calls"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["label"] == "plain" || c["label"] == "reference")
            {
                let args = call["args"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| restore(v, Device::Cpu))
                    .collect::<Vec<_>>();
                let sequence = |v: &V| match v {
                    V::List(x) | V::Tuple(x) => x.clone(),
                    _ => panic!("Expected sequence"),
                };
                let refs = sequence(&args[3])
                    .iter()
                    .map(|v| v.as_tensor().unwrap())
                    .collect::<Vec<_>>();
                let slots = sequence(&args[4])
                    .iter()
                    .map(|v| match v {
                        V::Int(x) => *x,
                        _ => panic!("Expected slot"),
                    })
                    .collect::<Vec<_>>();
                let a = Attention {
                    method: AttentionMethod::Pytorch,
                    priority_wrapper: true,
                    attention_precision: None,
                    low_precision_attention: false,
                    model_device: Device::Cuda(0),
                };
                let mut exec =
                    q::ResidentExecution::bind(&rt, &state, model.config.num_layers, a).unwrap();
                let out = unsafe {
                    model.forward(
                        &rt,
                        &args[0].as_tensor().unwrap(),
                        &args[1].as_tensor().unwrap(),
                        &args[2].as_tensor().unwrap(),
                        &refs,
                        &slots,
                        None,
                        &mut exec,
                    )
                }
                .unwrap();
                check(
                    &V::tensor(out),
                    &call["output"],
                    Device::Cuda(0),
                    "BF16 CPU/mixed offload",
                );
                count += 1;
            }
        }
    }
    assert_eq!(count, 12);
}
