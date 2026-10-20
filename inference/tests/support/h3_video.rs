use imagine_inference::{
    attention::{Attention, AttentionInput, AttentionMethod},
    checkpoint::PreparedParameters,
    execution::Runtime,
    execution_value::Value as V,
    models::{
        minimax_h3::video_vae::{Config, VideoExecution, VideoVae},
        qwen_image21::TextEncoderExecution,
    },
    native::Error,
};
use serde_json::Value;
use std::collections::BTreeMap;
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
struct Exec {
    free: u64,
    calls: usize,
    device: Device,
}
impl VideoExecution for Exec {
    fn free_memory(&mut self, device: Device) -> Result<u64, Error> {
        assert_eq!(device, self.device);
        self.calls += 1;
        Ok(self.free)
    }
}
impl TextEncoderExecution for Exec {
    fn queue_begin(&mut self, _: usize, _: Device) -> Result<(), Error> {
        Ok(())
    }
    fn queue_end(&mut self, _: Device) -> Result<(), Error> {
        Ok(())
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
        unsafe {
            Attention {
                method: AttentionMethod::Pytorch,
                priority_wrapper: self.device != Device::Cpu,
                attention_precision: None,
                low_precision_attention: true,
                model_device: self.device,
            }
            .run(
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
#[test]
fn h3_complete_video_vae_matches_original_causal_encoder_vit_decoder_spatial_blending_and_temporal_streaming()
 {
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    let _grad = tch::no_grad_guard();
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _restore = Restore(tch::get_num_threads());
    tch::set_num_threads(2);
    let f: Value = serde_json::from_str(include_str!("../../fixtures/h3-video.json")).unwrap();
    let selected = VideoVae::new(PreparedParameters::default(), Config::default());
    for plan in f["selected_plans"].as_array().unwrap() {
        let shape: Vec<i64> = serde_json::from_value(plan["shape"].clone()).unwrap();
        assert_eq!(
            selected.decode_output_shape(&shape),
            serde_json::from_value::<Vec<i64>>(plan["output"].clone()).unwrap()
        );
        let (a, b) = selected.temporal_chunks(shape[2]);
        assert_eq!(
            vec![a, b],
            serde_json::from_value::<Vec<i64>>(plan["chunks"].clone()).unwrap()
        );
    }
    for tiles in f["selected_tiles"].as_array().unwrap() {
        let (a, b, c) = selected.split_tiles(tiles["length"].as_i64().unwrap());
        assert_eq!(
            vec![a, b, c],
            serde_json::from_value::<Vec<Vec<i64>>>(tiles["result"].clone()).unwrap()
        );
    }
    let mut total = 0;
    for entry in f["models"].as_array().unwrap() {
        let kind = super::kind(entry["dtype"].as_str().unwrap());
        let mut state = PreparedParameters::default();
        for (name, p) in entry["factories"].as_object().unwrap() {
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
                .insert(name.clone(), V::tensor(x.to_kind(kind)));
        }
        for (name, b) in entry["buffers"].as_object().unwrap() {
            state.values.insert(name.clone(), V::tensor(tensor(b)));
        }
        let mut model = VideoVae::new(
            state,
            serde_json::from_value::<Config>(entry["config"].clone()).unwrap(),
        );
        for plan in entry["plans"].as_array().unwrap() {
            let shape: Vec<i64> = serde_json::from_value(plan["shape"].clone()).unwrap();
            assert_eq!(
                model.decode_output_shape(&shape),
                serde_json::from_value::<Vec<i64>>(plan["output"].clone()).unwrap()
            );
            let (c, n) = model.temporal_chunks(shape[2]);
            assert_eq!(
                vec![c, n],
                serde_json::from_value::<Vec<i64>>(plan["chunks"].clone()).unwrap()
            );
        }
        for tiles in entry["tiles"].as_array().unwrap() {
            let (a, b, c) = model.split_tiles(tiles["length"].as_i64().unwrap());
            assert_eq!(
                vec![a, b, c],
                serde_json::from_value::<Vec<Vec<i64>>>(tiles["result"].clone()).unwrap()
            );
        }
        for (case_index, case) in entry["cases"].as_array().unwrap().iter().enumerate() {
            total += 1;
            model.config.tiling = case["tiling"].as_bool().unwrap();
            model.config.still_frame = case["still_frame"].as_i64();
            let input = tensor(&case["input"]);
            let mut exec = Exec {
                device: Device::Cpu,
                free: case["free_bytes"].as_u64().unwrap(),
                calls: 0,
            };
            let buffer = case["output_buffer"].as_bool().unwrap().then(|| {
                Tensor::empty(
                    model.decode_output_shape(&input.size()),
                    (Kind::Float, Device::Cpu),
                )
            });
            let out = unsafe {
                if case["operation"] == "encode" {
                    model.encode(&rt, &input, Device::Cpu, &mut exec)
                } else {
                    model.decode(&rt, &input, buffer.as_ref(), &mut exec)
                }
            }
            .unwrap();
            super::binding_tests::check(
                &V::tensor(out.shallow_clone()),
                &case["output"],
                Device::Cpu,
                &format!(
                    "H3 video {kind:?} #{case_index} {} {:?}",
                    case["operation"],
                    input.size()
                ),
            );
            assert_eq!(exec.calls, case["memory_calls"].as_u64().unwrap() as usize);
            if let Some(buffer) = buffer {
                assert_eq!(out.data_ptr(), buffer.data_ptr());
            }
            assert_eq!(
                super::bits(&input),
                super::exact_hex(&case["input"]["bytes"])
            );
        }
    }
    assert_eq!(total, 30);
}

#[test]
fn h3_int8_video_vae_preserves_convrot_fused_norm_swiglu_and_residual_rounding() {
    let rt = Runtime::frozen_partial(super::kitchen()).unwrap();
    imagine_inference::checkpoint::bind_comfy_layouts(&rt);
    let _grad = tch::no_grad_guard();
    struct Restore(i32);
    impl Drop for Restore {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _restore = Restore(tch::get_num_threads());
    tch::set_num_threads(2);
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/h3-video-quant.json")).unwrap();
    let checkpoint = f["checkpoint"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, t)| (n.clone(), tensor(t)))
        .collect::<BTreeMap<_, _>>();
    let spec = serde_json::from_value(f["spec"].clone()).unwrap();
    let loaded = serde_json::from_value(f["loaded"].clone()).unwrap();
    let state = imagine_inference::models::minimax_h3::checkpoint::prepare_with_schema(
        &rt,
        imagine_inference::models::minimax_h3::checkpoint::Component::VideoVae,
        &spec,
        &loaded,
        checkpoint,
    )
    .unwrap();
    let mut model = VideoVae::new(state, serde_json::from_value(f["config"].clone()).unwrap());
    assert_eq!(f["cases"].as_array().unwrap().len(), 3);
    for (i, case) in f["cases"].as_array().unwrap().iter().enumerate() {
        model.config.tiling = case["tiling"].as_bool().unwrap();
        let input = tensor(&case["input"]);
        let buffer = case["output_buffer"].as_bool().unwrap().then(|| {
            Tensor::empty(
                model.decode_output_shape(&input.size()),
                (Kind::Float, Device::Cpu),
            )
        });
        let mut exec = Exec {
            device: Device::Cpu,
            free: 4 * 128 * 2u64.pow(20),
            calls: 0,
        };
        let out = unsafe { model.decode(&rt, &input, buffer.as_ref(), &mut exec) }.unwrap();
        super::binding_tests::check(
            &V::tensor(out.shallow_clone()),
            &case["output"],
            Device::Cpu,
            &format!("H3 INT8 video #{i}"),
        );
        if let Some(buffer) = buffer {
            assert_eq!(out.data_ptr(), buffer.data_ptr());
        }
        assert_eq!(
            super::bits(&input),
            super::exact_hex(&case["input"]["bytes"])
        );
    }
}

#[test]
#[ignore = "Requires an explicit canonical GPU lease and own CUDA oracle fixture"]
fn h3_tiny_cuda_video_matches_original_selected_policy_without_python_in_native_worker() {
    use imagine_inference::{checkpoint::bind_comfy_layouts, triton_compiler::Compiler};
    use std::{path::Path, sync::Arc};
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
    bind_comfy_layouts(&rt);
    let _grad = tch::no_grad_guard();
    let fixture = std::env::var("IMAGINE_H3_GPU_FIXTURE").expect("Explicit own tiny CUDA fixture");
    let f: Value = serde_json::from_str(&std::fs::read_to_string(fixture).unwrap()).unwrap();
    assert_eq!(f["trained_weights"], 0);
    assert_eq!(f["cpu_only"], false);
    assert_eq!(f["policy"]["fp16_accumulation"], false);
    assert_eq!(f["policy"]["conv_workaround"], false);
    assert_eq!(f["kitchen_triton_enabled"], false);
    rt.backends.policy.disable("triton");
    let dev = Device::Cuda(0);
    let mut total = 0;
    for entry in f["float_models"].as_array().unwrap() {
        let kind = super::kind(entry["dtype"].as_str().unwrap());
        let mut state = PreparedParameters::default();
        for (name, p) in entry["factories"].as_object().unwrap() {
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
                .insert(name.clone(), V::tensor(x.to_kind(kind).to_device(dev)));
        }
        for (name, b) in entry["buffers"].as_object().unwrap() {
            state
                .values
                .insert(name.clone(), V::tensor(tensor(b).to_device(dev)));
        }
        let mut model = VideoVae::new(
            state,
            serde_json::from_value(entry["config"].clone()).unwrap(),
        );
        for (i, case) in entry["cases"].as_array().unwrap().iter().enumerate() {
            model.config.tiling = case["tiling"].as_bool().unwrap();
            model.config.still_frame = case["still_frame"].as_i64();
            let input = tensor(&case["input"]).to_device(dev);
            let mut exec = Exec {
                device: dev,
                free: case["free_bytes"].as_u64().unwrap(),
                calls: 0,
            };
            let buffer = case["output_buffer"].as_bool().unwrap().then(|| {
                Tensor::empty(
                    model.decode_output_shape(&input.size()),
                    (Kind::Float, Device::Cpu),
                )
            });
            let out = unsafe {
                if case["operation"] == "encode" {
                    model.encode(&rt, &input, dev, &mut exec)
                } else {
                    model.decode(&rt, &input, buffer.as_ref(), &mut exec)
                }
            }
            .unwrap();
            super::binding_tests::check(
                &V::tensor(out),
                &case["output"],
                dev,
                &format!("CUDA video {kind:?} #{i} {}", case["operation"]),
            );
            total += 1;
            assert_eq!(
                super::bits(&input),
                super::exact_hex(&case["input"]["bytes"])
            );
        }
    }
    let q = &f["quantized"];
    let checkpoint = q["checkpoint"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(n, t)| (n.clone(), tensor(t)))
        .collect::<BTreeMap<_, _>>();
    let spec = serde_json::from_value(q["spec"].clone()).unwrap();
    let loaded = serde_json::from_value(q["loaded"].clone()).unwrap();
    let mut state = imagine_inference::models::minimax_h3::checkpoint::prepare_with_schema(
        &rt,
        imagine_inference::models::minimax_h3::checkpoint::Component::VideoVae,
        &spec,
        &loaded,
        checkpoint,
    )
    .unwrap();
    for v in state.values.values_mut() {
        *v = if let Ok(t) = v.as_tensor() {
            V::tensor(t.to_device(dev))
        } else {
            unsafe {
                rt.op(
                    "aten::to.dtype_layout",
                    std::slice::from_ref(v),
                    &vec![("device".into(), V::Device(dev))],
                )
                .unwrap()
            }
        };
    }
    let mut model = VideoVae::new(state, serde_json::from_value(q["config"].clone()).unwrap());
    for (i, case) in q["cases"].as_array().unwrap().iter().enumerate() {
        model.config.tiling = case["tiling"].as_bool().unwrap();
        let input = tensor(&case["input"]).to_device(dev);
        let buffer = case["output_buffer"].as_bool().unwrap().then(|| {
            Tensor::empty(
                model.decode_output_shape(&input.size()),
                (Kind::Float, Device::Cpu),
            )
        });
        let mut exec = Exec {
            device: dev,
            free: 4 * 128 * 2u64.pow(20),
            calls: 0,
        };
        let out = unsafe { model.decode(&rt, &input, buffer.as_ref(), &mut exec) }.unwrap();
        super::binding_tests::check(
            &V::tensor(out),
            &case["output"],
            dev,
            &format!("CUDA INT8 video #{i}"),
        );
        total += 1;
    }
    assert_eq!(total, 33);
}
