use super::{binding_tests::check, device, kitchen};
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
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
struct Exec {
    operations: usize,
    observed: BTreeMap<String, Tensor>,
    attention: Attention,
}
impl q::TextEncoderExecution for Exec {
    fn observe(&mut self, name: &str, t: &Tensor) -> Result<(), Error> {
        self.observed.insert(name.into(), t.shallow_clone());
        Ok(())
    }
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.operations += 1;
        Ok(())
    }
    fn queue_begin(&mut self, _layers: usize, _device: Device) -> Result<(), Error> {
        panic!("Vision has no language prefetch queue")
    }
    fn queue_end(&mut self, _device: Device) -> Result<(), Error> {
        panic!("Vision has no language prefetch queue")
    }
    fn block(
        &mut self,
        _i: usize,
        _f: &mut dyn FnMut(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        panic!("Vision has no language prefetch block")
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
fn data(shape: &[i64], dev: Device, kind: Kind, offset: i64) -> Tensor {
    Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev))
        .g_add_scalar(offset)
        .remainder(29)
        .g_mul_scalar(0.03125)
        .g_sub_scalar(0.40625)
        .reshape(shape)
        .to_kind(kind)
}
fn numbers(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap())
        .collect()
}
#[test]
fn qwen_image21_original_qwen3vl_vision_transformer_positions_mergers_and_deepstack_match() {
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
    rt.backends.policy.disable("triton");
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-vision-encoder.json"
    ))
    .unwrap();
    let groups = fixture["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 9);
    let mut count = 0;
    for g in groups {
        let dev = device(g["device"].as_str().unwrap());
        let (state, config) = prepare(&rt, g, &g["config"]);
        let model = q::QwenVisionEncoder::from_parameters(config, &state).unwrap();
        let mut exec = Exec {
            operations: 0,
            observed: BTreeMap::new(),
            attention: Attention {
                method: AttentionMethod::Pytorch,
                priority_wrapper: true,
                attention_precision: None,
                low_precision_attention: true,
                model_device: Device::Cuda(0),
            },
        };
        for call in g["calls"].as_array().unwrap() {
            let grid = call["grid"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| numbers(v).try_into().unwrap())
                .collect::<Vec<[i64; 3]>>();
            let label = format!("{}/{}/{}", g["device"], g["variant"], json!(grid));
            check(
                &V::tensor(model.rotary_positions(&grid).unwrap()),
                &call["rotary"],
                dev,
                &label,
            );
            check(
                &V::tensor(unsafe { model.learned_positions(&rt, &grid, &mut exec).unwrap() }),
                &call["learned"],
                dev,
                &format!("{label}/learned"),
            );
            exec.operations = 0;
            exec.observed.clear();
            let len = grid.iter().map(|g| g[0] * g[1] * g[2]).sum::<i64>();
            let x = data(&[len, 1536], dev, Kind::Float, 5);
            let (merged, deep) = unsafe { model.forward(&rt, &x, &grid, &mut exec).unwrap() };
            for (n, e) in call["observed"].as_object().unwrap() {
                check(
                    &V::tensor(exec.observed[n].shallow_clone()),
                    e,
                    dev,
                    &format!("{label}/{n}"),
                );
            }
            check(
                &V::Tuple(vec![
                    V::tensor(merged),
                    V::List(deep.into_iter().map(V::tensor).collect()),
                ]),
                &call["output"],
                dev,
                &label,
            );
            assert_eq!(
                exec.operations,
                call["operations"].as_u64().unwrap() as usize,
                "{label}: original operation checkpoints"
            );
            count += 1;
        }
    }
    assert_eq!(count, 27);
}

pub(super) fn prepare(
    rt: &Runtime,
    g: &Value,
    c: &Value,
) -> (PreparedParameters, q::VisionEncoderConfig) {
    let dev = device(g["device"].as_str().unwrap());
    let mut state = PreparedParameters::default();
    for (name, f) in g["factories"].as_object().unwrap() {
        let shape = numbers(&f["shape"]);
        let mut w = data(
            &shape,
            Device::Cpu,
            Kind::Float,
            f["offset"].as_i64().unwrap(),
        );
        w = if f["norm"] == true {
            w.g_mul_scalar(0.25).g_add_scalar(1.)
        } else {
            w.g_mul_scalar(0.125)
        };
        w = w.to_kind(Kind::BFloat16);
        let value = if f["quantized"] == true {
            let quant = if g["variant"] == "float8_e4m3fn" {
                let scale = Tensor::full([], 0.125, (Kind::Float, Device::Cpu));
                let payload = unsafe {
                    rt.call_kitchen(
                        "quantize_per_tensor_fp8",
                        &vec![
                            ("x".into(), V::tensor(w.shallow_clone())),
                            ("scale".into(), V::tensor(scale.shallow_clone())),
                            ("output_type".into(), V::DType(Kind::Float8e4m3fn)),
                        ],
                    )
                    .unwrap()
                }
                .as_tensor()
                .unwrap();
                let mut q =
                    imagine_inference::layouts::fp8_parts(payload, scale, w.kind(), w.size())
                        .unwrap();
                q.layout = "TensorCoreFP8E4M3Layout".into();
                q
            } else {
                unsafe {
                    rt.from_float(
                        &w,
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
                    &[V::quantized(quant)],
                    &vec![("device".into(), V::Device(dev))],
                )
                .unwrap()
            };
            V::quantized(moved.as_quantized().unwrap().detach(&rt.native).unwrap())
        } else {
            V::tensor(w.to_device(dev))
        };
        check(&value, &g["state"][name], dev, name);
        if shape.len() == 2
            && name != "visual.pos_embed.weight"
            && name != "model.embed_tokens.weight"
        {
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
    let n = |key: &str| c[key].as_i64().unwrap();
    let head = n("hidden_size") / n("num_heads") / 2;
    let inv = Tensor::arange_start_step(0, head, 2, (Kind::Float, Device::Cpu)).g_div_scalar(head);
    let inv = Tensor::f_pow_scalar(10000., &inv)
        .unwrap()
        .reciprocal()
        .g_mul_scalar(1.)
        .to_device(dev);
    check(
        &V::tensor(inv.shallow_clone()),
        &g["state"]["visual.rotary_pos_emb.inv_freq"],
        dev,
        "buffer",
    );
    state
        .values
        .insert("visual.rotary_pos_emb.inv_freq".into(), V::tensor(inv));
    let config = q::VisionEncoderConfig {
        hidden_size: n("hidden_size"),
        intermediate_size: n("intermediate_size"),
        depth: n("depth") as usize,
        heads: n("num_heads"),
        patch_size: n("patch_size"),
        temporal_patch_size: n("temporal_patch_size"),
        channels: n("in_channels"),
        merge_size: n("spatial_merge_size"),
        position_embeddings: n("num_position_embeddings"),
        out_hidden_size: n("out_hidden_size"),
        deepstack_indexes: numbers(&c["deepstack_visual_indexes"])
            .into_iter()
            .map(|n| n as usize)
            .collect(),
    };
    (state, config)
}
