use super::{
    binding_tests::{check, restore},
    kind, kitchen,
};
use imagine_inference::{
    execution::Runtime, execution_value::Value as V, models::qwen_image21 as q, native::Error,
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::Device;
fn intlist(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/qwen-image21-core.json")).unwrap()
}
#[test]
fn qwen_image21_original_component_math_sequences_and_int8_bf16_turbo_projections_match() {
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
    let rt = Runtime::frozen(native.clone(), compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let data = fixture();
    let mut counts = BTreeMap::<String, usize>::new();
    for c in data["cases"].as_array().unwrap() {
        let op = c["operation"].as_str().unwrap();
        if op == "detect" {
            continue;
        }
        *counts.entry(op.into()).or_default() += 1;
        let label = c["label"].as_str().unwrap();
        eprintln!("Qwen21 component: {label}");
        native
            .set_default_dtype(kind(c["default_dtype"].as_str().unwrap()))
            .unwrap();
        let args = c["args"].as_array().unwrap();
        let kw = &c["kwargs"];
        let values = args
            .iter()
            .map(|v| restore(v, Device::Cpu))
            .collect::<Vec<_>>();
        let t = |index: usize| values[index].as_tensor().unwrap();
        let result: Result<V, Error> = (|| {
            Ok(match op {
                "zero_rms" => V::tensor(q::zero_centered_rms_norm(&t(0), &t(1), 1e-6)?),
                "modulated_norm" => V::tensor(unsafe {
                    q::modulated_norm(
                        &rt,
                        &t(0),
                        (&t(1), &t(2)),
                        kw["prefix"].as_i64().unwrap(),
                        &t(3),
                        1e-6,
                        kw["training"].as_bool().unwrap(),
                    )?
                }),
                "gated" => V::tensor(q::gated_residual(
                    &t(0),
                    &t(1),
                    (&t(2), &t(3)),
                    kw["prefix"].as_i64().unwrap(),
                )?),
                "split" => {
                    let (a, b) = q::split_rows(&t(0))?;
                    V::Tuple(vec![V::tensor(a), V::tensor(b)])
                }
                "timestep" => V::tensor(q::timestep_embedding(
                    &t(0),
                    kw["dim"].as_i64().unwrap(),
                    10000.,
                    1000.,
                )?),
                "round_timestep" => V::tensor(q::round_timesteps(
                    &t(0),
                    kind(kw["dtype"].as_str().unwrap()),
                )?),
                "process_in" => V::tensor(q::LatentNormalization::new(&rt)?.process_in(&t(0))?),
                "process_out" => V::tensor(q::LatentNormalization::new(&rt)?.process_out(&t(0))?),
                "rope" => {
                    let a = intlist(&kw["axes"]);
                    V::tensor(
                        q::PositionEncoding {
                            axes: a.try_into().unwrap(),
                            theta: 10000.,
                            supports_fp64: kw["fp64"].as_bool().unwrap(),
                        }
                        .forward(&t(0))?,
                    )
                }
                "projection_stages" => {
                    let norm = q::zero_centered_rms_norm(&t(0), &t(1), 1e-6)?;
                    let linear = |x, w: &V| unsafe {
                        rt.op(
                            "aten::linear.default",
                            &[V::tensor(x), w.clone(), V::None],
                            &vec![],
                        )?
                        .as_tensor()
                    };
                    let first = linear(norm.shallow_clone(), &values[2])?;
                    let gelu = first.f_gelu("tanh")?;
                    let output = linear(gelu.shallow_clone(), &values[3])?;
                    V::Tuple(vec![
                        V::tensor(norm),
                        V::tensor(first),
                        V::tensor(gelu),
                        V::tensor(output),
                    ])
                }
                "sequence" => {
                    let x = t(0);
                    let ctx = t(1);
                    let V::List(ref_values) = &values[2] else {
                        panic!("Reference list required")
                    };
                    let refs = ref_values
                        .iter()
                        .map(|v| v.as_tensor().unwrap())
                        .collect::<Vec<_>>();
                    let text = q::TextProjection {
                        norm_weight: t(3),
                        in_weight: values[4].clone(),
                        out_weight: values[5].clone(),
                        eps: 1e-6,
                    };
                    let image = values[6].clone();
                    let sequence = unsafe {
                        q::build_sequence(
                            &rt,
                            &x,
                            &ctx,
                            &refs,
                            &intlist(&kw["slots"]),
                            &q::SequenceProjections {
                                text: &text,
                                image: &image,
                            },
                            &q::PositionEncoding::default(),
                        )?
                    };
                    let segments = sequence
                        .segments
                        .into_iter()
                        .map(|s| {
                            V::Tuple(vec![
                                V::Int(s.start),
                                V::Int(s.end),
                                s.mask.map_or(V::None, V::tensor),
                            ])
                        })
                        .collect();
                    V::Tuple(vec![
                        V::tensor(sequence.hidden),
                        V::tensor(sequence.position_encoding),
                        V::List(segments),
                        V::tensor(q::prefix_cache_key(
                            &x,
                            &ctx,
                            &refs,
                            &intlist(&kw["slots"]),
                        )?),
                    ])
                }
                _ => panic!("Unknown Qwen21 fixture operation {op}"),
            })
        })();
        let output = result.unwrap_or_else(|e| panic!("{label}: {e}"));
        if let V::Tuple(parts) = &output {
            for (index, part) in parts.iter().enumerate() {
                check(
                    part,
                    &c["output"]["items"][index],
                    Device::Cpu,
                    &format!("{label}/output[{index}]"),
                );
            }
        } else {
            check(&output, &c["output"], Device::Cpu, label);
        }
        if op == "gated" {
            assert_eq!(
                output.as_tensor().unwrap().data_ptr(),
                t(0).data_ptr(),
                "{label}: original in-place storage identity"
            );
        }
        for (value, expected) in values.iter().zip(c["post_args"].as_array().unwrap()) {
            check(value, expected, Device::Cpu, label);
        }
    }
    assert_eq!(counts.values().sum::<usize>(), 180);
    assert_eq!(counts["sequence"], 24);
}
#[test]
fn qwen_image21_original_logical_shape_detection_preserves_fused_split_and_prefix_rules() {
    let data = fixture();
    let mut count = 0;
    for c in data["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["operation"] == "detect")
    {
        let shapes = c["shapes"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), intlist(v)))
            .collect();
        let cfg = q::detect(&shapes, c["prefix"].as_str().unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(
            json!({"image_model":"qwen_image21","in_channels":cfg.in_channels,"out_channels":cfg.out_channels,"num_layers":cfg.num_layers,"attention_head_dim":cfg.attention_head_dim,"num_attention_heads":cfg.num_attention_heads,"context_in_dim":cfg.context_in_dim,"mlp_ratio":cfg.mlp_ratio,"fused_mlp":cfg.fused_mlp}),
            c["output_config"]
        );
        count += 1;
    }
    assert_eq!(count, 12);
}
