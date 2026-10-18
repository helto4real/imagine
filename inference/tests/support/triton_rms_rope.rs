use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution_value::Value as V,
    triton_compiler::Compiler,
    triton_frontend::{Count, Element, rope::rms::Specialization},
    triton_specialization::rms_rope_ast_digest_scalar,
};
use std::{collections::BTreeMap, sync::Arc};
fn element(v: &Value) -> Element {
    match v.as_str().unwrap() {
        "fp32" => Element::F32,
        "fp16" => Element::F16,
        "bf16" => Element::BF16,
        _ => unreachable!(),
    }
}
#[test]
fn original_rms_rope_specialization_keys_preserve_all_scalar_types_and_constexpr_positions() {
    let function = fixtures()["triton_rms_rope_function_cache_key"]
        .as_str()
        .unwrap();
    for c in fixtures()["triton_rms_rope_specialization"]
        .as_array()
        .unwrap()
    {
        let scalars = std::array::from_fn(|i| match c["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => unreachable!(),
        });
        let s = Specialization {
            input: element(&c["input"]),
            scale: element(&c["scale"]),
            frequencies: element(&c["frequencies"]),
            block_size: c["block_size"].as_u64().unwrap() as u32,
            split_half: c["split_half"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: c["source_path"].as_str().unwrap(),
            standard_source_path: c["standard_source_path"].as_str().unwrap(),
        };
        assert_eq!(
            rms_rope_ast_digest_scalar(&s, function, super::triton_scalar_tests::epsilon(c)),
            c["ast_digest"].as_str().unwrap(),
            "Original RMS-RoPE AST key {}",
            c["index"]
        );
    }
}
#[test]
fn registered_rms_rope_triton_bindings_preserve_paired_layout_precision_empty_shapes_strides_identity_and_errors()
 {
    use sha2::{Digest, Sha256};
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let rt = imagine_inference::execution::Runtime::frozen_partial(kitchen()).unwrap();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            rt.native.clone(),
        )
        .unwrap()
    });
    rt.bind_triton_fp8(compiler.clone());
    rt.bind_triton_rowwise(compiler.clone());
    rt.bind_triton_adaln(compiler.clone());
    rt.bind_triton_rope(compiler.clone());
    rt.bind_triton_rms_rope(compiler);
    assert_eq!(rt.backends.gaps().len(), 6);
    for (i, c) in fixtures()["triton_rms_rope_bindings"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        let kwargs = super::binding_tests::kwargs(&c["kwargs"], Device::Cuda(0));
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        match unsafe { selected.call(&rt, &args, &kwargs) } {
            Err(e) if c["error"].is_string() => reference_error(e, c),
            Err(e) => panic!("RMS-RoPE wrapper {i} {}: {e}", c["label"]),
            Ok(_) if c["error"].is_string() => {
                panic!("Expected original RMS-RoPE error {i} {}", c["label"])
            }
            Ok(v) => {
                check(
                    &v,
                    &c["output"],
                    Device::Cuda(0),
                    &format!("RMS-RoPE {i} {}", c["label"]),
                );
                let inputs = args
                    .iter()
                    .chain(kwargs.iter().map(|(_, v)| v))
                    .flat_map(leaves)
                    .collect::<Vec<_>>();
                let outputs = leaves(&v);
                let aliases = outputs
                    .iter()
                    .map(|t| {
                        inputs
                            .iter()
                            .map(|x| {
                                t.as_tensor().unwrap().data_ptr()
                                    == x.as_tensor().unwrap().data_ptr()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let identities = outputs
                    .iter()
                    .map(|t| inputs.iter().map(|x| t.same_object(x)).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                assert_eq!(serde_json::json!(aliases), c["aliases"], "Alias {i}");
                assert_eq!(
                    serde_json::json!(identities),
                    c["identities"],
                    "Identity {i}"
                );
            }
        }
        for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, Device::Cuda(0), &format!("Post args {i}"));
        }
        for ((name, v), e) in kwargs.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(name, e[0].as_str().unwrap());
            check(v, &e[1], Device::Cuda(0), &format!("Post kwarg {i}"));
        }
    }
    for c in fixtures()["triton_rms_rope_large"].as_array().unwrap() {
        let d = c["d"].as_i64().unwrap();
        let kind = match c["dtype"].as_str().unwrap() {
            "torch.float32" => Kind::Float,
            "torch.float16" => Kind::Half,
            "torch.bfloat16" => Kind::BFloat16,
            _ => unreachable!(),
        };
        let make = |shape: &[i64], kind: Kind, offset: i64| {
            (((Tensor::arange(
                shape.iter().product::<i64>(),
                (Kind::Float, Device::Cuda(0)),
            ) + offset)
                .remainder(97))
                * 0.03123
                - 1.5032)
                .reshape(shape)
                .to_kind(kind)
        };
        let args = [
            V::tensor(make(&[1, 2, 3, d], kind, 0)),
            V::tensor(make(&[1, 1, 3, d / 2, 2, 2], Kind::Float, 29)),
            V::tensor(make(&[d], Kind::BFloat16, 7)),
        ];
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let out = unsafe { selected.call(&rt, &args, &vec![]).unwrap() }
            .as_tensor()
            .unwrap();
        let bytes = out
            .f_contiguous()
            .unwrap()
            .f_view([-1])
            .unwrap()
            .f_view_dtype(Kind::Uint8)
            .unwrap()
            .to_device(Device::Cpu);
        let mut data = vec![0u8; bytes.numel()];
        let len = data.len();
        bytes.copy_data_u8(&mut data, len);
        let digest = Sha256::digest(&data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            digest,
            c["sha256"].as_str().unwrap(),
            "Complete RMS-RoPE bytes {c}"
        );
    }
}
fn leaves(v: &V) -> Vec<V> {
    match v {
        V::Tensor(_) => vec![v.clone()],
        V::Tuple(xs) => xs.iter().flat_map(leaves).collect(),
        _ => vec![],
    }
}
