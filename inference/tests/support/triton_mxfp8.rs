use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution_value::Value as V,
    triton_compiler::Compiler,
    triton_frontend::{Count, Element, mxfp8::Specialization},
    triton_specialization::mxfp8_ast_digest,
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
fn original_mxfp8_specialization_keys_preserve_all_scalar_types_and_constexpr_positions() {
    let function = fixtures()["triton_mxfp8_function_cache_key"]
        .as_str()
        .unwrap();
    for c in fixtures()["triton_mxfp8_specialization"]
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
            block_size: c["block_size"].as_u64().unwrap() as u32,
            blocks_per_program: c["blocks_per_program"].as_u64().unwrap() as u32,
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: c["source_path"].as_str().unwrap(),
            standard_source_path: c["standard_source_path"].as_str().unwrap(),
        };
        assert_eq!(
            mxfp8_ast_digest(&s, function),
            c["ast_digest"].as_str().unwrap(),
            "Original MXFP8 AST key {}",
            c["index"]
        );
    }
}
#[test]
fn registered_mxfp8_triton_bindings_preserve_paired_layout_precision_empty_shapes_strides_identity_and_errors()
 {
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
    rt.bind_triton_rms_rope(compiler.clone());
    rt.bind_triton_mxfp8(compiler);
    assert_eq!(rt.backends.gaps().len(), 5);
    for (i, c) in fixtures()["triton_mxfp8_bindings"]
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
            Err(e) => panic!("MXFP8 wrapper {i} {}: {e}", c["label"]),
            Ok(_) if c["error"].is_string() => {
                panic!("Expected original MXFP8 error {i} {}", c["label"])
            }
            Ok(v) => {
                check(
                    &v,
                    &c["output"],
                    Device::Cuda(0),
                    &format!("MXFP8 {i} {}", c["label"]),
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
}
fn leaves(v: &V) -> Vec<V> {
    match v {
        V::Tensor(_) => vec![v.clone()],
        V::Tuple(xs) => xs.iter().flat_map(leaves).collect(),
        _ => vec![],
    }
}
#[test]
fn original_mxfp8_known_tail_storage_and_large_complete_outputs_preserve_all_source_bytes() {
    use sha2::{Digest, Sha256};
    use std::sync::atomic::{AtomicUsize, Ordering};
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
    rt.bind_triton_mxfp8(compiler.clone());
    for (i, c) in fixtures()["triton_mxfp8_known_storage"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let x = restore(&c["args"][0], Device::Cuda(0));
        let expected = x.clone();
        let seen = Arc::new(AtomicUsize::new(0));
        let observed = seen.clone();
        compiler.set_pre_run_guard(Some(Arc::new(move |name, a, kw| {
            assert_eq!(name, "mxfp8");
            assert_eq!(a.len(), 9);
            assert!(a[0].same_object(&expected));
            let input = a[0].as_tensor()?;
            let mut output = a[1].as_tensor()?;
            let scales = a[2].as_tensor()?;
            assert_eq!(input.size(), output.size());
            assert_eq!(output.kind(), Kind::Float8e4m3fn);
            assert_eq!(output.device(), input.device());
            assert_eq!(scales.kind(), Kind::Uint8);
            let shape = input.size();
            assert_eq!(
                scales.size(),
                vec![(shape[0] + 127) / 128 * 128, (shape[1] / 32 + 3) / 4 * 4]
            );
            assert!(
                kw.iter()
                    .any(|(k, v)| k == "block_size" && matches!(v, V::Int(32)))
            );
            assert!(
                kw.iter()
                    .any(|(k, v)| k == "blocks_per_program" && matches!(v, V::Int(1)))
            );
            let _ = output.f_fill_(-1.5)?;
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })));
        let selected = rt
            .backends
            .resolve("quantize_mxfp8", Some("triton"), &BTreeMap::new())
            .unwrap();
        let out = unsafe {
            selected
                .call(&rt, std::slice::from_ref(&x), &vec![])
                .unwrap()
        };
        compiler.set_pre_run_guard(None);
        assert_eq!(
            seen.load(Ordering::SeqCst),
            c["hook_calls"].as_u64().unwrap() as usize
        );
        check(
            &out,
            &c["output"],
            Device::Cuda(0),
            &format!("Known MXFP8 tail {i}"),
        );
        check(
            &x,
            &c["post_args"][0],
            Device::Cuda(0),
            &format!("Known MXFP8 input {i}"),
        );
    }
    fn digest(t: &Tensor) -> String {
        Sha256::digest(bits(t))
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
    for c in fixtures()["triton_mxfp8_large"].as_array().unwrap() {
        let n = c["blocks"].as_i64().unwrap() * 32;
        let kind = kind(c["dtype"].as_str().unwrap());
        let x = ((Tensor::arange(n, (Kind::Float, Device::Cuda(0))).remainder(97) * 0.03123)
            - 1.5032)
            .reshape([1, n])
            .to_kind(kind);
        let selected = rt
            .backends
            .resolve("quantize_mxfp8", Some("triton"), &BTreeMap::new())
            .unwrap();
        let out = unsafe { selected.call(&rt, &[V::tensor(x)], &vec![]).unwrap() };
        let V::Tuple(items) = out else {
            panic!("Original MXFP8 pair")
        };
        let q = items[0].as_tensor().unwrap();
        let s = items[1].as_tensor().unwrap();
        assert_eq!(
            digest(&q),
            c["output_sha256"].as_str().unwrap(),
            "MXFP8 output {c}"
        );
        assert_eq!(
            digest(&s),
            c["scales_sha256"].as_str().unwrap(),
            "MXFP8 scales {c}"
        );
        assert_eq!(serde_json::json!([q.size(), s.size()]), c["shapes"]);
        assert_eq!(serde_json::json!([q.stride(), s.stride()]), c["strides"]);
        assert_eq!(rt.native.tensor_type(&s).unwrap().scalar_type, 44);
    }
}
