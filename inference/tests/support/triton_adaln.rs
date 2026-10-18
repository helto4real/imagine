use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution_value::Value as V,
    triton_compiler::Compiler,
    triton_frontend::{Count, Element, adaln::Specialization},
    triton_specialization::adaln_ast_digest_scalar,
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
fn original_adaln_specialization_keys_preserve_all_scalar_types_and_constexpr_positions() {
    let function = fixtures()["triton_adaln_function_cache_key"]
        .as_str()
        .unwrap();
    for c in fixtures()["triton_adaln_specialization"]
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
            shift: element(&c["shift"]),
            block_size: c["block_size"].as_u64().unwrap() as u32,
            subtract_mean: c["subtract_mean"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: c["source_path"].as_str().unwrap(),
            standard_source_path: c["standard_source_path"].as_str().unwrap(),
        };
        assert_eq!(
            adaln_ast_digest_scalar(&s, function, super::triton_scalar_tests::epsilon(c)),
            c["ast_digest"].as_str().unwrap(),
            "Original AdaLN AST key {}",
            c["index"]
        );
    }
}
#[test]
fn registered_adaln_triton_bindings_preserve_broadcast_two_pass_precision_empty_shapes_strides_and_errors()
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
    rt.bind_triton_adaln(compiler);
    assert_eq!(rt.backends.gaps().len(), 22);
    for (i, c) in fixtures()["triton_adaln_bindings"]
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
            Err(e) => panic!("AdaLN wrapper {i} {}: {e}", c["label"]),
            Ok(_) if c["error"].is_string() => {
                panic!("Expected original AdaLN error {i} {}", c["label"])
            }
            Ok(v) => check(
                &v,
                &c["output"],
                Device::Cuda(0),
                &format!("AdaLN {i} {}", c["label"]),
            ),
        }
    }
    fn data(shape: &[i64], kind: Kind, offset: i64) -> Tensor {
        Tensor::arange(
            shape.iter().product::<i64>(),
            (Kind::Float, Device::Cuda(0)),
        )
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(97)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(1.5)
        .unwrap()
        .reshape(shape)
        .to_kind(kind)
    }
    for c in fixtures()["triton_adaln_large"].as_array().unwrap() {
        let d = c["d"].as_i64().unwrap();
        let args = [
            V::tensor(data(&[2, d], kind(c["input_dtype"].as_str().unwrap()), 0)),
            V::tensor(data(&[d], Kind::Float, 13)),
            V::tensor(data(&[d], Kind::BFloat16, 29)),
        ];
        let selected = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let result = unsafe { selected.call(&rt, &args, &vec![]).unwrap() }
            .as_tensor()
            .unwrap();
        assert_eq!(serde_json::json!(result.size()), c["shape"]);
        assert_eq!(serde_json::json!(result.stride()), c["strides"]);
        let hash = Sha256::digest(bits(&result))
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(
            hash, c["sha256"],
            "Original multi-block AdaLN output {} {d}",
            c["operation"]
        );
    }
}
