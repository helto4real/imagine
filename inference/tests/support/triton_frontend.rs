use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution_value::Value as V,
    triton_driver::{Driver, Scalar},
};
use std::collections::BTreeMap;
#[test]
fn rust_frontend_cubins_and_original_driver_preserve_fp8_bytes_resources_and_pointer_errors() {
    let native = kitchen();
    let path = std::env::var("IMAGINE_TRITON_DRIVER_LIBRARY")
        .expect("Verified original driver binding required");
    let root = std::env::var("IMAGINE_TRITON_FRONTEND_CUBINS")
        .expect("Rust-fronted compiler output required");
    let driver = unsafe { Driver::open(Path::new(&path)).unwrap() };
    let mut kernels = BTreeMap::new();
    for (i, c) in fixtures()["triton_driver"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = Device::Cuda(0);
        let input = restore(&c["input"], dev).as_tensor().unwrap();
        if c["operation"] == "cpu_pointer" {
            match unsafe { driver.tensor_pointer(input.data_ptr() as u64) } {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected original CPU-pointer error"),
            }
            continue;
        }
        let scale = restore(&c["scale"], dev).as_tensor().unwrap();
        let out = Tensor::empty(
            c["output"]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect::<Vec<_>>(),
            (kind(c["output"]["dtype"].as_str().unwrap()), dev),
        );
        let stream = native.current_stream(0).unwrap();
        let index = c["cubin_index"].as_i64().unwrap();
        let kernel = kernels.entry(index).or_insert_with(|| {
            let data = std::fs::read(Path::new(&root).join(format!("case-{index}.cubin"))).unwrap();
            let name = if c["operation"] == "quantize" {
                "quantize_fp8_kernel_tl"
            } else {
                "dequantize_fp8_kernel_tl"
            };
            unsafe { driver.load(name, &data, 0, 0).unwrap() }
        });
        let resources = serde_json::json!({"registers":kernel.registers,"spills":kernel.spills,"max_threads":kernel.max_threads});
        assert_eq!(
            resources, c["resources"],
            "Original driver resource metadata {i}"
        );
        let mut args = vec![];
        for t in [&input, &out, &scale] {
            args.push(Scalar::Pointer(unsafe {
                driver.tensor_pointer(t.data_ptr() as u64).unwrap()
            }));
        }
        if c["operation"] == "quantize" {
            args.push(Scalar::F32(c["limit"].as_f64().unwrap() as f32));
        }
        args.push(Scalar::I32(input.numel().try_into().unwrap()));
        args.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
        unsafe {
            driver
                .launch(
                    kernel,
                    [(input.numel().div_ceil(128)).try_into().unwrap(), 1, 1],
                    4,
                    1,
                    false,
                    false,
                    0,
                    stream,
                    &mut args,
                )
                .unwrap();
        }
        check(
            &V::tensor(out),
            &c["output"],
            dev,
            &format!("Rust frontend/direct driver {i}"),
        );
    }
}

#[test]
fn triton_specialization_keys_preserve_original_ast_compiler_base32_and_unicode_environment_repr() {
    use imagine_inference::{
        triton_frontend::{Count, Element, Fp8Body, Fp8Specialization},
        triton_specialization::{
            CompilerHashInputs, ast_digest, cache_directory, compiler_digest, environment_repr,
            python_str_repr,
        },
    };
    use sha2::{Digest, Sha256};
    let profile = &fixtures()["triton_specialization_hash_inputs"];
    let inputs = CompilerHashInputs {
        triton_version_key: profile["triton_version_key"].as_str().unwrap(),
        backend_hash: profile["backend_hash"].as_str().unwrap(),
        options_hash: profile["options_hash"].as_str().unwrap(),
    };
    let element = |v: &Value| match v.as_str().unwrap() {
        "fp32" => Element::F32,
        "fp16" => Element::F16,
        "bf16" => Element::BF16,
        "fp8e4nv" => Element::E4M3,
        "fp8e5" => Element::E5M2,
        _ => panic!("Expected original element"),
    };
    let mut first_ast = None;
    for c in fixtures()["triton_specialization"].as_array().unwrap() {
        let v = &c["specialization"];
        let count = match v["count_type"].as_str().unwrap() {
            "i32" => Count::I32 {
                divisible_by_16: v["count_div16"].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["count_div16"].as_bool().unwrap(),
            },
            "one" => Count::One,
            _ => unreachable!(),
        };
        let s = Fp8Specialization {
            body: if v["operation"] == "quantize" {
                Fp8Body::Quantize
            } else {
                Fp8Body::Dequantize
            },
            input: element(&v["input"]),
            output: element(&v["output"]),
            scale: element(&v["scale"]),
            block_size: v["block_size"].as_u64().unwrap() as u32,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            count,
            source_path: v["source_path"].as_str().unwrap(),
        };
        let key = profile["function_keys"][v["operation"].as_str().unwrap()]
            .as_str()
            .unwrap();
        let ast = ast_digest(&s, key);
        assert_eq!(ast, c["ast_digest"]);
        let compiler = compiler_digest(&inputs, &ast, &BTreeMap::new());
        assert_eq!(compiler, c["compiler_digest"]);
        assert_eq!(cache_directory(&compiler).unwrap(), c["cache_directory"]);
        if first_ast.is_none() {
            first_ast = Some(ast);
        }
    }
    for c in fixtures()["triton_cache_environments"].as_array().unwrap() {
        let environment = c["values"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
            .collect();
        assert_eq!(environment_repr(&environment), c["python_repr"]);
        assert_eq!(
            compiler_digest(&inputs, first_ast.as_ref().unwrap(), &environment),
            c["digest"]
        );
    }
    let mut sha = Sha256::new();
    for code in 0..=0x10ffff {
        if let Some(c) = char::from_u32(code) {
            sha.update(python_str_repr(&c.to_string()).as_bytes());
            sha.update(b"\n");
        }
    }
    let digest = sha
        .finalize()
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect::<String>();
    assert_eq!(
        digest, profile["unicode_scalar_repr_sha256"],
        "Every Unicode scalar's original CPython repr"
    );
}
