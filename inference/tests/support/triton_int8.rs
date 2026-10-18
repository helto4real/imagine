use super::{bits, expected, fixtures, kind, kitchen};
use imagine_inference::{
    autotuning::INT8_CONFIGS,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, Element, int8},
    triton_specialization::int8_ast_digest,
};
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
fn element(v: &str) -> Element {
    match v {
        "fp64" | "torch.float64" => Element::F64,
        "fp32" | "torch.float32" => Element::F32,
        "fp16" | "torch.float16" => Element::F16,
        "bf16" | "torch.bfloat16" => Element::BF16,
        _ => panic!("Original INT8 floating signature"),
    }
}
#[test]
fn original_int8_specialization_keys_preserve_both_scale_paths_all_configs_and_scalar_attributes() {
    for c in fixtures()["triton_int8_specialization"].as_array().unwrap() {
        let per_row = c["per_row"].as_bool().unwrap();
        let s = int8::Specialization {
            per_row,
            output: element(c["output"].as_str().unwrap()),
            weight_scale: Element::F32,
            bias: element(c["bias"].as_str().unwrap()),
            has_bias: c["has_bias"].as_bool().unwrap(),
            block_m: c["block_m"].as_u64().unwrap() as u32,
            block_n: c["block_n"].as_u64().unwrap() as u32,
            block_k: c["block_k"].as_u64().unwrap() as u32,
            group_size_m: c["group_size_m"].as_u64().unwrap() as u32,
            scalars: std::array::from_fn(|i| match c["scalar_types"][i].as_str().unwrap() {
                "one" => Count::One,
                "i32" => Count::I32 {
                    divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
                },
                "i64" => Count::I64 {
                    divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
                },
                _ => panic!("Original scalar type"),
            }),
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: "",
            standard_source_path: "",
        };
        let key = fixtures()["triton_int8_function_cache_keys"]
            [if per_row { "per_row" } else { "tensorwise" }]
        .as_str()
        .unwrap();
        assert_eq!(
            int8_ast_digest(&s, key),
            c["ast_digest"].as_str().unwrap(),
            "{c}"
        );
    }
}
#[test]
fn runtime_int8_compiler_preserves_all_216_original_candidate_outputs_binaries_options_and_cache_restoration()
 {
    use sha2::{Digest, Sha256};
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let profile = std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap();
    let compiler = unsafe {
        Compiler::open(
            Path::new(&profile),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    };
    let source: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("IMAGINE_INT8_PROBE_FIXTURE").unwrap()).unwrap(),
    )
    .unwrap();
    let mut restored = [false; 2];
    for case in source["cases"].as_array().unwrap() {
        let (m, n, k) = (
            case["m"].as_i64().unwrap(),
            case["n"].as_i64().unwrap(),
            case["k"].as_i64().unwrap(),
        );
        let dtype_name = case["dtype"].as_str().unwrap();
        let dtype = kind(dtype_name);
        let dev = Device::Cuda(0);
        let per_row = case["weight_scale"] == "per-row";
        let has_bias = case["has_bias"].as_bool().unwrap();
        let q = Tensor::from_slice(
            &expected(&case["quantized"])
                .into_iter()
                .map(|b| b as i8)
                .collect::<Vec<_>>(),
        )
        .reshape([m, k])
        .to_device(dev);
        let scale_bytes = expected(&case["scale"]);
        let scale_values = scale_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let a_scale = Tensor::from_slice(&scale_values)
            .reshape([m, 1])
            .to_device(dev);
        let weight = (Tensor::arange(n * k, (Kind::Int, dev)).remainder(255) - 127)
            .to_kind(Kind::Int8)
            .reshape([n, k]);
        let b_scale = if per_row {
            Tensor::arange(n, (Kind::Float, dev)) * 0.001 + 0.07
        } else {
            Tensor::from(0.07_f32).to_device(dev).reshape([1])
        };
        let bias = (Tensor::arange(n, (Kind::Float, dev)) * 0.013).to_kind(dtype);
        let dummy = (Tensor::arange(m * k, (Kind::Float, dev)) * 0.03125 - 4.)
            .reshape([m, k])
            .to_kind(dtype);
        let output = Tensor::empty([m, n], (dtype, dev));
        let pointers = [
            q.data_ptr() as u64,
            weight.data_ptr() as u64,
            output.data_ptr() as u64,
            a_scale.data_ptr() as u64,
            b_scale.data_ptr() as u64,
            if has_bias {
                bias.data_ptr() as u64
            } else {
                dummy.data_ptr() as u64
            },
        ];
        let values = [m, n, k, k, 1, 1, k, n, 1];
        for (index, candidate) in case["candidates"].as_array().unwrap().iter().enumerate() {
            let config = INT8_CONFIGS[index];
            let original = &source["kernels"][candidate["id"].as_u64().unwrap() as usize];
            let aligned = |i: usize| {
                original["attrs"][format!("({i},)")]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|v| v == &serde_json::json!(["tt.divisibility", 16]))
            };
            let scalars = std::array::from_fn(|i| {
                match original["signature"][int8::SCALAR_NAMES[i]]
                    .as_str()
                    .unwrap()
                {
                    "constexpr" => {
                        assert_eq!(values[i], 1);
                        Count::One
                    }
                    "i32" => Count::I32 {
                        divisible_by_16: aligned(i + 6),
                    },
                    "i64" => Count::I64 {
                        divisible_by_16: aligned(i + 6),
                    },
                    _ => panic!("Original runtime scalar type"),
                }
            });
            let s = int8::Specialization {
                per_row,
                output: element(dtype_name),
                weight_scale: Element::F32,
                bias: element(dtype_name),
                has_bias,
                block_m: config.block_m,
                block_n: config.block_n,
                block_k: config.block_k,
                group_size_m: config.group_size_m,
                scalars,
                pointer_divisibility_16: std::array::from_fn(aligned),
                source_path: "",
                standard_source_path: "",
            };
            let kernel = unsafe { compiler.int8(0, &s, index).unwrap() };
            assert_eq!(kernel.metadata["num_warps"], config.warps);
            assert_eq!(kernel.metadata["num_stages"], config.stages);
            assert_eq!(kernel.metadata["shared"], original["shared"]);
            let image = &kernel
                .assembly
                .iter()
                .find(|(k, _)| k == "cubin")
                .unwrap()
                .1;
            let hash = Sha256::digest(image)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            assert_eq!(
                hash,
                original["cubin_sha256"].as_str().unwrap(),
                "Original complete candidate image {case_index}/{index}",
                case_index = case["quantizer_id"]
            );
            let mut args = pointers
                .into_iter()
                .map(Scalar::TensorPointer)
                .collect::<Vec<_>>();
            for (v, c) in values.into_iter().zip(scalars) {
                match c {
                    Count::One => {}
                    Count::I32 { .. } => args.push(Scalar::I32(v as i32)),
                    Count::I64 { .. } => args.push(Scalar::I64(v)),
                }
            }
            args.extend([Scalar::Pointer(0), Scalar::Pointer(0)]);
            let grid = (((m + i64::from(config.block_m) - 1) / i64::from(config.block_m))
                * ((n + i64::from(config.block_n) - 1) / i64::from(config.block_n)))
                as i32;
            unsafe {
                kernel
                    .run([grid, 1, 1], native.current_stream(0).unwrap(), &mut args)
                    .unwrap();
            }
            assert_eq!(bits(&output), expected(&candidate["bytes"]));
            assert_eq!(bits(&output), expected(&case["output"]));
            assert!(Arc::ptr_eq(&kernel, &unsafe {
                compiler.int8(0, &s, index).unwrap()
            }));
            if !restored[usize::from(per_row)] {
                let fresh = unsafe {
                    Compiler::open(
                        Path::new(&profile),
                        &own.join("cache"),
                        &own.join("work"),
                        native.clone(),
                    )
                    .unwrap()
                };
                let cached = unsafe { fresh.int8(0, &s, index).unwrap() };
                assert!(cached.disk_cache_hit);
                assert_eq!(cached.digest, kernel.digest);
                assert_eq!(cached.metadata, kernel.metadata);
                unsafe {
                    cached
                        .run([grid, 1, 1], native.current_stream(0).unwrap(), &mut args)
                        .unwrap();
                }
                assert_eq!(bits(&output), expected(&case["output"]));
                restored[usize::from(per_row)] = true;
            }
        }
    }
    assert_eq!(restored, [true; 2]);
}

#[test]
fn registered_int8_wrapper_preserves_original_boundaries_outputs_and_two_autotuner_states() {
    super::triton_wrapper_tests::compare("int8");
}
