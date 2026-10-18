//! Development driver: stdin specialization JSON -> source-positioned TTIR.
#[path = "../src/triton_frontend.rs"]
mod triton_frontend;
use std::io::{self, Read};
use triton_frontend::{Count, Element, Fp8Body, Fp8Specialization, fp8_ttir};
fn element(v: &serde_json::Value) -> Element {
    match v.as_str().unwrap() {
        "fp64" => Element::F64,
        "fp32" => Element::F32,
        "fp16" => Element::F16,
        "bf16" => Element::BF16,
        "fp8e4nv" => Element::E4M3,
        "fp8e5" => Element::E5M2,
        _ => panic!("Unsupported development signature"),
    }
}
fn epsilon(v: &serde_json::Value) -> triton_frontend::scalar::ScalarType {
    use triton_frontend::scalar::ScalarType;
    match v
        .get("epsilon_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("fp32")
    {
        "fp32" => ScalarType::F32,
        "fp64" => ScalarType::F64,
        "i32" => ScalarType::I32 {
            divisible_by_16: v["epsilon_div16"].as_bool().unwrap_or(false),
        },
        "i64" => ScalarType::I64 {
            divisible_by_16: v["epsilon_div16"].as_bool().unwrap_or(false),
        },
        "u64" => ScalarType::U64 {
            divisible_by_16: v["epsilon_div16"].as_bool().unwrap_or(false),
        },
        "one" => ScalarType::One,
        "u1" => ScalarType::Bool,
        _ => panic!("Original numeric epsilon type"),
    }
}
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let v: serde_json::Value = serde_json::from_str(&input).unwrap();
    if v["operation"] == "na3d" {
        let s = triton_frontend::na::Specialization {
            input: match v["input"].as_str().unwrap() {
                "fp32" => Element::F32,
                "fp16" => Element::F16,
                "bf16" => Element::BF16,
                _ => panic!("source element"),
            },
            key: match v.get("key").unwrap_or(&v["input"]).as_str().unwrap() {
                "fp32" => Element::F32,
                "fp16" => Element::F16,
                "bf16" => Element::BF16,
                _ => panic!("source key type"),
            },
            value: match v.get("value").unwrap_or(&v["input"]).as_str().unwrap() {
                "fp32" => Element::F32,
                "fp16" => Element::F16,
                "bf16" => Element::BF16,
                _ => panic!("source value type"),
            },
            scale: match v["scale_type"].as_str().unwrap() {
                "fp32" => triton_frontend::na::Scale::F32,
                "fp64" => triton_frontend::na::Scale::F64,
                "u64" => triton_frontend::na::Scale::U64 {
                    divisible_by_16: v["scale_div16"].as_bool().unwrap(),
                },
                "one" => triton_frontend::na::Scale::One,
                "u1" => triton_frontend::na::Scale::Bool,
                "i32" => triton_frontend::na::Scale::I32 {
                    divisible_by_16: v["scale_div16"].as_bool().unwrap(),
                },
                "i64" => triton_frontend::na::Scale::I64 {
                    divisible_by_16: v["scale_div16"].as_bool().unwrap(),
                },
                _ => panic!("source scale type"),
            },
            kt: v["kt"]
                .as_i64()
                .or_else(|| v["kt"].as_bool().map(i64::from))
                .unwrap_or_else(|| v["kt"].as_f64().unwrap() as i64),
            kh: v["kh"]
                .as_i64()
                .or_else(|| v["kh"].as_bool().map(i64::from))
                .unwrap_or_else(|| v["kh"].as_f64().unwrap() as i64),
            kw: v["kw"]
                .as_i64()
                .or_else(|| v["kw"].as_bool().map(i64::from))
                .unwrap_or_else(|| v["kw"].as_f64().unwrap() as i64),
            causal_t: v["causal_t"].as_bool().unwrap(),
            causal_h: v["causal_h"].as_bool().unwrap(),
            causal_w: v["causal_w"].as_bool().unwrap(),
            hd: v["hd"].as_u64().unwrap() as u32,
            hd_pad: v["hd_pad"].as_u64().unwrap() as u32,
            block_q: v["block_q"].as_u64().unwrap() as u32,
            block_k: v["block_k"].as_u64().unwrap() as u32,
            scalars: std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
                "one" => Count::One,
                "i32" => Count::I32 {
                    divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
                },
                "i64" => Count::I64 {
                    divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
                },
                _ => panic!("source scalar"),
            }),
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        if let Some(kernels) = v.get("kernel_values") {
            let kernels = std::array::from_fn(|i| {
                let value = &kernels[i];
                if let Some(b) = value.as_bool() {
                    triton_frontend::na::KernelValue::Bool(b)
                } else if let Some(n) = value.as_i64() {
                    triton_frontend::na::KernelValue::Int(n)
                } else if let Some(n) = value.as_u64() {
                    triton_frontend::na::KernelValue::UInt(n)
                } else {
                    triton_frontend::na::KernelValue::Float(value.as_f64().unwrap())
                }
            });
            match triton_frontend::na::checked_source_kernels(&s, kernels) {
                Ok(source) => print!("{source}"),
                Err(e) => {
                    print!(
                        "{}",
                        serde_json::json!({"error":e.class,"message":e.message})
                    );
                    std::process::exit(2);
                }
            }
            return;
        }
        match triton_frontend::na::checked_source(&s) {
            Ok(source) => print!("{source}"),
            Err(e) => {
                print!(
                    "{}",
                    serde_json::json!({"error":e.class,"message":e.message})
                );
                std::process::exit(2);
            }
        }
        return;
    }
    if v["operation"] == "w4a8_unpack" {
        let s = triton_frontend::w4a8::Specialization {
            has_cb: v["has_cb"].as_bool().unwrap(),
            bits: v["bits"].as_u64().unwrap() as u32,
            block_n: v["block_n"].as_u64().unwrap() as u32,
            block_kh: v["block_kh"].as_u64().unwrap() as u32,
            scalars: std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
                "one" => Count::One,
                "i32" => Count::I32 {
                    divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
                },
                "i64" => Count::I64 {
                    divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
                },
                _ => panic!("original scalar type"),
            }),
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
        };
        print!("{}", triton_frontend::w4a8::source_ttir(&s));
        return;
    }
    if v["operation"] == "int8_matmul" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Original scalar type"),
        });
        let s = triton_frontend::int8::Specialization {
            per_row: v["per_row"].as_bool().unwrap(),
            output: element(&v["output"]),
            weight_scale: Element::F32,
            bias: element(&v["bias"]),
            has_bias: v["has_bias"].as_bool().unwrap(),
            block_m: v["block_m"].as_u64().unwrap() as u32,
            block_n: v["block_n"].as_u64().unwrap() as u32,
            block_k: v["block_k"].as_u64().unwrap() as u32,
            group_size_m: v["group_size_m"].as_u64().unwrap() as u32,
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        print!("{}", triton_frontend::int8::source_ttir(&s));
        return;
    }
    if v["operation"] == "nvfp4" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Original count type"),
        });
        let s = triton_frontend::nvfp4::Specialization {
            input: element(&v["input"]),
            global_scale: Element::F32,
            block_size: v["block_size"].as_u64().unwrap() as u32,
            blocks_per_program: v["blocks_per_program"].as_u64().unwrap() as u32,
            hi_first: v["hi_first"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        if let Some(e) = triton_frontend::nvfp4::condition_failure(
            false,
            v.get("invalid_hi_first_type")
                .and_then(serde_json::Value::as_str),
        ) {
            print!(
                "{}",
                serde_json::json!({"error": e.class, "message": e.message})
            );
            std::process::exit(2);
        }
        print!("{}", triton_frontend::nvfp4::source_ttir(&s));
        return;
    }
    if v["operation"] == "dequant_nvfp4" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Original count type"),
        });
        let s = triton_frontend::nvfp4::dequantize::Specialization {
            output: element(&v["input"]),
            block_scale: Element::E4M3,
            global_scale: Element::F32,
            block_size: v["block_size"].as_u64().unwrap() as u32,
            tile_size: 128,
            hi_first: v["hi_first"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
        };
        if let Some(e) = triton_frontend::nvfp4::condition_failure(
            true,
            v.get("invalid_hi_first_type")
                .and_then(serde_json::Value::as_str),
        ) {
            print!(
                "{}",
                serde_json::json!({"error": e.class, "message": e.message})
            );
            std::process::exit(2);
        }
        print!("{}", triton_frontend::nvfp4::dequantize::source_ttir(&s));
        return;
    }
    if v["operation"] == "mxfp8" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Unsupported development scalar type"),
        });
        let s = triton_frontend::mxfp8::Specialization {
            input: element(&v["input"]),
            block_size: v["block_size"].as_u64().unwrap().try_into().unwrap(),
            blocks_per_program: v["blocks_per_program"]
                .as_u64()
                .unwrap()
                .try_into()
                .unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        print!("{}", triton_frontend::mxfp8::source_ttir(&s));
        return;
    }
    if v["operation"] == "rms_rope" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Unsupported development scalar type"),
        });
        let s = triton_frontend::rope::rms::Specialization {
            input: element(&v["input"]),
            scale: element(&v["scale"]),
            frequencies: element(&v["frequencies"]),
            block_size: v["block_size"].as_u64().unwrap().try_into().unwrap(),
            split_half: v["split_half"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        if v.get("epsilon_type").is_some_and(|v| v != "fp32") {
            print!(
                "{}",
                triton_frontend::rope::rms::source_ttir_scalar(&s, epsilon(&v))
            );
        } else {
            print!("{}", triton_frontend::rope::rms::source_ttir(&s));
        }
        return;
    }
    if v["operation"] == "rope" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Unsupported development scalar type"),
        });
        let s = triton_frontend::rope::Specialization {
            input: element(&v["input"]),
            key: if v["key"].is_null() {
                None
            } else {
                Some(element(&v["key"]))
            },
            frequencies: element(&v["frequencies"]),
            block_size: v["block_size"].as_u64().unwrap().try_into().unwrap(),
            split_half: v["split_half"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
        };
        print!("{}", triton_frontend::rope::source_ttir(&s));
        return;
    }
    if v["operation"] == "adaln" {
        let scalars = std::array::from_fn(|i| match v["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: v["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Unsupported development scalar type"),
        });
        let s = triton_frontend::adaln::Specialization {
            input: element(&v["input"]),
            scale: element(&v["scale"]),
            shift: element(&v["shift"]),
            block_size: v["block_size"].as_u64().unwrap().try_into().unwrap(),
            subtract_mean: v["subtract_mean"].as_bool().unwrap(),
            scalars,
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        if v.get("epsilon_type").is_some_and(|v| v != "fp32") {
            print!(
                "{}",
                triton_frontend::adaln::source_ttir_scalar(&s, epsilon(&v))
            );
        } else {
            print!("{}", triton_frontend::adaln::source_ttir(&s));
        }
        return;
    }
    let count = match v["count_type"].as_str().unwrap() {
        "i32" => Count::I32 {
            divisible_by_16: v["count_div16"].as_bool().unwrap(),
        },
        "i64" => Count::I64 {
            divisible_by_16: v["count_div16"].as_bool().unwrap(),
        },
        "one" => Count::One,
        _ => panic!("Unsupported development count type"),
    };
    if v["operation"] == "rowwise" {
        let s = triton_frontend::rowwise::Specialization {
            input: element(&v["input"]),
            input_dtype_code: v["input_dtype_code"].as_u64().unwrap().try_into().unwrap(),
            block_size: v["block_size"].as_u64().unwrap().try_into().unwrap(),
            pointer_divisibility_16: std::array::from_fn(|i| {
                v["pointer_div16"][i].as_bool().unwrap()
            }),
            count,
            source_path: v["source_path"].as_str().unwrap(),
            standard_source_path: v["standard_source_path"].as_str().unwrap(),
        };
        print!("{}", triton_frontend::rowwise::source_ttir(&s));
        return;
    }
    let s = Fp8Specialization {
        body: if v["operation"] == "quantize" {
            Fp8Body::Quantize
        } else {
            Fp8Body::Dequantize
        },
        input: element(&v["input"]),
        output: element(&v["output"]),
        scale: element(&v["scale"]),
        block_size: v["block_size"].as_u64().unwrap().try_into().unwrap(),
        pointer_divisibility_16: std::array::from_fn(|i| v["pointer_div16"][i].as_bool().unwrap()),
        count,
        source_path: v["source_path"].as_str().unwrap(),
    };
    print!(
        "{}",
        if v["raw"] == true {
            triton_frontend::fp8_source_ttir(&s)
        } else {
            fp8_ttir(&s)
        }
    );
}
