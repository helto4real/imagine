use super::{fixtures, kitchen};
use imagine_inference::{
    triton_compiler::Compiler,
    triton_frontend::{Count, Element, na},
    triton_specialization::na_ast_digest,
};
use std::{path::Path, sync::Arc};
fn element(name: &str) -> Element {
    match name {
        "fp32" => Element::F32,
        "fp16" => Element::F16,
        "bf16" => Element::BF16,
        _ => panic!("Original NA3D element"),
    }
}
fn specialization(c: &serde_json::Value) -> na::Specialization<'static> {
    na::Specialization {
        input: element(c["input"].as_str().unwrap()),
        key: element(c.get("key").unwrap_or(&c["input"]).as_str().unwrap()),
        value: element(c.get("value").unwrap_or(&c["input"]).as_str().unwrap()),
        scale: match c["scale_type"].as_str().unwrap() {
            "fp32" => na::Scale::F32,
            "fp64" => na::Scale::F64,
            "i32" => na::Scale::I32 {
                divisible_by_16: c["scale_div16"].as_bool().unwrap(),
            },
            "u64" => na::Scale::U64 {
                divisible_by_16: c["scale_div16"].as_bool().unwrap(),
            },
            "i64" => na::Scale::I64 {
                divisible_by_16: c["scale_div16"].as_bool().unwrap(),
            },
            "one" => na::Scale::One,
            "u1" => na::Scale::Bool,
            _ => panic!("Original scale signature"),
        },
        kt: c["kt"].as_i64().unwrap(),
        kh: c["kh"].as_i64().unwrap(),
        kw: c["kw"].as_i64().unwrap(),
        causal_t: c["causal_t"].as_bool().unwrap(),
        causal_h: c["causal_h"].as_bool().unwrap(),
        causal_w: c["causal_w"].as_bool().unwrap(),
        hd: c["hd"].as_u64().unwrap() as u32,
        hd_pad: c["hd_pad"].as_u64().unwrap() as u32,
        block_q: c["block_q"].as_u64().unwrap() as u32,
        block_k: c["block_k"].as_u64().unwrap() as u32,
        scalars: std::array::from_fn(|i| match c["scalar_types"][i].as_str().unwrap() {
            "one" => Count::One,
            "i32" => Count::I32 {
                divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
            },
            "i64" => Count::I64 {
                divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
            },
            _ => panic!("Original index signature"),
        }),
        pointer_divisibility_16: std::array::from_fn(|i| c["pointer_div16"][i].as_bool().unwrap()),
        source_path: "",
        standard_source_path: "",
    }
}
#[test]
fn original_na3d_keys_complete_images_cache_restoration_and_source_failures_preserve_all_signatures()
 {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let profile = std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap();
    let native = kitchen();
    let compiler = unsafe {
        Compiler::open(
            Path::new(&profile),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    };
    let key = fixtures()["triton_na_function_cache_keys"]["na3d"]
        .as_str()
        .unwrap();
    let mut images = 0;
    let times = super::cubin_source_times::stage1("na");
    for c in fixtures()["triton_na_specialization"].as_array().unwrap() {
        let s = specialization(c);
        assert_eq!(na_ast_digest(&s, key), c["ast_digest"].as_str().unwrap());
        let file = format!("case-{}.expected.cubin", c["index"]);
        if let Some(expected) = c["files"][&file].as_str() {
            let image = unsafe { compiler.na3d(0, &s).unwrap() };
            let cubin = &image
                .assembly
                .iter()
                .find(|(ext, _)| ext == "cubin")
                .unwrap()
                .1;
            assert_eq!(super::cubin_source_times::digest(cubin, &times), expected);
            assert_eq!(image.metadata["shared"], c["compiled_metadata"]["shared"]);
            assert_eq!(image.metadata["num_warps"], 4);
            assert_eq!(image.metadata["num_stages"], 3);
            assert!(Arc::ptr_eq(&image, &unsafe {
                compiler.na3d(0, &s).unwrap()
            }));
            let fresh = unsafe {
                Compiler::open(
                    Path::new(&profile),
                    &own.join("cache"),
                    &own.join("work"),
                    native.clone(),
                )
                .unwrap()
            };
            let restored = unsafe { fresh.na3d(0, &s).unwrap() };
            assert!(restored.disk_cache_hit);
            assert_eq!(restored.metadata, image.metadata);
            assert_eq!(restored.assembly, image.assembly);
            images += 1;
        }
    }
    assert_eq!(images, 36);
    for original in fixtures()["triton_na_source_errors"].as_array().unwrap() {
        let s = specialization(&original["request"]);
        assert_eq!(
            na_ast_digest(&s, key),
            original["request"]["ast_digest"].as_str().unwrap()
        );
        for _ in 0..2 {
            let error = match unsafe { compiler.na3d(0, &s) } {
                Err(e) => e,
                Ok(_) => panic!("Original source failure"),
            };
            super::binding_tests::reference_error(
                error,
                &serde_json::json!({"error":original["class_name"],"message":original["message"]}),
            );
        }
    }
}
#[test]
fn registered_na3d_preserves_original_windows_precision_empty_cases_scale_signatures_strides_and_errors()
 {
    super::triton_wrapper_tests::compare("na");
}
