use super::{fixtures, kitchen};
use imagine_inference::{
    triton_compiler::Compiler,
    triton_frontend::{Count, w4a8},
    triton_specialization::w4a8_ast_digest,
};
use std::{path::Path, sync::Arc};

#[test]
fn original_w4a8_unpack_keys_complete_binaries_and_cache_restoration() {
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
    let key = fixtures()["triton_w4a8_function_cache_keys"]["w4a8"]
        .as_str()
        .unwrap();
    let mut compiled = 0;
    let times = super::cubin_source_times::times(&fixtures()["triton_w4a8_compiler_source_files"]);
    for c in fixtures()["triton_w4a8_specialization"].as_array().unwrap() {
        let s = w4a8::Specialization {
            has_cb: c["has_cb"].as_bool().unwrap(),
            bits: c["bits"].as_u64().unwrap() as u32,
            block_n: c["block_n"].as_u64().unwrap() as u32,
            block_kh: c["block_kh"].as_u64().unwrap() as u32,
            scalars: std::array::from_fn(|i| match c["scalar_types"][i].as_str().unwrap() {
                "one" => Count::One,
                "i32" => Count::I32 {
                    divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
                },
                "i64" => Count::I64 {
                    divisible_by_16: c["scalar_div16"][i].as_bool().unwrap(),
                },
                _ => panic!("Original unpack scalar type"),
            }),
            pointer_divisibility_16: std::array::from_fn(|i| {
                c["pointer_div16"][i].as_bool().unwrap()
            }),
            source_path: "",
        };
        assert_eq!(w4a8_ast_digest(&s, key), c["ast_digest"].as_str().unwrap());
        let filename = format!("case-{}.expected.cubin", c["index"]);
        if let Some(expected) = c["files"][&filename].as_str() {
            let image = unsafe { compiler.w4a8_unpack(0, &s).unwrap() };
            let cubin = &image
                .assembly
                .iter()
                .find(|(ext, _)| ext == "cubin")
                .unwrap()
                .1;
            let hash = super::cubin_source_times::digest(cubin, &times);
            if hash != expected {
                eprintln!(
                    "W4A8_ASSEMBLY_DIAGNOSTIC {}",
                    serde_json::json!({"case":c["index"],"metadata":image.metadata,"assembly":image.assembly.iter().filter(|(name,_)|name!="cubin").map(|(name,bytes)|(name.clone(),String::from_utf8_lossy(bytes).into_owned())).collect::<std::collections::BTreeMap<_,_>>()})
                );
            }
            assert_eq!(hash, expected);
            assert!(Arc::ptr_eq(&image, &unsafe {
                compiler.w4a8_unpack(0, &s).unwrap()
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
            let restored = unsafe { fresh.w4a8_unpack(0, &s).unwrap() };
            assert!(restored.disk_cache_hit);
            assert_eq!(restored.digest, image.digest);
            assert_eq!(restored.metadata, image.metadata);
            assert_eq!(restored.assembly, image.assembly);
            compiled += 1;
        }
    }
    assert_eq!(compiled, 4);
}

#[test]
fn registered_w4a8_wrapper_preserves_original_unpack_gemm_boundaries_outputs_errors_and_shared_autotuners()
 {
    super::triton_wrapper_tests::compare("w4a8");
}
