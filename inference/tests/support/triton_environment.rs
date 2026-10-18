use super::{fixtures, kitchen};
use imagine_inference::{
    triton_compilation_modes::Modes,
    triton_compiler::Compiler,
    triton_environment::{boolean_knob, normalize},
    triton_file_cache::JsonPath,
    triton_frontend::{Count, Element, Fp8Body, Fp8Specialization},
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
};
fn sha(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
#[test]
fn named_environment_values_and_boolean_knobs_preserve_the_two_original_normalizations() {
    for c in fixtures()["triton_environment"].as_array().unwrap() {
        let values = serde_json::from_value(c["values"].clone()).unwrap();
        assert_eq!(serde_json::json!(normalize(&values)), c["expected"]);
    }
    for c in fixtures()["triton_boolean_knobs"].as_array().unwrap() {
        assert_eq!(
            boolean_knob(c["value"].as_str().unwrap()),
            c["expected"].as_bool().unwrap()
        );
    }
}
#[test]
fn original_environment_changes_preserve_warm_jit_identity_then_cold_disk_hash_metadata_and_restore()
 {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native,
        )
        .unwrap()
    };
    let spec = Fp8Specialization {
        body: Fp8Body::Quantize,
        input: Element::F32,
        output: Element::E4M3,
        scale: Element::F32,
        block_size: 128,
        pointer_divisibility_16: [true; 3],
        count: Count::I32 {
            divisible_by_16: true,
        },
        source_path: compiler.source_path(),
    };
    let mut previous = None;
    for c in fixtures()["triton_environment_cache"].as_array().unwrap() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        compiler.set_compilation_modes(Modes {
            environment: Some(serde_json::from_value(c["environment"].clone()).unwrap()),
            listener: Some(Arc::new(move |event| {
                seen.lock().unwrap().push(event.cache_hit);
                Ok(())
            })),
            ..Modes::default()
        });
        if c["clear"] == true {
            compiler.clear_specializations();
        }
        let kernel = unsafe { compiler.jit_fp8(0, &spec).unwrap() };
        assert_eq!(kernel.digest, c["digest"]);
        assert_eq!(kernel.metadata, c["metadata"]);
        assert_eq!(
            previous.as_ref().is_some_and(|p| Arc::ptr_eq(p, &kernel)),
            c["same_object_as_previous"].as_bool().unwrap()
        );
        assert_eq!(
            serde_json::json!(*events.lock().unwrap()),
            c["compiler_events"]
        );
        let assembly = kernel
            .assembly
            .iter()
            .map(|(name, data)| (name.clone(), sha(data)))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(serde_json::json!(assembly), c["assembly"]);
        let (_, path) = kernel
            .metadata_group
            .iter()
            .find(|(key, _)| key.to_utf8().unwrap().ends_with(".json"))
            .unwrap();
        let JsonPath::Text(path) = path else {
            unreachable!()
        };
        assert_eq!(
            sha(&std::fs::read(path.to_utf8().unwrap()).unwrap()),
            c["metadata_json_sha256"]
        );
        previous = Some(kernel);
    }
}
