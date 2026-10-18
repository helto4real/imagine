use super::fixtures;
use imagine_inference::triton_specialization::{
    CompilerHashInputs, compiler_digest, jit_source_digest,
};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
#[test]
fn all_7740_original_explicit_and_jit_binder_attribute_keys_preserve_distinct_hashes() {
    let profile: serde_json::Value = serde_json::from_slice(
        &std::fs::read(Path::new(
            &std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap(),
        ))
        .unwrap(),
    )
    .unwrap();
    let cases = fixtures()["triton_jit_keys"].as_array().unwrap();
    assert_eq!(cases.len(), 7740);
    for c in cases {
        let key = c["explicit_source_key"].as_str().unwrap();
        let explicit = Sha256::digest(key.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(explicit, c["explicit_ast_digest"]);
        let names = c["argument_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap())
            .collect::<Vec<_>>();
        let jit = jit_source_digest(key, &names);
        assert_eq!(jit, c["jit_ast_digest"], "{}", c["family"]);
        let inputs = CompilerHashInputs {
            triton_version_key: profile["triton_version_key"].as_str().unwrap(),
            backend_hash: profile["backend_hash"].as_str().unwrap(),
            options_hash: c["options_hash"].as_str().unwrap(),
        };
        assert_eq!(
            compiler_digest(&inputs, &jit, &BTreeMap::new()),
            c["jit_compiler_digest"]
        );
    }
}
