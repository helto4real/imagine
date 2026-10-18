use super::{fixtures, kitchen};
use imagine_inference::{
    triton_compilation_modes::Modes,
    triton_compiler::Compiler,
    triton_file_cache::JsonPath,
    triton_frontend::{Count, Element, Fp8Body, Fp8Specialization},
    triton_specialization::cache_directory,
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
fn original_compiler_modes_preserve_cache_layers_assemblies_metadata_overrides_and_sass_dumps() {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    for c in fixtures()["triton_compiler_modes"].as_array().unwrap() {
        let label = c["label"].as_str().unwrap();
        let root = own.join(label);
        let compiler = unsafe {
            Compiler::open(
                Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
                &root.join("cache"),
                &root.join("work"),
                native.clone(),
            )
            .unwrap()
        };
        let s = Fp8Specialization {
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
        if label == "override" {
            let seed = unsafe { compiler.fp8(0, &s).unwrap() };
            let path = root
                .join("override")
                .join(cache_directory(&seed.ast_digest).unwrap());
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join("quantize_fp8_kernel_tl.ttir"),
                &seed.assembly.iter().find(|(k, _)| k == "ttir").unwrap().1,
            )
            .unwrap();
            compiler.clear_specializations();
            std::fs::remove_dir_all(root.join("cache")).unwrap();
        }
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        compiler.set_compilation_modes(Modes{environment:None,always_compile:label=="always_compile",store_binary_only:label=="store_binary",override_directory:(label=="override").then(||root.join("override")),dump_directory:(label=="dump").then(||root.join("dump")),listener:Some(Arc::new(move|event|{seen.lock().unwrap().push(serde_json::json!({"cache_hit":event.cache_hit,"group":event.metadata_group.iter().map(|(k,_)|k.to_utf8().unwrap()).collect::<Vec<_>>(),"stages":event.lowering_stages}));Ok(())}))});
        for expected in c["calls"].as_array().unwrap() {
            let before = events.lock().unwrap().len();
            compiler.clear_specializations();
            let kernel = unsafe { compiler.fp8(0, &s).unwrap() };
            assert_eq!(kernel.digest, expected["hash"]);
            assert_eq!(kernel.metadata, expected["metadata"], "{label}");
            let assembly = kernel
                .assembly
                .iter()
                .map(|(k, v)| (k.clone(), sha(v)))
                .collect::<BTreeMap<_, _>>();
            assert_eq!(serde_json::json!(assembly), expected["asm"], "{label}");
            let listener = expected["events"]
                .as_array()
                .unwrap()
                .iter()
                .find(|e| e["method"] == "listener")
                .unwrap();
            let actual = events.lock().unwrap()[before].clone();
            assert_eq!(actual["cache_hit"], listener["cache_hit"]);
            assert_eq!(actual["group"], listener["group"]);
            assert_eq!(actual["stages"], listener["stages"]);
            assert_eq!(
                kernel.disk_cache_hit,
                listener["cache_hit"].as_bool().unwrap()
            );
            let warm = unsafe { compiler.fp8(0, &s).unwrap() };
            assert!(Arc::ptr_eq(&warm, &kernel));
            assert_eq!(
                events.lock().unwrap().len(),
                before + 1,
                "Warm JIT returns before compiler modes/listener"
            );
            for (key, path) in &kernel.metadata_group {
                let JsonPath::Text(path) = path else {
                    panic!("Original text path")
                };
                assert!(Path::new(&path.to_utf8().unwrap()).exists());
                assert!(!key.to_utf8().unwrap().is_empty());
            }
        }
        fn files(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
            if !dir.exists() {
                return;
            }
            for entry in std::fs::read_dir(dir).unwrap() {
                let p = entry.unwrap().path();
                if p.is_dir() {
                    files(root, &p, out);
                } else {
                    let name = p.file_name().unwrap().to_str().unwrap();
                    if !name.starts_with("__grp__") {
                        out.insert(
                            p.strip_prefix(root).unwrap().to_str().unwrap().into(),
                            sha(&std::fs::read(&p).unwrap()),
                        );
                    }
                }
            }
        }
        let mut actual = BTreeMap::new();
        for directory in ["cache", "dump", "override"] {
            files(&root, &root.join(directory), &mut actual);
        }
        let expected = c["files"]
            .as_object()
            .unwrap()
            .iter()
            .filter(|(name, _)| !name.split('/').next_back().unwrap().starts_with("__grp__"))
            .map(|(name, hash)| (name.clone(), hash.as_str().unwrap().to_string()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            actual, expected,
            "Exact stored source/stages/binary/metadata/dump files: {label}"
        );
    }
}
