use super::binding_tests::reference_error;
use super::*;
use imagine_inference::{
    native::Error,
    triton_file_cache::{FileCache, JsonPath, Mode},
};
use std::path::PathBuf;
pub(super) struct OwnDirectory {
    previous: PathBuf,
    root: PathBuf,
}
impl OwnDirectory {
    pub(super) fn create() -> Self {
        let previous = std::env::current_dir().unwrap();
        let root = (0..)
            .find_map(|index| {
                let root = PathBuf::from(format!(
                    "/tmp/imagine-parity-file-cache-{}-{index}",
                    std::process::id()
                ));
                match std::fs::create_dir(&root) {
                    Ok(()) => Some(root),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                    Err(e) => panic!("Own file-cache test root: {e}"),
                }
            })
            .unwrap();
        std::env::set_current_dir(&root).unwrap();
        Self { previous, root }
    }
}
impl Drop for OwnDirectory {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous).unwrap();
        std::fs::remove_dir_all(&self.root).unwrap();
    }
}
fn path_value(p: Option<PathBuf>) -> Value {
    p.map_or(Value::Null, |p| serde_json::json!(p.to_str().unwrap()))
}
fn json_path(p: JsonPath) -> Value {
    match p {
        JsonPath::Text(s) => serde_json::json!({"text":s.code_points()}),
        JsonPath::Signed(n) => serde_json::json!({"value":n}),
        JsonPath::Unsigned(n) => serde_json::json!({"value":n}),
        JsonPath::Bool(b) => serde_json::json!({"value":b}),
    }
}
fn decode(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
#[test]
fn local_triton_file_cache_preserves_modes_groups_missing_corrupt_records_and_atomic_file_state() {
    let cases = &fixtures()["triton_file_cache"];
    let _directory = OwnDirectory::create();
    for (i, c) in cases.as_array().unwrap().iter().enumerate() {
        if c["empty_base"] == true {
            match FileCache::new(Path::new(""), "EMPTY", Mode::Cache) {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected empty cache error"),
            };
            continue;
        }
        let mode = if c["dump"] == true {
            Mode::Dump
        } else if c["override"] == true {
            Mode::Override
        } else {
            Mode::Cache
        };
        let base = match mode {
            Mode::Cache => "cache/main",
            Mode::Dump => "cache/dump",
            Mode::Override => "cache/override",
        };
        let cache = FileCache::new(Path::new(base), c["key"].as_str().unwrap(), mode).unwrap();
        assert_eq!(cache.directory().to_str().unwrap(), c["directory"]);
        assert_eq!(
            cache.lock_path().map(|p| p.to_str().unwrap()),
            c["lock"].as_str()
        );
        assert_eq!(
            cache.directory().is_dir(),
            c["before_directory"].as_bool().unwrap()
        );
        for (j, step) in c["steps"].as_array().unwrap().iter().enumerate() {
            let args = &step["args"];
            let s = |n: usize| args[n].as_str().unwrap();
            let result = (|| -> Result<Value, Error> {
                Ok(match step["operation"].as_str().unwrap() {
                    "put" => path_value(Some(cache.put(&decode(s(0)), s(1))?)),
                    "put_group" => {
                        let group = args[1]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| {
                                (
                                    v[0].as_str().unwrap().to_string(),
                                    v[1].as_str().unwrap().to_string(),
                                )
                            })
                            .collect::<Vec<_>>();
                        let path = cache.put_group(s(0), &group)?;
                        if s(0) == "kernel.json" {
                            let bytes = std::fs::read(&path).unwrap();
                            let hex = bytes.iter().map(|x| format!("{x:02x}")).collect::<String>();
                            assert_eq!(hex, c["group_bytes"], "Original JSON bytes");
                        }
                        path_value(Some(path))
                    }
                    "get_file" => path_value(cache.get_file(s(0))),
                    "get_group" => cache.get_group(s(0))?.map_or(Value::Null, |group| {
                        Value::Array(
                            group
                                .into_iter()
                                .map(|(k, v)| serde_json::json!([k.code_points(), json_path(v)]))
                                .collect(),
                        )
                    }),
                    "write_group" => {
                        std::fs::write(
                            cache.directory().join(format!("__grp__{}", s(0))),
                            decode(s(1)),
                        )
                        .unwrap();
                        Value::Null
                    }
                    "put_group_points" => {
                        use imagine_inference::triton_file_cache::CacheText;
                        let text = |v: &Value| {
                            CacheText::from_code_points(
                                v.as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|n| n.as_u64().unwrap() as u32)
                                    .collect(),
                            )
                            .unwrap()
                        };
                        let group = args[1]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| (text(&v[0]), text(&v[1])))
                            .collect::<Vec<_>>();
                        let path = cache.put_group_text(s(0), &group)?;
                        let actual = std::fs::read(&path)
                            .unwrap()
                            .iter()
                            .map(|x| format!("{x:02x}"))
                            .collect::<String>();
                        assert_eq!(
                            actual, step["bytes"],
                            "Original surrogate-containing JSON bytes"
                        );
                        path_value(Some(path))
                    }
                    "prepare_nonutf8" => {
                        use std::os::unix::ffi::{OsStrExt, OsStringExt};
                        let mut path = cache.directory().as_os_str().as_bytes().to_vec();
                        path.extend_from_slice(b"/byte-");
                        path.push(255);
                        std::fs::write(std::ffi::OsString::from_vec(path), b"own byte filename")
                            .unwrap();
                        Value::Null
                    }
                    "prepare_override" => {
                        std::fs::create_dir_all(cache.directory()).unwrap();
                        std::fs::write(cache.directory().join("kernel.bin"), b"own override bytes")
                            .unwrap();
                        Value::Null
                    }
                    "mkdir" => {
                        std::fs::create_dir(cache.directory().join(s(0))).unwrap();
                        Value::Null
                    }
                    "unlink" => {
                        std::fs::remove_file(s(0)).unwrap();
                        Value::Null
                    }
                    _ => panic!("Unknown own cache operation"),
                })
            })();
            if step["error"].is_string() {
                match result {
                    Err(e) => reference_error(e, step),
                    Ok(v) => panic!("Expected original cache error {i}/{j}: {v}"),
                }
            } else {
                assert_eq!(
                    result.unwrap_or_else(|e| panic!("Cache {i}/{j}: {e}")),
                    step["output"],
                    "Cache {i}/{j}"
                );
            }
        }
        let mut remaining = std::fs::read_dir(cache.directory())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n.as_encoded_bytes().starts_with(b"tmp.pid_"))
            .map(|n| n.into_string().unwrap())
            .collect::<Vec<_>>();
        remaining.sort();
        assert_eq!(serde_json::json!(remaining), c["temporary_remains"]);
        assert_eq!(
            cache.lock_path().is_some_and(Path::exists),
            c["lock_exists"].as_bool().unwrap()
        );
    }
}
