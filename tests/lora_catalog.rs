//! Hermetic catalog and API contracts: own explicitly provided fixture root, no inference.
use axum::{
    Router,
    body::{Body, to_bytes},
    http::Request,
};
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    scheduler::{self, Action, Control},
    storage::Store,
    video::lora_catalog::{self as catalog, Error},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tokio::sync::{Semaphore, watch};
use tower::ServiceExt;

// Preserve fixtures as evidence; no automatic temp-directory cleanup or external root fallback.
fn fixture() -> PathBuf {
    let root = PathBuf::from(
        std::env::var_os("IMAGINE_LORA_TEST_ROOT")
            .expect("explicit own encrypted fixture root required"),
    );
    let root = root.join(uuid::Uuid::new_v4().to_string());
    fs::create_dir_all(&root).unwrap();
    root
}
fn write(path: &Path, bytes: &[u8]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}
fn safetensors(metadata: Value) -> Vec<u8> {
    let header =
        json!({"__metadata__":metadata, "weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}});
    tensor_bytes(header, &[0, 0, 0, 0])
}
fn tensor_bytes(header: Value, data: &[u8]) -> Vec<u8> {
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    bytes
}

#[test]
fn catalog_is_nested_sorted_extension_neutral_and_does_not_parse_files() {
    let root = fixture();
    write(&root.join("z.odd"), b"not a tensor");
    write(&root.join("a/deeper/model.bin"), b"123");
    write(&root.join("a/empty"), b"");
    let found = catalog::scan(&root).unwrap();
    assert_eq!(
        found
            .files
            .iter()
            .map(|v| (v.file.as_str(), v.size))
            .collect::<Vec<_>>(),
        vec![("a/deeper/model.bin", 3), ("a/empty", 0), ("z.odd", 12)]
    );
    assert!(found.warnings.is_empty());
    assert_eq!(found, catalog::scan(&root).unwrap());
}
#[test]
fn catalog_root_errors_are_typed() {
    let root = fixture();
    assert_eq!(
        catalog::scan(&root.join("missing")),
        Err(Error::CatalogUnavailable)
    );
    write(&root.join("file"), b"x");
    assert_eq!(
        catalog::scan(&root.join("file")),
        Err(Error::CatalogUnavailable)
    );
    assert!(
        catalog::scan(&root.join("missing"))
            .unwrap_err()
            .code()
            .ends_with("unavailable")
    );
}
#[cfg(unix)]
#[test]
fn catalog_symlink_cycle_and_alias_are_finite_and_deterministic() {
    use std::os::unix::fs::symlink;
    let root = fixture();
    write(&root.join("real/model.unknown"), b"x");
    symlink(&root, root.join("real/cycle")).unwrap();
    symlink(root.join("real"), root.join("alias")).unwrap();
    symlink(root.join("missing"), root.join("broken")).unwrap();
    let found = catalog::scan(&root).unwrap();
    assert_eq!(found.files.len(), 1);
    assert_eq!(found.files[0].file, "alias/model.unknown");
    assert_eq!(
        found.warnings,
        vec![
            "lora_catalog_directory_cycle_or_alias",
            "lora_catalog_entry_unreadable"
        ]
    );
    assert_eq!(found, catalog::scan(&root).unwrap());
}
#[cfg(unix)]
#[test]
fn catalog_non_utf8_names_are_reported_without_colliding_lossy_names() {
    use std::os::unix::ffi::OsStringExt;
    let root = fixture();
    write(&root.join(std::ffi::OsString::from_vec(vec![0xff])), b"x");
    let found = catalog::scan(&root).unwrap();
    assert!(found.files.is_empty());
    assert_eq!(found.warnings, vec!["lora_catalog_name_unrepresentable"]);
}
#[cfg(unix)]
#[test]
fn unreadable_catalog_and_info_return_safe_codes() {
    use std::os::unix::fs::PermissionsExt;
    let root = fixture();
    let blocked = root.join("blocked");
    fs::create_dir(&blocked).unwrap();
    fs::set_permissions(&blocked, fs::Permissions::from_mode(0o0)).unwrap();
    // This regression requires a non-root caller; exercising it under root would bypass DAC.
    assert_ne!(
        unsafe { libc::geteuid() },
        0,
        "run contracts as ordinary task user"
    );
    assert_eq!(catalog::scan(&blocked), Err(Error::CatalogUnavailable));
    write(&root.join("unreadable"), b"data");
    fs::set_permissions(root.join("unreadable"), fs::Permissions::from_mode(0o0)).unwrap();
    assert_eq!(
        catalog::info(&root, "unreadable"),
        Err(Error::InfoUnreadable)
    );
    let result = catalog::scan(&root).unwrap();
    assert_eq!(result.warnings, vec!["lora_catalog_directory_unreadable"]);
}
#[test]
fn info_reads_real_header_metadata_with_unknown_extension_and_ranked_words() {
    let root = fixture();
    let bytes = safetensors(json!({
        "ss_output_name":"Example", "ss_base_model_version":"custom-compatible-model",
        "ss_network_module":"networks.lora", "ss_tag_frequency":r#"{"first":{"word":2,"tie":3},"second":{"word":4,"alpha":3}}"#
    }));
    write(&root.join("nested/selected.custom"), &bytes);
    let info = catalog::info(&root, "nested/selected.custom").unwrap();
    assert_eq!(info.file, "nested/selected.custom");
    assert_eq!(info.name, "Example");
    assert_eq!(info.base_model, "custom-compatible-model");
    assert_eq!(info.model_type, "networks.lora");
    assert_eq!(info.format, "safetensors");
    assert_eq!(info.trained_words, vec!["word", "alpha", "tie"]);
    assert_eq!(info.sha256, format!("{:x}", Sha256::digest(&bytes)));
}
#[test]
fn info_manual_absolute_outside_catalog_and_relative_parent_paths_work() {
    let root = fixture();
    fs::create_dir(root.join("catalog")).unwrap();
    let path = root.join("outside.arbitrary");
    write(
        &path,
        &safetensors(json!({"modelspec.architecture":"future-family"})),
    );
    let info = catalog::info(&root.join("catalog"), path.to_str().unwrap()).unwrap();
    assert_eq!(info.base_model, "future-family");
    assert_eq!(
        catalog::info(&root.join("catalog"), "../outside.arbitrary")
            .unwrap()
            .sha256,
        info.sha256
    );
    assert!(
        catalog::scan(&root.join("catalog"))
            .unwrap()
            .files
            .is_empty()
    );
}
#[test]
fn metadata_fallbacks_and_empty_metadata_are_optional() {
    let root = fixture();
    write(
        &root.join("fallback.tensor"),
        &safetensors(json!({"ss_sd_model_name":"base", "ss_network_type":"LoRA"})),
    );
    let info = catalog::info(&root, "fallback.tensor").unwrap();
    assert_eq!(
        (
            info.name.as_str(),
            info.base_model.as_str(),
            info.model_type.as_str()
        ),
        ("fallback", "base", "LoRA")
    );
    write(
        &root.join("empty.tensor"),
        &tensor_bytes(
            json!({"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
            &[0; 4],
        ),
    );
    assert!(
        catalog::info(&root, "empty.tensor")
            .unwrap()
            .trained_words
            .is_empty()
    );
}
#[test]
fn malformed_optional_tag_frequency_does_not_block_information() {
    let root = fixture();
    write(
        &root.join("m"),
        &safetensors(json!({"ss_tag_frequency":"not JSON"})),
    );
    assert!(catalog::info(&root, "m").unwrap().trained_words.is_empty());
}
#[test]
fn missing_directory_empty_and_unknown_file_errors_are_safe() {
    let root = fixture();
    assert_eq!(catalog::info(&root, "missing"), Err(Error::InfoMissing));
    assert_eq!(catalog::info(&root, " "), Err(Error::InfoMissing));
    assert_eq!(catalog::info(&root, "."), Err(Error::InfoFormat));
    write(&root.join("unknown"), b"opaque format");
    assert_eq!(catalog::info(&root, "unknown"), Err(Error::InfoFormat));
    assert_eq!(catalog::scan(&root).unwrap().files[0].file, "unknown");
}
#[test]
fn oversized_truncated_and_non_object_headers_are_rejected_without_allocating_tensor_data() {
    let root = fixture();
    for (name, bytes) in [
        ("huge", u64::MAX.to_le_bytes().to_vec()),
        ("truncated", 128u64.to_le_bytes().to_vec()),
        ("not-object", tensor_bytes(json!([]), &[])),
        ("metadata-not-strings", safetensors(json!({"key":123}))),
    ] {
        write(&root.join(name), &bytes);
        assert_eq!(catalog::info(&root, name), Err(Error::InfoFormat), "{name}");
    }
}
#[test]
fn invalid_offsets_shapes_and_non_contiguous_data_are_rejected() {
    let root = fixture();
    let cases = [
        json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,5]}}),
        json!({"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}),
        json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[1,4]}}),
        json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
        json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0]}}),
    ];
    for (index, header) in cases.into_iter().enumerate() {
        let name = format!("bad-{index}");
        write(&root.join(&name), &tensor_bytes(header, &[0; 4]));
        assert_eq!(catalog::info(&root, &name), Err(Error::InfoFormat));
    }
}
#[test]
fn known_containers_are_informational_only_and_never_unpickled() {
    let root = fixture();
    let mut gguf = b"GGUF".to_vec();
    gguf.extend(3u32.to_le_bytes());
    gguf.extend([0; 16]);
    for (name, bytes, format) in [
        ("gguf", gguf, "GGUF"),
        (
            "zip",
            b"PK\x03\x04opaque".to_vec(),
            "ZIP (possible PyTorch)",
        ),
        ("pickle", vec![0x80, 2, b'.'], "Pickle (possible PyTorch)"),
    ] {
        write(&root.join(name), &bytes);
        let info = catalog::info(&root, name).unwrap();
        assert_eq!(info.format, format);
        assert_eq!(info.sha256, format!("{:x}", Sha256::digest(bytes)));
    }
    write(&root.join("short-gguf"), b"GGUF");
    assert_eq!(catalog::info(&root, "short-gguf"), Err(Error::InfoFormat));
}

struct NoInference;
impl Backend for NoInference {
    fn generate<'a>(
        &'a mut self,
        _: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async { anyhow::bail!("unexpected inference") })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
struct Harness {
    router: Router,
    store: Store,
    control: Control,
    scheduler: tokio::task::JoinHandle<anyhow::Result<()>>,
    library: Arc<Mutex<imagine_server::model_library::Library>>,
}
async fn setup(root: &Path) -> Harness {
    let data = root.join("api-data");
    fs::create_dir_all(&data).unwrap();
    let store = Store::open(&data).await.unwrap();
    let (control, scheduler) = scheduler::start(store.clone(), Box::new(NoInference))
        .await
        .unwrap();
    let (_, engine) = watch::channel(EngineState::default());
    let library = imagine_server::model_library::Library::load(
        root.join("models"),
        root.join("library-settings.json"),
    )
    .unwrap();
    let library = Arc::new(Mutex::new(library));
    let router = api::router(App {
        video: control.video.clone(),
        enhancement: imagine_server::enhancement::Service::open(data.join("enhancement.json"))
            .unwrap(),
        library: library.clone(),
        profiles: Arc::new(
            imagine_server::profiles::Catalog::load(Path::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/profiles"
            )))
            .unwrap(),
        ),
        store: store.clone(),
        control: control.clone(),
        engine,
        uploads: Arc::new(Semaphore::new(2)),
    });
    Harness {
        router,
        store,
        control,
        scheduler,
        library,
    }
}
async fn stop(h: Harness) {
    h.control.call(Action::Shutdown).await.unwrap();
    h.scheduler.await.unwrap().unwrap();
    h.store.close().await.unwrap();
}
async fn call(
    h: &Harness,
    method: &str,
    path: &str,
    body: Option<Value>,
    session: Option<&str>,
) -> (u16, Value) {
    let mut request = Request::builder()
        .method(method)
        .uri(path)
        .header("Content-Type", "application/json");
    if let Some(session) = session {
        request = request.header("X-Imagine-Session", session);
    }
    let response = h
        .router
        .clone()
        .oneshot(
            request
                .body(
                    body.map(|v| Body::from(v.to_string()))
                        .unwrap_or_else(Body::empty),
                )
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.headers()["Cache-Control"], "no-store");
    let status = response.status().as_u16();
    let body = to_bytes(response.into_body(), 2 * 1024 * 1024)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}
#[tokio::test]
async fn api_lists_and_inspects_manual_unknown_extension_without_catalog_membership() {
    let root = fixture();
    write(
        &root.join("models/loras/nested/m.custom"),
        &safetensors(json!({"ss_output_name":"Custom"})),
    );
    write(&root.join("external.manual"), &safetensors(json!({})));
    let h = setup(&root).await;
    let sid = Some(h.store.session.id.as_str());
    let (status, list) = call(&h, "GET", "/v1/video/loras", None, sid).await;
    assert_eq!(status, 200);
    assert_eq!(list["files"][0]["file"], "nested/m.custom");
    let (status, info) = call(
        &h,
        "POST",
        "/v1/video/loras/info",
        Some(json!({"schema_version":1,"file":"nested/m.custom"})),
        sid,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(info["name"], "Custom");
    let (status, info) = call(
        &h,
        "POST",
        "/v1/video/loras/info",
        Some(json!({"schema_version":1,"file":root.join("external.manual")})),
        sid,
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(info["format"], "safetensors");
    stop(h).await;
}
#[tokio::test]
async fn api_preserves_session_schema_and_safe_failure_contracts() {
    let root = fixture();
    write(&root.join("models/loras/bad"), b"bad");
    let h = setup(&root).await;
    for (path, body) in [
        ("/v1/video/loras", None),
        (
            "/v1/video/loras/info",
            Some(json!({"schema_version":1,"file":"bad"})),
        ),
    ] {
        let method = if body.is_some() { "POST" } else { "GET" };
        let (status, result) = call(&h, method, path, body.clone(), None).await;
        assert_eq!(
            (status, result["error"]["code"].as_str()),
            (401, Some("session_required"))
        );
        let (status, result) = call(&h, method, path, body, Some("wrong-session")).await;
        assert_eq!(
            (status, result["error"]["code"].as_str()),
            (409, Some("session_changed"))
        );
    }
    let sid = Some(h.store.session.id.as_str());
    for (body, status, code) in [
        (
            json!({"schema_version":2,"file":"bad"}),
            422,
            "unsupported_schema",
        ),
        (
            json!({"schema_version":1,"file":"bad","extra":true}),
            422,
            "invalid_request",
        ),
        (
            json!({"schema_version":1,"file":"bad"}),
            422,
            "lora_info_format",
        ),
        (
            json!({"schema_version":1,"file":"missing"}),
            404,
            "lora_info_missing",
        ),
    ] {
        let (actual, result) = call(&h, "POST", "/v1/video/loras/info", Some(body), sid).await;
        assert_eq!((actual, result), (status, json!({"error":{"code":code}})));
    }
    stop(h).await;
}
#[tokio::test]
async fn api_missing_catalog_is_optional_and_explicit_manual_info_still_works() {
    let root = fixture();
    write(&root.join("external"), &safetensors(json!({})));
    let h = setup(&root).await;
    let sid = Some(h.store.session.id.as_str());
    let (status, result) = call(&h, "GET", "/v1/video/loras", None, sid).await;
    assert_eq!(
        (status, result),
        (422, json!({"error":{"code":"lora_catalog_unavailable"}}))
    );
    let (status, _) = call(
        &h,
        "POST",
        "/v1/video/loras/info",
        Some(json!({"schema_version":1,"file":root.join("external")})),
        sid,
    )
    .await;
    assert_eq!(status, 200);
    stop(h).await;
}

#[tokio::test]
async fn api_manual_absolute_info_does_not_require_configured_catalog_root() {
    let root = fixture();
    write(&root.join("external"), &safetensors(json!({})));
    let h = setup(&root).await;
    *h.library.lock().unwrap() = imagine_server::model_library::Library::default();
    let sid = Some(h.store.session.id.as_str());
    let (status, result) = call(&h, "GET", "/v1/video/loras", None, sid).await;
    assert_eq!(
        (status, result),
        (422, json!({"error":{"code":"lora_catalog_unavailable"}}))
    );
    let (status, _) = call(
        &h,
        "POST",
        "/v1/video/loras/info",
        Some(json!({"schema_version":1,"file":root.join("external")})),
        sid,
    )
    .await;
    assert_eq!(status, 200);
    stop(h).await;
}
