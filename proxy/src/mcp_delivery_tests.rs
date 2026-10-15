use super::*;
use std::path::Path;

async fn setup() -> (tempfile::TempDir, Mcp) {
    let dir = tempfile::tempdir().unwrap();
    let access = dir.path().join("access.json");
    std::fs::write(&access, r#"{"admin_token":"synthetic"}"#).unwrap();
    let mcp = Mcp::new(Config {
        coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
        imagine_url: "http://127.0.0.1:1".into(),
        imagine_access_file: Some(access.to_string_lossy().into()),
        mcp_output_dir: Some(dir.path().join("output").to_string_lossy().into()),
        ..Default::default()
    })
    .await
    .unwrap();
    (dir, mcp)
}
fn job(mcp: &Mcp, id: &str, count: usize, batch: bool) -> Value {
    let directory = mcp.inner.root.join(format!("job-{id}"));
    references::private_dir(&directory).unwrap();
    let items: Vec<_> = (0..count).map(|index| {
        let bytes = format!("synthetic file bytes {id} {index}");
        json!({"index":index,"status":"succeeded","images":references::save_result(&directory,index,bytes.as_bytes()).unwrap(),"error":null})
    }).collect();
    let mut snapshot = if batch {
        json!({"items":items})
    } else {
        items[0].clone()
    };
    snapshot["job_id"] = json!(id);
    snapshot["status"] = json!("succeeded");
    let (result, _) = watch::channel(snapshot.clone());
    mcp.inner.jobs.lock().unwrap().insert(
        id.into(),
        Job {
            directory,
            fingerprint: vec![],
            result,
            task: None,
            count,
        },
    );
    snapshot
}
fn copy_and_verify(image: &Value, destination: &Path) -> Value {
    std::fs::copy(image["path"].as_str().unwrap(), destination).unwrap();
    let metadata = std::fs::metadata(destination).unwrap();
    assert!(metadata.is_file());
    let sha256 = references::hash(&std::fs::read(destination).unwrap());
    assert_eq!(image["bytes"], metadata.len());
    assert_eq!(image["sha256"], sha256);
    json!({"image_id":image["image_id"],"saved_path":destination,"bytes":metadata.len(),"sha256":sha256})
}
async fn confirm(mcp: &Mcp, id: &str, images: Value) -> Value {
    // Exercise the RPC dispatcher and published tool schema as well as deletion.
    dispatch_rpc(
        mcp.clone(),
        json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
        "params":{"name":"confirm_saved","arguments":{"job_id":id,"images":images}}}),
        false,
    )
    .await["result"]
        .clone()
}
#[tokio::test]
async fn single_confirmation_updates_recovery_and_is_idempotent() {
    let (dir, mcp) = setup().await;
    let result = job(&mcp, "single", 1, false);
    let image = &result["images"][0];
    let temporary = PathBuf::from(image["path"].as_str().unwrap());
    let saved = copy_and_verify(image, &dir.path().join("saved.png"));
    let discovered = dispatch_rpc(
        mcp.clone(),
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        false,
    )
    .await;
    assert!(
        discovered["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "confirm_saved" && t["annotations"]["destructiveHint"] == true)
    );
    let a = confirm(&mcp, "single", json!([saved])).await;
    assert_eq!(a["isError"], false, "{a}");
    assert!(!temporary.exists());
    assert!(!temporary.parent().unwrap().exists());
    assert!(dir.path().join("saved.png").exists());
    assert_eq!(confirm(&mcp, "single", json!([saved])).await, a);
    let recovered = mcp.get("single", true).await.unwrap();
    assert_eq!(recovered["images"], a["structuredContent"]["images"]);
    assert_eq!(recovered["images"][0]["path"], saved["saved_path"]);
    assert_eq!(recovered["images"][0]["storage"], "saved");
    let mut different = saved;
    different["saved_path"] = json!(dir.path().join("other.png"));
    assert_eq!(
        confirm(&mcp, "single", json!([different])).await["isError"],
        true
    );
}
#[tokio::test]
async fn invalid_acknowledgements_delete_nothing_and_batches_allow_subsets() {
    let (dir, mcp) = setup().await;
    let result = job(&mcp, "batch", 2, true);
    let other = job(&mcp, "other", 1, false);
    let a = &result["items"][0]["images"][0];
    let b = &result["items"][1]["images"][0];
    assert_eq!(
        Path::new(a["path"].as_str().unwrap()).parent(),
        Path::new(b["path"].as_str().unwrap()).parent()
    );
    let first = copy_and_verify(a, &dir.path().join("a.png"));
    let second = copy_and_verify(b, &dir.path().join("b.png"));
    let mut variants = vec![json!({})];
    for (key, value) in [
        ("bytes", json!(0)),
        ("sha256", json!("0".repeat(64))),
        ("image_id", other["images"][0]["image_id"].clone()),
        ("saved_path", a["path"].clone()),
        ("saved_path", json!("relative.png")),
        ("saved_path", json!("/workspace/../image.png")),
        ("saved_path", first["saved_path"].clone()),
        ("unexpected", json!(true)),
    ] {
        let mut wrong = second.clone();
        wrong[key] = value;
        variants.push(wrong);
    }
    variants.push(first.clone());
    for wrong in variants {
        assert_eq!(
            confirm(&mcp, "batch", json!([first, wrong])).await["isError"],
            true
        );
        assert!(Path::new(a["path"].as_str().unwrap()).exists());
        assert!(Path::new(b["path"].as_str().unwrap()).exists());
    }
    assert_eq!(confirm(&mcp, "batch", json!([])).await["isError"], true);
    assert_eq!(
        confirm(&mcp, "unknown", json!([first])).await["isError"],
        true
    );
    mcp.inner.jobs.lock().unwrap()["batch"]
        .result
        .send_modify(|v| v["status"] = json!("generating"));
    assert_eq!(
        confirm(&mcp, "batch", json!([first])).await["isError"],
        true
    );
    mcp.inner.jobs.lock().unwrap()["batch"]
        .result
        .send_modify(|v| {
            v["status"] = json!("partial");
            v["items"]
                .as_array_mut()
                .unwrap()
                .push(json!({"index":2,"status":"failed","images":[]}));
        });
    assert_eq!(
        confirm(&mcp, "batch", json!([first])).await["isError"],
        false
    );
    assert!(!Path::new(a["path"].as_str().unwrap()).exists());
    assert!(Path::new(b["path"].as_str().unwrap()).exists());
    assert_eq!(
        confirm(&mcp, "batch", json!([first, second])).await["isError"],
        false
    );
    assert!(
        !Path::new(b["path"].as_str().unwrap())
            .parent()
            .unwrap()
            .exists()
    );
    assert!(Path::new(other["images"][0]["path"].as_str().unwrap()).exists());
}
#[tokio::test]
async fn cleanup_failure_is_retryable_and_never_follows_directory_symlinks() {
    let (dir, mcp) = setup().await;
    let result = job(&mcp, "safe", 1, false);
    let image = &result["images"][0];
    let path = PathBuf::from(image["path"].as_str().unwrap());
    let directory = path.parent().unwrap();
    let saved = copy_and_verify(image, &dir.path().join("saved.png"));
    let moved = dir.path().join("moved");
    std::fs::rename(directory, &moved).unwrap();
    std::os::unix::fs::symlink(&moved, directory).unwrap();
    assert_eq!(confirm(&mcp, "safe", json!([saved])).await["isError"], true);
    assert!(moved.join("0.png").exists());
    assert_eq!(
        mcp.get("safe", false).await.unwrap()["images"][0]["storage"],
        "temporary"
    );
    std::fs::remove_file(directory).unwrap();
    std::fs::rename(&moved, directory).unwrap();
    // A failed unlink must retain metadata and permit retry, without recursive cleanup.
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert_eq!(confirm(&mcp, "safe", json!([saved])).await["isError"], true);
    assert!(path.is_dir());
    std::fs::remove_dir(&path).unwrap();
    assert_eq!(
        confirm(&mcp, "safe", json!([saved])).await["isError"],
        false
    );
    assert!(dir.path().join("saved.png").exists());
}
#[tokio::test]
async fn partial_cleanup_failure_retains_confirmed_progress() {
    let (dir, mcp) = setup().await;
    let result = job(&mcp, "partial-cleanup", 2, true);
    let a = copy_and_verify(&result["items"][0]["images"][0], &dir.path().join("a.png"));
    let b = copy_and_verify(&result["items"][1]["images"][0], &dir.path().join("b.png"));
    let path = PathBuf::from(result["items"][1]["images"][0]["path"].as_str().unwrap());
    std::fs::remove_file(&path).unwrap();
    std::fs::create_dir(&path).unwrap();
    assert_eq!(
        confirm(&mcp, "partial-cleanup", json!([a, b])).await["isError"],
        true
    );
    let state = mcp.get("partial-cleanup", false).await.unwrap();
    assert_eq!(state["items"][0]["images"][0]["storage"], "saved");
    assert_eq!(state["items"][1]["images"][0]["storage"], "temporary");
    std::fs::remove_dir(&path).unwrap();
    assert_eq!(
        confirm(&mcp, "partial-cleanup", json!([a, b])).await["isError"],
        false
    );
}
