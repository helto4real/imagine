use super::*;
use std::path::Path;

async fn setup() -> (tempfile::TempDir, Mcp) {
    let dir = tempfile::tempdir().unwrap();
    let published = dir.path().join("projects/own-workspace/outputs/images");
    std::fs::create_dir_all(&published).unwrap();
    let proof = test_destination(&published);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new()
        .route(
            "/v1/media-destination",
            axum::routing::get(move || {
                let proof = proof.clone();
                async move { axum::Json(proof) }
            }),
        )
        .route(
            "/v1/session",
            axum::routing::get(|| async { axum::Json(json!({"id":"synthetic-session"})) }),
        );
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mcp = Mcp::new(Config {
        coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
        imagine_url: url,
        // An obsolete cache outside the new output proof is harmless and unused.
        mcp_output_dir: Some(dir.path().join("old-cache").to_string_lossy().into()),
        mcp_input_dir: dir.path().to_string_lossy().into(),
        ..Default::default()
    })
    .await
    .unwrap();
    mcp.refresh_session().await.unwrap();
    assert!(!dir.path().join("old-cache").exists());
    (dir, mcp)
}
fn job(mcp: &Mcp, id: &str, count: usize, batch: bool) -> Value {
    let items=(0..count).map(|index| {
        let bytes=format!("own procedural published bytes {id} {index}");
        let path=mcp.inner.storage.persistent_root().join(format!("{id}-{index}.png"));
        std::fs::write(&path,bytes.as_bytes()).unwrap();
        let asset=json!({"saved_path":path,"persistent":true,"mime_type":"image/png","bytes":bytes.len(),"sha256":references::hash(bytes.as_bytes())});
        json!({"index":index,"status":"succeeded","images":references::persistent_result(&mcp.inner.storage,&asset).unwrap(),"error":null})
    }).collect::<Vec<_>>();
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
            fingerprint: vec![],
            result,
            task: None,
            count,
        },
    );
    snapshot
}
fn acknowledgement(image: &Value) -> Value {
    json!({"image_id":image["image_id"],"saved_path":image["saved_path"],"bytes":image["bytes"],"sha256":image["sha256"]})
}
async fn confirm(mcp: &Mcp, id: &str, images: Value) -> Value {
    dispatch_rpc(mcp.clone(),json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"confirm_saved","arguments":{"job_id":id,"images":images}}}),false).await["result"].clone()
}
#[tokio::test]
async fn persistent_confirmation_is_idempotent_and_never_copies_or_deletes() {
    let (dir, mcp) = setup().await;
    let snapshot = job(&mcp, "single", 1, false);
    let image = &snapshot["images"][0];
    let saved = acknowledgement(image);
    let path = Path::new(image["path"].as_str().unwrap());
    let before = std::fs::read(path).unwrap();
    assert_eq!(image["storage"], "saved");
    let tools = dispatch_rpc(
        mcp.clone(),
        json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
        false,
    )
    .await;
    assert!(
        tools["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "confirm_saved"
                && tool["annotations"]["destructiveHint"] == false)
    );
    let confirmed = confirm(&mcp, "single", json!([saved])).await;
    assert_eq!(confirmed["isError"], false, "{confirmed}");
    assert_eq!(confirm(&mcp, "single", json!([saved])).await, confirmed);
    assert_eq!(mcp.get("single", true).await.unwrap(), snapshot);
    assert_eq!(std::fs::read(path).unwrap(), before);
    assert_eq!(
        std::fs::read_dir(mcp.inner.storage.persistent_root())
            .unwrap()
            .count(),
        1
    );
    assert!(!dir.path().join("old-cache").exists());
    mcp.close().await;
    assert_eq!(std::fs::read(path).unwrap(), before);
}
#[tokio::test]
async fn invalid_acknowledgements_leave_outputs_and_batch_metadata_untouched() {
    let (_dir, mcp) = setup().await;
    let snapshot = job(&mcp, "batch", 2, true);
    let other = job(&mcp, "other", 1, false);
    let first = acknowledgement(&snapshot["items"][0]["images"][0]);
    let second = acknowledgement(&snapshot["items"][1]["images"][0]);
    for (key, value) in [
        ("bytes", json!(0)),
        ("sha256", json!("0".repeat(64))),
        ("image_id", other["images"][0]["image_id"].clone()),
        ("saved_path", first["saved_path"].clone()),
        ("saved_path", json!("relative.png")),
        ("unexpected", json!(true)),
    ] {
        let mut wrong = second.clone();
        wrong[key] = value;
        assert_eq!(
            confirm(&mcp, "batch", json!([first, wrong])).await["isError"],
            true
        );
        assert_eq!(mcp.get("batch", false).await.unwrap(), snapshot);
    }
    assert_eq!(confirm(&mcp, "batch", json!([])).await["isError"], true);
    assert_eq!(
        confirm(&mcp, "unknown", json!([first])).await["isError"],
        true
    );
    assert_eq!(
        confirm(&mcp, "batch", json!([first])).await["isError"],
        false
    );
    assert_eq!(
        confirm(&mcp, "batch", json!([first, second])).await["isError"],
        false
    );
    for item in snapshot["items"].as_array().unwrap() {
        assert!(Path::new(item["images"][0]["path"].as_str().unwrap()).exists());
    }
    mcp.close().await;
}
#[tokio::test]
async fn session_reset_discards_only_ram_and_leaves_published_and_historical_files() {
    let (dir, mcp) = setup().await;
    let historical = dir.path().join("own-historical-cache");
    std::fs::create_dir(&historical).unwrap();
    std::fs::write(historical.join("own-old.bin"), b"own old fixture").unwrap();
    let snapshot = job(&mcp, "retained", 1, false);
    let path = Path::new(snapshot["images"][0]["path"].as_str().unwrap());
    mcp.clear_session().await.unwrap();
    assert!(mcp.inner.jobs.lock().unwrap().is_empty());
    assert!(mcp.inner.references.lock().unwrap().is_empty());
    assert!(path.exists());
    assert_eq!(
        std::fs::read(historical.join("own-old.bin")).unwrap(),
        b"own old fixture"
    );
    assert!(!dir.path().join("old-cache").exists());
    mcp.close().await;
}
