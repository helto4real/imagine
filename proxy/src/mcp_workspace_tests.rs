//! Workspace confinement in the proxy, against a synthetic Imagine.
use super::*;
use axum::extract::Path as UrlPath;
use axum::response::IntoResponse;
use tower::ServiceExt;

const KEY_A: &str = "imk-aaaaaaaaaaaaaaaaaaaaaaaaaa";
const KEY_B: &str = "imk-bbbbbbbbbbbbbbbbbbbbbbbbbb";

struct Fixture {
    _dir: tempfile::TempDir,
    mcp: Mcp,
    /// Workspace A's folder, as Imagine would report it.
    folder: PathBuf,
    image: Vec<u8>,
}
fn png() -> Vec<u8> {
    let mut out = std::io::Cursor::new(vec![]);
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        40,
        20,
        image::Rgb([5, 50, 150]),
    ))
    .write_to(&mut out, image::ImageFormat::Png)
    .unwrap();
    out.into_inner()
}
async fn setup() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("own-mcp-root");
    let folder = root.join("2026-10-25/1432-own-a");
    let other = root.join("2026-10-25/1433-own-b");
    for path in [
        folder.join("uploads"),
        folder.join("images"),
        other.join("uploads"),
    ] {
        std::fs::create_dir_all(path).unwrap();
    }
    let image = png();
    std::fs::write(folder.join("uploads/ref.png"), &image).unwrap();
    std::fs::write(other.join("uploads/secret.png"), &image).unwrap();
    let proof = test_destination(dir.path());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (a, b) = (folder.clone(), other.clone());
    let entry = move |workspace: &str, file: &str| -> Option<Value> {
        let (id, path) = match (workspace, file) {
            ("ws_a", "fref") => ("fref", a.join("uploads/ref.png")),
            ("ws_b", "fsecret") => ("fsecret", b.join("uploads/secret.png")),
            _ => return None,
        };
        let id = crate::workspace_ids::file_id(workspace, id);
        Some(
            json!({"file_id":id,"name":path.file_name().unwrap().to_str(),"kind":"image","origin":"upload",
            "mime_type":"image/png","bytes":std::fs::metadata(&path).unwrap().len(),"sha256":"0".repeat(64),
            "created_ms":1,"relative":"uploads/x","path":path}),
        )
    };
    let (fa, fb) = (folder.clone(), other.clone());
    let lookup = std::sync::Arc::new(entry);
    let (list_lookup, info_lookup, bytes_lookup) = (lookup.clone(), lookup.clone(), lookup);
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
        )
        .route(
            "/v1/video/projects/private",
            axum::routing::get(|| async { axum::Json(json!({"projects":[]})) }),
        )
        .route(
            "/v1/workspaces/resolve",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let (fa, fb) = (fa.clone(), fb.clone());
                async move {
                    match body["key"].as_str() {
                        Some(KEY_A) => (
                            StatusCode::OK,
                            axum::Json(json!({"workspace_id":"ws_a","path":fa})),
                        ),
                        Some(KEY_B) => (
                            StatusCode::OK,
                            axum::Json(json!({"workspace_id":"ws_b","path":fb})),
                        ),
                        _ => (
                            StatusCode::NOT_FOUND,
                            axum::Json(json!({"error":{"code":"workspace_unknown"}})),
                        ),
                    }
                }
            }),
        )
        .route(
            "/v1/workspaces/{id}/files",
            axum::routing::get(move |UrlPath(id): UrlPath<String>| {
                let entries = list_lookup.clone();
                async move {
                    let file = if id == "ws_a" { "fref" } else { "fsecret" };
                    axum::Json(json!({"files":[entries(&id, file)]}))
                }
            }),
        )
        .route(
            "/v1/workspaces/{id}/files/{file}/info",
            axum::routing::get(move |UrlPath((id, file)): UrlPath<(String, String)>| {
                let entries = info_lookup.clone();
                async move {
                    let short = ["fref", "fsecret"]
                        .into_iter()
                        .find(|s| crate::workspace_ids::file_id(&id, s) == file);
                    match short.and_then(|s| entries(&id, s)) {
                        Some(entry) => (StatusCode::OK, axum::Json(entry)),
                        None => (
                            StatusCode::NOT_FOUND,
                            axum::Json(json!({"error":{"code":"file_not_found"}})),
                        ),
                    }
                }
            }),
        )
        .route(
            "/v1/workspaces/{id}/files/{file}",
            axum::routing::get(move |UrlPath((id, file)): UrlPath<(String, String)>| {
                let entries = bytes_lookup.clone();
                async move {
                    let short = ["fref", "fsecret"]
                        .into_iter()
                        .find(|s| crate::workspace_ids::file_id(&id, s) == file);
                    match short.and_then(|s| entries(&id, s)) {
                        Some(entry) => (
                            StatusCode::OK,
                            std::fs::read(entry["path"].as_str().unwrap()).unwrap(),
                        )
                            .into_response(),
                        None => StatusCode::NOT_FOUND.into_response(),
                    }
                }
            }),
        );
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mcp = Mcp::new(Config {
        coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
        imagine_url: url,
        mcp_port: Some(8190),
        ..Default::default()
    })
    .await
    .unwrap();
    mcp.refresh_session().await.unwrap();
    Fixture {
        _dir: dir,
        mcp,
        folder,
        image,
    }
}
fn structured(result: Value) -> Value {
    result["structuredContent"].clone()
}

#[tokio::test]
async fn tools_need_a_known_key_and_see_only_their_own_files() {
    let f = setup().await;
    // No key, or an unknown one, reaches nothing and says nothing about others.
    let missing = structured(f.mcp.call("workspace_files", &json!({})).await);
    assert_eq!(missing["error"]["category"], "validation");
    let unknown = structured(
        f.mcp
            .call(
                "workspace_files",
                &json!({"workspace_key":"imk-cccccccccccccccccccccccccc"}),
            )
            .await,
    );
    assert_eq!(unknown["error"]["category"], "not_found");
    // The guide needs no key; opening or creating a workspace returns it too.
    let guide = structured(f.mcp.call("imagine_guide", &json!({})).await);
    assert_eq!(guide["guide"], GUIDE);
    let opened = structured(
        f.mcp
            .call("workspace_open", &json!({"workspace_key":KEY_A}))
            .await,
    );
    assert_eq!(opened["guide"], GUIDE, "{opened}");
    let listed = structured(
        f.mcp
            .call("workspace_files", &json!({"workspace_key":KEY_A}))
            .await,
    );
    let files = listed["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0]["name"], "ref.png");
    let text = listed.to_string();
    assert!(!text.contains("own-mcp-root"), "no host path: {text}");
    assert!(
        files[0]["download_url"]
            .as_str()
            .unwrap()
            .starts_with("http://127.0.0.1:8190/v1/workspace/files/f")
    );
    // Workspace B's file ID is not a file of workspace A.
    let foreign = crate::workspace_ids::file_id("ws_b", "fsecret");
    let view = structured(
        f.mcp
            .call(
                "workspace_view",
                &json!({"workspace_key":KEY_A,"file_id":foreign}),
            )
            .await,
    );
    assert_eq!(view["error"]["category"], "not_found");
    // Its own image is shown as an image block, without a path.
    let own = crate::workspace_ids::file_id("ws_a", "fref");
    let shown = f
        .mcp
        .call(
            "workspace_view",
            &json!({"workspace_key":KEY_A,"file_id":own,"max_side":64}),
        )
        .await;
    assert_eq!(shown["content"][1]["type"], "image", "{shown}");
    assert!(!shown.to_string().contains("own-mcp-root"));
}

#[tokio::test]
async fn image_jobs_belong_to_the_workspace_that_started_them() {
    let f = setup().await;
    let (result, _) = watch::channel(json!({"job_id":"own-job","status":"succeeded","images":[]}));
    f.mcp.inner.jobs.lock().unwrap().insert(
        "own-job".into(),
        Job {
            fingerprint: vec![],
            result,
            task: None,
            count: 1,
            workspace: "ws_a".into(),
        },
    );
    let mine = structured(
        f.mcp
            .call(
                "get_job",
                &json!({"workspace_key":KEY_A,"job_id":"own-job","wait":false}),
            )
            .await,
    );
    assert_eq!(mine["status"], "succeeded", "{mine}");
    let theirs = structured(
        f.mcp
            .call(
                "get_job",
                &json!({"workspace_key":KEY_B,"job_id":"own-job","wait":false}),
            )
            .await,
    );
    assert_eq!(theirs["error"]["category"], "not_found");
}

#[test]
fn results_carry_file_references_and_never_host_paths() {
    let dir = tempfile::tempdir().unwrap();
    let folder = dir.path().join("2026-10-25/1432-own");
    std::fs::create_dir_all(folder.join("videos/chase/outputs/videos")).unwrap();
    let video = folder.join("videos/chase/outputs/videos/final.mp4");
    std::fs::write(&video, b"own").unwrap();
    let workspace = Bound {
        id: "ws_own".into(),
        folder: folder.clone(),
    };
    let image = folder.join("images/0001.png");
    std::fs::create_dir_all(image.parent().unwrap()).unwrap();
    std::fs::write(&image, b"own png").unwrap();
    let result = references::workspace_result(
        &workspace,
        &json!({"saved_path":image,"persistent":true,"mime_type":"image/png","bytes":7,"sha256":"0".repeat(64)}),
        |id| format!("http://127.0.0.1:8190/v1/workspace/files/{id}"),
    )
    .unwrap();
    assert_eq!(
        result[0]["file_id"],
        crate::workspace_ids::file_id("ws_own", "images/0001.png")
    );
    assert!(result[0].get("saved_path").is_none());
    // Outside the workspace is refused.
    assert!(
        references::workspace_result(
            &workspace,
            &json!({"saved_path":dir.path().join("elsewhere.png"),"persistent":true,"bytes":1,"sha256":"0"}),
            |id| id.into(),
        )
        .is_err()
    );
    // A generic result: workspace outputs become references, other paths go.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mcp = runtime.block_on(async {
        let proof = test_destination(dir.path());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let router = Router::new().route(
            "/v1/media-destination",
            axum::routing::get(move || {
                let proof = proof.clone();
                async move { axum::Json(proof) }
            }),
        );
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Mcp::new(Config {
            coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
            imagine_url: url,
            mcp_port: Some(8190),
            ..Default::default()
        })
        .await
        .unwrap()
    });
    let mut value = json!({"job":{"output":{"saved_path":video,"format":"mp4"}},
        "persistence":{"state":"saved","location":folder},
        "note":"/run/media/veracrypt1/mcp-root/2026-10-25/1432-own/project.json",
        "prompt":"a forest at dusk"});
    mcp.scrub(&mut value, Some(&workspace));
    assert_eq!(
        value["job"]["output"]["file_id"],
        crate::workspace_ids::file_id("ws_own", "videos/chase/outputs/videos/final.mp4")
    );
    assert_eq!(value["job"]["output"]["name"], "final.mp4");
    assert!(value["persistence"].get("location").is_none());
    assert_eq!(value["note"], "[hidden]");
    assert_eq!(value["prompt"], "a forest at dusk");
    assert!(!value.to_string().contains(folder.to_str().unwrap()));
}

#[tokio::test]
async fn http_file_routes_take_the_key_as_bearer_and_check_the_host() {
    let f = setup().await;
    let own = crate::workspace_ids::file_id("ws_a", "fref");
    let request = |key: Option<&str>, host: &str, file: &str| {
        let mut builder = Request::builder()
            .uri(format!("/v1/workspace/files/{file}"))
            .header("host", host);
        if let Some(key) = key {
            builder = builder.header("authorization", format!("Bearer {key}"));
        }
        builder.body(Body::empty()).unwrap()
    };
    let status = |response: Response<Body>| response.status().as_u16();
    let router = f.mcp.router();
    assert_eq!(
        status(
            router
                .clone()
                .oneshot(request(None, "127.0.0.1:8190", &own))
                .await
                .unwrap()
        ),
        401
    );
    assert_eq!(
        status(
            router
                .clone()
                .oneshot(request(Some(KEY_A), "evil.example:8190", &own))
                .await
                .unwrap()
        ),
        421
    );
    assert_eq!(
        status(
            router
                .clone()
                .oneshot(request(Some(KEY_B), "127.0.0.1:8190", &own))
                .await
                .unwrap()
        ),
        404
    );
    let response = router
        .clone()
        .oneshot(request(Some(KEY_A), "127.0.0.1:8190", &own))
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(body.as_ref(), f.image.as_slice());
    assert!(f.folder.join("uploads/ref.png").is_file());
}

#[tokio::test]
async fn the_interactive_video_prompt_is_listed_and_fills_in_the_idea() {
    let f = setup().await;
    let rpc = |method: &str, params: Value| {
        dispatch_rpc(
            f.mcp.clone(),
            json!({"jsonrpc":"2.0","id":1,"method":method,"params":params}),
            false,
        )
    };
    let listed = rpc("prompts/list", json!({})).await;
    assert_eq!(listed["result"]["prompts"][0]["name"], "video_interactive");
    assert_eq!(
        listed["result"]["prompts"][0]["arguments"][0]["required"],
        false
    );
    let got = rpc(
        "prompts/get",
        json!({"name":"video_interactive","arguments":{"idea":"a lighthouse keeper's last night"}}),
    )
    .await;
    let text = got["result"]["messages"][0]["content"]["text"]
        .as_str()
        .unwrap();
    assert!(text.contains("Call imagine_guide first"), "{text}");
    assert!(
        text.ends_with("My idea: a lighthouse keeper's last night"),
        "{text}"
    );
    let empty = rpc("prompts/get", json!({"name":"video_interactive"})).await;
    assert!(
        empty["result"]["messages"][0]["content"]["text"]
            .as_str()
            .unwrap()
            .contains("start with the settings")
    );
    for params in [
        json!({"name":"other"}),
        json!({"name":"video_interactive","arguments":{"idea":"x".repeat(4001)}}),
    ] {
        assert_eq!(rpc("prompts/get", params).await["error"]["code"], -32602);
    }
}
