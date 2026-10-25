//! Agent workspaces under an own synthetic MCP root.
use super::storage_links::{private_volume, setup_private_mcp};
use super::*;
use std::path::PathBuf;

async fn bytes(
    h: &Harness,
    path: &str,
    range: Option<&str>,
) -> (u16, Vec<u8>, axum::http::HeaderMap) {
    use tower::ServiceExt;
    let mut request = Request::builder().method("GET").uri(path);
    if let Some(range) = range {
        request = request.header("Range", range);
    }
    let response = h
        .router
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status().as_u16();
    let headers = response.headers().clone();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, body.to_vec(), headers)
}
fn png() -> Vec<u8> {
    let mut out = std::io::Cursor::new(vec![]);
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(8, 6, image::Rgb([9, 99, 199])))
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

#[tokio::test]
async fn workspaces_are_dated_folders_reached_only_by_key_with_isolated_files() {
    let directory = tempfile::tempdir().unwrap();
    let (private, _) = private_volume(directory.path());
    let mcp = directory.path().join("own-volume/own-mcp-root");
    let (private, ignored) = private.with_project_roots(std::slice::from_ref(&mcp));
    assert!(ignored.is_empty());
    let h = setup_private_mcp(&private, Some(&mcp)).await;
    // Created with its key file, metadata and folders in today's date folder.
    let (status, created) = call(
        &h,
        "POST",
        "/v1/workspaces",
        Some(json!({"schema_version":1,"title":"Own Forest Chase"})),
    )
    .await;
    assert_eq!(status, 201, "{created}");
    let key = created["key"].as_str().unwrap().to_owned();
    let id = created["workspace_id"].as_str().unwrap().to_owned();
    let folder = PathBuf::from(created["path"].as_str().unwrap());
    let date = imagine_video_contract::project::local_date();
    let relative = folder.strip_prefix(&mcp).unwrap().to_owned();
    assert_eq!(
        relative.iter().next().unwrap().to_str().unwrap(),
        format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8])
    );
    assert!(relative.to_str().unwrap().contains("own-forest-chase"));
    let key_file = std::fs::read_to_string(folder.join("AGENT-KEY.txt")).unwrap();
    assert_eq!(key_file.lines().next().unwrap(), key);
    let meta = std::fs::read_to_string(folder.join("workspace.json")).unwrap();
    assert!(
        !meta.contains(&key),
        "the key itself is not in workspace.json"
    );
    for child in ["uploads", "images", "videos"] {
        assert!(folder.join(child).is_dir());
    }
    // Resolving needs the exact key.
    let (status, resolved) = call(
        &h,
        "POST",
        "/v1/workspaces/resolve",
        Some(json!({"schema_version":1,"key":key})),
    )
    .await;
    assert_eq!(
        (status, resolved["workspace_id"].as_str()),
        (200, Some(id.as_str()))
    );
    for wrong in ["imk-aaaaaaaaaaaaaaaaaaaaaaaaaa", "not-a-key"] {
        let (status, error) = call(
            &h,
            "POST",
            "/v1/workspaces/resolve",
            Some(json!({"schema_version":1,"key":wrong})),
        )
        .await;
        assert_eq!(
            (status, error["error"]["code"].as_str()),
            (404, Some("workspace_unknown"))
        );
    }
    // Upload: type by content, safe name, no overwrite.
    let image = png();
    let (status, uploaded) = raw(
        &h,
        "PUT",
        &format!("/v1/workspaces/{id}/files?name=../My%20Ref.jpeg"),
        Body::from(image.clone()),
        "application/octet-stream",
        false,
    )
    .await;
    assert_eq!(status, 201, "{uploaded}");
    assert_eq!(uploaded["name"], "my-ref.png");
    assert_eq!(
        (uploaded["kind"].as_str(), uploaded["origin"].as_str()),
        (Some("image"), Some("upload"))
    );
    assert!(folder.join("uploads/my-ref.png").is_file());
    let (status, second) = raw(
        &h,
        "PUT",
        &format!("/v1/workspaces/{id}/files?name=my-ref.png"),
        Body::from(image.clone()),
        "application/octet-stream",
        false,
    )
    .await;
    assert_eq!(status, 201, "{second}");
    assert_eq!(second["name"], "my-ref-2.png");
    let (status, refused) = raw(
        &h,
        "PUT",
        &format!("/v1/workspaces/{id}/files?name=run.sh"),
        Body::from("#!/bin/sh\necho own\n"),
        "application/octet-stream",
        false,
    )
    .await;
    assert_eq!(
        (status, refused["error"]["code"].as_str()),
        (415, Some("upload_type_unsupported"))
    );
    // Listing and ranged download by file ID.
    let file = uploaded["file_id"].as_str().unwrap();
    let (_, listed) = call(&h, "GET", &format!("/v1/workspaces/{id}/files"), None).await;
    assert_eq!(listed["files"].as_array().unwrap().len(), 2);
    assert!(
        listed["files"][0].get("path").is_none(),
        "listings carry no host paths"
    );
    let (status, body, headers) =
        bytes(&h, &format!("/v1/workspaces/{id}/files/{file}"), None).await;
    assert_eq!((status, body == image), (200, true));
    assert!(
        headers["content-disposition"]
            .to_str()
            .unwrap()
            .contains("my-ref.png")
    );
    let (status, body, _) = bytes(
        &h,
        &format!("/v1/workspaces/{id}/files/{file}"),
        Some("bytes=1-3"),
    )
    .await;
    assert_eq!((status, body.as_slice()), (206, &image[1..4]));
    // Another workspace cannot reach this one's files.
    let (_, other) = call(
        &h,
        "POST",
        "/v1/workspaces",
        Some(json!({"schema_version":1})),
    )
    .await;
    let other_id = other["workspace_id"].as_str().unwrap();
    let (status, _, _) = bytes(&h, &format!("/v1/workspaces/{other_id}/files/{file}"), None).await;
    assert_eq!(status, 404);
    // Image jobs get the workspace's image folder as their destination.
    let (status, proof) = call(&h, "GET", &format!("/v1/workspaces/{id}/destination"), None).await;
    assert_eq!(status, 200, "{proof}");
    assert_eq!(proof["config"]["root"], json!(folder.join("images")));
    // A video project in videos/<title>, listed with its workspace.
    let (status, project) = call(
        &h,
        "POST",
        &format!("/v1/workspaces/{id}/video-projects"),
        Some(json!({"schema_version":1,"title":"Own Chase"})),
    )
    .await;
    assert_eq!(status, 201, "{project}");
    assert_eq!(project["path"], json!(folder.join("videos/own-chase")));
    assert!(folder.join("videos/own-chase/project.json").is_file());
    let (_, listed) = call(&h, "GET", "/v1/video/projects/private", None).await;
    let entry = listed["projects"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["path"] == json!(folder.join("videos/own-chase")))
        .unwrap_or_else(|| panic!("{listed}"));
    assert_eq!(entry["workspace"]["workspace_id"], json!(id));
    assert_eq!(entry["source"], "agent");
    // Replacing the key revokes the old one; the UI reads the current key.
    let (_, replaced) = call(&h, "POST", &format!("/v1/workspaces/{id}/key"), None).await;
    let new_key = replaced["key"].as_str().unwrap();
    assert_ne!(new_key, key);
    let (status, _) = call(
        &h,
        "POST",
        "/v1/workspaces/resolve",
        Some(json!({"schema_version":1,"key":key})),
    )
    .await;
    assert_eq!(status, 404);
    let (_, current) = call(&h, "GET", &format!("/v1/workspaces/{id}/key"), None).await;
    assert_eq!(current["key"], new_key);
    // A restarted server finds workspaces by scanning.
    let reopened = imagine_server::workspaces::Workspaces::open(private.rooted_at(&mcp).unwrap());
    assert_eq!(reopened.resolve(new_key).unwrap().meta.id, id);
    stop(h).await;
}

#[tokio::test]
async fn new_ui_projects_live_in_their_creation_date_folder() {
    let directory = tempfile::tempdir().unwrap();
    let (private, _) = private_volume(directory.path());
    let h = setup_private_mcp(&private, None).await;
    let (status, created) = call(
        &h,
        "POST",
        "/v1/video/projects",
        Some(serde_json::to_value(imagine_server::video_defaults::new_project()).unwrap()),
    )
    .await;
    assert_eq!(status, 201, "{created}");
    let id = created["project"]["id"].as_str().unwrap();
    let date = imagine_video_contract::project::local_date();
    assert!(id.starts_with(&format!("project_{date}_")), "{id}");
    let folder = private
        .layout()
        .root()
        .join("projects")
        .join(format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8]))
        .join(id);
    assert!(folder.join("project.json").is_file(), "{folder:?}");
    let (_, listed) = call(&h, "GET", "/v1/video/projects/private", None).await;
    assert!(
        listed["projects"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["path"] == json!(folder)),
        "{listed}"
    );
    stop(h).await;
}
