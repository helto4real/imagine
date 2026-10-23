//! Worker A route contracts; in-process Router only, no live server or inference.
use super::*;
use crate::destination::task_fixtures;
fn own_root() -> tempfile::TempDir {
    tempfile::tempdir_in(task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES")).unwrap()
}
#[tokio::test]
async fn media_health_http_relink_preview_is_nonmutating_and_apply_keeps_runtime_identity() {
    let root = own_root();
    let h = setup(root.path(), "health-relink").await;
    let (status, project) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    assert_eq!(status, 201);
    let id = project["project"]["id"].as_str().unwrap();
    let base = format!("/v1/video/projects/{id}");
    let (_, original) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(png()),
        "image/png",
        true,
    )
    .await;
    let (status, attached) = call(
        &h,
        "POST",
        &format!("{base}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"asset_id":original["id"]})),
    )
    .await;
    assert_eq!(status, 200);
    let target = attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .next()
        .unwrap()
        .clone();
    let mut bytes = Cursor::new(vec![]);
    image::RgbImage::from_pixel(16, 16, image::Rgb([180, 30, 20]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let (_, replacement) = raw(
        &h,
        "POST",
        "/v1/video/assets",
        Body::from(bytes.into_inner()),
        "image/png",
        true,
    )
    .await;
    let request = json!({"schema_version":1,"expected_project_revision":1,"asset_id":target,"expected_asset_revision":1,"replacement_asset_id":replacement["id"]});
    let (status, preview) = call(
        &h,
        "POST",
        &format!("{base}/media/relink-preview"),
        Some(request.clone()),
    )
    .await;
    assert_eq!(status, 200, "{preview}");
    assert_eq!(preview["impacts"][0]["kind"], "asset");
    let (_, unchanged) = call(&h, "GET", &base, None).await;
    assert_eq!(unchanged["project"]["project_revision"], 1);
    assert_eq!(unchanged["runtime_assets"][&target], original["id"]);
    let (status, health) = call(
        &h,
        "POST",
        &format!("{base}/media/health"),
        Some(json!({"schema_version":1,"expected_project_revision":1})),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(health["entries"][0]["status"], "ok");
    let (status, applied) = call(
        &h,
        "POST",
        &format!("{base}/media/relink"),
        Some(request.clone()),
    )
    .await;
    assert_eq!(status, 200, "{applied}");
    assert_eq!(
        applied["snapshot"]["project"]["assets"][&target]["revision"],
        2
    );
    assert_eq!(
        applied["snapshot"]["runtime_assets"][&target],
        replacement["id"]
    );
    assert_eq!(
        applied["snapshot"]["resources"].as_array().unwrap().len(),
        2
    );
    let (status, error) = call(&h, "POST", &format!("{base}/media/relink"), Some(request)).await;
    assert_eq!(status, 409);
    assert_eq!(error["snapshot"]["project"]["project_revision"], 2);
    let (status, _) = raw(
        &h,
        "POST",
        &format!("{base}/media/health"),
        Body::from(json!({"schema_version":1,"expected_project_revision":2}).to_string()),
        "application/json",
        false,
    )
    .await;
    assert_ne!(status, 200);
    stop(h).await;
}
#[tokio::test]
async fn reference_cache_http_unavailable_and_connected_accounting_are_distinct() {
    use imagine_server::video::reference_cache::{Cache, Control};
    use std::sync::Mutex;
    let root = own_root();
    let h = setup(root.path(), "reference-cache-contract").await;
    let (status, value) = call(&h, "GET", "/v1/video/reference-cache", None).await;
    assert_eq!(status, 200);
    assert_eq!(value["available"], false);
    let (status, _) = call(
        &h,
        "POST",
        "/v1/video/reference-cache/clear",
        Some(json!({})),
    )
    .await;
    assert_eq!(status, 422);
    let owner = Arc::new(Mutex::new(Cache::<u8>::new(1024)));
    {
        let mut c = owner.lock().unwrap();
        let t = c.begin("own-synthetic-encoding".into()).unwrap();
        assert!(c.finish(t, Arc::new(7), 128));
    }
    h.control.video.lock().await.reference_cache = Some(owner.clone());
    let (_, status) = call(&h, "GET", "/v1/video/reference-cache", None).await;
    assert_eq!(status["available"], true);
    assert_eq!(status["used_bytes"], 128);
    let (code, status) = call(
        &h,
        "POST",
        "/v1/video/reference-cache",
        Some(json!({"schema_version":1,"limit_bytes":64,"enabled":true})),
    )
    .await;
    assert_eq!(code, 200);
    assert_eq!(status["used_bytes"], 0);
    let (code, status) = call(
        &h,
        "POST",
        "/v1/video/reference-cache/clear",
        Some(json!({})),
    )
    .await;
    assert_eq!(code, 200);
    assert_eq!(status["entries"], 0);
    assert_eq!(owner.status().unwrap().used_bytes, 0);
    let (code, _) = call(
        &h,
        "POST",
        "/v1/video/reference-cache/clear",
        Some(json!({"path":"forbidden"})),
    )
    .await;
    assert_eq!(code, 422);
    stop(h).await;
}
