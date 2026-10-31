//! Synthetic projects, Draft requests and readers of the messages the local
//! provider receives, shared by the authoring tests.
use super::*;
fn colored(rgb: [u8; 3]) -> Vec<u8> {
    let mut out = Cursor::new(vec![]);
    image::RgbImage::from_pixel(16, 16, image::Rgb(rgb))
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}
pub(crate) async fn attach(h: &Harness, project: &str, rgb: [u8; 3]) -> String {
    let (code, uploaded) = raw(
        h,
        "POST",
        "/v1/video/assets",
        Body::from(colored(rgb)),
        "image/png",
        true,
    )
    .await;
    assert_eq!(code, 201, "{uploaded}");
    let (_, current) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    let (code, attached) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":current["project"]["project_revision"],"asset_id":uploaded["id"]})),
    )
    .await;
    assert_eq!(code, 200, "{attached}");
    attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .find(|k| current["project"]["assets"].get(k.as_str()).is_none())
        .unwrap()
        .clone()
}
pub(crate) async fn provider(h: &Harness, url: &str) {
    let profile = json!({"profile":{"id":"proxy","name":"Own fixture","base_url":url,"model":"own-story-fixture","system_prompt":"unused by video","max_tokens":1024,"temperature":null,"parameters":{},"has_api_key":false},"api_key":"own-fixture-key"});
    assert_eq!(
        call(h, "PUT", "/v1/enhancement/profiles/proxy", Some(profile))
            .await
            .0,
        200
    );
}
/// A project with four story images and two cast members; the keeper has an
/// enabled identity image.
pub(crate) async fn grounded(h: &Harness) -> (String, Value, Vec<String>) {
    let (code, d) = call(h, "POST", "/v1/video/projects", Some(create())).await;
    assert_eq!(code, 201, "{d}");
    let project = d["project"]["id"].as_str().unwrap().to_owned();
    let mut assets = vec![];
    for rgb in [
        [40, 90, 160],
        [200, 60, 30],
        [20, 160, 70],
        [230, 220, 40],
        [90, 20, 120],
    ] {
        assets.push(attach(h, &project, rgb).await);
    }
    let reference = |id: &str, alias: &str, images: Value| json!({"id":id,"name":alias,"alias":alias,"kind":"Character","description":format!("Own fictional {alias}"),"enabled":true,"strength":1.,"retention":"","images":images,"voice":{"language":"Swedish","description":"calm","assets":[]},"origin":null});
    let keeper = json!([{"id":"own_keeper_face","asset":assets[4],"enabled":true,"preview":false}]);
    let (_, current) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    let (code, edited) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":current["project"]["project_revision"],"commands":[{"operation":"set_creative","creative":{"references":[reference("keeper_id","keeper",keeper),reference("visitor_id","visitor",json!([]))],"sections":{}}}]})),
    )
    .await;
    assert_eq!(code, 200, "{edited}");
    let images = json!([
        {"id":"own_setting","asset":assets[0],"label":"Own lighthouse","usage":"setting_reference"},
        {"id":"own_start","asset":assets[1],"label":"Own door closed","usage":"start_frame"},
        {"id":"own_end","asset":assets[2],"label":"Own door open","usage":"end_frame"},
        {"id":"own_moment","asset":assets[3],"label":"Own beacon lit","usage":"required_moment"}
    ]);
    (project, images, assets)
}
pub(crate) async fn story_request(
    h: &Harness,
    project: &str,
    provider: &str,
    profile: &str,
) -> Value {
    let (_, d) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],"provider_profile_id":"proxy","provider":provider,"prompt_profile_id":profile,"operation":"outline","brief":"Own synthetic lighthouse story","instruction":"Own extra instruction","duration_seconds":10,"destination":"append"})
}
pub(crate) async fn settle(h: &Harness, id: &str, phase: &str) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let (code, d) = call(h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
            assert_eq!(code, 200, "{d}");
            if d["phase"] == phase {
                return d;
            }
            assert_ne!(d["phase"], "failed", "{d}");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
/// Outline, then every Scene, through the provider.
pub(crate) async fn provider_story(h: &Harness, project: &str, start: Value) -> Value {
    let (code, d) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap().to_owned();
    settle(h, &id, "outline").await;
    let (code, expanded) = call(
        h,
        "POST",
        &format!("/v1/video/authoring/{id}/expand"),
        Some(json!({"schema_version":1,"start_index":0})),
    )
    .await;
    assert_eq!(code, 200, "{expanded}");
    settle(h, &id, "review").await
}
pub(crate) fn user_text(body: &Value) -> String {
    let content = &body["messages"][1]["content"];
    content
        .as_str()
        .map(str::to_owned)
        .or_else(|| {
            content.as_array().map(|parts| {
                parts
                    .iter()
                    .filter_map(|p| p["text"].as_str())
                    .collect::<String>()
            })
        })
        .unwrap()
}
pub(crate) fn sent_images(body: &Value) -> Vec<String> {
    body["messages"][1]["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| p["image_url"]["url"].as_str())
        .map(|u| u.split_once(',').unwrap().1.to_owned())
        .collect()
}
