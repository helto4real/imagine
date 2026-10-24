//! Golden provider messages for synthetic Drafts: the system prompt digests,
//! user messages and image counts the local provider receives.
use super::authoring_tests::fixture;
use super::*;
use sha2::Digest;
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/authoring-messages.json"
);
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
fn recorded(requests: &[Value], assets: &[String], label: &str) -> Vec<Value> {
    requests
        .iter()
        .enumerate()
        .map(|(i, body)| {
            let mut user = user_text(body);
            for (n, asset) in assets.iter().enumerate() {
                user = user.replace(asset.as_str(), &format!("<asset_{n}>"));
            }
            let system = body["messages"][0]["content"].as_str().unwrap();
            json!({"case":format!("{label}_{i}"),"system_sha256":format!("{:x}", sha2::Sha256::digest(system.as_bytes())),"images":sent_images(body).len(),"user":user})
        })
        .collect()
}
#[tokio::test]
async fn provider_messages_match_the_golden_record() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "golden-messages").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let mut record = vec![];
    // Images, cast and dialogue.
    let (project, images, assets) = grounded(&h).await;
    let mut start = story_request(&h, &project, "koboldcpp", "nsfw").await;
    start["cast"] = json!(["keeper_id", "visitor_id"]);
    start["images"] = images;
    start["add_dialogue"] = json!(true);
    provider_story(&h, &project, start).await;
    record.extend(recorded(
        &std::mem::take(&mut *state.requests.lock().unwrap()),
        &assets,
        "grounded",
    ));
    // No story images, continuing an existing Scene that opens on a frame.
    let (_, d) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let project = d["project"]["id"].as_str().unwrap().to_owned();
    let frame = attach(&h, &project, [10, 10, 10]).await;
    let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"description":"Own harbour at dusk","model":"project_default","settings":[],"sections":[{"id":"own_section","start":{"num":0,"den":1},"duration":{"num":2,"den":1},"prompt":"Own keeper waits"},{"id":"own_later","start":{"num":2,"den":1},"duration":{"num":2,"den":1},"prompt":"Own keeper turns"}],"bindings":[{"id":"own_binding","asset":frame,"role":"first_frame","enabled":true,"video":null}],"continuity":null,"accepted_take":null});
    let (code, added) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":1,"commands":[{"operation":"add_scene","scene":scene}]})),
    )
    .await;
    assert_eq!(code, 200, "{added}");
    let mut start = story_request(&h, &project, "koboldcpp", "normal").await;
    start["destination"] = json!("continue");
    start["instruction"] = json!("");
    provider_story(&h, &project, start).await;
    record.extend(recorded(
        &std::mem::take(&mut *state.requests.lock().unwrap()),
        std::slice::from_ref(&frame),
        "continued",
    ));
    // Enhancements of that Scene's fields and Sections.
    for (operation, field, section) in [
        ("enhance_scene", Some("description"), None),
        ("enhance_scene", Some("summary"), None),
        ("enhance_section", None, Some("own_section")),
        ("enhance_section", None, Some("own_later")),
        ("enhance_scene_all", None, None),
    ] {
        let mut payload = story_request(&h, &project, "koboldcpp", "normal").await;
        payload["operation"] = json!(operation);
        payload["scene_id"] = json!("own_scene");
        payload["scene_field"] = json!(field);
        payload["section_id"] = json!(section);
        let (code, d) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(payload),
        )
        .await;
        assert_eq!(code, 202, "{d}");
        settle(&h, d["id"].as_str().unwrap(), "review").await;
    }
    record.extend(recorded(
        &std::mem::take(&mut *state.requests.lock().unwrap()),
        std::slice::from_ref(&frame),
        "enhanced",
    ));
    let actual = serde_json::to_string_pretty(&record).unwrap() + "\n";
    if std::env::var_os("IMAGINE_UPDATE_AUTHORING_GOLDEN").is_some() {
        std::fs::write(GOLDEN, &actual).unwrap();
    }
    let expected = std::fs::read_to_string(GOLDEN).unwrap();
    let expected: Vec<Value> = serde_json::from_str(&expected).unwrap();
    assert_eq!(expected.len(), record.len());
    for (expected, actual) in expected.iter().zip(&record) {
        assert_eq!(expected, actual, "{}", expected["case"]);
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
