//! Director's answer text normalization, the opening appearance in the
//! expansions, and lossless PNG images to the language model.
use super::authoring_messages::{grounded, provider, provider_story, story_request, user_text};
use super::authoring_tests::fixture;
use super::*;
use base64::Engine;

fn context() -> Value {
    json!({"latest_moment":[{"role":"assistant","name":"Own Visitor","text":"@keeper, the lamp!"}],
        "cast_state":[{"reference_id":"keeper_id","current_appearance":{"exposure":"clothed","garments":["grey coat"]}}],
        "protected_facts":["The lamp is broken."]})
}
async fn settle(h: &Harness, id: &str, phase: &str) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let (_, d) = call(h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
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
async fn start(h: &Harness, project: &str, request: Value) -> String {
    let (code, d) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(request),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    d["id"].as_str().unwrap().to_owned()
}
/// A project image of `width` x `height`.
async fn attach_sized(h: &Harness, project: &str, width: u32, height: u32) -> String {
    let mut png = Cursor::new(vec![]);
    image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 251) as u8, (y % 241) as u8, 90])
    })
    .write_to(&mut png, image::ImageFormat::Png)
    .unwrap();
    let (code, uploaded) = raw(
        h,
        "POST",
        "/v1/video/assets",
        Body::from(png.into_inner()),
        "image/png",
        true,
    )
    .await;
    assert_eq!(code, 201, "{uploaded}");
    let (_, current) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    let (code, attached) = call(h, "POST", &format!("/v1/video/projects/{project}/assets"),
        Some(json!({"schema_version":1,"expected_project_revision":current["project"]["project_revision"],"asset_id":uploaded["id"]}))).await;
    assert_eq!(code, 200, "{attached}");
    attached["project"]["assets"]
        .as_object()
        .unwrap()
        .keys()
        .find(|k| current["project"]["assets"].get(k.as_str()).is_none())
        .unwrap()
        .clone()
}
/// Every image a provider request carries, decoded, with its declared type.
fn images(body: &Value) -> Vec<(String, image::DynamicImage)> {
    let decode = |data: &str| {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        assert_eq!(
            image::guess_format(&bytes).unwrap(),
            image::ImageFormat::Png
        );
        image::load_from_memory(&bytes).unwrap()
    };
    if let Some(images) = body["images"].as_array() {
        return images
            .iter()
            .map(|i| ("ollama".to_owned(), decode(i.as_str().unwrap())))
            .collect();
    }
    body["messages"][1]["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| p["image_url"]["url"].as_str())
        .map(|url| {
            let (head, data) = url.split_once(',').unwrap();
            (head.to_owned(), decode(data))
        })
        .collect()
}
#[tokio::test]
async fn fenced_answers_produce_the_same_drafts_as_plain_ones() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "fenced-answers").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let mut drafts = vec![];
    for fenced in [false, true] {
        state
            .fence_answers
            .store(fenced, std::sync::atomic::Ordering::SeqCst);
        let (project, images, _) = grounded(&h).await;
        let mut request = story_request(&h, &project, "koboldcpp", "nsfw").await;
        request["cast"] = json!(["keeper_id", "visitor_id"]);
        request["images"] = images;
        request["add_dialogue"] = json!(true);
        let ready = provider_story(&h, &project, request).await;
        let mut enhance = story_request(&h, &project, "koboldcpp", "normal").await;
        let frame = attach_sized(&h, &project, 16, 16).await;
        let (_, current) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
        let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"description":"Own harbour","model":"project_default","settings":[],"sections":[{"id":"own_section","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"prompt":"Own keeper waits"}],"bindings":[{"id":"own_binding","asset":frame,"role":"first_frame","enabled":true,"video":null}],"continuity":null,"accepted_take":null});
        let (code, d) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"),
            Some(json!({"schema_version":1,"expected_project_revision":current["project"]["project_revision"],"commands":[{"operation":"add_scene","scene":scene}]}))).await;
        assert_eq!(code, 200, "{d}");
        enhance["expected_project_revision"] = d["project"]["project_revision"].clone();
        enhance["operation"] = json!("enhance_section");
        enhance["scene_id"] = json!("own_scene");
        enhance["section_id"] = json!("own_section");
        let enhanced = settle(&h, &start(&h, &project, enhance.clone()).await, "review").await;
        let mut animate = enhance.clone();
        animate["operation"] = json!("animate");
        animate["section_id"] = Value::Null;
        animate["brief"] = json!("");
        animate["instruction"] = json!("");
        let animated = settle(&h, &start(&h, &project, animate).await, "review").await;
        drafts.push(json!({"inventory":ready["inventory"],"outline":ready["outline"],"scenes":ready["scenes"],
            "text":enhanced["text"],"animate":animated["animate"]}));
    }
    assert!(drafts[0]["inventory"].as_array().unwrap().len() == 4);
    assert!(
        drafts[0]["scenes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s.is_object())
    );
    assert!(drafts[0]["text"].is_string() && drafts[0]["animate"].is_object());
    assert_eq!(drafts[0], drafts[1]);
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn expansions_start_from_the_resolved_and_then_the_reviewed_opening_appearance() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "opening-expansion").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, _, _) = grounded(&h).await;
    let mut request = story_request(&h, &project, "koboldcpp", "normal").await;
    request["brief"] = json!("");
    request["cast"] = json!(["keeper_id"]);
    request["context"] = context();
    let ready = provider_story(&h, &project, request).await;
    let id = ready["id"].as_str().unwrap().to_owned();
    let grounding = |body: &Value| {
        let user = user_text(body);
        user.split("<grounding_context>\n")
            .nth(1)
            .unwrap()
            .split("\n</grounding_context>")
            .next()
            .unwrap()
            .to_owned()
    };
    let requests = std::mem::take(&mut *state.requests.lock().unwrap());
    assert!(grounding(&requests[0]).contains("\"garments\": [\"grey coat\"]"));
    for scene in &requests[1..] {
        let sent = grounding(scene);
        assert!(
            sent.contains("\"exposure\": \"clothed\", \"garments\": [\"Own fixture coat\"]"),
            "{sent}"
        );
        assert!(sent.contains("\"protected_facts\": [\"The lamp is broken.\"]"));
    }
    let mut edited = ready["opening_appearance"].clone();
    edited["characters"][0]["garments"] = json!([" Own edited cloak ", "Own edited cloak"]);
    edited["characters"][0]["exposure"] = json!("partially_clothed");
    let path = format!("/v1/video/authoring/{id}/review");
    let (code, reviewed) = call(&h, "POST", &path, Some(json!({"schema_version":1,"outline":null,"scenes":null,"text":null,"opening_appearance":edited}))).await;
    assert_eq!(code, 200, "{reviewed}");
    assert_eq!(reviewed["opening_appearance"], edited);
    assert_eq!(reviewed["scenes"], ready["scenes"]);
    let expand = format!("/v1/video/authoring/{id}/expand");
    for only in [false, true] {
        let (code, d) = call(
            &h,
            "POST",
            &expand,
            Some(json!({"schema_version":1,"start_index":0,"only":only})),
        )
        .await;
        assert_eq!(code, 200, "{d}");
        settle(&h, &id, "review").await;
        let requests = std::mem::take(&mut *state.requests.lock().unwrap());
        assert!(!requests.is_empty());
        for scene in &requests {
            assert!(
                grounding(scene).contains(
                    "\"exposure\": \"partially_clothed\", \"garments\": [\"Own edited cloak\"]"
                ),
                "{}",
                grounding(scene)
            );
        }
    }
    for (change, at) in [
        (
            json!({"reference_tag":"@visitor"}),
            "opening_appearance.characters[0].reference_tag",
        ),
        (
            json!({"garments":(0..13).map(|i| i.to_string()).collect::<Vec<_>>()}),
            "opening_appearance.characters[0].garments",
        ),
        (
            json!({"exposure":"dressed"}),
            "opening_appearance.characters[0].exposure",
        ),
        (
            json!({"source":"default_wardrobe"}),
            "opening_appearance.characters[0].source",
        ),
    ] {
        let mut invalid = edited.clone();
        for (key, value) in change.as_object().unwrap() {
            invalid["characters"][0][key] = value.clone();
        }
        let (code, d) = call(&h, "POST", &path, Some(json!({"schema_version":1,"outline":null,"scenes":null,"text":null,"opening_appearance":invalid}))).await;
        assert_eq!(code, 422, "{d}");
        assert_eq!(d["error"]["code"], "story_opening_appearance_invalid");
        assert_eq!(d["error"]["at"], at, "{d}");
    }
    let (_, unchanged) = call(&h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
    assert_eq!(unchanged["opening_appearance"], edited);
    let plain = story_request(&h, &project, "koboldcpp", "normal").await;
    let other = provider_story(&h, &project, plain).await;
    let (code, d) = call(&h, "POST", &format!("/v1/video/authoring/{}/review", other["id"].as_str().unwrap()),
        Some(json!({"schema_version":1,"outline":null,"scenes":null,"text":null,"opening_appearance":edited}))).await;
    assert_eq!(
        (code, &d["error"]["at"]),
        (422, &json!("opening_appearance")),
        "{d}"
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn every_image_reaches_the_model_as_png_at_its_planned_size() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "png-images").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    for transport in ["koboldcpp", "ollama"] {
        let (project, _, _) = grounded(&h).await;
        let large = attach_sized(&h, &project, 2000, 1000).await;
        let small = attach_sized(&h, &project, 40, 30).await;
        let mut request = story_request(&h, &project, transport, "normal").await;
        request["images"] = json!([
            {"id":"own_large","asset":large,"label":"Own large","usage":"setting_reference"},
            {"id":"own_small","asset":small,"label":"Own small","usage":"setting_reference"}]);
        provider_story(&h, &project, request).await;
        let requests = std::mem::take(&mut *state.requests.lock().unwrap());
        let sizes = requests[..2]
            .iter()
            .flat_map(images)
            .map(|(head, image)| (head, image.width(), image.height()))
            .collect::<Vec<_>>();
        let head = if transport == "ollama" {
            "ollama"
        } else {
            "data:image/png;base64"
        };
        assert_eq!(
            sizes,
            vec![(head.to_owned(), 1280, 640), (head.to_owned(), 40, 30)],
            "{transport}"
        );
        let frame = attach_sized(&h, &project, 1600, 900).await;
        let (_, current) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
        let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"description":"Own harbour","model":"project_default","settings":[],"sections":[{"id":"own_section","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"prompt":"Own keeper waits"}],"bindings":[{"id":"own_binding","asset":frame,"role":"first_frame","enabled":true,"video":null}],"continuity":null,"accepted_take":null});
        let (code, d) = call(&h, "POST", &format!("/v1/video/projects/{project}/edits"),
            Some(json!({"schema_version":1,"expected_project_revision":current["project"]["project_revision"],"commands":[{"operation":"add_scene","scene":scene}]}))).await;
        assert_eq!(code, 200, "{d}");
        let mut animate = story_request(&h, &project, transport, "normal").await;
        animate["operation"] = json!("animate");
        animate["scene_id"] = json!("own_scene");
        animate["brief"] = json!("");
        settle(&h, &start(&h, &project, animate).await, "review").await;
        let requests = std::mem::take(&mut *state.requests.lock().unwrap());
        let sent = images(&requests[0]);
        assert_eq!(sent.len(), 1);
        assert_eq!(
            (sent[0].0.as_str(), sent[0].1.width(), sent[0].1.height()),
            (head, 1280, 720)
        );
        // The same input sends the same bytes.
        let first = serde_json::to_string(&requests[0]["messages"]).unwrap_or_default()
            + &requests[0]["images"].to_string();
        let mut again = story_request(&h, &project, transport, "normal").await;
        again["operation"] = json!("animate");
        again["scene_id"] = json!("own_scene");
        again["brief"] = json!("");
        settle(&h, &start(&h, &project, again).await, "review").await;
        let requests = std::mem::take(&mut *state.requests.lock().unwrap());
        let second = serde_json::to_string(&requests[0]["messages"]).unwrap_or_default()
            + &requests[0]["images"].to_string();
        assert_eq!(first, second);
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
