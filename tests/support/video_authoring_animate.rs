//! Animate: a Scene's description, first Section prompt and ambient sound
//! written from its First frame, as Director's Animate image.
use super::authoring_messages::{attach, grounded, provider, sent_images, user_text};
use super::authoring_tests::fixture;
use super::*;

/// A project with cast, and a Scene that opens on an enabled First frame.
/// `enabled` none leaves the Scene without a First frame.
async fn animated(h: &Harness, enabled: Option<bool>, start: i64) -> String {
    let (project, _, _) = grounded(h).await;
    let frame = attach(h, &project, [10, 10, 10]).await;
    let (_, current) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":10,"den":1},"description":"Own harbour","model":"project_default","settings":[],"sections":[{"id":"own_section","start":{"num":start,"den":1},"duration":{"num":10 - start,"den":1},"prompt":"Own keeper waits"}],"bindings":enabled.map(|enabled|json!([{"id":"own_binding","asset":frame,"role":"first_frame","enabled":enabled,"video":null}])).unwrap_or(json!([])),"continuity":null,"accepted_take":null});
    let (code, added) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":current["project"]["project_revision"],"commands":[{"operation":"add_scene","scene":scene}]})),
    )
    .await;
    assert_eq!(code, 200, "{added}");
    project
}
async fn request(h: &Harness, project: &str) -> Value {
    let (_, d) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    json!({"schema_version":1,"expected_project_revision":d["project"]["project_revision"],
        "provider_profile_id":"proxy","provider":"koboldcpp","prompt_profile_id":"normal",
        "operation":"animate","scene_id":"own_scene","instruction":"Make @keeper climb","duration_seconds":10,
        "cast":["keeper_id","visitor_id"],"mentions":[{"text":"@keeper","reference_id":"keeper_id"}],
        "add_dialogue":true,"add_ambient_sound":true,
        "context":{"latest_moment":[{"role":"assistant","name":"Own Keeper","text":"The lamp is out."},{"role":"user","name":"","text":"Hurry."}],
            "cast_state":[{"reference_id":"keeper_id","current_appearance":{"exposure":"clothed","garments":[" grey coat ","boots"]},"fallback_wardrobe":"Own wool sweater"}]}})
}
/// A change to a valid request.
type Mutation = Box<dyn Fn(&mut Value)>;
async fn settle(h: &Harness, id: &str, phase: &str) -> Value {
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let (_, d) = call(h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
            if d["phase"] == phase {
                return d;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap()
}
#[tokio::test]
async fn animate_writes_reviews_and_applies_in_one_revision() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "animate").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let project = animated(&h, Some(true), 0).await;
    state
        .fail_animate_once
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let start = request(&h, &project).await;
    let (code, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(start.clone()),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap().to_owned();
    let failed = settle(&h, &id, "failed").await;
    assert_eq!(failed["failure"], "animate_response_invalid");
    assert!(failed["animate"].is_null());
    let (code, _) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/retry"),
        Some(json!({})),
    )
    .await;
    assert_eq!(code, 200);
    let ready = settle(&h, &id, "review").await;
    assert_eq!(
        ready["animate"],
        json!({"description":"Own animated harbour","prompt":"","ambient_sound":"Own gulls"})
    );
    let requests = std::mem::take(&mut *state.requests.lock().unwrap());
    assert_eq!(requests.len(), 2);
    assert_eq!(user_text(&requests[0]), user_text(&requests[1]));
    assert_eq!(sent_images(&requests[1]).len(), 1);
    assert_eq!(
        user_text(&requests[1]),
        "<scene_context>\n<latest_moment>\n[Own Keeper] The lamp is out.\n[] Hurry.\n</latest_moment>\n</scene_context>\n<active_characters>\n[\n  {\n    \"id\": \"keeper_id\",\n    \"name\": \"keeper\",\n    \"appearance\": \"Own fictional keeper\\nExposure: clothed\\nGarments: grey coat; boots\\nFallback wardrobe: Own wool sweater\"\n  },\n  {\n    \"id\": \"visitor_id\",\n    \"name\": \"visitor\",\n    \"appearance\": \"Own fictional visitor\"\n  }\n]\n</active_characters>\n<user_instruction>\n{\n  \"text\": \"Make @keeper climb\",\n  \"mentions\": [\n    {\n      \"text\": \"@keeper\",\n      \"character_id\": \"keeper_id\"\n    }\n  ]\n}\n</user_instruction>\n<audio_request>\n{\n  \"generate_dialogue\": true,\n  \"generate_background_sound\": true\n}\n</audio_request>\n<generation_context>\n{\n  \"duration_seconds\": 10,\n  \"aspect_ratio\": \"16:9\",\n  \"orientation\": \"Landscape\"\n}\n</generation_context>"
    );
    assert_eq!(ready["request"]["mentions"], start["mentions"]);
    let edited = json!({"description":"Own reviewed harbour","prompt":"Own keeper climbs","ambient_sound":""});
    let (code, reviewed) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/review"),
        Some(json!({"schema_version":1,"outline":null,"scenes":null,"text":null,"animate":edited})),
    )
    .await;
    assert_eq!(code, 200, "{reviewed}");
    assert_eq!(reviewed["animate"], edited);
    let revision = start["expected_project_revision"].as_u64().unwrap();
    let (code, applied) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/apply"),
        Some(json!({"schema_version":1,"expected_project_revision":revision})),
    )
    .await;
    assert_eq!(code, 200, "{applied}");
    let (_, after) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    let p = &after["project"];
    assert_eq!(p["project_revision"], revision + 1);
    assert_eq!(p["scenes"][0]["description"], "Own reviewed harbour");
    assert_eq!(p["scenes"][0]["sections"][0]["prompt"], "Own keeper climbs");
    assert_eq!(
        p["creative"]["scene_prompts"]["own_scene"]["ambient_sound"],
        ""
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn animate_inputs_fail_with_named_codes_and_start_nothing() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "animate-invalid").await;
    for (enabled, start, code) in [
        (None, 0, "animate_first_frame_missing"),
        (Some(false), 0, "animate_first_frame_disabled"),
        (Some(true), 2, "animate_scene_invalid"),
    ] {
        let project = animated(&h, enabled, start).await;
        let (status, d) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(request(&h, &project).await),
        )
        .await;
        assert_eq!(status, 422, "{d}");
        assert_eq!(d["error"]["code"], code, "{d}");
        assert_eq!(d["error"]["at"], "scene_id", "{d}");
    }
    let project = animated(&h, Some(true), 0).await;
    let base = request(&h, &project).await;
    let messages = (0..11)
        .map(|_| json!({"role":"user","name":"","text":"x"}))
        .collect::<Vec<_>>();
    let cases: Vec<(Mutation, &str, &str)> = vec![
        (
            Box::new(|r| r["provider"] = json!("external")),
            "animate_external_unsupported",
            "provider",
        ),
        (
            Box::new(|r| r["duration_seconds"] = json!(16)),
            "animate_context_invalid",
            "duration_seconds",
        ),
        (
            Box::new(move |r| r["context"]["latest_moment"] = json!(messages.clone())),
            "animate_context_invalid",
            "context.latest_moment",
        ),
        (
            Box::new(|r| {
                r["context"]["earlier_continuity"] = json!([{"role":"user","name":"","text":"x"}])
            }),
            "animate_context_invalid",
            "context.earlier_continuity",
        ),
        (
            Box::new(|r| r["context"]["protected_facts"] = json!(["x"])),
            "animate_context_invalid",
            "context.protected_facts",
        ),
        (
            Box::new(|r| r["context"]["latest_moment"][1]["role"] = json!("system")),
            "animate_context_invalid",
            "context.latest_moment[1].role",
        ),
        (
            Box::new(|r| r["mentions"][0]["reference_id"] = json!("absent_id")),
            "animate_context_invalid",
            "mentions[0].reference_id",
        ),
        (
            Box::new(|r| {
                r["context"]["cast_state"][0]["current_appearance"]["exposure"] = json!("dressed")
            }),
            "animate_context_invalid",
            "context.cast_state[0].current_appearance.exposure",
        ),
        (
            Box::new(|r| r["cast"] = json!(["visitor_id"])),
            "animate_context_invalid",
            "mentions[0].reference_id",
        ),
        (
            Box::new(|r| r["brief"] = json!("Own brief")),
            "animate_context_invalid",
            "brief",
        ),
        (
            Box::new(|r| r["scene_id"] = json!("absent_scene")),
            "scene_not_found",
            "scene_id",
        ),
    ];
    for (mutate, code, at) in cases {
        let mut r = base.clone();
        mutate(&mut r);
        let (status, d) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(r),
        )
        .await;
        assert_eq!(status, 422, "{at}: {d}");
        assert_eq!(d["error"]["code"], code, "{d}");
        assert_eq!(d["error"]["at"], at, "{d}");
    }
    let mut outline = base.clone();
    outline["operation"] = json!("outline");
    outline["context"]["latest_moment"] = json!([{"role":"user","name":"","text":"x"}]);
    outline["context"]["cast_state"] = json!([]);
    let (status, d) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(outline),
    )
    .await;
    assert_eq!(
        (status, &d["error"]["code"]),
        (422, &json!("invalid_authoring_request")),
        "{d}"
    );
    stop(h).await;
}
