//! A client's Story context (conversation, cast state, protected facts) as
//! Director's Story bridge receives it.
use super::authoring_messages::{
    grounded, provider, provider_story, settle, story_request, user_text,
};
use super::authoring_tests::fixture;
use super::*;

fn context() -> Value {
    json!({
        "earlier_continuity":[
            {"role":"assistant","name":"Own Keeper","text":"The storm is coming."},
            {"role":"user","name":" ","text":"Close the door."}],
        "latest_moment":[{"role":"assistant","name":"Own Visitor","text":"@keeper, the lamp!"}],
        "cast_state":[{"reference_id":"keeper_id",
            "current_appearance":{"exposure":"partially_clothed","garments":[" grey coat ","","grey coat"],"worn_items":["lantern"]},
            "fallback_wardrobe":" Own wool sweater "}],
        "protected_facts":["The lamp is broken."]})
}
/// A change to a valid request.
type Mutation<'a> = &'a dyn Fn(&mut Value);
async fn start(h: &Harness, project: &str, request: Value) -> (u16, Value) {
    let (code, d) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(request),
    )
    .await;
    (code, d)
}
#[tokio::test]
async fn context_reaches_the_outline_and_every_expansion_as_bridge_context() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story-context").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, _, _) = grounded(&h).await;
    let mut request = story_request(&h, &project, "koboldcpp", "normal").await;
    request["brief"] = json!("");
    request["cast"] = json!(["keeper_id", "visitor_id"]);
    request["context"] = context();
    let ready = provider_story(&h, &project, request).await;
    let requests = std::mem::take(&mut *state.requests.lock().unwrap());
    let outline = user_text(&requests[0]);
    assert!(outline.ends_with("<story>\n<earlier_continuity>\n[Own Keeper] The storm is coming.\n[user] Close the door.\n</earlier_continuity>\n\n<latest_moment>\n[Own Visitor] [[DIRECTOR_REF_0001]], the lamp!\n</latest_moment>\n</story>"), "{outline}");
    assert!(outline.contains("\n\nOpening appearance example:\n"));
    assert!(outline.contains(
        "<opening_appearance_context>\n{\"schema_version\": \"1.0\", \"source_digest\": \"sha256:"
    ));
    assert!(outline.contains("\"characters\": [{\"character_id\": \"keeper_id\", \"reference_tag\": \"[[DIRECTOR_REF_0001]]\", \"fallback_wardrobe\": \"Own wool sweater\"}]}\n</opening_appearance_context>"));
    assert!(outline.contains("\"current_appearance\": {\"exposure\": \"partially_clothed\", \"garments\": [\"grey coat\"], \"worn_items\": [\"lantern\"], \"temporary_traits\": [], \"visible_body_state\": []}}], \"protected_facts\": [\"The lamp is broken.\"]}\n</grounding_context>"));
    assert!(
        requests[0]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("\"required\":[\"scenes\",\"visual_placements\",\"opening_appearance\"]")
    );
    let mut echoed = context();
    echoed["cast_state"][0]["current_appearance"]["temporary_traits"] = json!([]);
    echoed["cast_state"][0]["current_appearance"]["visible_body_state"] = json!([]);
    assert_eq!(ready["request"]["context"], echoed);
    assert_eq!(
        ready["opening_appearance"]["characters"][0]["reference_tag"],
        "@keeper"
    );
    assert_eq!(
        ready["opening_appearance"]["characters"][0]["garments"],
        json!(["Own fixture coat"])
    );
    for sent in &requests[1..] {
        let scene = user_text(sent);
        assert!(
            scene.contains("\n</dialogue_timing>\n\n<grounding_context>\n")
                || scene.contains("\n</continuity_cast>\n\n<grounding_context>\n"),
            "{scene}"
        );
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn a_reviewed_outline_starts_a_draft_whose_scenes_expand_through_the_provider() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story-seeded").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, _, _) = grounded(&h).await;
    let mut request = story_request(&h, &project, "koboldcpp", "normal").await;
    request["brief"] = json!("");
    request["cast"] = json!(["keeper_id", "visitor_id"]);
    request["context"] = context();
    let ready = provider_story(&h, &project, request.clone()).await;
    state.requests.lock().unwrap().clear();
    // A client may keep the cast state but not the conversation it wrote the outline from.
    let mut seeded = request.clone();
    seeded["context"]["earlier_continuity"] = json!([]);
    seeded["context"]["latest_moment"] = json!([]);
    seeded["outline"] = ready["outline"].clone();
    seeded["opening_appearance"] = ready["opening_appearance"].clone();
    seeded["opening_appearance"]["characters"][0]["garments"] = json!(["Own reviewed coat"]);
    let (code, d) = start(&h, &project, seeded.clone()).await;
    assert_eq!(code, 202, "{d}");
    assert_eq!(d["phase"], "outline");
    assert_eq!(d["outline"], ready["outline"]);
    assert_eq!(
        d["scenes"].as_array().unwrap().len(),
        ready["outline"]["scenes"].as_array().unwrap().len()
    );
    assert!(state.requests.lock().unwrap().is_empty());
    let id = d["id"].as_str().unwrap();
    let (code, expanded) = call(
        &h,
        "POST",
        &format!("/v1/video/authoring/{id}/expand"),
        Some(json!({"schema_version":1,"start_index":0})),
    )
    .await;
    assert_eq!(code, 200, "{expanded}");
    let done = settle(&h, id, "review").await;
    assert!(
        done["scenes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| !s.is_null())
    );
    let requests = std::mem::take(&mut *state.requests.lock().unwrap());
    assert_eq!(requests.len(), d["scenes"].as_array().unwrap().len());
    assert!(user_text(&requests[0]).contains("Own reviewed coat"));
    assert!(!user_text(&requests[0]).contains("The storm is coming."));
    // Without an opening appearance the Scenes start from the context's current appearance.
    let mut unopened = seeded.clone();
    unopened["opening_appearance"] = Value::Null;
    let (code, d) = start(&h, &project, unopened).await;
    assert_eq!(code, 202, "{d}");
    assert_eq!(
        (&d["phase"], &d["opening_appearance"]),
        (&json!("outline"), &Value::Null)
    );
    // A given opening appearance must resolve the cast exactly; it belongs to a reviewed outline.
    let cases: Vec<(Mutation, u16, &str)> = vec![
        (
            &|r| r["opening_appearance"]["characters"][0]["character_id"] = json!("visitor_id"),
            422,
            "story_opening_appearance_invalid",
        ),
        // Without a reviewed outline the conversation is needed again.
        (
            &|r| r["outline"] = Value::Null,
            422,
            "story_context_invalid",
        ),
        (
            &|r| {
                r["outline"] = Value::Null;
                r["context"] = context();
            },
            422,
            "invalid_authoring_request",
        ),
        (
            &|r| r["provider"] = json!("external"),
            422,
            "invalid_authoring_request",
        ),
    ];
    for (mutate, status, code) in cases {
        let mut request = seeded.clone();
        mutate(&mut request);
        let (got, d) = start(&h, &project, request).await;
        assert_eq!(
            (got, d["error"]["code"].as_str()),
            (status, Some(code)),
            "{d}"
        );
    }
    assert!(state.requests.lock().unwrap().is_empty());
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn an_invalid_context_names_where_and_starts_nothing() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story-context-invalid").await;
    let (project, _, _) = grounded(&h).await;
    let base = {
        let mut request = story_request(&h, &project, "koboldcpp", "normal").await;
        request["brief"] = json!("");
        request["cast"] = json!(["keeper_id"]);
        request["context"] = context();
        request
    };
    let cases: Vec<(Mutation, &str)> = vec![
        (&|r| r["brief"] = json!("Own brief"), "brief"),
        (
            &|r| r["context"]["latest_moment"] = json!([]),
            "context.latest_moment",
        ),
        (
            &|r| r["context"]["earlier_continuity"][1]["role"] = json!("system"),
            "context.earlier_continuity[1].role",
        ),
        (
            &|r| r["context"]["latest_moment"][0]["text"] = json!("  "),
            "context.latest_moment[0].text",
        ),
        (
            &|r| r["cast"] = json!(["visitor_id"]),
            "context.cast_state[0].reference_id",
        ),
        (
            &|r| r["context"]["cast_state"][0]["current_appearance"]["exposure"] = json!("dressed"),
            "context.cast_state[0].current_appearance.exposure",
        ),
        (
            &|r| {
                r["context"]["cast_state"][0]["current_appearance"]["garments"] =
                    json!((0..13).map(|i| i.to_string()).collect::<Vec<_>>())
            },
            "context.cast_state[0].current_appearance.garments",
        ),
        (
            &|r| {
                r["context"]["cast_state"][0]["current_appearance"]["worn_items"] =
                    json!(["y".repeat(4001)])
            },
            "context.cast_state[0].current_appearance.worn_items[0]",
        ),
        (
            &|r| r["context"]["cast_state"][0]["fallback_wardrobe"] = json!("x".repeat(2001)),
            "context.cast_state[0].fallback_wardrobe",
        ),
        (
            &|r| r["context"]["protected_facts"] = json!(["Own fact", " "]),
            "context.protected_facts[1]",
        ),
        (
            &|r| {
                r["context"]["cast_state"] = json!([
                    r["context"]["cast_state"][0].clone(),
                    r["context"]["cast_state"][0].clone()
                ])
            },
            "context.cast_state[1].reference_id",
        ),
    ];
    for (mutate, at) in cases {
        let mut request = base.clone();
        mutate(&mut request);
        let (code, d) = start(&h, &project, request).await;
        assert_eq!(code, 422, "{at}: {d}");
        assert_eq!(d["error"]["code"], "story_context_invalid", "{d}");
        assert_eq!(d["error"]["at"], at, "{d}");
        assert!(d["error"]["detail"].is_string(), "{d}");
    }
    let (_, listed) = call(
        &h,
        "GET",
        &format!("/v1/video/projects/{project}/authoring"),
        None,
    )
    .await;
    assert_eq!(listed["drafts"], json!([]));
    stop(h).await;
}
#[tokio::test]
async fn an_outline_without_the_opening_appearance_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story-context-opening").await;
    let (project, _, _) = grounded(&h).await;
    let mut request = story_request(&h, &project, "external", "normal").await;
    request["provider_profile_id"] = json!("");
    request["brief"] = json!("");
    request["cast"] = json!(["keeper_id"]);
    request["context"] = context();
    let (code, d) = start(&h, &project, request.clone()).await;
    assert_eq!(code, 202, "{d}");
    let id = d["id"].as_str().unwrap().to_owned();
    let bundle = &d["bundle"];
    assert!(
        bundle["outline"]["user"]
            .as_str()
            .unwrap()
            .contains("<opening_appearance_context>")
    );
    assert!(
        bundle["outline"]["system"]
            .as_str()
            .unwrap()
            .contains("\"opening_appearance\"")
    );
    assert_eq!(
        bundle["expansion"]["slots"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|s| *s == "grounding_context")
            .count(),
        1
    );
    assert!(
        bundle["expansion"]["slot_values"]["grounding_context"]["captured"]
            .as_str()
            .unwrap()
            .contains("\"protected_facts\": [\"The lamp is broken.\"]")
    );
    let scene = json!({"title":"T","summary":"S","description":"D","ambient_sound":"","non_diegetic_music":"","weight":1,"boundary_from_previous":"hard_cut","setting_reference_matches":{},"sections":[{"prompt":"@keeper waits","weight":1}]});
    let expansion = {
        let mut s = scene.clone();
        s.as_object_mut()
            .unwrap()
            .remove("setting_reference_matches");
        s.to_string()
    };
    let opening = json!({"schema_version":"1.0","characters":[{"character_id":"keeper_id","reference_tag":"[[DIRECTOR_REF_0001]]","source":"default_wardrobe","exposure":"clothed","garments":["Own wool sweater"],"worn_items":[],"temporary_traits":[],"visible_body_state":[]}]});
    let submit = |outline: Value, dry_run: bool| json!({"schema_version":1,"expected_project_revision":request["expected_project_revision"],"inventory":[],"outline":outline.to_string(),"expansions":[expansion],"dry_run":dry_run});
    let path = format!("/v1/video/authoring/{id}/submission");
    let (code, d) = call(
        &h,
        "POST",
        &path,
        Some(submit(
            json!({"scenes":[scene],"visual_placements":[]}),
            true,
        )),
    )
    .await;
    assert_eq!(code, 422, "{d}");
    assert_eq!(
        d["error"],
        json!({"code":"story_opening_appearance_invalid","at":"outline"})
    );
    let (code, d) = call(
        &h,
        "POST",
        &path,
        Some(submit(
            json!({"scenes":[scene],"visual_placements":[],"opening_appearance":opening}),
            false,
        )),
    )
    .await;
    assert_eq!(code, 200, "{d}");
    let (_, draft) = call(&h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
    assert_eq!(
        draft["opening_appearance"]["characters"][0]["reference_tag"],
        "@keeper"
    );
    assert_eq!(
        draft["opening_appearance"]["characters"][0]["source"],
        "default_wardrobe"
    );
    stop(h).await;
}
#[tokio::test]
async fn without_the_new_fields_a_draft_carries_none_of_them() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "story-context-absent").await;
    let (url, worker, _) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, _, _) = grounded(&h).await;
    let request = story_request(&h, &project, "koboldcpp", "normal").await;
    let ready = provider_story(&h, &project, request).await;
    for key in ["context", "mentions", "add_ambient_sound"] {
        assert!(ready["request"].get(key).is_none(), "{key}");
    }
    for key in ["opening_appearance", "animate"] {
        assert!(ready.get(key).is_none(), "{key}");
    }
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
