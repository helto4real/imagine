//! External-agent authoring: the context bundle, one submission, the
//! production block and its dry run, on synthetic projects and images.
use super::authoring_messages::{
    attach, grounded, provider, provider_story, sent_images, settle, story_request, user_text,
};
use super::authoring_tests::fixture;
use super::*;
use imagine_server::video::authoring::engine;
const PROFILES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/config/video/prompt-profiles");
async fn start(h: &Harness, project: &str, request: Value) -> Value {
    let (code, d) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(request),
    )
    .await;
    assert_eq!(code, 202, "{d}");
    d
}
async fn external(h: &Harness, project: &str, profile: &str) -> Value {
    let mut request = story_request(h, project, "external", profile).await;
    request["provider_profile_id"] = json!("");
    request
}
async fn submit(h: &Harness, id: &str, body: Value) -> (u16, Value) {
    call(
        h,
        "POST",
        &format!("/v1/video/authoring/{id}/submission"),
        Some(body),
    )
    .await
}
async fn draft(h: &Harness, id: &str) -> Value {
    let (code, mut d) = call(h, "GET", &format!("/v1/video/authoring/{id}"), None).await;
    assert_eq!(code, 200, "{d}");
    d.as_object_mut().unwrap().remove("progress");
    d
}
async fn project_revision(h: &Harness, project: &str) -> Value {
    let (_, d) = call(h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    d["project"]["project_revision"].clone()
}
fn fill(template: &str, slots: &[(&str, &str)]) -> String {
    slots
        .iter()
        .fold(template.to_owned(), |text, (name, value)| {
            text.replace(&format!("{{{{SLOT:{name}}}}}"), value)
        })
}
/// One block of a Director message.
fn block<'a>(text: &'a str, tag: &str) -> &'a str {
    text.split(&format!("<{tag}>\n"))
        .nth(1)
        .and_then(|s| s.split(&format!("\n</{tag}>")).next())
        .unwrap()
}
/// The bundle's messages, filled with the provider path's own values,
/// reproduce every message the provider received.
fn assert_equivalent(requests: &[Value], ready: &Value, bundle: &Value) {
    let images = ready["request"]["images"].as_array().unwrap().len();
    let (inventory, rest) = requests.split_at(images);
    for (i, sent) in inventory.iter().enumerate() {
        assert_eq!(
            sent["messages"][0]["content"],
            bundle["inventory"]["system"]
        );
        assert_eq!(
            user_text(sent),
            bundle["inventory"]["messages"][i]["user"].as_str().unwrap()
        );
        assert_eq!(
            sent_images(sent),
            vec![bundle["images"][i]["base64"].as_str().unwrap().to_owned()]
        );
    }
    let outline = &rest[0];
    assert_eq!(
        outline["messages"][0]["content"],
        bundle["outline"]["system"]
    );
    let notes = ready["inventory"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| {
            let escaped = serde_json::to_string(n["note"].as_str().unwrap()).unwrap();
            (
                format!("inventory_note:{}", n["id"].as_str().unwrap()),
                escaped[1..escaped.len() - 1].to_owned(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        user_text(outline),
        fill(
            bundle["outline"]["user"].as_str().unwrap(),
            &notes
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect::<Vec<_>>()
        )
    );
    let scenes = ready["outline"]["scenes"].as_array().unwrap();
    assert_eq!(rest.len(), 1 + scenes.len());
    let template = bundle["expansion"]["user"].as_str().unwrap();
    let rule = &bundle["expansion"]["slot_values"]["image_context"];
    for (i, sent) in rest[1..].iter().enumerate() {
        assert_eq!(
            sent["messages"][0]["content"],
            bundle["expansion"]["system"]
        );
        let user = user_text(sent);
        let summary = |j: Option<usize>| {
            j.and_then(|j| scenes.get(j))
                .map(|s| s["summary"].as_str().unwrap())
                .unwrap_or("")
        };
        assert_eq!(
            block(&user, "previous_scene_summary"),
            summary(i.checked_sub(1))
        );
        assert_eq!(block(&user, "next_scene_summary"), summary(Some(i + 1)));
        let scene: Value = serde_json::from_str(block(&user, "scene_json")).unwrap();
        assert_eq!(scene["id"], scenes[i]["id"]);
        if i == 0 {
            let first =
                &bundle["expansion"]["slot_values"]["previous_scene_context"]["first_scene"];
            if !first.as_str().unwrap().contains('@') {
                assert_eq!(
                    block(&user, "previous_scene_context"),
                    first.as_str().unwrap()
                );
            }
        }
        let context = if sent_images(sent).is_empty() {
            &rule["without_images"]
        } else {
            &rule["with_images"]
        };
        let timing = if user.contains("<dialogue_timing>") {
            block(&user, "dialogue_timing")
        } else {
            ""
        };
        assert_eq!(
            user,
            fill(
                template,
                &[
                    (
                        "previous_scene_summary",
                        block(&user, "previous_scene_summary")
                    ),
                    (
                        "previous_scene_context",
                        block(&user, "previous_scene_context")
                    ),
                    ("next_scene_summary", block(&user, "next_scene_summary")),
                    ("selected_cast", block(&user, "selected_cast")),
                    ("continuity_cast", block(&user, "continuity_cast")),
                    ("dialogue_timing", timing),
                    ("scene_json", block(&user, "scene_json")),
                    ("image_context", context.as_str().unwrap()),
                ]
            ),
            "scene {i}"
        );
    }
}
/// What the fixture provider answers to a recorded request.
async fn answer(url: &str, body: &Value) -> String {
    let reply: Value = reqwest::Client::new()
        .post(format!("{url}/chat/completions"))
        .json(body)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    reply["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned()
}
fn without_owned(text: &str) -> String {
    let mut scene: Value = serde_json::from_str(text).unwrap();
    let object = scene.as_object_mut().unwrap();
    object.remove("id");
    object.remove("image_ids");
    for section in object["sections"].as_array_mut().unwrap() {
        for key in ["id", "image_ids", "first_frame_id", "last_frame_id"] {
            section.as_object_mut().unwrap().remove(key);
        }
    }
    scene.to_string()
}
#[tokio::test]
async fn external_bundle_matches_the_provider_path_and_submission_parses_identically() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-equivalence").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, images, assets) = grounded(&h).await;
    let mut request = story_request(&h, &project, "koboldcpp", "nsfw").await;
    request["cast"] = json!(["keeper_id", "visitor_id"]);
    request["images"] = images;
    request["add_dialogue"] = json!(true);
    let ready = provider_story(&h, &project, request.clone()).await;
    let requests = std::mem::take(&mut *state.requests.lock().unwrap());
    request["provider"] = json!("external");
    request["provider_profile_id"] = json!("");
    let d = start(&h, &project, request).await;
    assert_eq!(d["phase"], "outline");
    assert!(d["outline"].is_null());
    assert_eq!(d["provider_snapshot"]["profile_id"], "external");
    assert!(d["provider_snapshot"]["generation"]["story_outline"].is_object());
    let bundle = &d["bundle"];
    assert_equivalent(&requests, &ready, bundle);
    assert_eq!(
        bundle["outline"]["slots"],
        json!([
            "inventory_note:own_setting",
            "inventory_note:own_start",
            "inventory_note:own_end",
            "inventory_note:own_moment"
        ]),
        "{}",
        bundle["outline"]
    );
    assert_eq!(
        bundle["expansion"]["slots"],
        json!([
            "previous_scene_summary",
            "previous_scene_context",
            "next_scene_summary",
            "selected_cast",
            "continuity_cast",
            "dialogue_timing",
            "scene_json",
            "image_context"
        ])
    );
    let (_, capabilities) = call(&h, "GET", "/v1/video/capabilities", None).await;
    assert_eq!(bundle["limits"], capabilities["authoring"]);
    assert_eq!(bundle["profile"]["id"], "nsfw");
    assert_eq!(
        bundle["code_owned_fields"],
        json!([
            "id",
            "image_ids",
            "sections[].id",
            "sections[].image_ids",
            "sections[].first_frame_id",
            "sections[].last_frame_id"
        ])
    );
    // Story images as the provider receives them, then cast identity images.
    let roles = bundle["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            (
                i["image_id"].as_str().unwrap(),
                i["role"].as_str().unwrap(),
                i["usage"].as_str().unwrap(),
                i["label"].as_str().unwrap(),
                i["mime"].as_str().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        vec![
            (
                "own_setting",
                "story",
                "setting_reference",
                "Own lighthouse",
                "image/png"
            ),
            (
                "own_start",
                "story",
                "start_frame",
                "Own door closed",
                "image/png"
            ),
            (
                "own_end",
                "story",
                "end_frame",
                "Own door open",
                "image/png"
            ),
            (
                "own_moment",
                "story",
                "required_moment",
                "Own beacon lit",
                "image/png"
            ),
            (
                "own_keeper_face",
                "cast",
                "identity_reference",
                "keeper (@keeper)",
                "image/png"
            ),
        ]
    );
    {
        use base64::Engine;
        let png = {
            let mut out = Cursor::new(vec![]);
            image::RgbImage::from_pixel(16, 16, image::Rgb([90, 20, 120]))
                .write_to(&mut out, image::ImageFormat::Png)
                .unwrap();
            out.into_inner()
        };
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(bundle["images"][4]["base64"].as_str().unwrap())
                .unwrap(),
            imagine_server::video::authoring::vision::normalize(&png)
                .unwrap()
                .png
        );
    }
    assert!(!bundle.to_string().contains(&assets[4]));
    // The provider's own answers, submitted once, parse to the same Draft.
    let images = ready["request"]["images"].as_array().unwrap().len();
    let outline = answer(&url, &requests[images]).await;
    let mut expansions = vec![];
    for sent in &requests[images + 1..] {
        expansions.push(without_owned(&answer(&url, sent).await));
    }
    state.requests.lock().unwrap().clear();
    let id = d["id"].as_str().unwrap();
    let mut inventory = ready["inventory"].as_array().unwrap().clone();
    inventory.reverse();
    let body = json!({"schema_version":1,"expected_project_revision":ready["request"]["expected_project_revision"],"inventory":inventory,"outline":outline,"expansions":expansions,"dry_run":true});
    let (code, report) = submit(&h, id, body).await;
    assert_eq!(code, 200, "{report}");
    assert_eq!(report["draft_id"], id);
    assert_eq!(report["scenes"].as_array().unwrap().len(), 2);
    let parsed = draft(&h, id).await;
    assert_eq!(parsed["phase"], "review");
    for key in [
        "outline",
        "scenes",
        "planned_durations",
        "inventory",
        "selected_cast",
    ] {
        assert_eq!(parsed[key], ready[key], "{key}");
    }
    assert!(parsed.get("production").is_none());
    // And to the same apply plan.
    let service = h.control.video.lock().await;
    let current = service
        .projects
        .read(&imagine_server::video::contract::ProjectId(project.clone()))
        .unwrap();
    let plan = |id: &str| {
        let draft = service.authoring.read(id).unwrap();
        serde_json::to_string(
            &engine::apply(
                &current,
                &draft,
                &imagine_server::video::presets::Catalog::builtin(),
            )
            .unwrap(),
        )
        .unwrap()
        .replace(id, "<draft>")
    };
    assert_eq!(plan(id), plan(ready["id"].as_str().unwrap()));
    drop(service);
    // No provider request on the external path.
    assert!(state.requests.lock().unwrap().is_empty());
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn external_continuation_bundle_matches_the_provider_path() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-continuation").await;
    let (url, worker, state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (_, d) = call(&h, "POST", "/v1/video/projects", Some(create())).await;
    let project = d["project"]["id"].as_str().unwrap().to_owned();
    let scene = json!({"id":"own_scene","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"description":"Own harbour at dusk","model":"project_default","settings":[],"sections":[{"id":"own_section","start":{"num":0,"den":1},"duration":{"num":4,"den":1},"prompt":"Own keeper waits"}],"bindings":[],"continuity":null,"accepted_take":null});
    let (code, added) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":0,"commands":[{"operation":"add_scene","scene":scene}]})),
    )
    .await;
    assert_eq!(code, 200, "{added}");
    let mut request = story_request(&h, &project, "koboldcpp", "nsfw").await;
    request["destination"] = json!("continue");
    let ready = provider_story(&h, &project, request.clone()).await;
    let requests = std::mem::take(&mut *state.requests.lock().unwrap());
    request["provider"] = json!("external");
    let d = start(&h, &project, request).await;
    let bundle = &d["bundle"];
    assert_equivalent(&requests, &ready, bundle);
    assert_eq!(bundle["outline"]["slots"], json!([]));
    assert_eq!(bundle["images"], json!([]));
    assert!(
        bundle["expansion"]["slot_values"]["previous_scene_context"]["first_scene"]
            .as_str()
            .unwrap()
            .contains("Own keeper waits")
    );
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
#[tokio::test]
async fn bundle_and_profile_description_deliver_every_profile_verbatim() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-verbatim").await;
    let (project, images, _) = grounded(&h).await;
    let (_, capabilities) = call(&h, "GET", "/v1/video/capabilities", None).await;
    let (_, catalog) = call(&h, "GET", "/v1/video/authoring/profiles", None).await;
    let file = |profile: &Value, key: &str| {
        let entry = profile["prompt_files"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["key"] == key)
            .unwrap()
            .clone();
        let raw = std::fs::read_to_string(format!(
            "{PROFILES}/{}/{}",
            entry["source_profile_id"].as_str().unwrap(),
            entry["file"].as_str().unwrap()
        ))
        .unwrap();
        // The provider receives the file without its surrounding whitespace.
        assert_eq!(raw.trim_start(), raw);
        (raw.trim_end().to_owned(), entry)
    };
    let mut seen = vec![];
    for profile in catalog["profiles"].as_array().unwrap() {
        let id = profile["id"].as_str().unwrap();
        let (code, described) = call(
            &h,
            "GET",
            &format!("/v1/video/authoring/profiles/{id}"),
            None,
        )
        .await;
        assert_eq!(code, 200, "{described}");
        assert_eq!(&described["profile"], profile);
        assert_eq!(described["limits"], capabilities["authoring"]);
        assert_eq!(
            described["agent_guidance"].as_str().unwrap(),
            include_str!("../../config/video/agent-guidance.txt")
        );
        for dialogue in [false, true] {
            let mut request = external(&h, &project, id).await;
            request["images"] = images.clone();
            request["cast"] = json!(["keeper_id", "visitor_id"]);
            request["add_dialogue"] = json!(dialogue);
            let d = start(&h, &project, request).await;
            let bundle = &d["bundle"];
            assert_eq!(&bundle["profile"]["prompt_files"], &profile["prompt_files"]);
            assert_eq!(bundle["profile"]["revision"], profile["revision"]);
            let schema = bundle["outline"]["output_schema"].to_string();
            let suffix = if dialogue { "_dialogue" } else { "" };
            let (outline, entry) = file(profile, &format!("story_outline{suffix}"));
            assert!(outline.contains("{{STORY_OUTLINE_JSON_SCHEMA}}"));
            assert_eq!(
                bundle["outline"]["system"].as_str().unwrap(),
                outline.replace("{{STORY_OUTLINE_JSON_SCHEMA}}", &schema)
            );
            seen.push(entry["source_profile_id"].clone());
            let (scene, _) = file(profile, &format!("story_scene{suffix}"));
            assert_eq!(bundle["expansion"]["system"].as_str().unwrap(), scene);
            let (inventory, entry) = file(profile, "story_image_inventory");
            assert_eq!(bundle["inventory"]["system"].as_str().unwrap(), inventory);
            if id == "nsfw" {
                assert_eq!(entry["inherited"], true);
                assert_eq!(entry["source_profile_id"], "normal");
            }
        }
    }
    assert!(seen.contains(&json!("nsfw")));
    let (code, missing) = call(&h, "GET", "/v1/video/authoring/profiles/own_missing", None).await;
    assert_eq!(code, 404);
    assert_eq!(missing["error"]["code"], "profile_not_found");
    let (code, error) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(external(&h, &project, "own_missing").await),
    )
    .await;
    assert_eq!(code, 422);
    assert_eq!(error["error"]["code"], "profile_not_found");
    // An external Draft never falls back to the default profile.
    for empty in ["", "  "] {
        let (code, error) = call(
            &h,
            "POST",
            &format!("/v1/video/projects/{project}/authoring"),
            Some(external(&h, &project, empty).await),
        )
        .await;
        assert_eq!(code, 422, "{error}");
        assert_eq!(error["error"]["code"], "profile_not_found");
    }
    stop(h).await;
}
#[tokio::test]
async fn drafts_are_listed_and_an_external_bundle_is_rebuilt_after_a_lost_start() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-recovery").await;
    let (url, worker, _state) = fixture::spawn().await;
    provider(&h, &url).await;
    let (project, images, _) = grounded(&h).await;
    let mut request = external(&h, &project, "nsfw").await;
    request["images"] = images;
    request["cast"] = json!(["keeper_id"]);
    let started = start(&h, &project, request).await;
    let id = started["id"].as_str().unwrap();
    let (code, provided) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/authoring"),
        Some(story_request(&h, &project, "koboldcpp", "nsfw").await),
    )
    .await;
    assert_eq!(code, 202, "{provided}");
    let (code, listed) = call(
        &h,
        "GET",
        &format!("/v1/video/projects/{project}/authoring"),
        None,
    )
    .await;
    assert_eq!(code, 200, "{listed}");
    assert_eq!(listed["project_id"], project.as_str());
    assert_eq!(
        listed["drafts"][0],
        json!({"id":id,"phase":"outline","provider":"external","prompt_profile_id":"nsfw"})
    );
    assert_eq!(listed["drafts"][1]["id"], provided["id"]);
    assert_eq!(listed["drafts"][1]["provider"], "koboldcpp");
    let (code, rebuilt) = call(&h, "GET", &format!("/v1/video/authoring/{id}/bundle"), None).await;
    assert_eq!(code, 200, "{rebuilt}");
    assert_eq!(rebuilt["draft_id"], id);
    assert_eq!(rebuilt["bundle"], started["bundle"]);
    let (code, error) = call(
        &h,
        "GET",
        &format!(
            "/v1/video/authoring/{}/bundle",
            provided["id"].as_str().unwrap()
        ),
        None,
    )
    .await;
    assert_eq!(code, 409, "{error}");
    assert_eq!(error["error"]["code"], "authoring_not_external");
    for path in [
        "/v1/video/authoring/own_missing/bundle".to_owned(),
        "/v1/video/projects/own_missing/authoring".to_owned(),
    ] {
        let (code, _) = call(&h, "GET", &path, None).await;
        assert_eq!(code, 404, "{path}");
    }
    settle(&h, provided["id"].as_str().unwrap(), "outline").await;
    stop(h).await;
    worker.abort();
    let _ = worker.await;
}
/// Five Scenes: a referenced setting, an I2V opening frame, a last-frame
/// continuation, a motion continuation and an imageless cut.
fn story() -> (Value, Vec<String>) {
    let scene = |title: &str, boundary: &str, setting: bool, sections: usize| json!({"title":title,"summary":format!("Own {title} summary"),"description":format!("Own {title}"),"ambient_sound":"Waves","non_diegetic_music":"","weight":1,"boundary_from_previous":boundary,"setting_reference_matches":{"own_setting":setting},"sections":(0..sections).map(|j|json!({"prompt":format!("Own {title} beat {j}"),"weight":1})).collect::<Vec<_>>()});
    let outline = json!({"scenes":[
        scene("arrival","hard_cut",true,2),
        scene("door","hard_cut",false,1),
        scene("entry","continuous_shot",false,1),
        scene("parcel","continuous_shot",false,1),
        scene("close","hard_cut",false,1)
    ],"visual_placements":[{"beat_key":"start_end_pair:beat_1","scene_ordinal":2,"section_ordinal":1,"semantic_role":"state_transition"}]});
    let expansions = outline["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            let mut s = s.clone();
            s.as_object_mut()
                .unwrap()
                .remove("setting_reference_matches");
            s["description"] = json!(format!("Expanded {}", s["description"].as_str().unwrap()));
            s.to_string()
        })
        .collect();
    (outline, expansions)
}
async fn production_project(h: &Harness) -> (String, Value) {
    let (_, d) = call(h, "POST", "/v1/video/projects", Some(create())).await;
    let project = d["project"]["id"].as_str().unwrap().to_owned();
    let setting = attach(h, &project, [40, 90, 160]).await;
    let open = attach(h, &project, [200, 60, 30]).await;
    let (_, capabilities) = call(h, "GET", "/v1/video/capabilities", None).await;
    let reference = capabilities["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["model"]["variant"] == "reference")
        .unwrap()["model"]
        .clone();
    let (code, edited) = call(
        h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":project_revision(h,&project).await,"commands":[{"operation":"set_default_model","model":reference}]})),
    )
    .await;
    assert_eq!(code, 200, "{edited}");
    let images = json!([
        {"id":"own_setting","asset":setting,"label":"Own workshop","usage":"setting_reference"},
        {"id":"own_open","asset":open,"label":"Own door opening","usage":"start_frame","pair_with":"own_open"}
    ]);
    (project, images)
}
fn body(outline: &Value, expansions: &[String], production: Value, revision: &Value) -> Value {
    json!({"schema_version":1,"expected_project_revision":revision,"inventory":[{"id":"own_setting","note":"Own setting note"},{"id":"own_open","note":"Own frame note"}],"outline":outline.to_string(),"expansions":expansions,"production":production,"dry_run":true})
}
fn production(models: [&str; 5], continuity: [Value; 5]) -> Value {
    json!({"scenes":models.iter().zip(continuity).map(|(m,c)|json!({"model":m,"continuity":c,"rationale":"Own note"})).collect::<Vec<_>>()})
}
fn planned() -> Value {
    production(
        ["reference", "i2v", "i2v", "reference", "reference"],
        [
            json!(null),
            json!(null),
            json!({"method":"auto","include_audio":false}),
            json!({"method":"influence","include_audio":false}),
            json!(null),
        ],
    )
}
#[tokio::test]
async fn production_block_dry_run_and_commit_apply_models_and_continuity() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-production").await;
    let (project, images) = production_project(&h).await;
    let mut request = external(&h, &project, "nsfw").await;
    request["images"] = images;
    request["duration_seconds"] = json!(25);
    let d = start(&h, &project, request).await;
    let id = d["id"].as_str().unwrap();
    let revision = project_revision(&h, &project).await;
    let (outline, expansions) = story();
    let (code, report) = submit(&h, id, body(&outline, &expansions, planned(), &revision)).await;
    assert_eq!(code, 200, "{report}");
    let scenes = report["scenes"].as_array().unwrap();
    let summary = scenes
        .iter()
        .map(|s| {
            (
                s["ordinal"].as_u64().unwrap(),
                s["model"].as_str().unwrap().to_owned(),
                s["continuity"]["method"].as_str().map(str::to_owned),
                s["planned_seconds"].as_f64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        summary,
        vec![
            (1, "reference".into(), None, 5.),
            (2, "i2v".into(), None, 5.),
            (3, "i2v".into(), Some("last_frame_first_frame".into()), 5.),
            (4, "reference".into(), Some("influence".into()), 5.),
            (5, "reference".into(), None, 5.),
        ]
    );
    assert_eq!(scenes[2]["continuity"]["include_audio"], false);
    assert!(
        scenes.iter().all(|s| !s["blockers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b == "scene_runtime_unsupported"
                || b == "h3_section_reference_unavailable"
                || b == "continuity_first_frame_conflict")),
        "{report}"
    );
    assert_eq!(
        report["warnings"],
        json!([
            {"ordinal":3,"code":"continuity_source_pending","source_scene":format!("scene_{id}_1")},
            {"ordinal":4,"code":"continuity_source_pending","source_scene":format!("scene_{id}_2")}
        ])
    );
    assert_eq!(
        report["valid"],
        scenes.iter().all(|s| s["blockers"] == json!([]))
    );
    // A dry run changes the Draft, never the project.
    assert_eq!(project_revision(&h, &project).await, revision);
    let parsed = draft(&h, id).await;
    assert_eq!(parsed["phase"], "review");
    assert_eq!(parsed["production"], planned());
    // Reviewing the same outline keeps the production block; a changed one clears it.
    let review =
        |outline: &Value| json!({"schema_version":1,"outline":outline,"scenes":null,"text":null});
    let path = format!("/v1/video/authoring/{id}/review");
    let (code, kept) = call(&h, "POST", &path, Some(review(&parsed["outline"]))).await;
    assert_eq!(code, 200, "{kept}");
    assert_eq!(kept["production"], planned());
    let mut changed = parsed["outline"].clone();
    changed["scenes"][0]["summary"] = json!("Own changed summary");
    let (code, cleared) = call(&h, "POST", &path, Some(review(&changed))).await;
    assert_eq!(code, 200, "{cleared}");
    assert!(cleared.get("production").is_none(), "{cleared}");
    let mut commit = body(&outline, &expansions, planned(), &revision);
    commit["dry_run"] = json!(false);
    let (code, applied) = submit(&h, id, commit.clone()).await;
    assert_eq!(code, 200, "{applied}");
    let scenes = applied["project"]["scenes"].as_array().unwrap();
    assert_eq!(scenes.len(), 5);
    let map = applied["scene_map"].as_array().unwrap();
    for (i, scene) in scenes.iter().enumerate() {
        assert_eq!(map[i]["ordinal"], i + 1);
        assert_eq!(map[i]["scene_id"], scene["id"]);
        assert_eq!(
            map[i]["section_ids"],
            json!(
                scene["sections"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|s| &s["id"])
                    .collect::<Vec<_>>()
            )
        );
    }
    assert_eq!(scenes[0]["model"], "project_default");
    assert_eq!(scenes[1]["model"]["override"]["variant"], "i2v");
    assert_eq!(scenes[2]["model"]["override"]["variant"], "i2v");
    assert_eq!(scenes[3]["model"], "project_default");
    assert!(scenes.iter().all(|s| s["settings"] == json!([])));
    // The I2V opening frame is its only input: one First Frame binding.
    assert_eq!(scenes[1]["bindings"].as_array().unwrap().len(), 1);
    assert_eq!(scenes[1]["bindings"][0]["role"], "first_frame");
    let opening = &applied["project"]["creative"]["sections"]
        [scenes[1]["sections"][0]["id"].as_str().unwrap()];
    assert_eq!(opening["images"].as_array().unwrap().len(), 1);
    assert_eq!(opening["images"][0]["role"], "first_frame");
    assert_eq!(scenes[2]["continuity"]["method"], "last_frame_first_frame");
    assert_eq!(scenes[2]["continuity"]["source_scene"], scenes[1]["id"]);
    assert_eq!(scenes[3]["continuity"]["method"], "influence");
    assert_eq!(applied["authoring"][0]["phase"], "applied");
    assert_eq!(applied["authoring"][0]["production"], planned());
    // The commit is what the dry run checked.
    let blockers = |s: &Value| {
        let service = h.control.video.try_lock().unwrap();
        let project = service
            .projects
            .read(&imagine_server::video::contract::ProjectId(project.clone()))
            .unwrap();
        imagine_server::video::execution::preflight(
            &project,
            &imagine_server::video::contract::SceneId(s["id"].as_str().unwrap().into()),
        )
        .into_iter()
        .filter(|b| b != "continuity_source_not_accepted" && b != "continuity_source_missing")
        .collect::<Vec<_>>()
    };
    for (scene, checked) in scenes.iter().zip(report["scenes"].as_array().unwrap()) {
        assert_eq!(json!(blockers(scene)), checked["blockers"]);
    }
    let (code, again) = submit(&h, id, commit).await;
    assert_eq!(code, 409, "{again}");
    stop(h).await;
}
#[tokio::test]
async fn submission_errors_name_where_and_leave_the_draft_unchanged() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-errors").await;
    let (project, images) = production_project(&h).await;
    let mut request = external(&h, &project, "nsfw").await;
    request["images"] = images;
    request["duration_seconds"] = json!(25);
    let d = start(&h, &project, request).await;
    let id = d["id"].as_str().unwrap();
    let revision = project_revision(&h, &project).await;
    let (outline, expansions) = story();
    // A valid dry run first, so errors must keep its result.
    let (code, report) = submit(&h, id, body(&outline, &expansions, planned(), &revision)).await;
    assert_eq!(code, 200, "{report}");
    let before = draft(&h, id).await;
    let mut cases = vec![];
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["outline"] = json!("{not json");
    cases.push((b, "authoring_invalid_json", "outline", None));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["expansions"].as_array_mut().unwrap().pop();
    cases.push((b, "story_expansion_count_mismatch", "expansions", None));
    let mut b = body(&outline, &expansions, planned(), &revision);
    let mut changed: Value = serde_json::from_str(&expansions[1]).unwrap();
    changed["sections"][0]["weight"] = json!(2);
    b["expansions"][1] = json!(changed.to_string());
    cases.push((
        b,
        "story_expansion_changed_structure",
        "expansions[1]",
        None,
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    let mut changed: Value = serde_json::from_str(&expansions[0]).unwrap();
    changed["sections"].as_array_mut().unwrap().pop();
    b["expansions"][0] = json!(changed.to_string());
    cases.push((
        b,
        "story_expansion_changed_structure",
        "expansions[0]",
        None,
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["expansions"][2] = json!("[]");
    cases.push((b, "authoring_invalid_json", "expansions[2]", None));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["inventory"].as_array_mut().unwrap().pop();
    cases.push((b, "authoring_inventory_mismatch", "inventory", None));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["inventory"][1]["id"] = json!("own_unknown");
    cases.push((b, "authoring_inventory_mismatch", "inventory", None));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["production"]["scenes"].as_array_mut().unwrap().pop();
    cases.push((
        b,
        "production_scene_count_mismatch",
        "production.scenes",
        None,
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["production"]["scenes"][2]["continuity"]["method"] = json!("influence");
    cases.push((
        b,
        "continuity_method_unsupported",
        "production.scenes[2]",
        None,
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["production"]["scenes"][2]["continuity"]["include_audio"] = json!(true);
    cases.push((
        b,
        "continuity_endpoint_incompatible",
        "production.scenes[2]",
        None,
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["production"]["scenes"][0]["continuity"] = json!({"method":"auto"});
    cases.push((
        b,
        "continuity_requires_continuous_shot",
        "production.scenes[0]",
        None,
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["production"]["scenes"][0]["model"] = json!("i2v");
    cases.push((
        b,
        "i2v_input_role_unsupported",
        "production.scenes[0]",
        Some("sections[0]"),
    ));
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["production"]["scenes"][4]["model"] = json!("i2v");
    cases.push((b, "i2v_start_frame_required", "production.scenes[4]", None));
    // A continuous shot into an I2V Scene that has its own start frame.
    let mut conflicting = outline.clone();
    conflicting["scenes"][1]["boundary_from_previous"] = json!("continuous_shot");
    let mut moved = expansions.clone();
    let mut second: Value = serde_json::from_str(&moved[1]).unwrap();
    second["boundary_from_previous"] = json!("continuous_shot");
    moved[1] = second.to_string();
    let b = body(&conflicting, &moved, planned(), &revision);
    cases.push((
        b,
        "continuity_first_frame_conflict",
        "production.scenes[1]",
        Some("sections[0]"),
    ));
    // Nothing precedes the first new Scene when appending, with or without production.
    let mut opening = outline.clone();
    opening["scenes"][0]["boundary_from_previous"] = json!("continuous_shot");
    let mut first = expansions.clone();
    let mut scene: Value = serde_json::from_str(&first[0]).unwrap();
    scene["boundary_from_previous"] = json!("continuous_shot");
    first[0] = scene.to_string();
    for production in [planned(), json!(null)] {
        cases.push((
            body(&opening, &first, production, &revision),
            "story_continuity_source_missing",
            "outline",
            Some("scenes[0]"),
        ));
    }
    // Too few Scenes for the duration: the minimum count is reported.
    let mut single = outline.clone();
    single["scenes"].as_array_mut().unwrap().truncate(1);
    single["visual_placements"][0]["scene_ordinal"] = json!(1);
    cases.push((
        body(&single, &expansions[..1], json!(null), &revision),
        "story_scene_duration_capacity",
        "outline",
        Some("minimum_scenes=2"),
    ));
    for (b, code, at, detail) in cases {
        let (status, error) = submit(&h, id, b).await;
        assert_eq!(status, 422, "{code}: {error}");
        assert_eq!(error["error"]["code"], code, "{error}");
        assert_eq!(error["error"]["at"], at, "{error}");
        assert_eq!(error["error"]["detail"].as_str(), detail, "{error}");
        assert_eq!(draft(&h, id).await, before, "{code}");
        assert_eq!(project_revision(&h, &project).await, revision);
    }
    let mut b = body(&outline, &expansions, planned(), &revision);
    b["expected_project_revision"] = json!(revision.as_u64().unwrap() + 1);
    let (status, error) = submit(&h, id, b).await;
    assert_eq!(status, 409, "{error}");
    assert_eq!(error["error"]["code"], "authoring_revision_conflict");
    assert!(error["error"].get("at").is_none());
    // Provider-only operations do not apply to an external Draft.
    for path in ["expand", "retry"] {
        let (status, error) = call(
            &h,
            "POST",
            &format!("/v1/video/authoring/{id}/{path}"),
            Some(json!({"schema_version":1,"start_index":0})),
        )
        .await;
        assert_eq!(status, 409, "{error}");
        assert_eq!(error["error"]["code"], "authoring_not_generated");
    }
    assert_eq!(draft(&h, id).await, before);
    stop(h).await;
}
#[tokio::test]
async fn apply_takes_a_production_block_and_a_dry_run() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-apply").await;
    let (project, images) = production_project(&h).await;
    let mut request = external(&h, &project, "nsfw").await;
    request["images"] = images;
    request["duration_seconds"] = json!(25);
    let d = start(&h, &project, request).await;
    let id = d["id"].as_str().unwrap();
    let revision = project_revision(&h, &project).await;
    let (outline, expansions) = story();
    let (code, report) = submit(&h, id, body(&outline, &expansions, json!(null), &revision)).await;
    assert_eq!(code, 200, "{report}");
    // Without a production block, the provider path's defaults.
    let models = |r: &Value| {
        r["scenes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| (s["model"].clone(), s["continuity"]["method"].clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        models(&report),
        vec![
            (json!("reference"), json!(null)),
            (json!("reference"), json!(null)),
            (json!("reference"), json!("motion_context_exp")),
            (json!("reference"), json!("motion_context_exp")),
            (json!("reference"), json!(null)),
        ]
    );
    let path = format!("/v1/video/authoring/{id}/apply");
    let mut wrong = planned();
    wrong["scenes"][4]["model"] = json!("i2v");
    let (code, error) = call(&h, "POST", &path, Some(json!({"schema_version":1,"expected_project_revision":revision,"production":wrong,"dry_run":true}))).await;
    assert_eq!(code, 422, "{error}");
    assert_eq!(error["error"]["code"], "i2v_start_frame_required");
    assert_eq!(error["error"]["at"], "production.scenes[4]");
    assert!(draft(&h, id).await.get("production").is_none());
    let (code, checked) = call(&h, "POST", &path, Some(json!({"schema_version":1,"expected_project_revision":revision,"production":planned(),"dry_run":true}))).await;
    assert_eq!(code, 200, "{checked}");
    assert_eq!(models(&checked)[1], (json!("i2v"), json!(null)));
    assert_eq!(draft(&h, id).await["production"], planned());
    assert_eq!(project_revision(&h, &project).await, revision);
    let (code, applied) = call(
        &h,
        "POST",
        &path,
        Some(json!({"schema_version":1,"expected_project_revision":revision})),
    )
    .await;
    assert_eq!(code, 200, "{applied}");
    assert_eq!(
        applied["project"]["scenes"][2]["continuity"]["method"],
        "last_frame_first_frame"
    );
    assert_eq!(applied["scene_map"].as_array().unwrap().len(), 5);
    stop(h).await;
}
#[tokio::test]
async fn capabilities_report_the_authoring_limits_and_continuity_table() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "authoring-capabilities").await;
    let (code, capabilities) = call(&h, "GET", "/v1/video/capabilities", None).await;
    assert_eq!(code, 200, "{capabilities}");
    assert_eq!(
        capabilities["authoring"],
        json!({"max_scene_seconds":15.0,"max_scenes":128,"max_sections_per_scene":12,
            "supported_boundaries":["hard_cut","continuous_shot"],"char_limit":100000,
            "continuity":{
                "reference":{"methods":["combo","influence","motion_context","motion_context_exp"],"auto":"motion_context_exp"},
                "i2v":{"methods":["last_frame_first_frame"],"auto":"last_frame_first_frame"}},
            "presets":[
                {"key":"Default ref2v","variant":"reference","description":"Standard MiniMax H3 reference-to-video","default":true,"available":true,"problem":null},
                {"key":"Default i2v","variant":"i2v","description":"Standard MiniMax H3 image-to-video","default":true,"available":true,"problem":null}],
            "presets_problem":null})
    );
    assert_eq!(
        capabilities["authoring"]["max_scene_seconds"],
        json!(engine::MAX_SCENE_SECONDS)
    );
    stop(h).await;
}
#[tokio::test]
async fn dry_run_warns_when_an_i2v_scene_names_cast_and_reports_boundaries() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-i2v-cast").await;
    let (project, images) = production_project(&h).await;
    let reference = json!({"id":"reference_own_keeper","name":"Keeper","alias":"keeper","kind":"character",
        "description":"Own keeper description","tags":[],"enabled":true,"strength":1.0,"retention":"",
        "images":[{"id":"image_own_keeper","asset":images[0]["asset"],"enabled":true,"preview":true}],
        "voice":null,"origin":null});
    let (code, edited) = call(
        &h,
        "POST",
        &format!("/v1/video/projects/{project}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":project_revision(&h,&project).await,
            "commands":[{"operation":"upsert_visual_reference","reference":reference}]})),
    )
    .await;
    assert_eq!(code, 200, "{edited}");
    let mut request = external(&h, &project, "nsfw").await;
    request["images"] = images;
    request["duration_seconds"] = json!(25);
    request["cast"] = json!(["reference_own_keeper"]);
    let d = start(&h, &project, request).await;
    let id = d["id"].as_str().unwrap();
    let revision = project_revision(&h, &project).await;
    let (outline, mut expansions) = story();
    // Scene 1 (reference) and Scene 2 (i2v) name the cast member; Scene 3 (i2v) does not.
    for i in [0, 1] {
        let mut scene: Value = serde_json::from_str(&expansions[i]).unwrap();
        scene["description"] = json!(format!(
            "@keeper walks. {}",
            scene["description"].as_str().unwrap()
        ));
        expansions[i] = scene.to_string();
    }
    let (code, report) = submit(&h, id, body(&outline, &expansions, planned(), &revision)).await;
    assert_eq!(code, 200, "{report}");
    assert_eq!(
        report["warnings"],
        json!([
            {"ordinal":2,"code":"i2v_cast_identity_text_only","cast":["keeper"]},
            {"ordinal":3,"code":"continuity_source_pending","source_scene":format!("scene_{id}_1")},
            {"ordinal":4,"code":"continuity_source_pending","source_scene":format!("scene_{id}_2")}
        ])
    );
    let boundaries = report["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["boundary"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        boundaries,
        [
            "hard_cut",
            "hard_cut",
            "continuous_shot",
            "continuous_shot",
            "hard_cut"
        ]
    );
    stop(h).await;
}
#[tokio::test]
async fn presets_apply_their_model_file_settings_and_loras_and_errors_name_the_scene() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "external-presets").await;
    let presets = root.path().join("video-presets.yaml");
    std::fs::write(
        &presets,
        "schema_version: 1\npresets:\n  - key: Own tuned ref2v\n    variant: reference\n    description: Own fine-tune\n    settings:\n      components:\n        model: diffusion_models/own_tuned.safetensors\n      stage_one:\n        steps: 24\n    loras:\n      - name: h3/own_style.safetensors\n        strength_model: 0.8\n  - key: Own broken\n    variant: reference\n    settings:\n      stage_one:\n        steps: 0\n",
    )
    .unwrap();
    h.control.video.lock().await.video_presets = Some(presets);
    let (code, capabilities) = call(&h, "GET", "/v1/video/capabilities", None).await;
    assert_eq!(code, 200, "{capabilities}");
    let listed = capabilities["authoring"]["presets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            (
                p["key"].as_str().unwrap(),
                p["available"].as_bool().unwrap(),
                p["problem"].clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        listed,
        vec![
            ("Default ref2v", true, Value::Null),
            ("Default i2v", true, Value::Null),
            ("Own tuned ref2v", true, Value::Null),
            ("Own broken", false, json!("preset_settings_invalid")),
        ]
    );
    assert!(!capabilities.to_string().contains("safetensors"));
    let (project, images) = production_project(&h).await;
    let mut request = external(&h, &project, "nsfw").await;
    request["images"] = images;
    request["duration_seconds"] = json!(25);
    let d = start(&h, &project, request).await;
    let id = d["id"].as_str().unwrap();
    let revision = project_revision(&h, &project).await;
    let (outline, expansions) = story();
    let mut scenes = planned()["scenes"].as_array().unwrap().clone();
    scenes[0] = json!({"preset":"Own tuned ref2v","continuity":null,"rationale":"Own note"});
    scenes[1]["preset"] = json!("Default i2v");
    let production = |scenes: &[Value]| json!({"scenes":scenes});
    for (scene, code) in [
        (json!({"preset":"Own missing"}), "preset_unknown"),
        (json!({"preset":"Own broken"}), "preset_settings_invalid"),
        (
            json!({"preset":"Own tuned ref2v","model":"i2v"}),
            "production_preset_variant_mismatch",
        ),
        (json!({"rationale":"Own note"}), "production_model_required"),
    ] {
        let mut wrong = scenes.clone();
        wrong[0] = scene;
        let (status, error) = submit(
            &h,
            id,
            body(&outline, &expansions, production(&wrong), &revision),
        )
        .await;
        assert_eq!(status, 422, "{error}");
        assert_eq!(error["error"]["code"], code);
        assert_eq!(error["error"]["at"], "production.scenes[0]");
    }
    let (code, report) = submit(
        &h,
        id,
        body(&outline, &expansions, production(&scenes), &revision),
    )
    .await;
    assert_eq!(code, 200, "{report}");
    let reported = report["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["preset"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        reported,
        [
            "Own tuned ref2v",
            "Default i2v",
            "Default i2v",
            "Default ref2v",
            "Default ref2v"
        ]
    );
    let mut commit = body(&outline, &expansions, production(&scenes), &revision);
    commit["dry_run"] = json!(false);
    let (code, applied) = submit(&h, id, commit).await;
    assert_eq!(code, 200, "{applied}");
    let scenes = applied["project"]["scenes"].as_array().unwrap();
    let tuned = &scenes[0];
    assert_eq!(tuned["model"], "project_default");
    let values = &tuned["settings"][0]["values"];
    assert_eq!(
        values["components"]["model"],
        "diffusion_models/own_tuned.safetensors"
    );
    assert_eq!(values["stage_one"]["steps"], 24);
    // Unnamed values keep the project's settings for that model.
    assert_eq!(values["stage_one"]["sampler"], "res_multistep");
    let loras = &applied["project"]["creative"]["scene_loras"][tuned["id"].as_str().unwrap()];
    assert_eq!(loras["merge_mode"], "Add To Global");
    // MCP records the preset with the same fingerprint the UI would.
    assert_eq!(
        applied["project"]["creative"]["scene_presets"][tuned["id"].as_str().unwrap()]["key"],
        "Own tuned ref2v"
    );
    let (_, snapshot) = call(&h, "GET", &format!("/v1/video/projects/{project}"), None).await;
    for scene in scenes {
        assert_eq!(
            snapshot["preset_states"][scene["id"].as_str().unwrap()],
            "applied",
            "{snapshot}"
        );
    }
    assert_eq!(
        loras["main"]["loras"][0]["name"],
        "h3/own_style.safetensors"
    );
    // Default presets keep today's behaviour: no own settings slot or LoRAs.
    assert_eq!(scenes[1]["model"]["override"]["variant"], "i2v");
    assert_eq!(scenes[1]["settings"], json!([]));
    assert!(
        applied["project"]["creative"]["scene_loras"][scenes[3]["id"].as_str().unwrap()].is_null()
    );
    stop(h).await;
}
#[tokio::test]
async fn ui_presets_plan_undoable_edits_track_modification_and_set_project_defaults() {
    let root = tempfile::tempdir().unwrap();
    let h = setup(root.path(), "ui-presets").await;
    let presets = root.path().join("video-presets.yaml");
    std::fs::write(
        &presets,
        "schema_version: 1\npresets:\n  - key: Own tuned ref2v\n    variant: reference\n    settings:\n      components:\n        model: diffusion_models/own_tuned.safetensors\n      stage_one:\n        steps: 24\n    loras:\n      - name: h3/own_style.safetensors\n        strength_model: 0.8\n",
    )
    .unwrap();
    h.control.video.lock().await.video_presets = Some(presets);
    let (_, catalog) = call(&h, "GET", "/v1/video/presets", None).await;
    let tuned = catalog["presets"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["key"] == "Own tuned ref2v")
        .unwrap()
        .clone();
    assert_eq!(
        tuned["settings"]["components"]["model"],
        "diffusion_models/own_tuned.safetensors"
    );
    assert_eq!(tuned["lora_merge"], "Add To Global");
    let (project, _) = production_project(&h).await;
    let path = format!("/v1/video/projects/{project}");
    // Two scenes to apply the preset to.
    let (_, snapshot) = call(&h, "GET", &path, None).await;
    let revision = snapshot["project"]["project_revision"].clone();
    let add = |id: &str, start: u64| {
        json!({"operation":"add_scene","scene":{"id":id,"start":{"num":start,"den":1},"duration":{"num":5,"den":1},
            "description":"Own scene","model":"project_default","settings":[],"sections":[{"id":format!("{id}_s"),"start":{"num":0,"den":1},"duration":{"num":5,"den":1},"prompt":"Own beat"}],
            "bindings":[],"continuity":null,"accepted_take":null}})
    };
    let (code, added) = call(&h, "POST", &format!("{path}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":revision,"commands":[add("own_a",0),add("own_b",5)]}))).await;
    assert_eq!(code, 200, "{added}");
    let plan_path = format!("{path}/presets/plan");
    let plan = |body: Value| call(&h, "POST", &plan_path, Some(body));
    let revision = added["project"]["project_revision"].clone();
    let (code, request) = plan(
        json!({"schema_version":1,"expected_project_revision":revision,
        "preset":"Own tuned ref2v","scenes":["own_a","own_b"]}),
    )
    .await;
    assert_eq!(code, 200, "{request}");
    // Planning changes nothing; the UI submits the batch through the edit path.
    let (_, unchanged) = call(&h, "GET", &path, None).await;
    assert_eq!(unchanged["project"]["project_revision"], revision);
    let (code, applied) = call(&h, "POST", &format!("{path}/edits"), Some(request)).await;
    assert_eq!(code, 200, "{applied}");
    for id in ["own_a", "own_b"] {
        assert_eq!(applied["preset_states"][id], "applied", "{applied}");
        let scene = applied["project"]["scenes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["id"] == id)
            .unwrap();
        assert_eq!(scene["settings"][0]["values"]["stage_one"]["steps"], 24);
        assert_eq!(
            applied["project"]["creative"]["scene_loras"][id]["merge_mode"],
            "Add To Global"
        );
    }
    // Changing a scene's settings by hand marks it Modified; Reapply restores it.
    let mut slot = applied["project"]["scenes"][applied["project"]["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|s| s["id"] == "own_a")
        .unwrap()]["settings"][0]
        .clone();
    slot["values"]["stage_one"]["steps"] = json!(30);
    let (code, edited) = call(&h, "POST", &format!("{path}/edits"),
        Some(json!({"schema_version":1,"expected_project_revision":applied["project"]["project_revision"],
            "commands":[{"operation":"set_model_settings","scene_id":"own_a","settings":slot}]}))).await;
    assert_eq!(code, 200, "{edited}");
    assert_eq!(edited["preset_states"]["own_a"], "modified");
    assert_eq!(edited["preset_states"]["own_b"], "applied");
    let (_, reapply) = plan(
        json!({"schema_version":1,"expected_project_revision":edited["project"]["project_revision"],
        "preset":"Own tuned ref2v","scenes":["own_a"]}),
    )
    .await;
    let (_, restored) = call(&h, "POST", &format!("{path}/edits"), Some(reapply)).await;
    assert_eq!(restored["preset_states"]["own_a"], "applied");
    // A default preset inherits project settings and drops the scene LoRAs.
    let (_, default) = plan(json!({"schema_version":1,"expected_project_revision":restored["project"]["project_revision"],
        "preset":"Default ref2v","scenes":["own_b"]})).await;
    let (_, plain) = call(&h, "POST", &format!("{path}/edits"), Some(default)).await;
    let b = plain["project"]["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["id"] == "own_b")
        .unwrap();
    assert_eq!(b["settings"], json!([]));
    assert!(
        plain["project"]["creative"]["scene_loras"]
            .get("own_b")
            .is_none()
    );
    assert_eq!(
        plain["project"]["creative"]["scene_presets"]["own_b"]["key"],
        "Default ref2v"
    );
    // Project defaults: mode, the project's settings for it, and the global LoRAs.
    let (_, request) = plan(
        json!({"schema_version":1,"expected_project_revision":plain["project"]["project_revision"],
        "preset":"Own tuned ref2v","project":true}),
    )
    .await;
    let (code, defaults) = call(&h, "POST", &format!("{path}/edits"), Some(request)).await;
    assert_eq!(code, 200, "{defaults}");
    let slot = defaults["project"]["generation"]["settings"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["model"]["variant"] == "reference")
        .unwrap()
        .clone();
    assert_eq!(
        slot["values"]["components"]["model"],
        "diffusion_models/own_tuned.safetensors"
    );
    assert_eq!(
        defaults["project"]["creative"]["h3_loras"]["loras"][0]["name"],
        "h3/own_style.safetensors"
    );
    // Errors.
    let revision = defaults["project"]["project_revision"].clone();
    for (body, status, code) in [
        (
            json!({"schema_version":1,"expected_project_revision":revision,"preset":"Own missing","scenes":["own_a"]}),
            404,
            "preset_unknown",
        ),
        (
            json!({"schema_version":1,"expected_project_revision":0,"preset":"Default ref2v","scenes":["own_a"]}),
            409,
            "revision_conflict",
        ),
        (
            json!({"schema_version":1,"expected_project_revision":revision,"preset":"Default ref2v"}),
            422,
            "invalid_request",
        ),
        (
            json!({"schema_version":1,"expected_project_revision":revision,"preset":"Default ref2v","scenes":["own_missing"]}),
            422,
            "scene_not_found",
        ),
    ] {
        let (got, error) = plan(body).await;
        assert_eq!(
            (got, error["error"]["code"].as_str().unwrap()),
            (status, code),
            "{error}"
        );
    }
    stop(h).await;
}
