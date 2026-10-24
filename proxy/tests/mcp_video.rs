#[path = "support/destination.rs"]
mod destination;
use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::get,
};
use serde_json::{Value, json};
use st_proxy::{config::Config, mcp::Mcp};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    session: String,
    key: Option<String>,
    body: Value,
}
#[derive(Default)]
struct Probe {
    session: Mutex<String>,
    seen: Mutex<Vec<Seen>>,
    run_polls: AtomicUsize,
    image: Mutex<(PathBuf, String, usize)>,
}
impl Probe {
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
    fn routes(&self) -> Vec<String> {
        self.seen()
            .into_iter()
            .map(|s| format!("{} {}", s.method, s.path))
            .collect()
    }
    fn clear(&self) {
        self.seen.lock().unwrap().clear();
    }
}
fn time(n: u64) -> Value {
    json!({"num":n,"den":1})
}
fn snapshot(revision: u64, attached: bool) -> Value {
    let mut assets = json!({"project_asset_old":{"id":"project_asset_old","name":"old.png","revision":1,"kind":"image"}});
    let mut runtime = json!({"project_asset_old":"session_old"});
    if attached {
        assets["project_asset_new"] =
            json!({"id":"project_asset_new","name":"own.png","revision":1,"kind":"image"});
        runtime["project_asset_new"] = json!("session_asset_1");
    }
    json!({"schema_version":1,"storage_layout":1,
    "project":{"schema_version":1,"id":"p1","project_revision":revision,"default_model":{"adapter":"h3"},
        "delivery":{"fps":time(24),"width":1280,"height":720},"global_prompt":"GLOBAL SYNTHETIC",
        "creative":{"references":[{"id":"reference_x","name":"Keeper","alias":"keeper","enabled":true,"description":"LONG CAST TEXT",
            "images":[{"id":"i1","asset":"project_asset_old","enabled":true,"preview":true}]}],
            "sections":{"s1":{"images":[]}},"scene_prompts":{"scene_1":{"title":"Opening","summary":"SUMMARY TEXT"}}},
        "assets":assets,
        "scenes":[{"id":"scene_1","start":time(0),"duration":time(5),"description":"Scene one","model":"project_default",
            "settings":[{"values":"SETTINGS BLOB"}],"sections":[{"id":"s1","start":time(0),"duration":time(5),"prompt":"Section prompt"}],
            "bindings":[{"id":"binding_blob"}],"continuity":null,
            "accepted_take":{"id":"take_1","model":null,"source_fingerprint":"FINGERPRINT"}}]},
    "resources":[{"id":"resource_1","source":"/private/original/path.png","sha256":"RESOURCE HASH"}],
    "takes":[{"id":"take_1","scene_id":"scene_1","status":"accepted","provenance":{"seed":"PROVENANCE"}}],
    "deliveries":[{"id":"delivery_1","asset_id":"asset_final","plan":{"blob":"DELIVERY PLAN"}}],
    "authoring":[{"id":"d1","phase":"review","request":{"brief":"BRIEF TEXT"}}],
    "persistence":{"state":"saved","saved_revision":revision,"error_code":null},
    "runtime_assets":runtime,"runtime_takes":{"take_1":"session_take"},
    "take_states":{"take_1":{"accepted":true,"outdated":false}}})
}
fn draft() -> Value {
    json!({"schema_version":1,"id":"d1","project_id":"p1","phase":"outline",
    "bundle":{"outline":{"system":"VERBATIM"},"images":[
        {"image_id":"img_a","role":"story","usage":"setting_reference","label":"Lobby","mime":"image/jpeg","base64":"QUJD"},
        {"image_id":"cast_keeper","role":"identity","usage":"identity","label":"@keeper","mime":"image/png","base64":"REVG"}]}})
}
async fn imagine(
    State(p): State<Arc<Probe>>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let body: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let header = |k: &str| headers.get(k).map(|v| v.to_str().unwrap().to_owned());
    p.seen.lock().unwrap().push(Seen {
        method: method.to_string(),
        path: uri.path().into(),
        session: header("X-Imagine-Session").unwrap_or_default(),
        key: header("Idempotency-Key"),
        body: body.clone(),
    });
    let ok = |v: Value| (StatusCode::OK, Json(v)).into_response();
    let fail =
        |status: u16, v: Value| (StatusCode::from_u16(status).unwrap(), Json(v)).into_response();
    match (method.as_str(), uri.path()) {
        ("GET", "/v1/video/authoring/profiles") => ok(json!({"profiles":[{"id":"nsfw"}]})),
        ("GET", "/v1/video/authoring/profiles/nsfw") => {
            ok(json!({"id":"nsfw","guidance":"SHARED"}))
        }
        ("GET", "/v1/video/authoring/profiles/missing") => {
            fail(404, json!({"error":{"code":"profile_not_found"}}))
        }
        ("GET", "/v1/video/capabilities") => ok(json!({"schema_version":1,"models":[]})),
        ("POST", "/v1/video/locations") => ok(json!({"location_id":"location_1"})),
        ("POST", "/v1/video/projects/at-location") => ok(snapshot(1, false)),
        ("GET", "/v1/video/projects/p1") => ok(snapshot(1, false)),
        ("GET", "/v1/video/projects/slow") => {
            tokio::time::sleep(Duration::from_secs(7)).await;
            ok(snapshot(1, false))
        }
        ("POST", "/v1/video/projects/slow/authoring") => {
            tokio::time::sleep(Duration::from_secs(6)).await;
            ok(draft())
        }
        ("POST", "/v1/video/projects/p1/edits") => {
            if body["commands"][0]["operation"] == "upsert_visual_reference" {
                assert_eq!(body["expected_project_revision"], 2);
                ok(snapshot(3, true))
            } else {
                ok(snapshot(2, false))
            }
        }
        ("POST", "/v1/video/projects/bad/edits") => fail(
            409,
            json!({"error":{"code":"revision_conflict","at":"commands[0].scene_id","detail":"Reload the project","extra":"LEAK"},
            "snapshot":snapshot(9, false)}),
        ),
        ("POST", "/v1/video/projects/odd/edits") => fail(
            422,
            json!({"error":{"code":"Not A Code /x","at":"has space","detail":"bad\u{1}detail"}}),
        ),
        ("POST", "/v1/video/projects/flat/edits") => fail(
            422,
            json!({"error":"story_expansion_count_mismatch","at":"expansions[3]","detail":"3 of 4"}),
        ),
        ("POST", "/v1/video/projects/p1/save") => ok(snapshot(2, false)),
        ("GET", "/v1/video/runs/run_failed") => ok(json!({"id":"run_failed","status":"failed",
            "failure":"run_job_failed","current_job":"job_failed",
            "targets":[{"scene_id":"s1","job_id":"job_ok","completed_take_id":"t1"},{"scene_id":"s2","job_id":"job_failed","completed_take_id":null}]})),
        ("GET", "/v1/jobs/job_failed") => ok(json!({"id":"job_failed","status":"failed",
            "error":{"code":"h3_generation_failed","stage":"sampling","kind":"shape_mismatch","exception":"RuntimeError",
                "sizes":[7,8],"message":"PRIVATE /home/x prompt","stage_text":"LEAK"}})),
        ("POST", "/v1/video/assets/link") => {
            let (path, digest, bytes) = p.image.lock().unwrap().clone();
            assert_eq!(body, json!({"path":path}));
            ok(
                json!({"id":"session_asset_1","mime_type":"image/png","sha256":digest,"bytes":bytes}),
            )
        }
        ("POST", "/v1/video/projects/p1/assets") => ok(snapshot(2, true)),
        ("POST", "/v1/video/projects/p1/authoring") => ok(draft()),
        ("POST", "/v1/video/authoring/d1/submission") => {
            ok(json!({"valid":true,"scene_map":[],"snapshot":snapshot(4, false)}))
        }
        ("GET" | "POST", "/v1/video/authoring/d1" | "/v1/video/authoring/d1/review") => {
            ok(json!({"id":"d1","phase":"review"}))
        }
        ("POST", "/v1/video/runs") => ok(json!({"id":"run_1","status":"running"})),
        ("GET", "/v1/video/runs/run_1") => {
            let n = p.run_polls.fetch_add(1, Ordering::SeqCst);
            ok(json!({"id":"run_1","status":if n < 2 {"running"} else {"completed"}}))
        }
        ("POST", "/v1/video/runs/run_1/stop") => ok(json!({"id":"run_1","status":"stopping"})),
        ("POST", "/v1/video/projects/p1/assembly") => ok(json!({"id":"assembly_1"})),
        ("POST", "/v1/video/assemblies/assembly_1/jobs") => {
            ok(json!({"id":"job_9","reused":false}))
        }
        ("GET", "/v1/video/assemblies/assembly_1") => ok(
            json!({"plan":{"id":"assembly_1"},"progress":{"phase":"done","completed":1,"total":1},"job_id":"job_9","rendered_result_available":false}),
        ),
        ("GET", "/v1/jobs/job_9") => ok(
            json!({"id":"job_9","status":"succeeded","request":{"secret":"REQUEST BODY"},"error":null,
            "output":{"format":"mp4","saved_path":"/own/final.mp4","persistent":true,
                "assets":[{"id":"asset_final","role":"video","mime_type":"video/mp4","bytes":10,"sha256":"ab"}],
                "video":{"duration_seconds":5.0},"delivery":{"blob":"DELIVERY PLAN"}}}),
        ),
        _ => fail(404, json!({"error":{"code":"unexpected_route"}})),
    }
}
struct Harness {
    mcp: Mcp,
    probe: Arc<Probe>,
    dir: tempfile::TempDir,
    server: tokio::task::JoinHandle<()>,
}
impl Harness {
    async fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let published = dir.path().join("outputs/images");
        std::fs::create_dir_all(&published).unwrap();
        std::fs::create_dir_all(dir.path().join("projects")).unwrap();
        let proof = destination::proof(&published);
        let probe = Arc::new(Probe::default());
        *probe.session.lock().unwrap() = "session-a".into();
        let session = probe.clone();
        let router = Router::new()
            .route(
                "/v1/media-destination",
                get(move || {
                    let p = proof.clone();
                    async move { Json(p) }
                }),
            )
            .route(
                "/v1/session",
                get(move || {
                    let id = session.session.lock().unwrap().clone();
                    async move { Json(json!({"id":id})) }
                }),
            )
            .fallback(imagine)
            .with_state(probe.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        let mcp = Mcp::new(Config {
            imagine_url: url,
            coordinator_socket: Some(dir.path().join("gpu.sock").to_string_lossy().into()),
            mcp_input_dir: dir.path().to_string_lossy().into(),
            mcp_output_dir: Some(dir.path().join("old-cache").to_string_lossy().into()),
            poll_interval: 0.01,
            ..Default::default()
        })
        .await
        .unwrap();
        Self {
            mcp,
            probe,
            dir,
            server,
        }
    }
    fn root(&self) -> PathBuf {
        self.dir.path().canonicalize().unwrap()
    }
    async fn call(&self, name: &str, args: Value) -> Value {
        let args = st_proxy::mcp_protocol::tool_arguments(name, &args).unwrap();
        self.mcp.call(name, &args).await
    }
    async fn close(self) {
        self.mcp.close().await;
        self.server.abort();
        let _ = self.server.await;
    }
}
fn ok(result: &Value) -> &Value {
    assert_eq!(result["isError"], false, "{result}");
    &result["structuredContent"]
}
fn fault(result: &Value) -> &Value {
    assert_eq!(result["isError"], true, "{result}");
    &result["structuredContent"]["error"]
}
fn png(path: &Path) -> (String, usize) {
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(8, 8)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    std::fs::write(path, png.get_ref()).unwrap();
    (
        st_proxy::references::hash(png.get_ref()),
        png.get_ref().len(),
    )
}

#[tokio::test]
async fn every_video_tool_and_action_routes_to_its_imagine_endpoint() {
    let h = Harness::new().await;
    let source = h.root().join("own-cast.png");
    let (digest, bytes) = png(&source);
    *h.probe.image.lock().unwrap() = (source.clone(), digest, bytes);
    let project = h.root().join("projects/own-film");
    let cases: Vec<(&str, Value, Vec<&str>)> = vec![
        (
            "list_video_profiles",
            json!({}),
            vec!["GET /v1/video/authoring/profiles"],
        ),
        (
            "describe_video_profile",
            json!({"profile":"nsfw"}),
            vec!["GET /v1/video/authoring/profiles/nsfw"],
        ),
        (
            "video_capabilities",
            json!({}),
            vec!["GET /v1/video/capabilities"],
        ),
        (
            "video_project",
            json!({"action":"create","path":project}),
            vec![
                "POST /v1/video/locations",
                "POST /v1/video/projects/at-location",
            ],
        ),
        (
            "video_project",
            json!({"action":"get","project_id":"p1"}),
            vec!["GET /v1/video/projects/p1"],
        ),
        (
            "video_project",
            json!({"action":"edit","project_id":"p1","request":{"expected_project_revision":1,"commands":[{"operation":"set_global_prompt","prompt":"x"}]}}),
            vec!["POST /v1/video/projects/p1/edits"],
        ),
        (
            "video_project",
            json!({"action":"save","project_id":"p1","expected_project_revision":2}),
            vec!["POST /v1/video/projects/p1/save"],
        ),
        (
            "video_register_reference",
            json!({"project_id":"p1","path":source,"expected_project_revision":1,"cast":{"name":"Keeper","alias":"@Keeper","description":"Tall"}}),
            vec![
                "POST /v1/video/assets/link",
                "GET /v1/video/projects/p1",
                "POST /v1/video/projects/p1/assets",
                "POST /v1/video/projects/p1/edits",
            ],
        ),
        (
            "video_storyboard",
            json!({"action":"start","project_id":"p1","request":{"expected_project_revision":3,"provider":"koboldcpp","provider_profile_id":"local","operation":"enhance_scene","brief":"b"}}),
            vec!["POST /v1/video/projects/p1/authoring"],
        ),
        (
            "video_storyboard",
            json!({"action":"submit","draft_id":"d1","submission":{"expected_project_revision":3,"dry_run":true}}),
            vec!["POST /v1/video/authoring/d1/submission"],
        ),
        (
            "video_storyboard",
            json!({"action":"get","draft_id":"d1"}),
            vec!["GET /v1/video/authoring/d1"],
        ),
        (
            "video_storyboard",
            json!({"action":"review","draft_id":"d1","review":{"outline":null}}),
            vec!["POST /v1/video/authoring/d1/review"],
        ),
        (
            "video_run",
            json!({"action":"start","job_id":"render-1","request":{"project_id":"p1","expected_project_revision":4,"mode":"missing_only","selected_scene":null}}),
            vec!["POST /v1/video/runs"],
        ),
        (
            "video_run",
            json!({"action":"get","run_id":"run_1"}),
            vec!["GET /v1/video/runs/run_1"],
        ),
        (
            "video_run",
            json!({"action":"stop","run_id":"run_1","cancel_current":true}),
            vec!["POST /v1/video/runs/run_1/stop"],
        ),
        (
            "video_assembly",
            json!({"action":"plan","project_id":"p1","request":{"options":{}}}),
            vec!["POST /v1/video/projects/p1/assembly"],
        ),
        (
            "video_assembly",
            json!({"action":"submit","plan_id":"assembly_1","job_id":"assemble-1"}),
            vec!["POST /v1/video/assemblies/assembly_1/jobs"],
        ),
        (
            "video_assembly",
            json!({"action":"get","plan_id":"assembly_1"}),
            vec!["GET /v1/video/assemblies/assembly_1", "GET /v1/jobs/job_9"],
        ),
    ];
    for (name, args, routes) in cases {
        h.probe.clear();
        let result = h.call(name, args.clone()).await;
        ok(&result);
        assert_eq!(h.probe.routes(), routes, "{name} {args}");
        assert!(
            h.probe.seen().iter().all(|s| s.session == "session-a"),
            "{name}"
        );
    }
    // Request bodies: forced external authoring, namespaced keys, typed cast edit.
    h.probe.clear();
    h.call(
        "video_project",
        json!({"action":"create","path":project,"project":{"schema_version":1}}),
    )
    .await;
    let seen = h.probe.seen();
    assert_eq!(
        seen[0].body,
        json!({"schema_version":1,"path":project,"create":true})
    );
    assert_eq!(
        seen[1].body,
        json!({"schema_version":1,"location_id":"location_1","project":{"schema_version":1},"canvas":null})
    );
    h.probe.clear();
    // Without project or canvas, MCP productions default to Draft.
    h.call("video_project", json!({"action":"create","path":project}))
        .await;
    assert_eq!(
        h.probe.seen()[1].body,
        json!({"schema_version":1,"location_id":"location_1","project":null,
            "canvas":{"ratio":"16:9","portrait":false,"quality":"Draft"}})
    );
    h.probe.clear();
    let canvas = json!({"ratio":"16:9","portrait":false,"quality":"Standard"});
    h.call(
        "video_project",
        json!({"action":"create","path":project,"canvas":canvas}),
    )
    .await;
    assert_eq!(
        h.probe.seen()[1].body,
        json!({"schema_version":1,"location_id":"location_1","project":null,"canvas":canvas})
    );
    h.probe.clear();
    let both = h
        .call(
            "video_project",
            json!({"action":"create","path":project,"project":{"schema_version":1},"canvas":canvas}),
        )
        .await;
    assert_eq!(fault(&both)["category"], "validation");
    assert!(h.probe.seen().is_empty());
    h.call("video_storyboard", json!({"action":"start","project_id":"p1","request":{"expected_project_revision":3,"provider":"koboldcpp","provider_profile_id":"local","operation":"enhance_scene","brief":"b"}})).await;
    assert_eq!(
        h.probe.seen()[0].body,
        json!({"expected_project_revision":3,"provider":"external","provider_profile_id":"","operation":"outline","brief":"b","schema_version":1})
    );
    h.probe.clear();
    let registered = h.call("video_register_reference", json!({"project_id":"p1","path":source,"expected_project_revision":1,"name":"Keeper face","cast":{"name":"Keeper","alias":"@Keeper","description":"Tall"}})).await;
    let registered = ok(&registered);
    assert_eq!(registered["asset_id"], "project_asset_new");
    assert_eq!(registered["project_revision"], 3);
    assert_eq!(registered["cast"]["alias"], "Keeper");
    assert_eq!(registered["cast"]["tag"], "@Keeper");
    let seen = h.probe.seen();
    assert_eq!(
        seen[2].body,
        json!({"schema_version":1,"expected_project_revision":1,"asset_id":"session_asset_1","name":"Keeper face"})
    );
    let command = &seen[3].body["commands"][0];
    assert_eq!(command["operation"], "upsert_visual_reference");
    let reference = &command["reference"];
    assert_eq!(reference["id"], registered["cast"]["reference_id"]);
    assert_eq!(reference["alias"], "Keeper");
    assert_eq!(
        reference["images"],
        json!([{"id":reference["images"][0]["id"],"asset":"project_asset_new","enabled":true,"preview":true}])
    );
    // The same alias always maps to the same reference, so a re-registration replaces it.
    h.probe.clear();
    let again = h.call("video_register_reference", json!({"project_id":"p1","path":source,"expected_project_revision":1,"cast":{"name":"Keeper","alias":"Keeper"}})).await;
    assert_eq!(
        ok(&again)["cast"]["reference_id"],
        registered["cast"]["reference_id"]
    );
    assert_eq!(ok(&again)["name"], "own-cast.png");
    // A projection never carries resources, provenance, settings or full creative blobs.
    let view = h
        .call("video_project", json!({"action":"get","project_id":"p1"}))
        .await;
    let view = ok(&view);
    assert_eq!(view["project_id"], "p1");
    assert_eq!(view["scenes"][0]["title"], "Opening");
    assert_eq!(view["scenes"][0]["sections"][0]["prompt"], "Section prompt");
    assert_eq!(view["scenes"][0]["accepted_take"]["id"], "take_1");
    assert_eq!(view["persistence"]["state"], "saved");
    assert_eq!(view["cast"][0]["alias"], "keeper");
    let text = view.to_string();
    for private in [
        "/private/original",
        "RESOURCE HASH",
        "FINGERPRINT",
        "SETTINGS BLOB",
        "binding_blob",
        "PROVENANCE",
        "DELIVERY PLAN",
        "BRIEF TEXT",
        "LONG CAST TEXT",
        "SUMMARY TEXT",
        "session_old",
    ] {
        assert!(!text.contains(private), "{private} in {text}");
    }
    let submitted = h.call("video_storyboard", json!({"action":"submit","draft_id":"d1","submission":{"expected_project_revision":3,"dry_run":false}})).await;
    let submitted = ok(&submitted);
    assert_eq!(submitted["valid"], true);
    assert_eq!(submitted["snapshot"]["project_revision"], 4);
    assert!(!submitted.to_string().contains("RESOURCE HASH"));
    let assembly = h
        .call(
            "video_assembly",
            json!({"action":"get","plan_id":"assembly_1"}),
        )
        .await;
    let assembly = ok(&assembly);
    assert_eq!(assembly["terminal"], true);
    assert_eq!(assembly["job"]["status"], "succeeded");
    assert_eq!(assembly["job"]["output"]["saved_path"], "/own/final.mp4");
    assert!(!assembly.to_string().contains("REQUEST BODY"));
    assert!(!assembly.to_string().contains("DELIVERY PLAN"));
    h.probe.clear();
    h.call(
        "video_assembly",
        json!({"action":"plan","project_id":"p1","request":{"options":{}}}),
    )
    .await;
    assert_eq!(
        h.probe.seen()[0].body,
        json!({"options":{},"schema_version":1,"project_id":"p1"})
    );
    let unknown = h
        .call("video_run", json!({"action":"restart","run_id":"run_1"}))
        .await;
    assert_eq!(fault(&unknown)["category"], "validation");
    h.close().await;
}

#[tokio::test]
async fn storyboard_bundle_images_become_image_blocks_without_base64() {
    let h = Harness::new().await;
    for (action, args) in [
        (
            "start",
            json!({"project_id":"p1","request":{"expected_project_revision":1}}),
        ),
        ("get", json!({"draft_id":"d1"})),
    ] {
        let mut args = args;
        args["action"] = json!(action);
        let result = h.call("video_storyboard", args).await;
        if action == "get" {
            assert_eq!(result["content"].as_array().unwrap().len(), 1);
            continue;
        }
        let payload = ok(&result);
        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 3);
        assert_eq!(content[0]["type"], "text");
        assert!(!content[0]["text"].as_str().unwrap().contains("QUJD"));
        assert_eq!(
            content[1],
            json!({"type":"image","data":"QUJD","mimeType":"image/jpeg"})
        );
        assert_eq!(
            content[2],
            json!({"type":"image","data":"REVG","mimeType":"image/png"})
        );
        let images = payload["bundle"]["images"].as_array().unwrap();
        assert_eq!(
            images[0],
            json!({"image_id":"img_a","role":"story","usage":"setting_reference","label":"Lobby","mime":"image/jpeg","index":0})
        );
        assert_eq!(images[1]["index"], 1);
        assert_eq!(images[1]["label"], "@keeper");
        assert!(!payload.to_string().contains("base64"));
        assert_eq!(payload["bundle"]["outline"]["system"], "VERBATIM");
    }
    h.close().await;
}

#[tokio::test]
async fn idempotency_keys_are_namespaced_by_session_and_follow_a_session_reset() {
    let h = Harness::new().await;
    let start = json!({"action":"start","job_id":"render-1","request":{"project_id":"p1","expected_project_revision":4,"mode":"missing_only","selected_scene":null}});
    ok(&h.call("video_run", start.clone()).await);
    ok(&h.call("video_run", start.clone()).await);
    ok(&h
        .call(
            "video_assembly",
            json!({"action":"submit","plan_id":"assembly_1","job_id":"assemble-1"}),
        )
        .await);
    let keys = h
        .probe
        .seen()
        .into_iter()
        .map(|s| s.key)
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        [
            Some("mcp-session-a-render-1".into()),
            Some("mcp-session-a-render-1".into()),
            Some("mcp-session-a-assemble-1".into())
        ]
    );
    // Reads and non-admission writes never carry a key.
    h.probe.clear();
    ok(&h
        .call("video_run", json!({"action":"get","run_id":"run_1"}))
        .await);
    assert_eq!(h.probe.seen()[0].key, None);
    for bad in ["../x", "", "has space", &"x".repeat(65)] {
        let mut args = start.clone();
        args["job_id"] = json!(bad);
        h.probe.clear();
        assert_eq!(
            fault(&h.call("video_run", args).await)["category"],
            "validation"
        );
        assert!(h.probe.seen().is_empty());
    }
    // A new Imagine session discards proxy session state; video keys and headers follow it.
    *h.probe.session.lock().unwrap() = "session-b".into();
    h.probe.clear();
    ok(&h.call("video_run", start).await);
    let seen = h.probe.seen();
    assert_eq!(seen[0].session, "session-b");
    assert_eq!(seen[0].key.as_deref(), Some("mcp-session-b-render-1"));
    h.close().await;
}

#[tokio::test]
async fn imagine_errors_forward_only_stable_code_field_path_and_detail() {
    let h = Harness::new().await;
    let edit = |project: &str| json!({"action":"edit","project_id":project,"request":{"expected_project_revision":1,"commands":[]}});
    let result = h.call("video_project", edit("bad")).await;
    assert_eq!(
        result["structuredContent"],
        json!({"job_id":null,"error":{"category":"conflict","message":"Imagine rejected the request: revision_conflict",
        "code":"revision_conflict","at":"commands[0].scene_id","detail":"Reload the project"}})
    );
    let text = result.to_string();
    for leak in ["LEAK", "GLOBAL SYNTHETIC", "Section prompt", "snapshot"] {
        assert!(!text.contains(leak), "{leak}");
    }
    let odd = h.call("video_project", edit("odd")).await;
    assert_eq!(
        fault(&odd).clone(),
        json!({"category":"validation","message":"Imagine rejected the request: operation_rejected","code":"operation_rejected"})
    );
    let flat = h.call("video_project", edit("flat")).await;
    assert_eq!(fault(&flat)["code"], "story_expansion_count_mismatch");
    assert_eq!(fault(&flat)["at"], "expansions[3]");
    assert_eq!(fault(&flat)["detail"], "3 of 4");
    let missing = h
        .call("describe_video_profile", json!({"profile":"missing"}))
        .await;
    assert_eq!(fault(&missing)["category"], "not_found");
    assert_eq!(fault(&missing)["code"], "profile_not_found");
    h.close().await;
}

#[tokio::test]
async fn project_folders_and_references_stay_inside_mcp_roots() {
    let h = Harness::new().await;
    let root = h.root();
    let outside = tempfile::tempdir().unwrap();
    let outside_path = outside.path().canonicalize().unwrap();
    std::os::unix::fs::symlink(&outside_path, root.join("projects/escape")).unwrap();
    std::fs::write(root.join("projects/file"), b"own").unwrap();
    for path in [
        json!(outside_path.join("film")),
        json!("relative/film"),
        json!(root.join("projects/../projects/film")),
        json!(root.join("missing-parent/film")),
        json!(root),
        json!(root.join("projects/escape")),
        json!(root.join("projects/escape/film")),
        json!(root.join("projects/file")),
    ] {
        let result = h
            .call("video_project", json!({"action":"create","path":path}))
            .await;
        assert_eq!(fault(&result)["category"], "validation", "{path}");
    }
    assert!(h.probe.seen().is_empty());
    // An existing empty folder and a symlinked parent inside the root resolve canonically.
    std::fs::create_dir(root.join("projects/empty")).unwrap();
    std::os::unix::fs::symlink(root.join("projects"), root.join("alias")).unwrap();
    for (path, expected) in [
        (root.join("projects/empty"), root.join("projects/empty")),
        (root.join("alias/new-film"), root.join("projects/new-film")),
    ] {
        h.probe.clear();
        let result = h
            .call("video_project", json!({"action":"create","path":path}))
            .await;
        assert_eq!(ok(&result)["path"], json!(expected));
        assert_eq!(h.probe.seen()[0].body["path"], json!(expected));
    }
    h.probe.clear();
    let outside_image = outside_path.join("own.png");
    png(&outside_image);
    let not_image = root.join("own.txt");
    std::fs::write(&not_image, b"own synthetic text").unwrap();
    for path in [json!(outside_image), json!("own.png"), json!(not_image)] {
        let result = h
            .call(
                "video_register_reference",
                json!({"project_id":"p1","path":path,"expected_project_revision":1}),
            )
            .await;
        assert_eq!(fault(&result)["category"], "validation", "{path}");
    }
    assert!(h.probe.seen().is_empty());
    // A changed file between hashing and linking is rejected before attaching.
    let source = root.join("own.png");
    png(&source);
    *h.probe.image.lock().unwrap() = (source.clone(), "0".repeat(64), 1);
    let changed = h
        .call(
            "video_register_reference",
            json!({"project_id":"p1","path":source,"expected_project_revision":1}),
        )
        .await;
    assert!(
        fault(&changed)["message"]
            .as_str()
            .unwrap()
            .contains("has changed")
    );
    assert_eq!(h.probe.routes(), ["POST /v1/video/assets/link"]);
    h.close().await;
}

#[tokio::test]
async fn video_calls_use_per_action_time_limits_without_holding_the_session_gate() {
    let h = Arc::new(Harness::new().await);
    let started = Instant::now();
    let read = {
        let h = h.clone();
        tokio::spawn(async move {
            h.call("video_project", json!({"action":"get","project_id":"slow"}))
                .await
        })
    };
    let authoring = {
        let h = h.clone();
        tokio::spawn(async move {
            h.call(
                "video_storyboard",
                json!({"action":"start","project_id":"slow","request":{"expected_project_revision":1}}),
            )
            .await
        })
    };
    // A concurrent quick call is not blocked behind the slow ones.
    tokio::time::sleep(Duration::from_millis(200)).await;
    ok(&h.call("video_capabilities", json!({})).await);
    assert!(started.elapsed() < Duration::from_secs(3));
    let read = read.await.unwrap();
    assert_eq!(fault(&read)["category"], "connection");
    let authoring = authoring.await.unwrap();
    assert_eq!(ok(&authoring)["id"], "d1");
    assert!(started.elapsed() >= Duration::from_secs(6));
    assert!(started.elapsed() < Duration::from_secs(20));
    Arc::try_unwrap(h).ok().unwrap().close().await;
}

#[tokio::test]
async fn run_and_assembly_waits_are_bounded_and_stop_at_terminal_status() {
    let h = Harness::new().await;
    let quick = h
        .call("video_run", json!({"action":"get","run_id":"run_1"}))
        .await;
    assert_eq!(ok(&quick)["status"], "running");
    let waited = h
        .call(
            "video_run",
            json!({"action":"get","run_id":"run_1","wait":true,"wait_seconds":5}),
        )
        .await;
    assert_eq!(ok(&waited)["status"], "completed");
    assert_eq!(h.probe.run_polls.load(Ordering::SeqCst), 3);
    for bad in [json!(61), json!(-1)] {
        let result = h
            .call(
                "video_run",
                json!({"action":"get","run_id":"run_1","wait":true,"wait_seconds":bad}),
            )
            .await;
        assert_eq!(fault(&result)["category"], "validation");
    }
    let stop = h
        .call("video_run", json!({"action":"stop","run_id":"run_1"}))
        .await;
    assert_eq!(fault(&stop)["category"], "validation");
    h.close().await;
}

#[test]
fn catalog_schemas_are_strict_objects() {
    let tools: Value = serde_json::from_str(include_str!("../mcp/tools.json")).unwrap();
    let tools = tools.as_array().unwrap();
    let mut names = std::collections::BTreeSet::new();
    for tool in tools {
        let name = tool["name"].as_str().unwrap();
        assert!(names.insert(name), "duplicate {name}");
        assert!(!tool["description"].as_str().unwrap().trim().is_empty());
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object", "{name}");
        assert_eq!(schema["additionalProperties"], false, "{name}");
        let properties = schema["properties"].as_object().unwrap();
        for required in schema["required"].as_array().unwrap() {
            assert!(
                properties.contains_key(required.as_str().unwrap()),
                "{name}"
            );
        }
        for annotation in [
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        ] {
            assert!(tool["annotations"][annotation].is_boolean(), "{name}");
        }
    }
    for name in [
        "list_video_profiles",
        "describe_video_profile",
        "video_capabilities",
        "video_project",
        "video_register_reference",
        "video_storyboard",
        "video_run",
        "video_assembly",
    ] {
        assert!(names.contains(name), "{name}");
    }
    let protocol = st_proxy::mcp_protocol::tool_arguments;
    assert!(
        protocol(
            "video_run",
            &json!({"action":"stop","run_id":"r","cancel_current":"maybe"})
        )
        .is_err()
    );
    assert!(protocol("video_run", &json!({"action":"get","run_id":"r","extra":1})).is_err());
    assert!(
        protocol(
            "video_project",
            &json!({"action":"create","path":"/x","project":null})
        )
        .is_ok()
    );
    let instructions = include_str!("../mcp/instructions.txt");
    assert!(instructions.contains("Do not open, render, visually inspect or analyze images"));
    assert!(instructions.contains("video_run start (the first and only GPU handover)"));
    assert!(instructions.contains("i2v only to animate one specific start image"));
}
#[tokio::test]
async fn failed_run_names_the_scene_and_only_bounded_job_diagnostics() {
    let h = Harness::new().await;
    let run = h
        .call("video_run", json!({"action":"get","run_id":"run_failed"}))
        .await;
    let run = ok(&run);
    assert_eq!(
        run["failed_scene"],
        json!({"scene_id":"s2","job_id":"job_failed","error":{"code":"h3_generation_failed",
            "stage":"sampling","kind":"shape_mismatch","exception":"RuntimeError","sizes":[7,8]}})
    );
    assert!(!run.to_string().contains("PRIVATE") && !run.to_string().contains("LEAK"));
    h.close().await;
}
