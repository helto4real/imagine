//! End to end: an agent drives the MCP video tools against an in-process Imagine with
//! a synthetic private root, an approved project root and fake H3 execution.
#[path = "../ui/tests/support/destination.rs"]
#[allow(dead_code)]
mod destination;
use axum::{
    Json, Router,
    routing::{get, post},
};
use destination::task_fixtures;
use imagine_server::{
    api::{self, App},
    engine::{Backend, EngineFuture, EngineState},
    h3_protocol::Profile,
    private_storage::Root,
    scheduler::{self, Action, Control},
    storage::Store,
    video_destination as d,
};
use serde_json::{Value, json};
use std::{
    io::Cursor,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
use tokio::{
    net::{TcpListener, UnixListener, UnixStream},
    sync::{Semaphore, oneshot, watch},
    task::JoinHandle,
};
const PROFILES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/config/video/prompt-profiles");
#[derive(Default)]
struct Probe {
    /// H3 profile, reference image count and opening frame bytes per scene job.
    videos: Mutex<Vec<(Profile, usize, usize)>>,
    /// Ticket connections from Imagine to the GPU coordinator.
    admissions: AtomicUsize,
    /// Requests that reached the LLM, including any authoring provider call.
    completions: AtomicUsize,
}
struct Fake(Arc<Probe>);
impl Backend for Fake {
    fn generate<'a>(
        &'a mut self,
        _: &'a imagine_server::generation::ImagePass,
    ) -> EngineFuture<'a, Vec<u8>> {
        Box::pin(async { anyhow::bail!("unexpected image generation") })
    }
    fn generate_video<'a>(
        &'a mut self,
        p: &'a imagine_server::h3_protocol::Request,
    ) -> EngineFuture<'a, imagine_server::short_av::Av> {
        Box::pin(async move {
            anyhow::ensure!(p.references.valid_payload());
            self.0.videos.lock().unwrap().push((
                p.profile,
                p.references.images.len(),
                p.image_rgb.len(),
            ));
            let continuation = p.capture_continuation.then(|| {
                let descriptor = imagine_server::h3_protocol::continuation::Descriptor {
                    schema_version: 1,
                    shape: p.shape,
                    sampled_frames: p.shape.frames,
                    visible_end_frame: p.shape.frames,
                    selected_end_frame: p.shape.frames,
                    handoff_upscale: p.sampling.as_ref().and_then(
                        imagine_server::h3_protocol::continuation::HandoffUpscale::from_sampling,
                    ),
                };
                let bytes = vec![0u8; descriptor.bytes()];
                imagine_server::h3_protocol::continuation::Artifact { descriptor, bytes }
            });
            Ok(imagine_server::short_av::Av {
                continuation,
                shape: p.shape,
                frames: vec![vec![70; p.shape.frame_bytes()]; p.shape.frames as usize],
                pcm: vec![0; p.shape.audio_samples() as usize * 8],
            })
        })
    }
    fn stop(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }
}
/// One Imagine session served over TCP, as the proxy reaches it.
struct Imagine {
    store: Store,
    control: Control,
    scheduler: JoinHandle<anyhow::Result<()>>,
    server: JoinHandle<()>,
    shutdown: oneshot::Sender<()>,
}
impl Imagine {
    async fn start(root: &Root, address: SocketAddr, socket: &Path, probe: &Arc<Probe>) -> Self {
        let mut store = Store::open_private(root.clone(), "own-image-workspace".into())
            .await
            .unwrap();
        // Agent workspaces live in the approved MCP root.
        store.workspaces = Some(Arc::new(imagine_server::workspaces::Workspaces::open(
            root.rooted_at(&root.project_roots()[0]).unwrap(),
        )));
        let (control, scheduler) = scheduler::start_coordinated(
            store.clone(),
            Box::new(Fake(probe.clone())),
            Some(socket.into()),
        )
        .await
        .unwrap();
        let (_, engine) = watch::channel(EngineState {
            backend: "rust".into(),
            h3_available: true,
            ..Default::default()
        });
        let router = api::router(App {
            video: control.video.clone(),
            enhancement: imagine_server::enhancement::Service::open_private(
                root.clone(),
                "library/enhancement-profiles.json".into(),
            )
            .unwrap(),
            library: Default::default(),
            profiles: Arc::new(
                imagine_server::profiles::Catalog::load(Path::new(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/config/image/profiles"
                )))
                .unwrap(),
            ),
            store: store.clone(),
            control: control.clone(),
            engine,
            uploads: Arc::new(Semaphore::new(2)),
        });
        let listener = TcpListener::bind(address).await.unwrap();
        let (shutdown, stopped) = oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        Self {
            store,
            control,
            scheduler,
            server,
            shutdown,
        }
    }
    /// Ends the session: the HTTP server closes its connections, then the store.
    async fn stop(self) {
        let _ = self.shutdown.send(());
        self.server.await.unwrap();
        self.control.call(Action::Shutdown).await.unwrap();
        self.scheduler.await.unwrap().unwrap();
        self.store.close().await.unwrap();
    }
    async fn http(&self, method: reqwest::Method, url: String, body: Value) -> (u16, Value) {
        let response = reqwest::Client::new()
            .request(method, url)
            .header("X-Imagine-Session", &self.store.session.id)
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = response.status().as_u16();
        (status, response.json().await.unwrap())
    }
}
/// The fake LLM behind the proxy coordinator and the authoring provider profile.
async fn llm(probe: Arc<Probe>) -> (String, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let router = Router::new()
        .route(
            "/health",
            get(|| async {
                Json(json!({"service":"strata","status":"ok","loaded":false,"model":"synthetic"}))
            }),
        )
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data":[{"id":"synthetic"}]})) }),
        )
        .route(
            "/v1/load",
            post(|| async { Json(json!({"status":"loaded"})) }),
        )
        .route(
            "/v1/unload",
            post(|| async { Json(json!({"status":"unloaded"})) }),
        )
        .route(
            "/v1/chat/completions",
            post(move || {
                probe.completions.fetch_add(1, Ordering::SeqCst);
                async { Json(json!({"choices":[]})) }
            }),
        );
    let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (url, task)
}
/// Counts every GPU ticket connection Imagine opens to the proxy's coordinator.
fn relay(from: PathBuf, to: PathBuf, probe: Arc<Probe>) -> JoinHandle<()> {
    let listener = UnixListener::bind(from).unwrap();
    tokio::spawn(async move {
        loop {
            let (mut inbound, _) = listener.accept().await.unwrap();
            probe.admissions.fetch_add(1, Ordering::SeqCst);
            let to = to.clone();
            tokio::spawn(async move {
                if let Ok(mut outbound) = UnixStream::connect(to).await {
                    let _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
                }
            });
        }
    })
}
/// A private root on the real encrypted /home volume of the approved task root,
/// so the proxy's own destination proof sees the same mount.
fn private_root(dir: &Path, inputs: &Path) -> Root {
    let root = dir.join("own-imagine");
    let runtime = dir.join("own-runtime");
    for path in [&root, &runtime, &inputs.to_path_buf()] {
        std::fs::create_dir(path).unwrap();
    }
    let config = d::Config {
        fsid: d::fsid(&d::directory(&root).unwrap()).unwrap(),
        root,
        mount: task_fixtures::MOUNT.into(),
        source: task_fixtures::SOURCE.into(),
        filesystem: task_fixtures::FILESYSTEM.into(),
    };
    let (root, ignored) = Root::with_environment(config, Arc::new(d::System))
        .unwrap()
        .with_runtime_directory(runtime)
        .unwrap()
        .with_project_roots(&[inputs.to_path_buf()]);
    assert!(ignored.is_empty());
    root
}
struct Agent {
    client: reqwest::Client,
    url: String,
    /// No response may contain this host folder or anything below it.
    hidden: String,
}
impl Agent {
    /// One `tools/call` exactly as an MCP client sends it: structured result and extra blocks.
    async fn call(&self, name: &str, args: Value) -> (Value, Vec<Value>) {
        let response = self
            .client
            .post(&self.url)
            .header("Accept", "application/json, text/event-stream")
            .json(&json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":name,"arguments":args}}))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        let body = response.text().await.unwrap();
        assert!(
            !body.contains(&self.hidden),
            "{name} leaked a host path: {body}"
        );
        let data = body
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap();
        let mut result = serde_json::from_str::<Value>(data).unwrap()["result"].take();
        let structured = result["structuredContent"].take();
        let content = result["content"].as_array().unwrap().clone();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(
            serde_json::from_str::<Value>(content[0]["text"].as_str().unwrap()).unwrap(),
            structured
        );
        assert_eq!(result["isError"], structured.get("error").is_some());
        (structured, content[1..].to_vec())
    }
    async fn ok(&self, name: &str, args: Value) -> Value {
        let (value, blocks) = self.call(name, args).await;
        assert!(value.get("error").is_none(), "{name}: {value}");
        assert!(blocks.is_empty(), "{name}");
        value
    }
    /// A rejected call carries only the category, message and Imagine's code, at and detail.
    async fn rejected(&self, name: &str, args: Value) -> Value {
        let (value, blocks) = self.call(name, args).await;
        assert!(blocks.is_empty());
        let error = value["error"]
            .as_object()
            .unwrap_or_else(|| panic!("{value}"));
        assert!(
            error
                .keys()
                .all(|k| ["category", "message", "code", "at", "detail"].contains(&k.as_str())),
            "{value}"
        );
        value["error"].clone()
    }
}
fn png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let mut out = Cursor::new(vec![]);
    image::RgbImage::from_pixel(width, height, image::Rgb(rgb))
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}
/// The profile file the provider receives, without surrounding whitespace.
fn prompt_file(profile: &Value, key: &str) -> String {
    let entry = profile["prompt_files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["key"] == key)
        .unwrap();
    std::fs::read_to_string(format!(
        "{PROFILES}/{}/{}",
        entry["source_profile_id"].as_str().unwrap(),
        entry["file"].as_str().unwrap()
    ))
    .unwrap()
    .trim()
    .to_owned()
}
fn saved(value: &Value) {
    let persistence = &value["persistence"];
    assert_eq!(persistence["state"], "saved", "{value}");
    assert_eq!(
        persistence["saved_revision"], value["project_revision"],
        "{value}"
    );
}
fn files(path: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(path)
        .map(|entries| entries.map(|e| e.unwrap().path()).collect())
        .unwrap_or_default()
}
#[tokio::test]
async fn agent_produces_a_video_in_its_workspace_and_resumes_with_the_key() {
    let dir = tempfile::tempdir().unwrap();
    assert!(task_fixtures::is_owned(dir.path()));
    let inputs = dir.path().join("own-mcp-root");
    let root = private_root(dir.path(), &inputs);
    let probe = Arc::new(Probe::default());
    let (llm_url, llm_task) = llm(probe.clone()).await;
    let coordinator = dir.path().join("g.sock");
    let tickets = dir.path().join("r.sock");
    let relay_task = relay(tickets.clone(), coordinator.clone(), probe.clone());
    let address = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .local_addr()
        .unwrap();
    let imagine = Imagine::start(&root, address, &tickets, &probe).await;
    let base = format!("http://{address}");
    // Any authoring provider request would reach the fake LLM.
    let provider = json!({"profile":{"id":"proxy","name":"Own fixture","base_url":format!("{llm_url}/v1"),"model":"own-story-fixture","system_prompt":"unused by video","max_tokens":1024,"temperature":null,"parameters":{},"has_api_key":false},"api_key":"own-fixture-key"});
    let (status, _) = imagine
        .http(
            reqwest::Method::PUT,
            format!("{base}/v1/enhancement/profiles/proxy"),
            provider,
        )
        .await;
    assert_eq!(status, 200);
    let running = st_proxy::service::Service::start(st_proxy::config::Config {
        mcp_port: Some(0),
        mcp_workflows_dir: "/nonexistent/no-comfy-graphs-needed".into(),
        mcp_output_dir: Some(dir.path().join("own-output").to_string_lossy().into()),
        chat_port: 0,
        image_port: 0,
        llm_backend: "sc117".into(),
        llm_url: llm_url.clone(),
        comfy_enabled: false,
        comfy_url: "http://127.0.0.1:1".into(),
        coordinator_socket: Some(coordinator.to_str().unwrap().into()),
        imagine_url: base.clone(),
        poll_interval: 0.01,
        ..Default::default()
    })
    .await
    .unwrap();
    let port = running.mcp_port.unwrap();
    let agent = Agent {
        client: reqwest::Client::new(),
        url: format!("http://127.0.0.1:{port}/mcp"),
        hidden: dir.path().to_string_lossy().into_owned(),
    };
    let untouched = |stage: &str| {
        assert_eq!(probe.completions.load(Ordering::SeqCst), 0, "{stage}");
        assert_eq!(probe.admissions.load(Ordering::SeqCst), 0, "{stage}");
        assert!(probe.videos.lock().unwrap().is_empty(), "{stage}");
        assert!(
            running.service.coordinator.status()["gpu_owner"].is_null(),
            "{stage}"
        );
    };

    // Discovery.
    let catalog = agent.ok("list_video_profiles", json!({})).await;
    let profile = catalog["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["id"] == "nsfw")
        .unwrap()
        .clone();
    let described = agent
        .ok("describe_video_profile", json!({"profile":"nsfw"}))
        .await;
    assert_eq!(described["profile"], profile);
    assert_eq!(
        described["agent_guidance"].as_str().unwrap(),
        include_str!("../config/video/agent-guidance.txt")
    );
    let capabilities = agent.ok("video_capabilities", json!({})).await;
    assert!(
        capabilities["project_operations"]
            .as_array()
            .unwrap()
            .contains(&json!("create_at_location"))
    );
    assert_eq!(described["limits"], capabilities["authoring"]);

    // A workspace, reachable only with its key; Imagine keeps it in a dated folder.
    let opened = agent
        .ok("workspace_create", json!({"title":"Own Agent Film"}))
        .await;
    let key = opened["workspace_key"].as_str().unwrap().to_owned();
    let workspace = imagine
        .store
        .workspaces
        .as_ref()
        .unwrap()
        .resolve(&key)
        .unwrap();
    assert!(workspace.folder.starts_with(&inputs));
    assert!(workspace.folder.join("AGENT-KEY.txt").is_file());
    let error = agent
        .rejected(
            "workspace_files",
            json!({"workspace_key":"imk-aaaaaaaaaaaaaaaaaaaaaaaaaa"}),
        )
        .await;
    assert_eq!(error["category"], "not_found");

    // Uploads: HTTP with the key as bearer, or the tool for a small file.
    let keeper = png(24, 16, [200, 150, 120]);
    let harbour = png(32, 20, [40, 90, 160]);
    let http = reqwest::Client::new();
    let uploaded: Value = http
        .put(format!(
            "http://127.0.0.1:{port}/v1/workspace/files?name=own-keeper.png"
        ))
        .header("Authorization", format!("Bearer {key}"))
        .body(keeper.clone())
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!uploaded.to_string().contains(&agent.hidden));
    let keeper_file = uploaded["file_id"].as_str().unwrap().to_owned();
    use base64::Engine;
    let harbour_file = agent
        .ok(
            "workspace_upload",
            json!({"workspace_key":key,"name":"own-harbour.png",
                "content_base64":base64::engine::general_purpose::STANDARD.encode(&harbour)}),
        )
        .await["file_id"]
        .as_str()
        .unwrap()
        .to_owned();
    let listed = agent
        .ok("workspace_files", json!({"workspace_key":key}))
        .await;
    assert_eq!(listed["files"].as_array().unwrap().len(), 2, "{listed}");
    assert!(workspace.folder.join("uploads/own-keeper.png").is_file());
    // A body that breaks off after a valid header publishes nothing.
    {
        use tokio::io::AsyncWriteExt;
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        let head = format!(
            "PUT /v1/workspaces/{}/files?name=own-broken.png HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n",
            workspace.meta.id, 64
        );
        socket.write_all(head.as_bytes()).await.unwrap();
        socket.write_all(&keeper[..64]).await.unwrap();
        socket.write_all(b"\r\n").await.unwrap();
        socket.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let mut uploads: Vec<_> = files(&workspace.folder.join("uploads"))
        .into_iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    uploads.sort();
    assert_eq!(uploads, ["own-harbour.png", "own-keeper.png"]);

    // A project in the workspace, saved there at once.
    let mut create = json!(imagine_server::video_defaults::new_project());
    create["delivery"]["width"] = json!(32);
    create["delivery"]["height"] = json!(32);
    let created = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"create","workspace_key":key,"title":"Own agent film","project":create}),
        )
        .await;
    let folder = workspace.folder.join("videos/own-agent-film");
    saved(&created);
    assert!(folder.join("project.json").is_file());
    let project = created["project_id"].as_str().unwrap().to_owned();
    // A create retried after a timeout returns the project already there.
    let retried = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"create","workspace_key":key,"title":"Own agent film","project":create}),
        )
        .await;
    assert_eq!(retried["project_id"], project.as_str());
    assert_eq!(retried["project_revision"], created["project_revision"]);
    assert!(!root.layout().project(&project).unwrap().root.exists());
    // Another workspace's key cannot reach this project.
    let other = agent.ok("workspace_create", json!({})).await;
    let error = agent
        .rejected(
            "video_project",
            json!({"workspace_key":key,"action":"get","workspace_key":other["workspace_key"],"project_id":project}),
        )
        .await;
    assert_eq!(error["category"], "not_found");

    // Two workspace images: a cast identity image and a setting reference.
    let cast = agent
        .ok(
            "video_register_reference",
            json!({"workspace_key":key,"project_id":project,"file_id":keeper_file,"expected_project_revision":created["project_revision"],
                "cast":{"name":"Keeper","alias":"@keeper","description":"Own fictional adult harbour keeper"}}),
        )
        .await;
    saved(&cast);
    assert_eq!(cast["cast"]["tag"], "@keeper");
    let setting = agent
        .ok(
            "video_register_reference",
            json!({"workspace_key":key,"project_id":project,"file_id":harbour_file,"expected_project_revision":cast["project_revision"],"name":"Own harbour"}),
        )
        .await;
    saved(&setting);
    assert_ne!(setting["asset_id"], cast["asset_id"]);
    let view = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"get","project_id":project}),
        )
        .await;
    assert_eq!(view["cast"][0]["alias"], "keeper");
    assert_eq!(view["cast"][0]["images"][0]["asset"], cast["asset_id"]);
    assert_eq!(view["assets"].as_array().unwrap().len(), 2);
    untouched("references");

    // The context bundle: verbatim profile prompts, images as MCP image blocks.
    let revision = setting["project_revision"].clone();
    let request = json!({"schema_version":1,"expected_project_revision":revision,"prompt_profile_id":"nsfw",
        "brief":"Own synthetic story: the keeper watches the harbour, then walks to the pier.","instruction":"",
        "duration_seconds":2,"destination":"append","cast":[cast["cast"]["reference_id"]],
        "images":[{"id":"own_harbour","asset":setting["asset_id"],"label":"Own harbour","usage":"setting_reference"}],
        "add_dialogue":false,"add_music":false});
    let (started, blocks) = agent
        .call(
            "video_storyboard",
            json!({"workspace_key":key,"action":"start","project_id":project,"request":request}),
        )
        .await;
    assert!(started.get("error").is_none(), "{started}");
    assert_eq!(started["phase"], "outline");
    assert_eq!(started["provider_snapshot"]["profile_id"], "external");
    let draft = started["id"].as_str().unwrap().to_owned();
    let bundle = &started["bundle"];
    assert!(!started.to_string().contains("base64"));
    let images = bundle["images"].as_array().unwrap();
    assert_eq!(images.len(), 2);
    assert_eq!(blocks.len(), 2);
    for (i, (image, original)) in images.iter().zip([&harbour, &keeper]).enumerate() {
        assert_eq!(image["index"], i);
        assert_eq!(blocks[i]["type"], "image");
        assert_eq!(blocks[i]["mimeType"], "image/png");
        assert_eq!(
            base64::engine::general_purpose::STANDARD
                .decode(blocks[i]["data"].as_str().unwrap())
                .unwrap(),
            imagine_server::video::authoring::vision::normalize(original)
                .unwrap()
                .png
        );
    }
    assert_eq!(
        (&images[0]["image_id"], &images[0]["role"]),
        (&json!("own_harbour"), &json!("story"))
    );
    assert_eq!(images[1]["role"], "cast");
    let schema = bundle["outline"]["output_schema"].to_string();
    let outline_prompt = prompt_file(&profile, "story_outline");
    assert!(outline_prompt.contains("{{STORY_OUTLINE_JSON_SCHEMA}}"));
    assert_eq!(
        bundle["outline"]["system"].as_str().unwrap(),
        outline_prompt.replace("{{STORY_OUTLINE_JSON_SCHEMA}}", &schema)
    );
    assert_eq!(
        bundle["expansion"]["system"].as_str().unwrap(),
        prompt_file(&profile, "story_scene")
    );
    assert_eq!(
        bundle["inventory"]["system"].as_str().unwrap(),
        prompt_file(&profile, "story_image_inventory")
    );
    assert_eq!(
        bundle["outline"]["slots"],
        json!(["inventory_note:own_harbour"])
    );
    for slot in ["previous_scene_context", "image_context", "scene_json"] {
        assert!(
            bundle["expansion"]["slots"]
                .as_array()
                .unwrap()
                .contains(&json!(slot)),
            "{slot}"
        );
    }
    // A lost start is recovered from the project's Drafts, never a new one.
    let listed = agent
        .ok(
            "video_storyboard",
            json!({"workspace_key":key,"action":"list","project_id":project}),
        )
        .await;
    assert_eq!(
        listed["drafts"],
        json!([{"id":draft,"phase":"outline","provider":"external","prompt_profile_id":"nsfw"}])
    );
    let (rebuilt, again) = agent
        .call(
            "video_storyboard",
            json!({"workspace_key":key,"action":"bundle","draft_id":draft}),
        )
        .await;
    assert_eq!(rebuilt["bundle"], started["bundle"]);
    assert_eq!(again, blocks);
    untouched("storyboard start");

    // One submission: a hard cut into a continuous shot, Reference then I2V.
    let scene = |title: &str, boundary: &str, setting: bool| {
        json!({"title":title,"summary":format!("Own {title} summary"),
        "description":format!("Own {title} at the harbour"),"ambient_sound":"Waves","non_diegetic_music":"","weight":1,
        "boundary_from_previous":boundary,"setting_reference_matches":{"own_harbour":setting},
        "sections":[{"prompt":format!("@keeper own {title} beat"),"weight":1}]})
    };
    let outline = json!({"scenes":[scene("watch","hard_cut",true),scene("walk","continuous_shot",false)],"visual_placements":[]});
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
        .collect::<Vec<_>>();
    let production = json!({"scenes":[{"model":"reference","continuity":null,"rationale":"Own setting shot"},
        {"model":"i2v","continuity":{"method":"auto","include_audio":false},"rationale":"Own continued motion"}]});
    let submission = |dry_run: bool| {
        json!({"schema_version":1,"expected_project_revision":revision,
        "inventory":[{"id":"own_harbour","note":"Own harbour at dusk"}],"outline":outline.to_string(),
        "expansions":expansions,"production":production,"dry_run":dry_run})
    };
    let mut short = submission(true);
    short["expansions"].as_array_mut().unwrap().pop();
    let error = agent
        .rejected(
            "video_storyboard",
            json!({"workspace_key":key,"action":"submit","draft_id":draft,"submission":short}),
        )
        .await;
    assert_eq!(error["category"], "validation");
    assert_eq!(error["code"], "story_expansion_count_mismatch");
    assert_eq!(error["at"], "expansions");
    let checked = agent
        .ok(
            "video_storyboard",
            json!({"workspace_key":key,"action":"submit","draft_id":draft,"submission":submission(true)}),
        )
        .await;
    assert_eq!(checked["valid"], true, "{checked}");
    let planned = checked["scenes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| (s["model"].clone(), s["continuity"]["method"].clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        planned,
        vec![
            (json!("reference"), json!(null)),
            (json!("i2v"), json!("last_frame_first_frame"))
        ]
    );
    let codes = checked["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|w| w["code"].as_str().unwrap())
        .collect::<Vec<_>>();
    // The I2V continuation names the cast, so it carries no identity reference.
    assert_eq!(
        codes,
        ["i2v_cast_identity_text_only", "continuity_source_pending"]
    );
    let view = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"get","project_id":project}),
        )
        .await;
    assert_eq!(
        view["project_revision"], revision,
        "A dry run commits nothing"
    );
    assert!(view["scenes"].as_array().unwrap().is_empty());
    let applied = agent
        .ok(
            "video_storyboard",
            json!({"workspace_key":key,"action":"submit","draft_id":draft,"submission":submission(false)}),
        )
        .await;
    saved(&applied);
    assert!(applied.get("resources").is_none(), "bounded projection");
    let scenes = applied["scenes"].as_array().unwrap();
    assert_eq!(scenes.len(), 2);
    assert_eq!(applied["scene_map"][1]["scene_id"], scenes[1]["id"]);
    assert_eq!(scenes[0]["model"], "project_default");
    assert_eq!(scenes[1]["model"]["override"]["variant"], "i2v");
    assert!(scenes[0]["continuity"].is_null());
    assert_eq!(scenes[1]["continuity"]["method"], "last_frame_first_frame");
    assert_eq!(scenes[1]["continuity"]["source_scene"], scenes[0]["id"]);
    assert_eq!(applied["authoring"][0]["phase"], "applied");
    let stored = agent
        .ok(
            "video_storyboard",
            json!({"workspace_key":key,"action":"get","draft_id":draft}),
        )
        .await;
    assert_eq!(stored["phase"], "applied");
    assert_eq!(stored["production"], production);
    untouched("storyboard commit");

    // The first and only GPU handover.
    let revision = applied["project_revision"].clone();
    let run_request = json!({"schema_version":1,"project_id":project,"expected_project_revision":revision,"mode":"missing_only","selected_scene":null});
    let run = agent
        .ok(
            "video_run",
            json!({"workspace_key":key,"action":"start","job_id":"own-run-1","request":run_request}),
        )
        .await;
    assert_ne!(run["reused"], true, "{run}");
    let run_id = run["id"].as_str().unwrap().to_owned();
    let again = agent
        .ok(
            "video_run",
            json!({"workspace_key":key,"action":"start","job_id":"own-run-1","request":run_request}),
        )
        .await;
    assert_eq!(again["id"], run_id.as_str());
    assert_eq!(again["reused"], true);
    let mut changed = run_request.clone();
    changed["mode"] = json!("all_fresh_seeds");
    let error = agent
        .rejected(
            "video_run",
            json!({"workspace_key":key,"action":"start","job_id":"own-run-1","request":changed}),
        )
        .await;
    assert_eq!(error["category"], "conflict");
    assert_eq!(error["code"], "idempotency_conflict");
    let mut done = Value::Null;
    for _ in 0..4 {
        done = agent
            .ok(
                "video_run",
                json!({"workspace_key":key,"action":"get","run_id":run_id,"wait":true,"wait_seconds":20}),
            )
            .await;
        if done["status"] != "running" {
            break;
        }
    }
    assert_eq!(done["status"], "completed", "{done}");
    let videos = probe.videos.lock().unwrap().clone();
    assert_eq!(
        videos
            .iter()
            .map(|(profile, _, _)| *profile)
            .collect::<Vec<_>>(),
        vec![Profile::AvNoReferenceV2, Profile::ImageToAvV1]
    );
    // Reference: the cast identity and the setting; I2V: the previous take's last frame only.
    assert_eq!((videos[0].1, videos[0].2), (2, 0));
    assert_eq!((videos[1].1, videos[1].2), (0, 32 * 32 * 3));
    assert!(probe.admissions.load(Ordering::SeqCst) >= 2);
    assert_eq!(probe.completions.load(Ordering::SeqCst), 0);
    let view = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"get","project_id":project}),
        )
        .await;
    saved(&view);
    let takes = view["takes"].as_array().unwrap();
    assert_eq!(takes.len(), 2);
    assert!(takes.iter().all(|t| t["status"] == "accepted"), "{view}");
    for (scene, take) in view["scenes"].as_array().unwrap().iter().zip(takes) {
        assert_eq!(scene["accepted_take"]["id"], take["id"]);
    }
    let outputs = folder.join("outputs/videos");
    let rendered = |outputs: &Path| {
        files(outputs)
            .into_iter()
            .filter(|p| p.extension().is_some_and(|e| e == "mp4"))
            .count()
    };
    assert_eq!(rendered(&outputs), 2, "takes live in the project folder");

    // CPU assembly of the accepted takes.
    let plan = agent
        .ok(
            "video_assembly",
            json!({"workspace_key":key,"action":"plan","project_id":project,"request":{"schema_version":1,"expected_project_revision":view["project_revision"]}}),
        )
        .await;
    let plan_id = plan["id"].as_str().unwrap().to_owned();
    let job = agent
        .ok(
            "video_assembly",
            json!({"workspace_key":key,"action":"submit","plan_id":plan_id,"job_id":"own-assembly-1"}),
        )
        .await;
    assert_eq!(job["reused"], false);
    let again = agent
        .ok(
            "video_assembly",
            json!({"workspace_key":key,"action":"submit","plan_id":plan_id,"job_id":"own-assembly-1"}),
        )
        .await;
    assert_eq!((&again["id"], &again["reused"]), (&job["id"], &json!(true)));
    let mut assembled = Value::Null;
    for _ in 0..4 {
        assembled = agent
            .ok(
                "video_assembly",
                json!({"workspace_key":key,"action":"get","plan_id":plan_id,"wait":true,"wait_seconds":20}),
            )
            .await;
        if assembled["terminal"] == true {
            break;
        }
    }
    assert_eq!(assembled["job"]["status"], "succeeded", "{assembled}");
    // The final video is a workspace file the agent downloads with its key.
    let output = &assembled["job"]["output"];
    assert!(output.get("saved_path").is_none());
    let final_file = output["file_id"].as_str().unwrap().to_owned();
    assert_eq!(rendered(&outputs), 3);
    let downloaded = http
        .get(output["download_url"].as_str().unwrap())
        .header("Authorization", format!("Bearer {key}"))
        .send()
        .await
        .unwrap();
    assert_eq!(downloaded.status(), 200);
    let bytes = downloaded.bytes().await.unwrap();
    let on_disk = files(&outputs)
        .into_iter()
        .find(|p| std::fs::read(p).is_ok_and(|b| b == bytes.as_ref()));
    assert!(on_disk.is_some(), "downloaded the published final video");
    let refused = http
        .get(output["download_url"].as_str().unwrap())
        .header(
            "Authorization",
            format!("Bearer {}", other["workspace_key"].as_str().unwrap()),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 404);
    let view = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"get","project_id":project}),
        )
        .await;
    saved(&view);
    assert_eq!(view["deliveries"].as_array().unwrap().len(), 1, "{view}");
    let explicit = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"save","project_id":project,"expected_project_revision":view["project_revision"]}),
        )
        .await;
    saved(&explicit);
    assert!(
        files(&root.layout().root().join("projects"))
            .iter()
            .all(|p| p.ends_with("own-image-workspace")),
        "nothing of the project lives in the private root"
    );
    let final_revision = view["project_revision"].clone();

    // Session restart: the proxy drops its session state; with the saved key
    // the agent finds its workspace and the project reopens from its folder.
    imagine.stop().await;
    let imagine = Imagine::start(&root, address, &tickets, &probe).await;
    let error = agent
        .rejected(
            "video_run",
            json!({"workspace_key":key,"action":"get","run_id":run_id}),
        )
        .await;
    assert_eq!(error["category"], "not_found", "{error}");
    let resumed = agent
        .ok("workspace_open", json!({"workspace_key":key}))
        .await;
    assert_eq!(resumed["video_projects"][0]["project_id"], project.as_str());
    assert!(
        agent
            .ok("workspace_files", json!({"workspace_key":key}))
            .await["files"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f["file_id"] == final_file.as_str())
    );
    let reopened = agent
        .ok(
            "video_project",
            json!({"workspace_key":key,"action":"get","project_id":project}),
        )
        .await;
    assert_eq!(reopened["project_revision"], final_revision);
    assert_eq!(reopened["scenes"].as_array().unwrap().len(), 2);
    assert!(
        reopened["scenes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["accepted_take"].is_object()),
        "{reopened}"
    );
    assert_eq!(
        reopened["scenes"][1]["continuity"]["method"],
        "last_frame_first_frame"
    );
    assert_eq!(probe.completions.load(Ordering::SeqCst), 0);

    // A key change: the new key works, the old does not, and the key file holds
    // only the new one.
    let workspaces = imagine.store.workspaces.as_ref().unwrap();
    let rotated = workspaces.replace_key(&workspace.meta.id).unwrap();
    assert_ne!(rotated, key);
    assert_eq!(workspaces.key(&workspace.meta.id).unwrap(), rotated);
    assert!(workspaces.resolve(&key).is_err());
    let key_file = workspace.folder.join("AGENT-KEY.txt");
    let text = std::fs::read_to_string(&key_file).unwrap();
    assert!(text.starts_with(&format!("{rotated}\n\n")) && !text.contains(&key));

    running.close().await;
    imagine.stop().await;
    relay_task.abort();
    llm_task.abort();
}
