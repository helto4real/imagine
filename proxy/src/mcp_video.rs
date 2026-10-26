//! Video production tools: argument shapes, workspace confinement and endpoint mapping
//! only. Imagine owns projects, Drafts, runs and assemblies; the proxy keeps no video state.
use super::*;
use reqwest::Method;
use std::time::Instant;
const TOOLS: &[&str] = &[
    "list_video_profiles",
    "describe_video_profile",
    "video_capabilities",
    "video_project",
    "video_register_reference",
    "video_storyboard",
    "video_run",
    "video_take",
    "video_assembly",
];
const DEFAULT_WAIT: u64 = 30;
/// The total video length when the user gives none.
const DEFAULT_SECONDS: u64 = 15;
/// MCP productions start on Draft; the UI's own default is smaller.
fn default_canvas() -> Value {
    json!({"ratio":"16:9","portrait":false,"quality":"Draft"})
}
const MAX_WAIT: u64 = 60;
/// Per-call limit for a video tool, or None for any other tool. Long work stays async in Imagine.
pub(super) fn limit(name: &str, args: &Value) -> Option<Duration> {
    if !TOOLS.contains(&name) {
        return None;
    }
    let seconds = match (name, args["action"].as_str().unwrap_or("")) {
        ("list_video_profiles" | "describe_video_profile" | "video_capabilities", _)
        | ("video_project" | "video_storyboard", "get")
        | ("video_storyboard", "list")
        | ("video_take", "list") => 5,
        ("video_storyboard", "start" | "submit" | "bundle") | ("video_take", "view") => 60,
        ("video_run" | "video_assembly", "get") => 5 + wait(args).as_secs(),
        _ => 15,
    };
    Some(Duration::from_secs(seconds))
}
fn check_wait(args: &Value) -> std::result::Result<(), Fault> {
    if !args["wait_seconds"].is_null() && args["wait_seconds"].as_u64().is_none_or(|s| s > MAX_WAIT)
    {
        return Err(validation("wait_seconds must be an integer from 0 to 60"));
    }
    Ok(())
}
fn wait(args: &Value) -> Duration {
    if args["wait"].as_bool() != Some(true) {
        return Duration::ZERO;
    }
    let seconds = args["wait_seconds"].as_u64().unwrap_or(DEFAULT_WAIT);
    Duration::from_secs(seconds.min(MAX_WAIT))
}
/// Forward only Imagine's stable code, field path and short detail; never snapshots or bodies.
fn rejected(error: anyhow::Error) -> Fault {
    let Some(r) = crate::imagine::rejection(&error) else {
        return Fault::new(
            "connection",
            "Imagine did not respond; retry within the same session",
        );
    };
    let category = match r.status {
        404 => "not_found",
        409 => "conflict",
        400..=499 => "validation",
        _ => "imagine",
    };
    Fault {
        category: category.into(),
        message: format!("Imagine rejected the request: {}", r.code),
        code: Some(r.code.into()),
        at: r.at.map(str::to_owned),
        detail: r.detail.map(str::to_owned),
    }
}
fn ident<'a>(args: &'a Value, key: &str) -> std::result::Result<&'a str, Fault> {
    args[key]
        .as_str()
        .filter(|s| {
            !s.is_empty()
                && s.len() <= 128
                && s.bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        })
        .ok_or_else(|| validation(format!("{key} must be 1-128 letters, digits, _ or -")))
}
fn revision(args: &Value) -> std::result::Result<u64, Fault> {
    args["expected_project_revision"]
        .as_u64()
        .ok_or_else(|| validation("expected_project_revision must be a non-negative integer"))
}
/// A request body object; schema_version defaults to 1.
fn body(args: &Value, key: &str) -> std::result::Result<Value, Fault> {
    let mut body = args[key]
        .as_object()
        .cloned()
        .ok_or_else(|| validation(format!("{key} must be an object")))?;
    body.entry("schema_version").or_insert(json!(1));
    Ok(Value::Object(body))
}
fn text<'a>(
    value: &'a Value,
    key: &str,
    max: usize,
    required: bool,
) -> std::result::Result<&'a str, Fault> {
    match &value[key] {
        Value::Null if !required => Ok(""),
        Value::String(s)
            if s.chars().count() <= max
                && !(required && s.trim().is_empty())
                && !s.chars().any(|c| c.is_control() && c != '\n') =>
        {
            Ok(s)
        }
        _ => Err(validation(format!(
            "{key} must be {}text of at most {max} characters",
            if required { "non-empty " } else { "" }
        ))),
    }
}
fn read_failure(e: anyhow::Error) -> Fault {
    if e.downcast_ref::<std::io::Error>().is_some()
        || e.downcast_ref::<image::ImageError>().is_some()
    {
        validation("reference could not be read or decoded")
    } else {
        validation(e)
    }
}
/// The call's workspace; every workspace-scoped action has one.
fn own(workspace: Option<&Bound>) -> std::result::Result<&Bound, Fault> {
    workspace.ok_or_else(|| validation("workspace_key is required"))
}
fn short_hash(text: &str) -> String {
    references::hash(text.as_bytes())[..16].to_owned()
}
fn terminal_run(run: &Value) -> bool {
    matches!(
        run["status"].as_str(),
        Some("completed" | "stopped" | "cancelled" | "failed" | "interrupted")
    )
}
/// Bounded projection of a project snapshot: identities, timing, prompts and status, no media.
fn project_view(snapshot: &Value) -> Value {
    let p = &snapshot["project"];
    let creative = &p["creative"];
    let list = |v: &Value, f: &dyn Fn(&Value) -> Value| {
        v.as_array()
            .map(|a| a.iter().map(f).collect::<Vec<_>>())
            .unwrap_or_default()
    };
    let scenes = list(&p["scenes"], &|s| {
        let id = s["id"].as_str().unwrap_or("");
        json!({"id":s["id"],"title":creative["scene_prompts"][id]["title"],
        // The preset the scene was set to, and whether it was changed since.
        "preset":creative["scene_presets"][id]["key"],"preset_state":snapshot["preset_states"][id],
        "start":s["start"],"duration":s["duration"],"description":s["description"],"model":s["model"],
        "continuity":s["continuity"],"accepted_take":s["accepted_take"].as_object().map(|t| json!({"id":t["id"],"model":t["model"]})),
        "sections":list(&s["sections"],&|x| json!({"id":x["id"],"start":x["start"],"duration":x["duration"],"prompt":x["prompt"]}))})
    });
    let assets = p["assets"]
        .as_object()
        .map(|a| {
            a.values()
                .map(|a| json!({"id":a["id"],"name":a["name"],"kind":a["kind"]}))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let cast = list(&creative["references"], &|r| {
        json!({"id":r["id"],"name":r["name"],"alias":r["alias"],"enabled":r["enabled"],
        "images":list(&r["images"],&|i| json!({"id":i["id"],"asset":i["asset"],"enabled":i["enabled"]}))})
    });
    let takes = list(&snapshot["takes"], &|t| {
        let state = &snapshot["take_states"][t["id"].as_str().unwrap_or("")];
        json!({"id":t["id"],"scene_id":t["scene_id"],"status":t["status"],"outdated":state["outdated"]})
    });
    json!({"project_id":p["id"],"project_revision":p["project_revision"],
        "default_model":p["default_model"],"delivery":p["delivery"],"persistence":snapshot["persistence"],
        "scenes":scenes,"assets":assets,"cast":cast,"takes":takes,
        "deliveries":list(&snapshot["deliveries"],&|d| json!({"id":d["id"],"asset_id":d["asset_id"]})),
        "authoring":list(&snapshot["authoring"],&|d| json!({"id":d["id"],"phase":d["phase"]}))})
}
/// A project's Takes in creation order, optionally of one Scene, with what an agent
/// needs to choose between them.
fn take_list(snapshot: &Value, scene: Option<&str>) -> Value {
    let p = &snapshot["project"];
    let current = |scene: &Value| {
        p["scenes"]
            .as_array()
            .and_then(|all| all.iter().find(|s| &s["id"] == scene))
            .map(|s| s["accepted_take"]["id"].clone())
            .unwrap_or(Value::Null)
    };
    let takes = snapshot["takes"]
        .as_array()
        .map(|all| {
            all.iter()
                .enumerate()
                .filter(|(_, t)| scene.is_none_or(|s| t["scene_id"] == s))
                .map(|(order, t)| {
                    let state = &snapshot["take_states"][t["id"].as_str().unwrap_or("")];
                    let provenance = &t["provenance"];
                    json!({"id":t["id"],"scene_id":t["scene_id"],"order":order,"status":t["status"],
                        "current":current(&t["scene_id"]) == t["id"],"seed":provenance["seed"],
                        "source_take":provenance["source_take"],"inputs_current":state["inputs_current"]})
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    json!({"project_id":p["id"],"project_revision":p["project_revision"],"takes":takes})
}
fn saved(snapshot: &Value) -> Value {
    json!({"project_id":snapshot["project"]["id"],"project_revision":snapshot["project"]["project_revision"],"persistence":snapshot["persistence"]})
}
/// Replace an embedded project snapshot with the bounded projection.
fn bounded(mut value: Value) -> Value {
    if let Some(snapshot) = value
        .get_mut("snapshot")
        .filter(|s| s["project"].is_object())
    {
        *snapshot = project_view(snapshot);
    }
    value
}
/// Move every bundle image's base64 into MCP image blocks, in order; JSON keeps the labels.
fn take_images(value: &mut Value) -> std::result::Result<Vec<Value>, Fault> {
    let Some(images) = ["/bundle/images", "/draft/bundle/images", "/frames"]
        .into_iter()
        .find(|p| value.pointer(p).is_some())
        .and_then(|p| value.pointer_mut(p))
        .and_then(Value::as_array_mut)
    else {
        return Ok(vec![]);
    };
    let mut blocks = vec![];
    for image in images.iter_mut().filter_map(Value::as_object_mut) {
        let Some(data) = image.remove("base64") else {
            continue;
        };
        let mime = image
            .get("mime")
            .or_else(|| image.get("mime_type"))
            .and_then(Value::as_str)
            .unwrap_or("image/jpeg");
        let valid = data.as_str().is_some_and(|d| {
            !d.is_empty()
                && d.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"+/=".contains(&b))
        });
        if !valid || !["image/jpeg", "image/png", "image/webp"].contains(&mime) {
            return Err(Fault::new("imagine", "invalid context bundle image"));
        }
        blocks.push(json!({"type":"image","data":data,"mimeType":mime}));
        image.insert("index".into(), json!(blocks.len() - 1));
    }
    Ok(blocks)
}
impl Mcp {
    async fn call_video(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
        key: Option<&str>,
    ) -> std::result::Result<Value, Fault> {
        self.client
            .json(method, path, body, key)
            .await
            .map_err(rejected)
    }
    fn video_key(&self, args: &Value) -> std::result::Result<String, Fault> {
        let job = args["job_id"]
            .as_str()
            .filter(|s| {
                !s.is_empty()
                    && s.len() <= 64
                    && s.bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
            })
            .ok_or_else(|| validation("job_id must be 1-64 letters, digits, _ or -"))?;
        let session = self
            .client
            .session_id()
            .ok_or_else(|| Fault::new("privacy", "Imagine is unavailable"))?;
        Some(format!("mcp-{session}-{job}"))
            .filter(|key| key.len() <= 128)
            .ok_or_else(|| validation("job_id is too long for this Imagine session"))
    }
    pub(super) async fn video(
        &self,
        name: &str,
        args: &Value,
    ) -> std::result::Result<(Value, Vec<Value>), Fault> {
        if self.inner.stop.is_cancelled() {
            return Err(Fault::new("shutdown", "broker is shutting down"));
        }
        let action = args["action"].as_str().unwrap_or("");
        // Everything but the catalog tools works inside one workspace.
        let workspace = match name {
            "list_video_profiles" | "describe_video_profile" | "video_capabilities" => None,
            _ => Some(self.bound(args).await?),
        };
        let ws = workspace.as_ref();
        let value = match (name, action) {
            ("list_video_profiles", _) => {
                self.call_video(Method::GET, "/v1/video/authoring/profiles", None, None)
                    .await?
            }
            ("describe_video_profile", _) => {
                let id = args["profile"]
                    .as_str()
                    .filter(|s| {
                        !s.is_empty()
                            && s.len() <= 64
                            && s.bytes().all(|b| {
                                b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(&b)
                            })
                    })
                    .ok_or_else(|| validation("invalid profile"))?;
                let path = format!("/v1/video/authoring/profiles/{id}");
                self.call_video(Method::GET, &path, None, None).await?
            }
            ("video_capabilities", _) => {
                let mut value = self
                    .call_video(Method::GET, "/v1/video/capabilities", None, None)
                    .await?;
                // Report the defaults MCP productions apply.
                value["canvas"]["default"] = default_canvas();
                let profiles = self
                    .call_video(Method::GET, "/v1/video/authoring/profiles", None, None)
                    .await
                    .unwrap_or(Value::Null);
                let profile = profiles["profiles"]
                    .as_array()
                    .and_then(|all| all.iter().find(|p| p["default"] == true))
                    .map(|p| p["id"].clone())
                    .unwrap_or(Value::Null);
                let preset = value["authoring"]["presets"]
                    .as_array()
                    .and_then(|all| {
                        all.iter()
                            .find(|p| p["default"] == true && p["variant"] == "reference")
                    })
                    .map(|p| p["key"].clone())
                    .unwrap_or(Value::Null);
                value["mcp_defaults"] = json!({"duration_seconds":DEFAULT_SECONDS,"canvas":default_canvas(),
                    "prompt_profile":profile,"scene_preset":preset,"add_music":false,"add_dialogue":false,
                    "character_sheet_profile":"qwen-character-sheet",
                    "characters_without_images":{
                        "interactive":"ask after the outline is accepted: character sheet or described in the prompts",
                        "headless":"character sheet when the character appears in more than one scene, otherwise described in the prompts"}});
                value
            }
            ("video_project", "create") => self.create_project(own(ws)?, args).await?,
            ("video_project", "get") => project_view(
                &self
                    .owned_project(own(ws)?, ident(args, "project_id")?)
                    .await?,
            ),
            ("video_project", "edit") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!("/v1/video/projects/{project}/edits");
                let request = body(args, "request")?;
                project_view(
                    &self
                        .call_video(Method::POST, &path, Some(&request), None)
                        .await?,
                )
            }
            ("video_project", "save") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!("/v1/video/projects/{project}/save");
                let request =
                    json!({"schema_version":1,"expected_project_revision":revision(args)?});
                saved(
                    &self
                        .call_video(Method::POST, &path, Some(&request), None)
                        .await?,
                )
            }
            ("video_register_reference", _) => {
                self.register_video_reference(own(ws)?, args).await?
            }
            ("video_storyboard", "start") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!("/v1/video/projects/{project}/authoring");
                let mut request = body(args, "request")?;
                // The MCP path never starts the local authoring provider or the GPU.
                request["provider"] = json!("external");
                request["provider_profile_id"] = json!("");
                request["operation"] = json!("outline");
                if request["duration_seconds"].is_null() {
                    request["duration_seconds"] = json!(DEFAULT_SECONDS);
                }
                self.call_video(Method::POST, &path, Some(&request), None)
                    .await?
            }
            ("video_storyboard", "submit") => {
                let draft = self.owned_draft(own(ws)?, args).await?;
                let path = format!("/v1/video/authoring/{draft}/submission");
                let request = body(args, "submission")?;
                let value = self
                    .call_video(Method::POST, &path, Some(&request), None)
                    .await?;
                // A commit returns the whole project snapshot; a dry run returns its report.
                if value["project"].is_object() {
                    let mut view = project_view(&value);
                    view["scene_map"] = value["scene_map"].clone();
                    view
                } else {
                    value
                }
            }
            ("video_storyboard", "list") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!("/v1/video/projects/{project}/authoring");
                self.call_video(Method::GET, &path, None, None).await?
            }
            ("video_storyboard", "bundle") => {
                let draft = self.owned_draft(own(ws)?, args).await?;
                let path = format!("/v1/video/authoring/{draft}/bundle");
                self.call_video(Method::GET, &path, None, None).await?
            }
            ("video_storyboard", "get") => {
                let draft = self.owned_draft(own(ws)?, args).await?;
                let path = format!("/v1/video/authoring/{draft}");
                self.call_video(Method::GET, &path, None, None).await?
            }
            ("video_storyboard", "review") => {
                let draft = self.owned_draft(own(ws)?, args).await?;
                let path = format!("/v1/video/authoring/{draft}/review");
                let request = body(args, "review")?;
                self.call_video(Method::POST, &path, Some(&request), None)
                    .await?
            }
            ("video_run", "start") => {
                let key = self.video_key(args)?;
                let request = body(args, "request")?;
                let project = ident(&request, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                self.call_video(Method::POST, "/v1/video/runs", Some(&request), Some(&key))
                    .await?
            }
            ("video_run", "plan") => {
                let request = body(args, "request")?;
                let project = ident(&request, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                self.call_video(Method::POST, "/v1/video/runs/plan", Some(&request), None)
                    .await?
            }
            ("video_take", "list") => {
                let snapshot = self
                    .owned_project(own(ws)?, ident(args, "project_id")?)
                    .await?;
                let scene = match &args["scene_id"] {
                    Value::Null => None,
                    _ => Some(ident(args, "scene_id")?),
                };
                take_list(&snapshot, scene)
            }
            ("video_take", "view") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!(
                    "/v1/video/projects/{project}/takes/{}/frames",
                    ident(args, "take_id")?
                );
                self.call_video(Method::GET, &path, None, None).await?
            }
            ("video_take", "select") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!(
                    "/v1/video/projects/{project}/takes/{}/select",
                    ident(args, "take_id")?
                );
                let restore = match &args["restore_prompt"] {
                    Value::Null => false,
                    value => value
                        .as_bool()
                        .ok_or_else(|| validation("restore_prompt must be true or false"))?,
                };
                let request = json!({"schema_version":1,"expected_project_revision":revision(args)?,
                    "restore_prompt":restore});
                self.call_video(Method::POST, &path, Some(&request), None)
                    .await?
            }
            ("video_run", "get") => {
                let path = format!("/v1/video/runs/{}", ident(args, "run_id")?);
                check_wait(args)?;
                // The first read checks the run's project and counts as the first poll.
                let first = self.call_video(Method::GET, &path, None, None).await?;
                self.owned_project(own(ws)?, ident(&first, "project_id")?)
                    .await?;
                let mut run = if args["wait"].as_bool() == Some(true) && !terminal_run(&first) {
                    self.poll(
                        args,
                        || self.call_video(Method::GET, &path, None, None),
                        terminal_run,
                    )
                    .await?
                } else {
                    first
                };
                // A failed run names the scene and its job's bounded diagnostics.
                if run["status"] == "failed"
                    && let Some(job) = run["current_job"].as_str().map(str::to_owned)
                {
                    let scene = run["targets"]
                        .as_array()
                        .and_then(|targets| targets.iter().find(|t| t["job_id"] == job.as_str()))
                        .map(|t| t["scene_id"].clone())
                        .unwrap_or(Value::Null);
                    let detail = self
                        .call_video(Method::GET, &format!("/v1/jobs/{job}"), None, None)
                        .await?;
                    run["failed_scene"] =
                        json!({"scene_id":scene,"job_id":job,"error":job_error(&detail)});
                }
                run
            }
            ("video_run", "stop") => {
                let run = ident(args, "run_id")?;
                let current = self
                    .call_video(Method::GET, &format!("/v1/video/runs/{run}"), None, None)
                    .await?;
                self.owned_project(own(ws)?, ident(&current, "project_id")?)
                    .await?;
                let path = format!("/v1/video/runs/{run}/stop");
                let cancel = args["cancel_current"]
                    .as_bool()
                    .ok_or_else(|| validation("cancel_current must be true or false"))?;
                let request = json!({"schema_version":1,"cancel_current":cancel});
                self.call_video(Method::POST, &path, Some(&request), None)
                    .await?
            }
            ("video_assembly", "plan") => {
                let project = ident(args, "project_id")?;
                self.owned_project(own(ws)?, project).await?;
                let path = format!("/v1/video/projects/{project}/assembly");
                let mut request = body(args, "request")?;
                if let Some(request) = request.as_object_mut() {
                    request.entry("project_id").or_insert(json!(project));
                }
                self.call_video(Method::POST, &path, Some(&request), None)
                    .await?
            }
            ("video_assembly", "submit") => {
                let plan = ident(args, "plan_id")?;
                self.owned_plan(own(ws)?, plan).await?;
                let path = format!("/v1/video/assemblies/{plan}/jobs");
                let key = self.video_key(args)?;
                self.call_video(Method::POST, &path, None, Some(&key))
                    .await?
            }
            ("video_assembly", "get") => {
                let plan = ident(args, "plan_id")?;
                self.owned_plan(own(ws)?, plan).await?;
                self.poll(args, || self.assembly(plan), |a| a["terminal"] == true)
                    .await?
            }
            (_, action) => {
                return Err(validation(format!(
                    "unknown action for {name}: {}",
                    if action.is_empty() { "missing" } else { action }
                )));
            }
        };
        let mut value = bounded(value);
        let images = take_images(&mut value)?;
        // Outputs inside the workspace become file references; other paths go.
        self.scrub(&mut value, ws);
        Ok((value, images))
    }
    /// The Draft's ID, if its project belongs to this workspace.
    async fn owned_draft<'a>(
        &self,
        workspace: &Bound,
        args: &'a Value,
    ) -> std::result::Result<&'a str, Fault> {
        let draft = ident(args, "draft_id")?;
        let value = self
            .call_video(
                Method::GET,
                &format!("/v1/video/authoring/{draft}"),
                None,
                None,
            )
            .await?;
        let project = value["project_id"]
            .as_str()
            .or(value["draft"]["project_id"].as_str())
            .ok_or_else(|| Fault::new("not_found", "draft_id is not a Draft of this workspace"))?;
        self.owned_project(workspace, project).await?;
        Ok(draft)
    }
    async fn owned_plan(&self, workspace: &Bound, plan: &str) -> std::result::Result<(), Fault> {
        let value = self
            .call_video(
                Method::GET,
                &format!("/v1/video/assemblies/{plan}"),
                None,
                None,
            )
            .await?;
        let project = value["plan"]["project_id"]
            .as_str()
            .ok_or_else(|| Fault::new("not_found", "plan_id is not a plan of this workspace"))?;
        self.owned_project(workspace, project).await.map(|_| ())
    }
    /// Read until the status is terminal or the bounded wait elapses; never longer than 60 s.
    async fn poll<F>(
        &self,
        args: &Value,
        read: impl Fn() -> F,
        terminal: impl Fn(&Value) -> bool,
    ) -> std::result::Result<Value, Fault>
    where
        F: std::future::Future<Output = std::result::Result<Value, Fault>>,
    {
        check_wait(args)?;
        let deadline = Instant::now() + wait(args);
        let epoch = self.inner.epoch.load(std::sync::atomic::Ordering::SeqCst);
        loop {
            let value = read().await?;
            let now = Instant::now();
            if terminal(&value) || now >= deadline {
                return Ok(value);
            }
            if epoch != self.inner.epoch.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(Fault::new("privacy", "Imagine session ended"));
            }
            let interval = Duration::from_secs_f64(self.inner.config.poll_interval.max(0.01));
            tokio::time::sleep(interval.min(deadline - now)).await;
        }
    }
    /// The assembly plan's progress plus its CPU job status and published output, if submitted.
    async fn assembly(&self, plan: &str) -> std::result::Result<Value, Fault> {
        let path = format!("/v1/video/assemblies/{plan}");
        let mut value = self.call_video(Method::GET, &path, None, None).await?;
        let Some(job) = value["job_id"].as_str().map(str::to_owned) else {
            value["terminal"] = json!(false);
            return Ok(value);
        };
        let job = self
            .call_video(Method::GET, &format!("/v1/jobs/{job}"), None, None)
            .await?;
        let output = &job["output"];
        let mut summary = json!({"id":job["id"],"status":job["status"]});
        if job["error"].is_object() {
            summary["error"] = job_error(&job);
        }
        if output.is_object() {
            summary["output"] = json!({"format":output["format"],"saved_path":output["saved_path"],
                "assets":output["assets"],"video":output["video"]});
        }
        value["terminal"] = json!(matches!(
            job["status"].as_str(),
            Some("succeeded" | "failed" | "cancelled" | "interrupted")
        ));
        value["job"] = summary;
        Ok(value)
    }
    async fn create_project(
        &self,
        workspace: &Bound,
        args: &Value,
    ) -> std::result::Result<Value, Fault> {
        let title = text(args, "title", 80, false)?.to_owned();
        let project = match &args["project"] {
            Value::Null => Value::Null,
            Value::Object(_) => args["project"].clone(),
            _ => return Err(validation("project must be an object or null")),
        };
        let canvas = match &args["canvas"] {
            // MCP productions default to Draft; the agent changes it only when the user asks.
            Value::Null if project.is_null() => default_canvas(),
            Value::Null => Value::Null,
            Value::Object(_) if project.is_null() => args["canvas"].clone(),
            Value::Object(_) => return Err(validation("give either project or canvas, not both")),
            _ => return Err(validation("canvas must be an object or null")),
        };
        let snapshot = self
            .call_video(
                Method::POST,
                &format!("/v1/workspaces/{}/video-projects", workspace.id),
                Some(&json!({"schema_version":1,"title":title,"project":project,"canvas":canvas})),
                None,
            )
            .await?;
        Ok(project_view(&snapshot))
    }
    async fn register_video_reference(
        &self,
        workspace: &Bound,
        args: &Value,
    ) -> std::result::Result<Value, Fault> {
        let project = ident(args, "project_id")?;
        self.owned_project(workspace, project).await?;
        let expected = revision(args)?;
        let name = text(args, "name", 128, false)?.to_owned();
        let cast = match &args["cast"] {
            Value::Null => None,
            cast @ Value::Object(_) => {
                let alias = text(cast, "alias", 64, true)?
                    .trim()
                    .trim_start_matches('@');
                if alias.is_empty() || alias.chars().any(char::is_whitespace) {
                    return Err(validation("cast alias must be one word, with or without @"));
                }
                Some((
                    text(cast, "name", 128, true)?.to_owned(),
                    alias.to_owned(),
                    text(cast, "description", 4000, false)?.to_owned(),
                ))
            }
            _ => {
                return Err(validation(
                    "cast must be an object with name, alias and description",
                ));
            }
        };
        let file = args["file_id"]
            .as_str()
            .ok_or_else(|| validation("file_id must be a file ID from this workspace"))?;
        let (path, _) = self.file_path(workspace, file).await?;
        let roots = vec![workspace.folder.clone()];
        let (path, digest, bytes) = tokio::task::spawn_blocking(move || {
            let path = path.canonicalize()?;
            let original = references::read_image(&path, &roots)?;
            let reader =
                image::ImageReader::new(std::io::Cursor::new(&original)).with_guessed_format()?;
            if !matches!(
                reader.format(),
                Some(image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP)
            ) {
                anyhow::bail!("reference image must be PNG, JPEG or WebP");
            }
            reader.into_dimensions()?;
            Ok((path, references::hash(&original), original.len() as u64))
        })
        .await
        .map_err(|_| validation("reference could not be read or decoded"))?
        .map_err(read_failure)?;
        let linked = self.client.link_video(&path).await.map_err(rejected)?;
        if linked["sha256"].as_str() != Some(digest.as_str())
            || linked["bytes"].as_u64() != Some(bytes)
        {
            return Err(validation(
                "registered reference has changed; register the image again",
            ));
        }
        let session_asset = ident(&linked, "id")
            .map_err(|_| validation("invalid reference storage response"))?
            .to_owned();
        let project_path = format!("/v1/video/projects/{project}");
        let before = self
            .call_video(Method::GET, &project_path, None, None)
            .await?;
        let name = if name.is_empty() {
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            name
        };
        let attached = self
            .call_video(
                Method::POST,
                &format!("{project_path}/assets"),
                Some(&json!({"schema_version":1,"expected_project_revision":expected,"asset_id":session_asset,"name":name})),
                None,
            )
            .await?;
        // The project asset is the new entry bound to this session asset.
        let known = |s: &Value, id: &str| s["project"]["assets"].get(id).is_some();
        let added = attached["runtime_assets"]
            .as_object()
            .map(|bindings| {
                bindings
                    .iter()
                    .filter(|(id, linked)| {
                        linked.as_str() == Some(&session_asset) && !known(&before, id)
                    })
                    .map(|(id, _)| id.clone())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let [asset] = added.as_slice() else {
            return Err(Fault::new("imagine", "attached project asset is ambiguous"));
        };
        let mut result = json!({"asset_id":asset,"name":name});
        let mut snapshot = attached;
        if let Some((cast_name, alias, description)) = cast {
            let id = format!("reference_mcp_{}", short_hash(&alias.to_lowercase()));
            let reference = json!({"id":id,"name":cast_name,"alias":alias,"kind":"character",
                "description":description,"tags":[],"enabled":true,"strength":1.0,"retention":"",
                "images":[{"id":format!("image_mcp_{}",short_hash(asset)),"asset":asset,"enabled":true,"preview":true}],
                "voice":null,"origin":null});
            let request = json!({"schema_version":1,"expected_project_revision":snapshot["project"]["project_revision"],
                "commands":[{"operation":"upsert_visual_reference","reference":reference}]});
            snapshot = self
                .call_video(
                    Method::POST,
                    &format!("{project_path}/edits"),
                    Some(&request),
                    None,
                )
                .await
                .map_err(|mut fault| {
                    fault.message = format!(
                        "{}; the image is attached as asset {asset}, the cast entry was not added",
                        fault.message
                    );
                    fault
                })?;
            result["cast"] = json!({"reference_id":id,"alias":alias,"tag":format!("@{alias}")});
        }
        let summary = saved(&snapshot);
        result["project_id"] = summary["project_id"].clone();
        result["project_revision"] = summary["project_revision"].clone();
        result["persistence"] = summary["persistence"].clone();
        Ok(result)
    }
}
/// A job's stable error code plus its bounded diagnostics: enum names, a class
/// name and integers only. Anything else in the job's error is dropped.
fn job_error(job: &Value) -> Value {
    let error = &job["error"];
    let code = error["code"]
        .as_str()
        .filter(|code| crate::imagine::stable_code(code));
    let mut out = json!({"code":code});
    let token = |value: &Value, max: usize, upper: bool| {
        value
            .as_str()
            .filter(|s| {
                !s.is_empty()
                    && s.len() <= max
                    && s.bytes().all(|b| {
                        b.is_ascii_lowercase() || b == b'_' || (upper && b.is_ascii_alphanumeric())
                    })
            })
            .map(str::to_owned)
    };
    if let Some(stage) = token(&error["stage"], 40, false) {
        out["stage"] = json!(stage);
    }
    if let Some(kind) = token(&error["kind"], 40, false) {
        out["kind"] = json!(kind);
    }
    if let Some(class) = token(&error["exception"], 48, true) {
        out["exception"] = json!(class);
    }
    if let Some(code) = error["accelerator_code"].as_i64() {
        out["accelerator_code"] = json!(code);
    }
    if let Some(sizes) = error["sizes"].as_array().filter(|s| s.len() <= 8) {
        let sizes = sizes.iter().filter_map(Value::as_i64).collect::<Vec<_>>();
        if !sizes.is_empty() {
            out["sizes"] = json!(sizes);
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn time_limits_follow_the_action_table() {
        let seconds = |name: &str, args: Value| limit(name, &args).map(|d| d.as_secs());
        assert_eq!(seconds("generate", json!({})), None);
        assert_eq!(seconds("get_job", json!({"wait":true})), None);
        for (name, action, expected) in [
            ("list_video_profiles", "", 5),
            ("describe_video_profile", "", 5),
            ("video_capabilities", "", 5),
            ("video_project", "get", 5),
            ("video_storyboard", "get", 5),
            ("video_storyboard", "list", 5),
            ("video_storyboard", "start", 60),
            ("video_storyboard", "submit", 60),
            ("video_storyboard", "bundle", 60),
            ("video_storyboard", "review", 15),
            ("video_project", "create", 15),
            ("video_project", "edit", 15),
            ("video_project", "save", 15),
            ("video_register_reference", "", 15),
            ("video_run", "start", 15),
            ("video_run", "plan", 15),
            ("video_run", "stop", 15),
            ("video_take", "list", 5),
            ("video_take", "view", 60),
            ("video_take", "select", 15),
            ("video_run", "get", 5),
            ("video_assembly", "plan", 15),
            ("video_assembly", "submit", 15),
            ("video_assembly", "get", 5),
        ] {
            assert_eq!(
                seconds(name, json!({"action":action})),
                Some(expected),
                "{name} {action}"
            );
        }
        for name in ["video_run", "video_assembly"] {
            assert_eq!(seconds(name, json!({"action":"get","wait":true})), Some(35));
            assert_eq!(
                seconds(name, json!({"action":"get","wait":true,"wait_seconds":600})),
                Some(65)
            );
            assert_eq!(
                seconds(name, json!({"action":"get","wait":false,"wait_seconds":20})),
                Some(5)
            );
        }
    }
}
