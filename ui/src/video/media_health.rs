//! Explicit Check Media and two-phase Relink UI. Epoch, revision and nonce guard all replies.
use super::*;
#[cfg(test)]
use crate::test_destination::task_fixtures;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

struct Cancellation(Arc<AtomicBool>);
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
#[derive(Clone, Copy)]
pub(in crate::app) enum Reply {
    Check,
    Preview,
    Apply,
}
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    project: Option<ProjectId>,
    revision: u64,
    nonce: u64,
    pending: bool,
    report: Option<Value>,
    preview: Option<Value>,
    error: Option<String>,
    cancel: Option<Cancellation>,
    apply_nonce: Option<u64>,
}
impl State {
    #[cfg(feature = "ui-preview")]
    pub(super) fn pending(&self) -> bool {
        self.pending
    }
    fn close(&mut self) {
        self.cancel = None;
        self.open = false;
        self.nonce = self.nonce.wrapping_add(1);
        self.pending = false;
        self.preview = None;
    }
    fn begin(&mut self) -> u64 {
        self.cancel = Some(Cancellation(Arc::new(AtomicBool::new(false))));
        self.nonce = self.nonce.wrapping_add(1);
        self.pending = true;
        self.error = None;
        self.nonce
    }
    fn cancel_pending(&mut self) {
        if self.apply_nonce.is_some() {
            return;
        }
        self.cancel = None;
        self.nonce = self.nonce.wrapping_add(1);
        self.pending = false;
        self.preview = None;
    }
    fn take_apply(&mut self, nonce: u64) -> bool {
        if self.apply_nonce != Some(nonce) {
            return false;
        }
        self.apply_nonce = None;
        true
    }
    #[cfg(test)]
    fn current(&self, project: &VideoProject, nonce: u64) -> bool {
        self.open
            && nonce == self.nonce
            && self.project.as_ref() == Some(&project.id)
            && self.revision == project.project_revision
    }
    fn receive(
        &mut self,
        project: &VideoProject,
        nonce: u64,
        kind: Reply,
        result: Result<Value, String>,
    ) -> Option<Value> {
        if !self.open || nonce != self.nonce || self.project.as_ref() != Some(&project.id) {
            return None;
        }
        self.pending = false;
        self.cancel = None;
        if self.revision != project.project_revision {
            self.preview = None;
            self.error = Some("Project changed. Check media again.".into());
            return None;
        }
        match result {
            Err(error) => {
                self.error = Some(error);
                self.preview = None;
            }
            Ok(value) => match kind {
                Reply::Check => {
                    if value["project_revision"].as_u64() != Some(self.revision) {
                        self.error = Some("Project changed. Check media again.".into());
                    } else {
                        self.report = Some(value);
                        self.preview = None;
                    }
                }
                Reply::Preview => {
                    if value["cancelled"] == true {
                        self.preview = None;
                    } else {
                        self.preview = Some(value);
                    }
                }
                Reply::Apply => {
                    self.close();
                    return Some(value["snapshot"].clone());
                }
            },
        }
        None
    }
}
async fn wait_cancel(flag: &AtomicBool) {
    while !flag.load(Ordering::Acquire) {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}
async fn replacement_preview(
    client: &client::Client,
    session: &str,
    project: &ProjectId,
    mut request: Value,
    selection: Option<PathBuf>,
    cancel: Arc<AtomicBool>,
) -> Result<Value, client::TimelineFailure> {
    if cancel.load(Ordering::Acquire) || selection.is_none() {
        return Ok(json!({"cancelled":true}));
    }
    let path = selection.unwrap();
    let label = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Selected replacement".into());
    let prepare = async {
        let uploaded = client
            .upload_media_cancellable(session, &path, cancel.clone())
            .await?;
        if cancel.load(Ordering::Acquire) {
            return Ok(json!({"cancelled":true}));
        }
        request["replacement_asset_id"] = uploaded["id"].clone();
        let mut preview = client
            .timeline_request(
                session,
                true,
                &format!("/v1/video/projects/{}/media/relink-preview", project.0),
                Some(&request),
            )
            .await?;
        preview["replacement_label"] = json!(label);
        Ok(preview)
    };
    tokio::select! {
        biased;
        _=wait_cancel(&cancel)=>Ok(json!({"cancelled":true})),
        result=prepare=>result,
    }
}
impl ImagineApp {
    pub(super) fn timeline_media_health_open(&mut self, ctx: &egui::Context) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        let project = document.project.clone();
        if self.timeline.busy {
            return;
        }
        let nonce = self.timeline.media_health.nonce.wrapping_add(1);
        self.timeline.media_health = State {
            open: true,
            nonce,
            project: Some(project.id.clone()),
            revision: project.project_revision,
            ..Default::default()
        };
        self.timeline_media_health_call(
            ctx,
            Reply::Check,
            json!({"schema_version":1,"expected_project_revision":project.project_revision}),
        );
    }
    fn timeline_media_health_call(&mut self, ctx: &egui::Context, kind: Reply, body: Value) {
        let (Some(client), Some(session), Some(project)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.media_health.project.clone(),
        ) else {
            return;
        };
        if matches!(kind, Reply::Apply) {
            if self.timeline.busy {
                return;
            }
            self.timeline.busy = true;
        }
        let nonce = self.timeline.media_health.begin();
        let cancel = self
            .timeline
            .media_health
            .cancel
            .as_ref()
            .unwrap()
            .0
            .clone();
        if matches!(kind, Reply::Apply) {
            self.timeline.media_health.apply_nonce = Some(nonce);
            self.timeline.media_health.cancel = None;
        }
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        let operation = match kind {
            Reply::Check => "health",
            Reply::Preview => "relink-preview",
            Reply::Apply => "relink",
        };
        let path = format!("/v1/video/projects/{}/media/{operation}", project.0);
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                if matches!(kind, Reply::Apply) {
                    return client
                        .timeline_request(&session, true, &path, Some(&body))
                        .await;
                }
                tokio::select! {
                    biased;
                    _=wait_cancel(&cancel)=>Ok(json!({"cancelled":true})),
                    result=client.timeline_request(&session,true,&path,Some(&body))=>result,
                }
            });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::MediaHealth {
                    epoch,
                    nonce,
                    kind,
                    result,
                }),
            );
        });
    }
    fn timeline_media_health_choose_replacement(
        &mut self,
        ctx: &egui::Context,
        asset: AssetId,
        kind: MediaKind,
    ) {
        let (Some(client), Some(session), Some(document)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.document.as_ref(),
        ) else {
            return;
        };
        let Some(source) = document.project.assets.get(&asset) else {
            return;
        };
        let request = json!({"schema_version":1,"expected_project_revision":document.project.project_revision,"asset_id":asset,"expected_asset_revision":source.revision});
        let project = document.project.id.clone();
        let nonce = self.timeline.media_health.begin();
        let cancel = self
            .timeline
            .media_health
            .cancel
            .as_ref()
            .unwrap()
            .0
            .clone();
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        #[cfg(feature = "ui-preview")]
        let qa_directory = super::qa::picker_directory(self);
        std::thread::spawn(move || {
            let dialog = rfd::FileDialog::new().set_title("Choose replacement media");
            #[cfg(feature = "ui-preview")]
            let dialog = super::qa::configure_picker(dialog, qa_directory);
            let dialog = match kind {
                MediaKind::Image => dialog.add_filter("Images", &["png", "jpg", "jpeg", "webp"]),
                MediaKind::Video => dialog.add_filter("Video", &["mp4", "mov", "mkv", "webm"]),
                MediaKind::Audio => dialog.add_filter("Audio", &["wav", "flac", "mp3", "ogg"]),
            };
            let selection = dialog.pick_file();
            let result = runtime().block_on(replacement_preview(
                &client, &session, &project, request, selection, cancel,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::MediaHealth {
                    epoch,
                    nonce,
                    kind: Reply::Preview,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_media_health_reply(
        &mut self,
        ctx: &egui::Context,
        nonce: u64,
        kind: Reply,
        result: Result<Value, client::TimelineFailure>,
    ) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        if matches!(kind, Reply::Apply) {
            if !self.timeline.media_health.take_apply(nonce) {
                return;
            }
            self.timeline.busy = false;
            // Epoch was checked by Event dispatch. Closing a dialog cannot discard a committed
            // mutation. Refresh authoritative state after an uncertain network result.
            self.timeline.media_health.close();
            let raw = match result {
                Ok(value) => Some(value["snapshot"].clone()),
                Err(error) => {
                    self.timeline.error = Some(error.message);
                    error.snapshot
                }
            };
            if let Some(raw) = raw {
                let current = self
                    .timeline
                    .document
                    .as_ref()
                    .map(|d| (&d.project.id.0, d.project.project_revision));
                if current.is_some_and(|(id, revision)| {
                    raw["project"]["id"].as_str() != Some(id.as_str())
                        || raw["project"]["project_revision"]
                            .as_u64()
                            .is_none_or(|v| v < revision)
                }) {
                    self.timeline_refresh(ctx);
                    return;
                }
                self.timeline_message(
                    ctx,
                    Event::Document {
                        epoch: self.timeline.epoch,
                        replace: false,
                        before: None,
                        history: 0,
                        result: Ok(raw),
                    },
                );
                self.timeline.thumbnails = Default::default();
            } else {
                self.timeline_refresh(ctx);
            }
            return;
        }
        let _ = self.timeline.media_health.receive(
            &document.project,
            nonce,
            kind,
            result.map_err(|e| e.message),
        );
    }
    pub(super) fn timeline_media_health_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.media_health.open {
            return;
        }
        #[cfg(feature = "ui-preview")]
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.timeline.media_health.close();
            return;
        }
        let Some(document) = self.timeline.document.clone() else {
            self.timeline.media_health.close();
            return;
        };
        let stale = self.timeline.media_health.project.as_ref() != Some(&document.project.id)
            || self.timeline.media_health.revision != document.project.project_revision;
        let mut open = true;
        let mut check = false;
        let mut preview = None;
        let mut choose = None;
        let mut apply = None;
        let mut back = false;
        let mut cancel = false;
        let privacy = self.prefs.privacy_mode;
        egui::Window::new("Check Media / Relink").open(&mut open).default_width(550.).max_height(620.).show(ctx,|ui|{
            #[cfg(feature="ui-preview")] {
                let r=ui.interact(ui.clip_rect(),ui.id().with("qa-media-health-scroll"),egui::Sense::hover());
                super::qa::control(ui,"media-health-scroll".into(),&r);
            }
            let hidden=!helto::reveal_in_viewport(ui,privacy);
            helto::private_scope(ui,hidden,|ui|{
                let state=&self.timeline.media_health;
                if stale{ui.label("Project changed. Check media again.");}
                if let Some(error)=&state.error{ui.colored_label(helto::ERROR,error);}
                if state.pending{
                    ui.spinner();ui.label("Checking media…");
                    if state.apply_nonce.is_none() {
                        let r=ui.button("Cancel");
                        #[cfg(feature="ui-preview")] super::qa::control(ui,"media-health-cancel".into(),&r);
                        if r.clicked(){cancel=true;}
                    }
                }
                if let Some(value)=&state.preview{
                    ui.strong("Relink impact preview");
                    let label = |asset: &Asset| if asset.name.is_empty() { asset.id.0.clone() } else { asset.name.clone() };
                    let current = value["request"]["asset_id"].as_str().and_then(|id| document.project.assets.get(&AssetId(id.into()))).map(label).unwrap_or_else(|| "Current media".into());
                    let replacement = value["replacement_label"].as_str().map(str::to_owned).or_else(|| document.project.assets.values().find(|asset| document.raw["runtime_assets"][&asset.id.0] == value["request"]["replacement_asset_id"]).map(label)).unwrap_or_else(|| "Selected replacement".into());
                    ui.label(format!("Current: {current}"));
                    ui.label(format!("Replacement: {replacement}"));
                    ui.label("Asset identity and listed bindings are kept. Existing Takes keep their source revision.");
                    egui::ScrollArea::vertical().max_height(300.).show(ui,|ui|{for (index,impact) in value["impacts"].as_array().into_iter().flatten().enumerate(){if hidden{ui.label(format!("Affected binding {}",index+1));}else{ui.label(format!("{} · {} · {}",impact["kind"].as_str().unwrap_or(""),impact["owner"].as_str().unwrap_or(""),impact["binding"].as_str().unwrap_or("")));}}});
                    ui.horizontal(|ui|{let r=ui.add_enabled(!state.pending,egui::Button::new("Back"));
                    #[cfg(feature="ui-preview")] super::qa::control(ui,"media-health-back".into(),&r);
                    if r.clicked(){back=true;}
                    let r=ui.add_enabled(!state.pending && !stale,egui::Button::new("Relink asset"));
                    #[cfg(feature="ui-preview")] super::qa::control(ui,"media-health-apply".into(),&r);
                    if r.clicked(){apply=Some(value["request"].clone());}});
                }else if let Some(report)=&state.report{
                    egui::ScrollArea::vertical().max_height(390.).show(ui,|ui|{for (index,entry) in report["entries"].as_array().into_iter().flatten().enumerate(){
                        let id=entry["asset_id"].as_str().unwrap_or("");let Some(asset)=document.project.assets.get(&AssetId(id.into()))else{continue;};
                        ui.group(|ui|{ui.horizontal_wrapped(|ui|{ui.strong(if hidden{format!("Media {}",index+1)}else if asset.name.is_empty(){asset.id.0.clone()}else{asset.name.clone()});ui.label(entry["status"].as_str().unwrap_or("Unavailable"));});
                            ui.label(format!("Affected bindings: {}",entry["impacts"].as_array().map_or(0,Vec::len)));
                            // Replacement uses another explicitly imported project media asset's immutable runtime descriptor.
                            // Importing goes through the existing contextual media picker; there are no raw path fields.
                            ui.add_enabled_ui(!state.pending && !stale,|ui|{let menu=ui.menu_button("Relink…",|ui|{
                                let r=ui.button("Choose replacement file…");
                                #[cfg(feature="ui-preview")] super::qa::control(ui,format!("relink-file-{}",asset.id.0),&r);
                                if r.clicked(){choose=Some((asset.id.clone(),asset.kind.clone()));ui.close();}
                                ui.separator();
                                for replacement in document.project.assets.values().filter(|a|a.id!=asset.id && a.kind==asset.kind){if let Some(runtime)=document.raw["runtime_assets"][&replacement.id.0].as_str(){let label=if hidden{"Replacement media".into()}else if replacement.name.is_empty(){replacement.id.0.clone()}else{replacement.name.clone()};let r=ui.button(label);
                                #[cfg(feature="ui-preview")] super::qa::control(ui,format!("relink-replacement-{}",replacement.id.0),&r);
                                if r.clicked(){preview=Some(json!({"schema_version":1,"expected_project_revision":document.project.project_revision,"asset_id":asset.id,"expected_asset_revision":asset.revision,"replacement_asset_id":runtime}));ui.close();}}}});
                            #[cfg(feature="ui-preview")] super::qa::control(ui,format!("relink-{}",asset.id.0),&menu.response);
                            let _=menu;});
                        });
                    }});
                }
                ui.add_enabled_ui(!state.pending,|ui|{
                    let r=helto::icon_button(ui,crate::icons::Icon::Retry,"Check media",false);
                    #[cfg(feature="ui-preview")] super::qa::control(ui,"media-health-check".into(),&r);
                    if r.clicked(){check=true;}
                });
            });
        });
        if !open {
            self.timeline.media_health.close();
            return;
        }
        if cancel {
            self.timeline.media_health.cancel_pending();
            return;
        }
        if back {
            self.timeline.media_health.preview = None;
        }
        if check {
            self.timeline_media_health_open(ctx);
        } else if let Some((asset, kind)) = choose {
            self.timeline_media_health_choose_replacement(ctx, asset, kind);
        } else if let Some(body) = preview {
            self.timeline_media_health_call(ctx, Reply::Preview, body);
        } else if let Some(body) = apply {
            self.timeline_media_health_call(ctx, Reply::Apply, body);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn media_health_late_close_and_changed_revision_are_ignored() {
        let p = VideoProject::create(
            serde_json::from_str(include_str!(
                "../../../design/video-contract/create-project.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let mut s = State {
            open: true,
            project: Some(p.id.clone()),
            revision: p.project_revision,
            ..Default::default()
        };
        let n = s.begin();
        s.close();
        assert!(!s.current(&p, n));
        s.open = true;
        let n = s.begin();
        let mut changed = p.clone();
        changed.project_revision += 1;
        assert!(!s.current(&changed, n));
        assert!(
            s.receive(
                &p,
                n,
                Reply::Check,
                Ok(json!({"project_revision":p.project_revision,"entries":[]}))
            )
            .is_none()
        );
        assert!(s.report.is_some());
        let n = s.begin();
        s.receive(&p, n, Reply::Preview, Ok(json!({"cancelled":true})));
        assert!(s.preview.is_none());
        assert!(s.error.is_none());
        assert!(s.report.is_some());
    }
}

#[cfg(test)]
#[path = "media_health_ui_tests.rs"]
pub(in crate::app::timeline) mod ui_evidence;

#[cfg(test)]
mod selector_tests {
    use super::*;
    fn project() -> VideoProject {
        VideoProject::create(
            serde_json::from_str(include_str!(
                "../../../design/video-contract/create-project.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn selector_cancel_supersede_and_drop_invalidate_all_late_results() {
        let project = project();
        let mut state = State {
            open: true,
            project: Some(project.id.clone()),
            revision: project.project_revision,
            ..Default::default()
        };
        let old = state.begin();
        let token = state.cancel.as_ref().unwrap().0.clone();
        let current = state.begin();
        assert!(token.load(Ordering::Acquire));
        state.receive(&project, old, Reply::Preview, Err("old failure".into()));
        assert!(state.error.is_none());
        assert!(state.pending);
        let token = state.cancel.as_ref().unwrap().0.clone();
        state.cancel_pending();
        assert!(token.load(Ordering::Acquire));
        assert!(!state.pending);
        state.receive(&project, current, Reply::Preview, Ok(json!({"request":{}})));
        assert!(state.preview.is_none());
        let nonce = state.begin();
        state.receive(
            &project,
            nonce,
            Reply::Preview,
            Err("own current failure".into()),
        );
        assert_eq!(state.error.as_deref(), Some("own current failure"));
        assert!(!state.pending);
        let _ = state.begin();
        let token = state.cancel.as_ref().unwrap().0.clone();
        drop(state);
        assert!(token.load(Ordering::Acquire));
    }
    #[test]
    fn selector_ui_exposes_cancel_only_before_confirmed_apply() {
        for applying in [false, true] {
            let ctx = egui::Context::default();
            let root = PathBuf::from(
                std::env::var_os("IMAGINE_WORKER_A_FIXTURES").expect("own encrypted fixtures"),
            )
            .canonicalize()
            .unwrap();
            assert!(task_fixtures::is_owned(&root));
            let mut app = ImagineApp::synthetic_preview(&ctx, root);
            app.client = None;
            let project = project();
            app.timeline.document = Some(Snapshot {
                raw: json!({}),
                project: project.clone(),
            });
            app.timeline.media_health = State {
                open: true,
                project: Some(project.id.clone()),
                revision: project.project_revision,
                ..Default::default()
            };
            let nonce = app.timeline.media_health.begin();
            if applying {
                app.timeline.media_health.apply_nonce = Some(nonce);
                app.timeline.media_health.cancel = None;
            }
            let evidence = super::ui_evidence::capture(
                &ctx,
                &format!("selector-pending-{applying}"),
                600.,
                760.,
                |ui| app.timeline_media_health_window(ui.ctx()),
            );
            assert_eq!(
                evidence["nodes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|node| node["label"] == "Cancel"),
                !applying
            );
        }
    }
    #[test]
    fn confirmed_apply_reconciles_after_close_but_only_for_its_nonce() {
        let mut state = State {
            open: true,
            ..Default::default()
        };
        let nonce = state.begin();
        state.apply_nonce = Some(nonce);
        state.cancel = None;
        state.close();
        assert!(!state.take_apply(nonce + 1));
        assert!(state.take_apply(nonce));
        assert!(!state.take_apply(nonce));
    }
    #[tokio::test]
    async fn selector_cancel_does_not_open_a_selected_path_and_errors_are_recoverable() {
        let project = project();
        let root = PathBuf::from(
            std::env::var_os("IMAGINE_WORKER_A_FIXTURES").expect("explicit encrypted fixture root"),
        )
        .canonicalize()
        .unwrap();
        assert!(task_fixtures::is_owned(&root));
        let absent = root.join("not-created-or-read.wav");
        // The production client registers a path with the server; it no longer
        // opens the file locally. Exercise that boundary with an owned fixture.
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls = requests.clone();
        let expected = absent.clone();
        let router = axum::Router::new().route(
            "/v1/video/assets/link",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(body["path"], json!(expected));
                async {
                    (
                        axum::http::StatusCode::NOT_FOUND,
                        axum::Json(json!({"error":{"code":"media_unavailable"}})),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        let client = client::Client::connect(&crate::settings::Connection {
            url: format!("http://{address}"),
        })
        .unwrap();
        let cancelled = replacement_preview(
            &client,
            "own-session",
            &project.id,
            json!({}),
            None,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .unwrap();
        assert_eq!(cancelled["cancelled"], true);
        let cancelled = replacement_preview(
            &client,
            "own-session",
            &project.id,
            json!({}),
            Some(absent.clone()),
            Arc::new(AtomicBool::new(true)),
        )
        .await
        .unwrap();
        assert_eq!(cancelled["cancelled"], true);
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        let failure = replacement_preview(
            &client,
            "own-session",
            &project.id,
            json!({}),
            Some(absent),
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .unwrap_err();
        assert_eq!(failure.code, "media_unavailable");
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        stop.send(()).unwrap();
        server.await.unwrap();
    }
}
