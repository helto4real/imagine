//! Native Timeline session. Server documents are authoritative; drafts/history stay in memory.
use super::*;
use imagine_video_contract::{
    contract::*,
    editing::{Edit, EditRequest},
};
use serde_json::json;
use std::collections::BTreeSet;
mod assembly;
mod audio;
mod authoring;
mod cast;
mod compiled;
mod continuity;
mod editing;
mod inspector;
mod jobs;
mod library;
mod media_binding;
mod player;
#[cfg(feature = "ui-preview")]
pub(in crate::app) mod qa;
mod review;
mod runs;
#[path = "timeline.rs"]
mod tracks;
mod workspace;
#[derive(Clone)]
struct Snapshot {
    project: VideoProject,
    raw: Value,
}
impl Snapshot {
    fn parse(raw: Value) -> Result<Self, String> {
        let project: VideoProject = serde_json::from_value(raw["project"].clone())
            .map_err(|_| "Invalid project response.")?;
        project
            .validate()
            .map_err(|_| "Invalid project response.")?;
        Ok(Self { project, raw })
    }
}
#[derive(Default)]
pub(super) struct Timeline {
    assembly: assembly::State,
    authoring: authoring::State,
    cast: cast::State,
    compiled: compiled::State,
    library: library::State,
    session: Option<String>,
    session_checked: bool,
    document: Option<Snapshot>,
    epoch: u64,
    busy: bool,
    admitting: bool,
    selected: BTreeSet<SceneId>,
    section: Option<SectionId>,
    boundary: bool,
    audio_selected: BTreeSet<AudioClipId>,
    audio_primary: Option<AudioClipId>,
    audio_draft: Option<AudioClip>,
    project_audio: bool,
    audio_settings_draft: Option<AudioSettings>,
    history: editing::History,
    snap: editing::Snap,
    mode: editing::Mode,
    zoom: f32,
    error: Option<String>,
    folder: Option<PathBuf>,
    draft_scene: Option<SceneId>,
    draft_description: String,
    draft_scene_prompt: imagine_video_contract::creative::ScenePrompt,
    draft_section: String,
    draft_global: String,
    jobs: Vec<jobs::Job>,
    runs: Vec<imagine_video_contract::runs::Run>,
    runs_loading: bool,
    run_acting: bool,
    run_poll_at: Option<std::time::Instant>,
    pending_run: Option<(Value, String)>,
    pending: Option<(Value, String)>,
    after_edit: Option<PostAction>,
    pub(super) quick_clip: bool,
    player: Option<player::Player>,
    comparison: Option<player::Comparison>,
    comparison_selection: [Option<String>; 2],
    loading: Option<(u64, std::sync::Arc<std::sync::atomic::AtomicBool>)>,
    load_nonce: u64,
    preview_asset: Option<String>,
    selected_take: Option<String>,
    refresh_after_busy: bool,
    drag: Option<tracks::Drag>,
}
pub(super) enum Event {
    Compiled {
        session: String,
        epoch: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    LibraryFolder {
        epoch: u64,
        path: Option<PathBuf>,
    },
    Library {
        session: String,
        epoch: u64,
        kind: library::Reply,
        result: Result<Value, client::TimelineFailure>,
    },
    ClipSource {
        epoch: u64,
        scene: SceneId,
        asset: AssetId,
        result: Result<Value, client::TimelineFailure>,
    },
    Assembly {
        session: String,
        epoch: u64,
        progress: bool,
        result: Result<Value, client::TimelineFailure>,
    },
    Comparison {
        epoch: u64,
        nonce: u64,
        result: Result<Box<player::Comparison>, String>,
    },
    Authoring {
        session: String,
        kind: authoring::Reply,
        result: Result<Value, client::TimelineFailure>,
    },
    Runs {
        session: String,
        kind: runs::Reply,
        result: Result<Value, client::TimelineFailure>,
    },
    Preview {
        epoch: u64,
        nonce: u64,
        asset: String,
        result: Result<player::Preview, String>,
    },
    Session(Result<String, client::TimelineFailure>),
    Document {
        epoch: u64,
        replace: bool,
        before: Option<Box<VideoProject>>,
        history: i8,
        result: Result<Value, client::TimelineFailure>,
    },
    Folder {
        epoch: u64,
        mode: FolderMode,
        path: Option<PathBuf>,
    },
    Files {
        epoch: u64,
        paths: Vec<PathBuf>,
    },
    Submitted {
        epoch: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    Job {
        session: String,
        job: Value,
    },
    MonitorLost {
        session: String,
        id: String,
        message: String,
    },
    Cancelled {
        session: String,
        id: String,
        result: Result<(), Failure>,
    },
}
#[derive(Clone, Copy)]
pub(super) enum FolderMode {
    Open,
    Save,
    SaveAs,
}
#[derive(Clone)]
pub(super) enum PostAction {
    Compiled,
    Library,
    UseClip {
        scene: SceneId,
        asset: AssetId,
    },
    Assembly,
    Authoring(imagine_video_contract::authoring::Operation),
    Run(imagine_video_contract::runs::Mode),
    Generate,
    Save,
    New,
    Open,
    SaveAs,
    Review {
        scene: SceneId,
        take: TakeId,
        action: imagine_video_contract::review::ReviewAction,
    },
}
impl ImagineApp {
    fn timeline_session(&mut self, ctx: &egui::Context) {
        let Some(client) = self.client.clone() else {
            return;
        };
        if self.timeline.busy || self.timeline.session.is_some() || self.timeline.session_checked {
            return;
        }
        self.timeline.busy = true;
        self.timeline.session_checked = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_session());
            send(&tx, &ctx, Message::Timeline(Event::Session(result)));
        });
    }
    pub(super) fn timeline_message(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::Compiled {
                session,
                epoch,
                result,
            } => {
                if self.timeline.session.as_deref() == Some(&session)
                    && self.timeline.epoch == epoch
                {
                    self.timeline.busy = false;
                    match result {
                        Ok(value) => self.timeline.compiled.value = Some(value),
                        Err(error) => self.timeline.compiled.error = Some(error.message),
                    }
                }
            }
            Event::LibraryFolder { epoch, path } => {
                if epoch == self.timeline.epoch {
                    self.timeline.busy = false;
                    self.timeline.library.busy = false;
                    if let Some(path) = path {
                        self.timeline.library.path_from_dialog(path);
                    }
                }
            }
            Event::Library {
                session,
                epoch,
                kind,
                result,
            } => self.timeline_library_reply(ctx, session, epoch, kind, result),
            Event::ClipSource {
                epoch,
                scene,
                asset,
                result,
            } => self.timeline_clip_source_reply(ctx, epoch, scene, asset, result),
            Event::Comparison {
                epoch,
                nonce,
                result,
            } => {
                if epoch != self.timeline.epoch
                    || self
                        .timeline
                        .loading
                        .as_ref()
                        .is_none_or(|(n, _)| *n != nonce)
                {
                    return;
                }
                self.timeline.loading = None;
                match result {
                    Ok(comparison) => {
                        self.timeline.comparison = Some(*comparison);
                        self.timeline.error = None;
                    }
                    Err(error) => self.timeline.error = Some(error),
                }
            }
            Event::Authoring {
                session,
                kind,
                result,
            } => self.timeline_authoring_reply(ctx, session, kind, result),
            Event::Assembly {
                session,
                epoch,
                progress,
                result,
            } => self.timeline_assembly_reply(session, epoch, progress, result),
            Event::Runs {
                session,
                kind,
                result,
            } => self.timeline_run_reply(ctx, session, kind, result),
            Event::Preview {
                epoch,
                nonce,
                asset,
                result,
            } => {
                if epoch != self.timeline.epoch
                    || self
                        .timeline
                        .loading
                        .as_ref()
                        .is_none_or(|(n, _)| *n != nonce)
                {
                    return;
                }
                self.timeline.loading = None;
                match result {
                    Ok(preview) => {
                        self.timeline.player = Some(player::Player::new(preview));
                        self.timeline.preview_asset = Some(asset);
                        self.timeline.error = None;
                    }
                    Err(error) => self.timeline.error = Some(error),
                }
            }
            Event::Session(result) => {
                self.timeline.busy = false;
                match result {
                    Ok(session) => {
                        self.timeline.authoring.session(&session);
                        self.timeline.library.session(&session);
                        self.timeline.session = Some(session);
                        self.timeline.error = None;
                    }
                    Err(e) => self.timeline.error = Some(e.message),
                }
            }
            Event::Document {
                epoch,
                replace,
                before,
                history,
                result,
            } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                self.timeline.busy = false;
                match result {
                    Ok(raw) => match Snapshot::parse(raw) {
                        Ok(snapshot) => {
                            let sync_draft =
                                replace || before.is_some() || self.timeline.draft_scene.is_none();
                            if replace {
                                self.timeline.clear_audio_selection();
                                self.timeline.clear_preview();
                                self.timeline.epoch += 1;
                                self.timeline.history = Default::default();
                                self.timeline.selected.clear();
                                self.timeline.section = None;
                                self.timeline.boundary = false;
                            }
                            if let Some(before) = before {
                                match history {
                                    -1 => {
                                        if let Some(entry) = self.timeline.history.undo.pop() {
                                            self.timeline.history.redo.push(entry);
                                        }
                                    }
                                    1 => {
                                        if let Some(entry) = self.timeline.history.redo.pop() {
                                            self.timeline.history.undo.push(entry);
                                        }
                                    }
                                    _ => self
                                        .timeline
                                        .history
                                        .record(*before, snapshot.project.clone()),
                                }
                            }
                            self.timeline
                                .selected
                                .retain(|id| snapshot.project.scenes.iter().any(|s| &s.id == id));
                            if self.timeline.selected.is_empty() {
                                self.timeline.boundary = false;
                            }
                            if self.timeline.selected.is_empty()
                                && let Some(scene) = snapshot.project.scenes.first()
                            {
                                self.timeline.selected.insert(scene.id.clone());
                            }
                            self.timeline.document = Some(snapshot);
                            let project = &self.timeline.document.as_ref().unwrap().project;
                            self.timeline.audio_selected.retain(|id| {
                                project
                                    .audio_tracks
                                    .iter()
                                    .flat_map(|t| &t.clips)
                                    .any(|c| &c.id == id)
                            });
                            if self
                                .timeline
                                .audio_primary
                                .as_ref()
                                .is_some_and(|id| !self.timeline.audio_selected.contains(id))
                            {
                                self.timeline.audio_primary =
                                    self.timeline.audio_selected.iter().next().cloned();
                                self.timeline.audio_draft = None;
                            }
                            self.timeline.error = None;
                            if sync_draft {
                                self.timeline.sync_draft();
                            }
                            if let Some(action) = self.timeline.after_edit.take() {
                                self.timeline_action(ctx, action);
                            }
                        }
                        Err(error) => self.timeline.error = Some(error),
                    },
                    Err(error) => {
                        self.timeline.after_edit = None;
                        if let Some(raw) = error.snapshot
                            && let Ok(snapshot) = Snapshot::parse(raw)
                        {
                            self.timeline.document = Some(snapshot);
                        }
                        if error.code == "session_changed" {
                            self.timeline.clear_preview();
                            self.timeline.session = None;
                            self.timeline.document = None;
                            self.timeline.epoch += 1;
                        }
                        self.timeline.error = Some(error.message);
                    }
                }
            }
            Event::Folder { epoch, mode, path } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                self.timeline.busy = false;
                if let Some(path) = path {
                    self.timeline_folder(ctx, path, mode);
                }
            }
            Event::Files { epoch, paths } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                self.timeline.busy = false;
                if !paths.is_empty() {
                    self.timeline_import(ctx, paths);
                }
            }
            Event::Submitted { epoch, result } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                self.timeline.admitting = false;
                match result {
                    Ok(value) => {
                        let job = self.timeline.pending.as_ref().and_then(|(request, _)| {
                            jobs::Job::new(value["id"].as_str()?.to_owned(), request.clone())
                        });
                        if let Some(job) = job {
                            self.timeline.error = None;
                            self.timeline.pending = None;
                            let id = job.id.clone();
                            if !self.timeline.jobs.iter().any(|old| old.id == id) {
                                self.timeline.jobs.push(job);
                            }
                            self.timeline_poll(ctx, id);
                        } else {
                            self.timeline.error=Some("Could not read admission. Retry submission to retrieve the same job.".into());
                        }
                    }
                    Err(error) => {
                        if !error.retryable {
                            self.timeline.pending = None;
                        }
                        self.timeline.error = Some(if error.retryable {
                            "Submission unconfirmed. Use Retrieve submission to check the existing job.".into()
                        } else {
                            error.message
                        });
                    }
                }
            }
            Event::Job { session, job } => {
                if self.timeline.session.as_deref() != Some(&session) {
                    return;
                }
                let Some(old) = self
                    .timeline
                    .jobs
                    .iter_mut()
                    .find(|old| Some(old.id.as_str()) == job["id"].as_str())
                else {
                    return;
                };
                if !old.observe(job) {
                    return;
                }
                let refresh = old.terminal()
                    && self
                        .timeline
                        .document
                        .as_ref()
                        .is_some_and(|d| json!(d.project.id) == old.request["project_id"]);
                if refresh {
                    if self.timeline.busy {
                        self.timeline.refresh_after_busy = true;
                    } else {
                        self.timeline_refresh(ctx);
                    }
                }
            }
            Event::MonitorLost {
                session,
                id,
                message,
            } => {
                if self.timeline.session.as_deref() != Some(&session) {
                    return;
                }
                if let Some(job) = self.timeline.jobs.iter_mut().find(|j| j.id == id) {
                    job.monitoring = false;
                    job.error = Some(message);
                }
            }
            Event::Cancelled {
                session,
                id,
                result,
            } => {
                if self.timeline.session.as_deref() != Some(&session) {
                    return;
                }
                if let Some(job) = self.timeline.jobs.iter_mut().find(|j| j.id == id) {
                    job.acting = false;
                    match result {
                        Ok(()) => {
                            if !job.terminal() {
                                job.observation["status"] = json!("cancelling");
                            }
                        }
                        Err(error) => job.error = Some(error.message),
                    }
                }
                self.timeline_poll(ctx, id);
            }
        }
        if self.timeline.refresh_after_busy && !self.timeline.busy {
            self.timeline.refresh_after_busy = false;
            self.timeline_refresh(ctx);
        }
    }
    fn timeline_action(&mut self, ctx: &egui::Context, action: PostAction) {
        match action {
            PostAction::Compiled => self.timeline_compiled(ctx),
            PostAction::Library => self.timeline.library.open = true,
            PostAction::UseClip { scene, asset } => self.timeline_clip_source(ctx, scene, asset),
            PostAction::Assembly => self.timeline.assembly.open = true,
            PostAction::Authoring(operation) => self.timeline_open_authoring(ctx, operation),
            PostAction::Run(mode) => self.timeline_start_run(ctx, mode),
            PostAction::Generate => self.timeline_generate(ctx),
            PostAction::Save => self.timeline_save(ctx),
            PostAction::New => self.timeline_new(ctx),
            PostAction::Open => self.timeline_choose_folder(ctx, FolderMode::Open),
            PostAction::SaveAs => self.timeline_choose_folder(ctx, FolderMode::SaveAs),
            PostAction::Review {
                scene,
                take,
                action,
            } => self.timeline_review(ctx, scene, take, action),
        }
    }
    fn timeline_request(
        &mut self,
        ctx: &egui::Context,
        post: bool,
        path: String,
        body: Option<Value>,
        replace: bool,
        edit: Option<(VideoProject, i8)>,
    ) {
        if self.timeline.busy {
            return;
        }
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.busy = true;
        self.timeline.error = None;
        let (before, history) = edit
            .map(|(p, h)| (Some(Box::new(p)), h))
            .unwrap_or((None, 0));
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.timeline_request(&session, post, &path, body.as_ref()));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Document {
                    epoch,
                    replace,
                    before,
                    history,
                    result,
                }),
            );
        });
    }
    fn timeline_refresh(&mut self, ctx: &egui::Context) {
        if let Some(document) = &self.timeline.document {
            self.timeline_request(
                ctx,
                false,
                format!("/v1/video/projects/{}", document.project.id.0),
                None,
                false,
                None,
            );
        }
    }
    fn timeline_edit(&mut self, ctx: &egui::Context, commands: Vec<Edit>, history: i8) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        if commands.is_empty() {
            return;
        }
        let project = document.project.clone();
        let request = EditRequest {
            schema_version: 1,
            expected_project_revision: project.project_revision,
            commands,
        };
        self.timeline_request(
            ctx,
            true,
            format!("/v1/video/projects/{}/edits", project.id.0),
            Some(serde_json::to_value(request).expect("edit")),
            false,
            Some((project, history)),
        );
    }
    fn timeline_new(&mut self, ctx: &egui::Context) {
        let shape = crate::h3_protocol::Selection {
            ratio: "16:9".into(),
            seconds: 5.,
            ..Default::default()
        }
        .resolve()
        .expect("default video geometry");
        let model = ModelSelection {
            adapter: "h3".into(),
            model: "minimax-h3".into(),
            variant: "reference".into(),
            profile: "h3-av-noref-v2".into(),
            profile_revision: "2".into(),
        };
        self.timeline.folder = None;
        self.timeline_request(
            ctx,
            true,
            "/v1/video/projects".into(),
            Some(json!(CreateProject {
                schema_version: 1,
                default_model: model,
                delivery: DeliverySettings {
                    width: shape.width,
                    height: shape.height,
                    fps: Time { num: 24, den: 1 }
                }
            })),
            true,
            None,
        );
    }
    fn timeline_add_scene(&mut self, ctx: &egui::Context) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        let project = &document.project;
        let start = project
            .scenes
            .iter()
            .map(|s| editing::seconds(s.start) + editing::seconds(s.duration))
            .fold(0., f64::max);
        let id = SceneId(format!("scene_{}", uuid::Uuid::new_v4().simple()));
        let duration = Time { num: 5, den: 1 };
        let settings = SettingsSlot {
            model: project.default_model.clone(),
            schema_version: 1,
            values: json!({"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}),
        };
        let scene = Scene {
            id: id.clone(),
            start: editing::time(start),
            duration,
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: vec![settings],
            sections: vec![Section {
                id: SectionId(format!("section_{}", uuid::Uuid::new_v4().simple())),
                start: Time { num: 0, den: 1 },
                duration,
                prompt: String::new(),
            }],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        };
        self.timeline.clear_preview();
        self.timeline.selected_take = None;
        self.timeline.selected = BTreeSet::from([id]);
        self.timeline_edit(
            ctx,
            vec![Edit::AddScene {
                scene: Box::new(scene),
            }],
            0,
        );
    }
    fn timeline_choose_folder(&mut self, ctx: &egui::Context, mode: FolderMode) {
        if self.timeline.busy {
            return;
        }
        self.timeline.busy = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        let epoch = self.timeline.epoch;
        std::thread::spawn(move || {
            let path = rfd::FileDialog::new()
                .set_title(match mode {
                    FolderMode::SaveAs => "Choose an empty folder for the project copy",
                    FolderMode::Save => "Choose an empty project folder",
                    FolderMode::Open => "Open project folder",
                })
                .pick_folder();
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Folder { epoch, mode, path }),
            );
        });
    }
    fn timeline_folder(&mut self, ctx: &egui::Context, path: PathBuf, mode: FolderMode) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        if self.timeline.busy {
            return;
        }
        self.timeline.busy = true;
        let document = self.timeline.document.as_ref().map(|d| d.project.clone());
        let epoch = self.timeline.epoch;
        self.timeline.folder = Some(path.clone());
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        let replace = matches!(mode, FolderMode::Open | FolderMode::SaveAs);
        std::thread::spawn(move || {
            let result=runtime().block_on(async {
            let location=client.timeline_request(&session,true,"/v1/video/locations",Some(&json!({"schema_version":1,"path":path}))).await?;
            let (route,body)=if let Some(document)=document.filter(|_|!matches!(mode,FolderMode::Open)) {(format!("/v1/video/projects/{}/{}",document.id.0,if matches!(mode,FolderMode::SaveAs){"save-as"}else{"save"}),json!({"schema_version":1,"expected_project_revision":document.project_revision,"location_id":location["location_id"]}))}else{("/v1/video/projects/open".into(),json!({"schema_version":1,"location_id":location["location_id"]}))};
            client.timeline_request(&session,true,&route,Some(&body)).await
        });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Document {
                    epoch,
                    replace,
                    before: None,
                    history: 0,
                    result,
                }),
            );
        });
    }
    fn timeline_save(&mut self, ctx: &egui::Context) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        if document.raw["persistence"]["state"] == "session_only" {
            self.timeline_choose_folder(ctx, FolderMode::Save);
            return;
        }
        self.timeline_request(ctx,true,format!("/v1/video/projects/{}/save",document.project.id.0),Some(json!({"schema_version":1,"expected_project_revision":document.project.project_revision})),false,None);
    }
    fn timeline_choose_media(&mut self, ctx: &egui::Context) {
        if self.timeline.busy {
            return;
        }
        self.timeline.busy = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        let epoch = self.timeline.epoch;
        std::thread::spawn(move || {
            let paths = rfd::FileDialog::new()
                .set_title("Import project media")
                .add_filter(
                    "Image, video or audio",
                    &[
                        "png", "jpg", "jpeg", "webp", "mp4", "mkv", "webm", "wav", "flac", "mp3",
                        "ogg",
                    ],
                )
                .pick_files()
                .unwrap_or_default();
            send(&tx, &ctx, Message::Timeline(Event::Files { epoch, paths }));
        });
    }
    fn timeline_import(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        let (Some(client), Some(session), Some(document)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.document.clone(),
        ) else {
            return;
        };
        if self.timeline.busy {
            return;
        }
        self.timeline.busy = true;
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result=runtime().block_on(async {
            let mut value=document.raw;
            for path in paths {let media=client.upload_media(&session,&path).await?;value=client.timeline_request(&session,true,&format!("/v1/video/projects/{}/assets",document.project.id.0),Some(&json!({"schema_version":1,"expected_project_revision":value["project"]["project_revision"],"asset_id":media["id"]}))).await?;}
            Ok(value)
        });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Document {
                    epoch,
                    replace: false,
                    before: None,
                    history: 0,
                    result,
                }),
            );
        });
    }
    fn timeline_generate(&mut self, ctx: &egui::Context) {
        if self.timeline.admitting
            || !self.timeline.audio_selected.is_empty()
            || self.timeline.selected.len() != 1
            || self.timeline.scene_active()
        {
            return;
        }
        let Some(document) = self.timeline.document.as_ref() else {
            return;
        };
        let Some(scene) = self.timeline.selected.iter().next() else {
            return;
        };
        if self.timeline.pending.as_ref().is_some_and(|(request, _)| {
            request["project_id"] != json!(document.project.id)
                || request["scene_id"] != json!(scene)
        }) {
            self.timeline.error = Some(
                "Retrieve the unconfirmed submission in the queue before generating another scene."
                    .into(),
            );
            return;
        }
        self.timeline.pending.get_or_insert_with(||(json!({"kind":"video_scene","schema_version":1,"project_id":document.project.id,"expected_project_revision":document.project.project_revision,"scene_id":scene,"seed":42}),uuid::Uuid::new_v4().to_string()));
        self.timeline_submit_pending(ctx);
    }
    fn timeline_submit_pending(&mut self, ctx: &egui::Context) {
        if self.timeline.admitting {
            return;
        }
        let (Some(client), Some(session), Some(submission)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.pending.clone(),
        ) else {
            return;
        };
        self.timeline.admitting = true;
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result =
                runtime().block_on(client.timeline_submit(&session, &submission.0, &submission.1));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Submitted { epoch, result }),
            );
        });
    }
    fn timeline_poll(&mut self, ctx: &egui::Context, id: String) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        let Some(job) = self
            .timeline
            .jobs
            .iter_mut()
            .find(|j| j.id == id && !j.terminal() && !j.monitoring)
        else {
            return;
        };
        job.monitoring = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            runtime().block_on(async {
                loop {
                    match client
                        .timeline_request(&session, false, &format!("/v1/jobs/{id}"), None)
                        .await
                    {
                        Ok(job) => {
                            let done = matches!(
                                job["status"].as_str(),
                                Some("succeeded" | "failed" | "cancelled" | "interrupted")
                            );
                            let valid = job["id"].as_str() == Some(&id)
                                && matches!(
                                    job["status"].as_str(),
                                    Some(
                                        "queued"
                                            | "running"
                                            | "cancelling"
                                            | "succeeded"
                                            | "failed"
                                            | "cancelled"
                                            | "interrupted"
                                    )
                                );
                            send(
                                &tx,
                                &ctx,
                                Message::Timeline(Event::Job {
                                    session: session.clone(),
                                    job,
                                }),
                            );
                            if done || !valid {
                                break;
                            }
                        }
                        Err(error) => {
                            send(
                                &tx,
                                &ctx,
                                Message::Timeline(Event::MonitorLost {
                                    session: session.clone(),
                                    id: id.clone(),
                                    message: error.message,
                                }),
                            );
                            break;
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }
            });
        });
    }
}
impl Timeline {
    fn scene_active(&self) -> bool {
        let (Some(document), Some(scene)) = (&self.document, self.selected.iter().next()) else {
            return false;
        };
        self.jobs
            .iter()
            .any(|j| j.matches(&document.project.id, scene) && !j.terminal())
    }

    fn clear_preview(&mut self) {
        if let Some((_, cancel)) = self.loading.take() {
            cancel.store(true, std::sync::atomic::Ordering::Release);
        }
        if let Some(player) = &mut self.player {
            player.pause();
        }
        self.player = None;
        if let Some(comparison) = &mut self.comparison {
            comparison.pause();
        }
        self.comparison = None;
        self.preview_asset = None;
    }
    fn scene(&self) -> Option<&Scene> {
        self.document
            .as_ref()?
            .project
            .scene(self.selected.iter().next()?)
            .ok()
    }
    fn sync_draft(&mut self) {
        self.audio_draft = self
            .document
            .as_ref()
            .and_then(|d| {
                self.audio_primary.as_ref().and_then(|id| {
                    d.project
                        .audio_tracks
                        .iter()
                        .flat_map(|t| &t.clips)
                        .find(|c| &c.id == id)
                })
            })
            .cloned();
        self.audio_settings_draft = self.document.as_ref().map(|d| d.project.audio.clone());
        self.draft_global = self
            .document
            .as_ref()
            .map(|d| d.project.global_prompt.clone())
            .unwrap_or_default();
        if let Some(scene) = self.scene().cloned() {
            self.draft_scene_prompt = self
                .document
                .as_ref()
                .and_then(|d| d.project.creative.scene_prompts.get(&scene.id))
                .cloned()
                .unwrap_or_default();
            self.draft_scene = Some(scene.id.clone());
            self.draft_description = scene.description;
            self.draft_section = self
                .section
                .as_ref()
                .and_then(|id| scene.sections.iter().find(|s| &s.id == id))
                .map(|s| s.prompt.clone())
                .unwrap_or_default();
        }
    }
}
impl Drop for Timeline {
    fn drop(&mut self) {
        self.clear_preview();
    }
}
impl ImagineApp {
    fn timeline_load_media(&mut self, ctx: &egui::Context, asset: String) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.clear_preview();
        self.timeline.load_nonce += 1;
        let nonce = self.timeline.load_nonce;
        let epoch = self.timeline.epoch;
        let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.timeline.loading = Some((nonce, cancelled.clone()));
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(async {
                let info = client
                    .timeline_request(&session, false, &format!("/v1/video/assets/{asset}"), None)
                    .await
                    .map_err(|e| e.message)?;
                let bytes = client
                    .timeline_media(&session, &asset)
                    .await
                    .map_err(|e| e.message)?;
                player::decode(bytes, info, cancelled)
            });
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Preview {
                    epoch,
                    nonce,
                    asset,
                    result,
                }),
            );
        });
    }
    pub(in crate::app) fn pause_timeline(&mut self) {
        if let Some(comparison) = &mut self.timeline.comparison {
            comparison.pause();
        }
        if let Some(player) = &mut self.timeline.player {
            player.pause();
        }
    }
}

impl Timeline {
    pub(super) fn draft_edits(&self) -> Vec<Edit> {
        let Some(document) = &self.document else {
            return vec![];
        };
        let mut edits = vec![];
        if let Some(draft) = &self.audio_draft
            && document
                .project
                .audio_tracks
                .iter()
                .flat_map(|t| &t.clips)
                .any(|c| c.id == draft.id && c != draft)
        {
            edits.push(Edit::UpdateAudioClip {
                clip: Box::new(draft.clone()),
            });
        }
        if let Some(settings) = &self.audio_settings_draft
            && settings != &document.project.audio
        {
            edits.push(Edit::SetAudioSettings {
                settings: settings.clone(),
            });
        }
        if self.draft_global != document.project.global_prompt {
            edits.push(Edit::SetGlobalPrompt {
                prompt: self.draft_global.clone(),
            });
        }
        if let Some(scene) = self.scene() {
            if self.draft_scene_prompt
                != document
                    .project
                    .creative
                    .scene_prompts
                    .get(&scene.id)
                    .cloned()
                    .unwrap_or_default()
            {
                let mut creative = document.project.creative.clone();
                creative
                    .scene_prompts
                    .insert(scene.id.clone(), self.draft_scene_prompt.clone());
                edits.push(Edit::SetCreative { creative });
            }
            if self.draft_description != scene.description {
                edits.push(Edit::SetSceneDescription {
                    scene_id: scene.id.clone(),
                    description: self.draft_description.clone(),
                });
            }
            if let Some(section) = self
                .section
                .as_ref()
                .and_then(|id| scene.sections.iter().find(|s| &s.id == id))
                && self.draft_section != section.prompt
            {
                let mut section = section.clone();
                section.prompt = self.draft_section.clone();
                edits.push(Edit::UpdateSection {
                    scene_id: scene.id.clone(),
                    section,
                });
            }
        }
        edits
    }
}
