//! Native Timeline session. Server documents are authoritative; drafts/history stay in memory.
use super::*;
use crate::video_defaults::new_project;
use imagine_video_contract::snapshot;
use imagine_video_contract::{
    contract::*,
    editing::{Edit, EditRequest},
};
use serde_json::json;
use std::collections::BTreeSet;
mod animate;
mod assembly;
mod audio;
mod authoring;
mod captures;
mod cast;
mod catalog;
mod compatibility;
mod compiled;
mod continuity;
mod editing;
mod generation_ui;
mod group_commands;
mod image_anchors;
mod inspector;
mod jobs;
mod library;
mod library_character;
mod loras;
mod media_binding;
mod media_health;
mod media_picker;
mod navigation;
mod playback;
mod player;
mod preflight;
mod presets;
mod private_projects;
mod project_prompt;
mod project_timing;
mod prompt_tools;
#[cfg(feature = "ui-preview")]
pub(in crate::app) mod qa;
mod recovery;
mod reference_cache;
mod review;
mod runs;
mod scene_commands;
mod section_block;
mod section_commands;
mod seed;
mod selection;
#[cfg(test)]
mod selection_oracle;
mod story;
mod take_card;
mod take_selection;
mod thumbnails;
#[path = "timeline.rs"]
mod tracks;
mod waveform;
mod workspace;
#[derive(Clone)]
struct Snapshot {
    project: VideoProject,
    /// Typed runtime and review state from the same response.
    view: snapshot::SnapshotView,
    /// The response itself, for rare one-shot extras such as export reports.
    raw: Value,
}
impl Snapshot {
    fn parse(raw: Value) -> Result<Self, String> {
        let (project, view) = snapshot::parse(&raw).ok_or("Invalid project response.")?;
        Ok(Self { project, view, raw })
    }
    /// A synthetic snapshot; panics on fixture data the server would never send.
    #[cfg(test)]
    fn with(project: VideoProject, raw: Value) -> Self {
        let view = serde_json::from_value(raw.clone()).expect("synthetic snapshot view");
        Self { project, view, raw }
    }
    /// The session media id serving project asset `id`.
    fn runtime_asset(&self, id: &str) -> Option<&str> {
        let (_, runtime) = self.view.runtime_assets.iter().find(|(a, _)| a.0 == id)?;
        Some(runtime)
    }
    /// The session media id serving Take `id`.
    fn runtime_take(&self, id: &str) -> Option<&str> {
        let (_, runtime) = self.view.runtime_takes.iter().find(|(t, _)| t.0 == id)?;
        Some(runtime)
    }
    fn save_state(&self) -> Option<&snapshot::SaveState> {
        self.view.persistence.as_ref().map(|p| &p.state)
    }
    fn take_state(&self, id: &TakeId) -> imagine_video_contract::review::TakeState {
        self.view.take_states.get(id).cloned().unwrap_or_default()
    }
    fn take(&self, id: &TakeId) -> Option<&snapshot::TakeView> {
        self.view.takes.iter().find(|t| &t.id == id)
    }
    fn capture(&self, id: &TakeId) -> Option<&imagine_video_contract::captures::Capture> {
        self.view.captures.get(id)
    }
}
#[derive(Default)]
pub(super) struct Timeline {
    /// A project-replacing action held until the user discards unsaved work.
    replace_pending: Option<Replace>,
    /// The user chose to close Imagine and discard unsaved work.
    close_confirmed: bool,
    captures: captures::State,
    reference_cache: reference_cache::State,
    preflight: preflight::State,
    playback: playback::State,
    /// Shelf media dropped on a scene or Section this frame.
    drop: Option<tracks::Dropped>,
    media_health: media_health::State,
    compatibility_open: bool,
    compatibility_report: Option<Value>,
    private_projects: private_projects::State,
    presets: presets::State,
    generation_editor: generation_ui::State,
    assembly: assembly::State,
    authoring: authoring::State,
    cast: cast::State,
    catalog: catalog::State,
    media_picker: media_picker::State,
    thumbnails: thumbnails::State,
    image_anchors: image_anchors::State,
    compiled: compiled::State,
    library: library::State,
    loras: loras::State,
    session: Option<String>,
    private_storage: bool,
    session_checked: bool,
    needs_reopen: bool,
    document: Option<Snapshot>,
    epoch: u64,
    busy: bool,
    admitting: bool,
    selected: BTreeSet<SceneId>,
    selection_order: Vec<selection::Item>,
    playhead: f64,
    section: Option<SectionId>,
    section_selection: Vec<SectionId>,
    audio_selection_order: Vec<AudioClipId>,
    boundary: bool,
    /// Scenes whose carried motion needs H3 Reference, awaiting the user's answer.
    model_switch: Vec<SceneId>,
    audio_selected: BTreeSet<AudioClipId>,
    audio_primary: Option<AudioClipId>,
    audio_draft: Option<AudioClip>,
    project_audio: bool,
    /// Project settings opened in the docked inspector from its project strip.
    project_open: bool,
    audio_settings_draft: Option<AudioSettings>,
    history: editing::History,
    snap: editing::Snap,
    mode: editing::Mode,
    zoom: f32,
    navigation: navigation::State,
    waveform: waveform::State,
    error: Option<String>,
    folder: Option<PathBuf>,
    draft_scene: Option<SceneId>,
    draft_description: String,
    draft_scene_prompt: imagine_video_contract::creative::ScenePrompt,
    draft_section: String,
    draft_global: String,
    global_prompt_options: Option<imagine_video_contract::creative::GlobalPromptOptions>,
    project_prompt_open: bool,
    project_timing: project_timing::State,
    jobs: Vec<jobs::Job>,
    runs: Vec<imagine_video_contract::runs::Run>,
    runs_loading: bool,
    run_acting: bool,
    run_poll_at: Option<std::time::Instant>,
    pending_run: Option<(Value, String)>,
    pending: Option<(Value, String)>,
    pending_seed_advance: Option<seed::Advance>,
    after_edit: Option<PostAction>,
    player: Option<player::Player>,
    comparison: Option<player::Comparison>,
    comparison_state: Option<player::ComparisonState>,
    comparison_selection: [Option<String>; 2],
    comparison_scene: Option<SceneId>,
    comparison_explicit: bool,
    loading: Option<(u64, std::sync::Arc<std::sync::atomic::AtomicBool>)>,
    load_nonce: u64,
    preview_asset: Option<String>,
    selected_take: Option<String>,
    refresh_after_busy: bool,
    shelf_collapsed: bool,
    /// The media bin shows the cast instead of project media.
    bin_cast: bool,
    inspector_collapsed: bool,
    keyboard_focus: Option<egui::Id>,
    drag: Option<tracks::Drag>,
    /// The project as the drag in progress would leave it, drawn until release.
    drag_preview: Option<VideoProject>,
}
pub(super) enum Event {
    CleanupPreview {
        epoch: u64,
        revision: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    Capabilities {
        epoch: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    PrivateProjects {
        epoch: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    Workspaces {
        epoch: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    WorkspaceKey {
        epoch: u64,
        replaced: bool,
        result: Result<Value, client::TimelineFailure>,
    },
    Presets {
        epoch: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    PresetPlan {
        epoch: u64,
        project: bool,
        result: Result<Value, client::TimelineFailure>,
    },
    Preflight {
        epoch: u64,
        key: (ProjectId, u64, SceneId),
        result: Result<Value, client::TimelineFailure>,
    },
    Animated {
        epoch: u64,
        result: Result<(client::TimelineSession, Value), client::TimelineFailure>,
    },
    ReferenceCache {
        epoch: u64,
        nonce: u64,
        result: Result<Value, client::TimelineFailure>,
    },
    MediaHealth {
        epoch: u64,
        nonce: u64,
        kind: media_health::Reply,
        result: Result<Value, client::TimelineFailure>,
    },
    Waveform {
        epoch: u64,
        project: ProjectId,
        asset: AssetId,
        revision: u64,
        peaks: usize,
        result: Result<Value, String>,
    },
    CatalogIo {
        nonce: u64,
        result: Result<catalog::Loaded, String>,
    },
    PickerLibrary {
        epoch: u64,
        nonce: u64,
        id: String,
        result: Result<Value, client::TimelineFailure>,
    },
    Thumbnail {
        epoch: u64,
        project: ProjectId,
        asset: AssetId,
        revision: u64,
        result: Result<client::Preview, String>,
    },
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
        elapsed_ms: u64,
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
    Session(Result<client::TimelineSession, client::TimelineFailure>),
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
        target: media_picker::Target,
    },
    Imported {
        epoch: u64,
        target: media_picker::Target,
        assets: Vec<AssetId>,
        result: Result<Value, client::TimelineFailure>,
    },
    LibraryImported {
        epoch: u64,
        editor: String,
        voice: bool,
        result: Result<Vec<AssetId>, client::TimelineFailure>,
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
    ExportCompatible,
    Open,
    Save,
    SaveAs,
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum SceneState {
    Empty,
    Candidates,
    Accepted,
    Outdated,
    Generating,
}
impl SceneState {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Empty => "No Take yet",
            Self::Candidates => "Takes to review",
            Self::Accepted => "Accepted Take",
            Self::Outdated => "Accepted Take is outdated",
            Self::Generating => "Generating",
        }
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Replace {
    New,
    Open,
    Animate,
    /// Closing the main window.
    Close,
}
#[derive(Clone)]
pub(super) enum PostAction {
    ImportTake {
        scene: SceneId,
        asset: AssetId,
    },
    Capture {
        scene: SceneId,
        take: TakeId,
        action: imagine_video_contract::captures::Action,
    },
    ClearSelection,
    SelectGroup(Vec<SectionId>, Vec<AudioClipId>),
    SelectSection(SceneId, SectionId),
    ExportCompatible,
    GenerationSaved,
    /// Plans and applies a preset once pending drafts are committed.
    Preset {
        key: String,
        target: presets::Target,
    },
    /// Reloads an open Project generation window with the preset's values.
    ProjectPresetApplied,
    EnhanceField(imagine_video_contract::authoring::SceneField),
    Compiled,
    ApproveCompiled,
    Library,
    LibrarySave {
        character: bool,
        replace: bool,
    },
    UseClip {
        scene: SceneId,
        asset: AssetId,
    },
    Assembly,
    RenderAssembly,
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
    pub(in crate::app) fn timeline_session(&mut self, ctx: &egui::Context) {
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
        if let Event::Animated { epoch, result } = event {
            if epoch != self.timeline.epoch {
                return;
            }
            self.timeline.busy = false;
            match result {
                Ok((connection, raw)) => {
                    self.timeline.folder = None;
                    self.timeline_message(ctx, Event::Session(Ok(connection)));
                    self.timeline_message(
                        ctx,
                        Event::Document {
                            epoch,
                            replace: true,
                            before: None,
                            history: 0,
                            result: Ok(raw),
                        },
                    );
                }
                Err(error) => self.timeline.error = Some(error.message),
            }
            return;
        }
        match event {
            Event::Animated { .. } => unreachable!(),
            Event::CleanupPreview {
                epoch,
                revision,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline_cleanup_reply(revision, result);
                }
            }
            Event::Capabilities { epoch, result } => {
                if epoch == self.timeline.epoch {
                    self.timeline_capabilities_reply(result);
                }
            }
            Event::PrivateProjects { epoch, result } => {
                if epoch == self.timeline.epoch {
                    self.timeline.private_projects.reply(result);
                }
            }
            Event::Workspaces { epoch, result } => {
                if epoch == self.timeline.epoch {
                    self.timeline.private_projects.workspaces_reply(result);
                }
            }
            Event::WorkspaceKey {
                epoch,
                replaced,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline
                        .private_projects
                        .key_reply(ctx, replaced, result);
                }
            }
            Event::Presets { epoch, result } => {
                if epoch == self.timeline.epoch {
                    self.timeline.presets.reply(result);
                }
            }
            Event::PresetPlan {
                epoch,
                project,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline_preset_planned(ctx, project, result);
                }
            }
            Event::Preflight { epoch, key, result } => {
                if epoch == self.timeline.epoch {
                    self.timeline_preflight_reply(key, result);
                }
            }
            Event::ReferenceCache {
                epoch,
                nonce,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline
                        .reference_cache
                        .receive(nonce, result.map_err(|e| e.message));
                }
            }
            Event::MediaHealth {
                epoch,
                nonce,
                kind,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline_media_health_reply(ctx, nonce, kind, result);
                }
            }
            Event::CatalogIo { nonce, result } => self.timeline_catalog_reply(nonce, result),
            Event::PickerLibrary {
                epoch,
                nonce,
                id,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline_picker_library_reply(nonce, id, result);
                }
            }
            Event::Thumbnail {
                epoch,
                project,
                asset,
                revision,
                result,
            } => {
                if epoch == self.timeline.epoch {
                    self.timeline_thumbnail_reply(ctx, project, asset, revision, result);
                }
            }
            Event::Waveform {
                epoch,
                project,
                asset,
                revision,
                peaks,
                result,
            } => {
                if epoch == self.timeline.epoch
                    && self.timeline.waveform.project.as_ref() == Some(&project)
                {
                    self.timeline
                        .waveform
                        .loading
                        .remove(&(asset.clone(), revision, peaks));
                    self.timeline.waveform.values.insert(
                        (asset, revision, peaks),
                        result.and_then(|v| {
                            serde_json::from_value(v)
                                .map_err(|_| "Invalid waveform response.".into())
                        }),
                    );
                }
            }
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
                        Ok(value) => self.timeline.compiled.receive(value),
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
                elapsed_ms,
                result,
            } => self.timeline_assembly_reply(ctx, session, epoch, progress, elapsed_ms, result),
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
                        self.timeline_cue_loaded(nonce);
                    }
                    Err(error) => self.timeline.error = Some(error),
                }
            }
            Event::Session(result) => {
                self.timeline.busy = false;
                match result {
                    Ok(connection) => {
                        self.timeline.private_storage = connection.private_storage;
                        let session = connection.id;
                        self.timeline.authoring.session(&session);
                        self.timeline.library.session(&session);
                        self.timeline.session = Some(session);
                        self.timeline.preflight.session_changed();
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
                            let reattaching = self.timeline.needs_reopen;
                            let replace =
                                replace
                                    || (reattaching
                                        && self.timeline.document.as_ref().is_some_and(|old| {
                                            old.project.id != snapshot.project.id
                                        }));
                            self.timeline.needs_reopen = false;
                            if reattaching {
                                self.timeline.history = Default::default();
                            }
                            if let Some(report) = snapshot.raw.get("compatibility_export") {
                                self.timeline.compatibility_report = Some(report.clone());
                                self.timeline.compatibility_open = true;
                            }
                            let sync_draft = replace
                                || (!reattaching
                                    && (before.is_some() || self.timeline.draft_scene.is_none()));
                            if replace {
                                if self
                                    .timeline
                                    .document
                                    .as_ref()
                                    .is_none_or(|d| d.project.id != snapshot.project.id)
                                {
                                    self.timeline.navigation = Default::default();
                                }
                                self.timeline.assembly.project_changed();
                                self.timeline.clear_audio_selection();
                                self.timeline.clear_preview();
                                self.timeline.thumbnails = Default::default();
                                self.timeline.epoch += 1;
                                self.timeline.history = Default::default();
                                self.timeline.selected.clear();
                                self.timeline.section = None;
                                self.timeline.boundary = false;
                            }
                            // Undo restores a recorded "before" project. An unrecorded
                            // authoring change in between would be reverted with it, so
                            // such a change starts a fresh history.
                            if before.is_none()
                                && self.timeline.document.as_ref().is_some_and(|old| {
                                    old.project.id == snapshot.project.id
                                        && !editing::difference(&old.project, &snapshot.project)
                                            .is_empty()
                                })
                            {
                                self.timeline.history = Default::default();
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
                            // Select the first scene of a newly opened project only; a
                            // cleared selection stays cleared across later refreshes.
                            if replace
                                && self.timeline.selected.is_empty()
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
                            self.timeline.session_ended();
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
            Event::Files {
                epoch,
                paths,
                target,
            } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                self.timeline.busy = false;
                if !paths.is_empty() {
                    self.timeline_import_into(ctx, paths, target);
                }
            }
            Event::Imported {
                epoch,
                target,
                assets,
                result,
            } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                let success = result.is_ok();
                self.timeline.media_picker.finish_import(
                    &target,
                    result.as_ref().err().map(|error| error.message.clone()),
                );
                self.timeline_message(
                    ctx,
                    Event::Document {
                        epoch,
                        replace: false,
                        before: None,
                        history: 0,
                        result,
                    },
                );
                if success {
                    self.timeline_use_media(ctx, target, assets);
                }
            }
            Event::LibraryImported {
                epoch,
                editor,
                voice,
                result,
            } => {
                if epoch != self.timeline.epoch {
                    return;
                }
                self.timeline.busy = false;
                match result {
                    Ok(assets) => self.library_character_media(&editor, voice, assets),
                    Err(error) => self.timeline.library.error = Some(error.message),
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
                                if let Some(advance) = self.timeline.pending_seed_advance.take()
                                    && let Some(document) = &self.timeline.document
                                    && let Some(generation) = advance.apply(&document.project)
                                {
                                    self.timeline_edit(
                                        ctx,
                                        vec![Edit::SetGenerationDefaults {
                                            generation: Some(generation),
                                        }],
                                        0,
                                    );
                                }
                            }
                            self.timeline.pending_seed_advance = None;
                            self.timeline_poll(ctx, id);
                        } else {
                            self.timeline.error=Some("Could not read admission. Retry submission to retrieve the same job.".into());
                        }
                    }
                    Err(error) => {
                        if !error.retryable {
                            self.timeline.pending = None;
                            self.timeline.pending_seed_advance = None;
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
                        .is_some_and(|d| old.project_id() == Some(&d.project.id.0));
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
                                job.set_status(jobs::JobStatus::Cancelling);
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
        let current_scene = self
            .timeline
            .document
            .as_ref()
            .and_then(|d| self.timeline.primary_scene(&d.project));
        if matches!(
            &action,
            PostAction::Generate
                | PostAction::Compiled
                | PostAction::ApproveCompiled
                | PostAction::Save
                | PostAction::SaveAs
                | PostAction::ExportCompatible
                | PostAction::Run(_)
        ) && let Some(project) = self.timeline.document.as_ref().map(|d| &d.project)
            && let Some(error) = self.timeline.generation_editor.input_error(
                project,
                if matches!(
                    &action,
                    PostAction::Generate | PostAction::Compiled | PostAction::ApproveCompiled
                ) {
                    current_scene.as_ref()
                } else {
                    None
                },
            )
        {
            self.timeline.error = Some(error);
            return;
        }
        match action {
            PostAction::ClearSelection => {
                if let Some(p) = self.timeline.document.as_ref().map(|d| d.project.clone()) {
                    self.timeline.set_selection(&p, vec![]);
                }
            }
            PostAction::ExportCompatible => {
                self.timeline.compatibility_report = None;
                self.timeline.compatibility_open = true;
            }
            PostAction::GenerationSaved => self.timeline.generation_editor.project_saved(),
            PostAction::Preset { key, target } => self.timeline_preset_plan(ctx, key, target),
            PostAction::ProjectPresetApplied => {
                if self.timeline.generation_editor.is_open() {
                    self.timeline_open_generation();
                }
            }
            PostAction::Compiled => self.timeline_compiled(ctx),
            PostAction::ApproveCompiled => self.timeline_approve_compiled(ctx),
            PostAction::Library => self.timeline.library.open = true,
            PostAction::LibrarySave { character, replace } => {
                self.library_save(ctx, character, replace)
            }
            PostAction::UseClip { scene, asset } => self.timeline_clip_source(ctx, scene, asset),
            PostAction::Assembly => self.timeline.assembly.open = true,
            PostAction::RenderAssembly => self.timeline_assembly_render_current(ctx),
            PostAction::SelectSection(scene, section) => {
                self.timeline.clear_audio_selection();
                self.timeline.selected = BTreeSet::from([scene]);
                self.timeline.section_selection = vec![section.clone()];
                self.timeline.section = Some(section);
                self.timeline.sync_draft();
            }
            PostAction::SelectGroup(sections, audio) => {
                self.timeline.clear_audio_selection();
                self.timeline.section_selection = sections;
                self.timeline.section = if audio.is_empty() {
                    self.timeline.section_selection.last().cloned()
                } else {
                    None
                };
                self.timeline.audio_selection_order = audio;
                self.timeline.audio_selected = self
                    .timeline
                    .audio_selection_order
                    .iter()
                    .cloned()
                    .collect();
                self.timeline.audio_primary = self.timeline.audio_selection_order.last().cloned();
                if let Some(scene) = self.timeline.document.as_ref().and_then(|d| {
                    d.project.scenes.iter().find(|s| {
                        s.sections
                            .iter()
                            .any(|s| self.timeline.section_selection.last() == Some(&s.id))
                    })
                }) {
                    self.timeline.selected = BTreeSet::from([scene.id.clone()]);
                }
                self.timeline.sync_draft();
            }
            PostAction::EnhanceField(field) => self.timeline_open_field_enhance(ctx, field),
            PostAction::Authoring(operation) => self.timeline_open_authoring(ctx, operation),
            PostAction::Run(mode) => self.timeline_start_run(ctx, mode),
            PostAction::Generate => self.timeline_generate(ctx),
            PostAction::Save => self.timeline_save(ctx),
            PostAction::New => {
                if !self.timeline_hold_replace(Replace::New) {
                    self.timeline_new(ctx)
                }
            }
            PostAction::Open => {
                if !self.timeline_hold_replace(Replace::Open) {
                    self.timeline_open_project(ctx)
                }
            }
            PostAction::SaveAs => self.timeline_choose_folder(ctx, FolderMode::SaveAs),
            PostAction::Review {
                scene,
                take,
                action,
            } => self.timeline_review(ctx, scene, take, action),
            PostAction::Capture {
                scene,
                take,
                action,
            } => self.timeline_capture(ctx, scene, take, action),
            PostAction::ImportTake { scene, asset } => self.timeline_import_take(ctx, scene, asset),
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
        if self.timeline.busy || (self.timeline.needs_reopen && path != "/v1/video/projects/open") {
            return;
        }
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            self.timeline.error =
                Some("Not connected to the video service. Reconnect and try again.".into());
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
        if self.timeline.needs_reopen {
            return;
        }
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
    /// Whether the open project has work that New, Open or Animate would discard.
    pub(super) fn timeline_unsaved(&self) -> bool {
        let Some(d) = &self.timeline.document else {
            return false;
        };
        let has_content = !d.project.scenes.is_empty()
            || !d.project.assets.is_empty()
            || !d.project.audio_tracks.is_empty();
        !self.timeline.draft_edits().is_empty()
            || has_content && d.save_state() != Some(&snapshot::SaveState::Saved)
    }
    /// Holds a main-window close while the open project has unsaved work; the
    /// close goes through once the user discards it.
    pub(super) fn timeline_guard_close(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested())
            && !self.timeline.close_confirmed
            && self.timeline_hold_replace(Replace::Close)
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        }
    }
    /// Holds `replace` for confirmation when it would discard unsaved work.
    pub(super) fn timeline_hold_replace(&mut self, replace: Replace) -> bool {
        if !self.timeline_unsaved() {
            return false;
        }
        self.timeline.replace_pending = Some(replace);
        self.workspace = Workspace::Video;
        true
    }
    pub(super) fn timeline_replace_window(&mut self, ctx: &egui::Context) {
        let Some(replace) = self.timeline.replace_pending else {
            return;
        };
        let mut decision = None;
        egui::Window::new("Unsaved project")
            .id(egui::Id::new("video-replace-unsaved"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                let _canvas = helto::window_canvas(ui);
                let answer = helto::confirm(
                    ui,
                    match replace {
                        Replace::New => {
                            "Start a new project? Unsaved changes in this project are lost."
                        }
                        Replace::Open => {
                            "Open another project? Unsaved changes in this project are lost."
                        }
                        Replace::Animate => {
                            "Animate into a new project? Unsaved changes in this project are lost."
                        }
                        Replace::Close => {
                            "Close Imagine? Unsaved changes in this project are lost."
                        }
                    },
                    (crate::icons::Icon::Trash, "Discard changes"),
                    "Keep project",
                    None,
                );
                #[cfg(feature = "ui-preview")]
                {
                    qa::control(ui, "replace-discard".into(), &answer.yes);
                    qa::control(ui, "replace-keep".into(), &answer.no);
                }
                if answer.yes.clicked() {
                    decision = Some(true);
                } else if answer.no.clicked() {
                    decision = Some(false);
                }
            });
        if let Some(discard) = decision {
            self.timeline.replace_pending = None;
            if !discard && replace == Replace::Animate {
                self.workspace = Workspace::Image;
            }
            if discard {
                match replace {
                    Replace::New => self.timeline_new(ctx),
                    Replace::Open => self.timeline_open_project(ctx),
                    Replace::Animate => self.animate_now(ctx),
                    Replace::Close => {
                        self.timeline.close_confirmed = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
        }
    }
    fn timeline_new(&mut self, ctx: &egui::Context) {
        self.timeline.folder = None;
        self.timeline_request(
            ctx,
            true,
            "/v1/video/projects".into(),
            Some(json!(new_project())),
            true,
            None,
        );
    }
    fn timeline_add_scene(&mut self, ctx: &egui::Context) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        let project = &document.project;
        let current = self.timeline.primary_scene(project);
        let Some((start, duration)) = scene_commands::insertion(
            project,
            current.as_ref(),
            self.timeline.playhead,
            self.timeline.snap,
        ) else {
            self.timeline.error = Some("No free space for a Scene on this timeline.".into());
            return;
        };
        let id = SceneId(format!("scene_{}", uuid::Uuid::new_v4().simple()));
        let settings = SettingsSlot {
            model: project.default_model.clone(),
            schema_version: 1,
            values: json!({"stage_one":{"steps":20,"sampler":"res_multistep","scheduler":"simple"},"stage_two":null}),
        };
        let scene = Scene {
            id: id.clone(),
            start,
            duration,
            description: String::new(),
            model: ModelChoice::ProjectDefault,
            settings: if project.generation.is_some() {
                vec![]
            } else {
                vec![settings]
            },
            sections: vec![],
            bindings: vec![],
            continuity: None,
            accepted_take: None,
        };
        self.timeline.clear_preview();
        self.timeline.selected_take = None;
        self.timeline.selected = BTreeSet::from([id]);
        self.timeline.section = None;
        self.timeline.section_selection.clear();
        self.timeline.clear_audio_selection();
        self.timeline.selection_order.clear();
        self.timeline_edit(
            ctx,
            vec![Edit::AddScene {
                scene: Box::new(scene),
            }],
            0,
        );
    }
    fn timeline_choose_folder(&mut self, ctx: &egui::Context, mode: FolderMode) {
        if self.timeline.private_storage && !matches!(mode, FolderMode::Open) {
            if let Some(document) = &self.timeline.document {
                let action = match mode {
                    FolderMode::SaveAs => "save-as",
                    FolderMode::ExportCompatible => "export-compatible",
                    _ => "save",
                };
                self.timeline_request(ctx, true, format!("/v1/video/projects/{}/{action}", document.project.id.0),
                    Some(json!({"schema_version":1,"expected_project_revision":document.project.project_revision})),
                    matches!(mode, FolderMode::SaveAs), None);
            }
            return;
        }
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
                    FolderMode::ExportCompatible => {
                        "Export older-compatible copy to an empty folder"
                    }
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
        let recovering = self.timeline.needs_reopen
            && matches!(mode, FolderMode::Open)
            && self.timeline.folder.as_ref() == Some(&path);
        if !matches!(mode, FolderMode::ExportCompatible) {
            self.timeline.folder = Some(path.clone());
        }
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        let replace = matches!(mode, FolderMode::Open | FolderMode::SaveAs) && !recovering;
        std::thread::spawn(move || {
            let result=runtime().block_on(async {
            let location=client.timeline_request(&session,true,"/v1/video/locations",Some(&json!({"schema_version":1,"path":path}))).await?;
            let (route,body)=if let Some(document)=document.filter(|_|!matches!(mode,FolderMode::Open)) {(format!("/v1/video/projects/{}/{}",document.id.0,match mode{FolderMode::SaveAs=>"save-as",FolderMode::ExportCompatible=>"export-compatible",_=>"save"}),json!({"schema_version":1,"expected_project_revision":document.project_revision,"location_id":location["location_id"]}))}else{("/v1/video/projects/open".into(),json!({"schema_version":1,"location_id":location["location_id"]}))};
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
        if document.save_state() == Some(&snapshot::SaveState::SessionOnly) {
            self.timeline_choose_folder(ctx, FolderMode::Save);
            return;
        }
        self.timeline_request(ctx,true,format!("/v1/video/projects/{}/save",document.project.id.0),Some(json!({"schema_version":1,"expected_project_revision":document.project.project_revision})),false,None);
    }
    fn timeline_choose_media(&mut self, ctx: &egui::Context) {
        self.timeline_choose_context_media(ctx, media_picker::Target::Project);
    }
    fn timeline_import_into(
        &mut self,
        ctx: &egui::Context,
        paths: Vec<PathBuf>,
        target: media_picker::Target,
    ) {
        self.timeline_import_sources(
            ctx,
            paths
                .into_iter()
                .map(media_picker::ImportSource::File)
                .collect(),
            target,
        );
    }
    fn timeline_import_sources(
        &mut self,
        ctx: &egui::Context,
        sources: Vec<media_picker::ImportSource>,
        target: media_picker::Target,
    ) {
        if let media_picker::Target::LibraryCharacter { editor, voice } = target {
            self.library_character_import(ctx, sources, editor, voice);
            return;
        }
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
            let mut assets = vec![];
            let mut value = document.raw;
            let result=runtime().block_on(async {
                for source in sources {
                    let (media,name)=match source {
                        media_picker::ImportSource::File(path)=>{let name=path.file_name().and_then(|n|n.to_str()).unwrap_or("").to_owned();(client.upload_media(&session,&path).await?,name)},
                        media_picker::ImportSource::Library{library,entry}=>{
                            let origin=LibraryMediaSource{library_id:library.clone(),entry_id:entry["entry_id"].as_str().unwrap_or("").into(),resource_id:entry["resource_id"].as_str().unwrap_or("").into()};
                            if let Some(asset)=serde_json::from_value::<VideoProject>(value["project"].clone()).ok().and_then(|p|p.assets.into_values().find(|a|a.library_source.as_ref()==Some(&origin))){assets.push(asset.id);continue;}
                            let media=client.timeline_request(&session,true,&format!("/v1/video/libraries/{library}/media-copy"),Some(&json!({"schema_version":1,"entry_id":entry["entry_id"],"expected_entry_revision":entry["entry_revision"],"resource_id":entry["resource_id"]}))).await?;(media,entry["name"].as_str().unwrap_or("").to_owned())},
                    };
                    value=client.timeline_request(&session,true,&format!("/v1/video/projects/{}/assets",document.project.id.0),Some(&json!({"schema_version":1,"expected_project_revision":value["project"]["project_revision"],"asset_id":media["id"],"name":name,"library_source":media["library_source"]}))).await?;
                    if let Some(id)=value["runtime_assets"].as_object().and_then(|m|m.iter().find(|(_,v)|*v==&media["id"])).map(|(k,_)|AssetId(k.clone())) {assets.push(id);}
                }
                Ok(value.clone())
            }).map_err(|mut error:client::TimelineFailure| {error.snapshot=Some(value);error});
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Imported {
                    epoch,
                    target,
                    assets,
                    result,
                }),
            );
        });
    }
    fn timeline_generate(&mut self, ctx: &egui::Context) {
        if self.timeline.admitting
            || self.timeline.scene().is_none()
            || self.timeline.scene_active()
        {
            return;
        }
        let Some(document) = self.timeline.document.as_ref() else {
            return;
        };
        let Some(scene) = self.timeline.primary_scene(&document.project) else {
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
        if self.timeline.pending.is_none() {
            let random = uuid::Uuid::new_v4().as_u128() as u64;
            let (seed, advance) = seed::prepare(&document.project, random);
            self.timeline.pending_seed_advance = advance;
            self.timeline.pending = Some((
                json!({"kind":"video_scene","schema_version":1,"project_id":document.project.id,"expected_project_revision":document.project.project_revision,"scene_id":scene,"seed":seed}),
                uuid::Uuid::new_v4().to_string(),
            ));
        }
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
    /// The server session this Timeline is connected to.
    pub(in crate::app) fn server_session(&self) -> Option<String> {
        self.session.clone()
    }
    pub(in crate::app) fn sampling_preview_session(&self) -> Option<(u64, String)> {
        Some((self.epoch, self.session.clone()?))
    }
    pub(in crate::app) fn sampling_preview_selection(&self) -> Option<(ProjectId, SceneId)> {
        if self.needs_reopen {
            return None;
        }
        let project = &self.document.as_ref()?.project;
        Some((project.id.clone(), self.primary_scene(project)?))
    }
    pub(in crate::app) fn sampling_preview_job_matches(
        &self,
        id: &str,
        project: &ProjectId,
        scene: &SceneId,
    ) -> bool {
        self.jobs.iter().any(|job| {
            job.id == id
                && !job.acting
                && job.matches(project, scene)
                && job.status() == jobs::JobStatus::Running
        })
    }
    /// Where a scene stands, as shown on its Timeline bar.
    fn scene_state(&self, project: &VideoProject, scene: &Scene) -> SceneState {
        if self
            .jobs
            .iter()
            .any(|j| j.matches(&project.id, &scene.id) && !j.terminal())
        {
            return SceneState::Generating;
        }
        let Some(document) = &self.document else {
            return SceneState::Empty;
        };
        if let Some(accepted) = &scene.accepted_take {
            return if document.take_state(&accepted.id).outdated {
                SceneState::Outdated
            } else {
                SceneState::Accepted
            };
        }
        if document
            .view
            .takes
            .iter()
            .any(|t| t.scene_id == scene.id && review::attached(document, t))
        {
            SceneState::Candidates
        } else {
            SceneState::Empty
        }
    }
    fn scene_active(&self) -> bool {
        let Some(document) = &self.document else {
            return false;
        };
        let Some(scene) = self.primary_scene(&document.project) else {
            return false;
        };
        self.jobs
            .iter()
            .any(|j| j.matches(&document.project.id, &scene) && !j.terminal())
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
        self.comparison_state = None;
        self.preview_asset = None;
    }
    fn scene(&self) -> Option<&Scene> {
        let project = &self.document.as_ref()?.project;
        project.scene(&self.primary_scene(project)?).ok()
    }
    fn sync_draft(&mut self) {
        self.reconcile_selection();
        if let Some(project) = self.document.as_ref().map(|d| &d.project) {
            self.section_selection.retain(|id| {
                project
                    .scenes
                    .iter()
                    .any(|s| s.sections.iter().any(|section| &section.id == id))
            });
            if self.section.is_some() {
                self.section = self
                    .section
                    .as_ref()
                    .filter(|id| {
                        project
                            .scenes
                            .iter()
                            .any(|s| s.sections.iter().any(|section| &section.id == *id))
                    })
                    .cloned()
                    .or_else(|| self.section_selection.last().cloned());
            }
            if let Some(id) = &self.section {
                if !self.section_selection.contains(id) {
                    self.section_selection = vec![id.clone()];
                }
            } else if self.audio_selected.is_empty()
                && !self
                    .selection_order
                    .iter()
                    .any(|item| matches!(item, selection::Item::Section(_)))
            {
                self.section_selection.clear();
            }
            self.audio_selection_order
                .retain(|id| self.audio_selected.contains(id));
        }
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
        self.global_prompt_options = self
            .document
            .as_ref()
            .and_then(|d| d.project.creative.global_prompt.clone());
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
        self.timeline_end_playback();
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
        let mut edits = self.generation_editor.scene_edits(&document.project);
        let mut creative = document.project.creative.clone();
        self.cast.merge_pending(&document.project, &mut creative);
        self.image_anchors
            .merge_pending(&document.project, &mut creative);
        if let Some(options) = &self.global_prompt_options {
            creative.global_prompt = Some(options.clone());
        }
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
                creative
                    .scene_prompts
                    .insert(scene.id.clone(), self.draft_scene_prompt.clone());
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
        if creative != document.project.creative {
            edits.push(Edit::SetCreative { creative });
        }
        edits
    }
}

#[cfg(test)]
mod mixed_selection_regression {
    use super::*;
    #[test]
    fn new_projects_start_at_quick_draft() {
        let project = new_project();
        let canvas = project.generation.unwrap().canvas.unwrap();
        assert_eq!(
            (canvas.ratio.as_str(), canvas.quality.as_str()),
            ("16:9", "Quick Draft")
        );
        assert_eq!(
            (project.delivery.width, project.delivery.height),
            (416, 224)
        );
    }
    #[test]
    fn audio_primary_keeps_secondary_sections_through_draft_refresh() {
        let mut p = super::media_picker::tests::project();
        let section = p.scenes[0].sections[0].id.clone();
        let audio = AudioClipId("own_mixed_audio".into());
        p.audio_tracks = vec![AudioTrack {
            id: AudioTrackId("own_track".into()),
            clips: vec![AudioClip::new(
                audio.clone(),
                Time { num: 0, den: 1 },
                Time { num: 1, den: 1 },
                &p.audio,
            )],
        }];
        let mut state = Timeline::default();
        state.document = Some(Snapshot::with(p, json!({})));
        state.section_selection = vec![section.clone()];
        state.audio_primary = Some(audio.clone());
        state.audio_selected = BTreeSet::from([audio.clone()]);
        state.audio_selection_order = vec![audio];
        state.sync_draft();
        assert_eq!(state.section_selection, vec![section]);
        assert!(state.section.is_none() && state.audio_draft.is_some());
        state.clear_audio_selection();
        state.sync_draft();
        assert!(state.section_selection.is_empty());
    }
}

#[cfg(test)]
mod undo_regression {
    use super::*;
    fn document(
        app: &mut ImagineApp,
        ctx: &egui::Context,
        project: &VideoProject,
        before: Option<&VideoProject>,
    ) {
        app.timeline_message(
            ctx,
            Event::Document {
                epoch: app.timeline.epoch,
                replace: false,
                before: before.map(|p| Box::new(p.clone())),
                history: 0,
                result: Ok(json!({ "project": project })),
            },
        );
    }
    #[test]
    fn undo_never_reverts_a_change_it_did_not_record() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        app.client = None;
        let original = super::media_picker::tests::project();
        app.timeline.document = Some(Snapshot::with(original.clone(), json!({})));
        // A recorded edit.
        let mut edited = original.clone();
        edited.global_prompt = "Own edited prompt".into();
        edited.project_revision += 1;
        document(&mut app, &ctx, &edited, Some(&original));
        assert!(!app.timeline.history.undo_commands(&edited).is_empty());
        // A server reply that only accepts a Take leaves undo intact.
        let mut accepted = edited.clone();
        accepted.scenes[0].accepted_take = Some(AcceptedTake {
            id: TakeId("own-take".into()),
            model: None,
            source_fingerprint: "own-fingerprint".into(),
        });
        accepted.project_revision += 1;
        document(&mut app, &ctx, &accepted, None);
        assert!(app.timeline.error.is_none(), "{:?}", app.timeline.error);
        assert!(!app.timeline.history.undo_commands(&accepted).is_empty());
        // An unrecorded authoring change, as a Library character copy makes:
        // undoing the earlier edit would silently revert it.
        let mut copied = accepted.clone();
        copied.creative.visual_references_enabled = Some(false);
        copied.project_revision += 1;
        document(&mut app, &ctx, &copied, None);
        let undo = app.timeline.history.undo_commands(&copied);
        assert!(
            !undo.iter().any(|e| matches!(e, Edit::SetCreative { .. })),
            "{undo:?}"
        );
        assert!(app.timeline.history.undo.is_empty() && app.timeline.history.redo.is_empty());
    }
}
