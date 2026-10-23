//! Review a frozen montage, then submit it to the shared queue as a CPU delivery.
use super::*;
use imagine_video_contract::assembly::{MixMode, Options, Plan, Request as FreezeRequest};
#[path = "assembly_flow.rs"]
mod flow;
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    pub(super) busy: bool,
    options: Options,
    pub(super) plan: Option<Plan>,
    pub(super) progress: Value,
    pub(super) error: Option<String>,
    poll_at: Option<std::time::Instant>,
    render_after_freeze: bool,
    pending_final: Option<flow::Pending>,
    freeze_options: Option<Options>,
    freeze_request: Option<FreezeRequest>,
    // Only typed project authoring metadata; never the raw snapshot, media,
    // resources or Take history. Move the baseline into the successful plan.
    freeze_project: Option<std::sync::Arc<VideoProject>>,
    plan_project: Option<std::sync::Arc<VideoProject>>,
    failure: Option<FailureDetails>,
}
#[derive(serde::Serialize)]
struct FailureDetails {
    operation: &'static str,
    code: String,
    retryable: bool,
    elapsed_ms: u64,
}
impl State {
    fn retry_request(&self) -> Option<FreezeRequest> {
        self.failure.as_ref().filter(|e| e.retryable)?;
        (!self.busy).then(|| self.freeze_request.clone()).flatten()
    }
    pub(super) fn project_changed(&mut self) {
        let pending = self.pending_final.take().is_some();
        self.busy = false;
        self.plan = None;
        self.progress = Value::Null;
        self.freeze_options = None;
        self.freeze_request = None;
        self.freeze_project = None;
        self.plan_project = None;
        self.failure = None;
        self.render_after_freeze = false;
        self.poll_at = None;
        self.error=pending.then(||"Final continuation ended because the project changed. The existing Run remains in the queue.".into());
    }
}
const UNFINISHED: &str = "Scenes without an accepted Take are generated first, with fresh seeds.";
const KEPT_EXPORT: &str =
    "The encoded export is kept for this session. Retry export in the queue to save it.";
fn assembly_phase_label(phase: &str) -> &'static str {
    match phase {
        "queued" => "Waiting in queue…",
        "mixing_generated_audio" | "mixing_timeline_audio" => "Mixing audio…",
        "decoding_clip" => "Reading clips…",
        "encoding_video" => "Encoding video…",
        "encoded" => "Encoded",
        "saving_result" => "Saving export…",
        "saved" => "Saved",
        "cancelled" => "Cancelled",
        "failed" => "Failed",
        "" => "Preparing…",
        _ => "Working…",
    }
}
fn earlier_snapshot(plan: &Plan, baseline: Option<&VideoProject>, document: &Snapshot) -> bool {
    if document.project.id != plan.project_id {
        return true;
    }
    let Some(baseline) = baseline.filter(|project| {
        project.id == plan.project_id && project.project_revision == plan.project_revision
    }) else {
        // No captured baseline: do not infer that an unseen revision is only a save.
        return document.project.project_revision != plan.project_revision;
    };
    let mut current = document.project.clone();
    current.project_revision = baseline.project_revision;
    for delivery in &document.view.deliveries {
        if delivery.plan.project_id != plan.project_id {
            continue;
        }
        let id = delivery.asset_id.clone();
        if baseline.assets.contains_key(&id) {
            continue;
        }
        // publication::prepare inserts precisely this new Asset as bookkeeping.
        // A renamed/revised delivery is an actual edit and must still be compared.
        let published = Asset {
            id: id.clone(),
            revision: 1,
            kind: MediaKind::Video,
            name: String::new(),
            library_source: None,
        };
        if current.assets.get(&id) == Some(&published) {
            current.assets.remove(&id);
        }
    }
    current != *baseline
}
fn control(ui: &mut egui::Ui, name: &str, response: &egui::Response) {
    #[cfg(feature = "ui-preview")]
    super::qa::control(ui, name.into(), response);
    let _ = (ui, name, response);
}
impl ImagineApp {
    pub(super) fn timeline_clip_source(
        &mut self,
        ctx: &egui::Context,
        scene: SceneId,
        asset: AssetId,
    ) {
        let (Some(client), Some(session), Some(document)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.document.as_ref(),
        ) else {
            return;
        };
        let Some(runtime_id) = document.runtime_asset(&asset.0).map(str::to_owned) else {
            return;
        };
        self.timeline.busy = true;
        let epoch = self.timeline.epoch;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                false,
                &format!("/v1/video/assets/{runtime_id}"),
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::ClipSource {
                    epoch,
                    scene,
                    asset,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_clip_source_reply(
        &mut self,
        ctx: &egui::Context,
        epoch: u64,
        scene: SceneId,
        asset: AssetId,
        result: Result<Value, client::TimelineFailure>,
    ) {
        if epoch != self.timeline.epoch {
            return;
        }
        self.timeline.busy = false;
        let result = result.map_err(|e| e.message).and_then(|info| {
            let current = self
                .timeline
                .document
                .as_ref()
                .and_then(|d| d.project.scene(&scene).ok())
                .ok_or_else(|| "The destination scene is no longer open.".to_owned())?;
            fitted_clip(current.duration, asset, &info)
        });
        match result {
            Ok(clip) => self.timeline_edit(
                ctx,
                vec![Edit::SetClipInstance {
                    scene_id: scene,
                    clip: Some(clip),
                }],
                0,
            ),
            Err(message) => self.timeline.error = Some(message),
        }
    }
    pub(super) fn timeline_assembly_boundary(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        commands: &mut Vec<Edit>,
    ) {
        use imagine_video_contract::assembly::BoundaryMode;
        let Some(index) = d.project.scenes.iter().position(|s| s.id == scene.id) else {
            return;
        };
        let Some(left) = index.checked_sub(1).and_then(|i| d.project.scenes.get(i)) else {
            return;
        };
        let mut boundary = boundary_for(&d.project, &left.id, &scene.id);
        let before = boundary.clone();
        ui.label("Incoming Scene boundary");
        for (mode, label, name) in [
            (BoundaryMode::HardCut, "Hard cut", "assembly-boundary-cut"),
            (
                BoundaryMode::ContinuousShot,
                "Continuous Shot · luminance match",
                "assembly-boundary-continuous",
            ),
            (
                BoundaryMode::BlendSeam,
                "Blend seam",
                "assembly-boundary-blend",
            ),
            (
                BoundaryMode::Transition,
                "Generated transition",
                "assembly-boundary-transition",
            ),
        ] {
            let r = ui.selectable_value(&mut boundary.mode, mode, label);
            control(ui, name, &r);
        }
        if boundary.mode == BoundaryMode::BlendSeam {
            ui.horizontal(|ui| {
                ui.label("Overlap frames");
                let (r, value) = helto::settled(
                    ui,
                    ("boundary-blend", &boundary.right.0),
                    &boundary.blend_frames,
                    |ui, v| ui.add(egui::DragValue::new(v).range(1..=u32::MAX)),
                );
                control(ui, "assembly-overlap", &r);
                if let Some(value) = value {
                    boundary.blend_frames = value;
                }
            });
        }
        if boundary.mode != BoundaryMode::HardCut {
            let mut custom = boundary.tail_frames.is_some();
            let response = ui.checkbox(&mut custom, "Custom reference tail (frames)");
            control(ui, "boundary-tail-custom", &response);
            if response.changed() {
                boundary.tail_frames = custom.then(|| {
                    boundary
                        .requested_tail_frames(d.project.delivery.fps)
                        .unwrap_or(0)
                });
            }
            if let Some(tail) = &mut boundary.tail_frames {
                let (response, value) =
                    helto::settled(ui, ("boundary-tail", &boundary.right.0), &*tail, |ui, v| {
                        ui.add(egui::DragValue::new(v))
                    });
                control(ui, "boundary-tail-frames", &response);
                if let Some(value) = value {
                    *tail = value;
                }
            } else {
                ui.weak(format!(
                    "Default: 3 seconds · {} requested frames",
                    boundary
                        .requested_tail_frames(d.project.delivery.fps)
                        .unwrap_or(0)
                ));
            }
            if boundary.mode == BoundaryMode::Transition {
                ui.label("Transition prompt");
                // Saved when the field loses focus, so typing keeps it.
                let (response, value) = helto::settled(
                    ui,
                    ("boundary-transition", &boundary.right.0),
                    &boundary.transition_prompt,
                    |ui, v| {
                        ui.add(
                            egui::TextEdit::multiline(v)
                                .desired_width(f32::INFINITY)
                                .desired_rows(3),
                        )
                    },
                );
                control(ui, "boundary-transition-prompt", &response);
                if let Some(value) = value {
                    boundary.transition_prompt = value;
                }
            }
            ui.checkbox(
                &mut boundary.reuse_visual_references,
                "Reuse visual references",
            );
            ui.checkbox(&mut boundary.reuse_style, "Reuse style");
            ui.label("Opening audio");
            use imagine_video_contract::assembly::AudioContinuity;
            for (value, label, name) in [
                (
                    AudioContinuity::NewSceneAudio,
                    "New Scene Audio",
                    "boundary-audio-new",
                ),
                (
                    AudioContinuity::ReferencePreviousAudio,
                    "Reference Previous Take Audio",
                    "boundary-audio-reference",
                ),
            ] {
                let response = ui.selectable_value(&mut boundary.audio_continuity, value, label);
                control(ui, name, &response);
            }
            ui.weak("Conditions the next Scene's ordinary Take. Generate that Scene again to apply changed boundary inputs. Final joins the resulting Takes; it does not add a third transition clip.");
        }
        if before != boundary {
            if let Some(continuity) =
                linked_continuity(&d.project, scene, &left.id, &before.mode, &boundary.mode)
            {
                commands.extend(self.timeline.draft_edits());
                commands.push(Edit::SetContinuity {
                    scene_id: scene.id.clone(),
                    continuity,
                });
            }
            commands.push(Edit::SetAssemblyBoundary { boundary });
        }
        ui.separator();
    }
    pub(super) fn timeline_assembly_clip(
        &mut self,
        ui: &mut egui::Ui,
        d: &Snapshot,
        scene: &Scene,
        commands: &mut Vec<Edit>,
    ) {
        if d.project.scene_kind(&scene.id) != imagine_video_contract::creative::SceneKind::Imported
        {
            return;
        }
        let pick = ui
            .horizontal(|ui| {
                ui.label("Scene clip");
                helto::icon_button(
                    ui,
                    crate::icons::Icon::Folder,
                    "Choose imported scene clip",
                    false,
                )
            })
            .inner;
        control(ui, "scene-clip-pick", &pick);
        if pick.clicked() {
            self.timeline_pick_media(media_picker::Target::SceneClip(scene.id.clone()));
        }
        let Some(mut clip) = d.project.assembly.clips.get(&scene.id).cloned() else {
            ui.weak("Choose a video for this imported scene.");
            return;
        };
        let before = clip.clone();
        ui.group(|ui| {
            ui.strong("Imported scene clip");
            ui.weak("Fitted to this scene. Change trim or speed to retime it.");
            let mut start = imagine_video_contract::assembly::seconds(clip.source_in);
            let mut end = clip
                .source_out
                .map(imagine_video_contract::assembly::seconds)
                .unwrap_or(
                    start + imagine_video_contract::assembly::seconds(scene.duration) * clip.speed,
                );
            ui.horizontal(|ui| {
                ui.label("Source in");
                let (r, value) = helto::settled(ui, ("clip-in", &scene.id.0), &start, |ui, v| {
                    ui.add(egui::DragValue::new(v).speed(0.01).range(0. ..=f64::MAX))
                });
                control(ui, "assembly-source-in", &r);
                if let Some(value) = value {
                    start = value;
                }
            });
            let mut trimmed = clip.source_out.is_some();
            let r = ui.checkbox(&mut trimmed, "Source out");
            control(ui, "assembly-source-out-enabled", &r);
            if trimmed {
                let (r, value) = helto::settled(ui, ("clip-out", &scene.id.0), &end, |ui, v| {
                    ui.add(egui::DragValue::new(v).speed(0.01).range(0. ..=f64::MAX))
                });
                control(ui, "assembly-source-out", &r);
                if let Some(value) = value {
                    end = value;
                }
            }
            clip.source_in = imagine_video_contract::assembly::time(start);
            clip.source_out = trimmed.then(|| imagine_video_contract::assembly::time(end));
            ui.horizontal(|ui| {
                ui.label("Speed");
                let (r, value) =
                    helto::settled(ui, ("clip-speed", &scene.id.0), &clip.speed, |ui, v| {
                        ui.add(
                            egui::DragValue::new(v)
                                .speed(0.05)
                                .range(f64::MIN_POSITIVE..=f64::MAX)
                                .suffix("×"),
                        )
                    });
                control(ui, "assembly-speed", &r);
                if let Some(value) = value {
                    clip.speed = value;
                }
            });
            let r = ui.checkbox(&mut clip.include_audio, "Include source audio");
            control(ui, "assembly-clip-audio", &r);
            let r = helto::icon_button(
                ui,
                crate::icons::Icon::Unlink,
                "Use generated Take instead",
                false,
            );
            control(ui, "assembly-remove-clip", &r);
            if r.clicked() {
                commands.push(Edit::SetClipInstance {
                    scene_id: scene.id.clone(),
                    clip: None,
                });
            }
        });
        if before != clip {
            let duration = if let Some(end) = clip.source_out {
                (imagine_video_contract::assembly::seconds(end)
                    - imagine_video_contract::assembly::seconds(clip.source_in))
                    / clip.speed
            } else {
                imagine_video_contract::assembly::seconds(scene.duration) * before.speed
                    / clip.speed
            };
            if duration > 0. {
                commands.extend(super::editing::resize_scene(
                    &d.project,
                    scene,
                    duration,
                    self.timeline.mode,
                ));
            }
            commands.push(Edit::SetClipInstance {
                scene_id: scene.id.clone(),
                clip: Some(clip),
            });
        }
    }
    fn assembly_call(&mut self, ctx: &egui::Context, progress: bool) {
        if self.timeline.assembly.busy {
            return;
        }
        let Some(document) = self.timeline.document.as_ref() else {
            return;
        };
        let (path, body) = if progress {
            let Some(plan) = &self.timeline.assembly.plan else {
                return;
            };
            (format!("/v1/video/assemblies/{}", plan.id), None)
        } else {
            let options = self
                .timeline
                .assembly
                .freeze_options
                .take()
                .unwrap_or_else(|| self.timeline.assembly.options.clone());
            let request = FreezeRequest {
                schema_version: 1,
                project_id: document.project.id.clone(),
                expected_project_revision: document.project.project_revision,
                options,
            };
            let body = json!(request);
            self.timeline.assembly.freeze_project =
                Some(std::sync::Arc::new(document.project.clone()));
            self.timeline.assembly.freeze_request = Some(request);
            (
                format!("/v1/video/projects/{}/assembly", document.project.id.0),
                Some(body),
            )
        };
        self.assembly_send(ctx, progress, path, body);
    }
    fn assembly_retry(&mut self, ctx: &egui::Context) {
        let Some(request) = self.timeline.assembly.retry_request() else {
            return;
        };
        self.timeline.assembly.render_after_freeze = true;
        self.assembly_send(
            ctx,
            false,
            format!("/v1/video/projects/{}/assembly", request.project_id.0),
            Some(json!(request)),
        );
    }
    fn assembly_send(
        &mut self,
        ctx: &egui::Context,
        progress: bool,
        path: String,
        body: Option<Value>,
    ) {
        if self.timeline.assembly.busy {
            return;
        }
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.assembly.busy = true;
        self.timeline.assembly.failure = None;
        let epoch = self.timeline.epoch;
        let ctx = ctx.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = runtime().block_on(client.timeline_request(
                &session,
                !progress,
                &path,
                body.as_ref(),
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Assembly {
                    session,
                    epoch,
                    progress,
                    elapsed_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_assembly_reply(
        &mut self,
        ctx: &egui::Context,
        session: String,
        epoch: u64,
        progress: bool,
        elapsed_ms: u64,
        result: Result<Value, client::TimelineFailure>,
    ) {
        if self.timeline.session.as_deref() != Some(&session) || self.timeline.epoch != epoch {
            return;
        }
        self.timeline.assembly.busy = false;
        match result {
            Ok(value) => {
                self.timeline.assembly.error = None;
                self.timeline.assembly.failure = None;
                if progress {
                    self.timeline.assembly.progress = value["progress"].clone();
                    self.timeline.assembly.progress["rendered_result_available"] =
                        value["rendered_result_available"].clone();
                } else {
                    match serde_json::from_value::<Plan>(value) {
                        Ok(plan) => {
                            self.timeline.assembly.freeze_request = None;
                            self.timeline.assembly.plan_project = self
                                .timeline
                                .assembly
                                .freeze_project
                                .take()
                                .filter(|project| {
                                    project.id == plan.project_id
                                        && project.project_revision == plan.project_revision
                                });
                            self.timeline.assembly.plan = Some(plan);
                            self.timeline.assembly.progress = Value::Null;
                        }
                        Err(_) => {
                            self.timeline.assembly.error =
                                Some("Could not read the frozen montage plan.".into())
                        }
                    }
                }
            }
            Err(error) => {
                if !progress && !error.retryable {
                    self.timeline.assembly.freeze_request = None;
                    self.timeline.assembly.freeze_project = None;
                }
                self.timeline.assembly.failure = Some(FailureDetails {
                    operation: if progress { "progress" } else { "freeze" },
                    code: error.code,
                    retryable: error.retryable,
                    elapsed_ms,
                });
                self.timeline.assembly.error = Some(error.message);
            }
        }
        if self.timeline.assembly.error.is_some() {
            self.timeline.assembly.render_after_freeze = false;
        }
        if !progress
            && self.timeline.assembly.render_after_freeze
            && self.timeline.assembly.plan.is_some()
        {
            self.timeline.assembly.render_after_freeze = false;
            if self.timeline.pending.is_some() || self.timeline.admitting {
                self.timeline.assembly.error=Some("Montage prepared. Retrieve the other unconfirmed submission before rendering this frozen plan.".into());
            } else {
                self.timeline_assembly_submit(ctx);
            }
        }
    }
    pub(super) fn timeline_assembly_render_current(&mut self, ctx: &egui::Context) {
        if self.timeline.assembly.busy
            || self.timeline.busy
            || self.timeline.admitting
            || self.timeline.pending.is_some()
            || self.timeline.assembly.pending_final.is_some()
        {
            return;
        }
        self.timeline_final_start(ctx);
    }
    pub(super) fn timeline_assembly_submit(&mut self, ctx: &egui::Context) {
        if self.timeline.pending.is_some() || self.timeline.admitting {
            return;
        }
        let Some(plan) = &self.timeline.assembly.plan else {
            return;
        };
        self.timeline.pending = Some((
            json!({"kind":"video_assembly","schema_version":1,"project_id":plan.project_id,"expected_project_revision":plan.project_revision,"assembly_plan":plan}),
            uuid::Uuid::new_v4().to_string(),
        ));
        self.timeline_submit_pending(ctx);
    }
    /// Optional inspection of a frozen plan before rendering it.
    fn assembly_plan_section(
        &self,
        ui: &mut egui::Ui,
        document: &Option<Snapshot>,
        busy: bool,
    ) -> (bool, bool) {
        let (mut capture, mut submit) = (false, false);
        let state = &self.timeline.assembly;
        egui::CollapsingHeader::new(helto::section("Frozen plan"))
            .id_salt("assembly-plan-section")
            .open(state.plan.is_some().then_some(true))
            .show(ui, |ui| {
                let r = helto::action(
                    ui,
                    crate::icons::Icon::Lock,
                    "Freeze current montage plan",
                    false,
                    if document.is_none() {
                        Some("Create or open a project first.")
                    } else {
                        busy.then_some("Wait for the current operation.")
                    },
                );
                control(ui, "assembly-freeze", &r);
                capture = r.clicked();
                let Some(plan) = &state.plan else {
                    return;
                };
                let fps = imagine_video_contract::assembly::seconds(plan.delivery.fps);
                ui.label(format!(
                    "{} × {} · {:.3} fps · {} frames · {:.2} s",
                    plan.delivery.width,
                    plan.delivery.height,
                    fps,
                    plan.frames,
                    plan.frames as f64 / fps
                ));
                ui.weak(format!(
                    "Revision {} · {} clips · {} timeline audio segments",
                    plan.project_revision,
                    plan.clips.len(),
                    plan.audio.len()
                ));
                if document
                    .as_ref()
                    .is_some_and(|d| earlier_snapshot(plan, state.plan_project.as_deref(), d))
                {
                    let r = ui.colored_label(
                        helto::ACCENT,
                        "This is an earlier project snapshot. Freeze again to include current edits.",
                    );
                    control(ui, "assembly-snapshot-warning", &r);
                }
                egui::ScrollArea::vertical()
                    .id_salt("assembly-plan")
                    .max_height(140.)
                    .show(ui, |ui| {
                        for (i, clip) in plan.clips.iter().enumerate() {
                            let source = if clip.take_id.is_some() {
                                "selected Take"
                            } else {
                                "imported video"
                            };
                            ui.label(format!(
                                "Scene {} · {:?} · {} frames / {} overlap · {source}",
                                i + 1,
                                clip.boundary,
                                clip.frames,
                                clip.overlap
                            ));
                        }
                        for warning in &plan.warnings {
                            ui.colored_label(helto::ACCENT, warning);
                        }
                    });
                let r = helto::action(
                    ui,
                    crate::icons::Icon::Generate,
                    "Render frozen plan",
                    true,
                    (busy || self.timeline.admitting || self.timeline.pending.is_some())
                        .then_some("Wait for the current operation."),
                );
                control(ui, "assembly-render", &r);
                submit = r.clicked();
            });
        (capture, submit)
    }
    pub(super) fn timeline_assembly_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.assembly.open {
            return;
        }
        let mut open = true;
        let mut capture = false;
        let mut submit = false;
        let mut preview = None;
        let mut render_current = false;
        let mut retry = false;
        let document = self.timeline.document.clone();
        let busy = self.timeline.assembly.busy;
        let before = self.timeline.assembly.options.clone();
        let wait = busy
            || self.timeline.busy
            || self.timeline.admitting
            || self.timeline.pending.is_some()
            || self.timeline.assembly.pending_final.is_some();
        let unfinished = document
            .as_ref()
            .is_some_and(|d| flow::needs_generation(&d.project));
        let width = (ctx.content_rect().width() - 24.).max(240.);
        egui::Window::new("Final video")
            .id(egui::Id::new("video-final-montage"))
            .open(&mut open)
            .default_width(540_f32.min(width))
            .default_height(640.)
            .max_width(width)
            .max_height((ctx.content_rect().height() - 70.).max(160.))
            .vscroll(true)
            .resizable(true)
            .show(ctx, |ui| {
                ui.set_max_width((ctx.content_rect().width() - 48.).max(216.));
                #[cfg(feature = "ui-preview")]
                {
                    let r = ui.interact(
                        ui.clip_rect(),
                        ui.id().with("qa-assembly-scroll"),
                        egui::Sense::hover(),
                    );
                    control(ui, "assembly-scroll", &r);
                }
                let hidden = !helto::reveal_in_viewport(ui, self.prefs.privacy_mode);
                helto::private_scope(ui, hidden, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        let r = helto::action(
                            ui,
                            crate::icons::Icon::Generate,
                            "Render current Timeline",
                            true,
                            if document.is_none() {
                                Some("Create or open a project first.")
                            } else {
                                wait.then_some("Wait for the current operation.")
                            },
                        );
                        control(ui, "assembly-render-current", &r);
                        render_current = r.clicked();
                        if busy {
                            ui.spinner();
                        }
                        if let Some(phase) = self.timeline.assembly.progress["phase"].as_str() {
                            let r = ui.weak(assembly_phase_label(phase));
                            control(ui, "assembly-progress", &r);
                        }
                    });
                    if self.timeline.assembly.pending_final.is_some() {
                        ui.colored_label(
                            helto::ACCENT,
                            "Generating scenes without a Take, with fresh seeds…",
                        );
                    } else if unfinished {
                        ui.weak(UNFINISHED);
                    }
                    if self.timeline.assembly.retry_request().is_some() {
                        ui.horizontal_wrapped(|ui| {
                            let r = helto::icon_button(
                                ui,
                                crate::icons::Icon::Retry,
                                "Retry Final preparation",
                                false,
                            );
                            control(ui, "assembly-retry-preparation", &r);
                            retry = r.clicked();
                            ui.weak("Retry uses the original Timeline revision and choices.");
                        });
                    }
                    if let Some(error) = &self.timeline.assembly.error {
                        ui.colored_label(helto::ERROR, error);
                    }
                    if self.timeline.assembly.progress["rendered_result_available"] == true {
                        ui.weak(KEPT_EXPORT);
                    }
                    ui.add_space(8.);
                    ui.horizontal(|ui| {
                        ui.label(helto::section("Output"));
                        ui.weak("MP4 · H.264 CRF 19 · AAC stereo 44.1 kHz");
                    });
                    ui.add_enabled_ui(!busy, |ui| {
                        export_options(ui, &mut self.timeline.assembly.options)
                    });
                    ui.add_space(8.);
                    let (frozen, render) = self.assembly_plan_section(ui, &document, busy);
                    capture = frozen;
                    submit = render;
                    ui.add_space(8.);
                    ui.label(helto::section("Deliveries"));
                    let deliveries = document
                        .as_ref()
                        .map_or(&[][..], |d| &d.view.deliveries[..]);
                    if deliveries.is_empty() {
                        ui.weak("No deliveries yet.");
                    }
                    for (i, delivery) in deliveries.iter().enumerate() {
                        let plan = &delivery.plan;
                        let fps = imagine_video_contract::assembly::seconds(plan.delivery.fps);
                        let r = ui
                            .selectable_label(
                                false,
                                format!(
                                    "Delivery {} · revision {} · {} × {} · {:.1} s",
                                    i + 1,
                                    plan.project_revision,
                                    plan.delivery.width,
                                    plan.delivery.height,
                                    plan.frames as f64 / fps
                                ),
                            )
                            .on_hover_text("Show in the viewer");
                        control(ui, &format!("assembly-delivery-{i}"), &r);
                        if r.clicked() {
                            preview = document
                                .as_ref()
                                .and_then(|d| d.runtime_asset(&delivery.asset_id.0))
                                .map(str::to_owned);
                        }
                    }
                });
            });
        self.timeline.assembly.open = open;
        if before != self.timeline.assembly.options {
            self.timeline.assembly.plan = None;
            self.timeline.assembly.plan_project = None;
        }
        if capture {
            self.timeline.assembly.render_after_freeze = false;
            self.assembly_call(ctx, false);
        }
        if render_current {
            let commands = self.timeline.draft_edits();
            if commands.is_empty() {
                self.timeline_assembly_render_current(ctx);
            } else {
                self.timeline.after_edit = Some(PostAction::RenderAssembly);
                self.timeline_edit(ctx, commands, 0);
            }
        }
        if submit {
            self.timeline_assembly_submit(ctx);
        }
        if retry {
            self.assembly_retry(ctx);
        }
        if let Some(asset) = preview {
            self.timeline_load_media(ctx, asset);
            self.timeline.assembly.open = false;
        }
        let active = self
            .timeline
            .jobs
            .iter()
            .any(|j| j.is_assembly() && !j.terminal());
        if active
            && !busy
            && self
                .timeline
                .assembly
                .poll_at
                .is_none_or(|t| t.elapsed().as_millis() >= 500)
        {
            self.timeline.assembly.poll_at = Some(std::time::Instant::now());
            self.assembly_call(ctx, true);
        }
    }
}
/// Filename, picture and audio choices for the next export.
fn export_options(ui: &mut egui::Ui, options: &mut Options) {
    egui::Grid::new("assembly-output")
        .num_columns(2)
        .spacing([12., 6.])
        .show(ui, |ui| {
            let label = ui.label("Filename");
            let mut name = options.output_filename.clone().unwrap_or_default();
            let r = ui
                .add(
                    egui::TextEdit::singleline(&mut name)
                        .hint_text("Automatic numbered name")
                        .desired_width(300.),
                )
                .labelled_by(label.id);
            control(ui, "assembly-output-filename", &r);
            if r.changed() {
                options.output_filename = (!name.is_empty()).then_some(name);
            }
            ui.end_row();
            ui.label("Color depth");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut options.bit_depth, 8, "8-bit");
                ui.selectable_value(&mut options.bit_depth, 10, "10-bit");
            });
            ui.end_row();
            ui.label("Continuous shots");
            ui.checkbox(&mut options.continuous_tone_match, "Match luminance");
            ui.end_row();
        });
    ui.add_space(8.);
    ui.label(helto::section("Audio"));
    egui::Grid::new("assembly-audio")
        .num_columns(2)
        .spacing([12., 6.])
        .show(ui, |ui| {
            ui.label("Source");
            ui.horizontal(|ui| {
                for (mode, label, name) in [
                    (MixMode::Generated, "Generated", "assembly-audio-generated"),
                    (MixMode::Timeline, "Timeline", "assembly-audio-timeline"),
                    (MixMode::Combined, "Combined", "assembly-audio-combined"),
                ] {
                    let r = ui.selectable_value(&mut options.audio_mode, mode, label);
                    control(ui, name, &r);
                }
            });
            ui.end_row();
            if options.audio_mode != MixMode::Timeline {
                ui.label("Generated gain");
                ui.add(
                    egui::DragValue::new(&mut options.generated_gain)
                        .speed(0.05)
                        .range(0. ..=f64::MAX)
                        .suffix("×"),
                );
                ui.end_row();
            }
            if options.audio_mode != MixMode::Generated {
                ui.label("Timeline gain");
                ui.add(
                    egui::DragValue::new(&mut options.timeline_gain)
                        .speed(0.05)
                        .range(0. ..=f64::MAX)
                        .suffix("×"),
                );
                ui.end_row();
            }
            ui.label("Level");
            let r = ui
                .checkbox(&mut options.normalize_mix, "Normalize mix")
                .on_hover_text("RMS −18 dB · peak −1 dB · at most 2× gain");
            control(ui, "assembly-normalize", &r);
            ui.end_row();
            if options.audio_mode == MixMode::Combined {
                ui.label("Ducking");
                ui.horizontal(|ui| {
                    let r = ui.checkbox(
                        &mut options.duck_generated,
                        "Duck generated under timeline audio",
                    );
                    control(ui, "assembly-duck", &r);
                    ui.add_enabled(
                        options.duck_generated,
                        egui::DragValue::new(&mut options.ducking_strength)
                            .speed(0.05)
                            .range(0. ..=1.),
                    )
                    .on_hover_text("Ducking strength");
                });
                ui.end_row();
            }
        });
}
// A new seam uses Director's three-frame overlap; persisted boundaries retain
// their explicit overlap when reopened or when only their mode is edited.
pub(super) fn boundary_for(
    project: &VideoProject,
    left: &SceneId,
    right: &SceneId,
) -> imagine_video_contract::assembly::Boundary {
    use imagine_video_contract::assembly::{Boundary, BoundaryMode};
    project
        .assembly
        .boundaries
        .iter()
        .find(|boundary| &boundary.left == left && &boundary.right == right)
        .cloned()
        .unwrap_or_else(|| Boundary {
            left: left.clone(),
            right: right.clone(),
            mode: BoundaryMode::HardCut,
            blend_frames: 3,
            ..Boundary::new(left.clone(), right.clone())
        })
}
/// A Continuous shot also carries the previous Scene's motion into this one;
/// a Hard cut removes that again. Other continuity the user chose is kept.
fn linked_continuity(
    project: &VideoProject,
    scene: &Scene,
    left: &SceneId,
    before: &imagine_video_contract::assembly::BoundaryMode,
    after: &imagine_video_contract::assembly::BoundaryMode,
) -> Option<Option<Continuity>> {
    use imagine_video_contract::assembly::BoundaryMode;
    use imagine_video_contract::creative::SceneKind;
    match (before, after) {
        (_, BoundaryMode::ContinuousShot)
            if scene.continuity.is_none()
                && ![left, &scene.id]
                    .into_iter()
                    .any(|id| project.scene_kind(id) == SceneKind::Imported) =>
        {
            Some(Some(Continuity {
                source_scene: left.clone(),
                method: ContinuityMethod::MotionContextExp,
                include_audio: false,
            }))
        }
        // Whether it continues the source audio is the user's choice either way.
        (BoundaryMode::ContinuousShot, BoundaryMode::HardCut)
            if scene.continuity.as_ref().is_some_and(|c| {
                &c.source_scene == left && c.method == ContinuityMethod::MotionContextExp
            }) =>
        {
            Some(None)
        }
        _ => None,
    }
}
fn fitted_clip(
    scene_duration: Time,
    asset: AssetId,
    info: &Value,
) -> Result<imagine_video_contract::assembly::ClipInstance, String> {
    let fps = serde_json::from_value::<Time>(info["media"]["fps"].clone())
        .ok()
        .filter(|t| t.positive());
    let duration = fps
        .zip(info["media"]["frames"].as_u64())
        .and_then(|(fps, frames)| {
            fps.den
                .checked_mul(frames)
                .map(|num| Time { num, den: fps.num })
        })
        .filter(|t| t.positive())
        .or_else(|| {
            serde_json::from_value::<Time>(info["media"]["duration"].clone())
                .ok()
                .filter(|t| t.positive())
        })
        .ok_or_else(|| "The source duration is unavailable. Check the selected video.".to_owned())?
        .normalized();
    Ok(imagine_video_contract::assembly::ClipInstance {
        asset,
        source_in: Time { num: 0, den: 1 },
        source_out: Some(duration),
        speed: imagine_video_contract::assembly::seconds(duration)
            / imagine_video_contract::assembly::seconds(scene_duration),
        include_audio: true,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn snapshot_and_plan() -> (Snapshot, Plan) {
        let project = media_picker::tests::project();
        let plan = Plan {
            schema_version: 1,
            id: "own-plan".into(),
            project_id: project.id.clone(),
            project_revision: project.project_revision,
            fingerprint: "own-test".into(),
            delivery: project.delivery.clone(),
            sample_rate: 44_100,
            frames: 48,
            samples: 88_200,
            options: Options::default(),
            clips: vec![],
            audio: vec![],
            warnings: vec![],
        };
        (Snapshot::with(project, json!({"deliveries":[]})), plan)
    }
    fn publish_delivery(document: &mut Snapshot, plan: &Plan, id: &str) {
        let asset = AssetId(id.into());
        document.project.assets.insert(
            asset.clone(),
            Asset {
                id: asset,
                revision: 1,
                kind: MediaKind::Video,
                name: String::new(),
                library_source: None,
            },
        );
        document.project.project_revision += 1;
        document.view.deliveries.push(snapshot::DeliveryView {
            id: format!("delivery-{id}"),
            asset_id: AssetId(id.into()),
            plan: plan.clone(),
        });
    }
    #[test]
    fn final_video_groups_choices_and_shows_only_the_relevant_audio_controls() {
        let root = tempfile::tempdir().unwrap();
        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let mut app = ImagineApp::synthetic_preview(&ctx, root.path().into());
        let (mut document, plan) = snapshot_and_plan();
        publish_delivery(&mut document, &plan, "own-delivery");
        app.workspace = Workspace::Video;
        app.timeline.session = Some("own-session".into());
        app.timeline.session_checked = true;
        app.timeline.document = Some(document);
        app.timeline.assembly.open = true;
        let frame = |app: &mut ImagineApp| {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280., 900.),
                    )),
                    focused: true,
                    ..Default::default()
                },
                |ui| app.content(ui),
            );
            let nodes = output
                .platform_output
                .accesskit_update
                .take()
                .unwrap()
                .nodes;
            (output, nodes)
        };
        let has = |nodes: &Vec<(egui::accesskit::NodeId, egui::accesskit::Node)>, text: &str| {
            nodes.iter().any(|(_, n)| {
                n.label()
                    .into_iter()
                    .chain(n.value())
                    .any(|t| t.contains(text))
            })
        };
        ctx.global_style_mut(|style| style.animation_time = 0.);
        let mut output = egui::FullOutput::default();
        let mut nodes = vec![];
        for _ in 0..4 {
            let (o, n) = frame(&mut app);
            output.append(o);
            nodes = n;
        }
        if let Some(dir) = std::env::var_os("IMAGINE_UI_PREVIEW_DIR") {
            crate::app::tests::render(&ctx, output, 1280, 900)
                .save(std::path::PathBuf::from(dir).join("final.png"))
                .unwrap();
        } else {
            output.textures_delta.clear();
        }
        for text in [
            "Render current Timeline",
            "Output",
            "Filename",
            "Audio",
            "Generated gain",
            "Timeline gain",
            "Ducking",
            "Frozen plan",
            "Deliveries",
            "Delivery 1 · revision",
            UNFINISHED,
        ] {
            assert!(has(&nodes, text), "{text}");
        }
        assert!(!has(&nodes, "Render frozen plan"));
        app.timeline.assembly.options.audio_mode = MixMode::Generated;
        let (mut o, nodes) = frame(&mut app);
        o.textures_delta.clear();
        assert!(has(&nodes, "Generated gain"));
        assert!(!has(&nodes, "Timeline gain") && !has(&nodes, "Ducking"));
        app.timeline.assembly.plan = Some(plan);
        let (mut o, nodes) = frame(&mut app);
        o.textures_delta.clear();
        assert!(has(&nodes, "Render frozen plan"));
    }
    #[test]
    fn final_phases_distinguish_encoding_saving_and_saved() {
        for (phase, label) in [
            ("queued", "Waiting in queue…"),
            ("mixing_generated_audio", "Mixing audio…"),
            ("mixing_timeline_audio", "Mixing audio…"),
            ("decoding_clip", "Reading clips…"),
            ("encoding_video", "Encoding video…"),
            ("encoded", "Encoded"),
            ("saving_result", "Saving export…"),
            ("saved", "Saved"),
            ("cancelled", "Cancelled"),
            ("failed", "Failed"),
            ("", "Preparing…"),
            ("future_phase", "Working…"),
        ] {
            assert_eq!(assembly_phase_label(phase), label);
        }
    }
    #[test]
    fn publication_alone_does_not_make_a_frozen_plan_stale() {
        let (mut document, plan) = snapshot_and_plan();
        let baseline = document.project.clone();
        assert!(!earlier_snapshot(&plan, Some(&baseline), &document));
        for id in ["own-delivery-1", "own-delivery-2"] {
            publish_delivery(&mut document, &plan, id);
            assert!(!earlier_snapshot(&plan, Some(&baseline), &document));
        }
        // The same publication revision may also include a real authoring edit.
        document.project.global_prompt = "own changed prompt".into();
        assert!(earlier_snapshot(&plan, Some(&baseline), &document));
    }
    #[test]
    fn snapshot_comparison_preserves_real_asset_and_authoring_edits() {
        let (document, plan) = snapshot_and_plan();
        let baseline = document.project.clone();
        let mut edited = document.clone();
        edited.project.global_prompt = "own edit before delivery".into();
        edited.project.project_revision += 1;
        assert!(earlier_snapshot(&plan, Some(&baseline), &edited));
        let mut delivered = document.clone();
        publish_delivery(&mut delivered, &plan, "own-delivery");
        delivered
            .project
            .assets
            .get_mut(&AssetId("own-delivery".into()))
            .unwrap()
            .name = "renamed".into();
        assert!(earlier_snapshot(&plan, Some(&baseline), &delivered));
        delivered
            .project
            .assets
            .get_mut(&AssetId("own-delivery".into()))
            .unwrap()
            .name
            .clear();
        delivered.view.deliveries.clear();
        assert!(earlier_snapshot(&plan, Some(&baseline), &delivered));
        let mut existing = document;
        existing
            .project
            .assets
            .get_mut(&AssetId("video".into()))
            .unwrap()
            .revision += 1;
        publish_delivery(&mut existing, &plan, "video");
        existing
            .project
            .assets
            .get_mut(&AssetId("video".into()))
            .unwrap()
            .revision += 1;
        assert!(earlier_snapshot(&plan, Some(&baseline), &existing));
    }
    #[test]
    fn absent_or_mismatched_baseline_does_not_guess_that_revision_is_a_delivery() {
        let (mut document, plan) = snapshot_and_plan();
        assert!(!earlier_snapshot(&plan, None, &document));
        let mut stale_baseline = document.project.clone();
        stale_baseline.project_revision += 1;
        publish_delivery(&mut document, &plan, "own-delivery");
        assert!(earlier_snapshot(&plan, None, &document));
        assert!(earlier_snapshot(&plan, Some(&stale_baseline), &document));
        let baseline = snapshot_and_plan().0.project;
        document.project.id = ProjectId("other-project".into());
        assert!(earlier_snapshot(&plan, Some(&baseline), &document));
    }
    #[test]
    fn a_continuous_shot_carries_the_previous_motion_and_a_hard_cut_removes_it() {
        use imagine_video_contract::assembly::BoundaryMode::*;
        use imagine_video_contract::creative::SceneKind;
        let mut project = media_picker::tests::project();
        let left = project.scenes[0].id.clone();
        let mut scene = project.scenes[0].clone();
        scene.id = SceneId("own-right-scene".into());
        scene.start = Time { num: 2, den: 1 };
        scene.sections.clear();
        project.scenes.push(scene.clone());
        let link = |project: &VideoProject, scene: &Scene, before, after| {
            linked_continuity(project, scene, &left, &before, &after)
        };
        let motion = Continuity {
            source_scene: left.clone(),
            method: ContinuityMethod::MotionContextExp,
            include_audio: false,
        };
        assert_eq!(
            link(&project, &scene, HardCut, ContinuousShot),
            Some(Some(motion.clone()))
        );
        let mut boundary = boundary_for(&project, &left, &scene.id);
        boundary.mode = ContinuousShot;
        let revision = project.project_revision;
        imagine_video_contract::editing::apply(
            &mut project,
            EditRequest {
                schema_version: 1,
                expected_project_revision: revision,
                commands: vec![
                    Edit::SetContinuity {
                        scene_id: scene.id.clone(),
                        continuity: Some(motion.clone()),
                    },
                    Edit::SetAssemblyBoundary { boundary },
                ],
            },
        )
        .unwrap();
        let mut scene = project.scenes[1].clone();
        assert_eq!(scene.continuity, Some(motion.clone()));
        assert_eq!(link(&project, &scene, ContinuousShot, HardCut), Some(None));
        scene.continuity = Some(Continuity {
            include_audio: true,
            ..motion.clone()
        });
        assert_eq!(link(&project, &scene, ContinuousShot, HardCut), Some(None));
        // Blend seams and transitions keep it; the Scene already continues.
        for after in [BlendSeam, Transition] {
            assert_eq!(link(&project, &scene, ContinuousShot, after), None);
        }
        assert_eq!(link(&project, &scene, BlendSeam, ContinuousShot), None);
        // Continuity the user chose stays through a Hard cut.
        for method in [
            ContinuityMethod::LastFrameFirstFrame,
            ContinuityMethod::MotionContext,
        ] {
            let mut own = scene.clone();
            own.continuity = Some(Continuity {
                method,
                ..motion.clone()
            });
            assert_eq!(link(&project, &own, ContinuousShot, HardCut), None);
            assert_eq!(link(&project, &own, HardCut, ContinuousShot), None);
        }
        // An imported clip has no Take to carry motion from.
        scene.continuity = None;
        for id in [&left, &scene.id] {
            let mut imported = project.clone();
            imported
                .creative
                .scene_kinds
                .insert(id.clone(), SceneKind::Imported);
            assert_eq!(link(&imported, &scene, HardCut, ContinuousShot), None);
        }
    }
    #[test]
    fn new_seam_uses_three_frames_but_reopened_edited_overlap_is_preserved() {
        use imagine_video_contract::assembly::BoundaryMode;
        let mut project = media_picker::tests::project();
        let left = project.scenes[0].id.clone();
        let right = SceneId("own-right-scene".into());
        let mut scene = project.scenes[0].clone();
        scene.id = right.clone();
        scene.start = Time { num: 2, den: 1 };
        scene.sections.clear();
        project.scenes.push(scene);
        let initial = boundary_for(&project, &left, &right);
        assert_eq!(initial.blend_frames, 3);
        assert_eq!(initial.mode, BoundaryMode::HardCut);
        assert!(project.assembly.boundaries.is_empty());
        // Six was the old default; neither it nor an authored value is migrated.
        for overlap in [6, 11] {
            let mut boundary = initial.clone();
            boundary.mode = BoundaryMode::BlendSeam;
            boundary.blend_frames = overlap;
            let revision = project.project_revision;
            imagine_video_contract::editing::apply(
                &mut project,
                EditRequest {
                    schema_version: 1,
                    expected_project_revision: revision,
                    commands: vec![Edit::SetAssemblyBoundary { boundary }],
                },
            )
            .unwrap();
            let mut reopened: VideoProject =
                serde_json::from_slice(&serde_json::to_vec(&project).unwrap()).unwrap();
            reopened.validate().unwrap();
            let mut edited = boundary_for(&reopened, &left, &right);
            assert_eq!(edited.blend_frames, overlap);
            edited.mode = BoundaryMode::ContinuousShot;
            let revision = reopened.project_revision;
            imagine_video_contract::editing::apply(
                &mut reopened,
                EditRequest {
                    schema_version: 1,
                    expected_project_revision: revision,
                    commands: vec![Edit::SetAssemblyBoundary { boundary: edited }],
                },
            )
            .unwrap();
            assert_eq!(boundary_for(&reopened, &left, &right).blend_frames, overlap);
        }
    }
    #[test]
    fn imported_clip_uses_video_frame_clock_and_preserves_authored_timing() {
        let clip = fitted_clip(
            Time { num: 3, den: 1 },
            AssetId("synthetic".into()),
            &json!({"media":{"frames":282,"fps":{"num":24,"den":1},"duration":{"num":12,"den":1}}}),
        )
        .unwrap();
        assert_eq!(clip.source_out, Some(Time { num: 47, den: 4 }));
        assert!((clip.speed - 47. / 12.).abs() < 1e-12);
    }
}
