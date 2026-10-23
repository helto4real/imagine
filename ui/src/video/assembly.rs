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
    for delivery in document.raw["deliveries"].as_array().into_iter().flatten() {
        if delivery["plan"]["project_id"].as_str() != Some(plan.project_id.0.as_str()) {
            continue;
        }
        let Some(id) = delivery["asset_id"].as_str() else {
            continue;
        };
        let id = AssetId(id.into());
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
        let Some(runtime_id) = document.raw["runtime_assets"][&asset.0]
            .as_str()
            .map(str::to_owned)
        else {
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
                let r =
                    ui.add(egui::DragValue::new(&mut boundary.blend_frames).range(1..=u32::MAX));
                control(ui, "assembly-overlap", &r);
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
                let response = ui.add(egui::DragValue::new(tail));
                control(ui, "boundary-tail-frames", &response);
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
                let response = ui.add(
                    egui::TextEdit::multiline(&mut boundary.transition_prompt)
                        .desired_width(f32::INFINITY)
                        .desired_rows(3),
                );
                control(ui, "boundary-transition-prompt", &response);
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
                let r = ui.add(
                    egui::DragValue::new(&mut start)
                        .speed(0.01)
                        .range(0. ..=f64::MAX),
                );
                control(ui, "assembly-source-in", &r);
            });
            let mut trimmed = clip.source_out.is_some();
            let r = ui.checkbox(&mut trimmed, "Source out");
            control(ui, "assembly-source-out-enabled", &r);
            if trimmed {
                let r = ui.add(
                    egui::DragValue::new(&mut end)
                        .speed(0.01)
                        .range(0. ..=f64::MAX),
                );
                control(ui, "assembly-source-out", &r);
            }
            clip.source_in = imagine_video_contract::assembly::time(start);
            clip.source_out = trimmed.then(|| imagine_video_contract::assembly::time(end));
            ui.horizontal(|ui| {
                ui.label("Speed");
                let r = ui.add(
                    egui::DragValue::new(&mut clip.speed)
                        .speed(0.05)
                        .range(f64::MIN_POSITIVE..=f64::MAX)
                        .suffix("×"),
                );
                control(ui, "assembly-speed", &r);
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
    pub(super) fn timeline_assembly_window(&mut self, ctx: &egui::Context) {
        if !self.timeline.assembly.open {
            return;
        }
        let mut open = true;
        let mut capture = false;
        let mut submit = false;
        let mut preview = None;
        let mut close = false;
        let mut render_current = false;
        let mut retry = false;
        let document = self.timeline.document.clone();
        let busy = self.timeline.assembly.busy;
        let before = self.timeline.assembly.options.clone();
        egui::Window::new("Final video").id(egui::Id::new("video-final-montage")).open(&mut open).default_width(590_f32.min((ctx.content_rect().width()-24.).max(240.))).max_width((ctx.content_rect().width()-24.).max(240.)).max_height((ctx.content_rect().height()-70.).max(160.)).vscroll(true).resizable(true).show(ctx,|ui|{
            ui.set_max_width((ctx.content_rect().width()-48.).max(216.));
            #[cfg(feature="ui-preview")] {
                let r=ui.interact(ui.clip_rect(),ui.id().with("qa-assembly-scroll"),egui::Sense::hover());
                control(ui,"assembly-scroll",&r);
            }
            let hidden=!helto::reveal_in_viewport(ui,self.prefs.privacy_mode);
            helto::private_scope(ui,hidden,|ui|{
            ui.horizontal_wrapped(|ui|{
                let r=helto::icon_button(ui,crate::icons::Icon::Back,"Back to Timeline",false);control(ui,"assembly-close",&r);close=r.clicked();
                let r=helto::action(ui,crate::icons::Icon::Generate,"Render current Timeline",true,if document.is_none(){Some("Create or open a project first.")}else{(busy||self.timeline.busy||self.timeline.admitting||self.timeline.pending.is_some()||self.timeline.assembly.pending_final.is_some()).then_some("Wait for the current operation.")});control(ui,"assembly-render-current",&r);render_current=r.clicked();
            });
            if self.timeline.assembly.retry_request().is_some() {
                let r=helto::icon_button(ui,crate::icons::Icon::Retry,"Retry Final preparation",false);control(ui,"assembly-retry-preparation",&r);retry=r.clicked();
                ui.weak("Retry uses the original Timeline revision and export choices. Render current Timeline uses your current choices.");
            }
            if let Some(error)=&self.timeline.assembly.error{ui.colored_label(helto::ERROR,error);}
            ui.label("Montage in scene order · MP4 / H.264 · CRF 19 · AAC stereo / 44.1 kHz");
            ui.weak("A frozen plan keeps its selected media even if the project changes. Deliveries appear below and in the queue.");
            ui.weak("An unfinished Timeline runs AllFreshSeeds first and accepts its results through the Run rules before export. Stop/Cancel remains in the queue.");
            if self.timeline.assembly.pending_final.is_some(){ui.strong("Final pending · generating all scenes with fresh seeds…");}
            ui.horizontal(|ui|{
                let label=ui.label("Filename");
                let mut name=self.timeline.assembly.options.output_filename.clone().unwrap_or_default();
                let r=ui.add(egui::TextEdit::singleline(&mut name).hint_text("Automatic numbered filename · MP4").desired_width(f32::INFINITY)).labelled_by(label.id);
                control(ui,"assembly-output-filename",&r);
                if r.changed(){self.timeline.assembly.options.output_filename=(!name.is_empty()).then_some(name);}
            });
            ui.add_enabled_ui(!busy,|ui|{
                let options=&mut self.timeline.assembly.options;
                ui.horizontal(|ui|{
                    ui.label("Audio");
                    for (mode,label,name) in [(MixMode::Generated,"Generated","assembly-audio-generated"),(MixMode::Timeline,"Timeline","assembly-audio-timeline"),(MixMode::Combined,"Combined","assembly-audio-combined")]{let r=ui.selectable_value(&mut options.audio_mode,mode,label);control(ui,name,&r);}
                });
                ui.horizontal(|ui|{
                    ui.label("Generated gain");ui.add(egui::DragValue::new(&mut options.generated_gain).speed(0.05).range(0. ..=f64::MAX));
                    ui.label("Timeline gain");ui.add(egui::DragValue::new(&mut options.timeline_gain).speed(0.05).range(0. ..=f64::MAX));
                });
                let r=ui.checkbox(&mut options.normalize_mix,"Normalize final mix · Director RMS −18 dB / peak −1 dB / max gain 2×");control(ui,"assembly-normalize",&r);
                ui.horizontal(|ui|{let r=ui.checkbox(&mut options.duck_generated,"Duck generated audio under timeline audio");control(ui,"assembly-duck",&r);ui.add(egui::DragValue::new(&mut options.ducking_strength).speed(0.05).range(0. ..=1.));});
                ui.checkbox(&mut options.continuous_tone_match,"Match luminance at Continuous Shot boundaries");
                ui.horizontal(|ui|{ui.label("Color depth");ui.selectable_value(&mut options.bit_depth,8,"8-bit · standard");ui.selectable_value(&mut options.bit_depth,10,"10-bit");});
            });
            ui.separator();
            ui.weak("Render freezes the current montage internally. Frozen-plan inspection below is optional.");
            let response=helto::action(ui,crate::icons::Icon::Lock,"Freeze current montage plan",false,if document.is_none(){Some("Create or open a project first.")}else{busy.then_some("Wait for the current operation.")});control(ui,"assembly-freeze",&response);capture=response.clicked();
            if busy{ui.spinner();}
            if let Some(plan)=&self.timeline.assembly.plan{
                let fps=imagine_video_contract::assembly::seconds(plan.delivery.fps);
                ui.label(format!("{} × {} · {:.6} fps · {} frames · {:.6} s",plan.delivery.width,plan.delivery.height,fps,plan.frames,plan.frames as f64/fps));
                ui.weak(format!("Project revision {} · {} clips · {} timeline audio segments · {} samples/channel",plan.project_revision,plan.clips.len(),plan.audio.len(),plan.samples));
                if document.as_ref().is_some_and(|d|earlier_snapshot(plan,self.timeline.assembly.plan_project.as_deref(),d)){let response=ui.colored_label(helto::ACCENT,"This is an earlier project snapshot. Freeze again to include current edits.");control(ui,"assembly-snapshot-warning",&response);}
                egui::ScrollArea::vertical().id_salt("assembly-plan").max_height(140.).show(ui,|ui|{
                    for (i,clip) in plan.clips.iter().enumerate(){ui.label(format!("Scene {} · {:?} · {} frames / {} overlap · {}",i+1,clip.boundary,clip.frames,clip.overlap,if clip.take_id.is_some(){"selected Take"}else{"imported video"}));}
                    for warning in &plan.warnings{ui.colored_label(helto::ACCENT,warning);}
                });
                let r=helto::action(ui,crate::icons::Icon::Generate,"Render frozen plan",true,(busy||self.timeline.admitting||self.timeline.pending.is_some()).then_some("Wait for the current operation."));control(ui,"assembly-render",&r);submit=r.clicked();
            }
            if let Some(phase)=self.timeline.assembly.progress["phase"].as_str(){let response=ui.weak(assembly_phase_label(phase));control(ui,"assembly-progress",&response);}
            if self.timeline.assembly.progress["rendered_result_available"]==true{ui.weak("The encoded export is retained in this session. Retry export in the queue to save it again.");}
            ui.separator();ui.weak("Saved deliveries");
            if let Some(d)=&document{
                for (i,delivery) in d.raw["deliveries"].as_array().into_iter().flatten().enumerate(){
                    let r=ui.button(format!("Delivery {} · revision {}",i+1,delivery["plan"]["project_revision"]));control(ui,&format!("assembly-delivery-{i}"),&r);
                    if r.clicked(){preview=delivery["asset_id"].as_str().and_then(|id|d.raw["runtime_assets"][id].as_str()).map(str::to_owned);}
                }
            }
            ui.weak("Exports are separate deliveries. They do not replace Accepted Takes.");
            });
        });
        self.timeline.assembly.open = open && !close;
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
            .any(|j| j.request["kind"] == "video_assembly" && !j.terminal());
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
// A new seam uses Director's three-frame overlap; persisted boundaries retain
// their explicit overlap when reopened or when only their mode is edited.
fn boundary_for(
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
        (
            Snapshot {
                project,
                raw: json!({"deliveries":[]}),
            },
            plan,
        )
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
        document.raw["deliveries"]
            .as_array_mut()
            .unwrap()
            .push(json!({"asset_id":id,"plan":plan}));
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
        delivered.raw["deliveries"] = json!([]);
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
