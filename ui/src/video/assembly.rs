//! Review a frozen montage, then submit it to the shared queue as a CPU delivery.
use super::*;
use imagine_video_contract::assembly::{MixMode, Options, Plan};
#[derive(Default)]
pub(super) struct State {
    pub open: bool,
    pub(super) busy: bool,
    options: Options,
    pub(super) plan: Option<Plan>,
    pub(super) progress: Value,
    pub(super) error: Option<String>,
    poll_at: Option<std::time::Instant>,
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
        use imagine_video_contract::assembly::{Boundary, BoundaryMode};
        let Some(index) = d.project.scenes.iter().position(|s| s.id == scene.id) else {
            return;
        };
        let Some(left) = index.checked_sub(1).and_then(|i| d.project.scenes.get(i)) else {
            return;
        };
        let mut boundary = d
            .project
            .assembly
            .boundaries
            .iter()
            .find(|b| b.left == left.id && b.right == scene.id)
            .cloned()
            .unwrap_or(Boundary {
                left: left.id.clone(),
                right: scene.id.clone(),
                mode: BoundaryMode::HardCut,
                blend_frames: 6,
            });
        let before = boundary.clone();
        ui.label("Final montage boundary");
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
        if boundary.mode == BoundaryMode::Transition {
            ui.weak("A generated bridge Take is required. This runtime cannot assemble that boundary yet.");
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
        let Some(mut clip) = d.project.assembly.clips.get(&scene.id).cloned() else {
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
            let r = ui.button("Use generated Take instead");
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
        let (Some(client), Some(session), Some(document)) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.document.as_ref(),
        ) else {
            return;
        };
        let (path, body) = if progress {
            let Some(plan) = &self.timeline.assembly.plan else {
                return;
            };
            (format!("/v1/video/assemblies/{}", plan.id), None)
        } else {
            (
                format!("/v1/video/projects/{}/assembly", document.project.id.0),
                Some(
                    json!({"schema_version":1,"project_id":document.project.id,"expected_project_revision":document.project.project_revision,"options":self.timeline.assembly.options}),
                ),
            )
        };
        self.timeline.assembly.busy = true;
        let epoch = self.timeline.epoch;
        let ctx = ctx.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
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
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_assembly_reply(
        &mut self,
        session: String,
        epoch: u64,
        progress: bool,
        result: Result<Value, client::TimelineFailure>,
    ) {
        self.timeline.assembly.busy = false;
        if self.timeline.session.as_deref() != Some(&session) || self.timeline.epoch != epoch {
            return;
        }
        match result {
            Ok(value) => {
                self.timeline.assembly.error = None;
                if progress {
                    self.timeline.assembly.progress = value["progress"].clone();
                    self.timeline.assembly.progress["rendered_result_available"] =
                        value["rendered_result_available"].clone();
                } else {
                    match serde_json::from_value(value) {
                        Ok(plan) => {
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
            Err(error) => self.timeline.assembly.error = Some(error.message),
        }
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
        let document = self.timeline.document.clone();
        let busy = self.timeline.assembly.busy;
        let before = self.timeline.assembly.options.clone();
        egui::Window::new("Final video").id(egui::Id::new("video-final-montage")).open(&mut open).default_width(590.).max_height((ctx.content_rect().height()-70.).max(160.)).vscroll(true).resizable(true).show(ctx,|ui|{
            let r=ui.button("Back to Timeline");control(ui,"assembly-close",&r);close=r.clicked();
            ui.label("Montage in scene order · MP4 / H.264 · CRF 19 · AAC stereo / 44.1 kHz");
            ui.weak("A frozen plan keeps its selected media even if the project changes. Deliveries appear below and in the queue.");
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
            let response=ui.add_enabled(!busy&&document.is_some(),egui::Button::new("Freeze current montage plan"));control(ui,"assembly-freeze",&response);capture=response.clicked();
            if busy{ui.spinner();}
            if let Some(plan)=&self.timeline.assembly.plan{
                let fps=imagine_video_contract::assembly::seconds(plan.delivery.fps);
                ui.label(format!("{} × {} · {:.6} fps · {} frames · {:.6} s",plan.delivery.width,plan.delivery.height,fps,plan.frames,plan.frames as f64/fps));
                ui.weak(format!("Project revision {} · {} clips · {} timeline audio segments · {} samples/channel",plan.project_revision,plan.clips.len(),plan.audio.len(),plan.samples));
                if document.as_ref().is_some_and(|d|d.project.project_revision!=plan.project_revision||d.project.id!=plan.project_id){ui.colored_label(helto::ACCENT,"This is an earlier project snapshot. Freeze again to include current edits.");}
                egui::ScrollArea::vertical().id_salt("assembly-plan").max_height(140.).show(ui,|ui|{
                    for (i,clip) in plan.clips.iter().enumerate(){ui.label(format!("Scene {} · {:?} · {} frames / {} overlap · {}",i+1,clip.boundary,clip.frames,clip.overlap,if clip.take_id.is_some(){"selected Take"}else{"imported video"}));}
                    for warning in &plan.warnings{ui.colored_label(helto::ACCENT,warning);}
                });
                let r=ui.add_enabled(!busy&&!self.timeline.admitting&&self.timeline.pending.is_none(),egui::Button::new("Render frozen plan"));control(ui,"assembly-render",&r);submit=r.clicked();
            }
            if let Some(phase)=self.timeline.assembly.progress["phase"].as_str(){ui.weak(match phase{"queued"=>"Waiting in queue…","mixing_generated_audio"|"mixing_timeline_audio"=>"Mixing audio…","decoding_clip"=>"Reading clips…","encoding_video"=>"Encoding video…","encoded"=>"Encoded","cancelled"=>"Cancelled","failed"=>"Failed",_=>"Preparing…"});}
            if self.timeline.assembly.progress["rendered_result_available"]==true{ui.weak("The encoded export is retained in this session. Retry export in the queue to save it again.");}
            if let Some(error)=&self.timeline.assembly.error{ui.colored_label(helto::ERROR,error);}
            ui.separator();ui.weak("Saved deliveries");
            if let Some(d)=&document{
                for (i,delivery) in d.raw["deliveries"].as_array().into_iter().flatten().enumerate(){
                    let r=ui.button(format!("Delivery {} · revision {}",i+1,delivery["plan"]["project_revision"]));control(ui,&format!("assembly-delivery-{i}"),&r);
                    if r.clicked(){preview=delivery["asset_id"].as_str().and_then(|id|d.raw["runtime_assets"][id].as_str()).map(str::to_owned);}
                }
            }
            ui.weak("Exports are separate deliveries. They do not replace Accepted Takes.");
        });
        self.timeline.assembly.open = open && !close;
        if before != self.timeline.assembly.options {
            self.timeline.assembly.plan = None;
        }
        if capture {
            self.assembly_call(ctx, false);
        }
        if submit {
            self.timeline_assembly_submit(ctx);
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
