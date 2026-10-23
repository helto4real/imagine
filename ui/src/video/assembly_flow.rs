//! Session-local Director AllFreshSeeds → Final continuation, using server-owned Runs.
use super::*;
use imagine_video_contract::runs::{Mode, Run, Status};

pub(super) struct Pending {
    project: ProjectId,
    epoch: u64,
    session: String,
    intent: Value,
    options: Options,
    key: Option<String>,
    run: Option<String>,
}
fn needs_generation(project: &VideoProject) -> bool {
    project
        .scenes
        .iter()
        .filter(|s| {
            project.scene_kind(&s.id) != imagine_video_contract::creative::SceneKind::Imported
        })
        .any(|s| s.accepted_take.is_none())
}
fn intent(project: &VideoProject, options: &Options) -> Result<Value, String> {
    let mut clean = project.clone();
    for scene in &mut clean.scenes {
        scene.accepted_take = None;
    }
    let inputs=clean.scenes.iter().map(|s|{
        if clean.scene_kind(&s.id)==imagine_video_contract::creative::SceneKind::Imported {
            let clip=clean.scene_clip(&s.id);
            Ok(json!({"id":s.id,"start":s.start,"duration":s.duration,"clip":clip,"asset":clip.and_then(|c|clean.assets.get(&c.asset)).map(|a|(&a.id,a.revision))}))
        }else{imagine_video_contract::planning::input_revision(&clean,&s.id).map(|r|json!({"id":s.id,"input":r})).map_err(|_|"Could not prepare the Final generation inputs.".to_owned())}
    }).collect::<Result<Vec<_>,_>>()?;
    let audio = if options.audio_mode == MixMode::Generated {
        Value::Null
    } else {
        let clips=project.audio_tracks.iter().flat_map(|t|&t.clips).filter(|c|c.enabled&&c.usage.final_mix).map(|c|{
            let mut clip=json!(c);
            for field in ["name","locked","lane","reference","usage"]{clip.as_object_mut().unwrap().remove(field);}
            json!({"clip":clip,"asset":c.asset.as_ref().and_then(|id|project.assets.get(id)).map(|a|(&a.id,a.revision))})
        }).collect::<Vec<_>>();
        json!({"settings":project.audio,"clips":clips})
    };
    Ok(
        json!({"scenes":inputs,"delivery":project.delivery,"boundaries":project.assembly.boundaries,"audio":audio}),
    )
}
impl ImagineApp {
    #[cfg(feature = "ui-preview")]
    pub(in crate::app::timeline) fn timeline_final_diagnostics(&self) -> Value {
        let pending = self.timeline.assembly.pending_final.as_ref();
        json!({"pending":pending.is_some(),"run":pending.and_then(|p|p.run.as_ref()),"intent_matches":pending.and_then(|p|self.timeline.document.as_ref().map(|d|intent(&d.project,&p.options).ok().as_ref()==Some(&p.intent))),"busy":self.timeline.busy,"assembly_busy":self.timeline.assembly.busy,"error":self.timeline.assembly.error,"failure":self.timeline.assembly.failure,"retry_preparation":self.timeline.assembly.retry_request().is_some(),"project_error":self.timeline.error,"admitting":self.timeline.admitting,"unconfirmed_job":self.timeline.pending.is_some()})
    }
    pub(in crate::app::timeline) fn timeline_final_run_reply(&mut self, run: &Run) {
        if let Some(pending) = &mut self.timeline.assembly.pending_final
            && pending.project == run.project_id
            && run.mode == Mode::AllFreshSeeds
            && pending.key.as_deref()
                == self
                    .timeline
                    .pending_run
                    .as_ref()
                    .map(|(_, key)| key.as_str())
        {
            pending.run = Some(run.id.clone());
        }
    }
    pub(in crate::app::timeline) fn timeline_final_run_failure(&mut self, message: &str) {
        if self.timeline.assembly.pending_final.take().is_some() {
            self.timeline.assembly.error =
                Some(format!("Final generation was not started: {message}"));
        }
    }
    pub(in crate::app) fn timeline_final_tick(&mut self, ctx: &egui::Context) {
        if self.timeline.assembly.pending_final.is_none() {
            return;
        }
        self.timeline_poll_runs(ctx);
        let pending = self.timeline.assembly.pending_final.as_ref().unwrap();
        if pending.epoch != self.timeline.epoch
            || self.timeline.session.as_ref() != Some(&pending.session)
            || self
                .timeline
                .document
                .as_ref()
                .is_none_or(|d| d.project.id != pending.project)
        {
            self.timeline.assembly.pending_final = None;
            self.timeline.assembly.error=Some("Final continuation ended because the project or server session changed. No export was resumed.".into());
            return;
        }
        let Some(run) = pending
            .run
            .as_ref()
            .and_then(|id| self.timeline.runs.iter().find(|r| &r.id == id))
        else {
            return;
        };
        if !run.status.terminal() {
            return;
        }
        if run.status != Status::Completed {
            let status = run.status;
            let reason = run.failure.clone().unwrap_or_default();
            self.timeline.assembly.pending_final = None;
            self.timeline.assembly.error = Some(format!(
                "Final generation ended ({status:?}) {reason}. Existing Takes are kept. Review the run, then Render to start a new fresh-seed run."
            ));
            return;
        }
        if self.timeline.busy || self.timeline.admitting || self.timeline.assembly.busy {
            return;
        }
        let project = &self.timeline.document.as_ref().unwrap().project;
        // Run completion can arrive before the refreshed project snapshot.
        if run.targets.iter().any(|t| {
            project
                .scene(&t.scene_id)
                .ok()
                .and_then(|s| s.accepted_take.as_ref().map(|v| &v.id))
                != t.completed_take_id.as_ref()
        }) {
            self.timeline_refresh(ctx);
            return;
        }
        if intent(project, &pending.options).ok().as_ref() != Some(&pending.intent) {
            self.timeline.assembly.pending_final = None;
            self.timeline.assembly.error=Some("Final inputs changed during generation. Completed Takes are kept; Render again to use the current Timeline.".into());
            return;
        }
        let pending = self.timeline.assembly.pending_final.take().unwrap();
        self.timeline.assembly.freeze_options = Some(pending.options);
        self.timeline.assembly.plan = None;
        self.timeline.assembly.render_after_freeze = true;
        self.assembly_call(ctx, false);
    }
    pub(super) fn timeline_final_start(&mut self, ctx: &egui::Context) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        let mut options = self.timeline.assembly.options.clone();
        match imagine_video_contract::assembly::output_filename(options.output_filename.as_deref())
        {
            Ok(name) => options.output_filename = name,
            Err(_) => {
                self.timeline.assembly.error=Some("Choose a filename, without a folder path, inside the selected output destination.".into());
                return;
            }
        }
        self.timeline.assembly.error = None;
        self.timeline.assembly.failure = None;
        self.timeline.assembly.freeze_request = None;
        if !needs_generation(&document.project) {
            self.timeline.assembly.freeze_options = Some(options);
            self.timeline.assembly.plan = None;
            self.timeline.assembly.render_after_freeze = true;
            self.assembly_call(ctx, false);
            return;
        }
        if self.timeline.has_active_run() || self.timeline.pending_run.is_some() {
            self.timeline.assembly.error=Some("A Timeline run is already active or unconfirmed. Retrieve it or use the existing Stop/Cancel controls first.".into());
            return;
        }
        let signature = match intent(&document.project, &options) {
            Ok(value) => value,
            Err(error) => {
                self.timeline.assembly.error = Some(error);
                return;
            }
        };
        let Some(session) = self.timeline.session.clone() else {
            return;
        };
        self.timeline.assembly.pending_final = Some(Pending {
            project: document.project.id.clone(),
            epoch: self.timeline.epoch,
            session,
            intent: signature,
            options,
            key: None,
            run: None,
        });
        self.timeline_start_run(ctx, Mode::AllFreshSeeds);
        if let Some(pending) = &mut self.timeline.assembly.pending_final {
            pending.key = self
                .timeline
                .pending_run
                .as_ref()
                .map(|(_, key)| key.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_final_retry_keeps_original_request_when_next_draft_changes() {
        let p = super::super::super::media_picker::tests::project();
        let request = FreezeRequest {
            schema_version: 1,
            project_id: p.id,
            expected_project_revision: 42,
            options: Options {
                output_filename: Some("original.mp4".into()),
                ..Default::default()
            },
        };
        let state = State {
            freeze_request: Some(request.clone()),
            options: Options {
                output_filename: Some("later draft.mp4".into()),
                ..Default::default()
            },
            failure: Some(FailureDetails {
                operation: "freeze",
                code: "network_request".into(),
                retryable: true,
                elapsed_ms: 1,
            }),
            ..Default::default()
        };
        assert_eq!(state.retry_request(), Some(request));
        assert_eq!(
            state.options.output_filename.as_deref(),
            Some("later draft.mp4")
        );
    }
    #[test]
    fn final_generation_needs_missing_generated_scenes_but_not_imported_scenes() {
        let mut p = super::super::super::media_picker::tests::project();
        assert!(needs_generation(&p));
        p.creative.scene_kinds.insert(
            p.scenes[0].id.clone(),
            imagine_video_contract::creative::SceneKind::Imported,
        );
        assert!(!needs_generation(&p));
    }
    #[test]
    fn final_intent_ignores_new_take_acceptance_but_keeps_generation_mix_and_geometry() {
        let mut p = super::super::super::media_picker::tests::project();
        let options = Options::default();
        let original = intent(&p, &options).unwrap();
        p.project_revision += 1;
        p.scenes[0].accepted_take = Some(AcceptedTake {
            id: TakeId("own-fresh-result".into()),
            model: Some(p.default_model.clone()),
            source_fingerprint: "own".into(),
        });
        assert_eq!(intent(&p, &options).unwrap(), original);
        p.scenes[0].description = "Changed after starting".into();
        assert_ne!(intent(&p, &options).unwrap(), original);
    }
    #[test]
    fn changing_project_clears_only_session_continuation_and_keeps_next_export_choices() {
        let p = super::super::super::media_picker::tests::project();
        let options = Options {
            output_filename: Some("Own next export.mp4".into()),
            ..Default::default()
        };
        let pending = Pending {
            project: p.id.clone(),
            epoch: 1,
            session: "own-session".into(),
            intent: intent(&p, &options).unwrap(),
            options: options.clone(),
            key: Some("own-key".into()),
            run: Some("own-run".into()),
        };
        let mut state = State {
            busy: true,
            render_after_freeze: true,
            pending_final: Some(pending),
            freeze_options: Some(options.clone()),
            options: options.clone(),
            freeze_request: Some(FreezeRequest {
                schema_version: 1,
                project_id: p.id,
                expected_project_revision: 1,
                options: options.clone(),
            }),
            failure: Some(FailureDetails {
                operation: "freeze",
                code: "network_request".into(),
                retryable: true,
                elapsed_ms: 1,
            }),
            ..Default::default()
        };
        state.project_changed();
        assert!(
            state.pending_final.is_none()
                && state.freeze_options.is_none()
                && !state.busy
                && !state.render_after_freeze
        );
        assert_eq!(state.options, options);
        assert!(state.error.is_some());
        assert!(state.freeze_request.is_none() && state.failure.is_none());
    }
}
