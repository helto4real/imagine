//! Native projection of server-owned runs; polling never advances generation in the UI.
use super::*;
use imagine_video_contract::runs::{Mode, Run, Status};
#[derive(Clone, Copy)]
pub(in crate::app) enum Reply {
    List,
    Submitted,
    Stopped,
}
impl Timeline {
    pub(super) fn has_active_run(&self) -> bool {
        self.document.as_ref().is_some_and(|d| {
            self.runs
                .iter()
                .any(|r| r.project_id == d.project.id && !r.status.terminal())
        })
    }
}
impl ImagineApp {
    pub(super) fn timeline_poll_runs(&mut self, ctx: &egui::Context) {
        if self.timeline.needs_reopen {
            return;
        }
        ctx.request_repaint_after(Duration::from_millis(500));
        if self.timeline.runs_loading
            || self
                .timeline
                .run_poll_at
                .is_some_and(|at| at.elapsed() < Duration::from_millis(500))
        {
            return;
        }
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.runs_loading = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                false,
                "/v1/video/runs",
                None,
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Runs {
                    session,
                    kind: Reply::List,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_start_run(&mut self, ctx: &egui::Context, mode: Mode) {
        let Some(document) = &self.timeline.document else {
            return;
        };
        if self.timeline.admitting || self.timeline.has_active_run() {
            return;
        }
        if self.timeline.pending_run.is_none() {
            self.timeline.pending_run = Some((
                json!({"schema_version":1,"project_id":document.project.id,"expected_project_revision":document.project.project_revision,"mode":mode,"selected_scene":self.timeline.primary_scene(&document.project)}),
                uuid::Uuid::new_v4().to_string(),
            ));
        }
        self.timeline_submit_run(ctx);
    }
    fn timeline_submit_run(&mut self, ctx: &egui::Context) {
        if self.timeline.admitting {
            return;
        }
        let (Some(client), Some(session), Some((request, key))) = (
            self.client.clone(),
            self.timeline.session.clone(),
            self.timeline.pending_run.clone(),
        ) else {
            return;
        };
        self.timeline.admitting = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_start_run(&session, &request, &key));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Runs {
                    session,
                    kind: Reply::Submitted,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_run_reply(
        &mut self,
        ctx: &egui::Context,
        session: String,
        kind: Reply,
        result: Result<Value, client::TimelineFailure>,
    ) {
        if self.timeline.session.as_deref() != Some(&session) {
            return;
        }
        match kind {
            Reply::List => {
                self.timeline.runs_loading = false;
                self.timeline.run_poll_at = Some(std::time::Instant::now());
            }
            Reply::Submitted => self.timeline.admitting = false,
            Reply::Stopped => self.timeline.run_acting = false,
        }
        let value = match result {
            Ok(v) => v,
            Err(e) => {
                if e.code == "session_changed" {
                    self.timeline.session_ended();
                }
                if matches!(kind, Reply::Submitted) && !e.retryable {
                    self.timeline_final_run_failure(&e.message);
                    self.timeline.pending_run = None;
                }
                if !matches!(kind, Reply::List) || !self.timeline.runs.is_empty() {
                    self.timeline.error = Some(e.message);
                }
                return;
            }
        };
        let parsed: Result<Vec<Run>, _> = if matches!(kind, Reply::List) {
            serde_json::from_value(value["runs"].clone())
        } else {
            serde_json::from_value(value).map(|run| vec![run])
        };
        let Ok(runs) = parsed else {
            self.timeline.error =
                Some("Could not read timeline runs. Reconnect to the server.".into());
            return;
        };
        if matches!(kind, Reply::Submitted) {
            for run in &runs {
                self.timeline_final_run_reply(run);
            }
            self.timeline.pending_run = None;
            self.timeline.error = None;
        }
        let mut refresh = false;
        let mut jobs = vec![];
        let mut completed_jobs = vec![];
        for run in runs {
            let changed = self
                .timeline
                .runs
                .iter()
                .find(|r| r.id == run.id)
                .is_none_or(|old| old.completed() != run.completed() || old.status != run.status);
            refresh |= changed
                && self
                    .timeline
                    .document
                    .as_ref()
                    .is_some_and(|d| d.project.id == run.project_id);
            for target in &run.targets {
                if let Some(id) = &target.job_id {
                    if !self.timeline.jobs.iter().any(|j| j.id == *id)
                        && let Some(job) = jobs::Job::new(
                            id.clone(),
                            json!({"kind":"video_scene","run_id":run.id,"schema_version":1,"project_id":run.project_id,"scene_id":target.scene_id,"seed":target.seed}),
                        )
                    {
                        self.timeline.jobs.push(job);
                    }
                    jobs.push(id.clone());
                    if target.completed_take_id.is_some()
                        && let Some(job) = self.timeline.jobs.iter_mut().find(|job| job.id == *id)
                    {
                        // The server reports completed_take_id only after job commit and review.
                        if !job.terminal() {
                            job.observation["status"] = json!("succeeded");
                            job.monitoring = false;
                            job.acting = false;
                        }
                        if job.observation["output"].is_null() {
                            completed_jobs.push(id.clone());
                        }
                    }
                }
            }
            self.timeline.runs.retain(|r| r.id != run.id);
            self.timeline.runs.push(run);
        }
        for id in jobs {
            self.timeline_poll(ctx, id);
        }
        for id in completed_jobs {
            let (Some(client), Some(session)) =
                (self.client.clone(), self.timeline.session.clone())
            else {
                break;
            };
            let tx = self.tx.clone();
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                if let Ok(job) = runtime().block_on(client.timeline_request(
                    &session,
                    false,
                    &format!("/v1/jobs/{id}"),
                    None,
                )) {
                    send(&tx, &ctx, Message::Timeline(Event::Job { session, job }));
                }
            });
        }
        if refresh {
            if self.timeline.busy {
                self.timeline.refresh_after_busy = true;
            } else {
                self.timeline_refresh(ctx);
            }
        }
    }
    fn timeline_stop_run(&mut self, ctx: &egui::Context, id: String, cancel: bool) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        self.timeline.run_acting = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.timeline_request(
                &session,
                true,
                &format!("/v1/video/runs/{id}/stop"),
                Some(&json!({"schema_version":1,"cancel_current":cancel})),
            ));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Runs {
                    session,
                    kind: Reply::Stopped,
                    result,
                }),
            );
        });
    }
    pub(super) fn timeline_run_rows(&mut self, ui: &mut egui::Ui) {
        if self.timeline.pending_run.is_some() {
            let r = ui.add_enabled(
                !self.timeline.admitting,
                egui::Button::new("Retrieve run request"),
            );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, "run-retrieve".into(), &r);
            if r.clicked() {
                self.timeline_submit_run(ui.ctx());
            }
        }
        let mut stop = None;
        for (index, run) in self
            .timeline
            .runs
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.status.terminal())
            .chain(
                self.timeline
                    .runs
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.status.terminal())
                    .rev()
                    .take(3),
            )
        {
            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "Timeline run · {} / {}",
                    run.completed(),
                    run.targets.len()
                ));
                ui.weak(match run.status {
                    Status::Running => "Running",
                    Status::Stopping => "Stopping after current",
                    Status::Cancelling => "Cancelling",
                    Status::Completed => "Complete",
                    Status::Stopped => "Stopped",
                    Status::Cancelled => "Cancelled",
                    Status::Failed => "Needs review",
                    Status::Interrupted => "Interrupted",
                });
                if !run.status.terminal() {
                    for (cancel, label, name) in [
                        (false, "Stop after current", "run-stop"),
                        (true, "Cancel current", "run-cancel"),
                    ] {
                        let response =
                            ui.add_enabled(!self.timeline.run_acting, egui::Button::new(label));
                        #[cfg(feature = "ui-preview")]
                        super::qa::control(ui, format!("{name}-{index}"), &response);
                        #[cfg(feature = "ui-preview")]
                        if self
                            .timeline
                            .document
                            .as_ref()
                            .is_some_and(|d| d.project.id == run.project_id)
                        {
                            super::qa::control(ui, format!("{name}-current"), &response);
                        }
                        let _ = name;
                        if response.clicked() {
                            stop = Some((run.id.clone(), cancel));
                        }
                    }
                }
            });
            if let Some(code) = &run.failure {
                ui.colored_label(
                    helto::ERROR,
                    match code.as_str() {
                        "run_inputs_changed" => {
                            "Scene inputs changed. Review before starting another run."
                        }
                        "run_take_selection_changed" => {
                            "Take selection changed. Your choice was kept."
                        }
                        "all_scenes_require_locked_seeds" => {
                            "Every scene needs an accepted or previously accepted Take."
                        }
                        _ => "Run stopped. Review its scene job before retrying.",
                    },
                );
            }
            ui.separator();
            let _ = index;
        }
        if let Some((id, cancel)) = stop {
            self.timeline_stop_run(ui.ctx(), id, cancel);
        }
    }
}
