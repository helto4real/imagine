//! Projection of shared FIFO jobs. A lost observation never creates a new job.
use super::*;

pub(super) struct Job {
    pub id: String,
    pub request: Value,
    pub observation: Value,
    pub monitoring: bool,
    pub acting: bool,
    pub error: Option<String>,
}
impl Job {
    pub(super) fn failure_message(&self) -> Option<String> {
        self.error
            .clone()
            .or_else(|| {
                self.observation["error"]["message"]
                    .as_str()
                    .map(str::to_owned)
            })
            .or_else(|| {
                self.observation["error"]["code"].as_str().map(|code| {
                    if code == "generation_failed" {
                        "Video generation failed. Retry the scene.".into()
                    } else {
                        client::error_message(code)
                    }
                })
            })
    }
    pub fn new(id: String, request: Value) -> Option<Self> {
        if id.is_empty()
            || id.len() > 128
            || !id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            return None;
        }
        Some(Self {
            observation: json!({"id":id,"status":"queued"}),
            id,
            request,
            monitoring: false,
            acting: false,
            error: None,
        })
    }
    pub fn terminal(&self) -> bool {
        matches!(
            self.observation["status"].as_str(),
            Some("succeeded" | "failed" | "cancelled" | "interrupted")
        )
    }
    pub fn queued(&self) -> bool {
        self.observation["status"] == "queued"
    }
    pub fn take_id(&self) -> Option<&str> {
        let take = &self.observation["output"]["take"];
        (self.observation["status"] == "succeeded"
            && take["project_id"] == self.request["project_id"])
            .then(|| take["id"].as_str())
            .flatten()
    }
    pub fn matches(&self, project: &ProjectId, scene: &SceneId) -> bool {
        self.request["project_id"].as_str() == Some(&project.0)
            && self.request["scene_id"].as_str() == Some(&scene.0)
    }
    pub fn retryable(&self, project: &VideoProject) -> bool {
        self.terminal()
            && self.observation["status"] != "succeeded"
            && self.request["project_id"].as_str() == Some(&project.id.0)
            && (self.request["kind"] == "video_assembly"
                || self.request["expected_project_revision"] == json!(project.project_revision))
    }
    pub fn observe(&mut self, value: Value) -> bool {
        if value["id"].as_str() != Some(&self.id)
            || !matches!(
                value["status"].as_str(),
                Some(
                    "queued"
                        | "running"
                        | "cancelling"
                        | "succeeded"
                        | "failed"
                        | "cancelled"
                        | "interrupted"
                )
            )
        {
            self.monitoring = false;
            self.error = Some("Could not read job status. Reconnect to this job.".into());
            return false;
        }
        if self.terminal() && value["status"] != self.observation["status"] {
            return false;
        }
        self.observation = value;
        self.error = None;
        if self.terminal() {
            self.monitoring = false;
            self.acting = false;
        }
        true
    }
}

impl ImagineApp {
    pub(in crate::app) fn timeline_has_job_errors(&self) -> bool {
        if self.timeline.pending.is_some() {
            return true;
        }
        self.timeline
            .jobs
            .iter()
            .any(|j| j.error.is_some() || j.terminal() && j.observation["status"] != "succeeded")
    }
    /// "Scene N" for a scene job in the open project, from its position on the Timeline.
    fn job_scene_label(&self, job: &Job) -> String {
        self.timeline
            .document
            .as_ref()
            .filter(|d| job.request["project_id"] == json!(d.project.id))
            .and_then(|d| {
                d.project
                    .scenes
                    .iter()
                    .position(|s| job.request["scene_id"] == json!(s.id))
            })
            .map_or_else(|| "Video scene".into(), |n| format!("Scene {}", n + 1))
    }
    pub(in crate::app) fn timeline_report_error(&mut self, message: String) {
        self.timeline.error = Some(message);
    }
    /// The video service was checked and no session could be opened.
    pub(in crate::app) fn timeline_unreachable(&self) -> bool {
        self.timeline.session_checked && self.timeline.session.is_none() && !self.timeline.busy
    }
    /// Timeline work shown in the queue dock; growth means new work was submitted.
    pub(in crate::app) fn timeline_queue_work(&self) -> usize {
        self.timeline.jobs.len()
            + self.timeline.runs.len()
            + usize::from(self.timeline.pending.is_some())
            + usize::from(self.timeline.pending_run.is_some())
            + usize::from(self.timeline.admitting)
    }
    pub(in crate::app) fn timeline_queue_counts(&self) -> (usize, usize) {
        let pending = self
            .timeline
            .jobs
            .iter()
            .filter(|j| !j.terminal() && j.queued())
            .count();
        let active = self
            .timeline
            .jobs
            .iter()
            .filter(|j| !j.terminal() && !j.queued())
            .count();
        (pending + usize::from(self.timeline.admitting), active)
    }
    pub(in crate::app) fn timeline_queue_rows(&mut self, ui: &mut egui::Ui, revealed: bool) {
        self.timeline_run_rows(ui);
        let mut cancel = None;
        let mut reconnect = None;
        let mut retry = None;
        if self.timeline.pending.is_some() {
            ui.horizontal(|ui| {
                ui.label(if self.timeline.admitting {
                    "Submitting video scene…"
                } else {
                    "Submission unconfirmed"
                });
                let response = helto::action(
                    ui,
                    crate::icons::Icon::Download,
                    "Retrieve submission",
                    false,
                    self.timeline.admitting.then_some("Submitting now."),
                );
                #[cfg(feature = "ui-preview")]
                super::qa::control(ui, "submission-reconnect".into(), &response);
                if response.clicked() {
                    self.timeline_submit_pending(ui.ctx());
                }
            });
        }
        for (index, job) in self.timeline.jobs.iter().enumerate() {
            if job.observation["status"] == "succeeded" {
                continue;
            }
            let _ = index; // Names QA controls in preview builds.
            ui.push_id(("scene-job",&job.id),|ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(if job.request["kind"]=="video_assembly"{"Final video".into()}else{self.job_scene_label(job)});
                    let label=if job.terminal() {match job.observation["status"].as_str() {Some("cancelled")=>"Cancelled",Some("interrupted")=>"Interrupted",_=>"Failed"}.to_owned()} else if !job.monitoring {"Connection lost".into()} else if job.request["kind"]=="video_assembly"{match job.observation["status"].as_str(){Some("queued")=>"Queued",Some("cancelling")=>"Cancelling",_=>"Exporting"}.into()}else {let Progress::Status{label,..}=client::job_progress(&job.observation) else {unreachable!()};label};
                    ui.weak(label);
                    if !job.terminal() {
                        if !job.monitoring {
                            let response=helto::icon_button(ui,crate::icons::Icon::Link,"Reconnect",false);
                            #[cfg(feature="ui-preview")]
                            super::qa::control(ui,format!("job-reconnect-{index}"),&response);
                            if response.clicked(){reconnect=Some(job.id.clone());}
                        }
                        let response=helto::action(ui,crate::icons::Icon::Stop,"Cancel",false,(job.acting || job.observation["status"]=="cancelling").then_some("Cancelling."));
                        #[cfg(feature="ui-preview")]
                        super::qa::control(ui,format!("job-cancel-{index}"),&response);
                        if response.clicked(){cancel=Some(job.id.clone());}
                    } else {
                        let can_retry=self.timeline.document.as_ref().is_some_and(|d| job.retryable(&d.project)) && !self.timeline.admitting && self.timeline.pending.is_none();
                        let response=helto::action(ui,crate::icons::Icon::Retry,if job.request["kind"]=="video_assembly"{"Retry export"}else{"Retry scene"},false,(!can_retry).then_some("Open the source project to retry. A montage retry uses its frozen media.")).on_hover_text("A completed montage is saved again without rendering. Otherwise its frozen sources are rendered again.");
                        #[cfg(feature="ui-preview")]
                        super::qa::control(ui,format!("job-retry-{index}"),&response);
                        if response.clicked() {retry=Some(job.id.clone());}
                    }
                });
                if let Progress::Status{sampling:Some(p),..}=client::job_progress(&job.observation) {
                    ui.add(egui::ProgressBar::new(p.completed as f32/p.total.max(1) as f32).text(format!("{} / {} steps",p.completed,p.total)).fill(helto::ACCENT));
                }
                let error=job.failure_message();
                if let Some(error)=error {
                    ui.colored_label(helto::ERROR,if revealed {error.as_str()} else {"Hidden error"});
                }
                ui.separator();
            });
        }
        if let Some(id) = cancel {
            self.timeline_cancel_job(ui.ctx(), id);
        }
        if let Some(id) = reconnect {
            self.timeline_poll(ui.ctx(), id);
        }
        if let Some(id) = retry {
            self.timeline_retry_job(ui.ctx(), &id);
        }
    }
    pub(in crate::app) fn timeline_result_count(&self) -> usize {
        self.timeline
            .jobs
            .iter()
            .filter(|j| {
                j.take_id().is_some()
                    || j.observation["status"] == "succeeded"
                        && j.observation["output"]["delivery"].is_object()
            })
            .count()
    }
    pub(in crate::app) fn timeline_result_rows(&mut self, ui: &mut egui::Ui, revealed: bool) {
        let mut view = None;
        let mut delivery = None;
        for (index, job) in self
            .timeline
            .jobs
            .iter()
            .enumerate()
            .filter(|(_, j)| j.observation["status"] == "succeeded")
        {
            let Some(take) = job.take_id() else {
                if job.observation["output"]["delivery"].is_object() {
                    let r = ui.button(if revealed {
                        format!("Final video {}", index + 1)
                    } else {
                        "Hidden result".into()
                    });
                    #[cfg(feature = "ui-preview")]
                    super::qa::control(ui, format!("job-result-{index}"), &r);
                    if r.clicked() {
                        delivery = job.observation["output"]["assets"][0]["id"]
                            .as_str()
                            .map(str::to_owned);
                    }
                }
                continue;
            };
            let current = self
                .timeline
                .document
                .as_ref()
                .is_some_and(|d| job.request["project_id"] == json!(d.project.id));
            let enabled = current && !self.timeline.busy && self.timeline.draft_edits().is_empty();
            let label = if revealed {
                format!("{} result", self.job_scene_label(job))
            } else {
                "Hidden result".into()
            };
            let response = ui
                .add_enabled(enabled, egui::Button::new(label))
                .on_disabled_hover_text(
                    "Open the result project and save pending prompt edits first.",
                );
            #[cfg(feature = "ui-preview")]
            super::qa::control(ui, format!("job-result-{index}"), &response);
            if response.clicked() {
                view = Some((
                    job.request["scene_id"].as_str().unwrap_or("").to_owned(),
                    take.to_owned(),
                ));
            }
        }
        if let Some((scene, take)) = view {
            let asset = self
                .timeline
                .document
                .as_ref()
                .and_then(|d| d.raw["runtime_takes"][&take].as_str())
                .map(str::to_owned);
            if let Some(asset) = asset {
                self.workspace = Workspace::Video;
                self.timeline.clear_audio_selection();
                self.timeline.selected.clear();
                self.timeline.selected.insert(SceneId(scene));
                self.timeline.section = None;
                self.timeline.sync_draft();
                self.timeline.selected_take = Some(take);
                self.timeline_load_media(ui.ctx(), asset);
            }
        }
        if let Some(asset) = delivery {
            self.workspace = Workspace::Video;
            self.timeline.selected_take = None;
            self.timeline_load_media(ui.ctx(), asset);
        }
    }
    fn timeline_cancel_job(&mut self, ctx: &egui::Context, id: String) {
        let (Some(client), Some(session)) = (self.client.clone(), self.timeline.session.clone())
        else {
            return;
        };
        let Some(job) = self
            .timeline
            .jobs
            .iter_mut()
            .find(|j| j.id == id && !j.terminal() && !j.acting)
        else {
            return;
        };
        job.acting = true;
        let tx = self.tx.clone();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = runtime().block_on(client.cancel(&id));
            send(
                &tx,
                &ctx,
                Message::Timeline(Event::Cancelled {
                    session,
                    id,
                    result,
                }),
            );
        });
    }
    fn timeline_retry_job(&mut self, ctx: &egui::Context, id: &str) {
        if self.timeline.admitting || self.timeline.pending.is_some() {
            return;
        }
        let Some(document) = self.timeline.document.as_ref() else {
            return;
        };
        let Some(job) = self
            .timeline
            .jobs
            .iter()
            .find(|j| j.id == id && j.retryable(&document.project))
        else {
            return;
        };
        if self.timeline.jobs.iter().any(|other| {
            !other.terminal()
                && other.request["project_id"] == job.request["project_id"]
                && other.request["scene_id"] == job.request["scene_id"]
        }) {
            return;
        }
        self.timeline.pending = Some((job.request.clone(), uuid::Uuid::new_v4().to_string()));
        self.timeline_submit_pending(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_observation_rejects_late_progress_but_accepts_completed_output() {
        let mut job = Job::new("job_1".into(), json!({"project_id":"project"})).unwrap();
        assert!(job.observe(json!({"id":"job_1","status":"running"})));
        assert!(job.observe(json!({"id":"job_1","status":"succeeded"})));
        assert!(!job.observe(json!({"id":"job_1","status":"running"})));
        assert!(job.terminal());
        assert!(!job.monitoring);
        assert!(job.observe(json!({"id":"job_1","status":"succeeded","output":{"take":{"id":"take_1","project_id":"project"}}})));
        assert!(!job.observe(json!({"id":"job_1","status":"queued"})));
        assert_eq!(job.take_id(), Some("take_1"));
    }
    #[test]
    fn lost_monitoring_retains_job_identity_and_active_state() {
        assert!(Job::new("../../jobs".into(), Value::Null).is_none());
        let mut job = Job::new(
            "job_1".into(),
            json!({"scene_id":"scene","project_id":"project"}),
        )
        .unwrap();
        job.monitoring = true;
        assert!(!job.observe(json!({"id":"job_1","status":"poll_failed"})));
        assert!(!job.terminal());
        assert!(job.queued());
        assert_eq!(job.id, "job_1");
        assert!(!job.monitoring);
        assert!(job.observe(json!({"id":"job_1","status":"cancelled"})));
        assert!(job.terminal());
        assert!(job.error.is_none());
    }
    #[test]
    fn published_result_uses_public_output_contract_and_project_identity() {
        let mut job = Job::new("job_1".into(), json!({"project_id":"project"})).unwrap();
        job.observe(json!({"id":"job_1","status":"succeeded","output":{"take":{"id":"take_1","project_id":"project"}}}));
        assert_eq!(job.take_id(), Some("take_1"));
        job.observe(json!({"id":"job_1","status":"succeeded","output":{"take":{"id":"take_other","project_id":"other"}}}));
        assert!(job.take_id().is_none());
    }
}
