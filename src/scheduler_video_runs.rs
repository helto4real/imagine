//! Serial run advancement reuses exactly the same admission, cancellation and FIFO.
use super::*;
use crate::video::{
    contract::{ProjectId, TakeId},
    execution,
    runs::{self, Status},
};
/// Stable request errors keep their own code; anything else is an invalid request.
fn fault(e: crate::video::persistence::StorageError) -> Fault {
    const KEPT: [&str; 11] = [
        "all_scenes_require_locked_seeds",
        "selected_scene_required",
        "rerun_required",
        "rerun_requires_rerun_mode",
        "rerun_scenes_required",
        "rerun_scene_duplicate",
        "rerun_scene_imported",
        "rerun_take_not_found",
        "rerun_would_repeat_take",
        "rerun_prompt_empty",
        "section_not_found",
    ];
    match e.code.as_str() {
        "revision_conflict" => Fault::new(409, "revision_conflict"),
        "scene_not_found" => Fault::new(422, "scene_not_found"),
        code => KEPT
            .into_iter()
            .find(|kept| *kept == code)
            .map_or(Fault::new(422, "invalid_run_request"), |code| {
                Fault::new(422, code)
            }),
    }
}
impl Scheduler {
    pub(super) async fn start_video_run(
        &mut self,
        owner: String,
        request: runs::Request,
        key: String,
    ) -> std::result::Result<Value, Fault> {
        if key.is_empty() || key.len() > 128 {
            return Err(Fault::new(400, "idempotency_key_required"));
        }
        let hash = hash(&serde_json::to_vec(&request).unwrap());
        if let Some(run) = self
            .video_runs
            .values()
            .find(|r| r.owner == owner && r.key == key)
        {
            if run.request_hash != hash {
                return Err(Fault::new(409, "idempotency_conflict"));
            }
            let mut value = json!(run.run);
            value["reused"] = json!(true);
            return Ok(value);
        }
        if self
            .video_runs
            .values()
            .any(|r| r.run.project_id == request.project_id && !r.run.status.terminal())
        {
            return Err(Fault::new(409, "run_already_active"));
        }
        let mut service = self.video.lock().await;
        let mut plan = runs::plan(
            service
                .projects
                .session(&request.project_id)
                .map_err(|_| Fault::new(404, "project_not_found"))?
                .document(),
            &request,
            owner,
            key,
        )
        .map_err(fault)?;
        // A rerun's prompt edits become one revision together with the run.
        if let Some(edited) = plan.edited.take() {
            service
                .projects
                .session_mut(&request.project_id)
                .map_err(crate::video::persistence::StorageError::from)
                .and_then(|session| session.publish(edited))
                .map_err(|_| Fault::new(503, "project_save_failed"))?;
        }
        let value = json!(plan.run);
        self.video_runs.insert(plan.run.id.clone(), plan);
        Ok(value)
    }
    /// What a start with this request would render, without changing anything.
    pub(super) async fn plan_video_run(
        &mut self,
        owner: String,
        request: runs::Request,
    ) -> std::result::Result<Value, Fault> {
        if self
            .video_runs
            .values()
            .any(|r| r.run.project_id == request.project_id && !r.run.status.terminal())
        {
            return Err(Fault::new(409, "run_already_active"));
        }
        let service = self.video.lock().await;
        let session = service
            .projects
            .session(&request.project_id)
            .map_err(|_| Fault::new(404, "project_not_found"))?;
        let plan = runs::plan(session.document(), &request, owner, String::new()).map_err(fault)?;
        Ok(plan.report)
    }
    pub(super) fn read_video_runs(
        &self,
        owner: &str,
        project: Option<&ProjectId>,
        id: Option<&str>,
    ) -> std::result::Result<Value, Fault> {
        if let Some(id) = id {
            return self
                .video_runs
                .get(id)
                .filter(|r| r.owner == owner)
                .map(|r| json!(r.run))
                .ok_or(Fault::new(404, "run_not_found"));
        }
        Ok(
            json!({"schema_version":1,"runs":self.video_runs.values().filter(|r|r.owner==owner && project.is_none_or(|p|p==&r.run.project_id)).map(|r|&r.run).collect::<Vec<_>>()}),
        )
    }
    pub(super) fn stop_video_run(
        &mut self,
        owner: &str,
        id: &str,
        cancel: bool,
    ) -> std::result::Result<Value, Fault> {
        let plan = self
            .video_runs
            .get_mut(id)
            .filter(|r| r.owner == owner)
            .ok_or(Fault::new(404, "run_not_found"))?;
        if !plan.run.status.terminal() {
            plan.run.status = if plan.run.current_job.is_none() {
                if cancel {
                    Status::Cancelled
                } else {
                    Status::Stopped
                }
            } else if cancel || plan.run.status == Status::Cancelling {
                Status::Cancelling
            } else {
                Status::Stopping
            };
        }
        Ok(json!(plan.run))
    }
    pub(super) async fn advance_video_runs(&mut self) -> Result<()> {
        let ids = self
            .video_runs
            .iter()
            .filter(|(_, p)| !p.run.status.terminal())
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in ids {
            let mut plan = self.video_runs.remove(&id).expect("listed run");
            let index = plan.run.completed();
            if let Some(job) = plan.run.current_job.clone() {
                if plan.run.status == Status::Cancelling && !plan.cancel_sent {
                    let _ = self
                        .command(Action::Cancel {
                            owner: plan.owner.clone(),
                            id: job.clone(),
                        })
                        .await;
                    plan.cancel_sent = true;
                }
                match self.store.job(&job, &plan.owner).await? {
                    None => plan.fail("run_job_orphaned"),
                    Some(value) => match value["status"].as_str() {
                        Some("succeeded") if plan.run.status == Status::Cancelling => {
                            plan.run.status = Status::Cancelled;
                            plan.run.current_job = None;
                        }
                        Some("succeeded") => {
                            let take = value["output"]["take"]["id"]
                                .as_str()
                                .map(|s| TakeId(s.into()));
                            let mut service = self.video.lock().await;
                            let result = take
                                .ok_or_else(|| crate::video::persistence::StorageError {
                                    code: "run_result_missing".into(),
                                })
                                .and_then(|take| {
                                    let session =
                                        service.projects.session_mut(&plan.run.project_id)?;
                                    plan.accept(session, index, take)
                                });
                            if let Err(e) = result {
                                plan.fail(&e.code);
                            }
                        }
                        Some("cancelled") => {
                            plan.run.status = Status::Cancelled;
                            plan.run.current_job = None;
                        }
                        Some("failed" | "interrupted") => plan.fail("run_job_failed"),
                        _ => {}
                    },
                }
            }
            if !plan.run.status.terminal() && plan.run.current_job.is_none() {
                if plan.run.status != Status::Running {
                    plan.run.status = Status::Stopped;
                } else {
                    let target = &plan.run.targets[plan.run.completed()];
                    let admission = {
                        let service = self.video.lock().await;
                        service
                            .projects
                            .session(&plan.run.project_id)
                            .map_err(crate::video::persistence::StorageError::from)
                            .and_then(|s| {
                                plan.guard(s.document(), &target.scene_id)?;
                                Ok(execution::Admission {
                                    kind: execution::Kind::VideoScene,
                                    schema_version: 1,
                                    project_id: plan.run.project_id.clone(),
                                    expected_project_revision: s
                                        .document()
                                        .project
                                        .project_revision,
                                    scene_id: target.scene_id.clone(),
                                    seed: target.seed,
                                })
                            })
                    };
                    match admission {
                        Err(e) => plan.fail(&e.code),
                        Ok(request) => match self
                            .command(Action::SubmitScene {
                                owner: plan.owner.clone(),
                                request,
                                key: Some(format!("{}-{}", plan.run.id, plan.run.completed())),
                            })
                            .await
                        {
                            Err(e) => plan.fail(e.code),
                            Ok(value) => {
                                let job = value["id"].as_str().expect("admitted scene").to_owned();
                                let index = plan.run.completed();
                                plan.run.targets[index].job_id = Some(job.clone());
                                plan.run.current_job = Some(job);
                            }
                        },
                    }
                }
            }
            self.video_runs.insert(id, plan);
        }
        Ok(())
    }
}
