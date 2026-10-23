use crate::{
    contract::JobRequest,
    engine::{Backend, EngineError},
    storage::{Store, hash, id, now},
};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use sqlx::Row;
use std::time::Instant;
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;
#[path = "scheduler_assembly.rs"]
mod assembly;
#[path = "scheduler_video_runs.rs"]
mod video_runs;

fn log_job(id: &str, event: &str, started: Option<Instant>) {
    let seconds = now() / 1000;
    let elapsed = started
        .map(|start| format!(" elapsed={:.1}s", start.elapsed().as_secs_f64()))
        .unwrap_or_default();
    eprintln!(
        "[{:02}:{:02}:{:02} UTC] job={id} {event}{elapsed}",
        seconds / 3600 % 24,
        seconds / 60 % 60,
        seconds % 60,
    );
}

#[derive(Debug, Clone)]
pub struct Fault {
    pub status: u16,
    pub code: &'static str,
}
impl Fault {
    pub fn new(status: u16, code: &'static str) -> Self {
        Self { status, code }
    }
}
impl From<anyhow::Error> for Fault {
    fn from(_: anyhow::Error) -> Self {
        Self::new(500, "internal_error")
    }
}
impl From<sqlx::Error> for Fault {
    fn from(_: sqlx::Error) -> Self {
        Self::new(500, "storage_error")
    }
}
fn scene_fault(error: crate::video::persistence::StorageError) -> Fault {
    match error.code.as_str() {
        "revision_conflict" => Fault::new(409, "revision_conflict"),
        "project_not_found" => Fault::new(404, "project_not_found"),
        "scene_not_found" => Fault::new(404, "scene_not_found"),
        "scene_runtime_unsupported" => Fault::new(422, "scene_runtime_unsupported"),
        "scene_profile_mismatch" => Fault::new(422, "scene_profile_mismatch"),
        "unsupported_h3_sampling" => Fault::new(422, "unsupported_h3_sampling"),
        "h3_trained_upscaler_unavailable" => Fault::new(422, "h3_trained_upscaler_unavailable"),
        "continuity_method_unbound" => Fault::new(422, "continuity_method_unbound"),
        "continuity_source_not_accepted" => Fault::new(422, "continuity_source_not_accepted"),
        "continuity_endpoint_missing_or_ambiguous" => {
            Fault::new(422, "continuity_endpoint_missing_or_ambiguous")
        }
        "continuity_endpoint_incompatible" => Fault::new(422, "continuity_endpoint_incompatible"),
        "continuity_first_frame_conflict" => Fault::new(422, "continuity_first_frame_conflict"),
        "invalid_endpoint" | "endpoint_geometry_mismatch" => {
            Fault::new(422, "invalid_continuity_endpoint")
        }
        "model_settings_required" => Fault::new(422, "model_settings_required"),
        "reference_resource_missing" => Fault::new(422, "reference_resource_missing"),
        "resource_integrity_failed" => Fault::new(422, "resource_integrity_failed"),
        "unsupported_h3_geometry" => Fault::new(422, "unsupported_h3_geometry"),
        "imported_scene_not_generatable" => Fault::new(422, "imported_scene_not_generatable"),
        _ => Fault::new(422, "invalid_scene_request"),
    }
}
pub enum Action {
    SubmitAssembly {
        owner: String,
        plan_id: String,
        key: String,
    },
    StartVideoRun {
        owner: String,
        request: crate::video::runs::Request,
        key: String,
    },
    ReadVideoRuns {
        owner: String,
        project_id: Option<crate::video::contract::ProjectId>,
        id: Option<String>,
    },
    StopVideoRun {
        owner: String,
        id: String,
        cancel_current: bool,
    },
    SubmitScene {
        owner: String,
        request: crate::video::execution::Admission,
        key: Option<String>,
    },
    Submit {
        owner: String,
        request: Box<JobRequest>,
        key: Option<String>,
    },
    SubmitBatch {
        owner: String,
        requests: Vec<JobRequest>,
        key: Option<String>,
    },
    Cancel {
        owner: String,
        id: String,
    },
    Unload {
        cancel: bool,
    },
    Resume,
    YieldGpu,
    Shutdown,
}
struct Message {
    action: Action,
    reply: oneshot::Sender<std::result::Result<Value, Fault>>,
}
#[derive(Clone)]
pub struct Control {
    coordinated: bool,
    tx: mpsc::Sender<Message>,
    pub state: watch::Receiver<Value>,
    pub video: std::sync::Arc<tokio::sync::Mutex<crate::video::service::Service>>,
}
impl Control {
    pub fn is_coordinated(&self) -> bool {
        self.coordinated
    }
    pub async fn call(&self, action: Action) -> std::result::Result<Value, Fault> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Message { action, reply: tx })
            .await
            .map_err(|_| Fault::new(503, "scheduler_unavailable"))?;
        rx.await
            .map_err(|_| Fault::new(503, "scheduler_unavailable"))?
    }
}
struct Active {
    cpu: bool,
    id: String,
    owner: String,
    started: Instant,
    cancel: CancellationToken,
    task: JoinHandle<Finished>,
    target: Option<crate::video::execution::Target>,
}
struct Finished {
    ticket: Option<crate::coordination::Ticket>,
    engine: Box<dyn Backend>,
    result: Result<crate::generation::Generated>,
    stopped: Result<()>,
}
struct Prepared {
    id: String,
    effective: String,
    fingerprint: String,
}
struct Pending {
    id: String,
    task: JoinHandle<anyhow::Result<crate::coordination::Ticket>>,
}
struct Scheduler {
    assemblies: std::collections::HashMap<
        String,
        (
            crate::video::assembly::Frozen,
            crate::video::assembly::Progress,
            crate::video::assembly::Rendered,
        ),
    >,
    video_runs: std::collections::BTreeMap<String, crate::video::runs::Planned>,
    video: std::sync::Arc<tokio::sync::Mutex<crate::video::service::Service>>,
    scenes: std::collections::HashMap<String, crate::video::execution::Frozen>,
    store: Store,
    engine: Option<Box<dyn Backend>>,
    active: Option<Active>,
    coordinator: Option<std::path::PathBuf>,
    tickets: std::collections::HashMap<String, crate::coordination::Ticket>,
    pending: Option<Pending>,
    granted: Option<crate::coordination::Ticket>,
    paused: bool,
    unload: Option<String>,
    state: watch::Sender<Value>,
}
pub async fn start(
    store: Store,
    engine: Box<dyn Backend>,
) -> Result<(Control, JoinHandle<Result<()>>)> {
    start_coordinated(store, engine, None).await
}
pub async fn start_coordinated(
    store: Store,
    engine: Box<dyn Backend>,
    coordinator: Option<std::path::PathBuf>,
) -> Result<(Control, JoinHandle<Result<()>>)> {
    let paused = store.paused().await?;
    let (tx, rx) = mpsc::channel(128);
    let (state, receiver) = watch::channel(json!({"paused":paused,"active_job":false,"queued":0}));
    let coordinated = coordinator.is_some();
    let video = std::sync::Arc::new(tokio::sync::Mutex::new(
        crate::video::service::Service::with_reference_cache(engine.reference_cache()),
    ));
    let scheduler = Scheduler {
        assemblies: Default::default(),
        video_runs: Default::default(),
        video: video.clone(),
        scenes: Default::default(),
        coordinator,
        tickets: Default::default(),
        pending: None,
        granted: None,
        store,
        engine: Some(engine),
        active: None,
        paused,
        unload: None,
        state,
    };
    Ok((
        Control {
            video,
            coordinated,
            tx,
            state: receiver,
        },
        tokio::spawn(scheduler.run(rx)),
    ))
}
impl Scheduler {
    async fn run(mut self, mut rx: mpsc::Receiver<Message>) -> Result<()> {
        loop {
            self.advance_video_runs().await?;
            self.dispatch().await?;
            self.publish().await?;
            tokio::select! {
                biased;
                message=rx.recv()=> {
                    let Some(message)=message else {self.shutdown().await?;break};
                    if matches!(message.action,Action::Shutdown) {
                        let result=self.shutdown().await;
                        let _=message.reply.send(result.map(|_|json!({"stopped":true})).map_err(Fault::from));
                        break;
                    }
                    let result=self.command(message.action).await;
                    self.publish().await?;
                    let _=message.reply.send(result);
                }
                granted=async {(&mut self.pending.as_mut().unwrap().task).await}, if self.pending.is_some()=> {
                    let pending=self.pending.take().unwrap();
                    match granted.context("GPU ticket task panicked")? {
                        Ok(ticket)=> self.granted=Some(ticket),
                        Err(_)=> {self.scenes.remove(&pending.id);sqlx::query("UPDATE jobs SET status='failed',error='coordinator_unavailable',finished=? WHERE id=?")
                            .bind(now()).bind(&pending.id).execute(&self.store.db).await?;
                            log_job(&pending.id, "failed", None);}
                    }
                }
                finished=async {(&mut self.active.as_mut().unwrap().task).await}, if self.active.is_some()=> {
                    let finished=finished.context("generation task panicked")?;
                    self.complete(finished).await?;
                }
            }
        }
        Ok(())
    }
    async fn publish(&mut self) -> Result<()> {
        let queued: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM jobs WHERE status='queued'")
            .fetch_one(&self.store.db)
            .await?;
        self.state.send_replace(json!({"paused":self.paused,"active_job":self.active.is_some(),"queued":queued,"unload_operation":self.unload}));
        Ok(())
    }
    async fn command(&mut self, action: Action) -> std::result::Result<Value, Fault> {
        match action {
            Action::SubmitAssembly {
                owner,
                plan_id,
                key,
            } => self.submit_assembly(owner, plan_id, key).await,
            Action::StartVideoRun {
                owner,
                request,
                key,
            } => self.start_video_run(owner, request, key).await,
            Action::ReadVideoRuns {
                owner,
                project_id,
                id,
            } => self.read_video_runs(&owner, project_id.as_ref(), id.as_deref()),
            Action::StopVideoRun {
                owner,
                id,
                cancel_current,
            } => self.stop_video_run(&owner, &id, cancel_current),
            Action::SubmitScene {
                owner,
                request,
                key,
            } => {
                if key.as_ref().is_none_or(|k| k.is_empty() || k.len() > 128) {
                    return Err(Fault::new(400, "idempotency_key_required"));
                }
                let fingerprint = hash(&serde_json::to_vec(&request).expect("scene admission"));
                if let Some(row) = sqlx::query(
                    "SELECT id,request_hash FROM jobs WHERE owner=? AND idempotency_key=?",
                )
                .bind(&owner)
                .bind(&key)
                .fetch_optional(&self.store.db)
                .await?
                {
                    if row.get::<String, _>("request_hash") != fingerprint {
                        return Err(Fault::new(409, "idempotency_conflict"));
                    }
                    return Ok(json!({"id":row.get::<String,_>("id"),"reused":true}));
                }
                if self.paused {
                    return Err(Fault::new(409, "runtime_suspended"));
                }
                let socket = self
                    .coordinator
                    .as_ref()
                    .ok_or(Fault::new(503, "coordinator_unavailable"))?;
                let staging = crate::video::execution::staging(&self.store).map_err(scene_fault)?;
                let frozen =
                    crate::video::execution::freeze(&*self.video.lock().await, &request, &staging)
                        .map_err(scene_fault)?;
                let destination = self
                    .store
                    .output_destination(Some(&frozen.target.project_id.0), true)
                    .map_err(|c| Fault::new(422, c))?;
                let proof = destination.proof().map_err(|c| Fault::new(422, c))?;
                destination
                    .preflight(
                        &json!(proof),
                        frozen.request.shape.raw_bytes() + 64 * 1024 * 1024,
                    )
                    .map_err(|c| Fault::new(422, c))?;
                let summary = frozen.summary().to_string();
                let job = id("job");
                let ticket = crate::coordination::Ticket::register(socket, &job, summary.len())
                    .await
                    .map_err(|_| Fault::new(503, "coordinator_unavailable"))?;
                sqlx::query("INSERT INTO jobs (id,owner,status,request,request_hash,idempotency_key,created) VALUES (?,?,'queued',?,?,?,?)").bind(&job).bind(owner).bind(summary).bind(fingerprint).bind(key).bind(now()).execute(&self.store.db).await?;
                self.scenes.insert(job.clone(), frozen);
                self.tickets.insert(job.clone(), ticket);
                log_job(&job, "received", None);
                Ok(json!({"id":job,"reused":false}))
            }
            Action::SubmitBatch {
                owner,
                requests,
                key,
            } => {
                if requests.is_empty() {
                    return Err(Fault::new(422, "invalid_request"));
                }
                if key.as_ref().is_some_and(|k| k.is_empty() || k.len() > 128) {
                    return Err(Fault::new(400, "invalid_idempotency_key"));
                }
                for request in &requests {
                    request.validate().map_err(|e| Fault::new(422, e))?;
                }
                let fingerprint =
                    hash(&serde_json::to_vec(&requests).expect("serializable requests"));
                if let Some(key) = &key
                    && let Some(row) = sqlx::query("SELECT request_hash,job_ids FROM job_batches WHERE owner=? AND idempotency_key=?")
                        .bind(&owner).bind(key).fetch_optional(&self.store.db).await? {
                        if row.get::<String, _>("request_hash") != fingerprint { return Err(Fault::new(409, "idempotency_conflict")); }
                        let ids: Vec<String> = serde_json::from_str(row.get("job_ids")).map_err(|_| Fault::new(500, "storage_error"))?;
                        return Ok(json!({"jobs":ids.into_iter().map(|id|json!({"id":id,"reused":true})).collect::<Vec<_>>()}));
                    }
                if self.paused {
                    return Err(Fault::new(409, "runtime_suspended"));
                }
                let mut prepared = Vec::new();
                for request in requests {
                    prepared.push(self.prepare_request(&owner, request).await?);
                }
                let ids: Vec<_> = prepared.iter().map(|p| p.id.clone()).collect();
                let tickets = if let Some(socket) = &self.coordinator {
                    let jobs = prepared
                        .iter()
                        .map(|p| (p.id.clone(), p.effective.len()))
                        .collect::<Vec<_>>();
                    crate::coordination::Ticket::register_many(socket, &jobs)
                        .await
                        .map_err(|_| Fault::new(503, "coordinator_unavailable"))?
                } else {
                    Vec::new()
                };
                let mut tx = self.store.db.begin().await?;
                for p in prepared {
                    sqlx::query("INSERT INTO jobs (id,owner,status,request,request_hash,created) VALUES (?,?,'queued',?,?,?)")
                        .bind(p.id).bind(&owner).bind(p.effective).bind(p.fingerprint).bind(now()).execute(&mut *tx).await?;
                }
                if let Some(key) = key {
                    sqlx::query("INSERT INTO job_batches (owner,idempotency_key,request_hash,job_ids) VALUES (?,?,?,?)")
                        .bind(owner).bind(key).bind(fingerprint).bind(json!(ids).to_string()).execute(&mut *tx).await?;
                }
                tx.commit().await?;
                for id in &ids {
                    log_job(id, "received", None);
                }
                for (id, ticket) in ids.iter().zip(tickets) {
                    self.tickets.insert(id.clone(), ticket);
                }
                Ok(
                    json!({"jobs":ids.into_iter().map(|id|json!({"id":id,"reused":false})).collect::<Vec<_>>()}),
                )
            }
            Action::Submit {
                owner,
                request,
                key,
            } => {
                request.validate().map_err(|e| Fault::new(422, e))?;
                let fingerprint =
                    hash(&serde_json::to_vec(&request).expect("serializable request"));
                if let Some(key) = &key {
                    if key.is_empty() || key.len() > 128 {
                        return Err(Fault::new(400, "invalid_idempotency_key"));
                    }
                    if let Some(row) = sqlx::query(
                        "SELECT id,request_hash FROM jobs WHERE owner=? AND idempotency_key=?",
                    )
                    .bind(&owner)
                    .bind(key)
                    .fetch_optional(&self.store.db)
                    .await?
                    {
                        if row.get::<String, _>("request_hash") != fingerprint {
                            return Err(Fault::new(409, "idempotency_conflict"));
                        }
                        return Ok(json!({"id":row.get::<String,_>("id"),"reused":true}));
                    }
                }
                if self.paused {
                    return Err(Fault::new(409, "runtime_suspended"));
                }
                let p = self.prepare_request(&owner, *request).await?;
                let ticket = if let Some(socket) = &self.coordinator {
                    Some(
                        crate::coordination::Ticket::register(socket, &p.id, p.effective.len())
                            .await
                            .map_err(|_| Fault::new(503, "coordinator_unavailable"))?,
                    )
                } else {
                    None
                };
                sqlx::query("INSERT INTO jobs (id,owner,status,request,request_hash,idempotency_key,created) VALUES (?,?,'queued',?,?,?,?)")
                    .bind(&p.id).bind(owner).bind(p.effective).bind(p.fingerprint).bind(key).bind(now()).execute(&self.store.db).await?;
                if let Some(ticket) = ticket {
                    self.tickets.insert(p.id.clone(), ticket);
                }
                log_job(&p.id, "received", None);
                Ok(json!({"id":p.id,"reused":false}))
            }
            Action::Cancel { owner, id } => {
                let job = self
                    .store
                    .job(&id, &owner)
                    .await?
                    .ok_or(Fault::new(404, "job_not_found"))?;
                match job["status"].as_str() {
                    Some("queued") => {
                        if let Some((_, progress, _)) = self.assemblies.remove(&id) {
                            progress.set("cancelled", 0, 1);
                        }
                        self.scenes.remove(&id);
                        self.tickets.remove(&id);
                        if self.pending.as_ref().is_some_and(|p| p.id == id) {
                            let pending = self.pending.take().unwrap();
                            pending.task.abort();
                            let _ = pending.task.await;
                        }
                        self.granted = None;
                        sqlx::query("UPDATE jobs SET status='cancelled',finished=? WHERE id=?")
                            .bind(now())
                            .bind(&id)
                            .execute(&self.store.db)
                            .await?;
                        log_job(&id, "cancelled", None);
                    }
                    Some("running" | "cancelling") => {
                        self.cancel_active().await?;
                    }
                    Some("cancelled") => {}
                    _ => return Err(Fault::new(409, "job_already_finished")),
                }
                Ok(json!({"id":id}))
            }
            Action::Unload { cancel } => {
                if self.unload.is_some() {
                    return Err(Fault::new(409, "unload_in_progress"));
                }
                self.tickets.clear();
                self.granted = None;
                if let Some(pending) = self.pending.take() {
                    pending.task.abort();
                    let _ = pending.task.await;
                }

                if self.unload.is_some() {
                    return Err(Fault::new(409, "unload_in_progress"));
                }
                self.store.set_paused(true).await?;
                self.paused = true;
                let id = id("operation");
                sqlx::query(
                    "INSERT INTO operations (id,status,mode,created) VALUES (?,'running',?,?)",
                )
                .bind(&id)
                .bind(if cancel { "cancel" } else { "wait" })
                .bind(now())
                .execute(&self.store.db)
                .await?;
                self.unload = Some(id.clone());
                if cancel && self.active.is_some() {
                    self.cancel_active().await?;
                }
                if self.active.is_none() {
                    self.finish_unload().await?;
                }
                Ok(json!({"id":id}))
            }
            Action::YieldGpu => {
                if self.active.is_some() {
                    return Err(Fault::new(409, "gpu_operation_active"));
                }
                self.engine
                    .as_mut()
                    .ok_or(Fault::new(503, "engine_unavailable"))?
                    .stop()
                    .await?;
                Ok(json!({"released":true}))
            }
            Action::Resume => {
                if self.unload.is_some() {
                    return Err(Fault::new(409, "unload_in_progress"));
                }
                self.store.set_paused(false).await?;
                self.paused = false;
                Ok(json!({"paused":false}))
            }
            Action::Shutdown => unreachable!(),
        }
    }
    async fn prepare_request(
        &mut self,
        owner: &str,
        mut request: JobRequest,
    ) -> std::result::Result<Prepared, Fault> {
        request.validate().map_err(|e| Fault::new(422, e))?;
        if request.model == "minimax-h3" && self.coordinator.is_none() {
            return Err(Fault::new(503, "coordinator_unavailable"));
        }
        // API clients without a local-folder selection use the configured server destination.
        // UI clients send a proof, which must match; no client path becomes authoritative.
        if self.store.private_root().is_some()
            || (request.model != "minimax-h3"
                && !request.client_metadata.contains_key("video_destination"))
        {
            let proof = self
                .store
                .output_destination(None, request.model == "minimax-h3")
                .and_then(|d| d.proof())
                .map_err(|e| Fault::new(422, e))?;
            request
                .client_metadata
                .insert("video_destination".into(), json!(proof));
        }
        let required = if request.model == "minimax-h3" {
            crate::video::job::request(&request)
                .map_err(|e| Fault::new(422, e))?
                .shape
                .raw_bytes()
                + 64 * 1024 * 1024
        } else if request.operation == crate::contract::Operation::CharacterSheet {
            96 * 1024 * 1024
        } else {
            32 * 1024 * 1024
        };
        self.store
            .destination_for_proof(&request.client_metadata["video_destination"])
            .and_then(|d| d.preflight(&request.client_metadata["video_destination"], required))
            .map_err(|e| Fault::new(422, e))?;
        let fingerprint = hash(&serde_json::to_vec(&request).expect("serializable request"));
        let reference_preprocessing =
            crate::generation::reference_metadata(&request, owner, &self.store).await?;
        if request.seed.is_none() {
            // Use only random UUID bytes (exclude version/variant bits), then
            // mask to the API's complete, exactly representable 53-bit range.
            let uuid = uuid::Uuid::new_v4();
            let b = uuid.as_bytes();
            request.seed = Some(
                u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[9], b[10]])
                    & crate::contract::MAX_SEED,
            );
        }
        let mut effective = request.effective();
        effective["reference_preprocessing"] = json!(reference_preprocessing);
        Ok(Prepared {
            id: id("job"),
            effective: effective.to_string(),
            fingerprint,
        })
    }
    async fn cancel_active(&mut self) -> Result<()> {
        if let Some(active) = &self.active {
            sqlx::query("UPDATE jobs SET status='cancelling' WHERE id=?")
                .bind(&active.id)
                .execute(&self.store.db)
                .await?;
            active.cancel.cancel();
        }
        Ok(())
    }
    async fn dispatch(&mut self) -> Result<()> {
        if self.paused || self.active.is_some() || self.pending.is_some() {
            return Ok(());
        }
        let row=sqlx::query("SELECT id,owner,request FROM jobs WHERE status='queued' ORDER BY created,rowid LIMIT 1").fetch_optional(&self.store.db).await?;
        if let Some(row) = row {
            let job_id: String = row.get("id");
            let cpu = self.assemblies.contains_key(&job_id);
            if !cpu
                && let Some(socket) = &self.coordinator
                && self.granted.is_none()
            {
                let ticket = if let Some(ticket) = self.tickets.remove(&job_id) {
                    ticket
                } else {
                    match crate::coordination::Ticket::register(
                        socket,
                        &job_id,
                        row.get::<String, _>("request").len(),
                    )
                    .await
                    {
                        Ok(ticket) => ticket,
                        Err(_) => {
                            self.scenes.remove(&job_id);
                            sqlx::query("UPDATE jobs SET status='failed',error='coordinator_unavailable',finished=? WHERE id=?").bind(now()).bind(&job_id).execute(&self.store.db).await?;
                            log_job(&job_id, "failed", None);
                            return Ok(());
                        }
                    }
                };
                self.pending = Some(Pending {
                    id: job_id,
                    task: tokio::spawn(ticket.acquire()),
                });
                return Ok(());
            }
            let owner: String = row.get("owner");
            let scene = self.scenes.remove(&job_id);
            let assembly = self.assemblies.remove(&job_id);
            let request = if scene.is_some() || assembly.is_some() {
                None
            } else {
                let mut value: Value = serde_json::from_str(row.get("request"))?;
                value.as_object_mut().unwrap().remove("profile_revision");
                value.as_object_mut().unwrap().remove("api_revision");
                value
                    .as_object_mut()
                    .unwrap()
                    .remove("reference_preprocessing");
                let plan = value.as_object_mut().unwrap().remove("execution_plan");
                let request: JobRequest = serde_json::from_value(value)?;
                if plan.as_ref() != request.effective().get("execution_plan") {
                    sqlx::query("UPDATE jobs SET status='failed',error='execution_plan_mismatch',finished=? WHERE id=?")
                    .bind(now()).bind(&job_id).execute(&self.store.db).await?;
                    if let Some(ticket) = self.granted.take() {
                        let _ = ticket.finish().await;
                    }
                    log_job(&job_id, "failed", None);
                    return Ok(());
                }
                Some(request)
            };
            sqlx::query("UPDATE jobs SET status='running',started=? WHERE id=?")
                .bind(now())
                .bind(&job_id)
                .execute(&self.store.db)
                .await?;
            let mut engine = self.engine.take().context("missing engine")?;
            if !cpu {
                engine.begin_job(&job_id);
            }
            let store = self.store.clone();
            let worker_owner = owner.clone();
            let cancel = CancellationToken::new();
            let token = cancel.clone();
            let mut ticket = self.granted.take();
            let (target, scene_input) = if let Some(scene) = scene {
                let project = scene.target.project_id.clone();
                (
                    Some(scene.target),
                    Some((
                        scene.request,
                        scene.first_frame,
                        scene.reference_inputs,
                        project,
                    )),
                )
            } else {
                (None, None)
            };
            let started = Instant::now();
            log_job(&job_id, "started", None);
            let task = tokio::spawn(async move {
                let result = if let Some((frozen, progress, rendered)) = assembly {
                    assembly::generate(store, frozen, progress, rendered, token.clone()).await
                } else {
                    tokio::select! {
                        biased;
                        _=token.cancelled()=>Err(anyhow::anyhow!("cancelled")),
                        _=async {ticket.as_mut().unwrap().lost().await}, if ticket.is_some()=>Err(anyhow::anyhow!(EngineError("coordinator_unavailable"))),
                        result=async {
                            if let Some((input,image,references,project))=scene_input {
                                let proof=store.output_destination(Some(&project.0), true).and_then(|d|d.proof()).map_err(EngineError)?;
                                crate::video::execution::generate(engine.as_mut(),input,image,references,json!(proof),&store).await
                            } else {
                                crate::generation::generate(engine.as_mut(),request.as_ref().expect("image or legacy video"),&worker_owner,&store).await
                            }
                        }=>result
                    }
                };
                let stopped = if result.is_err() && !cpu {
                    engine.stop().await
                } else {
                    Ok(())
                };
                Finished {
                    ticket,
                    engine,
                    result,
                    stopped,
                }
            });
            self.active = Some(Active {
                cpu,
                id: job_id,
                owner,
                started,
                cancel,
                task,
                target,
            });
        }
        Ok(())
    }
    async fn complete(&mut self, finished: Finished) -> Result<()> {
        let active = self.active.take().unwrap();
        let ticket = finished.ticket;
        self.engine = Some(finished.engine);
        finished.stopped.context("could not confirm native exit")?;
        if active.cancel.is_cancelled() {
            // A result that raced with cancellation is never published.
            if finished.result.is_ok() && !active.cpu {
                self.engine.as_mut().unwrap().stop().await?;
            }
            sqlx::query("UPDATE jobs SET status='cancelled',finished=? WHERE id=?")
                .bind(now())
                .bind(&active.id)
                .execute(&self.store.db)
                .await?;
            log_job(&active.id, "cancelled", Some(active.started));
        } else {
            let result = match finished.result {
                Ok(artifacts) => self
                    .publish_result(&active.id, &active.owner, active.target, artifacts)
                    .await
                    .map_err(|error| {
                        if error.downcast_ref::<EngineError>().is_some() {
                            error
                        } else {
                            error.context(EngineError("output_save_failed"))
                        }
                    }),
                Err(error) => Err(error),
            };
            match result {
                Ok(()) => log_job(&active.id, "succeeded", Some(active.started)),
                Err(error) => {
                    let details =
                        error.downcast_ref::<crate::component_error::ComponentLoadError>();
                    let code = if details.is_some() {
                        "model_component_load_failed"
                    } else {
                        error
                            .downcast_ref::<EngineError>()
                            .map(|e| e.0)
                            .unwrap_or("generation_failed")
                    };
                    sqlx::query("UPDATE jobs SET status='failed',error=?,error_details=?,finished=? WHERE id=?")
                        .bind(code)
                        .bind(details.filter(|details|details.valid()).map(serde_json::to_string).transpose()?)
                        .bind(now())
                        .bind(&active.id)
                        .execute(&self.store.db)
                        .await?;
                    log_job(&active.id, "failed", Some(active.started));
                }
            }
        }
        if !active.cpu {
            self.engine.as_mut().unwrap().mark_idle();
        }
        if self.unload.is_some() {
            self.finish_unload().await?;
        }
        if let Some(ticket) = ticket {
            let _ = ticket.finish().await;
        }
        Ok(())
    }
    async fn publish_result(
        &mut self,
        job: &str,
        owner: &str,
        target: Option<crate::video::execution::Target>,
        artifacts: crate::generation::Generated,
    ) -> Result<()> {
        if let crate::generation::Generated::Delivery {
            bytes,
            plan,
            pending,
        } = artifacts
        {
            let staging = crate::video::execution::staging(&self.store)?;
            let mut service = self.video.lock().await;
            let mut prepared = crate::video::assembly::publication::prepare(
                &service,
                (*plan).clone(),
                &bytes,
                &staging,
            )?;
            let plan_id = plan.id.clone();
            self.store
                .publish_delivery(
                    job,
                    owner,
                    crate::generation::Generated::Delivery {
                        bytes,
                        pending,
                        plan,
                    },
                    prepared.asset_id.0.clone(),
                    prepared.media.take().expect("staged delivery"),
                )
                .await?;
            prepared.commit(&mut service);
            if let Some(entry) = service.assemblies.get(&plan_id) {
                entry.rendered.clear();
                entry.progress.set("saved", 1, 1);
            }
            return Ok(());
        }
        let Some(target) = target else {
            return self.store.publish_output(job, owner, artifacts).await;
        };
        let crate::generation::Generated::Video {
            bytes,
            take_artifacts,
            ..
        } = &artifacts
        else {
            anyhow::bail!("invalid_scene_output");
        };
        let staging = crate::video::execution::staging(&self.store)?;
        let mut service = self.video.lock().await;
        let mut prepared =
            crate::video::takes::prepare(&service, target, bytes, take_artifacts, &staging)?;
        let info = json!({"project_id":prepared.project_id,"id":prepared.take_id});
        let asset = self
            .store
            .publish_scene(
                job,
                owner,
                artifacts,
                info,
                prepared.media.take().expect("staged video"),
            )
            .await?;
        // No await between successful SQL commit and the already validated session swap.
        prepared.commit(&mut service, asset);
        Ok(())
    }
    async fn finish_unload(&mut self) -> Result<()> {
        self.engine
            .as_mut()
            .context("engine missing during unload")?
            .stop()
            .await?;
        if let Some(id) = &self.unload {
            sqlx::query("UPDATE operations SET status='succeeded',finished=? WHERE id=?")
                .bind(now())
                .bind(id)
                .execute(&self.store.db)
                .await?;
        }
        self.unload = None;
        Ok(())
    }
    async fn shutdown(&mut self) -> Result<()> {
        self.tickets.clear();
        self.granted = None;
        if let Some(pending) = self.pending.take() {
            pending.task.abort();
            let _ = pending.task.await;
        }
        if let Some(active) = self.active.take() {
            active.cancel.cancel();
            let finished = active.task.await?;
            self.engine = Some(finished.engine);
            finished.stopped?;
            self.engine.as_mut().unwrap().stop().await?;
            if let Some(ticket) = finished.ticket {
                let _ = ticket.finish().await;
            }
            sqlx::query(
                "UPDATE jobs SET status='interrupted',error='server_stopped',finished=? WHERE id=?",
            )
            .bind(now())
            .bind(&active.id)
            .execute(&self.store.db)
            .await?;
            log_job(&active.id, "interrupted", Some(active.started));
        }
        if let Some(engine) = &mut self.engine {
            engine.stop().await?;
        }
        self.scenes.clear();
        self.assemblies.clear();
        sqlx::query("UPDATE jobs SET status='interrupted',error='server_stopped',finished=? WHERE status='queued'")
            .bind(now()).execute(&self.store.db).await?;
        Ok(())
    }
}
