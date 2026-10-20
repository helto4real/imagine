use crate::{
    contract::JobRequest,
    engine::{Backend, EngineError},
    storage::{Store, hash, id, now},
};
use anyhow::{Context, Result};
use serde_json::{Value, json};
use sqlx::Row;
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

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
pub enum Action {
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
    id: String,
    owner: String,
    cancel: CancellationToken,
    task: JoinHandle<Finished>,
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
    let scheduler = Scheduler {
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
                        Err(_)=> {sqlx::query("UPDATE jobs SET status='failed',error='coordinator_unavailable',finished=? WHERE id=?")
                            .bind(now()).bind(pending.id).execute(&self.store.db).await?;}
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
            if let Some(socket) = &self.coordinator
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
                            sqlx::query("UPDATE jobs SET status='failed',error='coordinator_unavailable',finished=? WHERE id=?").bind(now()).bind(&job_id).execute(&self.store.db).await?;
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
                return Ok(());
            }
            sqlx::query("UPDATE jobs SET status='running',started=? WHERE id=?")
                .bind(now())
                .bind(&job_id)
                .execute(&self.store.db)
                .await?;
            let mut engine = self.engine.take().context("missing engine")?;
            engine.begin_job(&job_id);
            let store = self.store.clone();
            let worker_owner = owner.clone();
            let cancel = CancellationToken::new();
            let token = cancel.clone();
            let mut ticket = self.granted.take();
            let task = tokio::spawn(async move {
                let result = tokio::select! {
                    biased;
                    _=token.cancelled()=>Err(anyhow::anyhow!("cancelled")),
                    _=async {ticket.as_mut().unwrap().lost().await}, if ticket.is_some()=>Err(anyhow::anyhow!(EngineError("coordinator_unavailable"))),
                    result=crate::generation::generate(engine.as_mut(),&request,&worker_owner,&store)=>result
                };
                let stopped = if result.is_err() {
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
                id: job_id,
                owner,
                cancel,
                task,
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
            if finished.result.is_ok() {
                self.engine.as_mut().unwrap().stop().await?;
            }
            sqlx::query("UPDATE jobs SET status='cancelled',finished=? WHERE id=?")
                .bind(now())
                .bind(&active.id)
                .execute(&self.store.db)
                .await?;
        } else {
            let result = match finished.result {
                Ok(artifacts) => self
                    .store
                    .publish_output(&active.id, &active.owner, artifacts)
                    .await
                    .context(EngineError("output_save_failed")),
                Err(error) => Err(error),
            };
            match result {
                Ok(()) => {}
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
                }
            }
        }
        self.engine.as_mut().unwrap().mark_idle();
        if self.unload.is_some() {
            self.finish_unload().await?;
        }
        if let Some(ticket) = ticket {
            let _ = ticket.finish().await;
        }
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
        }
        if let Some(engine) = &mut self.engine {
            engine.stop().await?;
        }
        Ok(())
    }
}
