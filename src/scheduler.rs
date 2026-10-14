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
    Cancel {
        owner: String,
        id: String,
    },
    Unload {
        cancel: bool,
    },
    Resume,
    Shutdown,
}
struct Message {
    action: Action,
    reply: oneshot::Sender<std::result::Result<Value, Fault>>,
}
#[derive(Clone)]
pub struct Control {
    tx: mpsc::Sender<Message>,
    pub state: watch::Receiver<Value>,
}
impl Control {
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
    engine: Box<dyn Backend>,
    result: Result<Vec<u8>>,
    stopped: Result<()>,
}
struct Scheduler {
    store: Store,
    engine: Option<Box<dyn Backend>>,
    active: Option<Active>,
    paused: bool,
    unload: Option<String>,
    state: watch::Sender<Value>,
}
pub async fn start(
    store: Store,
    engine: Box<dyn Backend>,
) -> Result<(Control, JoinHandle<Result<()>>)> {
    let paused = store.paused().await?;
    let (tx, rx) = mpsc::channel(128);
    let (state, receiver) = watch::channel(json!({"paused":paused,"active_job":false,"queued":0}));
    let scheduler = Scheduler {
        store,
        engine: Some(engine),
        active: None,
        paused,
        unload: None,
        state,
    };
    Ok((
        Control {
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
            Action::Submit {
                owner,
                mut request,
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
                let mut reference_preprocessing = Vec::new();
                for a in &request.inputs.reference_images {
                    let asset = self
                        .store
                        .asset(a, &owner)
                        .await?
                        .ok_or(Fault::new(404, "asset_not_found"))?;
                    let w = asset["width"].as_u64().unwrap();
                    let h = asset["height"].as_u64().unwrap();
                    let pixels = u64::from(
                        request
                            .model_options
                            .qwen_image_2_1
                            .reference_max_pixels_per_image,
                    )
                    .min(u64::from(request.width) * u64::from(request.height));
                    let raw_width = ((pixels * w / h) as f64).sqrt();
                    let width = (raw_width / 32.0).round() as u32 * 32;
                    let height = (raw_width * h as f64 / w as f64 / 32.0).round() as u32 * 32;
                    if width == 0 || height == 0 || width > 8192 || height > 8192 {
                        return Err(Fault::new(422, "unsupported_reference_geometry"));
                    }
                    reference_preprocessing.push(json!({"asset_id":a,"source_width":w,"source_height":h,"vae_width":width,"vae_height":height,
                        "pixels":width*height,"source":"pinned_native_resize_rule"}));
                }
                if request.seed.is_none() {
                    request.seed = Some(u32::from_le_bytes(
                        uuid::Uuid::new_v4().as_bytes()[..4].try_into().unwrap(),
                    ));
                }
                let id = id("job");
                let mut effective = request.effective();
                effective["reference_preprocessing"] = json!(reference_preprocessing);
                sqlx::query("INSERT INTO jobs (id,owner,status,request,request_hash,idempotency_key,created) VALUES (?,?,'queued',?,?,?,?)")
                    .bind(&id).bind(owner).bind(effective.to_string()).bind(fingerprint).bind(key).bind(now()).execute(&self.store.db).await?;
                Ok(json!({"id":id,"reused":false}))
            }
            Action::Cancel { owner, id } => {
                let job = self
                    .store
                    .job(&id, &owner)
                    .await?
                    .ok_or(Fault::new(404, "job_not_found"))?;
                match job["status"].as_str() {
                    Some("queued") => {
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
        if self.paused || self.active.is_some() {
            return Ok(());
        }
        let row=sqlx::query("SELECT id,owner,request FROM jobs WHERE status='queued' ORDER BY created,rowid LIMIT 1").fetch_optional(&self.store.db).await?;
        if let Some(row) = row {
            let job_id: String = row.get("id");
            let owner: String = row.get("owner");
            let mut value: Value = serde_json::from_str(row.get("request"))?;
            value.as_object_mut().unwrap().remove("profile_revision");
            value.as_object_mut().unwrap().remove("api_revision");
            value
                .as_object_mut()
                .unwrap()
                .remove("reference_preprocessing");
            let request: JobRequest = serde_json::from_value(value)?;
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
            let task = tokio::spawn(async move {
                let result = tokio::select! {
                    biased;
                    _=token.cancelled()=>Err(anyhow::anyhow!("cancelled")),
                    result=engine.generate(&request,&worker_owner,&store)=>result
                };
                let stopped = if result.is_err() {
                    engine.stop().await
                } else {
                    Ok(())
                };
                Finished {
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
                Ok(bytes) => self
                    .store
                    .save_asset(&active.owner, bytes)
                    .await
                    .context(EngineError("output_save_failed")),
                Err(error) => Err(error),
            };
            match result {
                Ok(asset) => {
                    let output = json!({"assets":[asset],"format":"png"});
                    sqlx::query(
                        "UPDATE jobs SET status='succeeded',output=?,finished=? WHERE id=?",
                    )
                    .bind(output.to_string())
                    .bind(now())
                    .bind(&active.id)
                    .execute(&self.store.db)
                    .await?;
                }
                Err(error) => {
                    let code = error
                        .downcast_ref::<EngineError>()
                        .map(|e| e.0)
                        .unwrap_or("generation_failed");
                    sqlx::query("UPDATE jobs SET status='failed',error=?,finished=? WHERE id=?")
                        .bind(code)
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
        if let Some(active) = self.active.take() {
            active.cancel.cancel();
            let finished = active.task.await?;
            self.engine = Some(finished.engine);
            finished.stopped?;
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
