use super::*;
use crate::video::assembly;
impl Scheduler {
    pub(super) async fn submit_assembly(
        &mut self,
        owner: String,
        plan_id: String,
        key: String,
    ) -> std::result::Result<Value, Fault> {
        if key.is_empty() || key.len() > 128 {
            return Err(Fault::new(400, "idempotency_key_required"));
        }
        let fingerprint = hash(plan_id.as_bytes());
        if let Some(row) =
            sqlx::query("SELECT id,request_hash FROM jobs WHERE owner=? AND idempotency_key=?")
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
        let mut service = self.video.lock().await;
        let entry = service
            .assemblies
            .get(&plan_id)
            .ok_or(Fault::new(404, "assembly_plan_not_found"))?;
        let frozen = entry
            .frozen
            .try_clone()
            .map_err(|_| Fault::new(500, "storage_error"))?;
        let progress = entry.progress.clone();
        let rendered = entry.rendered.clone();
        let job = id("job");
        let plan = &frozen.plan;
        let summary = json!({"kind":"video_assembly","schema_version":1,"project_id":plan.project_id,"project_revision":plan.project_revision,"assembly_plan":plan});
        // CPU jobs use the existing FIFO and never reserve/acquire a GPU ticket.
        sqlx::query("INSERT INTO jobs (id,owner,status,request,request_hash,idempotency_key,created) VALUES (?,?,'queued',?,?,?,?)").bind(&job).bind(&owner).bind(summary.to_string()).bind(fingerprint).bind(&key).bind(now()).execute(&self.store.db).await?;
        service.assemblies.get_mut(&plan_id).unwrap().job = Some(job.clone());
        progress.set("queued", 0, 1);
        self.assemblies
            .insert(job.clone(), (frozen, progress, rendered));
        log_job(&job, "received", None);
        Ok(json!({"id":job,"reused":false}))
    }
}
pub(super) async fn generate(
    store: Store,
    frozen: assembly::Frozen,
    progress: assembly::Progress,
    rendered: assembly::Rendered,
    token: CancellationToken,
) -> Result<crate::generation::Generated> {
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = cancelled.clone();
    let worker_progress = progress.clone();
    let mut task = tokio::task::spawn_blocking(move || -> Result<crate::generation::Generated> {
        let destination = store.videos.destination().map_err(EngineError)?;
        let proof = destination.proof().map_err(EngineError)?;
        let pending = destination
            .reserve(&json!(proof), 64 * 1024 * 1024)
            .map_err(EngineError)?;
        let plan = Box::new(frozen.plan.clone());
        let bytes = if let Some(bytes) = rendered.get() {
            worker_progress.set("saving_result", 0, 1);
            bytes
        } else {
            let bytes = std::sync::Arc::new(
                assembly::render(frozen, flag, worker_progress).map_err(EngineError)?,
            );
            rendered.set(bytes.clone());
            bytes
        };
        Ok(crate::generation::Generated::Delivery {
            bytes,
            plan,
            pending: Box::new(pending),
        })
    });
    let result = tokio::select! {
        biased;
        _=token.cancelled()=>{
            cancelled.store(true,std::sync::atomic::Ordering::Release);
            // Always join our CPU child pipeline before the FIFO advances.
            let _=task.await;Err(anyhow::anyhow!("cancelled"))
        }
        result=&mut task=>result.context("assembly_worker_failed")?,
    };
    if result.is_err() {
        progress.set(
            if token.is_cancelled() {
                "cancelled"
            } else {
                "failed"
            },
            0,
            1,
        );
    }
    result
}
