use crate::{
    backend::{Backends, Target},
    comfy::Comfy,
    config::{Config, label},
    logging::log,
};
use anyhow::{Result, anyhow, bail};
use axum::http::HeaderMap;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{
    sync::{Mutex as AsyncMutex, Notify, oneshot},
    task::JoinHandle,
    time::timeout,
};
use tokio_util::sync::CancellationToken;
const LOG: &str = "st_proxy.coordinator";
#[derive(Clone, Debug)]
pub struct Buffered {
    pub status: u16,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}
struct Chat {
    sequence: u64,
    target: Option<Target>,
    ready: oneshot::Sender<Result<Lease>>,
    cancel: CancellationToken,
}
struct Image {
    sequence: u64,
    path: String,
    query: String,
    headers: HeaderMap,
    body: Vec<u8>,
    submitted: oneshot::Sender<Result<Buffered>>,
    completion: Option<oneshot::Sender<Result<Value>>>,
    cancel: CancellationToken,
}
struct Imagine {
    stream: oneshot::Receiver<ImagineStream>,
    bytes: usize,
    cancel: CancellationToken,
}
struct ImagineStream {
    stream: tokio::net::UnixStream,
    watcher: JoinHandle<()>,
}
impl Drop for ImagineStream {
    fn drop(&mut self) {
        self.watcher.abort();
    }
}
type Registration = (oneshot::Sender<ImagineStream>, CancellationToken);
enum Work {
    Chat(Chat),
    Image(Image),
    Imagine(Imagine),
}
struct State {
    queue: VecDeque<Work>,
    sequence: u64,
    active_chats: usize,
    metadata: usize,
    controls: usize,
    waiting_chats: usize,
    waiting_images: usize,
    bytes: usize,
    owner: Option<&'static str>,
    phase: &'static str,
    changed: Instant,
    fatal: Option<String>,
    recoverable: Option<String>,
    last_error: Option<String>,
    closing: bool,
    active_prompt: Option<String>,
    recovering: bool,
    attempts: usize,
    dispatcher_alive: bool,
}
#[derive(Clone)]
pub struct Coordinator {
    pub config: Config,
    pub backends: Arc<AsyncMutex<Backends>>,
    pub comfy: Comfy,
    state: Arc<Mutex<State>>,
    notify: Arc<Notify>,
    stop: CancellationToken,
    registrations: Arc<Mutex<std::collections::HashMap<String, Registration>>>,
}
pub struct Lease {
    coordinator: Coordinator,
    kind: &'static str,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut s = self.coordinator.state.lock().unwrap();
        match self.kind {
            "chat" => {
                s.active_chats -= 1;
                log(
                    10,
                    LOG,
                    format!("chat lease released: active_chats={}", s.active_chats),
                );
            }
            "metadata" => s.metadata -= 1,
            "control" => {
                s.controls -= 1;
                log(
                    10,
                    LOG,
                    format!(
                        "ComfyUI control lease released: active_controls={}",
                        s.controls
                    ),
                );
            }
            _ => {}
        }
        drop(s);
        self.coordinator.notify.notify_waiters();
    }
}
struct CancelOnDrop(CancellationToken, Coordinator);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
        self.1.remove_cancelled();
    }
}
impl Coordinator {
    pub async fn start(
        config: Config,
        client: reqwest::Client,
        limit: crate::connection_limit::Limit,
    ) -> (Self, JoinHandle<()>) {
        let backends = Backends::limited(&config, client.clone(), limit.clone());
        let this = Self {
            comfy: Comfy {
                config: config.clone(),
                client,
                limit,
            },
            config,
            backends: Arc::new(AsyncMutex::new(backends)),
            state: Arc::new(Mutex::new(State {
                queue: VecDeque::new(),
                sequence: 0,
                active_chats: 0,
                metadata: 0,
                controls: 0,
                waiting_chats: 0,
                waiting_images: 0,
                bytes: 0,
                owner: None,
                phase: "initializing",
                changed: Instant::now(),
                fatal: None,
                recoverable: None,
                last_error: None,
                closing: false,
                active_prompt: None,
                recovering: false,
                attempts: 0,
                dispatcher_alive: true,
            })),
            notify: Arc::new(Notify::new()),
            stop: CancellationToken::new(),
            registrations: Default::default(),
        };
        log(20, LOG, "broker initialization started");
        if this.config.llm_backend == "ollama" {
            let init = async {
                this.backends.lock().await.validate().await?;
                this.phase("cleaning_comfy");
                this.comfy.free().await?;
                this.phase("verifying_llm");
                let t = this.backends.lock().await.entries[0].snapshot().await?;
                Ok::<_, anyhow::Error>(t)
            }
            .await;
            match init {
                Ok(t) => {
                    this.backends.lock().await.active = Some(t);
                    this.state.lock().unwrap().owner = Some("llm");
                    this.phase("llm_ready");
                    log(
                        20,
                        LOG,
                        format!(
                            "broker initialization completed: GPU owner=Ollama idle_timeout={:.1}s",
                            this.config.idle_timeout
                        ),
                    );
                }
                Err(e) => {
                    log(40, LOG, format!("broker initialization failed: {e}"));
                    let mut s = this.state.lock().unwrap();
                    s.last_error = Some(e.to_string());
                    s.fatal = Some("broker initialization failed".into());
                    drop(s);
                    this.phase("error");
                }
            }
        } else {
            this.phase("awaiting_request");
            log(
                20,
                LOG,
                format!(
                    "broker initialization completed without contacting {}; waiting for active work",
                    label(&this.config.llm_backend)
                ),
            );
        }
        let cloned = this.clone();
        let task = tokio::spawn(async move {
            cloned.dispatch().await;
        });
        (this, task)
    }
    fn phase(&self, phase: &'static str) {
        let mut s = self.state.lock().unwrap();
        if s.phase != phase {
            log(20, LOG, format!("state transition: {} -> {phase}", s.phase));
            s.phase = phase;
            s.changed = Instant::now();
        }
    }
    fn admission(&self, s: &State) -> Result<()> {
        if s.closing {
            bail!("broker is shutting down")
        }
        if let Some(e) = &s.fatal {
            bail!("{e}")
        }
        Ok(())
    }
    fn remove_cancelled(&self) {
        let mut s = self.state.lock().unwrap();
        let mut kept = VecDeque::new();
        while let Some(w) = s.queue.pop_front() {
            match w {
                Work::Imagine(w) if w.cancel.is_cancelled() => {
                    s.waiting_images -= 1;
                    s.bytes -= w.bytes;
                }
                Work::Chat(w) if w.cancel.is_cancelled() => {
                    s.waiting_chats -= 1;
                }
                Work::Image(w) if w.cancel.is_cancelled() => {
                    s.waiting_images -= 1;
                    s.bytes -= w.body.len();
                    let _ = w
                        .submitted
                        .send(Err(anyhow!("image request was cancelled")));
                    if let Some(c) = w.completion {
                        let _ = c.send(Err(anyhow!("image request was cancelled")));
                    }
                    log(
                        20,
                        LOG,
                        format!(
                            "cancelled image request removed before submission: sequence={}",
                            w.sequence
                        ),
                    );
                }
                w => kept.push_back(w),
            }
        }
        s.queue = kept;
        drop(s);
        self.notify.notify_waiters();
    }
    pub async fn chat(&self, target: Option<Target>) -> Result<Lease> {
        let (tx, rx) = oneshot::channel();
        let cancel = CancellationToken::new();
        let _guard = CancelOnDrop(cancel.clone(), self.clone());
        {
            let mut s = self.state.lock().unwrap();
            self.admission(&s)?;
            s.sequence += 1;
            let seq = s.sequence;
            s.waiting_chats += 1;
            s.queue.push_back(Work::Chat(Chat {
                sequence: seq,
                target,
                ready: tx,
                cancel,
            }));
            log(
                20,
                LOG,
                format!(
                    "chat request queued: sequence={seq} waiting_chats={}",
                    s.waiting_chats
                ),
            );
        }
        self.notify.notify_waiters();
        rx.await.map_err(|_| anyhow!("broker is shutting down"))?
    }
    pub fn metadata(&self) -> Option<Lease> {
        let mut s = self.state.lock().unwrap();
        if !s.closing
            && s.fatal.is_none()
            && s.recoverable.is_none()
            && s.owner == Some("llm")
            && matches!(s.phase, "llm_ready" | "llm_reserved")
            && (!self.config.kobold_router_mode || s.active_chats == 0)
        {
            s.metadata += 1;
            Some(Lease {
                coordinator: self.clone(),
                kind: "metadata",
            })
        } else {
            None
        }
    }
    pub fn router_control(&self, abort: bool) -> Option<Lease> {
        let mut s = self.state.lock().unwrap();
        if !s.closing
            && ((abort && s.active_chats > 0)
                || (!abort
                    && s.active_chats == 0
                    && matches!(
                        s.phase,
                        "awaiting_request"
                            | "llm_reserved"
                            | "llm_ready"
                            | "comfy_ready"
                            | "image_active"
                    )))
        {
            s.metadata += 1;
            Some(Lease {
                coordinator: self.clone(),
                kind: "metadata",
            })
        } else {
            None
        }
    }
    pub fn control(&self) -> Result<Lease> {
        let mut s = self.state.lock().unwrap();
        self.admission(&s)?;
        if s.owner != Some("comfy") || !matches!(s.phase, "comfy_ready" | "image_active") {
            bail!("ComfyUI does not currently own the GPU")
        };
        s.controls += 1;
        log(
            10,
            LOG,
            format!(
                "ComfyUI control lease granted: active_controls={}",
                s.controls
            ),
        );
        Ok(Lease {
            coordinator: self.clone(),
            kind: "control",
        })
    }
    pub async fn submit(
        &self,
        path: String,
        query: String,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Result<Buffered> {
        let (tx, rx) = oneshot::channel();
        let cancel = CancellationToken::new();
        let _guard = CancelOnDrop(cancel.clone(), self.clone());
        {
            let mut s = self.state.lock().unwrap();
            self.admission(&s)?;
            if body.len() > self.config.max_workflow_body_bytes {
                bail!("workflow request body exceeds the configured size limit")
            }
            if s.waiting_images + 1 > self.config.max_queued_images {
                bail!("workflow queue has reached its request limit")
            }
            if s.bytes + body.len() > self.config.max_queued_workflow_bytes {
                bail!("workflow queue has reached its memory limit")
            }
            s.sequence += 1;
            let sequence = s.sequence;
            s.waiting_images += 1;
            s.bytes += body.len();
            s.queue.push_back(Work::Image(Image {
                sequence,
                path,
                query,
                headers,
                body,
                submitted: tx,
                completion: None,
                cancel,
            }));
            log(
                20,
                LOG,
                format!(
                    "image request queued: sequence={sequence} waiting_images={}",
                    s.waiting_images
                ),
            );
        }
        self.notify.notify_waiters();
        rx.await.map_err(|_| anyhow!("broker is shutting down"))?
    }
    pub async fn wait_history(&self, ids: &[String]) {
        loop {
            let notified = self.notify.notified();
            if !self
                .state
                .lock()
                .unwrap()
                .active_prompt
                .as_ref()
                .is_some_and(|id| ids.contains(id))
            {
                return;
            }
            notified.await;
        }
    }
    pub fn status(&self) -> Value {
        let s = self.state.lock().unwrap();
        let age = s.changed.elapsed().as_secs_f64();
        let stall = match s.phase {
            "verifying_llm" | "reloading_llm" => Some(self.config.reload_timeout),
            "draining_llm" => Some(self.config.chat_drain_timeout),
            "unloading_llm" => Some(self.config.unload_timeout),
            "image_active" => Some(self.config.image_timeout + self.config.request_timeout),
            "cleaning_comfy" => Some(self.config.cleanup_timeout * 2.0),
            _ => None,
        };
        let stalled = stall.is_some_and(|v| age > v + 5.0);
        let healthy = s.fatal.is_none()
            && s.recoverable.is_none()
            && !s.closing
            && s.dispatcher_alive
            && !stalled;
        json!({"responding":true,"ready":healthy,"state":s.phase,"state_age_seconds":(age*1000.0).round()/1000.0,"state_stalled":stalled,"healthy":healthy,"dispatcher_alive":s.dispatcher_alive,"recovering":s.recovering,"recovery_attempts":s.attempts,"recovery_retry_seconds":null,"gpu_owner":s.owner,"active_chats":s.active_chats,"active_llm_metadata":s.metadata,"waiting_chats":s.waiting_chats,"waiting_images":s.waiting_images,"queued_workflow_bytes":s.bytes,"max_queued_images":self.config.max_queued_images,"max_queued_workflow_bytes":self.config.max_queued_workflow_bytes,"active_comfy_controls":s.controls,"active_prompt_id":s.active_prompt,"last_error":s.last_error,"chat_available":s.fatal.is_none() && !s.closing,"llm_backend":self.config.llm_backend,"kobold_router_mode":self.config.kobold_router_mode,"idle_timeout":self.config.idle_timeout,"idle_restore_enabled":false,"idle_restore_scheduled":false,"comfy_route_policy":if self.config.allow_unknown_comfy_routes{"transparent"}else{"strict"}})
    }
    pub async fn request_failed(&self, target: Option<&Target>) {
        self.backends.lock().await.failed(target);
        self.recoverable("KoboldCpp router request did not complete reliably");
    }
    fn recoverable(&self, error: &str) {
        self.state.lock().unwrap().recoverable = Some(error.into());
        self.phase("error");
        self.notify.notify_waiters();
    }
    async fn drain(&self, cancel: &CancellationToken, timed: bool) -> Result<()> {
        let wait = async {
            loop {
                let n = self.notify.notified();
                if cancel.is_cancelled() {
                    bail!("image request was cancelled")
                }
                {
                    let s = self.state.lock().unwrap();
                    if s.active_chats == 0 && s.metadata == 0 {
                        return Ok(());
                    }
                }
                tokio::select! {_=n=>{},_=cancel.cancelled()=>bail!("image request was cancelled")}
            }
        };
        if timed {
            timeout(
                Duration::from_secs_f64(self.config.chat_drain_timeout),
                wait,
            )
            .await
            .map_err(|_| anyhow!("TimeoutError"))?
        } else {
            wait.await
        }
    }
    async fn dispatch(self) {
        let mut restore = self.backends.lock().await.active.clone();
        let mut validated = self.config.llm_backend == "ollama" && restore.is_some();
        loop {
            let notified = self.notify.notified();
            let work = {
                let mut s = self.state.lock().unwrap();
                if s.closing || s.fatal.is_some() {
                    None
                } else {
                    let w = s.queue.pop_front();
                    if let Some(w) = &w {
                        match w {
                            Work::Chat(_) => s.waiting_chats -= 1,
                            Work::Image(w) => {
                                s.waiting_images -= 1;
                                s.bytes -= w.body.len();
                            }
                            Work::Imagine(w) => {
                                s.waiting_images -= 1;
                                s.bytes -= w.bytes;
                            }
                        }
                    }
                    w
                }
            };
            if let Some(work) = work {
                let operation = async {
                    match work {
                        Work::Chat(w) => self.dispatch_chat(w, &mut restore, &mut validated).await,
                        Work::Image(w) => {
                            self.dispatch_image(w, &mut restore, &mut validated).await
                        }
                        Work::Imagine(w) => {
                            self.dispatch_imagine(w, &mut restore, &mut validated).await
                        }
                    }
                };
                tokio::select! {_=self.stop.cancelled()=>break,_=operation=>{}}
            } else {
                if self.state.lock().unwrap().closing || self.state.lock().unwrap().fatal.is_some()
                {
                    break;
                }
                tokio::select! {_=self.stop.cancelled()=>break,_=notified=>{}}
            }
        }
        self.state.lock().unwrap().dispatcher_alive = false;
    }
    async fn dispatch_chat(&self, w: Chat, restore: &mut Option<Target>, validated: &mut bool) {
        if w.cancel.is_cancelled() {
            return;
        }
        if self.config.kobold_router_mode
            || matches!(
                self.config.llm_backend.as_str(),
                "prism" | "all" | "sc117" | "orca" | "sc117_128k" | "orca_128k"
            )
        {
            if self.drain(&w.cancel, false).await.is_err() {
                return;
            }
            self.phase("draining_llm");
        }
        let need = {
            let s = self.state.lock().unwrap();
            s.owner != Some("llm")
                || s.recoverable.is_some()
                || (w.target.is_some() && &w.target != restore)
        };
        if need && let Err(e) = self.restore_llm(w.target.clone(), restore, validated).await {
            let _ = w.ready.send(Err(e));
            return;
        }
        if w.cancel.is_cancelled() {
            return;
        }
        if self.config.kobold_router_mode || self.config.llm_backend == "all" {
            self.phase("llm_reserved")
        } else if matches!(
            self.config.llm_backend.as_str(),
            "prism" | "sc117" | "orca" | "sc117_128k" | "orca_128k"
        ) {
            self.phase("llm_ready")
        }
        {
            let mut s = self.state.lock().unwrap();
            s.active_chats += 1;
            log(
                10,
                LOG,
                format!(
                    "chat lease granted: sequence={} active_chats={}",
                    w.sequence, s.active_chats
                ),
            );
        }
        let _ = w.ready.send(Ok(Lease {
            coordinator: self.clone(),
            kind: "chat",
        }));
    }
    async fn restore_llm(
        &self,
        target: Option<Target>,
        restore: &mut Option<Target>,
        validated: &mut bool,
    ) -> Result<()> {
        {
            let mut s = self.state.lock().unwrap();
            s.recovering = true;
            s.attempts += 1;
        }
        let cleanup = self.state.lock().unwrap().owner != Some("llm") || target.is_none();
        if cleanup {
            if let Err(e) = self.yield_imagine().await {
                self.recoverable("Imagine GPU release is not confirmed");
                return Err(e);
            }
            self.phase("cleaning_comfy");
            let wait = timeout(
                Duration::from_secs_f64(self.config.cleanup_timeout),
                async {
                    loop {
                        let n = self.notify.notified();
                        if self.state.lock().unwrap().controls == 0 {
                            break;
                        }
                        n.await
                    }
                },
            )
            .await;
            let result = if wait.is_err() {
                Err(anyhow!(
                    "timed out waiting for ComfyUI control requests to finish"
                ))
            } else {
                self.comfy.free().await
            };
            if let Err(e) = result {
                log(
                    40,
                    LOG,
                    format!("ComfyUI cleanup failed; GPU release is unconfirmed: {e}"),
                );
                {
                    let mut s = self.state.lock().unwrap();
                    s.last_error = Some(e.to_string());
                    s.recovering = false;
                }
                self.recoverable("ComfyUI GPU release is not confirmed");
                bail!("ComfyUI GPU release is not confirmed")
            }
        }
        self.state.lock().unwrap().owner = None;
        let result = async {
            let mut b = self.backends.lock().await;
            if !*validated {
                b.validate().await?;
                *validated = true;
            }
            let t = if target.is_some() {
                self.phase("reloading_llm");
                b.acquire(target).await?
            } else if self.config.kobold_router_mode {
                if self.state.lock().unwrap().recoverable.is_some()
                    && let Some(r) = restore
                {
                    b.release(r).await?;
                }
                b.acquire(None).await?
            } else {
                let skip = self.config.llm_backend == "tabbyapi"
                    && self.state.lock().unwrap().recoverable.is_some();
                let observed = if restore.is_none() && self.config.llm_backend != "ollama" && !skip
                {
                    b.observe().await?
                } else {
                    None
                };
                if let Some(t) = observed {
                    t
                } else {
                    self.phase("reloading_llm");
                    b.acquire(restore.clone()).await?
                }
            };
            *restore = Some(t);
            Ok::<_, anyhow::Error>(())
        }
        .await;
        if let Err(e) = result {
            log(
                40,
                LOG,
                format!("{} reload failed: {e}", label(&self.config.llm_backend)),
            );
            let error = if e.to_string().contains("insufficient_vram") {
                e.to_string()
            } else {
                format!("{} is not confirmed ready", label(&self.config.llm_backend))
            };
            {
                let mut s = self.state.lock().unwrap();
                s.last_error = Some(e.to_string());
                s.recovering = false;
            }
            self.recoverable(&error);
            bail!("{error}")
        }
        {
            let mut s = self.state.lock().unwrap();
            s.owner = Some("llm");
            s.recovering = false;
            s.fatal = None;
            s.recoverable = None;
            s.last_error = None;
        }
        self.phase(
            if self.config.kobold_router_mode || self.config.llm_backend == "all" {
                "llm_reserved"
            } else {
                "llm_ready"
            },
        );
        log(
            20,
            LOG,
            format!(
                "GPU ownership transferred: owner={}",
                label(&self.config.llm_backend)
            ),
        );
        Ok(())
    }
    async fn activate_comfy(
        &self,
        w: &Image,
        restore: &mut Option<Target>,
        validated: &mut bool,
    ) -> Result<()> {
        self.phase("draining_llm");
        {
            let s = self.state.lock().unwrap();
            log(
                20,
                LOG,
                format!(
                    "draining LLM requests: active_chats={} active_metadata={}",
                    s.active_chats, s.metadata
                ),
            );
        }
        self.drain(&w.cancel, true).await?;
        log(20, LOG, "LLM requests drained");
        self.yield_imagine().await?;
        let mut b = self.backends.lock().await;
        if self.config.kobold_router_mode {
            if !*validated {
                b.validate().await?;
                *validated = true
            }
            if restore.is_none() {
                *restore = Some(b.acquire(None).await?)
            }
            self.state.lock().unwrap().owner = None;
            self.phase("unloading_llm");
            b.release(restore.as_mut().unwrap()).await?;
        } else {
            if self.state.lock().unwrap().owner.is_none() && self.config.llm_backend != "ollama" {
                if !*validated {
                    b.validate().await?;
                    *validated = true
                }
                if let Some(t) = b.observe().await? {
                    *restore = Some(t);
                    self.state.lock().unwrap().owner = Some("llm");
                }
            }
            if self.state.lock().unwrap().owner == Some("llm") {
                self.state.lock().unwrap().owner = None;
                self.phase("unloading_llm");
                b.release(
                    restore
                        .as_mut()
                        .ok_or_else(|| anyhow!("LLM restore point is unavailable"))?,
                )
                .await?;
            }
        }
        {
            let mut s = self.state.lock().unwrap();
            s.owner = Some("comfy");
            s.recoverable = None;
        }
        self.phase("comfy_ready");
        if !self.config.kobold_router_mode {
            log(20, LOG, "GPU ownership transferred: owner=ComfyUI");
        }
        Ok(())
    }
    async fn dispatch_image(
        &self,
        mut w: Image,
        restore: &mut Option<Target>,
        validated: &mut bool,
    ) {
        self.state.lock().unwrap().last_error = None;
        let mut submitted = Some(w.submitted);
        w.submitted = oneshot::channel().0;
        let result = async {
            if w.cancel.is_cancelled() {
                bail!("image request was cancelled")
            }
            if self.state.lock().unwrap().owner != Some("comfy") {
                self.activate_comfy(&w, restore, validated).await?;
            }
            if w.cancel.is_cancelled() {
                bail!("image request was cancelled")
            }
            self.phase("image_active");
            let url = format!(
                "{}{}{}{}",
                self.config.comfy_url,
                w.path,
                if w.query.is_empty() { "" } else { "?" },
                w.query
            );
            let permit = self.comfy.limit.acquire().await?;
            let r = self
                .comfy
                .client
                .post(url)
                .headers(std::mem::take(&mut w.headers))
                .body(std::mem::take(&mut w.body))
                .timeout(Duration::from_secs_f64(self.config.request_timeout))
                .send()
                .await?;
            let status = r.status().as_u16();
            let headers = r.headers().clone();
            let body = r.bytes().await?.to_vec();
            drop(permit);
            let id = serde_json::from_slice::<Value>(&body).ok().and_then(|v| {
                v.get("prompt_id").filter(|v| !v.is_null()).map(|v| {
                    v.as_str()
                        .map(str::to_string)
                        .unwrap_or_else(|| v.to_string())
                })
            });
            if let Some(tx) = submitted.take() {
                let _ = tx.send(Ok(Buffered {
                    status,
                    headers,
                    body,
                }));
            }
            if status >= 400 {
                bail!("ComfyUI rejected prompt with HTTP {status}")
            }
            let id = id
                .filter(|v| !v.is_empty())
                .ok_or_else(|| anyhow!("ComfyUI response did not contain a prompt_id"))?;
            self.state.lock().unwrap().active_prompt = Some(id.clone());
            log(
                20,
                LOG,
                format!("ComfyUI image job started: prompt_id={id}"),
            );
            let outputs = self.comfy.wait_prompt(&id).await?;
            log(
                20,
                LOG,
                format!("ComfyUI image job completed: prompt_id={id}"),
            );
            self.phase("comfy_ready");
            Ok::<_, anyhow::Error>(outputs)
        }
        .await;
        match result {
            Ok(v) => {
                if let Some(tx) = w.completion {
                    let _ = tx.send(Ok(v));
                }
            }
            Err(e) => {
                log(40, LOG, format!("image handoff failed: {e}"));
                let error = e.to_string();
                {
                    let mut s = self.state.lock().unwrap();
                    s.last_error = Some(error.clone());
                }
                let owner = self.state.lock().unwrap().owner;
                match owner {
                    Some("comfy") => self.phase("comfy_ready"),
                    Some("llm") => self.phase("llm_ready"),
                    _ => self.recoverable(
                        "GPU ownership is not confirmed after workflow handoff failure",
                    ),
                }
                if let Some(tx) = submitted {
                    let _ = tx.send(Err(anyhow!(error.clone())));
                }
                if let Some(tx) = w.completion {
                    let _ = tx.send(Err(anyhow!(error)));
                }
            }
        }
        self.state.lock().unwrap().active_prompt = None;
        self.notify.notify_waiters();
    }

    pub async fn listen_imagine(&self) -> Result<Option<JoinHandle<()>>> {
        use std::os::unix::fs::PermissionsExt;
        use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
        let Some(path) = &self.config.coordinator_socket else {
            return Ok(None);
        };
        let listener = tokio::net::UnixListener::bind(path)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        let this = self.clone();
        let socket = path.clone();
        Ok(Some(tokio::spawn(async move {
            loop {
                let stream = tokio::select! {_=this.stop.cancelled()=>break,v=listener.accept()=>match v{Ok((s,_))=>s,Err(_)=>break}};
                let this = this.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stream);
                    let mut line = String::new();
                    let mut limited = (&mut reader).take(65536);
                    let read = tokio::select! {
                        _ = this.stop.cancelled() => return,
                        read = timeout(Duration::from_secs(30), limited.read_line(&mut line)) => read,
                    };
                    if !matches!(read, Ok(Ok(1..))) || !line.ends_with('\n') {
                        return;
                    }
                    let Ok(v) = serde_json::from_str::<Value>(&line) else {
                        return;
                    };
                    let stream = reader.into_inner();
                    if let Some(token) = v["ticket"].as_str() {
                        let sender = this.registrations.lock().unwrap().remove(token);
                        if let Some((sender, cancel)) = sender
                            && stream.try_write(b"{\"queued\":true}\n").is_ok()
                        {
                            match this.watch_imagine(stream, cancel.clone()) {
                                Ok(stream) => {
                                    let _ = sender.send(stream);
                                }
                                Err(_) => {
                                    cancel.cancel();
                                    this.remove_cancelled();
                                }
                            }
                        }
                        return;
                    }
                    let batch = v["jobs"].is_array();
                    let jobs = if let Some(jobs) = v["jobs"].as_array() {
                        jobs.clone()
                    } else {
                        vec![v]
                    };
                    if jobs.is_empty() {
                        return;
                    }
                    let sizes: Option<Vec<usize>> = jobs
                        .iter()
                        .map(|job| {
                            job["job"]
                                .as_str()
                                .and_then(|_| job["bytes"].as_u64())
                                .and_then(|v| usize::try_from(v).ok())
                        })
                        .collect();
                    let Some(sizes) = sizes else { return };
                    let Some(total) = sizes.iter().try_fold(0usize, |a, b| a.checked_add(*b))
                    else {
                        return;
                    };
                    let cancel = CancellationToken::new();
                    let mut tokens = Vec::new();
                    let mut single = None;
                    {
                        let mut s = this.state.lock().unwrap();
                        if this.admission(&s).is_err()
                            || sizes
                                .iter()
                                .any(|b| *b > this.config.max_workflow_body_bytes)
                            || s.waiting_images + jobs.len() > this.config.max_queued_images
                            || s.bytes.saturating_add(total) > this.config.max_queued_workflow_bytes
                        {
                            let _ = stream.try_write(b"{\"queued\":false}\n");
                            return;
                        }
                        for bytes in sizes {
                            let (tx, rx) = oneshot::channel();
                            let job_cancel = cancel.child_token();
                            if batch {
                                let token = uuid::Uuid::new_v4().simple().to_string();
                                this.registrations
                                    .lock()
                                    .unwrap()
                                    .insert(token.clone(), (tx, job_cancel.clone()));
                                tokens.push(token);
                            } else {
                                single = Some(tx)
                            }
                            s.sequence += 1;
                            s.waiting_images += 1;
                            s.bytes += bytes;
                            s.queue.push_back(Work::Imagine(Imagine {
                                stream: rx,
                                bytes,
                                cancel: job_cancel,
                            }));
                        }
                        let ack = format!("{}\n", json!({"queued":true,"tickets":tokens}));
                        if stream.try_write(ack.as_bytes()).ok() != Some(ack.len()) {
                            cancel.cancel();
                        }
                    }
                    this.remove_cancelled();
                    this.notify.notify_waiters();
                    if let Some(tx) = single {
                        match this.watch_imagine(stream, cancel.clone()) {
                            Ok(stream) => {
                                let _ = tx.send(stream);
                            }
                            Err(_) => {
                                cancel.cancel();
                                this.remove_cancelled();
                            }
                        }
                        return;
                    }
                    // The allocation connection pins the batch until all per-job sockets
                    // have attached. Losing it removes every unattached queue position.
                    let mut reader = BufReader::new(stream);
                    let mut line = String::new();
                    tokio::select! {_=this.stop.cancelled()=>{},_=reader.read_line(&mut line)=>{}}
                    {
                        let mut pending = this.registrations.lock().unwrap();
                        for token in tokens {
                            if let Some((_, cancel)) = pending.remove(&token) {
                                cancel.cancel();
                            }
                        }
                    }
                    this.remove_cancelled();
                });
            }
            let _ = std::fs::remove_file(socket);
        })))
    }
    fn watch_imagine(
        &self,
        stream: tokio::net::UnixStream,
        cancel: CancellationToken,
    ) -> Result<ImagineStream> {
        use std::os::fd::AsFd;
        let duplicate = std::os::unix::net::UnixStream::from(stream.as_fd().try_clone_to_owned()?);
        let duplicate = tokio::net::UnixStream::from_std(duplicate)?;
        let this = self.clone();
        let watcher = tokio::spawn(async move {
            loop {
                if duplicate.readable().await.is_err() {
                    break;
                }
                match duplicate.try_read(&mut [0u8]) {
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    _ => break,
                }
            }
            cancel.cancel();
            this.remove_cancelled();
        });
        Ok(ImagineStream { stream, watcher })
    }
    async fn yield_imagine(&self) -> Result<()> {
        if self.config.coordinator_socket.is_none() {
            return Ok(());
        }
        let access: Value = serde_json::from_slice(
            &tokio::fs::read(self.config.imagine_access_file.as_ref().unwrap()).await?,
        )?;
        let token = access["admin_token"]
            .as_str()
            .ok_or_else(|| anyhow!("Imagine credentials unavailable"))?;
        let _permit = self.comfy.limit.acquire().await?;
        let r = self
            .comfy
            .client
            .post(format!("{}/v1/runtime/yield", self.config.imagine_url))
            .bearer_auth(token)
            .timeout(Duration::from_secs_f64(self.config.unload_timeout))
            .send()
            .await?;
        if !r.status().is_success() || r.json::<Value>().await?["released"] != true {
            bail!("Imagine GPU release is not confirmed")
        }
        Ok(())
    }
    async fn dispatch_imagine(
        &self,
        w: Imagine,
        restore: &mut Option<Target>,
        validated: &mut bool,
    ) {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let mut connection = match w.stream.await {
            Ok(stream) => stream,
            Err(_) => return,
        };
        if w.cancel.is_cancelled() {
            return;
        }
        // A ticket cancelled before dispatch must not initiate a handoff.
        let mut probe = [0];
        if connection.stream.try_read(&mut probe).is_ok() {
            return;
        }
        let result = async {
            let owner = self.state.lock().unwrap().owner;
            let recovery = self.state.lock().unwrap().recoverable.is_some();
            if owner == Some("imagine") && recovery {
                self.yield_imagine().await?;
            }
            if owner != Some("imagine") {
                self.phase("draining_llm");
                if let Err(error) = self.drain(&w.cancel, true).await {
                    if w.cancel.is_cancelled() {
                        return Ok(());
                    }
                    return Err(error);
                }
                let mut b = self.backends.lock().await;
                if !*validated {
                    b.validate().await?;
                    *validated = true;
                }
                if restore.is_none() {
                    *restore = b.observe().await?;
                }
                if let Some(t) = restore {
                    self.state.lock().unwrap().owner = None;
                    self.phase("unloading_llm");
                    b.release(t).await?;
                }
                drop(b);
                self.phase("cleaning_comfy");
                self.comfy.free().await?;
                self.state.lock().unwrap().owner = Some("imagine");
            }
            if w.cancel.is_cancelled() {
                return Ok(());
            }
            self.phase("image_active");
            connection.watcher.abort();
            let _ = (&mut connection.watcher).await;
            connection.stream.write_all(b"{\"granted\":true}\n").await?;
            let mut reader = BufReader::new(&mut connection.stream);
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            let finished = serde_json::from_str::<Value>(&line)
                .ok()
                .is_some_and(|v| v["finished"] == true);
            if !finished {
                self.yield_imagine().await?;
            }
            self.state.lock().unwrap().recoverable = None;
            self.phase("imagine_ready");
            Ok::<_, anyhow::Error>(())
        }
        .await;
        if let Err(e) = result {
            self.state.lock().unwrap().last_error = Some(e.to_string());
            self.recoverable("Imagine GPU ownership is not confirmed");
        }
    }

    pub async fn close(&self) {
        {
            let mut s = self.state.lock().unwrap();
            if s.closing {
                return;
            }
            s.closing = true;
            s.queue.clear();
            s.waiting_chats = 0;
            s.waiting_images = 0;
            s.bytes = 0;
        }
        self.registrations.lock().unwrap().clear();
        self.phase("shutting_down");
        log(20, LOG, "handoff coordinator shutdown started");
        if self.state.lock().unwrap().active_prompt.is_some() {
            self.comfy.interrupt().await;
        }
        self.stop.cancel();
        self.notify.notify_waiters();
        log(20, LOG, "handoff coordinator shutdown completed");
    }
}
