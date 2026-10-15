use crate::{
    backend::{Backend, Target, kobold_models},
    config::{Config, label},
    coordinator::{Coordinator, Lease},
    http,
    logging::log,
    profiles::Profiles,
    requests,
    routes::{self, Kind},
};
use anyhow::{Result, anyhow};
use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{FromRequestParts, State, ws::WebSocketUpgrade},
    http::{Request, Response},
    routing::any,
};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_util::sync::CancellationToken;
const LOG: &str = "st_proxy.service";
#[derive(Clone)]
pub struct Service {
    pub config: Config,
    pub coordinator: Coordinator,
    pub chat: reqwest::Client,
    pub image: reqwest::Client,
    pub profiles: Profiles,
    catalog_backends: Vec<Backend>,
    router_models: Arc<Mutex<Value>>,
    active_chat: Arc<AtomicUsize>,
    active_image: Arc<AtomicUsize>,
    active_ws: Arc<AtomicUsize>,
    chat_limit: crate::connection_limit::Limit,
    image_limit: crate::connection_limit::Limit,
    websocket_limit: crate::connection_limit::Limit,
}
struct Forward<'a> {
    origin: &'a str,
    path: &'a str,
    query: Option<String>,
    body: Option<Vec<u8>>,
    target: &'a str,
    lease: Option<Lease>,
    router: bool,
    selection: Option<Target>,
}
pub struct Running {
    pub service: Service,
    pub chat_port: u16,
    pub image_port: u16,
    pub mcp_port: Option<u16>,
    mcp: Option<crate::mcp::Mcp>,
    stop: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
}
impl Running {
    pub async fn close(self) {
        if let Some(mcp) = &self.mcp {
            mcp.close().await;
        }
        self.stop.cancel();
        self.service.coordinator.close().await;
        for task in self.tasks {
            task.abort();
            let _ = task.await;
        }
        log(20, LOG, "broker stopped");
    }
}
impl Service {
    pub async fn start(config: Config) -> Result<Running> {
        let profiles = Profiles::load(config.model_profiles.as_deref())?;
        let builder = || {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs_f64(config.connect_timeout))
                .read_timeout(Duration::from_secs_f64(config.request_timeout))
                .redirect(reqwest::redirect::Policy::none())
        };
        let control = builder().pool_max_idle_per_host(0).build()?;
        let chat = builder()
            .pool_idle_timeout(Duration::from_secs_f64(
                15.0_f64.min(config.request_timeout / 2.0),
            ))
            .build()?;
        let image = chat.clone();
        let control_limit = crate::connection_limit::Limit::new(16, config.connect_timeout);
        let catalog_backends =
            crate::backend::Backends::limited(&config, control.clone(), control_limit.clone())
                .entries;
        let router_models = if config.kobold_router_mode {
            if let Some(path) = &config.kobold_model_cache {
                kobold_models(&serde_json::from_slice(&std::fs::read(path)?)?)?
            } else {
                kobold_models(&json!({"data":[{"id":"initial_model"}]}))?
            }
        } else {
            Value::Null
        };
        let chat_listener =
            TcpListener::bind((config.listen_host.as_str(), config.chat_port as u16)).await?;
        let image_listener =
            TcpListener::bind((config.listen_host.as_str(), config.image_port as u16)).await?;
        let chat_port = chat_listener.local_addr()?.port();
        let image_port = image_listener.local_addr()?.port();
        let mcp_listener = if let Some(port) = config.mcp_port {
            let listener = TcpListener::bind(("127.0.0.1", port)).await?;
            let port = listener.local_addr()?.port();
            let mut mcp_config = config.clone();
            mcp_config.mcp_port = Some(port);
            Some((listener, crate::mcp::Mcp::new(mcp_config).await?, port))
        } else {
            None
        };
        let (coordinator, dispatcher) =
            Coordinator::start(config.clone(), control, control_limit).await;
        let this = Self {
            chat_limit: crate::connection_limit::Limit::new(64, config.connect_timeout),
            image_limit: crate::connection_limit::Limit::new(64, config.connect_timeout),
            websocket_limit: crate::connection_limit::Limit::new(32, config.connect_timeout),
            config,
            coordinator,
            chat,
            image,
            profiles,
            catalog_backends,
            router_models: Arc::new(Mutex::new(router_models)),
            active_chat: Default::default(),
            active_image: Default::default(),
            active_ws: Default::default(),
        };
        let stop = CancellationToken::new();
        let mut tasks = vec![dispatcher];
        match this.coordinator.listen_imagine().await {
            Ok(Some(listener)) => tasks.push(listener),
            Ok(None) => {}
            Err(error) => {
                this.coordinator.close().await;
                for task in tasks {
                    task.abort();
                    let _ = task.await;
                }
                return Err(error);
            }
        }
        for (listener, router) in [
            (
                chat_listener,
                Router::new()
                    .fallback(any(chat_handler))
                    .with_state(this.clone()),
            ),
            (
                image_listener,
                Router::new()
                    .fallback(any(image_handler))
                    .with_state(this.clone()),
            ),
        ] {
            let stopped = stop.clone();
            tasks.push(tokio::spawn(async move {
                let _ = axum::serve(listener, router)
                    .with_graceful_shutdown(stopped.cancelled_owned())
                    .await;
            }));
        }
        let mut mcp = None;
        let mut mcp_port = None;
        if let Some((listener, server, port)) = mcp_listener {
            let router = server.router();
            let stopped = stop.clone();
            tasks.push(tokio::spawn(async move {
                let _ = axum::serve(listener, router)
                    .with_graceful_shutdown(stopped.cancelled_owned())
                    .await;
            }));
            log(
                20,
                "st_proxy.mcp_server",
                format!("MCP ready: http://127.0.0.1:{port}/mcp"),
            );
            mcp = Some(server);
            mcp_port = Some(port);
        }
        Ok(Running {
            service: this,
            chat_port,
            image_port,
            mcp,
            mcp_port,
            stop,
            tasks,
        })
    }
    pub async fn status(&self) -> Response<Body> {
        let mut s = self.coordinator.status();
        for (k, v) in [
            (
                "active_chat_requests",
                self.active_chat.load(Ordering::Relaxed),
            ),
            (
                "active_image_requests",
                self.active_image.load(Ordering::Relaxed),
            ),
            ("active_websockets", self.active_ws.load(Ordering::Relaxed)),
            ("control_connection_limit", 16),
            ("chat_connection_limit", 64),
            ("image_connection_limit", 64),
            ("websocket_connection_limit", 32),
        ] {
            s[k] = v.into();
        }
        if self.config.llm_backend == "all" {
            let b = self.coordinator.backends.lock().await;
            s["active_llm_backend"] = b
                .active
                .as_ref()
                .map(|t| json!(t.engine))
                .unwrap_or(Value::Null);
            s["active_model"] = b
                .active
                .as_ref()
                .map(|t| json!(format!("{}/{}", t.engine, t.model)))
                .unwrap_or(Value::Null);
        }
        http::json(200, s)
    }
    async fn catalog(&self, engine: Option<&str>) -> Result<Value> {
        let mut cards = Vec::new();
        for b in &self.catalog_backends {
            if engine.is_some_and(|e| e != b.kind) {
                continue;
            }
            let v = b.catalog().await?;
            for c in v["data"].as_array().unwrap() {
                let mut c = c.clone();
                c["id"] = format!("{}/{}", b.kind, c["id"].as_str().unwrap()).into();
                c["owned_by"] = b.kind.clone().into();
                cards.push(c);
            }
        }
        Ok(json!({"object":"list","data":cards}))
    }
    async fn target(&self, model: Option<&str>) -> Result<Target> {
        if self.config.llm_backend != "all" {
            return self.catalog_backends[0].target(model).await;
        }
        let model =
            model.ok_or_else(|| anyhow!("select a model ID from /v1/models (backend/model)"))?;
        let (e, m) = model
            .split_once('/')
            .ok_or_else(|| anyhow!("select a model ID from /v1/models (backend/model)"))?;
        if m.is_empty() || !self.catalog_backends.iter().any(|b| b.kind == e) {
            return Err(anyhow!("unknown backend/model"));
        }
        if !self.catalog(Some(e)).await?["data"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["id"] == model)
        {
            return Err(anyhow!("unknown backend/model"));
        }
        Ok(Target::new(e, m))
    }
    async fn forward(
        &self,
        request: Request<Body>,
        options: Forward<'_>,
    ) -> Result<Response<Body>> {
        let Forward {
            origin,
            path,
            query,
            body,
            target,
            lease,
            router,
            selection,
        } = options;
        let path = if self.config.llm_backend != "all"
            && !self.config.kobold_router_mode
            && path == decoded_path(request.uri().path()).trim_end_matches('/')
        {
            request.uri().path().to_string()
        } else {
            path.to_string()
        };
        http::proxy(
            request,
            self.chat.clone(),
            http::Upstream {
                origin: origin.into(),
                path,
                query,
                body,
                target: target.into(),
                lease,
                limit: self.chat_limit.clone(),
                on_failure: if router {
                    Some((self.coordinator.clone(), selection))
                } else {
                    None
                },
            },
        )
        .await
    }
    async fn chat(&self, mut req: Request<Body>) -> Result<Response<Body>> {
        let path = decoded_path(req.uri().path())
            .trim_end_matches('/')
            .to_string();
        let method = req.method().to_string();
        let kind = self.config.llm_backend.as_str();
        let origin = self.config.llm_url.clone();
        let qs: Vec<(String, String)> = req
            .uri()
            .query()
            .map(|q| {
                url::form_urlencoded::parse(q.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_default();
        let query_model = qs
            .iter()
            .find(|(k, _)| k == "model")
            .map(|(_, v)| v.as_str());
        let safe = matches!(method.as_str(), "GET" | "HEAD");
        let managed = kind == "prism" || self.catalog_backends[0].strata();
        if kind == "all" || managed {
            if qs.iter().any(|(k, _)| k == "autoload" || k == "reload")
                || path.starts_with("/models/")
                || (kind == "all"
                    && (path.starts_with("/api/admin/")
                        || path.starts_with("/v1/model/")
                        || path == "/noscript"))
                || (managed
                    && ["/models", "/v1/models", "/props"].contains(&path.as_str())
                    && !safe)
            {
                return Ok(http::text(403, "model lifecycle is owned by the broker"));
            }
            if kind == "all" && safe && ["/v1/models", "/models"].contains(&path.as_str()) {
                return Ok(http::json(200, self.catalog(None).await?));
            }
            if kind == "all" && safe && path == "/health" {
                return Ok(self.status().await);
            }
            let strata = self.catalog_backends[0].strata();
            if managed
                && strata
                && !((method == "POST" && requests::STRATA_PATHS.contains(&path.as_str()))
                    || (safe
                        && [
                            "/health",
                            "/api/health",
                            "/models",
                            "/v1/models",
                            "/v1/status",
                            "/props",
                            "/slots",
                        ]
                        .contains(&path.as_str())))
            {
                return Ok(http::text(
                    403,
                    "unsupported request or broker-owned lifecycle",
                ));
            }
            if managed
                && safe
                && [
                    "/health",
                    "/api/health",
                    "/models",
                    "/v1/models",
                    "/v1/status",
                ]
                .contains(&path.as_str())
            {
                return self
                    .forward(
                        req,
                        Forward {
                            origin: &origin,
                            path: &path,
                            query: None,
                            body: None,
                            target: label(kind),
                            lease: None,
                            router: false,
                            selection: None,
                        },
                    )
                    .await;
            }
            if managed && safe && ["/props", "/metrics", "/slots"].contains(&path.as_str()) {
                let lease = self.coordinator.metadata();
                let active = if lease.is_some() {
                    self.coordinator.backends.lock().await.entries[0]
                        .observe()
                        .await?
                } else {
                    None
                };
                let Some(active) = active.filter(|a| query_model.is_none_or(|m| m == a.model))
                else {
                    return Ok(http::json(
                        503,
                        json!({"error":format!("{} model is not active",label(kind))}),
                    ));
                };
                let query = with_model(&qs, &active.model);
                return self
                    .forward(
                        req,
                        Forward {
                            origin: &origin,
                            path: &path,
                            query: Some(query),
                            body: None,
                            target: label(kind),
                            lease,
                            router: false,
                            selection: None,
                        },
                    )
                    .await;
            }
            let mut body = match read_body(&mut req, self.config.max_chat_body_bytes).await {
                Ok(v) => v,
                Err(r) => return Ok(r),
            };
            let mut payload = if body.is_empty() {
                Value::Null
            } else {
                match serde_json::from_slice::<Value>(&body) {
                    Ok(v) if v.is_object() => v,
                    _ => return Ok(http::text(400, "expected a JSON object")),
                }
            };
            if !payload["model"].is_null() && !payload["model"].is_string() {
                let message = if kind == "all" {
                    "select a model ID from /v1/models (backend/model)"
                } else if kind == "prism" {
                    "model must be a nonempty Prism preset ID"
                } else {
                    "unknown Strata model"
                };
                return Ok(http::text(400, message));
            }
            let model = if payload.is_null() {
                query_model
            } else {
                payload["model"].as_str()
            };
            if !payload.is_null() && query_model.is_some() && query_model != model {
                return Ok(http::text(400, "conflicting body and query model IDs"));
            }
            let target = match self.target(model).await {
                Ok(t) => t,
                Err(e) => return Ok(http::text(400, e.to_string())),
            };
            let engine = target.engine.as_str();
            let backend = self
                .catalog_backends
                .iter()
                .find(|b| b.kind == engine)
                .unwrap();
            if !payload.is_null() && (kind == "all" || payload["model"].is_null()) {
                payload["model"] = target.model.clone().into();
                body = crate::json_wire::dumps(&payload, kind != "all", false).into_bytes();
            }
            let query = with_model(&qs, &target.model);
            if kind == "all"
                && safe
                && ["/props", "/metrics", "/slots", "/v1/model", "/api/v1/model"]
                    .contains(&path.as_str())
            {
                let lease = self.coordinator.metadata();
                let active = self.coordinator.backends.lock().await.active.clone();
                if lease.is_none() || active.as_ref() != Some(&target) {
                    return Ok(http::json(
                        503,
                        json!({"error":"requested model is not active"}),
                    ));
                }
                return self
                    .forward(
                        req,
                        Forward {
                            origin: &backend.origin,
                            path: &path,
                            query: Some(query),
                            body: Some(body),
                            target: backend.label(),
                            lease,
                            router: false,
                            selection: None,
                        },
                    )
                    .await;
            }
            let active = if backend.strata() {
                requests::STRATA_PATHS.contains(&path.as_str())
            } else {
                requests::TEXT_PATHS.contains(&path.as_str())
                    || (engine == "prism"
                        && [
                            "/completion",
                            "/tokenize",
                            "/detokenize",
                            "/apply-template",
                            "/v1/embeddings",
                            "/embeddings",
                            "/rerank",
                            "/v1/rerank",
                        ]
                        .contains(&path.as_str()))
            };
            if kind == "all" && (method != "POST" || !active) {
                return Ok(http::text(
                    403,
                    "unsupported model request in all-backend mode",
                ));
            }
            if kind == "all" && payload.is_null() {
                return Ok(http::text(400, "expected a JSON object with model"));
            }
            if method == "POST" && path == "/v1/chat/completions" && !payload.is_null() {
                let profiled = self.profiles.apply(&payload, engine, Some(&target.model));
                if profiled != payload {
                    payload = profiled;
                    body = crate::json_wire::dumps(&payload, false, false).into_bytes();
                }
            }
            if engine == "koboldcpp" {
                body = requests::default_model(&body, &self.config.kobold_default_model)?;
            }
            if engine == "tabbyapi" && path == "/v1/chat/completions" {
                body = requests::tabby(&body)
            }
            if engine == "prism" && path == "/v1/chat/completions" && !payload.is_null() {
                body = requests::request_usage(payload)
            }
            if body.len() > self.config.max_chat_body_bytes {
                return Ok(http::text(413, "Maximum request body size exceeded"));
            }
            let lease = match self.coordinator.chat(Some(target.clone())).await {
                Ok(l) => l,
                Err(e) => return Ok(http::json(503, json!({"error":e.to_string()}))),
            };
            let upstream_path = if engine == "koboldcpp" {
                requests::kobold_path(&path)
            } else {
                &path
            };
            return self
                .forward(
                    req,
                    Forward {
                        origin: &backend.origin,
                        path: upstream_path,
                        query: Some(query),
                        body: Some(body),
                        target: backend.label(),
                        lease: Some(lease),
                        router: engine == "koboldcpp",
                        selection: Some(target),
                    },
                )
                .await;
        }
        if self.config.kobold_router_mode {
            if method == "GET" && ["/v1/models", "/api/v1/model"].contains(&path.as_str()) {
                let lease = self.coordinator.metadata();
                if path == "/v1/models" {
                    if lease.is_some() {
                        match self.catalog_backends[0].catalog().await {
                            Ok(v) => *self.router_models.lock().unwrap() = v,
                            Err(_) => log(
                                40,
                                LOG,
                                "router model discovery failed; retaining cached list",
                            ),
                        }
                    }
                    return Ok(http::json(200, self.router_models.lock().unwrap().clone()));
                }
                if lease.is_none() {
                    return Ok(inactive(&path, kind));
                }
                return self
                    .forward(
                        req,
                        Forward {
                            origin: &origin,
                            path: &path,
                            query: None,
                            body: None,
                            target: label(kind),
                            lease,
                            router: false,
                            selection: None,
                        },
                    )
                    .await;
            }
            if path.starts_with("/api/admin/reload_config") || path == "/noscript" {
                return Ok(http::text(403, "model lifecycle is owned by the broker"));
            }
            let abort = method == "POST" && path == "/api/extra/abort";
            if safe || method == "OPTIONS" || abort {
                let lease = self.coordinator.router_control(abort);
                if lease.is_none() {
                    return Ok(http::json(
                        503,
                        json!({"error":"router control is unavailable during GPU transition"}),
                    ));
                }
                let body = match read_body(&mut req, self.config.max_chat_body_bytes).await {
                    Ok(v) => v,
                    Err(r) => return Ok(r),
                };
                return self
                    .forward(
                        req,
                        Forward {
                            origin: &origin,
                            path: &path,
                            query: None,
                            body: Some(body),
                            target: label(kind),
                            lease,
                            router: false,
                            selection: None,
                        },
                    )
                    .await;
            }
        }
        if kind == "tabbyapi"
            && safe
            && ["/v1/models", "/v1/model/list", "/health"].contains(&path.as_str())
        {
            return self
                .forward(
                    req,
                    Forward {
                        origin: &origin,
                        path: &path,
                        query: None,
                        body: None,
                        target: label(kind),
                        lease: None,
                        router: false,
                        selection: None,
                    },
                )
                .await;
        }
        let passive =
            (kind == "tabbyapi" && safe && ["/v1/model", "/props"].contains(&path.as_str()))
                || (kind == "koboldcpp"
                    && method == "GET"
                    && ["/api/v1/model", "/v1/models"].contains(&path.as_str()));
        if passive {
            log(
                10,
                LOG,
                format!("passive LLM metadata request bypasses GPU lease: path={path}"),
            );
            let lease = self.coordinator.metadata();
            if lease.is_none() {
                return Ok(inactive(&path, kind));
            }
            return self
                .forward(
                    req,
                    Forward {
                        origin: &origin,
                        path: &path,
                        query: None,
                        body: None,
                        target: label(kind),
                        lease,
                        router: false,
                        selection: None,
                    },
                )
                .await;
        }
        let lease = match self.coordinator.chat(None).await {
            Ok(l) => l,
            Err(e) => return Ok(http::json(503, json!({"error":e.to_string()}))),
        };
        let mut body = None;
        let mut upstream_path = path.clone();
        if self.config.kobold_router_mode
            || (kind == "tabbyapi" && method == "POST" && path == "/v1/chat/completions")
        {
            let mut bytes = match read_body(&mut req, self.config.max_chat_body_bytes).await {
                Ok(v) => v,
                Err(r) => return Ok(r),
            };
            if self.config.kobold_router_mode
                && method == "POST"
                && requests::TEXT_PATHS.contains(&path.as_str())
            {
                bytes = match requests::default_model(&bytes, &self.config.kobold_default_model) {
                    Ok(v) => v,
                    Err(_) => return Ok(http::text(400, "expected a JSON object")),
                };
                upstream_path = requests::kobold_path(&path).into();
            } else if kind == "tabbyapi" {
                bytes = requests::tabby(&self.profiles.body(&bytes, kind, None));
                let [effort, tokens, budget] = requests::thinking_diagnostics(&bytes);
                log(
                    20,
                    LOG,
                    format!(
                        "TabbyAPI request: reasoning_effort={effort} max_tokens={tokens} reasoning_budget_tokens={budget}"
                    ),
                );
            }
            if bytes.len() > self.config.max_chat_body_bytes {
                return Ok(http::text(413, "Maximum request body size exceeded"));
            }
            body = Some(bytes);
        }
        self.forward(
            req,
            Forward {
                origin: &origin,
                path: &upstream_path,
                query: None,
                body,
                target: label(kind),
                lease: Some(lease),
                router: self.config.kobold_router_mode,
                selection: None,
            },
        )
        .await
    }
}
fn inactive(path: &str, kind: &str) -> Response<Body> {
    if kind == "tabbyapi" {
        http::json(
            503,
            json!({"detail":"LLM readiness is not currently verified"}),
        )
    } else if path == "/v1/models" {
        http::json(200, json!({"object":"list","data":[]}))
    } else {
        http::json(200, json!({"result":"inactive"}))
    }
}
fn with_model(q: &[(String, String)], model: &str) -> String {
    let mut ser = url::form_urlencoded::Serializer::new(String::new());
    for (k, v) in q {
        if k != "model" {
            ser.append_pair(k, v);
        }
    }
    ser.append_pair("model", model).finish()
}
async fn read_body(
    req: &mut Request<Body>,
    limit: usize,
) -> std::result::Result<Vec<u8>, Response<Body>> {
    let body = std::mem::replace(req.body_mut(), Body::empty());
    to_bytes(body, limit)
        .await
        .map(|b| b.to_vec())
        .map_err(|_| http::text(413, "Maximum request body size exceeded"))
}
async fn chat_handler(State(s): State<Service>, mut req: Request<Body>) -> Response<Body> {
    if req.uri().path() == "/broker/status" {
        return s.status().await;
    }
    let method = req.method().to_string();
    let path = decoded_path(req.uri().path());
    let label = label(&s.config.llm_backend);
    let completion =
        crate::request_log::RequestLog::new(label, &method, &path, 20, s.active_chat.clone());
    req.extensions_mut().insert(completion.clone());
    log(
        20,
        LOG,
        format!("request started: target={label} method={method} path={path}"),
    );
    let response = match s.chat(req).await {
        Ok(r) => r,
        Err(e) => {
            let kind = e
                .downcast_ref::<reqwest::Error>()
                .map(|e| {
                    if e.is_timeout() {
                        "TimeoutError"
                    } else if e.is_connect() {
                        "ClientConnectorError"
                    } else {
                        "ClientError"
                    }
                })
                .unwrap_or(if e.is::<tokio::time::error::Elapsed>() {
                    "TimeoutError"
                } else {
                    "UpstreamError"
                });
            completion.failed(kind);
            return http::json(
                502,
                json!({"error":format!("{label} upstream request failed")}),
            );
        }
    };
    if response
        .extensions()
        .get::<crate::request_log::Streamed>()
        .is_none()
    {
        if response.status().is_client_error() || response.status().is_server_error() {
            completion.rejected(
                response.status().as_u16(),
                response.status().canonical_reason().unwrap_or(""),
            )
        } else {
            completion.complete(response.status().as_u16())
        }
    }
    response
}
// Imagine owns job admission and the coordinator ticket. Taking an HTTP-level
// image lease here would deadlock the job behind its own forwarded request.
async fn imagine_handler(s: &Service, mut req: Request<Body>) -> Response<Body> {
    let Some(path) = req.uri().path().strip_prefix("/imagine/") else {
        return http::json(404, json!({"error":{"code":"not_found"}}));
    };
    let path = format!("/{path}");
    if s.config.coordinator_socket.is_none() {
        return http::json(
            503,
            json!({"error":{"code":"imagine_proxy_not_configured"}}),
        );
    }
    let level = if req.method() == axum::http::Method::GET {
        10
    } else {
        20
    };
    let completion = crate::request_log::RequestLog::new(
        "Imagine",
        req.method().as_str(),
        &path,
        level,
        s.active_image.clone(),
    );
    log(
        level,
        LOG,
        format!(
            "request started: target=Imagine method={} path={path}",
            req.method()
        ),
    );
    req.extensions_mut().insert(completion.clone());
    // Keep the caller's bearer token and idempotency key. The upstream API
    // enforces owner isolation and admin permissions; never inject admin auth.
    match http::proxy(
        req,
        s.image.clone(),
        http::Upstream {
            origin: s.config.imagine_url.clone(),
            path,
            query: None,
            body: None,
            target: "Imagine".into(),
            lease: None,
            limit: s.image_limit.clone(),
            on_failure: None,
        },
    )
    .await
    {
        Ok(response) => response,
        Err(_) => {
            completion.failed("UpstreamError");
            http::json(
                502,
                json!({"error":{"code":"imagine_upstream_unavailable"}}),
            )
        }
    }
}

async fn image_handler(State(s): State<Service>, mut req: Request<Body>) -> Response<Body> {
    let path = decoded_path(req.uri().path());
    if path == "/imagine" || path.starts_with("/imagine/") {
        return imagine_handler(&s, req).await;
    }

    if req.uri().path() == "/broker/status" {
        return s.status().await;
    }
    if !s.config.comfy_enabled {
        return http::json(503, json!({"error":{"code":"comfy_disabled"}}));
    }
    let path = decoded_path(req.uri().path());
    let method = req.method().to_string();
    let kind = routes::classify(&method, &path);
    let started = Instant::now();
    let level = if method == "GET"
        && (path == "/history"
            || path.starts_with("/history/")
            || path == "/api/jobs"
            || path.ends_with("/jobs/status"))
    {
        10
    } else {
        20
    };
    log(
        level,
        LOG,
        format!("request started: target=ComfyUI method={method} path={path}"),
    );
    let rejection = match kind {
        Kind::Lifecycle => Some((
            "broker-managed lifecycle endpoint",
            "ComfyUI lifecycle endpoints are managed by the broker",
        )),
        Kind::BlockedGpu => Some((
            "uncoordinated GPU route",
            "ComfyUI route blocked because it can start GPU work outside the workflow coordinator",
        )),
        Kind::UnknownMutation if !s.config.allow_unknown_comfy_routes => Some((
            "unclassified mutating route",
            "unclassified mutating ComfyUI route rejected by strict proxy policy",
        )),
        _ => None,
    };
    if let Some((reason, error)) = rejection {
        log(
            30,
            LOG,
            format!(
                "request rejected: target=ComfyUI method={method} path={path} status=403 reason={reason} duration={:.3}s",
                started.elapsed().as_secs_f64()
            ),
        );
        return http::json(403, json!({"error":error}));
    }
    let completion = crate::request_log::RequestLog::new(
        "ComfyUI",
        &method,
        &path,
        level,
        s.active_image.clone(),
    );
    req.extensions_mut().insert(completion.clone());
    let upstream = routes::upstream_path(req.uri().path());
    if kind == Kind::Workflow {
        let body = match read_body(&mut req, s.config.max_workflow_body_bytes).await {
            Ok(v) => v,
            Err(_) => {
                return http::json(
                    413,
                    json!({"error":"workflow request body exceeds the configured size limit"}),
                );
            }
        };
        return match s
            .coordinator
            .submit(
                upstream,
                req.uri().query().unwrap_or("").into(),
                http::request_headers(req.headers(), &s.config.comfy_url),
                body,
            )
            .await
        {
            Ok(r) => {
                log(
                    20,
                    LOG,
                    format!(
                        "request completed: target=ComfyUI method={method} path={path} status={} duration={:.3}s",
                        r.status,
                        started.elapsed().as_secs_f64()
                    ),
                );
                http::buffered(r)
            }
            Err(e) => {
                let message = e.to_string();
                let status = if message.starts_with("workflow queue") {
                    429
                } else {
                    503
                };
                let mut r = http::json(status, json!({"error":message}));
                if status == 429 {
                    r.headers_mut().insert("retry-after", "1".parse().unwrap());
                }
                r
            }
        };
    }
    if req
        .headers()
        .get("upgrade")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"))
    {
        let (mut parts, body) = req.into_parts();
        let upgrade = match WebSocketUpgrade::from_request_parts(&mut parts, &s).await {
            Ok(u) => u,
            Err(_) => return http::text(400, "invalid WebSocket upgrade"),
        };
        let req = Request::from_parts(parts, body);
        return match http::websocket(
            req.headers().clone(),
            req.uri().clone(),
            upgrade,
            &s.config.comfy_url,
            completion,
            s.active_ws.clone(),
            s.websocket_limit.clone(),
        )
        .await
        {
            Ok(r) => r,
            Err(_) => http::json(502, json!({"error":"ComfyUI upstream request failed"})),
        };
    }
    let mut body = None;
    if method == "POST" && path == "/history" {
        let bytes = match read_body(&mut req, 1024 * 1024 * 1024).await {
            Ok(b) => b,
            Err(r) => return r,
        };
        if let Ok(v) = serde_json::from_slice::<Value>(&bytes)
            && let Some(ids) = v["delete"].as_array()
        {
            let ids = ids
                .iter()
                .filter_map(|v| v.as_str().filter(|v| !v.is_empty()).map(str::to_string))
                .collect::<Vec<_>>();
            s.coordinator.wait_history(&ids).await;
        }
        body = Some(bytes);
    }
    let lease = if kind == Kind::Control {
        match s.coordinator.control() {
            Ok(l) => Some(l),
            Err(e) => {
                completion.rejected(409, &e.to_string());
                return http::json(409, json!({"error":e.to_string()}));
            }
        }
    } else {
        None
    };
    match http::proxy(
        req,
        s.image.clone(),
        http::Upstream {
            origin: s.config.comfy_url.clone(),
            path: upstream,
            query: None,
            body,
            target: "ComfyUI".into(),
            lease,
            limit: s.image_limit.clone(),
            on_failure: None,
        },
    )
    .await
    {
        Ok(r) => r,
        Err(_) => http::json(502, json!({"error":"ComfyUI upstream request failed"})),
    }
}

fn decoded_path(raw: &str) -> String {
    percent_encoding::percent_decode_str(raw)
        .decode_utf8_lossy()
        .into_owned()
}

#[cfg(test)]
mod imagine_bridge_tests {
    use super::*;

    #[tokio::test]
    async fn bridge_fails_closed_without_coordination_or_upstream() {
        let dir = tempfile::tempdir().unwrap();
        let unused = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", unused.local_addr().unwrap());
        drop(unused);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        for coordinated in [false, true] {
            let running = Service::start(Config {
                chat_port: 0,
                image_port: 0,
                imagine_url: origin.clone(),
                coordinator_socket: coordinated
                    .then(|| dir.path().join("gpu.sock").to_string_lossy().into_owned()),
                imagine_access_file: coordinated.then(|| {
                    dir.path()
                        .join("access.json")
                        .to_string_lossy()
                        .into_owned()
                }),
                ..Default::default()
            })
            .await
            .unwrap();
            let base = format!("http://127.0.0.1:{}", running.image_port);
            let response = client
                .get(format!("{base}/imagine/v1/models"))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status().as_u16(),
                if coordinated { 502 } else { 503 }
            );
            assert_eq!(
                response.json::<Value>().await.unwrap()["error"]["code"],
                if coordinated {
                    "imagine_upstream_unavailable"
                } else {
                    "imagine_proxy_not_configured"
                }
            );
            assert_eq!(
                client
                    .get(format!("{base}/im%61gine/v1/models"))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                404
            );
            assert_eq!(running.service.active_image.load(Ordering::Relaxed), 0);
            running.close().await;
        }
    }
}
