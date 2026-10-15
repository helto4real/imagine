use crate::{coordinator::Lease, logging::log};
use anyhow::Result;
use axum::{
    body::{Body, Bytes},
    extract::ws::{Message, WebSocketUpgrade},
    http::{HeaderMap, HeaderValue, Request, Response, StatusCode},
};
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
const HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];
pub fn request_headers(headers: &HeaderMap, origin: &str) -> HeaderMap {
    let mut out = HeaderMap::new();
    let url = reqwest::Url::parse(origin).unwrap();
    let authority = format!(
        "{}://{}",
        url.scheme(),
        &url[url::Position::BeforeHost..url::Position::AfterPort]
    );
    let host = headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let downstream = format!("http://{host}");
    for (k, v) in headers {
        if HOP.contains(&k.as_str()) || matches!(k.as_str(), "host" | "content-length") {
            continue;
        }
        let value = if k == "origin" {
            HeaderValue::from_str(&authority).unwrap()
        } else if k == "referer" {
            v.to_str()
                .ok()
                .and_then(|v| v.strip_prefix(&downstream))
                .and_then(|suffix| HeaderValue::from_str(&format!("{authority}{suffix}")).ok())
                .unwrap_or_else(|| v.clone())
        } else {
            v.clone()
        };
        out.append(k, value);
    }
    out
}
pub fn response_headers(
    headers: &HeaderMap,
    origin: Option<&str>,
    host: Option<&str>,
) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (k, v) in headers {
        if !HOP.contains(&k.as_str()) {
            out.append(k, v.clone());
        }
    }
    if let (Some(origin), Some(host), Some(loc)) = (
        origin,
        host,
        out.get("location").and_then(|v| v.to_str().ok()),
    ) && let (Ok(base), Ok(target)) = (reqwest::Url::parse(origin), reqwest::Url::parse(loc))
        && base.scheme() == target.scheme()
        && base.authority() == target.authority()
        && let Ok(v) = HeaderValue::from_str(&format!(
            "http://{host}{}",
            &target[url::Position::BeforePath..]
        ))
    {
        out.insert("location", v);
    }
    out
}
pub fn json(status: u16, v: serde_json::Value) -> Response<Body> {
    let mut r = Response::new(Body::from(v.to_string()));
    *r.status_mut() = StatusCode::from_u16(status).unwrap();
    r.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("application/json; charset=utf-8"),
    );
    r
}
pub fn text(status: u16, s: impl Into<String>) -> Response<Body> {
    let mut r = Response::new(Body::from(s.into()));
    *r.status_mut() = StatusCode::from_u16(status).unwrap();
    r.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    r
}
pub fn buffered(r: crate::coordinator::Buffered) -> Response<Body> {
    let mut out = Response::new(Body::from(r.body));
    *out.status_mut() = StatusCode::from_u16(r.status).unwrap();
    *out.headers_mut() = response_headers(&r.headers, None, None);
    out
}
pub struct Upstream {
    pub origin: String,
    pub path: String,
    pub query: Option<String>,
    pub body: Option<Vec<u8>>,
    pub target: String,
    pub lease: Option<Lease>,
    pub limit: crate::connection_limit::Limit,
    pub on_failure: Option<(
        crate::coordinator::Coordinator,
        Option<crate::backend::Target>,
    )>,
}
pub async fn proxy(
    request: Request<Body>,
    client: reqwest::Client,
    upstream: Upstream,
) -> Result<Response<Body>> {
    let Upstream {
        origin,
        path,
        query,
        body,
        target,
        lease,
        limit,
        on_failure,
    } = upstream;
    let (parts, body_stream) = request.into_parts();
    let host = parts
        .headers
        .get("host")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let query = query.as_deref().or(parts.uri.query()).unwrap_or("");
    let url = format!(
        "{}{}{}{}",
        origin.trim_end_matches('/'),
        path,
        if query.is_empty() { "" } else { "?" },
        query
    );
    let call = client
        .request(parts.method.clone(), url)
        .headers(request_headers(&parts.headers, &origin));
    let mut observer = if target == "Prism"
        && parts.method == "POST"
        && parts.uri.path().trim_end_matches('/') == "/v1/chat/completions"
    {
        body.as_ref()
            .and_then(|body| crate::diagnostics::Prism::new(body))
    } else {
        None
    };
    let completion = parts
        .extensions
        .get::<crate::request_log::RequestLog>()
        .cloned();
    let call = if let Some(body) = body {
        call.body(body)
    } else {
        // Streaming erases the body's exact size. Preserve framing for GET/HEAD
        // bodies too; hyper otherwise assumes those methods have no body.
        let call = if let Some(length) = parts.headers.get("content-length") {
            call.header("content-length", length)
        } else if parts.headers.contains_key("transfer-encoding") {
            call.header("transfer-encoding", "chunked")
        } else {
            call
        };
        call.body(reqwest::Body::wrap_stream(body_stream.into_data_stream()))
    };
    let permit = limit.acquire().await?;
    let upstream = match call.send().await {
        Ok(r) => r,
        Err(e) => {
            if let Some((c, t)) = &on_failure {
                c.request_failed(t.as_ref()).await;
            }
            return Err(e.into());
        }
    };
    let status = upstream.status();
    if let Some(o) = &mut observer {
        o.start(
            status.as_u16(),
            upstream
                .headers()
                .get("content-type")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("")
                .split(';')
                .next()
                .unwrap_or(""),
        );
    }
    let headers = response_headers(upstream.headers(), Some(&origin), host.as_deref());
    if status.as_u16() >= 400 {
        log(
            40,
            "st_proxy.http",
            format!(
                "upstream request failed: target={target} method={} path={} status={}",
                parts.method,
                parts.uri.path(),
                status.as_u16()
            ),
        );
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(1);
    tokio::spawn(async move {
        let _lease = lease;
        let _permit = permit;
        let mut connected = true;
        let mut stream = upstream.bytes_stream();
        let mut failed = status.as_u16() >= 500;
        let mut complete = true;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if let Some(o) = &mut observer {
                        o.feed(&bytes)
                    }
                    if connected && tx.send(Ok(bytes)).await.is_err() {
                        connected = false;
                    }
                }
                Err(e) => {
                    failed = true;
                    complete = false;
                    let _ = tx.send(Err(std::io::Error::other(e))).await;
                    break;
                }
            }
        }
        if let Some(o) = observer {
            log(
                20,
                "st_proxy.llm.prism_diagnostics",
                o.finish(complete, connected),
            );
        }
        if failed && let Some((c, t)) = on_failure {
            c.request_failed(t.as_ref()).await;
        }
        drop(_lease);
        if let Some(completion) = completion {
            if complete {
                completion.complete(status.as_u16())
            } else {
                completion.failed("ClientPayloadError")
            }
        }
    });
    let stream =
        futures_util::stream::unfold(rx, |mut rx| async { rx.recv().await.map(|v| (v, rx)) });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    response
        .extensions_mut()
        .insert(crate::request_log::Streamed);
    Ok(response)
}
pub async fn websocket(
    headers: HeaderMap,
    uri: axum::http::Uri,
    upgrade: WebSocketUpgrade,
    origin: &str,
    completion: crate::request_log::RequestLog,
    active: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    limit: crate::connection_limit::Limit,
) -> Result<Response<Body>> {
    use tokio_tungstenite::tungstenite::{Message as Up, client::IntoClientRequest};
    let url = format!("{}{}", origin.replacen("http", "ws", 1), uri);
    let mut req = url.into_client_request()?;
    for (k, v) in request_headers(&headers, origin) {
        if let Some(k) = k
            && !k.as_str().starts_with("sec-websocket-")
        {
            req.headers_mut().append(k, v);
        }
    }
    if let Some(v) = headers.get("sec-websocket-protocol") {
        req.headers_mut()
            .insert("sec-websocket-protocol", v.clone());
    }
    let cfg = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default()
        .max_message_size(Some(64 * 1024 * 1024));
    let permit = limit.acquire().await?;
    let (upstream, resp) =
        tokio_tungstenite::connect_async_with_config(req, Some(cfg), false).await?;
    let protocol = resp
        .headers()
        .get("sec-websocket-protocol")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let upgrade = if let Some(protocol) = protocol {
        upgrade.protocols([protocol])
    } else {
        upgrade
    };
    struct Active(std::sync::Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    Ok(upgrade.max_message_size(64 * 1024 * 1024).on_upgrade(move |mut downstream| async move {
        let _permit = permit;
        active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _active = Active(active);
        let mut upstream = upstream;
        let mut up_deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        let mut down_deadline = up_deadline;
        let mut up_ping = false;
        let mut down_ping = false;
        let mut close_code = 1000;
        loop {
            tokio::select! {
                msg = downstream.recv() => {
                    down_deadline = tokio::time::Instant::now() + Duration::from_secs(30);
                    down_ping = false;
                    match msg {
                        Some(Ok(Message::Text(v))) => if upstream.send(Up::Text(v.to_string().into())).await.is_err() { break },
                        Some(Ok(Message::Binary(v))) => if upstream.send(Up::Binary(v)).await.is_err() { break },
                        Some(Ok(Message::Ping(v))) => if downstream.send(Message::Pong(v)).await.is_err() { break },
                        Some(Ok(Message::Pong(_))) => {},
                        _ => break,
                    }
                },
                msg = upstream.next() => {
                    up_deadline = tokio::time::Instant::now() + Duration::from_secs(30);
                    up_ping = false;
                    match msg {
                        Some(Ok(Up::Text(v))) => if downstream.send(Message::Text(v.to_string().into())).await.is_err() { break },
                        Some(Ok(Up::Binary(v))) => if downstream.send(Message::Binary(v)).await.is_err() { break },
                        Some(Ok(Up::Ping(v))) => if upstream.send(Up::Pong(v)).await.is_err() { break },
                        Some(Ok(Up::Pong(_))) => {},
                        Some(Ok(Up::Close(frame))) => { close_code = frame.map_or(1000, |f| f.code.into()); break },
                        _ => break,
                    }
                },
                _ = tokio::time::sleep_until(up_deadline) => {
                    if up_ping { close_code = 1001; break; }
                    if upstream.send(Up::Ping(Bytes::new())).await.is_err() { break; }
                    up_ping = true;
                    up_deadline = tokio::time::Instant::now() + Duration::from_secs(15);
                },
                _ = tokio::time::sleep_until(down_deadline) => {
                    if down_ping { break; }
                    if downstream.send(Message::Ping(Bytes::new())).await.is_err() { break; }
                    down_ping = true;
                    down_deadline = tokio::time::Instant::now() + Duration::from_secs(15);
                }
            }
        }
        let _ = upstream.close(None).await;
        let _ = downstream.send(Message::Close(Some(axum::extract::ws::CloseFrame {
            code: close_code, reason: "".into(),
        }))).await;
        completion.complete(101);
    }).map(Body::new))
}
