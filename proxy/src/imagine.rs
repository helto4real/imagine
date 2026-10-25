//! Local Imagine API client. Profiles and validation belong to Imagine.
use crate::config::Config;
use anyhow::{Result, anyhow, bail};
use serde_json::Value;
#[derive(Debug)]
struct Rejected(reqwest::StatusCode, String, Option<String>, Option<String>);
impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Imagine {}: {}", self.0, self.1)
    }
}
impl std::error::Error for Rejected {}
/// A validated Imagine rejection: HTTP status, stable code, field path and short detail.
pub struct Rejection<'a> {
    pub status: u16,
    pub code: &'a str,
    pub at: Option<&'a str>,
    pub detail: Option<&'a str>,
}
pub fn rejection(error: &anyhow::Error) -> Option<Rejection<'_>> {
    error.downcast_ref::<Rejected>().map(|e| Rejection {
        status: e.0.as_u16(),
        code: &e.1,
        at: e.2.as_deref(),
        detail: e.3.as_deref(),
    })
}
pub(crate) fn stable_code(code: &str) -> bool {
    code.len() <= 128
        && code.starts_with(|c: char| c.is_ascii_lowercase())
        && code
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}
/// Field paths such as `expansions[3]` or `production.scenes[0].continuity`.
fn field_path(at: &str) -> bool {
    !at.is_empty()
        && at.len() <= 128
        && at
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.[]".contains(&b))
}
fn short_detail(detail: &str) -> bool {
    !detail.is_empty() && detail.chars().count() <= 500 && !detail.chars().any(char::is_control)
}
pub fn retryable(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<Rejected>()
        .is_none_or(|e| e.0.is_server_error() || matches!(e.0.as_u16(), 408 | 429))
}
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
    session: Option<String>,
}
impl Client {
    pub async fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(std::time::Duration::from_secs_f64(config.request_timeout))
                .build()?,
            base: config.imagine_url.clone(),
            session: None,
        })
    }
    pub fn with_session(&self, session: String) -> Self {
        Self {
            session: Some(session),
            ..self.clone()
        }
    }
    pub fn session_id(&self) -> Option<&str> {
        self.session.as_deref()
    }
    fn bound(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(id) = &self.session {
            request.header("X-Imagine-Session", id)
        } else {
            request
        }
    }
    pub async fn session(&self) -> Result<String> {
        let response = self
            .http
            .get(format!("{}/v1/session", self.base))
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await?;
        if !response.status().is_success() {
            bail!("Imagine session unavailable");
        }
        let value: Value = response.json().await?;
        value["id"]
            .as_str()
            .filter(|s| !s.is_empty() && s.len() <= 64)
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("Invalid Imagine session"))
    }
    /// A raw request bound to the Imagine session, for streamed bodies.
    pub fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.bound(self.http.request(method, format!("{}{path}", self.base)))
    }
    pub async fn json(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
        key: Option<&str>,
    ) -> Result<Value> {
        let mut req = self.bound(self.http.request(method, format!("{}{path}", self.base)));
        if let Some(body) = body {
            req = req.json(body)
        }
        if let Some(key) = key {
            req = req.header("Idempotency-Key", key)
        }
        let response = req.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            // Imagine errors contain stable codes, never prompts or credentials.
            let value = response.json::<Value>().await.unwrap_or(Value::Null);
            // Both `{"error":{"code","at","detail"}}` and `{"error":code,"at","detail"}`.
            let error = if value["error"].is_string() {
                &value
            } else {
                &value["error"]
            };
            let text = |v: &Value, valid: fn(&str) -> bool| {
                v.as_str().filter(|s| valid(s)).map(str::to_owned)
            };
            let code = value["error"]
                .as_str()
                .or(error["code"].as_str())
                .filter(|s| stable_code(s))
                .unwrap_or("operation_rejected");
            return Err(Rejected(
                status,
                code.to_owned(),
                text(&error["at"], field_path),
                text(&error["detail"], short_detail),
            )
            .into());
        };
        Ok(response.json().await?)
    }
    /// Link an original for a video project in place, as the Director media import does.
    pub async fn link_video(&self, path: &std::path::Path) -> Result<Value> {
        self.json(
            reqwest::Method::POST,
            "/v1/video/assets/link",
            Some(&serde_json::json!({"path":path})),
            None,
        )
        .await
    }
    pub async fn link(&self, path: &std::path::Path) -> Result<Value> {
        let r = self
            .bound(self.http.post(format!("{}/v1/assets/link", self.base)))
            .json(&serde_json::json!({"path":path,"normalize_image":true}))
            .send()
            .await?;
        if !r.status().is_success() {
            bail!("reference storage was rejected by Imagine")
        };
        Ok(r.json().await?)
    }
}
