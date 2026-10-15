//! Authenticated Imagine API client. Profiles and validation belong to Imagine.
use crate::config::Config;
use anyhow::{Result, anyhow, bail};
use serde_json::Value;
#[derive(Debug)]
struct Rejected(reqwest::StatusCode, String);
impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Imagine {}: {}", self.0, self.1)
    }
}
impl std::error::Error for Rejected {}
pub fn retryable(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<Rejected>()
        .is_none_or(|e| e.0.is_server_error() || matches!(e.0.as_u16(), 408 | 429))
}
#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
    token: String,
}
impl Client {
    pub async fn new(config: &Config) -> Result<Self> {
        let access: Value = serde_json::from_slice(
            &tokio::fs::read(config.imagine_access_file.as_ref().ok_or_else(|| {
                anyhow!("MCP Imagine integration requires --imagine-access-file")
            })?)
            .await?,
        )?;
        // Admin requests retain their own Imagine owner; other clients cannot access MCP assets.
        let token = access["admin_token"]
            .as_str()
            .ok_or_else(|| anyhow!("Imagine credentials unavailable"))?
            .to_string();
        Ok(Self {
            http: reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(std::time::Duration::from_secs_f64(config.request_timeout))
                .build()?,
            base: config.imagine_url.clone(),
            token,
        })
    }
    pub async fn json(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
        key: Option<&str>,
    ) -> Result<Value> {
        let mut req = self
            .http
            .request(method, format!("{}{path}", self.base))
            .bearer_auth(&self.token);
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
            let code = value["error"]["code"]
                .as_str()
                .filter(|s| {
                    s.len() <= 128 && s.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
                })
                .unwrap_or("operation_rejected");
            return Err(Rejected(status, code.to_owned()).into());
        };
        Ok(response.json().await?)
    }
    pub async fn upload(&self, bytes: Vec<u8>) -> Result<String> {
        let r = self
            .http
            .post(format!("{}/v1/assets", self.base))
            .bearer_auth(&self.token)
            .header("Content-Type", "image/png")
            .body(bytes)
            .send()
            .await?;
        if !r.status().is_success() {
            bail!("reference storage was rejected by Imagine")
        };
        r.json::<Value>().await?["id"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("invalid reference storage response"))
    }
    pub async fn download(&self, id: &str) -> Result<Vec<u8>> {
        if !id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c))
        {
            bail!("invalid image reference")
        };
        let mut r = self
            .http
            .get(format!("{}/v1/assets/{id}", self.base))
            .bearer_auth(&self.token)
            .send()
            .await?;
        if !r.status().is_success() {
            bail!("image download failed")
        };
        let mut bytes = Vec::new();
        while let Some(chunk) = r.chunk().await? {
            if bytes.len() + chunk.len() > 64 * 1024 * 1024 {
                bail!("image exceeds the transfer limit")
            };
            bytes.extend(chunk)
        }
        if bytes.is_empty() {
            bail!("image is empty")
        };
        Ok(bytes)
    }
}
