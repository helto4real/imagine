use crate::{config::Config, logging::log};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
use tokio::time::{sleep, timeout};
#[derive(Clone)]
pub struct Comfy {
    pub config: Config,
    pub client: reqwest::Client,
    pub limit: crate::connection_limit::Limit,
}
impl Comfy {
    pub async fn interrupt(&self) {
        log(20, "st_proxy.comfy", "ComfyUI interrupt requested");
        let Ok(_permit) = self.limit.acquire().await else {
            return;
        };
        match self
            .client
            .post(format!("{}/interrupt", self.config.comfy_url))
            .timeout(Duration::from_secs_f64(self.config.cleanup_timeout))
            .send()
            .await
        {
            Ok(r) => {
                let status = r.status().as_u16();
                match r.bytes().await {
                    Ok(_) => log(
                        20,
                        "st_proxy.comfy",
                        format!("ComfyUI interrupt completed: status={status}"),
                    ),
                    Err(_) => log(
                        30,
                        "st_proxy.comfy",
                        "ComfyUI interrupt failed: ClientPayloadError",
                    ),
                }
            }
            Err(e) => log(
                30,
                "st_proxy.comfy",
                format!(
                    "ComfyUI interrupt failed: {}",
                    if e.is_timeout() {
                        "TimeoutError"
                    } else {
                        "ClientConnectorError"
                    }
                ),
            ),
        }
    }
    pub async fn wait_prompt(&self, id: &str) -> Result<Value> {
        let start = Instant::now();
        let mut failures = 0;
        while start.elapsed().as_secs_f64() < self.config.image_timeout {
            let mut failure = None;
            let seconds = self
                .config
                .request_timeout
                .min((self.config.image_timeout - start.elapsed().as_secs_f64()).max(0.01));
            let call = async {
                let _permit = self
                    .limit
                    .acquire()
                    .await
                    .map_err(|_| "TimeoutError".to_string())?;
                let r = self
                    .client
                    .get(format!("{}/history/{id}", self.config.comfy_url))
                    .send()
                    .await
                    .map_err(|_| "ClientConnectorError".to_string())?;
                if r.status().as_u16() >= 400 {
                    return Err(format!("HTTP {}", r.status().as_u16()));
                }
                r.json::<Value>()
                    .await
                    .map_err(|_| "ValueError".to_string())
            };
            match timeout(Duration::from_secs_f64(seconds), call).await {
                Ok(Ok(v)) => {
                    if !v.is_object() {
                        failure = Some("invalid history response".into())
                    } else {
                        failures = 0;
                        let e = &v[id];
                        let s = &e["status"];
                        if e.is_object() {
                            if s.is_object() {
                                let state = s["status_str"].as_str().unwrap_or("").to_lowercase();
                                if matches!(
                                    state.as_str(),
                                    "cancelled" | "canceled" | "error" | "failed" | "failure"
                                ) {
                                    bail!("ComfyUI image job failed: {state}")
                                }
                                if s["completed"] == true {
                                    if !matches!(state.as_str(), "success" | "completed") {
                                        bail!(
                                            "ComfyUI image job failed: {}",
                                            if state.is_empty() { "unknown" } else { &state }
                                        )
                                    }
                                    log(
                                        20,
                                        "st_proxy.comfy",
                                        format!(
                                            "ComfyUI job state confirmed: prompt_id={id} duration={:.3}s",
                                            start.elapsed().as_secs_f64()
                                        ),
                                    );
                                    return Ok(e.get("outputs").cloned().unwrap_or(json!({})));
                                }
                            }
                            if e["outputs"].as_object().is_some_and(|v| !v.is_empty())
                                && (s.is_null() || s.as_object().is_some_and(|s| s.is_empty()))
                            {
                                log(
                                    20,
                                    "st_proxy.comfy",
                                    format!(
                                        "ComfyUI job outputs confirmed: prompt_id={id} duration={:.3}s",
                                        start.elapsed().as_secs_f64()
                                    ),
                                );
                                return Ok(e["outputs"].clone());
                            }
                        }
                    }
                }
                Ok(Err(e)) => failure = Some(e),
                Err(_) => failure = Some("TimeoutError".into()),
            }
            if let Some(error) = failure {
                failures += 1;
                if failures == 1 {
                    log(
                        30,
                        "st_proxy.comfy",
                        format!("ComfyUI history monitoring failed; retrying: error={error}"),
                    );
                }
                if failures >= self.config.comfy_poll_failure_limit {
                    log(
                        40,
                        "st_proxy.comfy",
                        format!(
                            "ComfyUI history monitoring aborted after consecutive failures: count={failures} error={error}"
                        ),
                    );
                    self.interrupt().await;
                    bail!(
                        "ComfyUI history monitoring failed: {error} after {failures} consecutive failures"
                    )
                }
            }
            sleep(Duration::from_secs_f64(self.config.poll_interval)).await;
        }
        self.interrupt().await;
        bail!("ComfyUI image job failed: timed out")
    }
    async fn status(&self, path: &str) -> Result<Value> {
        let _permit = self.limit.acquire().await?;
        let r = self
            .client
            .get(format!("{}{path}", self.config.comfy_url))
            .send()
            .await?;
        if r.status() != 200 {
            bail!(
                "ComfyUI cleanup observation failed: HTTP {}",
                r.status().as_u16()
            )
        }
        let v: Value = r.json().await?;
        if !v.is_object() {
            bail!("ComfyUI cleanup observation failed: invalid status")
        }
        Ok(v)
    }
    async fn idle(&self) -> Result<()> {
        loop {
            let q = self.status("/queue").await?;
            let (Some(r), Some(p)) = (q["queue_running"].as_array(), q["queue_pending"].as_array())
            else {
                bail!("ComfyUI cleanup observation failed: invalid queue status")
            };
            if r.is_empty() && p.is_empty() {
                return Ok(());
            }
            sleep(Duration::from_secs_f64(self.config.poll_interval)).await;
        }
    }
    pub async fn free(&self) -> Result<()> {
        if !self.config.comfy_enabled {
            return Ok(());
        }
        log(20, "st_proxy.comfy", "ComfyUI VRAM cleanup requested");
        timeout(Duration::from_secs_f64(self.config.cleanup_timeout),async{
        let settle=matches!(self.config.llm_backend.as_str(),"tabbyapi"|"all");
        if settle{self.idle().await?;}
        let permit = self.limit.acquire().await?;
        let r=self.client.post(format!("{}/free",self.config.comfy_url)).json(&json!({"unload_models":true,"free_memory":true})).send().await?;
        let s=r.status().as_u16();let _=r.bytes().await?;drop(permit);if s>=400{bail!("ComfyUI model cleanup failed: HTTP {s}")}
        log(20,"st_proxy.comfy",format!("ComfyUI VRAM cleanup accepted: status={s}; waiting 5 seconds before LLM reload"));
        sleep(Duration::from_secs(5)).await;
        if settle {
            let mut previous=None;
            loop{
                self.idle().await?;let stats=self.status("/system_stats").await?;
                let Some(devices)=stats["devices"].as_array().filter(|v|!v.is_empty())else{bail!("ComfyUI cleanup observation failed: missing memory status")};
                let mut current=Vec::new();
                for d in devices {
                    let vals:Option<Vec<u64>>=["vram_total","vram_free","torch_vram_total","torch_vram_free"].iter().map(|k|d[k].as_u64()).collect();
                    let Some(v)=vals else{bail!("ComfyUI cleanup observation failed: invalid memory status")};
                    if v[1]>v[0] || v[3]>v[2] || v[3]>v[1]{bail!("ComfyUI cleanup observation failed: invalid memory status")}
                    current.push((v[0],v[1]-v[3],v[2],v[3]));
                }
                if previous.as_ref()==Some(&current){log(20,"st_proxy.comfy","ComfyUI queue idle and memory observations stable");break}
                previous=Some(current);sleep(Duration::from_secs_f64(self.config.poll_interval)).await;
            }
        }
        log(20,"st_proxy.comfy","ComfyUI cleanup wait completed");Ok(())
    }).await.map_err(|_|anyhow::anyhow!("ComfyUI model cleanup failed: TimeoutError"))?
    }
}
