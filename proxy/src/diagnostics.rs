//! Bounded, content-free Prism response diagnostics.
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
const MAX: usize = 1024 * 1024;
fn count(v: &Value) -> String {
    v.as_u64()
        .filter(|v| *v <= i64::MAX as u64)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unavailable".into())
}
pub struct Prism {
    max_tokens: String,
    effort: String,
    enabled: String,
    status: Option<u16>,
    sse: bool,
    valid: bool,
    done: bool,
    buffer: Vec<u8>,
    data: Vec<Vec<u8>>,
    size: usize,
    chars: usize,
    reasons: BTreeSet<String>,
    usage: HashMap<&'static str, String>,
}
impl Prism {
    pub fn new(body: &[u8]) -> Option<Self> {
        let p: Value = serde_json::from_slice(body).ok()?;
        let effort = p["reasoning_effort"]
            .as_str()
            .filter(|s| ["none", "minimal", "low", "medium", "high", "xhigh", "unset"].contains(s))
            .unwrap_or("unavailable")
            .into();
        let enabled = p["chat_template_kwargs"]["enable_thinking"]
            .as_bool()
            .map(|b| b.to_string())
            .unwrap_or_else(|| "unavailable".into());
        Some(Self {
            max_tokens: count(&p["max_tokens"]),
            effort,
            enabled,
            status: None,
            sse: false,
            valid: true,
            done: false,
            buffer: Vec::new(),
            data: Vec::new(),
            size: 0,
            chars: 0,
            reasons: BTreeSet::new(),
            usage: HashMap::new(),
        })
    }
    pub fn start(&mut self, status: u16, content_type: &str) {
        self.status = Some(status);
        self.sse = content_type == "text/event-stream";
        self.valid =
            status == 200 && ["text/event-stream", "application/json"].contains(&content_type)
    }
    fn invalidate(&mut self) {
        self.valid = false;
        self.buffer.clear();
        self.data.clear();
        self.size = 0;
    }
    pub fn feed(&mut self, chunk: &[u8]) {
        if !self.valid {
            return;
        }
        if !self.sse {
            if self.buffer.len() + chunk.len() > MAX {
                self.invalidate()
            } else {
                self.buffer.extend_from_slice(chunk)
            }
            return;
        }
        for piece in chunk.split_inclusive(|b| *b == b'\n') {
            if self.buffer.len() + self.size + piece.len() > MAX {
                self.invalidate();
                return;
            }
            self.buffer.extend_from_slice(piece);
            if self.buffer.last() != Some(&b'\n') {
                continue;
            }
            while self
                .buffer
                .last()
                .is_some_and(|c| *c == b'\r' || *c == b'\n')
            {
                self.buffer.pop();
            }
            let line = std::mem::take(&mut self.buffer);
            if let Some(data) = line.strip_prefix(b"data:") {
                let data = data.strip_prefix(b" ").unwrap_or(data).to_vec();
                self.size += data.len() + 1;
                self.data.push(data);
            } else if line.is_empty() && !self.data.is_empty() {
                let event = self.data.join(&b'\n');
                self.data.clear();
                self.size = 0;
                if event.trim_ascii() == b"[DONE]" {
                    self.done = true
                } else {
                    self.event(&event)
                }
                if !self.valid {
                    return;
                }
            }
        }
    }
    fn event(&mut self, bytes: &[u8]) {
        let parse = (|| -> Option<()> {
            let p: Value = serde_json::from_slice(bytes).ok()?;
            if !p.is_object() || p.get("error").is_some() {
                return None;
            }
            let empty = Vec::new();
            let choices = match p.get("choices") {
                None => &empty,
                Some(c) => c.as_array()?,
            };
            for c in choices {
                if !c.is_object() {
                    return None;
                }
                if let Some(reason) = c.get("finish_reason").filter(|r| !r.is_null()) {
                    self.reasons.insert(
                        reason
                            .as_str()
                            .filter(|s| {
                                [
                                    "stop",
                                    "length",
                                    "tool_calls",
                                    "function_call",
                                    "content_filter",
                                ]
                                .contains(s)
                            })
                            .unwrap_or("other")
                            .into(),
                    );
                }
                if let Some(m) = c.get(if self.sse { "delta" } else { "message" }) {
                    let m = m.as_object()?;
                    if let Some(s) = m.get("reasoning_content").and_then(Value::as_str) {
                        self.chars += s.chars().count()
                    }
                }
            }
            if p["usage"].is_object() {
                for k in ["prompt_tokens", "completion_tokens"] {
                    self.usage.insert(k, count(&p["usage"][k]));
                }
                self.usage.insert(
                    "cached_tokens",
                    count(&p["usage"]["prompt_tokens_details"]["cached_tokens"]),
                );
                self.usage.insert(
                    "reasoning_tokens",
                    count(&p["usage"]["completion_tokens_details"]["reasoning_tokens"]),
                );
            }
            Some(())
        })();
        if parse.is_none() {
            self.invalidate()
        }
    }
    pub fn finish(mut self, complete: bool, connected: bool) -> String {
        if self.valid && !self.sse && complete {
            let bytes = std::mem::take(&mut self.buffer);
            self.event(&bytes)
        }
        if self.sse && (!self.buffer.is_empty() || !self.data.is_empty()) {
            self.valid = false
        }
        let status = if self.status.is_some_and(|s| s != 200) {
            "http_error"
        } else if !complete {
            "interrupted"
        } else if !self.valid || self.reasons.is_empty() || (self.sse && !self.done) {
            "incomplete"
        } else {
            "complete"
        };
        let fields = [
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "reasoning_tokens",
        ]
        .map(|k| {
            self.usage
                .get(k)
                .map(String::as_str)
                .unwrap_or("unavailable")
        });
        format!(
            "Prism response: max_tokens={} reasoning_effort={} enable_thinking={} prompt_tokens={} cached_tokens={} completion_tokens={} reasoning_tokens={} reasoning_chars={} finish_reason={} stream_status={} client_disconnected={}",
            self.max_tokens,
            self.effort,
            self.enabled,
            fields[0],
            fields[1],
            fields[2],
            fields[3],
            if self.valid {
                self.chars.to_string()
            } else {
                "unavailable".into()
            },
            if self.reasons.is_empty() {
                "unavailable".into()
            } else {
                self.reasons.into_iter().collect::<Vec<_>>().join(",")
            },
            status,
            !connected
        )
    }
}
