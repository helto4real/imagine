use anyhow::{Result, bail};
use serde_json::{Map, Value};
pub const TEXT_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/chat/completions",
    "/v1/completions",
    "/v1/completion",
    "/completions",
    "/api/v1/generate",
    "/api/latest/generate",
    "/api/extra/generate/stream",
    "/api/extra/tokencount",
    "/api/extra/tokenize",
];
pub const STRATA_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/messages/count_tokens",
];
pub fn kobold_path(p: &str) -> &str {
    match p {
        "/api/latest/generate" => "/api/v1/generate",
        "/api/extra/tokenize" => "/api/extra/tokencount",
        _ => p,
    }
}
pub fn default_model(body: &[u8], default: &str) -> Result<Vec<u8>> {
    let mut v: Value = serde_json::from_slice(body)?;
    if !v.is_object() {
        bail!("request body must be a JSON object")
    }
    if v["model"].is_null()
        || v["model"] == ""
        || (v["model"] == "initial_model" && default != "initial_model")
    {
        v["model"] = default.into();
        return Ok(crate::json_wire::dumps(&v, false, false).into_bytes());
    }
    Ok(body.into())
}
fn first<'a>(v: &'a Value, keys: &[&str]) -> &'a Value {
    keys.iter().find_map(|k| v.get(k)).unwrap_or(&Value::Null)
}
fn integer(v: &Value) -> Option<i128> {
    match v {
        Value::Bool(b) => Some(i128::from(*b)),
        Value::Number(n) => n
            .as_i64()
            .map(i128::from)
            .or_else(|| n.as_u64().map(i128::from))
            .or_else(|| n.as_f64().filter(|v| v.fract() == 0.0).map(|v| v as i128)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}
fn falls_back(v: &Value) -> bool {
    v.is_null() || integer(v).is_some_and(|v| v < 0)
}
pub fn tabby(body: &[u8]) -> Vec<u8> {
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else {
        return body.into();
    };
    if !v.is_object() {
        return body.into();
    }
    let reasoning = v
        .get("reasoning")
        .filter(|v| !v.is_null())
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let t = first(&v, &["template_vars", "chat_template_kwargs"]);
    if !reasoning.is_object() || (!t.is_null() && !t.is_object()) {
        return body.into();
    }
    let template = t.clone();
    let mut effort = &reasoning["effort"];
    let mut enabled = &reasoning["enabled"];
    if !v["reasoning_effort"].is_null() {
        effort = &v["reasoning_effort"]
    }
    if !v["enable_thinking"].is_null() {
        enabled = &v["enable_thinking"]
    }
    if let Some(x) = template.get("reasoning_effort") {
        effort = x
    }
    if let Some(x) = template.get("enable_thinking") {
        enabled = x
    }
    let effort = effort.as_str().map(|s| s.trim().to_lowercase());
    let enabled = enabled.clone();
    let mut changed = false;
    if matches!(effort.as_deref(), Some("none" | "off"))
        && enabled.is_null()
        && template.get("enable_thinking").is_none()
    {
        v["enable_thinking"] = false.into();
        changed = true;
    }
    let keys = [
        "reasoning_budget_tokens",
        "reasoning_budget",
        "thinking_budget",
        "thinking_token_budget",
    ];
    let wins = !falls_back(first(&v, &keys)) || !falls_back(&reasoning["max_tokens"]);
    if !wins && v.get("thinking_budget_tokens").is_some() {
        let b = v
            .as_object_mut()
            .unwrap()
            .remove("thinking_budget_tokens")
            .unwrap();
        v["reasoning_budget_tokens"] = b;
        changed = true;
    } else if !wins
        && !keys.iter().any(|k| v.get(k).is_some())
        && reasoning.get("max_tokens").is_none()
        && enabled != false
    {
        let divisor = match effort.as_deref() {
            Some("minimal") => Some(10),
            Some("low") => Some(4),
            Some("medium") => Some(2),
            _ => None,
        };
        if let (Some(d), Some(n)) = (
            divisor,
            integer(first(
                &v,
                &["max_tokens", "max_completion_tokens", "max_length"],
            )),
        ) && n > 0
            && let Ok(n) = u64::try_from(n / d)
        {
            v["reasoning_budget_tokens"] = n.into();
            changed = true;
        }
    }
    if changed {
        crate::json_wire::dumps(&v, true, true).into_bytes()
    } else {
        body.into()
    }
}
pub fn thinking_diagnostics(body: &[u8]) -> [String; 3] {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return std::array::from_fn(|_| "<other>".into());
    };
    if !v.is_object() {
        return std::array::from_fn(|_| "<other>".into());
    }
    ["reasoning_effort", "max_tokens", "reasoning_budget_tokens"].map(|k| match v.get(k) {
        None => "<absent>".into(),
        Some(Value::Null) => "null".into(),
        Some(Value::String(s))
            if k == "reasoning_effort"
                && ["minimal", "low", "medium", "high", "none", "off", "unset"]
                    .contains(&s.as_str()) =>
        {
            s.clone()
        }
        Some(Value::Number(n)) if k != "reasoning_effort" => n.to_string(),
        _ => "<other>".into(),
    })
}
pub fn request_usage(mut payload: Value) -> Vec<u8> {
    if payload["stream"] == true {
        if payload["stream_options"].is_null() {
            payload["stream_options"] = Value::Object(Map::new());
        }
        let opts = payload
            .as_object_mut()
            .unwrap()
            .entry("stream_options")
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(opts) = opts.as_object_mut() {
            opts.insert("include_usage".into(), true.into());
        }
    }
    crate::json_wire::dumps(&payload, false, false).into_bytes()
}
