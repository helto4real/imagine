use crate::logging::log;
use anyhow::{Result, bail};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
const REP: &[&str] = &["repetition_penalty", "repeat_penalty", "rep_pen"];
const BUDGET: &[&str] = &[
    "reasoning_budget_tokens",
    "reasoning_budget",
    "thinking_budget",
    "thinking_token_budget",
    "thinking_budget_tokens",
];
const STRATA: &[&str] = &[
    "sc117",
    "orca",
    "sc117_128k",
    "orca_128k",
    "swift",
    "swift_128k",
];
#[derive(Clone, Default)]
pub struct Profiles {
    models: HashMap<String, Value>,
}
fn mode(v: &Value, engine: &str) -> Option<&'static str> {
    let mut controls = Vec::new();
    let mut modes = HashSet::new();
    if let Some(x) = v.get("reasoning_effort") {
        controls.push(("effort", x))
    }
    if let Some(x) = v.get("enable_thinking") {
        if engine != "tabbyapi" {
            return None;
        }
        controls.push(("enabled", x))
    }
    for key in [
        "reasoning",
        "chat_template_kwargs",
        "template_vars",
        "thinking",
    ] {
        let Some(obj) = v.get(key) else { continue };
        if engine == "tabbyapi" && key == "chat_template_kwargs" && v.get("template_vars").is_some()
        {
            continue;
        }
        let obj = obj.as_object()?;
        if key == "thinking" || (key == "template_vars" && engine != "tabbyapi") {
            return None;
        }
        let effort = if key == "reasoning" {
            "effort"
        } else {
            "reasoning_effort"
        };
        let enabled = if key == "reasoning" {
            "enabled"
        } else {
            "enable_thinking"
        };
        for (name, value) in obj {
            if name == effort {
                if engine == "prism" && (key == "reasoning" || value == "none") {
                    return None;
                }
                controls.push(("effort", value))
            } else if name == enabled {
                if key == "reasoning" && engine != "tabbyapi" {
                    return None;
                }
                if STRATA.contains(&engine) && value == true {
                    continue;
                }
                controls.push(("enabled", value))
            }
        }
    }
    for (kind, v) in controls {
        if kind == "enabled" {
            modes.insert(if v.as_bool()? {
                "thinking"
            } else {
                "non_thinking"
            });
        } else {
            let text = v.as_str()?;
            let text = if engine == "prism" {
                text.to_owned()
            } else {
                text.trim().to_lowercase()
            };
            if text == "none" || (engine != "prism" && text == "off") {
                modes.insert("non_thinking");
            } else if ["low", "medium", "xhigh"].contains(&text.as_str())
                || (engine != "prism" && text == "high")
            {
                modes.insert("thinking");
            } else {
                return None;
            }
        }
    }
    if modes.len() == 1 {
        modes.into_iter().next()
    } else {
        None
    }
}
fn thinking(v: &Value) -> bool {
    BUDGET
        .iter()
        .chain(
            [
                "reasoning_effort",
                "enable_thinking",
                "reasoning",
                "thinking",
            ]
            .iter(),
        )
        .any(|k| v.get(k).is_some())
        || ["chat_template_kwargs", "template_vars"].iter().any(|k| {
            v.get(k).is_some_and(|o| {
                !o.is_object()
                    || ["enable_thinking", "reasoning_effort"]
                        .iter()
                        .chain(BUDGET.iter())
                        .any(|k| o.get(k).is_some())
            })
        })
}
impl Profiles {
    pub fn load(path: Option<&str>) -> Result<Self> {
        let Some(path) = path.filter(|p| !p.is_empty()) else {
            return Ok(Self::default());
        };
        let read = || -> Result<Self> {
            let bytes = std::fs::read(path)?;
            let data = crate::json_wire::unique(&bytes)?;
            Self::parse(data)
        };
        read().map_err(|e| anyhow::anyhow!("model profiles {path}: {e}"))
    }
    pub fn parse(data: Value) -> Result<Self> {
        if data.as_object().is_none_or(|o| o.len() != 2)
            || data["version"].as_i64() != Some(1)
            || !data["profiles"].is_array()
        {
            bail!("expected version 1 and a profiles list")
        }
        let mut this = Self::default();
        let mut names = HashSet::new();
        for p in data["profiles"].as_array().unwrap() {
            let fields = [
                "id",
                "models",
                "defaults",
                "thinking",
                "non_thinking",
                "sources",
                "checked_at",
                "notes",
                "installed_revision",
            ];
            if p.as_object()
                .is_none_or(|o| o.keys().any(|k| !fields.contains(&k.as_str())))
            {
                bail!("unknown profile fields")
            }
            let Some(name) = p["id"].as_str().filter(|s| !s.trim().is_empty()) else {
                bail!("profile IDs must be nonempty and unique")
            };
            if !names.insert(name) {
                bail!("profile IDs must be nonempty and unique")
            }
            let Some(models) = p["models"].as_array().filter(|v| !v.is_empty()) else {
                bail!("{name}: models must be a nonempty list")
            };
            let mut engines = HashSet::new();
            for m in models {
                let Some(model) = m.as_str() else {
                    bail!("{name}: expected an exact supported backend/model ID")
                };
                let Some((engine, native)) = model.split_once('/') else {
                    bail!("{name}: expected an exact supported backend/model ID")
                };
                if (!STRATA.contains(&engine) && !["prism", "tabbyapi"].contains(&engine))
                    || native.trim().is_empty()
                    || model.contains(['*', '?', '[', ']'])
                {
                    bail!("{name}: expected an exact supported backend/model ID")
                }
                if this.models.insert(model.into(), p.clone()).is_some() {
                    bail!("duplicate model match: {model}")
                }
                engines.insert(engine);
            }
            let strata = engines.iter().any(|e| STRATA.contains(e));
            for key in ["defaults", "thinking", "non_thinking"] {
                let empty = Value::Object(Map::new());
                let values = p.get(key).unwrap_or(&empty);
                let params = [
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "presence_penalty",
                    "frequency_penalty",
                    "reasoning_budget_tokens",
                ];
                let Some(values) = values
                    .as_object()
                    .filter(|v| v.keys().all(|k| params.contains(&k.as_str())))
                else {
                    bail!("{name}.{key}: expected supported parameter names")
                };
                for (k, v) in values {
                    let Some(n) = v.as_f64().filter(|v| v.is_finite()) else {
                        bail!("{name}.{key}.{k}: expected a finite number")
                    };
                    let valid = match k.as_str() {
                        "temperature" => (0.0..=2.0).contains(&n),
                        "top_p" => n > 0.0 && n <= 1.0,
                        "min_p" => (0.0..=1.0).contains(&n),
                        "repetition_penalty" => n > 0.0,
                        "top_k" | "reasoning_budget_tokens" => {
                            v.as_u64().is_some()
                                && (k != "top_k" || !strata || (1.0..=64.0).contains(&n))
                        }
                        _ => (if strata { 0.0 } else { -2.0 }..=2.0).contains(&n),
                    };
                    if !valid {
                        bail!("{name}.{key}.{k}: value outside supported backend range")
                    }
                }
            }
            for key in ["checked_at", "notes", "installed_revision"] {
                if p.get(key).is_some_and(|v| !v.is_string()) {
                    bail!("{name}.{key}: expected text")
                }
            }
            if p.get("sources").is_some_and(|v| {
                v.as_array()
                    .is_none_or(|a| a.iter().any(|v| !v.is_string()))
            }) {
                bail!("{name}.sources: expected a list of source URLs")
            }
        }
        Ok(this)
    }
    pub fn apply(&self, payload: &Value, engine: &str, model: Option<&str>) -> Value {
        let Some(profile) = model.and_then(|m| self.models.get(&format!("{engine}/{m}"))) else {
            return payload.clone();
        };
        let mode = mode(payload, engine);
        let mut defaults = profile["defaults"].as_object().cloned().unwrap_or_default();
        if let Some(m) = mode
            && let Some(o) = profile[m].as_object()
        {
            defaults.extend(o.clone());
        }
        let mut result = payload.clone();
        let mut added = Vec::new();
        for (k, v) in defaults {
            if k == "reasoning_budget_tokens" && thinking(payload) {
                continue;
            }
            let wire = if k == "repetition_penalty" && engine == "prism" {
                "repeat_penalty"
            } else {
                &k
            };
            if if k == "repetition_penalty" {
                REP.iter().any(|k| payload.get(k).is_some())
            } else {
                payload.get(wire).is_some()
            } {
                continue;
            }
            result[wire] = v;
            added.push(wire.to_owned());
        }
        let wire = if engine == "prism" {
            "repeat_penalty"
        } else {
            "repetition_penalty"
        };
        if payload.get(wire).is_none() {
            let aliases = if engine == "tabbyapi" {
                &["rep_pen", "repeat_penalty"][..]
            } else {
                REP
            };
            if let Some(v) = aliases.iter().find_map(|k| payload.get(k)) {
                result[wire] = v.clone();
            }
        }
        if !added.is_empty() {
            log(
                10,
                "st_proxy.model_profiles",
                format!(
                    "model profile={} mode={} default_fields={}",
                    profile["id"].as_str().unwrap(),
                    mode.unwrap_or("unknown"),
                    added.join(",")
                ),
            );
        }
        result
    }
    pub fn body(&self, body: &[u8], engine: &str, model: Option<&str>) -> Vec<u8> {
        if self.models.is_empty() {
            return body.into();
        }
        let Ok(payload) = serde_json::from_slice::<Value>(body) else {
            return body.into();
        };
        if !payload.is_object() {
            return body.into();
        }
        let result = self.apply(
            &payload,
            engine,
            model.or_else(|| payload["model"].as_str()),
        );
        if result == payload {
            body.into()
        } else {
            crate::json_wire::dumps(&result, false, false).into_bytes()
        }
    }
}
