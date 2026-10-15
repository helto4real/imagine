//! Wire validation used by the pinned MCP SDK, without a Python runtime.
use serde_json::{Value, json};
pub const HANDSHAKE: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];
fn repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::String(s) => {
            let quote = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut r = String::from(quote);
            for c in s.chars() {
                match c {
                    '\n' => r.push_str("\\n"),
                    '\r' => r.push_str("\\r"),
                    '\t' => r.push_str("\\t"),
                    '\\' => r.push_str("\\\\"),
                    c if c == quote => {
                        r.push('\\');
                        r.push(c)
                    }
                    c if c.is_control() => r.push_str(&format!("\\x{:02x}", c as u32)),
                    c => r.push(c),
                }
            }
            r.push(quote);
            r
        }
        Value::Array(a) => format!("[{}]", a.iter().map(repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", repr(&json!(k)), repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => v.to_string(),
    }
}
fn kind(v: &Value) -> &str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
        Value::Number(n) if n.is_f64() => "float",
        _ => "int",
    }
}
fn issue(location: &str, code: &str, message: &str, input: &Value) -> String {
    let r = repr(input);
    let c = r.chars().collect::<Vec<_>>();
    let r = if c.len() > 50 {
        format!(
            "{}...{}",
            c[..25].iter().collect::<String>(),
            c[c.len() - 24..].iter().collect::<String>()
        )
    } else {
        r
    };
    format!(
        "{location}\n  {message} [type={code}, input_value={r}, input_type={}]\n    For further information visit https://errors.pydantic.dev/2.14/v/{code}",
        kind(input)
    )
}
fn report(name: &str, issues: &[String]) -> String {
    format!(
        "{} validation error{} for {name}\n{}",
        issues.len(),
        if issues.len() == 1 { "" } else { "s" },
        issues.join("\n")
    )
}
fn check(value: &Value, ty: &str, location: &str, issues: &mut Vec<String>) {
    let ok = match ty {
        "string" => value.is_string(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "integer" => value.is_i64() || value.is_u64(),
        _ => true,
    };
    if !ok {
        let (code, message) = match ty {
            "string" => ("string_type", "Input should be a valid string"),
            "object" => ("dict_type", "Input should be a valid dictionary"),
            "array" => ("list_type", "Input should be a valid list"),
            _ => ("int_type", "Input should be a valid integer"),
        };
        issues.push(issue(location, code, message, value));
    }
}
pub fn tool_arguments(name: &str, args: &Value) -> Result<Value, String> {
    let tools: Value = serde_json::from_str(include_str!("../mcp/tools.json")).unwrap();
    let Some(tool) = tools.as_array().unwrap().iter().find(|t| t["name"] == name) else {
        return Err(format!("Unknown tool: {name}"));
    };
    let schema = &tool["inputSchema"];
    let mut args = if args.is_null() {
        json!({})
    } else {
        args.clone()
    };
    let mut issues = Vec::new();
    if !args.is_object() {
        return Err(format!("Error executing tool {name}"));
    }
    if let Some(key) = args
        .as_object()
        .unwrap()
        .keys()
        .find(|k| schema["properties"].get(*k).is_none())
    {
        return Err(format!(
            "Error executing tool {name}: unknown argument: {key}"
        ));
    }
    for (key, spec) in schema["properties"].as_object().unwrap() {
        if spec["type"] != "string"
            && let Some(s) = args[key].as_str()
            && let Ok(v) = serde_json::from_str::<Value>(s)
            && (v.is_null() || v.is_array() || v.is_object())
        {
            args[key] = v;
        }
    }
    for (key, spec) in schema["properties"].as_object().unwrap() {
        let Some(value) = args.get(key) else {
            if schema["required"]
                .as_array()
                .is_some_and(|a| a.contains(&json!(key)))
            {
                issues.push(issue(key, "missing", "Field required", &args));
            }
            continue;
        };
        if key == "wait" {
            let parsed = if let Some(b) = value.as_bool() {
                Some(b)
            } else if value == 0 {
                Some(false)
            } else if value == 1 {
                Some(true)
            } else {
                value
                    .as_str()
                    .and_then(|s| match s.to_ascii_lowercase().as_str() {
                        "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
                        "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
                        _ => None,
                    })
            };
            if let Some(b) = parsed {
                args[key] = json!(b)
            } else {
                let code = if value.is_string() || value.is_number() {
                    "bool_parsing"
                } else {
                    "bool_type"
                };
                let message = if code == "bool_parsing" {
                    "Input should be a valid boolean, unable to interpret input"
                } else {
                    "Input should be a valid boolean"
                };
                issues.push(issue(key, code, message, value));
            }
        } else if key == "references" {
            if !value.is_null() {
                check(value, "array", key, &mut issues);
                if let Some(a) = value.as_array() {
                    for (i, v) in a.iter().enumerate() {
                        check(v, "string", &format!("{key}.{i}"), &mut issues)
                    }
                }
            }
        } else {
            check(value, spec["type"].as_str().unwrap_or(""), key, &mut issues);
            if key == "items"
                && let Some(a) = value.as_array()
            {
                for (i, v) in a.iter().enumerate() {
                    check(v, "object", &format!("items.{i}"), &mut issues)
                }
            }
        }
    }
    if issues.is_empty() {
        Ok(args)
    } else {
        Err(format!(
            "Error executing tool {name}: {}",
            report(&format!("{name}Arguments"), &issues)
        ))
    }
}
pub fn envelope(value: &Value) -> Result<(), String> {
    let mut all = Vec::new();
    for name in [
        "JSONRPCRequest",
        "JSONRPCNotification",
        "JSONRPCResponse",
        "JSONRPCError",
    ] {
        let mut issues = Vec::new();
        if !value.is_object() {
            issues.push(issue(
                name,
                "model_type",
                &format!("Input should be a valid dictionary or instance of {name}"),
                value,
            ));
            all.extend(issues);
            continue;
        }
        if value.get("jsonrpc").is_none() {
            issues.push(issue(
                &format!("{name}.jsonrpc"),
                "missing",
                "Field required",
                value,
            ));
        } else if value["jsonrpc"] != "2.0" {
            issues.push(issue(
                &format!("{name}.jsonrpc"),
                "literal_error",
                "Input should be '2.0'",
                &value["jsonrpc"],
            ));
        }
        if name != "JSONRPCNotification" {
            if let Some(id) = value.get("id") {
                if !(id.is_string()
                    || id.is_i64()
                    || id.is_u64()
                    || (name == "JSONRPCError" && id.is_null()))
                {
                    check(id, "integer", &format!("{name}.id.int"), &mut issues);
                    check(id, "string", &format!("{name}.id.str"), &mut issues);
                }
            } else {
                issues.push(issue(
                    &format!("{name}.id"),
                    "missing",
                    "Field required",
                    value,
                ));
            }
        }
        let field = match name {
            "JSONRPCResponse" => "result",
            "JSONRPCError" => "error",
            _ => "method",
        };
        if let Some(v) = value.get(field) {
            check(
                v,
                if field == "method" {
                    "string"
                } else {
                    "object"
                },
                &format!("{name}.{field}"),
                &mut issues,
            );
        } else {
            issues.push(issue(
                &format!("{name}.{field}"),
                "missing",
                "Field required",
                value,
            ));
        }
        if field == "method"
            && let Some(v) = value.get("params").filter(|v| !v.is_null())
        {
            check(v, "object", &format!("{name}.params"), &mut issues);
        }
        if issues.is_empty() {
            return Ok(());
        }
        all.extend(issues);
    }
    Err(format!(
        "Validation error: {}",
        report(
            "union[JSONRPCRequest,JSONRPCNotification,JSONRPCResponse,JSONRPCError]",
            &all
        )
    ))
}

pub const MODERN: &str = "2026-07-28";
/// The per-request routing envelope used by the reference SDK's modern HTTP entry.
pub fn modern(v: &Value, headers: &axum::http::HeaderMap) -> Result<(), Value> {
    use base64::Engine;
    let error = |code, message: String| json!({"code":code,"message":message});
    let header = |key: &str| headers.get(key).and_then(|v| v.to_str().ok());
    let unsupported = |version: &str| json!({"code":-32022,"message":"Unsupported protocol version","data":{"supported":[MODERN],"requested":version}});
    if v.get("id").is_none() {
        let version = header("mcp-protocol-version").unwrap_or("");
        return if version == MODERN {
            Ok(())
        } else {
            Err(unsupported(version))
        };
    }
    for key in ["mcp-protocol-version", "mcp-method", "mcp-name"] {
        if headers.get_all(key).iter().count() > 1 {
            return Err(error(
                -32020,
                format!("{key} header appears more than once"),
            ));
        }
    }
    let version_key = "io.modelcontextprotocol/protocolVersion";
    let caps_key = "io.modelcontextprotocol/clientCapabilities";
    let Some(meta) = v["params"]["_meta"].as_object() else {
        return Err(error(
            -32602,
            format!(
                "params._meta must be an object carrying the required '{version_key}' and '{caps_key}' envelope keys"
            ),
        ));
    };
    let missing = [version_key, caps_key]
        .into_iter()
        .filter(|k| !meta.contains_key(*k))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(error(
            -32602,
            format!(
                "params._meta is missing the required envelope key(s): {}",
                missing.join(", ")
            ),
        ));
    }
    if header("mcp-protocol-version").is_none()
        || header("mcp-protocol-version") != meta[version_key].as_str()
    {
        return Err(error(
            -32020,
            "mcp-protocol-version header does not match the request envelope's protocol version"
                .into(),
        ));
    }
    if header("mcp-method") != v["method"].as_str() {
        return Err(error(
            -32020,
            "mcp-method header does not match the request body's method".into(),
        ));
    }
    let name_key = match v["method"].as_str() {
        Some("tools/call" | "prompts/get") => Some("name"),
        Some("resources/read") => Some("uri"),
        _ => None,
    };
    if let Some(key) = name_key
        && !v["params"][key].is_null()
    {
        let name = header("mcp-name").and_then(|name| {
            if let Some(encoded) = name
                .strip_prefix("=?base64?")
                .and_then(|s| s.strip_suffix("?="))
            {
                String::from_utf8(
                    base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .ok()?,
                )
                .ok()
            } else {
                Some(name.to_string())
            }
        });
        if name.as_deref() != v["params"][key].as_str() {
            return Err(error(
                -32020,
                format!("mcp-name header does not match the request body's '{key}' parameter"),
            ));
        }
    }
    let version = meta[version_key].as_str().unwrap();
    if version != MODERN {
        return Err(unsupported(version));
    }
    Ok(())
}
