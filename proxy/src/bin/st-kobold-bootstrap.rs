use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        bail!("expected source and destination")
    };
    let v: Value = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let fields = [
        "host",
        "port",
        "port_param",
        "password",
        "adminpassword",
        "ssl",
        "nocertify",
        "reqtimeout",
        "quiet",
        "debugmode",
        "remotetunnel",
        "adminunloadtimeout",
        "downloaddir",
        "baseconfig",
    ];
    let mut out: serde_json::Map<_, _> = v
        .as_object()
        .context("KoboldCpp profile must be a JSON object")?
        .iter()
        .filter(|(k, _)| fields.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for k in ["nomodel", "usecpu", "admin", "routermode"] {
        out.insert(k.into(), json!(true));
    }
    std::fs::write(
        &args[1],
        format!("{}\n", st_proxy::mcp::python_json(&Value::Object(out))),
    )?;
    Ok(())
}
