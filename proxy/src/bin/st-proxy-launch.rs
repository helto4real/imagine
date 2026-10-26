//! Resolve launcher options using the proxy's CLI, then run its stack supervisor.
use anyhow::{Context, Result, bail};
use st_proxy::config::{Config, default_origin};
use std::{collections::HashMap, os::unix::process::CommandExt, process::Command};

struct Plan {
    env: HashMap<String, String>,
    stack_args: Vec<String>,
    proxy_args: Vec<String>,
    direct: bool,
}
fn plan(args: Vec<String>, mut env: HashMap<String, String>) -> Result<Plan> {
    let mut stack_args = Vec::new();
    let mut proxy_args = Vec::new();
    let mut direct = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--log" | "--stop" => {
                stack_args.push(arg);
                continue;
            }
            "--proxy-only" => {
                direct = true;
                continue;
            }
            _ => {}
        }
        let (flag, inline) = arg
            .split_once('=')
            .map_or((arg.as_str(), None), |(f, v)| (f, Some(v)));
        let flag = Config::resolve_option(flag)?;
        let switch = matches!(
            flag.as_str(),
            "--help"
                | "--check-backend"
                | "--strict-comfy-routes"
                | "--allow-unknown-comfy-routes"
                | "--kobold-router-mode"
                | "--restore-llm-on-idle"
        );
        if switch && inline.is_some() {
            bail!("argument {flag} does not accept a value");
        }
        let value = if switch {
            "true".to_owned()
        } else {
            inline
                .map(str::to_owned)
                .or_else(|| args.next())
                .with_context(|| format!("argument {flag}: expected one argument"))?
        };
        if matches!(flag.as_str(), "--help" | "--check-backend") {
            direct = true;
        }
        let key = if flag == "--kobold-url" {
            "LLM_URL".into()
        } else {
            flag[2..].replace('-', "_").to_uppercase()
        };
        if !matches!(
            flag.as_str(),
            "--help" | "--check-backend" | "--write-kobold-model-cache"
        ) {
            env.insert(format!("ST_PROXY_{key}"), value.clone());
        }
        if flag == "--allow-unknown-comfy-routes" {
            env.remove("ST_PROXY_STRICT_COMFY_ROUTES");
        }
        proxy_args.push(flag);
        if !switch {
            proxy_args.push(value);
        }
    }
    if direct && !stack_args.is_empty() {
        bail!("stack controls cannot be combined with direct proxy commands");
    }
    let backend = env
        .entry("ST_PROXY_LLM_BACKEND".into())
        .or_insert_with(|| "all".into())
        .clone();
    if env.get("ST_PROXY_LLM_URL").is_none_or(String::is_empty) {
        let key = match backend.as_str() {
            "koboldcpp" | "all" => "ST_PROXY_KOBOLD_URL".to_owned(),
            _ => format!("ST_PROXY_{}_URL", backend.to_uppercase()),
        };
        let origin = env.get(&key).filter(|s| !s.is_empty()).cloned().unwrap_or(
            if matches!(backend.as_str(), "koboldcpp" | "all") {
                "http://127.0.0.1:5001".into()
            } else {
                default_origin(&backend)?
            },
        );
        env.insert("ST_PROXY_LLM_URL".into(), origin);
    }
    // Validate before the supervisor starts or stops anything. Retain the
    // canonical argument list so mutual exclusions are checked as well.
    Config::parse(proxy_args.clone(), &env)?;
    Ok(Plan {
        env,
        stack_args,
        proxy_args,
        direct,
    })
}
fn main() {
    if run().is_err() {
        eprintln!("st-proxy: launch failed");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let plan = plan(
        std::env::args().skip(1).collect(),
        std::env::vars().collect(),
    )?;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap();
    let mut command = if plan.direct {
        if plan.proxy_args.iter().any(|a| a == "--help") {
            println!(
                "start-proxy.sh [--log] [--stop] [proxy options]: starts the selected backends and proxy (default: all).\n--proxy-only starts only the proxy. --help and --check-backend never start the stack."
            );
        }
        let mut command = Command::new(std::env::current_exe()?.with_file_name("st-proxy"));
        command.args(plan.proxy_args);
        command
    } else {
        let executable = std::env::current_exe()?;
        let packaged = executable.with_file_name("st-stack.zsh");
        let script = if packaged.is_file() {
            packaged
        } else {
            root.join("proxy/st-stack.zsh")
        };
        let mut command = Command::new("zsh");
        command.env(
            "ST_STACK_PROXY_BINARY",
            executable.with_file_name("st-proxy"),
        );
        command.arg(script).args(plan.stack_args);
        command
    };
    Err(command.envs(plan.env).exec().into())
}
#[cfg(test)]
mod tests {
    use super::*;
    fn env() -> HashMap<String, String> {
        HashMap::from([("ST_PROXY_MCP_OUTPUT_DIR".into(), "/tmp/synthetic".into())])
    }
    #[test]
    fn ordinary_options_keep_stack_and_resolve_abbreviations() {
        for flag in ["--llm-backend", "--llm-b"] {
            let p = plan(
                [flag, "sc117", "--chat-port", "5002"]
                    .map(str::to_owned)
                    .to_vec(),
                env(),
            )
            .unwrap();
            assert!(!p.direct);
            assert_eq!(p.env["ST_PROXY_LLM_URL"], "http://127.0.0.1:48640");
            assert_eq!(p.env["ST_PROXY_CHAT_PORT"], "5002");
        }
        let p = plan(vec!["--llm-b=orca_128k".into()], env()).unwrap();
        assert_eq!(p.env["ST_PROXY_LLM_URL"], "http://127.0.0.1:48646");
        for (kind, port) in [("swift", 48648), ("swift_128k", 48650)] {
            let p = plan(vec![format!("--llm-b={kind}")], env()).unwrap();
            assert!(!p.direct);
            assert_eq!(
                p.env["ST_PROXY_LLM_URL"],
                format!("http://127.0.0.1:{port}")
            );
            assert_eq!(p.env["ST_PROXY_LLM_BACKEND"], kind);
        }
    }
    #[test]
    fn explicit_origin_and_stack_controls_survive() {
        let p = plan(
            [
                "--log",
                "--llm-b",
                "sc117",
                "--llm-url",
                "http://127.0.0.1:12345",
            ]
            .map(str::to_owned)
            .to_vec(),
            env(),
        )
        .unwrap();
        assert_eq!(p.stack_args, ["--log"]);
        assert_eq!(p.env["ST_PROXY_LLM_URL"], "http://127.0.0.1:12345");
        assert!(plan(vec!["--proxy-only".into()], env()).unwrap().direct);
        assert!(plan(vec!["--image-port".into(), "invalid".into()], env()).is_err());
        assert!(plan(vec!["--stop".into(), "--check-backend".into()], env()).is_err());
    }
}
