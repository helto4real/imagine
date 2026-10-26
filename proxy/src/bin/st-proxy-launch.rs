//! Resolve launcher options using the proxy's CLI, then run its stack supervisor.
use anyhow::{Context, Result, bail};
use st_proxy::config::Config;
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
        if matches!(
            flag.as_str(),
            "--help" | "--check-backend" | "--print-backends"
        ) {
            direct = true;
        }
        let key = if flag == "--kobold-url" {
            "LLM_URL".into()
        } else if flag == "--backend-url" {
            // The stack and proxy both read a backend's origin as ST_PROXY_<ID>_URL.
            let (id, _) = value
                .split_once('=')
                .with_context(|| "argument --backend-url: expected ID=URL")?;
            format!("{}_URL", id.trim().to_uppercase())
        } else {
            flag[2..].replace('-', "_").to_uppercase()
        };
        if !matches!(
            flag.as_str(),
            "--help" | "--check-backend" | "--write-kobold-model-cache" | "--print-backends"
        ) {
            let value = match value.split_once('=') {
                Some((_, origin)) if flag == "--backend-url" => origin.trim().to_owned(),
                _ => value.clone(),
            };
            env.insert(format!("ST_PROXY_{key}"), value);
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
    env.entry("ST_PROXY_LLM_BACKEND".into())
        .or_insert_with(|| "all".into());
    // Validate before the supervisor starts or stops anything. Retain the
    // canonical argument list so mutual exclusions are checked as well.
    let (config, _) = Config::parse(proxy_args.clone(), &env)?;
    // One backend's origin, from the backends file and its overrides.
    if !config.all() && env.get("ST_PROXY_LLM_URL").is_none_or(String::is_empty) {
        env.insert("ST_PROXY_LLM_URL".into(), config.llm_url);
    }
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
        // Synthetic backends: no launcher paths or weights are needed.
        // Each call has its own file: tests run in parallel.
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "st-launch-backends-{}-{n}.json",
            std::process::id()
        ));
        let backends = [("koboldcpp", "koboldcpp", 5001), ("sc117", "strata", 48640),
            ("orca_128k", "strata", 48646), ("swift", "strata", 48648), ("swift_128k", "strata", 48650)]
            .map(|(id, adapter, port)| {
                format!(r#"{{"id":"{id}","label":"Own {id}","adapter":"{adapter}","origin":"http://127.0.0.1:{port}"}}"#)
            })
            .join(",");
        std::fs::write(&path, format!(r#"{{"schema":1,"backends":[{backends}]}}"#)).unwrap();
        HashMap::from([
            ("ST_PROXY_MCP_OUTPUT_DIR".into(), "/tmp/synthetic".into()),
            (
                "ST_PROXY_BACKENDS".into(),
                path.to_string_lossy().into_owned(),
            ),
        ])
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
    #[test]
    fn backend_origins_and_listing_come_from_the_backends_file() {
        let p = plan(
            ["--llm-b", "swift", "--swift-url", "http://127.0.0.1:12001"]
                .map(str::to_owned)
                .to_vec(),
            env(),
        )
        .unwrap();
        assert_eq!(p.env["ST_PROXY_SWIFT_URL"], "http://127.0.0.1:12001");
        assert_eq!(p.env["ST_PROXY_LLM_URL"], "http://127.0.0.1:12001");
        let p = plan(
            ["--backend-url", "sc117=http://127.0.0.1:12002"]
                .map(str::to_owned)
                .to_vec(),
            env(),
        )
        .unwrap();
        assert_eq!(p.env["ST_PROXY_SC117_URL"], "http://127.0.0.1:12002");
        assert_eq!(p.env["ST_PROXY_LLM_BACKEND"], "all");
        assert!(!p.env.contains_key("ST_PROXY_LLM_URL"));
        assert!(plan(vec!["--llm-b=orca".into()], env()).is_err());
        assert!(
            plan(
                vec!["--nonexistent-url".into(), "http://127.0.0.1:1".into()],
                env()
            )
            .is_err()
        );
        assert!(
            plan(
                ["--print-backends", "shell"].map(str::to_owned).to_vec(),
                env()
            )
            .unwrap()
            .direct
        );
    }
}
