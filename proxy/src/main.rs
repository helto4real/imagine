use anyhow::Result;
use st_proxy::{config::Config, logging::log, service::Service};
#[tokio::main]
async fn main() {
    if run().await.is_err() {
        eprintln!("st-proxy: startup or shutdown failed");
        std::process::exit(1);
    }
}
async fn run() -> Result<()> {
    let limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::setrlimit(libc::RLIMIT_CORE, &limit) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let (config, cli) = match Config::parse(std::env::args().skip(1), &std::env::vars().collect()) {
        Ok(v) => v,
        Err(_) => {
            eprintln!("st-proxy: invalid command line or configuration");
            std::process::exit(2)
        }
    };
    if cli.help {
        // Retain the imported reference fixture; hide the removed Imagine option.
        let help = include_str!("../reference/cli-help.txt")
            .replace("                [--imagine-access-file IMAGINE_ACCESS_FILE]\n", "")
            .replace("  --imagine-access-file IMAGINE_ACCESS_FILE\n                        operator-owned Imagine access.json for MCP assets and\n                        GPU release\n", "");
        // Backends come from the backends file, not a fixed list.
        let help = help
            .replace(
                "{koboldcpp,ollama,sc117,orca,sc117_128k,orca_128k,prism,tabbyapi,all}",
                "BACKEND",
            )
            .lines()
            .filter(|l| {
                !["tabby", "prism", "sc117", "orca"].iter().any(|id| {
                    l.contains(&format!("--{id}-url")) || l.contains(&format!("--{id}-128k-url"))
                })
            })
            .map(|l| format!("{l}\n"))
            .collect::<String>();
        print!("{help}");
        println!(
            "  --backends BACKENDS   LLM backends file (default: CONFIG/proxy/backends.json)\n  --llm-backend BACKEND a backend id from that file, or all\n  --exclude-backends IDS\n                        comma-separated backend ids that all leaves out\n  --backend-url ID=URL, --ID-url URL\n                        a backend's origin (also ST_PROXY_<ID>_URL)\n  --print-backends table|shell\n                        list the configured backends and exit"
        );
        println!(
            "\n  --comfy-enabled true|false\n                        enable Comfy routes and GPU cleanup (binary default: true; root launcher: false)\n  --config CONFIG       editable configuration root (MCP texts under mcp/)"
        );
        return Ok(());
    }
    if let Some(format) = &cli.print_backends {
        print!(
            "{}",
            if format == "shell" {
                st_proxy::registry::shell(&config.registry)
            } else {
                st_proxy::registry::table(&config.registry)
            }
        );
        return Ok(());
    }
    st_proxy::logging::configure(&cli.log_level);
    if cli.check_backend {
        if cli.backend_check_timeout <= 0.0 {
            eprintln!("st-proxy: error: backend_check_timeout must be greater than zero");
            std::process::exit(2)
        }
        let mut config = config;
        let t = cli.backend_check_timeout;
        config.request_timeout = config.request_timeout.min(t);
        config.reload_timeout = config.reload_timeout.min(t);
        config.unload_timeout = config.unload_timeout.min(t);
        config.poll_interval = config.poll_interval.min(t);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(t))
            .pool_max_idle_per_host(0)
            .build()?;
        let mut b = st_proxy::backend::Backends::new(&config, client);
        b.validate().await?;
        use st_proxy::registry::Adapter;
        if config.kobold_router_mode
            || !matches!(config.adapter(), Some(Adapter::Koboldcpp | Adapter::Ollama))
        {
            b.observe().await?;
        } else {
            b.entries[0].snapshot().await?;
        }
        if let Some(path) = cli.write_kobold_model_cache {
            if config.adapter() != Some(Adapter::Koboldcpp) || !config.kobold_router_mode {
                anyhow::bail!("model cache export requires KoboldCpp Router mode")
            };
            let bytes = serde_json::to_vec(&b.entries[0].catalog().await?)?;
            let path = std::path::Path::new(&path);
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, bytes)?;
            std::fs::rename(tmp, path)?;
        }
        log(
            20,
            "st_proxy.cli",
            format!("LLM backend ready: backend={}", config.label()),
        );
        return Ok(());
    }
    // The MCP texts are read once; a restart applies edits.
    st_proxy::config_files::init(config.config.clone().into());
    if config.mcp_port.is_some() && st_proxy::mcp::check_config().is_err() {
        eprintln!("st-proxy: MCP configuration under --config is missing or invalid");
        std::process::exit(2)
    }
    for (label, reason) in config.skipped() {
        log(
            30,
            "st_proxy.cli",
            format!("LLM backend skipped: {label}: {reason}"),
        );
    }
    let running = Service::start(config).await?;
    log(
        20,
        "st_proxy.cli",
        format!(
            "broker ready: chat=http://{}:{} image=http://{}:{} state={}",
            running.service.config.listen_host,
            running.chat_port,
            running.service.config.listen_host,
            running.image_port,
            running.service.coordinator.status()["state"]
                .as_str()
                .unwrap()
        ),
    );
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
    running.close().await;
    Ok(())
}
