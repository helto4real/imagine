//! Run an explicitly selected isolated developer worker under the shared lease.
use anyhow::{Context, Result, bail};
use imagine_native_gpu_reservation as gpu_reserved_job;
use std::{ffi::OsString, path::PathBuf, time::Duration};
#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let (mut socket, mut job, mut bytes, mut timeout, mut manifest) =
        (None, None, None, None, None);
    loop {
        let flag=args.next().context("Expected --socket PATH --job ID --bytes N --timeout SECONDS --manifest FILE -- COMMAND [ARG...]")?;
        if flag == "--" {
            break;
        }
        let value = args.next().context("Missing option value")?;
        match flag.to_str() {
            Some("--socket") => socket = Some(PathBuf::from(value)),
            Some("--job") => {
                job = Some(
                    value
                        .into_string()
                        .map_err(|_| anyhow::anyhow!("job must be UTF-8"))?,
                )
            }
            Some("--bytes") => {
                bytes = Some(value.to_str().context("invalid bytes")?.parse::<usize>()?)
            }
            Some("--timeout") => {
                timeout = Some(Duration::from_secs(
                    value.to_str().context("invalid timeout")?.parse::<u64>()?,
                ))
            }
            Some("--manifest") => manifest = Some(PathBuf::from(value)),
            _ => bail!("Unknown option"),
        }
    }
    let executable = PathBuf::from(args.next().context("Missing worker executable")?);
    gpu_reserved_job::validate_gpu_worker(&executable)?;
    let arguments: Vec<OsString> = args.collect();
    let socket = socket.context("Missing --socket")?;
    if !socket.is_absolute() {
        bail!("--socket must be absolute, including in a worktree");
    }
    let job = job.context("Missing --job")?;
    let manifest = manifest.context("Missing --manifest")?;
    let report = gpu_reserved_job::run(
        gpu_reserved_job::Request {
            socket: &socket,
            job: &job,
            bytes: bytes.context("Missing --bytes")?,
            executable: &executable,
            arguments: &arguments,
            timeout: timeout.context("Missing --timeout")?,
        },
        async {
            let terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
            if let Ok(mut terminate) = terminate {
                tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
            }
            // Signal setup failure also completes cancellation rather than
            // leaving a worker running without its stop handlers.
        },
    )
    .await?;
    let temporary = manifest.with_extension(format!("{}.tmp", std::process::id()));
    let record = serde_json::json!({"schema":1,"job_id":job,"coordinator_socket":socket,"direct_worker":executable,"report":&report});
    std::fs::write(&temporary, serde_json::to_vec_pretty(&record)?)?;
    std::fs::rename(temporary, manifest)?;
    if report.state != "completed" || report.exit_code != Some(0) || !report.finish_sent {
        bail!("Reserved worker failed: {}", report.state);
    }
    Ok(())
}
