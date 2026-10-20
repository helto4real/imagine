//! Run an explicitly selected isolated developer worker under the shared lease.
use anyhow::{Context, Result, bail};
use imagine_native_gpu_reservation as gpu_reserved_job;
use std::{ffi::OsString, path::PathBuf, time::Duration};
#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let (mut socket, mut job, mut bytes, mut timeout, mut manifest) =
        (None, None, None, None, None);
    let mut process_form = String::from("native");
    let mut namespace_entry = None;
    loop {
        let flag=args.next().context("Expected --socket PATH --job ID --bytes N --timeout SECONDS --manifest FILE -- COMMAND [ARG...]")?;
        if flag == "--" {
            break;
        }
        let value = args.next().context("Missing option value")?;
        match flag.to_str() {
            Some("--namespace-entry") => namespace_entry = Some(PathBuf::from(value)),
            Some("--process-form") => {
                process_form = value
                    .into_string()
                    .map_err(|_| anyhow::anyhow!("process form must be UTF-8"))?
            }
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
    if process_form == "native" {
        gpu_reserved_job::validate_gpu_worker(&executable)?;
    } else if process_form != "bwrap-pid-namespace" {
        bail!("Unknown process form");
    }
    let arguments: Vec<OsString> = args.collect();
    let socket = socket.context("Missing --socket")?;
    if !socket.is_absolute() {
        bail!("--socket must be absolute, including in a worktree");
    }
    let job = job.context("Missing --job")?;
    let manifest = manifest.context("Missing --manifest")?;
    let request = gpu_reserved_job::Request {
        socket: &socket,
        job: &job,
        bytes: bytes.context("Missing --bytes")?,
        executable: &executable,
        arguments: &arguments,
        timeout: timeout.context("Missing --timeout")?,
    };
    let (report, success) = if process_form == "native" {
        let report = gpu_reserved_job::run(request, cancellation()).await?;
        let success =
            report.state == "completed" && report.exit_code == Some(0) && report.finish_sent;
        (serde_json::to_value(report)?, success)
    } else {
        let report = gpu_reserved_job::run_bwrap(
            request,
            gpu_reserved_job::Admission {
                entry: namespace_entry
                    .as_deref()
                    .context("Missing --namespace-entry")?,
                own_root: manifest
                    .parent()
                    .context("Manifest requires a parent directory")?,
            },
            cancellation(),
        )
        .await?;
        let success = report.namespace_exited
            && report.worker.state == "completed"
            && report.worker.exit_code == Some(0)
            && report.worker.finish_sent;
        (serde_json::to_value(report)?, success)
    };
    let temporary = manifest.with_extension(format!("{}.tmp", std::process::id()));
    let record = serde_json::json!({"schema":1,"job_id":job,"coordinator_socket":socket,"direct_worker":executable,"process_form":process_form,"report":&report});
    std::fs::write(&temporary, serde_json::to_vec_pretty(&record)?)?;
    std::fs::rename(temporary, manifest)?;
    if !success {
        bail!("Reserved worker did not complete successfully; see the owned manifest");
    }
    Ok(())
}

async fn cancellation() {
    let terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate());
    if let Ok(mut terminate) = terminate {
        tokio::select! {_=tokio::signal::ctrl_c()=>{},_=terminate.recv()=>{}}
    }
    // Signal setup failure requests cancellation as in the direct-native route.
}
