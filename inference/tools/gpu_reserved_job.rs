//! Development controller using the product's existing Ticket implementation.
//! The command must be an already isolated development worker. No fallback
//! coordinator or alternate lock is created, and command arguments are not logged.
use crate::coordination::Ticket;
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{ffi::OsString, future::Future, path::Path, process::Stdio, time::Duration};
use tokio::{
    process::Command,
    time::{Instant, sleep_until},
};
/// This small controller owns one direct GPU process. GPU-capable descendants
/// and launchers that create a new session are outside this harness's contract.
/// Invoke the native worker itself, not Python/test_isolation/bwrap/a shell.
/// ELF and wrapper checks prevent accidental launcher use; the reviewed native
/// worker must also satisfy the single-GPU-process precondition.
pub fn validate_gpu_worker(path: &Path) -> Result<()> {
    use std::io::Read;
    let path = std::fs::canonicalize(path).context("worker executable unavailable")?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("invalid worker executable name")?;
    if name.starts_with("python")
        || matches!(
            name,
            "sh" | "bash" | "dash" | "zsh" | "env" | "bwrap" | "timeout" | "setsid" | "sudo"
        )
    {
        bail!("GPU command must be the direct native worker, not an interpreter or launcher");
    }
    let mut magic = [0u8; 4];
    std::fs::File::open(path)?.read_exact(&mut magic)?;
    if magic != *b"\x7fELF" {
        bail!("GPU worker must be a native ELF executable, not a script");
    }
    Ok(())
}
pub struct Request<'a> {
    pub socket: &'a Path,
    pub job: &'a str,
    /// Product protocol request cost; this is not a VRAM allocation estimate.
    pub bytes: usize,
    pub executable: &'a Path,
    pub arguments: &'a [OsString],
    pub timeout: Duration,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub state: &'static str,
    pub child_started: bool,
    pub child_reaped: bool,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub finish_sent: bool,
}
impl Report {
    fn idle(state: &'static str) -> Self {
        Self {
            state,
            child_started: false,
            child_reaped: false,
            exit_code: None,
            signal: None,
            finish_sent: false,
        }
    }
}
/// Cancellation of an acquired job keeps the Ticket alive until its own worker
/// has actually terminated. A caller must await this future through cleanup.
pub async fn run(request: Request<'_>, cancel: impl Future<Output = ()>) -> Result<Report> {
    let deadline = Instant::now() + request.timeout;
    tokio::pin!(cancel);
    let acquire = async {
        Ticket::register(request.socket, request.job, request.bytes)
            .await?
            .acquire()
            .await
    };
    let mut ticket = tokio::select! {
        result=acquire=>result.context("GPU reservation failed; worker was not started")?,
        ()=&mut cancel=>return Ok(Report::idle("cancelled_while_queued")),
        ()=sleep_until(deadline)=>return Ok(Report::idle("timed_out_while_queued")),
    };
    let child = Command::new(request.executable)
        .args(request.arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            // No child exists, so releasing this grant is safe.
            let _ = ticket.finish().await;
            return Err(error).context("reserved worker could not be started");
        }
    };
    let (state, observed) = tokio::select! {
        result=child.wait()=>("completed",Some(result)),
        ()=ticket.lost()=>("reservation_lost",None),
        ()=&mut cancel=>("cancelled",None),
        ()=sleep_until(deadline)=>("timed_out",None),
    };
    // Do not propagate a wait/kill error past the live Ticket. Keep ownership
    // and retry reaping: losing that ownership would release the reservation
    // while the worker's actual termination was still unknown.
    let status = match observed {
        Some(Ok(status)) => status,
        _ => {
            let _ = child.start_kill();
            loop {
                match child.wait().await {
                    Ok(status) => break status,
                    Err(_) => {
                        let _ = child.start_kill();
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }
    };
    use std::os::unix::process::ExitStatusExt;
    let mut report = Report {
        state,
        child_started: true,
        child_reaped: true,
        exit_code: status.code(),
        signal: status.signal(),
        finish_sent: false,
    };
    // The send is deliberately after wait. It has no release acknowledgement;
    // the report therefore says 'finish_sent', not 'release observed'.
    report.finish_sent = ticket.finish().await.is_ok();
    Ok(report)
}
