//! Development-only process-tree reservation. Bubblewrap creates a PID namespace;
//! the Ticket survives both launcher wait and a pidfd-confirmed namespace-init
//! exit. The command stays blocked until its init pidfd is open. This is separate
//! from the original direct-native-worker contract.
//! Kernel semantics: https://man7.org/linux/man-pages/man7/pid_namespaces.7.html
use crate::{Report, Request, coordination::Ticket};
use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::{
    future::Future,
    os::unix::fs::DirBuilderExt,
    os::{
        fd::{AsRawFd, FromRawFd, OwnedFd},
        unix::process::ExitStatusExt,
    },
    path::Path,
    process::Stdio,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, unix::AsyncFd},
    net::{UnixListener, UnixStream},
    process::{Child, Command},
    time::{Instant, sleep_until},
};

pub struct Admission<'a> {
    pub entry: &'a Path,
    pub own_root: &'a Path,
}
struct GateDirectory(std::path::PathBuf);
impl Drop for GateDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0.join("gate.sock"));
        let _ = std::fs::remove_dir(&self.0);
    }
}
fn gate_directory(root: &Path) -> Result<GateDirectory> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if !root.is_absolute() {
        bail!("Admission root must be absolute");
    }
    loop {
        let path = root.join(format!(
            "h3-admission-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::DirBuilder::new().mode(0o700).create(&path) {
            Ok(()) => return Ok(GateDirectory(path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct NamespaceReport {
    pub worker: Report,
    pub namespace_init: Option<i32>,
    pub namespace_exited: bool,
}
fn idle(state: &'static str) -> NamespaceReport {
    NamespaceReport {
        worker: Report {
            state,
            child_started: false,
            child_reaped: false,
            exit_code: None,
            signal: None,
            finish_sent: false,
        },
        namespace_init: None,
        namespace_exited: false,
    }
}
fn stop_init(fd: &AsyncFd<OwnedFd>) -> std::io::Result<()> {
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            fd.get_ref().as_raw_fd(),
            libc::SIGKILL,
            std::ptr::null::<libc::siginfo_t>(),
            0,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        let e = std::io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(e)
        }
    }
}
async fn reap(child: &mut Child) -> std::process::ExitStatus {
    loop {
        match child.wait().await {
            Ok(s) => return s,
            Err(_) => {
                let _ = child.start_kill();
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}
async fn confirm_namespace_exit(fd: &AsyncFd<OwnedFd>) {
    loop {
        match fd.readable().await {
            Ok(_) => return,
            Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}
/// The caller provides reviewed filesystem/device mounts and an explicit ELF
/// bubblewrap path. No mounts, model paths, source trees or GPU devices are added
/// implicitly. Unlike `run`, this route supports a trusted interpreter/launcher
/// only inside the owned PID namespace. Callers must await cleanup to completion.
/// It is not an adversarial sandbox: kernel/runtime failure of the controller
/// itself remains outside the ordinary cancel/timeout/lost-reservation contract.
pub async fn run_bwrap(
    request: Request<'_>,
    admission: Admission<'_>,
    cancel: impl Future<Output = ()>,
) -> Result<NamespaceReport> {
    if !request.socket.is_absolute() {
        bail!("Coordinator socket must be absolute");
    }
    let separator = request
        .arguments
        .iter()
        .position(|a| a.as_os_str() == "--")
        .context("Namespace args require -- before the explicit worker")?;
    if separator + 1 == request.arguments.len() {
        bail!("Missing namespace worker");
    }
    if !admission.entry.is_absolute() {
        bail!("Native admission executable must be absolute");
    }
    crate::validate_gpu_worker(admission.entry)?;
    let directory = gate_directory(admission.own_root)?;
    let mut token_bytes = [0u8; 32];
    {
        use std::io::Read;
        std::fs::File::open("/dev/urandom")?.read_exact(&mut token_bytes)?;
    }
    let token = token_bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    // Resolve the owned directory through its fd so deep worktree paths do
    // not exceed sockaddr_un.sun_path. The socket itself stays in own_root.
    let directory_fd = std::fs::File::open(&directory.0)?;
    let gate_listener = UnixListener::bind(format!(
        "/proc/self/fd/{}/gate.sock",
        directory_fd.as_raw_fd()
    ))?;
    let name = std::fs::canonicalize(request.executable)?;
    if name.file_name().and_then(|n| n.to_str()) != Some("bwrap") {
        bail!("PID-namespace execution requires the explicit bubblewrap ELF");
    }
    // Keep the executable open across the spawn; diagnostics record its inode.
    let executable = std::fs::File::open(&name)?;
    let mut magic = [0u8; 4];
    use std::os::unix::fs::FileExt;
    executable.read_exact_at(&mut magic, 0)?;
    if magic != *b"\x7fELF" {
        bail!("bubblewrap must be ELF");
    }
    for arg in request
        .arguments
        .iter()
        .take_while(|a| a.as_os_str() != "--")
    {
        if matches!(
            arg.to_str().and_then(|a| a.split('=').next()),
            Some(
                "--info-fd"
                    | "--json-status-fd"
                    | "--block-fd"
                    | "--userns-block-fd"
                    | "--args"
                    | "--pidns"
                    | "--as-pid-1"
            )
        ) {
            bail!("Caller cannot override namespace supervision descriptors or PID ownership");
        }
    }
    let (info_parent, info_child) = std::os::unix::net::UnixStream::pair()?;
    info_parent.set_nonblocking(true)?;
    let info = UnixStream::from_std(info_parent)?;
    let info_fd = info_child.as_raw_fd();
    let exe_fd = executable.as_raw_fd();
    let mut command = Command::new(format!("/proc/self/fd/{exe_fd}"));
    command
        .args([
            "--unshare-pid",
            "--die-with-parent",
            "--new-session",
            "--info-fd",
            &info_fd.to_string(),
        ])
        .args(&request.arguments[..separator])
        .arg("--bind")
        .arg(&directory.0)
        .arg("/imagine-admission")
        .arg("--ro-bind")
        .arg(admission.entry)
        .arg("/imagine-namespace-entry")
        .args([
            "--",
            "/imagine-namespace-entry",
            "/imagine-admission/gate.sock",
        ])
        .arg(&token)
        .args(&request.arguments[separator + 1..])
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);
    unsafe {
        command.pre_exec(move || {
            for fd in [info_fd, exe_fd] {
                if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                    return Err(std::io::Error::last_os_error());
                }
            }
            Ok(())
        });
    }
    let deadline = Instant::now() + request.timeout;
    tokio::pin!(cancel);
    let acquire = async {
        Ticket::register(request.socket, request.job, request.bytes)
            .await?
            .acquire()
            .await
    };
    let mut ticket = tokio::select! {result=acquire=>result.context("GPU reservation failed; namespace was not started")?,()=&mut cancel=>return Ok(idle("cancelled_while_queued")),()=sleep_until(deadline)=>return Ok(idle("timed_out_while_queued"))};
    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            let _ = ticket.finish().await;
            return Err(e).context("reserved namespace launcher could not start");
        }
    };
    drop(info_child);

    let mut bytes = vec![];
    let mut bounded_info = info.take(8192);
    let setup = tokio::select! {
        result=bounded_info.read_to_end(&mut bytes)=>result.map(|_|"ready"),
        ()=&mut cancel=>Ok("cancelled_before_ready"),
        ()=ticket.lost()=>Ok("reservation_lost_before_ready"),
        ()=sleep_until(deadline)=>Ok("timed_out_before_ready"),
    };
    let pid = if matches!(setup, Ok("ready")) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v["child-pid"].as_i64())
            .and_then(|p| i32::try_from(p).ok())
            .filter(|p| *p > 0)
    } else {
        None
    };
    let init = pid.and_then(|pid| {
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
        if fd < 0 {
            return None;
        }
        unsafe { AsyncFd::register(OwnedFd::from_raw_fd(fd as i32)) }.ok()
    });
    let Some(init) = init else {
        // No native entry received G. Closing this admission listener can only
        // cause refusal/EOF; it never admits an interpreter to GPU execution.
        let _ = child.start_kill();
        let status = reap(&mut child).await;
        let mut report = idle(match setup {
            Ok("ready") | Err(_) => "namespace_setup_failed",
            Ok(s) => s,
        });
        report.worker.child_started = true;
        report.worker.child_reaped = true;
        report.worker.exit_code = status.code();
        report.worker.signal = status.signal();
        report.worker.finish_sent = ticket.finish().await.is_ok();
        return Ok(report);
    };
    let accepted = tokio::select! {peer=gate_listener.accept()=>peer.map(|(p,_)|p),()=&mut cancel=>Err(std::io::Error::new(std::io::ErrorKind::Interrupted,"cancelled")),()=ticket.lost()=>Err(std::io::Error::new(std::io::ErrorKind::ConnectionAborted,"reservation_lost")),()=sleep_until(deadline)=>Err(std::io::Error::new(std::io::ErrorKind::TimedOut,"timed_out"))};
    let state = match accepted {
        Ok(mut gate) => {
            let valid = gate
                .peer_cred()
                .map(|peer| peer.uid() == unsafe { libc::geteuid() } && peer.pid().is_some())
                .unwrap_or(false);
            if !valid {
                "namespace_admission_rejected"
            } else {
                let mut proof = vec![0u8; token.len()];
                let ready = tokio::select! {ready=gate.read_exact(&mut proof)=>ready,()=&mut cancel=>Err(std::io::Error::new(std::io::ErrorKind::Interrupted,"cancelled")),()=ticket.lost()=>Err(std::io::Error::new(std::io::ErrorKind::ConnectionAborted,"reservation_lost")),()=sleep_until(deadline)=>Err(std::io::Error::new(std::io::ErrorKind::TimedOut,"timed_out"))};
                if ready.is_err()
                    || proof != token.as_bytes()
                    || gate.write_all(b"G").await.is_err()
                {
                    "namespace_admission_failed"
                } else {
                    tokio::select! {result=child.wait()=>if result.is_ok(){"completed"}else{"wait_failed"},()=&mut cancel=>"cancelled",()=ticket.lost()=>"reservation_lost",()=sleep_until(deadline)=>"timed_out"}
                }
            }
        }
        Err(e) => match e.kind() {
            std::io::ErrorKind::Interrupted => "cancelled",
            std::io::ErrorKind::ConnectionAborted => "reservation_lost",
            std::io::ErrorKind::TimedOut => "timed_out",
            _ => "namespace_admission_failed",
        },
    };
    // The pidfd is opened before the first executable can run. Killing namespace
    // init also kills setsid/double-fork/nested-namespace descendants. A launcher
    // exit alone is never taken as proof that those processes have terminated.
    if stop_init(&init).is_err() {
        let _ = child.start_kill();
    }
    confirm_namespace_exit(&init).await;
    let status = reap(&mut child).await;
    let mut report = NamespaceReport {
        worker: Report {
            state,
            child_started: true,
            child_reaped: true,
            exit_code: status.code(),
            signal: status.signal(),
            finish_sent: false,
        },
        namespace_init: pid,
        namespace_exited: true,
    };
    report.worker.finish_sent = ticket.finish().await.is_ok();
    Ok(report)
}
