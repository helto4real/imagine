//! Hand off Wayland text selections to an owner that outlives the UI process.
use eframe::egui;
use std::{
    fs::File,
    io::{self, Seek, Write},
    os::{fd::FromRawFd, unix::fs::PermissionsExt},
    process::{Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

const COPY_TIMEOUT: Duration = Duration::from_secs(1);

enum Work {
    Copy(u64, String),
    Finish,
}

pub(crate) struct Clipboard {
    tx: mpsc::Sender<Work>,
    finished: mpsc::Receiver<()>,
}

impl Clipboard {
    pub fn install(
        ctx: &egui::Context,
        report_error: impl Fn() + Send + Sync + 'static,
    ) -> Option<Self> {
        if std::env::var_os("WAYLAND_DISPLAY").is_none_or(|value| value.is_empty()) {
            return None;
        }
        let program = std::env::split_paths(&std::env::var_os("PATH")?)
            .map(|dir| dir.join("wl-copy"))
            .find(|path| {
                path.metadata()
                    .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
            })?;
        Some(Self::with_command(
            ctx,
            Command::new(program),
            COPY_TIMEOUT,
            report_error,
        ))
    }

    fn with_command(
        ctx: &egui::Context,
        program: Command,
        timeout: Duration,
        report_error: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        let (failed_tx, failed_rx) = mpsc::channel();
        let (finished_tx, finished) = mpsc::channel();
        let worker_ctx = ctx.clone();
        std::thread::spawn(move || {
            while let Ok(work) = rx.recv() {
                let Work::Copy(mut sequence, mut text) = work else {
                    break;
                };
                let mut finish = false;
                // Only the newest queued copy matters. Serial delivery prevents an
                // older helper from taking ownership after a newer copy.
                while let Ok(work) = rx.try_recv() {
                    match work {
                        Work::Copy(next, value) => {
                            sequence = next;
                            text = value;
                        }
                        Work::Finish => {
                            finish = true;
                            break;
                        }
                    }
                }
                if hand_off(&program, &text, timeout).is_err() {
                    let _ = failed_tx.send((sequence, text));
                    worker_ctx.request_repaint();
                }
                if finish {
                    break;
                }
            }
            let _ = finished_tx.send(());
        });
        let hook_tx = tx.clone();
        let state = Mutex::new((0_u64, failed_rx));
        ctx.on_end_pass(
            "persistent clipboard",
            Arc::new(move |ui| {
                let ctx = ui.ctx();
                if ctx.will_discard() {
                    return;
                }
                let mut state = state.lock().unwrap();
                let mut failed = false;
                ctx.output_mut(|output| {
                    output.commands.retain(|command| {
                        let egui::OutputCommand::CopyText(text) = command else {
                            return true;
                        };
                        state.0 += 1;
                        // If the worker is unavailable, leave native clipboard handling intact.
                        hook_tx.send(Work::Copy(state.0, text.clone())).is_err()
                    });
                    while let Ok((sequence, text)) = state.1.try_recv() {
                        if sequence == state.0 {
                            // This is appended after interception, so it reaches the native
                            // backend once instead of being sent back to the failed helper.
                            output.commands.push(egui::OutputCommand::CopyText(text));
                            failed = true;
                        }
                    }
                });
                if failed {
                    report_error();
                    ctx.request_repaint();
                }
            }),
        );
        Self { tx, finished }
    }

    pub fn finish(&self) {
        let _ = self.tx.send(Work::Finish);
        // Only shutdown waits. At most one active and one coalesced copy remain.
        let _ = self
            .finished
            .recv_timeout(COPY_TIMEOUT * 2 + Duration::from_millis(500));
    }
}

fn hand_off(program: &Command, text: &str, timeout: Duration) -> io::Result<()> {
    // Anonymous RAM-backed input avoids blocking pipe writes and plaintext temp files.
    // The child owns the descriptor even if Imagine exits immediately afterwards.
    let fd = unsafe { libc::memfd_create(c"imagine-clipboard".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: memfd_create returned a new, valid descriptor owned exclusively here.
    let mut input = unsafe { File::from_raw_fd(fd) };
    input.write_all(text.as_bytes())?;
    input.rewind()?;
    let mut child = Command::new(program.get_program())
        .args(program.get_args())
        .args(["--type", "text/plain;charset=utf-8"])
        .env_remove("WAYLAND_DEBUG")
        .stdin(Stdio::from(input))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return Ok(()),
            Ok(Some(_)) => return Err(io::Error::other("Clipboard helper failed")),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            result => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(result.err().unwrap_or_else(|| {
                    io::Error::new(io::ErrorKind::TimedOut, "Clipboard helper timed out")
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn helper(root: &Path, body: &str) -> PathBuf {
        let path = root.join("clipboard-helper");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    fn scripted(path: &Path) -> Command {
        // Interpret the fixture instead of exec'ing a just-written file: parallel
        // test process creation can briefly inherit a writable fd (ETXTBSY).
        let mut command = Command::new("/bin/sh");
        command.arg(path);
        command
    }

    fn frame(ctx: &egui::Context, copy: Option<&str>) -> Vec<egui::OutputCommand> {
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            if let Some(text) = copy {
                ui.ctx().copy_text(text.to_owned());
            }
        });
        output.textures_delta.clear();
        output.platform_output.commands
    }

    #[test]
    fn helper_receives_exact_text_through_stdin_not_arguments() {
        let root = tempfile::tempdir().unwrap();
        let program = helper(
            root.path(),
            r#"
[ "$#" -eq 2 ] && [ "$1" = '--type' ] && [ "$2" = 'text/plain;charset=utf-8' ] || exit 9
cat > "$0.input"
"#,
        );
        let text = "Synthetic åäö\n'\" $(never-run) `never-run` --option\n".repeat(4096);
        hand_off(&scripted(&program), &text, Duration::from_secs(1)).unwrap();
        assert_eq!(
            std::fs::read_to_string(program.with_extension("input")).unwrap(),
            text
        );
    }

    #[test]
    fn latest_copy_is_delivered_before_shutdown_and_helper_can_outlive_ui() {
        let root = tempfile::tempdir().unwrap();
        let program = helper(
            root.path(),
            r#"
cat > "$0.input"
(sleep 0.05; cp "$0.input" "$0.served") >/dev/null 2>&1 &
"#,
        );
        let ctx = egui::Context::default();
        let clipboard = Clipboard::with_command(&ctx, scripted(&program), COPY_TIMEOUT, || {
            panic!("unexpected failure")
        });
        for text in ["first", "second", "last: åäö\n"] {
            assert!(frame(&ctx, Some(text)).is_empty());
        }
        clipboard.finish();
        drop(clipboard);
        drop(ctx);
        let served = program.with_extension("served");
        let deadline = Instant::now() + Duration::from_secs(2);
        // A file can exist while the synthetic helper is still copying into it.
        while std::fs::read_to_string(&served).ok().as_deref() != Some("last: åäö\n")
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(std::fs::read_to_string(served).unwrap(), "last: åäö\n");
    }

    #[test]
    fn failures_and_timeouts_restore_native_copy_once() {
        for body in ["exit 1", "exec sleep 5"] {
            let root = tempfile::tempdir().unwrap();
            let program = helper(root.path(), body);
            let ctx = egui::Context::default();
            let errors = Arc::new(AtomicUsize::new(0));
            let report = errors.clone();
            let clipboard = Clipboard::with_command(
                &ctx,
                scripted(&program),
                Duration::from_millis(30),
                move || {
                    report.fetch_add(1, Ordering::SeqCst);
                },
            );
            assert!(frame(&ctx, Some("synthetic fallback")).is_empty());
            clipboard.finish();
            let output = frame(&ctx, None);
            assert!(
                matches!(output.as_slice(), [egui::OutputCommand::CopyText(text)] if text == "synthetic fallback")
            );
            assert_eq!(errors.load(Ordering::SeqCst), 1);
            assert!(frame(&ctx, None).is_empty());
        }
    }

    #[test]
    fn stale_failure_does_not_replace_a_newer_copy() {
        let root = tempfile::tempdir().unwrap();
        let program = helper(root.path(), "exit 1");
        let ctx = egui::Context::default();
        let clipboard = Clipboard::with_command(&ctx, scripted(&program), COPY_TIMEOUT, || {});
        frame(&ctx, Some("old"));
        clipboard.finish();
        // A dead worker leaves this newer command with the native backend. The old
        // failure must not append its stale text after the newer command.
        let output = frame(&ctx, Some("new"));
        assert!(
            matches!(output.as_slice(), [egui::OutputCommand::CopyText(text)] if text == "new")
        );
    }
}
