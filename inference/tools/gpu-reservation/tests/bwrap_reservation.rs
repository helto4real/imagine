use imagine_native_gpu_reservation::{Admission, Request, run_bwrap};
use serde_json::{Value, json};
use std::{ffi::OsString, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::oneshot,
};
async fn line(reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>) -> Value {
    let mut s = String::new();
    assert_ne!(reader.read_line(&mut s).await.unwrap(), 0);
    serde_json::from_str(&s).unwrap()
}
async fn example(mode: &str) {
    let dir = tempfile::tempdir().unwrap();
    let socket = dir.path().join("mock.sock");
    let observed = dir.path().join("children.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let children = UnixListener::bind(&observed).unwrap();
    let (cancel, receive) = oneshot::channel();
    let (controller_send, controller_receive) = oneshot::channel::<u32>();
    let mode = mode.to_owned();
    let server_mode = mode.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (r, mut w) = stream.into_split();
        let mut r = BufReader::new(r);
        assert_eq!(
            line(&mut r).await,
            json!({"job":"own-h3-namespace","bytes":17})
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(30), children.accept())
                .await
                .is_err(),
            "a worker started before grant"
        );
        w.write_all(b"{\"queued\":true}\n{\"granted\":true}\n")
            .await
            .unwrap();
        let mut peers: Vec<(i32, UnixStream)> = vec![];
        let mut root = None;
        for _ in 0..4 {
            let (mut child, _) = tokio::time::timeout(Duration::from_secs(5), children.accept())
                .await
                .unwrap()
                .unwrap();
            let pid = child.peer_cred().unwrap().pid().unwrap();
            let mut label = vec![];
            loop {
                let b = child.read_u8().await.unwrap();
                if b == b'\n' {
                    break;
                }
                label.push(b);
            }
            if label == b"root" {
                root = Some(peers.len());
            }
            peers.push((pid, child));
        }
        peers[root.unwrap()].1.write_all(&[1]).await.unwrap();
        if server_mode.starts_with("signal-") {
            let pid = controller_receive.await.unwrap();
            let signal = if server_mode == "signal-int" {
                libc::SIGINT
            } else {
                libc::SIGTERM
            };
            assert_eq!(unsafe { libc::kill(pid as i32, signal) }, 0);
        } else if server_mode == "cancel" {
            cancel.send(()).unwrap();
        } else if server_mode == "loss" {
            w.shutdown().await.unwrap();
        } else {
            drop(cancel);
        }
        // Closing the sender is not cancellation; the receiver future in run
        // remains pending when the test requests normal completion/timeout.
        assert_eq!(line(&mut r).await, json!({"finished":true}));
        for (pid, mut peer) in peers {
            assert_eq!(
                tokio::time::timeout(Duration::from_secs(2), peer.read_u8())
                    .await
                    .unwrap()
                    .unwrap_err()
                    .kind(),
                std::io::ErrorKind::UnexpectedEof
            );
            assert_eq!(
                unsafe { libc::kill(pid, 0) },
                -1,
                "finished before owned process {pid} was reaped"
            );
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
        }
    });
    let probe = std::path::Path::new(env!("CARGO_BIN_EXE_namespace_probe"));
    let mut args: Vec<OsString> = [
        "--unshare-all",
        "--clearenv",
        "--ro-bind",
        "/usr/lib",
        "/usr/lib",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--ro-bind",
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    args.extend([
        probe.as_os_str().to_owned(),
        "/probe".into(),
        "--bind".into(),
        dir.path().as_os_str().to_owned(),
        "/work".into(),
        "--".into(),
        "/probe".into(),
        "/work/children.sock".into(),
        if mode == "exit" {
            "exit".into()
        } else {
            "wait".into()
        },
    ]);
    let report: Value = if mode.starts_with("signal-") {
        let manifest = dir.path().join("own-manifest.json");
        let mut controller =
            tokio::process::Command::new(env!("CARGO_BIN_EXE_imagine-native-gpu-reservation"))
                .args(["--socket"])
                .arg(&socket)
                .args([
                    "--job",
                    "own-h3-namespace",
                    "--bytes",
                    "17",
                    "--timeout",
                    "10",
                    "--process-form",
                    "bwrap-pid-namespace",
                    "--namespace-entry",
                    env!("CARGO_BIN_EXE_namespace_entry"),
                    "--manifest",
                ])
                .arg(&manifest)
                .args(["--", "/usr/bin/bwrap"])
                .args(&args)
                .stderr(std::process::Stdio::null())
                .kill_on_drop(true)
                .spawn()
                .unwrap();
        controller_send.send(controller.id().unwrap()).unwrap();
        let status = tokio::time::timeout(Duration::from_secs(15), controller.wait())
            .await
            .unwrap()
            .unwrap();
        assert!(!status.success());
        let record: Value = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
        assert_eq!(record["process_form"], "bwrap-pid-namespace");
        record["report"].clone()
    } else {
        let report = tokio::time::timeout(
            Duration::from_secs(15),
            run_bwrap(
                Request {
                    socket: &socket,
                    job: "own-h3-namespace",
                    bytes: 17,
                    executable: std::path::Path::new("/usr/bin/bwrap"),
                    arguments: &args,
                    timeout: Duration::from_secs(if mode == "timeout" { 2 } else { 10 }),
                },
                Admission {
                    entry: std::path::Path::new(env!("CARGO_BIN_EXE_namespace_entry")),
                    own_root: dir.path(),
                },
                async {
                    if receive.await.is_ok() {
                        return;
                    }
                    std::future::pending::<()>().await
                },
            ),
        )
        .await
        .unwrap()
        .unwrap();
        serde_json::to_value(report).unwrap()
    };
    assert_eq!(report["namespace_exited"], true);
    assert_eq!(report["worker"]["child_reaped"], true);
    assert_eq!(report["worker"]["finish_sent"], true);
    assert_eq!(
        report["worker"]["state"],
        match mode.as_str() {
            "exit" => "completed",
            "cancel" | "signal-int" | "signal-term" => "cancelled",
            "loss" => "reservation_lost",
            _ => "timed_out",
        }
    );
    server.await.unwrap();
}
#[tokio::test]
async fn pid_namespace_completion_reaps_detached_descendants_before_finished() {
    example("exit").await;
}
#[tokio::test]
async fn pid_namespace_cancel_reaps_detached_descendants_before_finished() {
    example("cancel").await;
}
#[tokio::test]
async fn pid_namespace_timeout_reaps_detached_descendants_before_finished() {
    example("timeout").await;
}
#[tokio::test]
async fn pid_namespace_lease_loss_reaps_detached_descendants_before_finished() {
    example("loss").await;
}

#[tokio::test]
async fn pid_namespace_sigint_reaps_detached_descendants_before_finished() {
    example("signal-int").await;
}
#[tokio::test]
async fn pid_namespace_sigterm_reaps_detached_descendants_before_finished() {
    example("signal-term").await;
}
