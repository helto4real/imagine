use gpu_reserved_job::{Report, Request};
use imagine_native_gpu_reservation as gpu_reserved_job;
use serde_json::{Value, json};
use std::{ffi::OsString, path::Path, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
    sync::oneshot,
};
#[test]
fn gpu_entry_requires_a_direct_native_worker_instead_of_a_script_or_launcher() {
    gpu_reserved_job::validate_gpu_worker(&std::env::current_exe().unwrap()).unwrap();
    assert!(gpu_reserved_job::validate_gpu_worker(Path::new("/usr/bin/sh")).is_err());
    let directory = tempfile::tempdir().unwrap();
    let script = directory.path().join("own.sh");
    std::fs::write(&script, b"#!/bin/sh\nexit 0\n").unwrap();
    assert!(gpu_reserved_job::validate_gpu_worker(&script).is_err());
}
fn request<'a>(socket: &'a Path, arguments: &'a [OsString]) -> Request<'a> {
    Request {
        socket,
        job: "own-synthetic-av-reservation",
        bytes: 17,
        executable: Path::new("/usr/bin/sh"),
        arguments,
        timeout: Duration::from_secs(10),
    }
}
fn child_arguments(marker: &Path, long: bool) -> Vec<OsString> {
    let script = if long {
        "printf '%s' \"$$\" > \"$1\"; exec sleep 60"
    } else {
        "printf '%s' \"$$\" > \"$1\""
    };
    [
        OsString::from("-c"),
        OsString::from(script),
        OsString::from("own-av-child"),
        marker.as_os_str().into(),
    ]
    .into()
}
fn assert_reaped(marker: &Path) {
    let pid = std::fs::read_to_string(marker)
        .unwrap()
        .parse::<i32>()
        .unwrap();
    // Signal zero checks only the known harmless child started by this test.
    assert_eq!(
        unsafe { libc::kill(pid, 0) },
        -1,
        "finished arrived while our child still existed"
    );
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}
async fn await_marker(marker: &Path) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while !marker.is_file() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
}
async fn line(reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>) -> Value {
    let mut text = String::new();
    assert_ne!(reader.read_line(&mut text).await.unwrap(), 0);
    serde_json::from_str(&text).unwrap()
}
fn assert_stopped(report: &Report, state: &str) {
    assert_eq!(report.state, state);
    assert!(report.child_started && report.child_reaped && report.finish_sent);
}
#[tokio::test]
async fn shared_ticket_grants_before_worker_start_and_finishes_after_real_exit() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("own.sock");
    let marker = directory.path().join("own.pid");
    let listener = UnixListener::bind(&socket).unwrap();
    let server_marker = marker.clone();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        assert_eq!(
            line(&mut reader).await,
            json!({"job":"own-synthetic-av-reservation","bytes":17})
        );
        assert!(!server_marker.exists());
        writer.write_all(b"{\"queued\":true}\n").await.unwrap();
        assert!(!server_marker.exists());
        writer.write_all(b"{\"granted\":true}\n").await.unwrap();
        assert_eq!(line(&mut reader).await, json!({"finished":true}));
        assert_reaped(&server_marker);
    });
    let arguments = child_arguments(&marker, false);
    let report = gpu_reserved_job::run(request(&socket, &arguments), std::future::pending())
        .await
        .unwrap();
    assert_stopped(&report, "completed");
    assert_eq!(report.exit_code, Some(0));
    server.await.unwrap();
}
async fn stop_active(loss: bool) -> Report {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("own.sock");
    let marker = directory.path().join("own.pid");
    let listener = UnixListener::bind(&socket).unwrap();
    let server_marker = marker.clone();
    let (cancel, receive) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        let _ = line(&mut reader).await;
        writer
            .write_all(b"{\"queued\":true}\n{\"granted\":true}\n")
            .await
            .unwrap();
        await_marker(&server_marker).await;
        if loss {
            writer.shutdown().await.unwrap();
            let _keep_sender_alive = cancel;
            assert_eq!(line(&mut reader).await, json!({"finished":true}));
            assert_reaped(&server_marker);
        } else {
            cancel.send(()).unwrap();
            assert_eq!(line(&mut reader).await, json!({"finished":true}));
            assert_reaped(&server_marker);
        }
    });
    let arguments = child_arguments(&marker, true);
    let report = gpu_reserved_job::run(request(&socket, &arguments), async {
        let _ = receive.await;
    })
    .await
    .unwrap();
    server.await.unwrap();
    report
}
#[tokio::test]
async fn explicit_cancellation_reaps_the_owned_child_before_finished() {
    let report = stop_active(false).await;
    assert_stopped(&report, "cancelled");
    assert!(report.signal.is_some());
}
#[tokio::test]
async fn coordinator_loss_reaps_the_owned_child_without_standalone_fallback() {
    let report = stop_active(true).await;
    assert_stopped(&report, "reservation_lost");
    assert!(report.signal.is_some());
}
#[tokio::test]
async fn cancellation_while_queued_never_starts_a_worker() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("own.sock");
    let marker = directory.path().join("own.pid");
    let listener = UnixListener::bind(&socket).unwrap();
    let (cancel, receive) = oneshot::channel();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let (reader, mut writer) = stream.into_split();
        let mut reader = BufReader::new(reader);
        let _ = line(&mut reader).await;
        writer.write_all(b"{\"queued\":true}\n").await.unwrap();
        cancel.send(()).unwrap();
        let mut text = String::new();
        match reader.read_line(&mut text).await {
            Ok(0) => {}
            Err(error) if error.kind() == std::io::ErrorKind::ConnectionReset => {}
            other => panic!("Expected cancelled queue connection to close, got {other:?}"),
        }
    });
    let arguments = child_arguments(&marker, true);
    let report = gpu_reserved_job::run(request(&socket, &arguments), async {
        let _ = receive.await;
    })
    .await
    .unwrap();
    assert_eq!(report.state, "cancelled_while_queued");
    assert!(!report.child_started && !report.child_reaped && !report.finish_sent);
    assert!(!marker.exists());
    server.await.unwrap();
}
#[tokio::test]
async fn rejection_or_missing_coordinator_never_starts_a_worker() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("own.sock");
    let marker = directory.path().join("own.pid");
    let arguments = child_arguments(&marker, true);
    assert!(
        gpu_reserved_job::run(request(&socket, &arguments), std::future::pending())
            .await
            .is_err()
    );
    assert!(!marker.exists());
    for reply in [
        b"{\"queued\":false}\n".as_slice(),
        b"{\"queued\":true}\n{\"granted\":false}\n".as_slice(),
    ] {
        let listener = UnixListener::bind(&socket).unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = stream.into_split();
            let mut reader = BufReader::new(reader);
            let _ = line(&mut reader).await;
            writer.write_all(reply).await.unwrap();
        });
        assert!(
            gpu_reserved_job::run(request(&socket, &arguments), std::future::pending())
                .await
                .is_err()
        );
        server.await.unwrap();
        std::fs::remove_file(&socket).unwrap();
        assert!(!marker.exists());
    }
}
