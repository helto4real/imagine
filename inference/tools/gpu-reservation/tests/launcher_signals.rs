//! Real stop signals, using only a private fake coordinator and a tiny C child.
use serde_json::{Value, json};
use std::{path::Path, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::UnixListener,
};
async fn check_signal(signal: i32) {
    let own = tempfile::tempdir().unwrap();
    let socket = own.path().join("own.sock");
    let marker = own.path().join("own-worker.pid");
    let report = own.path().join("own-report.json");
    let source = own.path().join("own-worker.c");
    let worker = own.path().join("own-worker");
    std::fs::write(&source,b"#include <stdio.h>\n#include <unistd.h>\nint main(int argc,char**argv){if(argc!=2)return 2;FILE*f=fopen(argv[1],\"w\");if(!f)return 3;fprintf(f,\"%d\",(int)getpid());fclose(f);sleep(60);return 0;}\n").unwrap();
    assert!(
        std::process::Command::new("/usr/bin/cc")
            .arg(&source)
            .arg("-o")
            .arg(&worker)
            .env("CCACHE_DIR", own.path().join("ccache"))
            .status()
            .unwrap()
            .success()
    );
    let executable = env!("CARGO_BIN_EXE_imagine-native-gpu-reservation");
    let listener = UnixListener::bind(&socket).unwrap();
    let mut launcher = tokio::process::Command::new(executable)
        .args(["--socket"])
        .arg(&socket)
        .args([
            "--job",
            "own-signal-fixture",
            "--bytes",
            "17",
            "--timeout",
            "15",
            "--manifest",
        ])
        .arg(&report)
        .arg("--")
        .arg(&worker)
        .arg(&marker)
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let pid = launcher.id().unwrap();
    let (stream, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
        .await
        .unwrap()
        .unwrap();
    let (reader, mut writer) = stream.into_split();
    let mut reader = BufReader::new(reader);
    let mut text = String::new();
    reader.read_line(&mut text).await.unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap(),
        json!({"job":"own-signal-fixture","bytes":17})
    );
    writer
        .write_all(b"{\"queued\":true}\n{\"granted\":true}\n")
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !marker.is_file() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    // Only signal our own controller. Its direct worker must be killed/reaped
    // by the production controller path before it releases the Ticket.
    assert_eq!(unsafe { libc::kill(pid as i32, signal) }, 0);
    text.clear();
    tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut text))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap(),
        json!({"finished":true})
    );
    let child = std::fs::read_to_string(&marker)
        .unwrap()
        .parse::<i32>()
        .unwrap();
    assert_eq!(unsafe { libc::kill(child, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
    assert!(
        !tokio::time::timeout(Duration::from_secs(5), launcher.wait())
            .await
            .unwrap()
            .unwrap()
            .success()
    );
    let record: Value = serde_json::from_slice(&std::fs::read(&report).unwrap()).unwrap();
    assert_eq!(record["job_id"], "own-signal-fixture");
    assert_eq!(record["coordinator_socket"], socket.to_str().unwrap());
    assert_eq!(record["report"]["state"], "cancelled");
    assert_eq!(record["report"]["child_reaped"], true);
    assert_eq!(record["report"]["finish_sent"], true);
    assert!(Path::new(record["direct_worker"].as_str().unwrap()).is_file());
}
#[tokio::test]
async fn launcher_sigterm_reaps_direct_native_worker_before_finished() {
    check_signal(libc::SIGTERM).await;
}
#[tokio::test]
async fn launcher_sigint_reaps_direct_native_worker_before_finished() {
    check_signal(libc::SIGINT).await;
}
