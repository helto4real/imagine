//! The stack's proxy port release stops only an st-proxy of this user; any other
//! listener and its process group keep running and startup fails.
use std::{
    io::{BufRead, BufReader},
    os::unix::process::{CommandExt, ExitStatusExt},
    path::Path,
    process::{Child, Command, Stdio},
};

/// A listener on a free loopback port, run as `program` in its own process group.
fn listener(program: &Path) -> (Child, u16) {
    let mut child = Command::new(program)
        .args([
            "-c",
            "import socket,sys,time\ns=socket.socket();s.bind(('127.0.0.1',0));s.listen()\nprint(s.getsockname()[1],flush=True)\ntime.sleep(60)",
        ])
        .stdout(Stdio::piped())
        .process_group(0)
        .spawn()
        .unwrap();
    let mut line = String::new();
    BufReader::new(child.stdout.as_mut().unwrap())
        .read_line(&mut line)
        .unwrap();
    (child, line.trim().parse().unwrap())
}
fn release(port: u16) -> (bool, String) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/stack_ports.zsh");
    let out = Command::new("zsh")
        .arg(script)
        .arg(port.to_string())
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into(),
    )
}

#[test]
fn foreign_port_owners_survive_and_stack_proxies_are_stopped() {
    let python = Path::new("/usr/bin/python3");
    let (mut foreign, port) = listener(python);
    let (released, log) = release(port);
    assert!(!released, "{log}");
    assert!(log.contains("not an st-proxy of this user"), "{log}");
    assert!(foreign.try_wait().unwrap().is_none());
    assert!(std::net::TcpStream::connect(("127.0.0.1", port)).is_ok());
    foreign.kill().unwrap();
    foreign.wait().unwrap();

    // An st-proxy left from another checkout is the stack's own and is stopped.
    let dir = tempfile::tempdir().unwrap();
    let proxy = dir.path().join("st-proxy");
    std::fs::copy(std::fs::canonicalize(python).unwrap(), &proxy).unwrap();
    let (mut stale, port) = listener(&proxy);
    let (released, log) = release(port);
    assert!(released, "{log}");
    assert!(stale.wait().unwrap().signal().is_some());
}
