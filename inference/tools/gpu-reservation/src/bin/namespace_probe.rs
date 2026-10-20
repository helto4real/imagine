//! Harmless CPU-only lifecycle fixture. Never links an inference or GPU library.
use std::{
    io::{Read, Write},
    os::unix::net::UnixStream,
    path::Path,
};
fn connect(path: &Path, label: &str) -> UnixStream {
    let mut s = UnixStream::connect(path).unwrap();
    s.write_all(label.as_bytes()).unwrap();
    s.write_all(b"\n").unwrap();
    s
}
fn park(path: &Path, label: &str) -> ! {
    let _connection = connect(path, label);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let path = Path::new(&args[0]);
    for mode in 0..3 {
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            if mode > 0 {
                assert!(unsafe { libc::setsid() } >= 0);
            }
            if mode == 2 {
                let pid = unsafe { libc::fork() };
                assert!(pid >= 0);
                if pid > 0 {
                    unsafe { libc::_exit(0) }
                }
            }
            park(
                path,
                match mode {
                    0 => "child",
                    1 => "setsid",
                    _ => "double-fork",
                },
            );
        }
    }
    let mut connection = connect(path, "root");
    let mut start = [0u8];
    connection.read_exact(&mut start).unwrap();
    if args[1] == "exit" {
        return;
    }
    park(path, "root-after-ready");
}
