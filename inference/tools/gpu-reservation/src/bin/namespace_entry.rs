//! GPU-free admission executable. EOF is rejection, never permission to exec.
use std::{
    io::{Read, Write},
    os::unix::{net::UnixStream, process::CommandExt},
    process::Command,
};
fn main() {
    let mut args = std::env::args_os().skip(1);
    let gate = args.next().expect("Missing owned admission socket");
    let token = args.next().expect("Missing one-use admission token");
    let program = args.next().expect("Missing explicit worker");
    let mut connection = UnixStream::connect(gate).expect("Admission socket unavailable");
    connection
        .write_all(token.as_encoded_bytes())
        .expect("Admission peer unavailable");
    let mut ready = [0u8];
    if connection.read_exact(&mut ready).is_err() || ready != *b"G" {
        std::process::exit(125);
    }
    drop(connection);
    let error = Command::new(program).args(args).exec();
    eprintln!("Reserved namespace worker could not exec: {error}");
    std::process::exit(126);
}
