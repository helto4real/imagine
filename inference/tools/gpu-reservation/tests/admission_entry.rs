use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::UnixListener,
};
async fn check(reply: Option<u8>, expected: i32) {
    let own = tempfile::tempdir().unwrap();
    let socket = own.path().join("gate.sock");
    let marker = own.path().join("must-not-start");
    let listener = UnixListener::bind(&socket).unwrap();
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_namespace_entry"))
        .arg(&socket)
        .arg("own-token")
        .arg("/usr/bin/touch")
        .arg(&marker)
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let (mut peer, _) = listener.accept().await.unwrap();
    let mut token = [0u8; 9];
    peer.read_exact(&mut token).await.unwrap();
    assert_eq!(&token, b"own-token");
    if let Some(reply) = reply {
        peer.write_all(&[reply]).await.unwrap();
    }
    drop(peer);
    assert_eq!(child.wait().await.unwrap().code(), Some(expected));
    assert_eq!(marker.exists(), expected == 0);
}
#[tokio::test]
async fn eof_never_admits_a_worker() {
    check(None, 125).await;
}
#[tokio::test]
async fn wrong_byte_never_admits_a_worker() {
    check(Some(b'X'), 125).await;
}
#[tokio::test]
async fn explicit_admission_executes_the_reviewed_worker() {
    check(Some(b'G'), 0).await;
}
