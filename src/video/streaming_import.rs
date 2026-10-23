//! Descriptor-only media ingress. Source size never determines a RAM allocation.
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    os::fd::AsRawFd,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

pub const CHUNK_BYTES: usize = 64 * 1024;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Cancelled,
    Idle,
    Read,
    Write,
    Space,
    Empty,
    NumericSize,
    Length,
    Target,
}
impl Error {
    pub fn code(self) -> &'static str {
        match self {
            Self::Cancelled => "media_import_cancelled",
            Self::Idle => "upload_timeout",
            Self::Read => "upload_interrupted",
            Self::Write => "media_import_write_failed",
            Self::Space => "media_import_storage_full",
            Self::Empty => "invalid_media_size",
            Self::NumericSize => "media_size_unrepresentable",
            Self::Length => "upload_length_mismatch",
            Self::Target => "invalid_import_target",
        }
    }
}
pub struct Receipt {
    pub file: File,
    pub bytes: u64,
    pub sha256: String,
}
pub fn checked_size(previous: u64, added: usize) -> Result<u64, Error> {
    previous
        .checked_add(added as u64)
        .filter(|v| *v <= super::contract::MAX_INTEGER)
        .ok_or(Error::NumericSize)
}
fn capacity(file: &impl AsRawFd) -> Result<u64, Error> {
    let mut value = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::fstatvfs(file.as_raw_fd(), value.as_mut_ptr()) } != 0 {
        return Err(Error::Target);
    }
    let value = unsafe { value.assume_init() };
    Ok(value.f_bavail.saturating_mul(value.f_frsize))
}
fn writable(error: std::io::Error) -> Error {
    if matches!(error.raw_os_error(), Some(libc::ENOSPC | libc::EDQUOT)) {
        Error::Space
    } else {
        Error::Write
    }
}
async fn cancelled(flag: &AtomicBool) {
    while !flag.load(Ordering::Acquire) {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
/// Caller owns a fresh target from the verified encrypted SessionStaging boundary.
/// No target or asset is published if receive, flush, cancellation or validation fails.
pub async fn receive<R: AsyncRead + Unpin>(
    reader: &mut R,
    target: File,
    expected: Option<u64>,
    flag: &AtomicBool,
    idle: Duration,
) -> Result<Receipt, Error> {
    if flag.load(Ordering::Acquire) {
        return Err(Error::Cancelled);
    }
    let metadata = target.metadata().map_err(|_| Error::Target)?;
    if !metadata.is_file() || metadata.len() != 0 {
        return Err(Error::Target);
    }
    if let Some(size) = expected {
        checked_size(size, 0)?;
        if size == 0 {
            return Err(Error::Empty);
        }
        if size > capacity(&target)? {
            return Err(Error::Space);
        }
    }
    let mut output = tokio::fs::File::from_std(target);
    let mut buffer = vec![0u8; CHUNK_BYTES];
    let mut bytes = 0u64;
    let mut hash = Sha256::new();
    loop {
        let read = tokio::select! {
            result=tokio::time::timeout(idle,reader.read(&mut buffer))=>result.map_err(|_|Error::Idle)?.map_err(|_|Error::Read)?,
            _=cancelled(flag)=>return Err(Error::Cancelled),
        };
        if read == 0 {
            break;
        }
        bytes = checked_size(bytes, read)?;
        if expected.is_some_and(|size| bytes > size) {
            return Err(Error::Length);
        }
        // Known-length transfers were checked against actual free space once above.
        // Writes remain authoritative if space changes; ENOSPC/EDQUOT roll back the
        // private descriptor. Avoid a volume-wide statvfs syscall per 64 KiB chunk.
        output.write_all(&buffer[..read]).await.map_err(writable)?;
        hash.update(&buffer[..read]);
    }
    if flag.load(Ordering::Acquire) {
        return Err(Error::Cancelled);
    }
    if bytes == 0 {
        return Err(Error::Empty);
    }
    if expected.is_some_and(|size| size != bytes) {
        return Err(Error::Length);
    }
    output.flush().await.map_err(writable)?;
    output.sync_all().await.map_err(writable)?;
    output.rewind().await.map_err(writable)?;
    if flag.load(Ordering::Acquire) {
        return Err(Error::Cancelled);
    }
    Ok(Receipt {
        file: output.into_std().await,
        bytes,
        sha256: format!("{:x}", hash.finalize()),
    })
}

#[cfg(test)]
use crate::task_fixtures;
#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        pin::Pin,
        task::{Context, Poll},
    };
    use tokio::io::ReadBuf;
    fn root() -> PathBuf {
        super::task_fixtures::env_root("IMAGINE_WORKER_A_FIXTURES")
    }
    fn target() -> File {
        tempfile::tempfile_in(root()).unwrap()
    }
    #[test]
    fn streaming_import_size_is_u64_and_not_a_two_gib_gate() {
        assert_eq!(
            checked_size(2 * 1024 * 1024 * 1024, 65536),
            Ok(2 * 1024 * 1024 * 1024 + 65536)
        );
        assert_eq!(
            checked_size(super::super::contract::MAX_INTEGER, 1),
            Err(Error::NumericSize)
        );
        assert_eq!(checked_size(u64::MAX, 1), Err(Error::NumericSize));
    }
    #[tokio::test]
    async fn streaming_import_unknown_length_hashes_and_preserves_bytes() {
        let data = b"OWN SYNTHETIC SOURCE".repeat(CHUNK_BYTES / 8);
        let mut reader = data.as_slice();
        let receipt = receive(
            &mut reader,
            target(),
            None,
            &AtomicBool::new(false),
            Duration::from_secs(1),
        )
        .await
        .unwrap();
        assert_eq!(receipt.bytes, data.len() as u64);
        assert_eq!(receipt.sha256, format!("{:x}", Sha256::digest(&data)));
        let mut actual = vec![];
        std::io::Read::read_to_end(&mut receipt.file.try_clone().unwrap(), &mut actual).unwrap();
        assert_eq!(actual, data);
    }
    #[tokio::test]
    async fn streaming_import_length_empty_space_and_write_errors() {
        for expected in [Some(2), Some(6)] {
            let mut input = b"four".as_slice();
            assert_eq!(
                receive(
                    &mut input,
                    target(),
                    expected,
                    &AtomicBool::new(false),
                    Duration::from_secs(1)
                )
                .await
                .err(),
                Some(Error::Length)
            );
        }
        let mut input = b"".as_slice();
        assert_eq!(
            receive(
                &mut input,
                target(),
                None,
                &AtomicBool::new(false),
                Duration::from_secs(1)
            )
            .await
            .err(),
            Some(Error::Empty)
        );
        let file = target();
        let space = capacity(&file).unwrap();
        let mut input = b"x".as_slice();
        assert_eq!(
            receive(
                &mut input,
                file,
                Some(space + 1),
                &AtomicBool::new(false),
                Duration::from_secs(1)
            )
            .await
            .err(),
            Some(Error::Space)
        );
        let path = root().join(format!("readonly-{}", uuid::Uuid::new_v4()));
        drop(File::create(&path).unwrap());
        let readonly = File::open(path).unwrap();
        let mut input = b"x".as_slice();
        assert_eq!(
            receive(
                &mut input,
                readonly,
                None,
                &AtomicBool::new(false),
                Duration::from_secs(1)
            )
            .await
            .err(),
            Some(Error::Write)
        );
    }
    struct MustNotRead;
    impl AsyncRead for MustNotRead {
        fn poll_read(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            _: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            panic!("cancelled source must not be read")
        }
    }
    struct FailingRead;
    impl AsyncRead for FailingRead {
        fn poll_read(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            _: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Err(std::io::Error::other("own synthetic read failure")))
        }
    }
    #[tokio::test]
    async fn streaming_import_cancel_read_failure_and_idle_are_terminal() {
        assert_eq!(
            receive(
                &mut MustNotRead,
                target(),
                None,
                &AtomicBool::new(true),
                Duration::from_secs(1)
            )
            .await
            .err(),
            Some(Error::Cancelled)
        );
        assert_eq!(
            receive(
                &mut FailingRead,
                target(),
                None,
                &AtomicBool::new(false),
                Duration::from_secs(1)
            )
            .await
            .err(),
            Some(Error::Read)
        );
        let (mut reader, _writer) = tokio::io::duplex(16);
        assert_eq!(
            receive(
                &mut reader,
                target(),
                None,
                &AtomicBool::new(false),
                Duration::from_millis(5)
            )
            .await
            .err(),
            Some(Error::Idle)
        );
        let (mut reader, writer) = tokio::io::duplex(16);
        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let copy = flag.clone();
        let task = tokio::spawn(async move {
            let _writer = writer;
            tokio::time::sleep(Duration::from_millis(5)).await;
            copy.store(true, Ordering::Release);
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        assert_eq!(
            receive(&mut reader, target(), None, &flag, Duration::from_secs(1))
                .await
                .err(),
            Some(Error::Cancelled)
        );
        task.await.unwrap();
    }
}
