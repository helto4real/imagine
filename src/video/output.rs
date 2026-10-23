//! Persistent image/video publication on an explicitly pinned volume. No fallback directory.
use crate::video_destination::{self as d, Config, Environment, Proof, Result, System};
use std::ffi::CString;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::MetadataExt,
    },
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
#[derive(Clone)]
pub struct Destination {
    config: Config,
    env: Arc<dyn Environment>,
}
impl Destination {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            env: Arc::new(System),
        }
    }
    pub fn with_environment(config: Config, env: Arc<dyn Environment>) -> Self {
        Self { config, env }
    }
    pub fn proof(&self) -> Result<Proof> {
        d::inspect(&self.config, self.env.as_ref()).map(|v| v.0)
    }
    pub fn preflight(&self, proof: &serde_json::Value, required: u64) -> Result<()> {
        let (actual, fd) = d::inspect(&self.config, self.env.as_ref())?;
        if serde_json::to_value(actual).map_err(|_| "video_destination_invalid")? != *proof {
            return Err("video_destination_mismatch");
        }
        d::writable(&fd, required)
    }
    pub fn reserve(&self, proof: &serde_json::Value, required: u64) -> Result<Pending> {
        self.reserve_image(proof, required, None)
    }
    pub fn reserve_image(
        &self,
        proof: &serde_json::Value,
        required: u64,
        source: Option<&Published>,
    ) -> Result<Pending> {
        self.preflight(proof, required)?;
        let (identity, root) = d::inspect(&self.config, self.env.as_ref())?;
        let (date, upscale_stem) = if let Some(source) = source {
            let relative = source
                .path
                .strip_prefix(&self.config.root)
                .map_err(|_| "video_destination_mismatch")?;
            let parts = relative.components().collect::<Vec<_>>();
            if parts.len() != 2 {
                return Err("video_destination_invalid");
            }
            let folder = parts[0]
                .as_os_str()
                .to_str()
                .ok_or("video_destination_invalid")?;
            if folder.len() != 10
                || !folder.bytes().enumerate().all(|(i, b)| {
                    if i == 4 || i == 7 {
                        b == b'-'
                    } else {
                        b.is_ascii_digit()
                    }
                })
            {
                return Err("video_destination_invalid");
            }
            let stem = source
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("video_destination_invalid")?;
            let stem = stem.split("_upscaled").next().unwrap();
            if !stem.starts_with("image_")
                || !stem.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            {
                return Err("video_destination_invalid");
            }
            (folder.to_owned(), Some(stem.to_owned()))
        } else {
            (date()?, None)
        };
        let name = CString::new(date.clone()).unwrap();
        if unsafe { libc::mkdirat(root.as_raw_fd(), name.as_ptr(), 0o700) } != 0
            && std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST)
        {
            return Err("video_destination_not_writable");
        }
        let fd = unsafe {
            libc::openat(
                root.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err("video_destination_invalid");
        }
        let folder = unsafe { File::from_raw_fd(fd) };
        if folder
            .metadata()
            .map_err(|_| "video_destination_unavailable")?
            .dev()
            != identity.device
            || d::fsid(&folder)? != self.config.fsid
        {
            return Err("video_volume_mismatch");
        }
        d::writable(&folder, required)?;
        let name =
            CString::new(format!(".media_{}.partial", uuid::Uuid::new_v4().simple())).unwrap();
        let fd = unsafe {
            libc::openat(
                folder.as_raw_fd(),
                name.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err(io_code());
        }
        Ok(Pending {
            destination: self.clone(),
            identity,
            folder: Arc::new(folder),
            file: Some(unsafe { File::from_raw_fd(fd) }),
            partial: name,
            date,
            final_name: None,
            upscale_stem,
            committed: false,
        })
    }
}
fn io_code() -> &'static str {
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ENOSPC | libc::EDQUOT) => "video_destination_full",
        Some(libc::EROFS) => "video_destination_read_only",
        _ => "video_destination_write_failed",
    }
}
fn date() -> Result<String> {
    let t = unsafe { libc::time(std::ptr::null_mut()) };
    let mut out = std::mem::MaybeUninit::<libc::tm>::uninit();
    if unsafe { libc::localtime_r(&t, out.as_mut_ptr()) }.is_null() {
        return Err("video_destination_invalid");
    }
    let t = unsafe { out.assume_init() };
    Ok(format!(
        "{:04}-{:02}-{:02}",
        t.tm_year + 1900,
        t.tm_mon + 1,
        t.tm_mday
    ))
}
struct DirectoryLock(Arc<File>);
impl DirectoryLock {
    fn acquire(folder: Arc<File>) -> Result<Self> {
        if unsafe { libc::flock(folder.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err("video_destination_write_failed");
        }
        Ok(Self(folder))
    }
}
impl Drop for DirectoryLock {
    fn drop(&mut self) {
        unsafe {
            libc::flock(self.0.as_raw_fd(), libc::LOCK_UN);
        }
    }
}
fn image_number_used(folder: &File, number: u64) -> Result<bool> {
    for role in ["", "_character_sheet", "_portrait", "_turnaround"] {
        for extension in ["png", "jpg", "webp"] {
            let name = CString::new(format!("image_{number:04}{role}.{extension}")).unwrap();
            let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
            if unsafe {
                libc::fstatat(
                    folder.as_raw_fd(),
                    name.as_ptr(),
                    stat.as_mut_ptr(),
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } == 0
            {
                return Ok(true);
            }
            if std::io::Error::last_os_error().raw_os_error() != Some(libc::ENOENT) {
                return Err("video_destination_unavailable");
            }
        }
    }
    Ok(false)
}
pub struct Pending {
    destination: Destination,
    identity: Proof,
    folder: Arc<File>,
    file: Option<File>,
    partial: CString,
    date: String,
    final_name: Option<CString>,
    committed: bool,
    upscale_stem: Option<String>,
}
impl Pending {
    pub fn publish(&mut self, bytes: &[u8]) -> Result<Published> {
        self.publish_named(bytes, "video", "", "mp4", None)
    }
    pub fn publish_image(
        &mut self,
        bytes: &[u8],
        role: &str,
        mime: &str,
        group: Option<&str>,
    ) -> Result<Published> {
        let extension = match mime {
            "image/png" => "png",
            "image/jpeg" => "jpg",
            "image/webp" => "webp",
            _ => return Err("invalid_image_output"),
        };
        let suffix = match role {
            "image" => "",
            "character_sheet" => "_character_sheet",
            "portrait" => "_portrait",
            "turnaround" => "_turnaround",
            _ => return Err("invalid_image_output"),
        };
        if group.is_some_and(|g| {
            !g.starts_with("image_") || !g.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
        }) {
            return Err("invalid_image_output");
        }
        self.publish_named(bytes, "image", suffix, extension, group)
    }
    fn publish_named(
        &mut self,
        bytes: &[u8],
        prefix: &str,
        suffix: &str,
        extension: &str,
        group: Option<&str>,
    ) -> Result<Published> {
        if self.destination.proof()? != self.identity {
            return Err("video_destination_mismatch");
        }
        d::writable(&self.folder, bytes.len() as u64)?;
        let file = self.file.as_mut().ok_or("video_destination_write_failed")?;
        write_bytes(file, bytes)?;
        file.sync_all()
            .map_err(|_| "video_destination_write_failed")?;
        self.file.take();
        // The first sheet role reserves the shared number; exact-role collisions still fail closed.
        let _lock = DirectoryLock::acquire(self.folder.clone())?;
        for number in 1..=1_000_000_u64 {
            if prefix == "image"
                && group.is_none()
                && self.upscale_stem.is_none()
                && image_number_used(&self.folder, number)?
            {
                continue;
            }
            let stem = if let Some(group) = group {
                group.to_owned()
            } else if let Some(base) = &self.upscale_stem {
                if number == 1 {
                    format!("{base}_upscaled")
                } else {
                    format!("{base}_upscaled_{number}")
                }
            } else {
                format!("{prefix}_{number:04}")
            };
            let name = CString::new(format!("{stem}{suffix}.{extension}")).unwrap();
            let status = unsafe {
                libc::renameat2(
                    self.folder.as_raw_fd(),
                    self.partial.as_ptr(),
                    self.folder.as_raw_fd(),
                    name.as_ptr(),
                    libc::RENAME_NOREPLACE,
                )
            };
            if status == 0 {
                self.final_name = Some(name.clone());
                self.folder
                    .sync_all()
                    .map_err(|_| "video_destination_write_failed")?;
                let fd = unsafe {
                    libc::openat(
                        self.folder.as_raw_fd(),
                        name.as_ptr(),
                        libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    )
                };
                if fd < 0 {
                    return Err("video_destination_write_failed");
                }
                return Ok(Published {
                    path: self
                        .destination
                        .config
                        .root
                        .join(&self.date)
                        .join(name.to_str().unwrap()),
                    file: Arc::new(unsafe { File::from_raw_fd(fd) }),
                });
            }
            if group.is_some()
                || std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST)
            {
                return Err(io_code());
            }
        }
        Err("video_destination_names_exhausted")
    }
    pub fn commit(mut self) {
        self.committed = true;
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        if !self.committed {
            let name = self.final_name.as_ref().unwrap_or(&self.partial);
            unsafe {
                libc::unlinkat(self.folder.as_raw_fd(), name.as_ptr(), 0);
            }
        }
    }
}
#[derive(Clone)]
pub struct Published {
    pub path: PathBuf,
    pub file: Arc<File>,
}
#[derive(Clone, Default)]
pub struct Outputs {
    pub destination: Option<Destination>,
    pub published: Arc<Mutex<HashMap<String, Published>>>,
}
impl Outputs {
    pub fn configure(path: &Path) -> Self {
        Self {
            destination: std::fs::read(path)
                .ok()
                .and_then(|b| serde_json::from_slice::<Config>(&b).ok())
                .map(Destination::new),
            ..Default::default()
        }
    }
    pub fn destination(&self) -> Result<&Destination> {
        self.destination
            .as_ref()
            .ok_or("video_destination_unconfigured")
    }
    pub fn open(&self, id: &str) -> Result<Option<File>> {
        self.published
            .lock()
            .map_err(|_| "video_destination_unavailable")?
            .get(id)
            .map(|p| {
                File::open(format!("/proc/self/fd/{}", p.file.as_raw_fd()))
                    .map_err(|_| "video_destination_unavailable")
            })
            .transpose()
    }
    pub fn read(&self, id: &str) -> Result<Option<Vec<u8>>> {
        let Some(mut f) = self.open(id)? else {
            return Ok(None);
        };
        use std::io::Seek;
        f.rewind().map_err(|_| "video_destination_unavailable")?;
        let mut b = Vec::new();
        f.read_to_end(&mut b)
            .map_err(|_| "video_destination_unavailable")?;
        Ok(Some(b))
    }
}

fn write_bytes(writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes).map_err(|e| match e.raw_os_error() {
        Some(libc::ENOSPC | libc::EDQUOT) => "video_destination_full",
        Some(libc::EACCES | libc::EPERM) => "video_destination_not_writable",
        Some(libc::EROFS) => "video_destination_read_only",
        _ => "video_destination_write_failed",
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    struct Failure(i32);
    impl Write for Failure {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::from_raw_os_error(self.0))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn simulated_disk_full_and_permission_errors_are_explicit() {
        for (code, label) in [
            (libc::ENOSPC, "video_destination_full"),
            (libc::EDQUOT, "video_destination_full"),
            (libc::EACCES, "video_destination_not_writable"),
            (libc::EROFS, "video_destination_read_only"),
        ] {
            assert_eq!(
                write_bytes(&mut Failure(code), b"own fixture").unwrap_err(),
                label
            );
        }
    }
}
