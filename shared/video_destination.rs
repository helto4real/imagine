//! Trusted local destination identity; no media reads, writes or client-selected paths.
use serde::{Deserialize, Serialize};
use std::{
    ffi::CString,
    fs::{File, OpenOptions},
    os::{
        fd::AsRawFd,
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt},
        },
    },
    path::{Path, PathBuf},
};
pub type Result<T> = std::result::Result<T, &'static str>;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub root: PathBuf,
    pub mount: PathBuf,
    pub source: String,
    pub filesystem: String,
    pub fsid: [i32; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Proof {
    pub config: Config,
    pub host: String,
    pub device: u64,
    pub inode: u64,
}
pub trait Environment: Send + Sync {
    fn mounts(&self) -> Result<String>;
    fn host(&self) -> Result<String>;
}
pub struct System;
impl Environment for System {
    fn mounts(&self) -> Result<String> {
        std::fs::read_to_string("/proc/self/mountinfo").map_err(|_| "video_mount_unavailable")
    }
    fn host(&self) -> Result<String> {
        std::fs::read_to_string("/etc/machine-id")
            .map(|v| v.trim().to_owned())
            .map_err(|_| "video_host_unavailable")
    }
}
fn unescape(v: &str) -> String {
    v.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}
pub fn mounted(config: &Config, text: &str) -> Result<()> {
    let target = config.mount.to_str().ok_or("video_destination_invalid")?;
    let root = config.root.to_str().ok_or("video_destination_invalid")?;
    let mut matched = false;
    for line in text.lines() {
        let Some((a, b)) = line.split_once(" - ") else {
            continue;
        };
        let a: Vec<_> = a.split_whitespace().collect();
        let b: Vec<_> = b.split_whitespace().collect();
        if a.len() < 6 || b.len() < 3 {
            continue;
        }
        let p = unescape(a[4]);
        if p == target {
            if unescape(b[1]) != config.source || b[0] != config.filesystem {
                return Err("video_volume_mismatch");
            }
            if !a[5].split(',').any(|s| s == "rw") || b[2].split(',').any(|s| s == "ro") {
                return Err("video_destination_read_only");
            }
            matched = true;
        } else if p != target
            && Path::new(&p).starts_with(&config.mount)
            && Path::new(root).starts_with(&p)
        {
            return Err("video_volume_mismatch");
        }
    }
    if !matched {
        return Err("video_volume_not_mounted");
    }
    Ok(())
}
pub fn fsid(file: &File) -> Result<[i32; 2]> {
    let mut s = std::mem::MaybeUninit::<libc::statfs>::uninit();
    if unsafe { libc::fstatfs(file.as_raw_fd(), s.as_mut_ptr()) } != 0 {
        return Err("video_destination_unavailable");
    }
    let s = unsafe { s.assume_init() };
    Ok(unsafe { std::mem::transmute_copy(&s.f_fsid) })
}
pub fn directory(path: &Path) -> Result<File> {
    if !path.is_absolute()
        || path.components().any(|c| {
            !matches!(
                c,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )
        })
    {
        return Err("video_destination_invalid");
    }
    let mut fd = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open("/")
        .map_err(|_| "video_destination_unavailable")?;
    use std::os::fd::FromRawFd;
    for part in path.components() {
        if let std::path::Component::Normal(name) = part {
            let name = CString::new(name.as_bytes()).map_err(|_| "video_destination_invalid")?;
            let next = unsafe {
                libc::openat(
                    fd.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if next < 0 {
                return Err("video_destination_unavailable");
            }
            fd = unsafe { File::from_raw_fd(next) };
        }
    }
    Ok(fd)
}
pub fn inspect(config: &Config, env: &dyn Environment) -> Result<(Proof, File)> {
    if config.root == config.mount || !config.root.starts_with(&config.mount) {
        return Err("video_destination_invalid");
    }
    mounted(config, &env.mounts()?)?;
    let mount = directory(&config.mount)?;
    let root = directory(&config.root)?;
    let m = mount
        .metadata()
        .map_err(|_| "video_destination_unavailable")?;
    let r = root
        .metadata()
        .map_err(|_| "video_destination_unavailable")?;
    if m.dev() != r.dev() || fsid(&root)? != config.fsid || fsid(&mount)? != config.fsid {
        return Err("video_volume_mismatch");
    }
    let host = env.host()?;
    if host.is_empty() {
        return Err("video_host_unavailable");
    }
    Ok((
        Proof {
            config: config.clone(),
            host,
            device: r.dev(),
            inode: r.ino(),
        },
        root,
    ))
}
pub fn writable(fd: &File, required: u64) -> Result<()> {
    let dot = c".";
    if unsafe { libc::faccessat(fd.as_raw_fd(), dot.as_ptr(), libc::W_OK, libc::AT_EACCESS) } != 0 {
        return Err("video_destination_not_writable");
    }
    let mut v = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::fstatvfs(fd.as_raw_fd(), v.as_mut_ptr()) } != 0 {
        return Err("video_destination_unavailable");
    }
    let v = unsafe { v.assume_init() };
    if v.f_flag & libc::ST_RDONLY != 0 {
        return Err("video_destination_read_only");
    }
    if v.f_bavail.saturating_mul(v.f_frsize) < required {
        return Err("video_destination_full");
    }
    Ok(())
}
