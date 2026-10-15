use anyhow::{Result, anyhow, bail};
use image::{ImageDecoder, ImageFormat, ImageReader};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{Cursor, Read, Write},
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
};
#[derive(Clone)]
pub struct Reference {
    pub workflow: String,
    pub path: PathBuf,
    pub asset: String,
    pub width: u32,
    pub height: u32,
    pub digest: String,
}
pub const LIMIT: usize = 10 * 1024 * 1024;
pub fn read_image(path: &Path, roots: &[PathBuf]) -> Result<Vec<u8>> {
    let path = path.canonicalize()?;
    let root = roots
        .iter()
        .find(|root| path.starts_with(root))
        .ok_or_else(|| anyhow!("image path is outside the configured input/output directories"))?;
    let components: Vec<_> = path.strip_prefix(root)?.components().collect();
    if components.is_empty() {
        bail!("image path must name a regular file")
    }
    let c = std::ffi::CString::new(root.as_os_str().as_encoded_bytes())?;
    let fd = unsafe {
        libc::open(
            c.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    };
    let mut dir = unsafe { OwnedFd::from_raw_fd(fd) };
    for (part_index, part) in components.iter().enumerate() {
        let name = std::ffi::CString::new(part.as_os_str().as_encoded_bytes())?;
        let flags = libc::O_RDONLY
            | libc::O_NOFOLLOW
            | libc::O_CLOEXEC
            | if part_index + 1 == components.len() {
                libc::O_NONBLOCK
            } else {
                libc::O_DIRECTORY
            };
        let fd = unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags) };
        if fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        };
        dir = unsafe { OwnedFd::from_raw_fd(fd) };
    }
    let file = std::fs::File::from(dir);
    if !file.metadata()?.is_file() {
        bail!("image path must name a regular file")
    };
    let mut data = Vec::new();
    file.take((LIMIT + 1) as u64).read_to_end(&mut data)?;
    if data.is_empty() || data.len() > LIMIT {
        bail!("reference image must be non-empty and at most 10 MiB")
    };
    Ok(data)
}
pub fn normalize(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let mut reader = ImageReader::new(Cursor::new(data)).with_guessed_format()?;
    if reader.format().is_none() {
        bail!("reference could not be read, decoded or stored");
    }
    if !matches!(
        reader.format(),
        Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP)
    ) {
        bail!("reference image must be PNG, JPEG or WebP")
    };
    reader.no_limits();
    let mut decoder = reader.into_decoder()?;
    let (w, h) = decoder.dimensions();
    if u64::from(w) * u64::from(h) > 89_478_485 {
        bail!("reference image dimensions exceed the decode limit")
    };
    let orientation = decoder.orientation()?;
    let mut image = image::DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let image = if image.color().has_alpha() {
        image::DynamicImage::ImageRgba8(image.to_rgba8())
    } else {
        image::DynamicImage::ImageRgb8(image.to_rgb8())
    };
    let (w, h) = (image.width(), image.height());
    let mut output = Cursor::new(Vec::new());
    image.write_to(&mut output, ImageFormat::Png)?;
    let output = output.into_inner();
    if output.len() > LIMIT {
        bail!("normalized reference exceeds 10 MiB")
    };
    Ok((output, w, h))
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}
pub fn store_preview(root: &Path, data: &[u8]) -> Result<(PathBuf, String)> {
    let digest = hash(data);
    let dir = root.join("references");
    private_dir(&dir)?;
    let path = dir.join(format!("{digest}.png"));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&path)
    {
        Ok(mut f) => {
            f.write_all(data)?;
            f.sync_all()?
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if hash(&read_image(&path, &[root.to_path_buf()])?) != digest {
                bail!("registered reference has changed; register the image again")
            }
        }
        Err(e) => return Err(e.into()),
    }
    Ok((path, digest))
}
impl Reference {
    pub fn verify(&self, root: &Path) -> Result<()> {
        if hash(&read_image(&self.path, &[root.into()])?) != self.digest {
            bail!("registered reference has changed; register the image again")
        };
        Ok(())
    }
    pub fn result(&self, id: &str) -> Value {
        json!({"image_id":id,"width":self.width,"height":self.height,"mime_type":"image/png","preview_path":self.path})
    }
}
/// Save one item inside the broker-owned directory for the whole MCP job.
pub fn save_result(directory: &Path, index: usize, bytes: &[u8]) -> Result<Vec<Value>> {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let stage = directory.join(format!(".{id}.part"));
    let path = directory.join(format!("{index}.png"));
    let result = (|| {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&stage)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&stage, &path)?;
        Ok(vec![
            json!({"image_id":id,"path":path,"mime_type":"image/png",
            "bytes":bytes.len(),"sha256":hash(bytes),"storage":"temporary"}),
        ])
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(stage);
    }
    result
}

/// Delete only an owned result filename, never traverse client paths or symlinks.
/// Missing files are safe on retries; other errors leave confirmation pending.
pub fn remove_result(root: &Path, directory: &Path, filename: &str) -> Result<()> {
    let name = directory.strip_prefix(root)?;
    if name.components().count() != 1
        || !matches!(
            name.components().next(),
            Some(std::path::Component::Normal(_))
        )
        || Path::new(filename).components().count() != 1
        || !matches!(
            Path::new(filename).components().next(),
            Some(std::path::Component::Normal(_))
        )
    {
        bail!("invalid result location");
    }
    let root = std::ffi::CString::new(root.as_os_str().as_encoded_bytes())?;
    let name = std::ffi::CString::new(name.as_os_str().as_encoded_bytes())?;
    let filename = std::ffi::CString::new(filename)?;
    let flags = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    let fd = unsafe { libc::open(root.as_ptr(), flags) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let root = unsafe { OwnedFd::from_raw_fd(fd) };
    let fd = unsafe { libc::openat(root.as_raw_fd(), name.as_ptr(), flags) };
    if fd < 0 {
        let error = std::io::Error::last_os_error();
        return if error.kind() == std::io::ErrorKind::NotFound {
            Ok(())
        } else {
            Err(error.into())
        };
    }
    let directory = unsafe { OwnedFd::from_raw_fd(fd) };
    if unsafe { libc::unlinkat(directory.as_raw_fd(), filename.as_ptr(), 0) } < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::NotFound {
            return Err(error.into());
        }
    }
    // Remove only an empty job directory. Unconfirmed files and unrelated entries stay.
    unsafe {
        libc::unlinkat(root.as_raw_fd(), name.as_ptr(), libc::AT_REMOVEDIR);
    }
    Ok(())
}
