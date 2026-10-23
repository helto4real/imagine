use anyhow::{Result, anyhow, bail};
use image::{ImageDecoder, ImageFormat, ImageReader};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{Cursor, Read},
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
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
pub fn store_preview(
    storage: &crate::mcp_storage::Storage,
    data: &[u8],
) -> Result<(PathBuf, String)> {
    let digest = hash(data);
    let dir = storage.directory("references")?;
    let filename = format!("{digest}.png");
    let path = dir.join(&filename);
    if let Ok(existing) = storage.read(&path) {
        if hash(&existing) != digest {
            bail!("registered reference has changed; register the image again");
        }
    } else {
        storage.save(&dir, &filename, data)?;
    }
    Ok((path, digest))
}
impl Reference {
    pub fn verify(&self, storage: &crate::mcp_storage::Storage) -> Result<()> {
        if hash(&storage.read(&self.path)?) != self.digest {
            bail!("registered reference has changed; register the image again")
        };
        Ok(())
    }
    pub fn result(&self, id: &str) -> Value {
        json!({"image_id":id,"width":self.width,"height":self.height,"mime_type":"image/png","preview_path":self.path})
    }
}
/// Save an ephemeral MCP delivery copy on the pinned encrypted destination.
pub fn save_result(
    storage: &crate::mcp_storage::Storage,
    directory: &Path,
    index: usize,
    bytes: &[u8],
) -> Result<Vec<Value>> {
    let id = uuid::Uuid::new_v4().simple().to_string();
    let path = storage.save(directory, &format!("{index}.png"), bytes)?;
    Ok(vec![
        json!({"image_id":id,"path":path,"mime_type":"image/png",
        "bytes":bytes.len(),"sha256":hash(bytes),"storage":"temporary","persistent_root":storage.persistent_root()}),
    ])
}
