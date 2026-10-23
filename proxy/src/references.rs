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
    pub mime_type: String,
    /// Normalized preview pixels exist only for this registered RAM reference.
    pub preview_png: std::sync::Arc<Vec<u8>>,
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
impl Reference {
    pub fn verify(&self, roots: &[PathBuf]) -> Result<()> {
        if hash(&read_image(&self.path, roots)?) != self.digest {
            bail!("registered reference has changed; register the image again")
        };
        Ok(())
    }
    pub fn result(&self, id: &str) -> Value {
        json!({"image_id":id,"width":self.width,"height":self.height,"mime_type":self.mime_type,"preview_path":self.path,"source_path":self.path,"preview_storage":"memory"})
    }
}
/// Expose the existing persistent Imagine output without a duplicate media file.
pub fn persistent_result(
    storage: &crate::mcp_storage::Storage,
    asset: &Value,
) -> Result<Vec<Value>> {
    let path = asset["saved_path"]
        .as_str()
        .map(Path::new)
        .ok_or_else(|| anyhow!("Imagine persistent image path is missing"))?;
    if asset["persistent"] != true
        || asset["bytes"].as_u64().is_none()
        || asset["sha256"].as_str().is_none()
    {
        bail!("Imagine persistent image metadata is missing");
    }
    storage.validate_saved_path(path)?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    Ok(vec![
        json!({"image_id":id,"path":path,"saved_path":path,"mime_type":asset["mime_type"],
        "bytes":asset["bytes"],"sha256":asset["sha256"],"storage":"saved","persistent":true,"persistent_root":storage.persistent_root()}),
    ])
}

#[cfg(test)]
use crate::task_fixtures;
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_preview_is_the_previous_oriented_rgb_contract_without_rewriting_original() {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(32, 16, |x, y| {
            image::Rgb([x as u8 * 7, y as u8 * 11, (x + y) as u8 * 3])
        }));
        let mut jpeg = Cursor::new(Vec::new());
        image.write_to(&mut jpeg, ImageFormat::Jpeg).unwrap();
        // Procedural EXIF TIFF entry: orientation 6, 90 degrees clockwise.
        let exif = b"Exif\0\0II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let mut original = jpeg.get_ref()[..2].to_vec();
        original.extend_from_slice(&[0xff, 0xe1]);
        original.extend_from_slice(&((exif.len() + 2) as u16).to_be_bytes());
        original.extend_from_slice(exif);
        original.extend_from_slice(&jpeg.get_ref()[2..]);
        let raw = image::load_from_memory(&original).unwrap();
        let expected = raw.rotate90().to_rgb8();
        let before = original.clone();
        let (png, width, height) = normalize(&original).unwrap();
        assert_eq!((width, height), (16, 32));
        assert_eq!(image::load_from_memory(&png).unwrap().to_rgb8(), expected);
        assert_eq!(original, before);
        assert_eq!(normalize(&png).unwrap().0, png);
    }

    #[test]
    fn reference_retains_rgba_preview_only_in_ram_and_verifies_original_source() {
        let root = tempfile::tempdir_in(task_fixtures::env_root("TMPDIR")).unwrap();
        let source = root.path().join("own-source.png");
        let rgba = image::RgbaImage::from_fn(8, 8, |x, y| {
            image::Rgba([x as u8, y as u8, 42, (x + y) as u8])
        });
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(rgba.clone())
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let original = encoded.into_inner();
        std::fs::write(&source, &original).unwrap();
        let (preview, width, height) = normalize(&original).unwrap();
        let reference = Reference {
            workflow: "own".into(),
            path: source.clone(),
            asset: "own-linked".into(),
            width,
            height,
            digest: hash(&original),
            mime_type: "image/png".into(),
            preview_png: std::sync::Arc::new(preview),
        };
        reference.verify(&[root.path().into()]).unwrap();
        assert_eq!(
            image::load_from_memory(&reference.preview_png)
                .unwrap()
                .to_rgba8(),
            rgba
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        assert_eq!(std::fs::read(&source).unwrap(), original);
        std::fs::write(&source, b"own changed source").unwrap();
        assert!(
            reference
                .verify(&[root.path().into()])
                .unwrap_err()
                .to_string()
                .contains("registered reference has changed")
        );
    }
}
