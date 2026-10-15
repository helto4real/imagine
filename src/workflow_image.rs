//! CPU reference geometry for the approved workflow profile. The default
//! frozen profile continues to use its original native reference preprocessing.
use anyhow::{Result, bail};
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use std::io::Cursor;
pub fn dimensions(width: u32, height: u32, pixels: u32) -> (u32, u32, u32, u32) {
    let scale = if pixels > 0 && u64::from(width) * u64::from(height) > u64::from(pixels) {
        (f64::from(pixels) / (f64::from(width) * f64::from(height))).sqrt()
    } else {
        1.
    };
    let w = (f64::from(width) * scale) as u32;
    let h = (f64::from(height) * scale) as u32;
    (w, h, w / 32 * 32, h / 32 * 32)
}
pub fn prepare(bytes: &[u8], pixels: u32) -> Result<Vec<u8>> {
    let image = image::load_from_memory(bytes)?;
    let (w, h, cw, ch) = dimensions(image.width(), image.height(), pixels);
    if cw < 32 || ch < 32 {
        bail!("reference dimensions are too small")
    };
    let image = if (w, h) != (image.width(), image.height()) {
        image.resize_exact(w, h, FilterType::Lanczos3)
    } else {
        image
    };
    let image = image.crop_imm((w - cw) / 2, (h - ch) / 2, cw, ch);
    encode(&image)
}
pub fn encode(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png)?;
    Ok(bytes.into_inner())
}
