//! Read an owned session image into bounded RGB; resize belongs to the native Pillow boundary.
use crate::{
    h3_protocol::{ImageInput, MAX_INPUT_IMAGE},
    storage,
};
pub fn validate_asset(v: &serde_json::Value) -> Result<(), &'static str> {
    let w = v["width"].as_u64().unwrap_or(0);
    let h = v["height"].as_u64().unwrap_or(0);
    if v["bytes"]
        .as_u64()
        .is_none_or(|v| v > MAX_INPUT_IMAGE as u64)
        || w == 0
        || h == 0
        || w > 8192
        || h > 8192
        || w * h * 3 > MAX_INPUT_IMAGE as u64
        || !matches!(
            v["mime_type"].as_str(),
            Some("image/png" | "image/jpeg" | "image/webp")
        )
    {
        return Err("invalid_h3_start_image");
    }
    Ok(())
}
pub fn decode(bytes: &[u8]) -> anyhow::Result<(ImageInput, Vec<u8>)> {
    anyhow::ensure!(bytes.len() <= MAX_INPUT_IMAGE, "invalid_h3_start_image");
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    let (w, h) = reader.into_dimensions()?;
    let d = ImageInput {
        width: w,
        height: h,
        sha256: "0".repeat(64),
    };
    anyhow::ensure!(d.valid(), "invalid_h3_start_image");
    let rgb = image::load_from_memory(bytes)?.to_rgb8().into_raw();
    Ok((
        ImageInput {
            sha256: storage::hash(&rgb),
            ..d
        },
        rgb,
    ))
}
