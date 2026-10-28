//! Shared reference preprocessing: AIO's shrink-only geometry and /32 center crop.
use anyhow::{Result, bail};
use image::{DynamicImage, ImageDecoder, ImageFormat, imageops::FilterType};
use std::io::Cursor;
/// Preserve the MCP's existing display-image decode contract in memory. The
/// original file and its integrity identity remain raw and are never rewritten.
pub fn normalize_reference(bytes: &[u8]) -> Result<Vec<u8>> {
    let reader = image::ImageReader::new(Cursor::new(bytes)).with_guessed_format()?;
    let mut decoder = reader.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut image = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);
    let image = if image.color().has_alpha() {
        DynamicImage::ImageRgba8(image.to_rgba8())
    } else {
        DynamicImage::ImageRgb8(image.to_rgb8())
    };
    encode(&image)
}
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
/// A MOSS edit reference at the output size: scaled (Lanczos) to cover
/// `width` × `height` and center-cropped. ComfyUI's MOSS edit encoder uses
/// references at the size they come in, and edits keep their layout only when
/// that size is the output's.
pub fn cover(bytes: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    let image = image::load_from_memory(bytes)?;
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 || width == 0 || height == 0 {
        bail!("reference dimensions are too small")
    }
    let scale = (f64::from(width) / f64::from(w)).max(f64::from(height) / f64::from(h));
    let (sw, sh) = (
        ((f64::from(w) * scale).round() as u32).max(width),
        ((f64::from(h) * scale).round() as u32).max(height),
    );
    let image = if (sw, sh) != (w, h) {
        image.resize_exact(sw, sh, FilterType::Lanczos3)
    } else {
        image
    };
    encode(&image.crop_imm((sw - width) / 2, (sh - height) / 2, width, height))
}
pub fn encode(image: &DynamicImage) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    image.write_to(&mut bytes, ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aio_geometry_never_enlarges_and_caps_each_reference_independently() {
        // AIO formulas with an explicit pixel budget, before and after /32 crop.
        for (w, h, cap, expected) in [
            (512, 512, 1_000_000, (512, 512, 512, 512)),
            (1600, 896, 1_000_000, (1336, 748, 1312, 736)),
            (896, 1600, 1_000_000, (748, 1336, 736, 1312)),
            (2048, 2048, 1_000_000, (1000, 1000, 992, 992)),
            (1600, 896, 1_048_576, (1368, 766, 1344, 736)),
            (65, 67, 0, (65, 67, 64, 64)),
        ] {
            assert_eq!(dimensions(w, h, cap), expected);
        }
    }

    #[test]
    fn moss_references_cover_the_output_and_crop_the_center() {
        let original =
            image::RgbaImage::from_fn(60, 40, |x, _| image::Rgba([x as u8 * 4, 0, 0, 255]));
        let bytes = encode(&DynamicImage::ImageRgba8(original.clone())).unwrap();
        // Same aspect scales; a wider output crops top and bottom; same size is untouched.
        for (w, h) in [(120, 80), (120, 40), (60, 40), (32, 64)] {
            let out = image::load_from_memory(&cover(&bytes, w, h).unwrap()).unwrap();
            assert_eq!((out.width(), out.height()), (w, h));
        }
        let same = image::load_from_memory(&cover(&bytes, 60, 40).unwrap())
            .unwrap()
            .to_rgba8();
        assert_eq!(same, original);
        // A tall output keeps the middle columns.
        let tall = image::load_from_memory(&cover(&bytes, 20, 40).unwrap())
            .unwrap()
            .to_rgba8();
        assert_eq!(tall.get_pixel(0, 0), original.get_pixel(20, 0));
    }

    #[test]
    fn preprocessing_preserves_small_pixels_and_center_crops_all_channels() {
        let original = image::RgbaImage::from_fn(67, 69, |x, y| {
            image::Rgba([x as u8, y as u8, 42, (x + y) as u8])
        });
        let bytes = encode(&DynamicImage::ImageRgba8(original.clone())).unwrap();
        let prepared = prepare(&bytes, 1_000_000).unwrap();
        let cropped = image::load_from_memory(&prepared).unwrap().to_rgba8();
        assert_eq!(cropped.dimensions(), (64, 64));
        for (x, y, pixel) in cropped.enumerate_pixels() {
            assert_eq!(pixel, original.get_pixel(x + 1, y + 2));
        }
        let aligned = prepare(&prepared, 1_000_000).unwrap();
        assert_eq!(
            image::load_from_memory(&aligned).unwrap().to_rgba8(),
            cropped
        );
        assert_eq!(
            image::load_from_memory(&bytes).unwrap().to_rgba8(),
            original
        );
    }

    #[test]
    fn preprocessing_shrinks_before_crop_and_rejects_too_thin_inputs() {
        let image = image::RgbaImage::from_pixel(1600, 896, image::Rgba([64, 128, 192, 255]));
        let bytes = encode(&DynamicImage::ImageRgba8(image)).unwrap();
        let result = image::load_from_memory(&prepare(&bytes, 1_000_000).unwrap()).unwrap();
        assert_eq!((result.width(), result.height()), (1312, 736));
        let tiny = encode(&DynamicImage::new_rgb8(31, 64)).unwrap();
        assert!(prepare(&tiny, 1_000_000).is_err());
    }
}
