//! Director vision normalization. Existing libjpeg tools run only on in-memory pipes.
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use std::{
    io::{Cursor, Read, Write},
    process::{Command, Stdio},
    thread,
};
pub struct Normalized {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub jpeg: Vec<u8>,
}
fn codec(program: &str, args: &[&str], input: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| "vision_codec_unavailable")?;
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(input));
        let reader = scope.spawn(move || {
            let mut output = vec![];
            stdout
                .take(crate::video::media::MAX_UPLOAD + 1)
                .read_to_end(&mut output)
                .map(|_| output)
        });
        let status = child.wait().map_err(|_| {
            let _ = child.kill();
            let _ = child.wait();
            "vision_codec_failed"
        });
        let written = writer.join().map_err(|_| "vision_codec_failed")?;
        let output = reader
            .join()
            .map_err(|_| "vision_codec_failed")?
            .map_err(|_| "vision_codec_failed")?;
        if !status?.success()
            || written.is_err()
            || output.len() as u64 > crate::video::media::MAX_UPLOAD
        {
            return Err("vision_codec_failed");
        }
        Ok(output)
    })
}
fn ppm(bytes: &[u8], width: u32, height: u32) -> Result<Vec<u8>, &'static str> {
    let mut pos = 0;
    let mut fields = vec![];
    while fields.len() < 4 {
        while bytes.get(pos).is_some_and(u8::is_ascii_whitespace) {
            pos += 1;
        }
        if bytes.get(pos) == Some(&b'#') {
            while bytes.get(pos).is_some_and(|b| *b != b'\n') {
                pos += 1;
            }
            continue;
        }
        let from = pos;
        while bytes.get(pos).is_some_and(|b| !b.is_ascii_whitespace()) {
            pos += 1;
        }
        if from == pos {
            return Err("vision_codec_failed");
        }
        fields.push(&bytes[from..pos]);
    }
    let channels = match fields[0] {
        b"P6" => 3,
        b"P5" => 1,
        _ => return Err("vision_codec_failed"),
    };
    if fields[1] != width.to_string().as_bytes()
        || fields[2] != height.to_string().as_bytes()
        || fields[3] != b"255"
    {
        return Err("vision_codec_failed");
    }
    if bytes.get(pos) == Some(&b'\r') && bytes.get(pos + 1) == Some(&b'\n') {
        pos += 2;
    } else {
        pos += 1;
    }
    let rgb = bytes.get(pos..).ok_or("vision_codec_failed")?;
    if rgb.len() != width as usize * height as usize * channels {
        return Err("vision_codec_failed");
    }
    Ok(if channels == 1 {
        rgb.iter().flat_map(|v| [*v; 3]).collect()
    } else {
        rgb.to_vec()
    })
}
pub fn decode_rgb(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), &'static str> {
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| "invalid_media")?;
    let format = reader.format().ok_or("unsupported_media_type")?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(crate::video::media::MAX_UPLOAD);
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(|_| "invalid_media")?;
    let orientation = decoder.orientation().map_err(|_| "invalid_media")?;
    let (width, height) = decoder.dimensions();
    if width as u64 * height as u64 * 3 > crate::video::media::MAX_UPLOAD {
        return Err("invalid_media_dimensions");
    }
    let mut image = if format == ImageFormat::Jpeg {
        let rgb = ppm(&codec("djpeg", &["-dct", "int"], bytes)?, width, height)?;
        DynamicImage::ImageRgb8(
            image::RgbImage::from_raw(width, height, rgb).ok_or("invalid_media")?,
        )
    } else {
        DynamicImage::from_decoder(decoder).map_err(|_| "invalid_media")?
    };
    image.apply_orientation(orientation);
    // Pillow loads a 16-bit grayscale PNG as integer samples; convert("RGB")
    // clips those values to 8 bits, rather than rescaling the 16-bit range.
    let rgb = match image {
        DynamicImage::ImageLuma16(luma) => {
            image::RgbImage::from_fn(luma.width(), luma.height(), |x, y| {
                image::Rgb([luma[(x, y)].0[0].min(255) as u8; 3])
            })
        }
        DynamicImage::ImageLumaA16(luma) => {
            image::RgbImage::from_fn(luma.width(), luma.height(), |x, y| {
                image::Rgb([(luma[(x, y)].0[0] >> 8) as u8; 3])
            })
        }
        DynamicImage::ImageRgb16(rgb) => {
            image::RgbImage::from_fn(rgb.width(), rgb.height(), |x, y| {
                image::Rgb(rgb[(x, y)].0.map(|v| (v >> 8) as u8))
            })
        }
        DynamicImage::ImageRgba16(rgba) => {
            image::RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
                let [r, g, b, _] = rgba[(x, y)].0;
                image::Rgb([r, g, b].map(|v| (v >> 8) as u8))
            })
        }
        other => other.to_rgb8(),
    };
    let (width, height) = rgb.dimensions();
    Ok((width, height, rgb.into_raw()))
}
pub fn normalize(bytes: &[u8]) -> Result<Normalized, &'static str> {
    let (width, height, bytes) = decode_rgb(bytes)?;
    let mut rgb = image::RgbImage::from_raw(width, height, bytes).ok_or("invalid_media")?;
    let largest = width.max(height);
    if largest > 1280 {
        let scale = 1280. / largest as f64;
        let w = (width as f64 * scale).round_ties_even().max(1.) as u32;
        let h = (height as f64 * scale).round_ties_even().max(1.) as u32;
        rgb = image::RgbImage::from_raw(
            w,
            h,
            crate::video::rgb_resize::resize(rgb.as_raw(), width, height, w, h),
        )
        .ok_or("invalid_media")?;
    }
    let (width, height) = rgb.dimensions();
    let rgb = rgb.into_raw();
    let mut packed = format!("P6\n{width} {height}\n255\n").into_bytes();
    packed.extend_from_slice(&rgb);
    let jpeg = codec(
        "cjpeg",
        &[
            "-quality",
            "90",
            "-optimize",
            "-sample",
            "2x2",
            "-dct",
            "int",
        ],
        &packed,
    )?;
    Ok(Normalized {
        width,
        height,
        rgb,
        jpeg,
    })
}
