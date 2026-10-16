//! Model-independent sheet inputs and deterministic, CPU-only composition.
use anyhow::Result;
use image::{DynamicImage, Rgb, RgbImage};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct CharacterSheet {
    pub character_description: String,
    pub clothing_description: String,
    pub layout: Layout,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Layout {
    pub output_width: u32,
    pub portrait_fraction: f64,
}
impl Default for Layout {
    fn default() -> Self {
        Self {
            output_width: 1536,
            portrait_fraction: 0.45,
        }
    }
}
impl CharacterSheet {
    pub fn validate(&self, references: usize) -> Result<(), &'static str> {
        if self.character_description.trim().is_empty() && references == 0 {
            return Err("character_description_or_reference_required");
        }
        // Leave space for the server-owned instructions and reference role labels.
        if self.character_description.len() > 12000 || self.clothing_description.len() > 12000 {
            return Err("invalid_prompt_length");
        }
        self.layout.pass_sizes()?;
        Ok(())
    }
}
impl Layout {
    pub fn dimensions(&self) -> (u32, u32) {
        (self.output_width, self.output_width * 9 / 16)
    }
    pub fn boundary(&self) -> u32 {
        (self.output_width as f64 * self.portrait_fraction + 0.5).floor() as u32
    }
    pub fn pass_sizes(&self) -> Result<[(u32, u32); 2], &'static str> {
        if !(512..=4096).contains(&self.output_width)
            || !self.output_width.is_multiple_of(512)
            || !self.portrait_fraction.is_finite()
            || !(0.0..1.0).contains(&self.portrait_fraction)
        {
            return Err("invalid_character_sheet_layout");
        }
        let (w, h) = self.dimensions();
        let boundary = self.boundary();
        if boundary.min(w - boundary) < 32 {
            return Err("invalid_character_sheet_layout");
        }
        let aligned = |w: u32| ((w as f64 / 32.).round_ties_even() as u32 * 32).max(32);
        Ok([(aligned(boundary), h), (aligned(w - boundary), h)])
    }
}

/// One uniform continuous scale, with white outside the source. Portrait covers
/// its panel; turnaround contains all pixels. RGBA is composited onto white first.
pub fn compose(portrait: &[u8], turnaround: &[u8], layout: &Layout) -> Result<Vec<u8>> {
    layout.pass_sizes().map_err(anyhow::Error::msg)?;
    let (w, h) = layout.dimensions();
    let boundary = layout.boundary();
    let mut canvas = RgbImage::from_pixel(w, h, Rgb([255; 3]));
    for (bytes, offset, width, cover) in [
        (portrait, 0, boundary, true),
        (turnaround, boundary, w - boundary, false),
    ] {
        let src = image::load_from_memory(bytes)?.to_rgba8();
        let sx = width as f64 / src.width() as f64;
        let sy = h as f64 / src.height() as f64;
        let scale = if cover { sx.max(sy) } else { sx.min(sy) };
        let left = (width as f64 - src.width() as f64 * scale) / 2.;
        let top = (h as f64 - src.height() as f64 * scale) / 2.;
        for y in 0..h {
            for x in 0..width {
                let px = (x as f64 + 0.5 - left) / scale - 0.5;
                let py = (y as f64 + 0.5 - top) / scale - 0.5;
                let (ix, iy) = (px.floor() as i64, py.floor() as i64);
                let (fx, fy) = (px - ix as f64, py - iy as f64);
                let mut rgb = [0.; 3];
                for (dx, wx) in [(0, 1. - fx), (1, fx)] {
                    for (dy, wy) in [(0, 1. - fy), (1, fy)] {
                        let (xx, yy) = (ix + dx, iy + dy);
                        let pixel = if xx >= 0
                            && yy >= 0
                            && xx < src.width() as i64
                            && yy < src.height() as i64
                        {
                            src.get_pixel(xx as u32, yy as u32).0
                        } else {
                            [255; 4]
                        };
                        let alpha = pixel[3] as f64 / 255.;
                        for c in 0..3 {
                            rgb[c] += (pixel[c] as f64 * alpha + 255. * (1. - alpha)) * wx * wy;
                        }
                    }
                }
                canvas.put_pixel(
                    offset + x,
                    y,
                    Rgb(rgb.map(|v| v.round().clamp(0., 255.) as u8)),
                );
            }
        }
    }
    crate::workflow_image::encode(&DynamicImage::ImageRgb8(canvas))
}

/// Synthetic guide: front, right profile and back; no identity or body volume.
pub fn pose_guide(width: u32, height: u32) -> Result<Vec<u8>> {
    let mut image = RgbImage::from_pixel(width, height, Rgb([255; 3]));
    let radius = (height / 200).max(2) as i32;
    fn line(img: &mut RgbImage, a: (f64, f64), b: (f64, f64), color: Rgb<u8>, radius: i32) {
        let steps = (a.0 - b.0).abs().max((a.1 - b.1).abs()).ceil().max(1.) as u32;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let (x, y) = (
                (a.0 + (b.0 - a.0) * t).round() as i32,
                (a.1 + (b.1 - a.1) * t).round() as i32,
            );
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let (xx, yy) = (x + dx, y + dy);
                    if dx * dx + dy * dy <= radius * radius
                        && xx >= 0
                        && yy >= 0
                        && xx < img.width() as i32
                        && yy < img.height() as i32
                    {
                        img.put_pixel(xx as u32, yy as u32, color);
                    }
                }
            }
        }
    }
    for (index, color) in [Rgb([30, 100, 190]), Rgb([190, 80, 30]), Rgb([80, 140, 70])]
        .into_iter()
        .enumerate()
    {
        let center = (index as f64 + 0.5) * width as f64 / 3.;
        let point = |x: f64, y: f64| (center + x * width as f64 / 3., y * height as f64);
        let mut joints = [
            (0., 0.13),
            (0., 0.23),
            (0., 0.51),
            (-0.15, 0.26),
            (0.15, 0.26),
            (-0.21, 0.40),
            (0.21, 0.40),
            (-0.24, 0.53),
            (0.24, 0.53),
            (-0.09, 0.52),
            (0.09, 0.52),
            (-0.10, 0.71),
            (0.10, 0.71),
            (-0.11, 0.91),
            (0.11, 0.91),
        ];
        if index == 1 {
            for (n, j) in joints.iter_mut().enumerate().skip(3) {
                j.0 = if n % 2 == 1 { -0.025 } else { 0.025 };
            }
        }
        for (a, b) in [
            (0, 1),
            (1, 2),
            (3, 4),
            (9, 10),
            (1, 3),
            (1, 4),
            (3, 5),
            (5, 7),
            (4, 6),
            (6, 8),
            (2, 9),
            (2, 10),
            (9, 11),
            (11, 13),
            (10, 12),
            (12, 14),
        ] {
            line(
                &mut image,
                point(joints[a].0, joints[a].1),
                point(joints[b].0, joints[b].1),
                color,
                radius,
            );
        }
        if index == 0 {
            for x in [-0.055, 0.055] {
                line(&mut image, point(0., 0.13), point(x, 0.12), color, radius);
            }
        }
        if index == 1 {
            line(
                &mut image,
                point(0., 0.13),
                point(0.12, 0.13),
                color,
                radius,
            );
            for n in [13, 14] {
                let (x, y) = joints[n];
                line(&mut image, point(x, y), point(x + 0.10, y), color, radius);
            }
        }
    }
    crate::workflow_image::encode(&DynamicImage::ImageRgb8(image))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_matches_aio_geometry_and_rejects_invalid_panels() {
        let layout = Layout::default();
        assert_eq!(layout.dimensions(), (1536, 864));
        assert_eq!(layout.boundary(), 691);
        assert_eq!(layout.pass_sizes().unwrap(), [(704, 864), (832, 864)]);
        for (width, fraction) in [
            (513, 0.45),
            (4097, 0.45),
            (512, 0.),
            (512, 1.),
            (512, 0.001),
            (512, f64::NAN),
        ] {
            assert!(
                Layout {
                    output_width: width,
                    portrait_fraction: fraction
                }
                .pass_sizes()
                .is_err()
            );
        }
    }
    #[test]
    fn composition_is_deterministic_preserves_aspect_and_flattens_alpha() {
        let portrait = crate::workflow_image::encode(&DynamicImage::ImageRgba8(
            image::RgbaImage::from_pixel(128, 128, image::Rgba([255, 0, 0, 128])),
        ))
        .unwrap();
        let body = crate::workflow_image::encode(&DynamicImage::ImageRgb8(RgbImage::from_pixel(
            256,
            128,
            Rgb([0, 0, 255]),
        )))
        .unwrap();
        let layout = Layout {
            output_width: 512,
            portrait_fraction: 0.5,
        };
        let bytes = compose(&portrait, &body, &layout).unwrap();
        assert_eq!(bytes, compose(&portrait, &body, &layout).unwrap());
        let result = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(result.dimensions(), (512, 288));
        assert_eq!(result.get_pixel(128, 144).0, [255, 127, 127]);
        assert_eq!(result.get_pixel(384, 144).0, [0, 0, 255]);
        assert_eq!(result.get_pixel(384, 20).0, [255; 3]);
        // Body remains 256x128 centered: no stretching to the 288px panel height.
        assert_eq!(result.get_pixel(384, 79).0, [255; 3]);
        assert_eq!(result.get_pixel(384, 80).0, [0, 0, 255]);
    }
    #[test]
    fn pose_guide_is_deterministic_and_has_three_distinct_orientations() {
        let bytes = pose_guide(288, 288).unwrap();
        assert_eq!(bytes, pose_guide(288, 288).unwrap());
        let image = image::load_from_memory(&bytes).unwrap().to_rgb8();
        assert_eq!(image.dimensions(), (288, 288));
        for rgb in [[30, 100, 190], [190, 80, 30], [80, 140, 70]] {
            assert!(image.pixels().any(|p| p.0 == rgb));
        }
        assert_ne!(
            image::imageops::crop_imm(&image, 0, 0, 96, 288).to_image(),
            image::imageops::crop_imm(&image, 192, 0, 96, 288).to_image()
        );
    }
}
