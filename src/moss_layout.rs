//! MOSS-Image's native layout prompt, as ComfyUI's `MOSSImageLayoutPrompt` writes
//! it: the scene caption and one item per region, its box taken from where the
//! region's color lies in the control color map at the output size.
use crate::contract::RegionPrompt;
use anyhow::{Result, bail};
use serde::Serialize;

/// Channel difference still counted as a region's color (antialiased edges).
const TOLERANCE: u8 = 8;

#[derive(Serialize)]
struct Layout<'a> {
    summary: Summary<'a>,
    items: Vec<Item<'a>>,
    bbox_coord_space: &'static str,
    bbox_format: &'static str,
}
#[derive(Serialize)]
struct Summary<'a> {
    caption: &'a str,
}
#[derive(Serialize)]
struct Item<'a> {
    label: &'static str,
    caption: &'a str,
    bbox: [i64; 4],
}
/// `#rrggbb` as RGB.
pub fn parse_color(color: &str) -> Option<[u8; 3]> {
    let hex = color.strip_prefix('#').filter(|h| h.len() == 6)?;
    let channel = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
    Some([channel(0)?, channel(2)?, channel(4)?])
}
/// The pixel box `[y0, x0, y1, x1]` (end edges exclusive) of `color` in the map.
fn bounds(map: &image::RgbImage, color: [u8; 3]) -> Option<[u32; 4]> {
    let mut found: Option<[u32; 4]> = None;
    for (x, y, p) in map.enumerate_pixels() {
        if (0..3).all(|c| p[c].abs_diff(color[c]) <= TOLERANCE) {
            let b = found.get_or_insert([y, x, y + 1, x + 1]);
            *b = [b[0].min(y), b[1].min(x), b[2].max(y + 1), b[3].max(x + 1)];
        }
    }
    found
}
/// The layout prompt for `width` × `height`; the scene prompt alone without
/// regions. The map is placed as the control places it: scaled to cover the
/// output and center-cropped. Regions cropped away are left out.
pub fn layout_prompt(
    prompt: &str,
    regions: &[RegionPrompt],
    color_map: &[u8],
    width: u32,
    height: u32,
) -> Result<String> {
    if regions.is_empty() {
        return Ok(prompt.to_owned());
    }
    let map = image::load_from_memory(color_map)?.to_rgb8();
    let (mw, mh) = (f64::from(map.width()), f64::from(map.height()));
    let (w, h) = (i64::from(width), i64::from(height));
    let scale = (width as f64 / mw).max(height as f64 / mh);
    let (sw, sh) = (
        ((mw * scale).round() as i64).max(w),
        ((mh * scale).round() as i64).max(h),
    );
    let (ox, oy) = ((sw - w) / 2, (sh - h) / 2);
    let mut items = Vec::new();
    for region in regions {
        let Some(color) = parse_color(&region.color) else {
            bail!("invalid region color");
        };
        let Some([y0, x0, y1, x1]) = bounds(&map, color) else {
            bail!(crate::engine::EngineError("region_not_in_map"));
        };
        let edge = |v: u32, offset: i64, limit: i64, up: bool| {
            let v = f64::from(v) * scale;
            ((if up { v.ceil() } else { v.floor() }) as i64 - offset).clamp(0, limit)
        };
        let (y0, x0) = (edge(y0, oy, h, false), edge(x0, ox, w, false));
        let (y1, x1) = (edge(y1, oy, h, true), edge(x1, ox, w, true));
        if y1 <= y0 || x1 <= x0 {
            continue;
        }
        items.push(Item {
            label: if region.character {
                "character"
            } else {
                "object"
            },
            caption: &region.prompt,
            bbox: [y0 - h / 2, x0 - w / 2, y1 - h / 2, x1 - w / 2],
        });
    }
    if items.is_empty() {
        return Ok(prompt.to_owned());
    }
    Ok(serde_json::to_string(&Layout {
        summary: Summary {
            caption: prompt.trim(),
        },
        items,
        bbox_coord_space: "abs_centralized",
        bbox_format: "yxyx",
    })?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(width: u32, height: u32, boxes: &[([u8; 3], [u32; 4])]) -> Vec<u8> {
        let image = image::RgbImage::from_fn(width, height, |x, y| {
            boxes
                .iter()
                .find(|(_, [y0, x0, y1, x1])| (*y0..*y1).contains(&y) && (*x0..*x1).contains(&x))
                .map_or(image::Rgb([0, 0, 0]), |(c, _)| image::Rgb(*c))
        });
        let mut png = std::io::Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        png.into_inner()
    }
    fn region(color: &str, prompt: &str, character: bool) -> RegionPrompt {
        RegionPrompt {
            color: color.into(),
            prompt: prompt.into(),
            character,
        }
    }

    #[test]
    fn regions_become_comfy_layout_items_with_centered_pixel_boxes() {
        let png = map(
            100,
            50,
            &[
                ([255, 0, 0], [10, 20, 30, 60]),
                ([0, 255, 255], [0, 0, 50, 10]),
            ],
        );
        let prompt = layout_prompt(
            "  A park, daytime  ",
            &[
                region("#ff0000", "a woman in a red coat", true),
                region("#00ffff", "a tree", false),
            ],
            &png,
            200,
            100,
        )
        .unwrap();
        // Each box doubles with the output and moves to the canvas centre.
        assert_eq!(
            prompt,
            r#"{"summary":{"caption":"A park, daytime"},"items":[{"label":"character","caption":"a woman in a red coat","bbox":[-30,-60,10,20]},{"label":"object","caption":"a tree","bbox":[-50,-100,50,-80]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}"#
        );
        assert_eq!(
            layout_prompt("scene", &[], &png, 200, 100).unwrap(),
            "scene"
        );
    }

    #[test]
    fn cropped_regions_drop_out_and_missing_colors_fail() {
        // A square output keeps the middle 50 of the 100 columns.
        let png = map(
            100,
            50,
            &[
                ([255, 0, 0], [0, 0, 50, 20]),
                ([0, 0, 255], [0, 40, 50, 60]),
            ],
        );
        let prompt = layout_prompt(
            "scene",
            &[
                region("#ff0000", "cropped", false),
                region("#0000ff", "kept", false),
            ],
            &png,
            50,
            50,
        )
        .unwrap();
        assert!(prompt.contains(r#""caption":"kept","bbox":[-25,-10,25,10]"#));
        assert!(!prompt.contains("cropped"));
        assert!(layout_prompt("scene", &[region("#123456", "x", false)], &png, 50, 50).is_err());
        assert_eq!(parse_color("#E6194b"), Some([230, 25, 75]));
        assert_eq!(parse_color("e6194b"), None);
    }
}
