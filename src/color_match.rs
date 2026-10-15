//! CPU port of AIO's guided Lab chrominance transfer and reference tone curve.
//! Only workflow requests opt in; the frozen native profile bypasses this module.
use anyhow::Result;
use image::{DynamicImage, RgbaImage};
type Pixel = [f32; 4];
fn lab(rgb: [f32; 3]) -> [f32; 3] {
    let c = rgb.map(|x| {
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).max(0.).powf(2.4)
        }
    });
    let m = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.072175],
        [0.0193339, 0.119192, 0.9503041],
    ];
    let white = [0.95047, 1., 1.08883];
    let mut f = [0.; 3];
    for j in 0..3 {
        let x = (0..3).map(|k| m[j][k] * c[k]).sum::<f32>() / white[j];
        f[j] = if x > (6f32 / 29.).powi(3) {
            x.max(0.).powf(1. / 3.)
        } else {
            x / (3. * (6f32 / 29.).powi(2)) + 4. / 29.
        }
    }
    [
        116. * f[1] - 16.,
        500. * (f[0] - f[1]),
        200. * (f[1] - f[2]),
    ]
}
fn linear(lab: [f32; 3]) -> [f32; 3] {
    let y = (lab[0] + 16.) / 116.;
    let f = [y + lab[1] / 500., y, y - lab[2] / 200.];
    let white = [0.95047, 1., 1.08883];
    let xyz = std::array::from_fn::<_, 3, _>(|i| {
        white[i]
            * if f[i] > 6. / 29. {
                f[i].powi(3)
            } else {
                3. * (6f32 / 29.).powi(2) * (f[i] - 4. / 29.)
            }
    });
    let m = [
        [3.240_455, -1.537_138_8, -0.49853155],
        [-0.969_266_4, 1.876_010_9, 0.04155608],
        [0.05564342, -0.20402585, 1.057_225_1],
    ];
    std::array::from_fn(|i| (0..3).map(|k| m[i][k] * xyz[k]).sum())
}
fn gamut(l: [f32; 3]) -> [f32; 3] {
    let mut rgb = linear(l);
    let valid = |rgb: [f32; 3]| rgb.iter().all(|v| *v >= -1e-6 && *v <= 1. + 1e-6);
    if !valid(rgb) {
        let (mut low, mut high) = (0., 1.);
        for _ in 0..12 {
            let mid = (low + high) / 2.;
            if valid(linear([l[0], l[1] * mid, l[2] * mid])) {
                low = mid
            } else {
                high = mid
            }
        }
        rgb = linear([l[0], l[1] * low, l[2] * low]);
    }
    rgb.map(|v| {
        let v = v.clamp(0., 1.);
        if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1. / 2.4) - 0.055
        }
    })
}
fn pixels(image: &DynamicImage) -> Vec<Pixel> {
    image.to_rgba32f().pixels().map(|p| p.0).collect()
}
fn weights(source: usize, target: usize, index: usize, antialias: bool) -> Vec<(usize, f32)> {
    let ratio = source as f32 / target as f32;
    let scale = if antialias { ratio.max(1.) } else { 1. };
    let center = (index as f32 + 0.5) * ratio;
    let start = (center - scale - 0.5).ceil() as isize;
    let stop = (center + scale - 0.5).floor() as isize;
    let mut values = Vec::new();
    for pos in start..=stop {
        let weight = (1. - ((pos as f32 + 0.5 - center) / scale).abs()).max(0.);
        if antialias && (pos < 0 || pos >= source as isize) {
            continue;
        };
        if weight > 0. {
            values.push((pos.clamp(0, source as isize - 1) as usize, weight))
        }
    }
    let sum: f32 = values.iter().map(|(_, v)| v).sum();
    for (_, v) in &mut values {
        *v /= sum
    }
    values
}
fn resize<const N: usize>(
    input: &[[f32; N]],
    sw: usize,
    sh: usize,
    w: usize,
    h: usize,
    aa: bool,
) -> Vec<[f32; N]> {
    let xs: Vec<_> = (0..w).map(|x| weights(sw, w, x, aa)).collect();
    let ys: Vec<_> = (0..h).map(|y| weights(sh, h, y, aa)).collect();
    let mut horizontal = vec![[0.; N]; w * sh];
    for y in 0..sh {
        for x in 0..w {
            for (s, v) in &xs[x] {
                for c in 0..N {
                    horizontal[y * w + x][c] += input[y * sw + s][c] * v
                }
            }
        }
    }
    let mut out = vec![[0.; N]; w * h];
    for y in 0..h {
        for x in 0..w {
            for (s, v) in &ys[y] {
                for c in 0..N {
                    out[y * w + x][c] += horizontal[s * w + x][c] * v
                }
            }
        }
    }
    out
}
fn work(input: &[Pixel], sw: usize, sh: usize, w: usize, h: usize) -> Vec<Pixel> {
    let premul: Vec<_> = input
        .iter()
        .map(|p| [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]])
        .collect();
    resize(&premul, sw, sh, w, h, true)
        .iter()
        .map(|p| {
            let l = lab([
                p[0] / p[3].max(1e-6),
                p[1] / p[3].max(1e-6),
                p[2] / p[3].max(1e-6),
            ]);
            [l[0], l[1], l[2], p[3]]
        })
        .collect()
}
fn box_mean<const N: usize>(input: &[[f32; N]], w: usize, h: usize) -> Vec<[f32; N]> {
    let mut tmp = vec![[0.; N]; w * h];
    let mut out = tmp.clone();
    for y in 0..h {
        for x in 0..w {
            for xx in x.saturating_sub(8)..=(x + 8).min(w - 1) {
                for c in 0..N {
                    tmp[y * w + x][c] += input[y * w + xx][c] / 17.
                }
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for yy in y.saturating_sub(8)..=(y + 8).min(h - 1) {
                for c in 0..N {
                    out[y * w + x][c] += tmp[yy * w + x][c] / 17.
                }
            }
        }
    }
    out
}
fn chroma(
    source: &mut [Pixel],
    reference: &[Pixel],
    w: usize,
    h: usize,
    ww: usize,
    wh: usize,
    strength: f32,
) {
    let s = work(source, w, h, ww, wh);
    let moments: Vec<[f32; 7]> = s
        .iter()
        .zip(reference)
        .map(|(s, r)| {
            let weight = s[3] * r[3];
            let guide = s[0] / 100.;
            let a = r[1] - s[1];
            let b = r[2] - s[2];
            [
                weight,
                weight * guide,
                weight * guide * guide,
                weight * a,
                weight * b,
                weight * guide * a,
                weight * guide * b,
            ]
        })
        .collect();
    let moments = box_mean(&moments, ww, wh);
    let coefficients: Vec<[f32; 5]> = moments
        .iter()
        .map(|m| {
            let weight = m[0];
            let mean = m.map(|v| v / weight.max(1e-6));
            let variance = (mean[2] - mean[1] * mean[1]).max(0.) + 1e-4;
            let a = (mean[5] - mean[1] * mean[3]) / variance;
            let b = (mean[6] - mean[1] * mean[4]) / variance;
            [
                a * weight,
                b * weight,
                (mean[3] - a * mean[1]) * weight,
                (mean[4] - b * mean[1]) * weight,
                weight,
            ]
        })
        .collect();
    let fields: Vec<[f32; 5]> = box_mean(&coefficients, ww, wh)
        .iter()
        .zip(reference)
        .map(|(c, r)| {
            [
                c[0] / c[4].max(1e-6),
                c[1] / c[4].max(1e-6),
                c[2] / c[4].max(1e-6),
                c[3] / c[4].max(1e-6),
                r[3],
            ]
        })
        .collect();
    let xs: Vec<_> = (0..w).map(|x| weights(ww, w, x, false)).collect();
    for y in 0..h {
        let ys = weights(wh, h, y, false);
        for x in 0..w {
            let p = &mut source[y * w + x];
            if p[3] <= 0. {
                continue;
            }
            let mut f = [0.; 5];
            for (yy, wy) in &ys {
                for (xx, wx) in &xs[x] {
                    for c in 0..5 {
                        f[c] += fields[yy * ww + xx][c] * wy * wx
                    }
                }
            }
            let l = lab([p[0], p[1], p[2]]);
            let da = strength * (f[0] * (l[0] / 100.) + f[2]);
            let db = strength * (f[1] * (l[0] / 100.) + f[3]);
            if f[4] > 1e-6 && da.abs().max(db.abs()) > 1e-5 {
                let rgb = gamut([l[0], l[1] + da, l[2] + db]);
                p[..3].copy_from_slice(&rgb)
            }
        }
    }
}
fn quantiles(p: &[Pixel]) -> Option<[f64; 3]> {
    let mut values: Vec<_> = p.iter().map(|p| (p[0] / 100., p[3])).collect();
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f32 = values.iter().map(|v| v.1).sum();
    if total <= 1e-6 {
        return None;
    }
    Some([0.05, 0.5, 0.95].map(|q| {
        let mut sum = 0.;
        for (v, a) in &values {
            sum += a;
            if sum >= total * q {
                return f64::from(v.clamp(0., 1.));
            }
        }
        f64::from(values.last().unwrap().0.clamp(0., 1.))
    }))
}
fn knots(
    source: [f64; 3],
    reference: [f64; 3],
    brightness: f64,
    contrast: f64,
) -> Option<Vec<[f32; 3]>> {
    let [low, middle, high] = source;
    let target = (middle + brightness * (reference[1] - middle))
        .clamp(0.02 * middle, 1. - 0.02 * (1. - middle));
    let span = high - low;
    let mut ratio = if span > 1e-5 {
        1. + contrast * ((reference[2] - reference[0]) / span - 1.)
    } else {
        1.
    };
    ratio = ratio.max(0.02);
    let (left, right) = (middle - low, high - middle);
    if left > 1e-5 {
        ratio = ratio.min((target - 0.02 * low) / left)
    }
    if right > 1e-5 {
        ratio = ratio.min((1. - 0.02 * (1. - high) - target) / right)
    }
    let mut pairs = vec![(0., 0.), (1., 1.)];
    for (x, y) in [
        (middle, target),
        (low, target - left * ratio),
        (high, target + right * ratio),
    ] {
        if pairs.iter().all(|(v, _)| (x - v).abs() > 1e-5) {
            pairs.push((x, y))
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    if pairs.iter().all(|(x, y)| (x - y).abs() < 1e-6) {
        return None;
    }
    let slopes: Vec<_> = pairs
        .windows(2)
        .map(|p| (p[1].1 - p[0].1) / (p[1].0 - p[0].0))
        .collect();
    Some(
        pairs
            .iter()
            .enumerate()
            .map(|(i, (x, y))| {
                let d = if i == 0 {
                    slopes[0]
                } else if i == pairs.len() - 1 {
                    slopes[i - 1]
                } else {
                    2. * slopes[i - 1] * slopes[i] / (slopes[i - 1] + slopes[i])
                };
                [*x as f32, *y as f32, d as f32]
            })
            .collect(),
    )
}
fn tone(
    source: &mut [Pixel],
    reference: &[Pixel],
    dimensions: (usize, usize),
    work_dimensions: (usize, usize),
    brightness: f32,
    contrast: f32,
) {
    let (w, h) = dimensions;
    let (ww, wh) = work_dimensions;
    let (Some(s), Some(r)) = (quantiles(&work(source, w, h, ww, wh)), quantiles(reference)) else {
        return;
    };
    let Some(knots) = knots(s, r, f64::from(brightness), f64::from(contrast)) else {
        return;
    };
    for p in source {
        if p[3] <= 0. {
            continue;
        }
        let mut l = lab([p[0], p[1], p[2]]);
        let v = (l[0] / 100.).clamp(0., 1.);
        let i = knots
            .partition_point(|k| k[0] <= v)
            .saturating_sub(1)
            .min(knots.len() - 2);
        let [x0, y0, d0] = knots[i];
        let [x1, y1, d1] = knots[i + 1];
        let position = (v - x0) / (x1 - x0);
        let slope = (y1 - y0) / (x1 - x0);
        let cross = position * (1. - position);
        let target = 100.
            * (y0
                + (y1 - y0) * (slope * position * position + d0 * cross)
                    / (slope + (d0 + d1 - 2. * slope) * cross));
        if (target - l[0]).abs() > 1e-5 {
            l[0] = target;
            p[..3].copy_from_slice(&gamut(l))
        }
    }
}
pub fn apply(
    source: &[u8],
    reference: &[u8],
    chroma_strength: f32,
    brightness: f32,
    contrast: f32,
) -> Result<Vec<u8>> {
    if chroma_strength == 0. && brightness == 0. && contrast == 0. {
        return Ok(source.to_vec());
    };
    let image = image::load_from_memory(source)?;
    let ref_image = image::load_from_memory(reference)?;
    let (w, h) = (image.width() as usize, image.height() as usize);
    let scale = (512. / w.max(h) as f64).min(1.);
    let (ww, wh) = (
        (w as f64 * scale).round_ties_even().max(1.) as usize,
        (h as f64 * scale).round_ties_even().max(1.) as usize,
    );
    let reference = work(
        &pixels(&ref_image),
        ref_image.width() as usize,
        ref_image.height() as usize,
        ww,
        wh,
    );
    let mut source = pixels(&image);
    if chroma_strength > 0. {
        chroma(&mut source, &reference, w, h, ww, wh, chroma_strength)
    }
    if brightness > 0. || contrast > 0. {
        tone(
            &mut source,
            &reference,
            (w, h),
            (ww, wh),
            brightness,
            contrast,
        )
    }
    let mut out = RgbaImage::new(w as u32, h as u32);
    for (pixel, value) in out.pixels_mut().zip(source) {
        *pixel = image::Rgba(value.map(|v| (v.clamp(0., 1.) * 255.).round() as u8));
    }
    crate::workflow_image::encode(&DynamicImage::ImageRgba8(out))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn color_roundtrip() {
        for rgb in [[0., 0., 0.], [1., 1., 1.], [0.2, 0.4, 0.7], [1., 0., 0.]] {
            let actual = gamut(lab(rgb));
            for (a, b) in rgb.into_iter().zip(actual) {
                assert!((a - b).abs() < 0.001, "{a} {b}")
            }
        }
    }
    #[test]
    fn tone_identity() {
        assert!(knots([0.1, 0.4, 0.8], [0.1, 0.4, 0.8], 1., 1.).is_none());
    }
}
