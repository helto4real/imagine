//! RGB8 Lanczos padding with Pillow's separable, quantized 8-bit semantics.
//! Reference: Pillow 12.3.0 src/libImaging/Resample.c (HPND); no Python dependency.
const PRECISION: u32 = 22;
fn sinc(x: f64) -> f64 {
    if x == 0. {
        1.
    } else {
        let p = x * std::f64::consts::PI;
        p.sin() / p
    }
}
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i64>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let support = 3. * filter_scale;
    (0..output)
        .map(|i| {
            let centre = (i as f64 + 0.5) * scale;
            let first = ((centre - support + 0.5) as i64).max(0) as usize;
            let end = ((centre + support + 0.5) as usize).min(input);
            let weights = (first..end)
                .map(|j| {
                    let x = (j as f64 - centre + 0.5) / filter_scale;
                    if (-3. ..3.).contains(&x) {
                        sinc(x) * sinc(x / 3.)
                    } else {
                        0.
                    }
                })
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            let fixed = weights
                .into_iter()
                .map(|w| {
                    let w = w / sum * (1u64 << PRECISION) as f64;
                    (w + if w < 0. { -0.5 } else { 0.5 }) as i64
                })
                .collect();
            (first, fixed)
        })
        .collect()
}
fn filtered(
    source: &[u8],
    width: usize,
    height: usize,
    target: usize,
    horizontal: bool,
) -> Vec<u8> {
    let input = if horizontal { width } else { height };
    if input == target {
        return source.to_vec();
    }
    let table = coefficients(input, target);
    let (w, h) = if horizontal {
        (target, height)
    } else {
        (width, target)
    };
    let mut result = vec![0; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let (first, weights) = &table[if horizontal { x } else { y }];
            for channel in 0..3 {
                let mut sum = 1i64 << (PRECISION - 1);
                for (j, weight) in weights.iter().enumerate() {
                    let (sx, sy) = if horizontal {
                        (first + j, y)
                    } else {
                        (x, first + j)
                    };
                    sum += source[(sy * width + sx) * 3 + channel] as i64 * weight;
                }
                result[(y * w + x) * 3 + channel] = (sum >> PRECISION).clamp(0, 255) as u8;
            }
        }
    }
    result
}
pub(crate) fn resize(
    source: &[u8],
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
) -> Vec<u8> {
    let horizontal = filtered(
        source,
        width as usize,
        height as usize,
        target_width as usize,
        true,
    );
    filtered(
        &horizontal,
        target_width as usize,
        height as usize,
        target_height as usize,
        false,
    )
}
