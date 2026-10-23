//! Director's endpoint frame fitting, seam correction and stereo mix arithmetic.
pub type Stereo = [f32; 2];
fn sum_f32(values: &[f32]) -> f32 {
    if values.len() <= 128 {
        return values.iter().sum();
    }
    let (a, b) = values.split_at(values.len() / 2);
    sum_f32(a) + sum_f32(b)
}
fn energy_f32(values: &[Stereo]) -> f32 {
    if values.len() <= 128 {
        return values.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum();
    }
    let (a, b) = values.split_at(values.len() / 2);
    energy_f32(a) + energy_f32(b)
}
pub fn blend(left: &mut [f32], right: &[f32], index: u64, count: u64) {
    let alpha = super::linspace_f32(
        index,
        1. / (count + 1) as f64,
        count as f64 / (count + 1) as f64,
        count,
    );
    for (a, b) in left.iter_mut().zip(right) {
        *a = *a * (1. - alpha) + b * alpha;
    }
}
pub fn fit_audio(source: &[Stereo], count: usize) -> Vec<Stereo> {
    if source.is_empty() {
        return vec![[0.; 2]; count];
    }
    if source.len() == count {
        return source.to_vec();
    }
    (0..count)
        .map(|i| {
            let scale = source.len() as f32 / count as f32;
            let x = (i as f32 + 0.5).mul_add(scale, -0.5).max(0.);
            let a = (x.floor() as usize).min(source.len() - 1);
            let b = (a + 1).min(source.len() - 1);
            let w = x - a as f32;
            [
                source[a][0] * (1. - w) + source[b][0] * w,
                source[a][1] * (1. - w) + source[b][1] * w,
            ]
        })
        .collect()
}
pub fn gain_fades(samples: &mut [Stereo], gain: f32, fade_in: usize, fade_out: usize) {
    let n = samples.len();
    let fi = fade_in.min(n);
    let fo = fade_out.min(n);
    for (i, sample) in samples.iter_mut().enumerate() {
        sample[0] *= gain;
        sample[1] *= gain;
        if i < fi {
            let fade = super::linspace_f32(i as u64, 0., 1., fi as u64);
            sample[0] *= fade;
            sample[1] *= fade;
        }
        if i >= n - fo {
            let fade = super::linspace_f32((i - (n - fo)) as u64, 1., 0., fo as u64);
            sample[0] *= fade;
            sample[1] *= fade;
        }
    }
}
pub fn add(output: &mut [Stereo], source: &[Stereo], start: usize) {
    for (a, b) in output.iter_mut().skip(start).zip(source) {
        a[0] += b[0];
        a[1] += b[1];
    }
}
pub fn normalize(samples: &mut [Stereo]) {
    if samples.is_empty() {
        return;
    }
    let rms = (energy_f32(samples) / (samples.len() * 2) as f32).sqrt() as f64;
    let peak = samples.iter().flatten().map(|v| v.abs()).fold(0., f32::max) as f64;
    if rms <= 1e-8 || peak <= 1e-8 {
        return;
    }
    let gain = (10f64.powf(-18. / 20.) / rms)
        .min(10f64.powf(-1. / 20.) / peak)
        .min(2.) as f32;
    for s in samples.iter_mut().flatten() {
        *s *= gain;
    }
    let peak = samples.iter().flatten().map(|v| v.abs()).fold(0., f32::max) as f64;
    let limit = 10f64.powf(-1. / 20.);
    if peak > limit {
        let correction = (limit / peak) as f32;
        for sample in samples.iter_mut().flatten() {
            *sample *= correction;
        }
    }
}
pub fn combine(generated: &mut [Stereo], timeline: &[Stereo], options: &super::Options) {
    for (g, t) in generated.iter_mut().zip(timeline) {
        let t = t.map(|v| v * options.timeline_gain as f32);
        let duck = if options.duck_generated && t[0].abs().max(t[1].abs()) > 1e-4 {
            1. - options.ducking_strength as f32
        } else {
            1.
        };
        for c in 0..2 {
            g[c] = g[c] * options.generated_gain as f32 * duck + t[c];
        }
    }
    if options.normalize_mix {
        normalize(generated);
    }
}
pub fn luma(rgb: &[f32]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}
fn quantile(sorted: &[f32], q: f32) -> f32 {
    let x = (sorted.len() - 1) as f32 * q;
    let a = x.floor() as usize;
    let b = x.ceil() as usize;
    sorted[a] + (sorted[b] - sorted[a]) * (x - a as f32)
}
pub fn tone_match(left: &[Vec<f32>], right: &mut [Vec<f32>], fps: f64) {
    let n = 3.min(left.len()).min(right.len());
    if n == 0 {
        return;
    }
    let values = |frames: &[Vec<f32>]| {
        frames
            .iter()
            .flat_map(|f| f.chunks_exact(3).map(luma))
            .collect::<Vec<_>>()
    };
    let mut a = values(&left[left.len() - n..]);
    let mut b = values(&right[..n]);
    if a.is_empty() || b.is_empty() {
        return;
    }
    let mean_a = sum_f32(&a) / a.len() as f32;
    let mean_b = sum_f32(&b) / b.len() as f32;
    let delta = mean_a - mean_b;
    if delta.abs() > 0.08 {
        return;
    }
    a.sort_by(f32::total_cmp);
    b.sort_by(f32::total_cmp);
    let sa = quantile(&a, 0.9) - quantile(&a, 0.1);
    let sb = quantile(&b, 0.9) - quantile(&b, 0.1);
    let ratio = if sa >= 1. / 255. && sb >= 1. / 255. {
        (f64::from(sa) / f64::from(sb)).clamp(0.9, 1.1)
    } else {
        1.
    };
    // Director skips only when neither correction qualifies. If one qualifies,
    // both measured terms participate, including a sub-threshold companion term.
    if (ratio - 1.).abs() < 0.005 && delta.abs() < 1. / 1024. {
        return;
    }
    let count = right.len().min(fps.round_ties_even().max(1.) as usize);
    for (i, frame) in right.iter_mut().take(count).enumerate() {
        let fade = super::linspace_f32(i as u64, 1., 0., count as u64);
        for rgb in frame.chunks_exact_mut(3) {
            let change = (delta + (ratio - 1.) as f32 * (luma(rgb) - mean_b)) * fade;
            for v in rgb {
                *v = (*v + change).clamp(0., 1.);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_fit_preserves_torch_fp32_half_endpoint_ties() {
        assert_eq!(super::super::fit_index(5000, 4, 10001), 1);
        assert_eq!(super::super::fit_index(95, 73, 241), 29);
        assert_eq!(super::super::fit_index(57, 1001, 241), 237);
    }
    #[test]
    fn continuous_correction_keeps_subthreshold_companion_term() {
        for (left, expected) in [([0.1844, 0.6164], 0.1844), ([0.2296, 0.6304], 0.2296)] {
            let source = vec![vec![left[0], left[0], left[0], left[1], left[1], left[1]]; 3];
            let mut target = vec![vec![0.2, 0.2, 0.2, 0.6, 0.6, 0.6]; 3];
            tone_match(&source, &mut target, 3.);
            assert!((target[0][0] - expected).abs() < 1e-6);
        }
    }
    #[test]
    fn fitting_uses_pixel_centres_while_frame_fitting_uses_endpoints() {
        assert_eq!(
            fit_audio(&[[0., 0.], [1., 1.]], 4),
            vec![[0., 0.], [0.25, 0.25], [0.75, 0.75], [1., 1.]]
        );
        assert_eq!(
            (0..4)
                .map(|i| super::super::fit_index(i, 2, 4))
                .collect::<Vec<_>>(),
            vec![0, 0, 1, 1]
        );
    }
    #[test]
    fn fades_overlap_and_ducking_preserves_stereo() {
        let mut source = vec![[1., -1.]; 5];
        gain_fades(&mut source, 0.5, 3, 3);
        assert_eq!(
            source,
            vec![
                [0., 0.],
                [0.25, -0.25],
                [0.5, -0.5],
                [0.25, -0.25],
                [0., 0.]
            ]
        );
        let options = super::super::Options {
            normalize_mix: false,
            duck_generated: true,
            ..Default::default()
        };
        combine(&mut source, &[[0.2, 0.1]; 5], &options);
        assert_eq!(source[2], [0.45, -0.15]);
    }
    #[test]
    fn normalization_caps_gain_and_silence_is_unchanged() {
        let mut a = vec![[0.01, -0.01]; 4];
        normalize(&mut a);
        assert_eq!(a, vec![[0.02, -0.02]; 4]);
        let mut silence = vec![[0.; 2]; 4];
        normalize(&mut silence);
        assert_eq!(silence, vec![[0.; 2]; 4]);
    }
    #[test]
    fn continuous_tone_fades_and_skips_large_exposure_jump() {
        let left = vec![vec![0.5; 6]; 3];
        let mut right = vec![vec![0.45; 6]; 4];
        tone_match(&left, &mut right, 3.);
        assert!((right[0][0] - 0.5).abs() < 1e-6);
        assert_eq!(right[2][0], 0.45);
        assert_eq!(right[3][0], 0.45);
        let mut right = vec![vec![0.1; 6]; 4];
        let original = right.clone();
        tone_match(&left, &mut right, 3.);
        assert_eq!(right, original);
    }
}
