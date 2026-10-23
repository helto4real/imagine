//! Native Reference-to-AV preprocessing from Director and nodes_minimax_h3.
use super::protocol::{Shape, references as wire};
use crate::{models::qwen_image21::Pillow, native::Error};
use std::path::Path;
use tch::Tensor;
fn error(message: &str) -> Error {
    Error::Native(message.into())
}

pub fn geometry(
    width: u32,
    height: u32,
    quality: &str,
    shape: Shape,
) -> Result<(u32, u32, u32, u32), Error> {
    let budget = match quality {
        "Standard" => Some(1_000_000u64),
        "High" => Some(2_000_000),
        "Extra High" => Some(4_000_000),
        _ => None,
    };
    if width == 0 || height == 0 {
        return Err(error("Empty reference image"));
    }
    if let Some(budget) = budget {
        let longest = u64::from(width.max(height));
        let dimensions = |edge: u64| {
            let w = ((2 * u64::from(width) * edge + longest) / (2 * longest)).max(1) as u32;
            let h = ((2 * u64::from(height) * edge + longest) / (2 * longest)).max(1) as u32;
            (w, h, w.div_ceil(32) * 32, h.div_ceil(32) * 32)
        };
        let (mut low, mut high) = (1, longest);
        while low < high {
            let middle = (low + high).div_ceil(2);
            let (_, _, w, h) = dimensions(middle);
            if u64::from(w) * u64::from(h) <= budget {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(dimensions(low))
    } else {
        let scale = match quality {
            "match" => (f64::from(shape.width) * f64::from(shape.height)
                / (f64::from(width) * f64::from(height)))
            .sqrt()
            .min(1.),
            "max" => (2048. / f64::from(width.min(height))).min(1.),
            _ => return Err(error("Unsupported reference image preprocessing policy")),
        };
        let w = ((f64::from(width) * scale / 32.).round_ties_even() * 32.).max(32.) as u32;
        let h = ((f64::from(height) * scale / 32.).round_ties_even() * 32.).max(32.) as u32;
        Ok((w, h, w, h))
    }
}
pub fn images(input: &wire::Input, shape: Shape, pillow_path: &Path) -> Result<Vec<Tensor>, Error> {
    if input.images.is_empty() {
        return Ok(vec![]);
    }
    let pillow = unsafe { Pillow::open(pillow_path) }
        .map_err(|_| error("Reference image Pillow unavailable"))?;
    input
        .images
        .iter()
        .map(|image| {
            let (w, h, pw, ph) = geometry(image.width, image.height, &input.quality, shape)?;
            let rgb = if (w, h) == (image.width, image.height) {
                image.rgb.clone()
            } else {
                pillow
                    .resize_rgb(
                        &image.rgb,
                        image.width as i32,
                        image.height as i32,
                        3,
                        w as i32,
                        h as i32,
                    )
                    .map_err(|_| error("Reference image resize failed"))?
            };
            let values = rgb.iter().map(|v| f32::from(*v) / 255.).collect::<Vec<_>>();
            let tensor =
                Tensor::f_from_slice(&values)?.f_view([1, i64::from(h), i64::from(w), 3])?;
            if w == pw && h == ph {
                Ok(tensor)
            } else {
                Ok(tensor
                    .f_permute([0, 3, 1, 2])?
                    .f_pad(
                        [0, i64::from(pw - w), 0, i64::from(ph - h)],
                        "replicate",
                        None,
                    )?
                    .f_permute([0, 2, 3, 1])?)
            }
        })
        .collect()
}
/// Comfy audio.resample's default sinc_interp_hann, on the waveform's device.
pub fn resample(waveform: &Tensor, source: i64, target: i64) -> Result<Tensor, Error> {
    if source == target {
        return Ok(waveform.shallow_clone());
    }
    if source <= 0 || target <= 0 {
        return Err(error("Invalid reference audio sample rate"));
    }
    let gcd = |mut a: i64, mut b: i64| {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    };
    let divisor = gcd(source, target);
    let (source, target) = (source / divisor, target / divisor);
    let base = source.min(target) as f64 * 0.99;
    let width = (6. * source as f64 / base).ceil() as i64;
    let options = (waveform.kind(), waveform.device());
    let idx = Tensor::f_arange_start(-width, width + source, options)?
        .f_view([1, 1, -1])?
        .f_div_scalar(source)?;
    let t = Tensor::f_arange_start_step(0, -target, -1, options)?
        .f_view([-1, 1, 1])?
        .f_div_scalar(target)?
        .f_add(&idx)?
        .f_mul_scalar(base)?
        .f_clamp(-6., 6.)?;
    let window = t
        .f_mul_scalar(std::f64::consts::PI)?
        .f_div_scalar(6.)?
        .f_div_scalar(2.)?
        .f_cos()?
        .f_pow_tensor_scalar(2)?;
    let t = t.f_mul_scalar(std::f64::consts::PI)?;
    let kernel = Tensor::f_ones_like(&t)?
        .f_where_self(&t.f_eq(0.)?, &t.f_sin()?.f_div(&t)?)?
        .f_mul(&window.f_mul_scalar(base / source as f64)?)?;
    let shape = waveform.size();
    let length = *shape.last().ok_or_else(|| error("Reference audio shape"))?;
    let rows = waveform.numel() as i64 / length;
    let value = waveform
        .f_reshape([-1, length])?
        .f_pad([width, width + source], "constant", None)?
        .f_unsqueeze(1)?
        .f_conv1d(&kernel, None::<&Tensor>, [source], [0], [1], 1)?
        .f_transpose(1, 2)?
        .f_reshape([rows, -1])?;
    let target_length = (((target * length) as f64 / source as f64) as f32).ceil() as i64;
    let mut output_shape = shape;
    *output_shape.last_mut().unwrap() = target_length;
    Ok(value
        .f_narrow(1, 0, target_length)?
        .f_reshape(&output_shape)?)
}
pub fn audio(input: &wire::Audio) -> Result<Tensor, Error> {
    let values = input
        .pcm
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let tensor = Tensor::f_from_slice(&values)?
        .f_view([i64::from(input.samples), 2])?
        .f_transpose(0, 1)?
        .f_unsqueeze(0)?;
    resample(&tensor, i64::from(input.sample_rate), 32_000)
}
