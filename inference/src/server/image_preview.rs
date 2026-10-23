//! Disposable RGB projection of the sampler's borrowed denoised image latent.
//! Cadence belongs to the caller. No VAE loading, RNG use or input mutation.
use crate::native::Error;
use tch::{Device, Kind, Tensor};

#[path = "../../../shared/image_preview_rgb.rs"]
mod factors;

struct Projection {
    weights: &'static [[f32; 3]],
    bias: [f32; 3],
    pack: i64,
}
fn projection(model: &str) -> Result<Projection, Error> {
    match model {
        "moss-image-sft-int8" => Ok(Projection {
            weights: &factors::FLUX2_RGB,
            bias: factors::FLUX2_BIAS,
            pack: 2,
        }),
        "qwen-image-2.1-int8" | "qwen-image-2.1-bf16" | "qwen-image-2.1-turbo" => Ok(Projection {
            weights: &factors::QWEN_RGB,
            bias: factors::QWEN_BIAS,
            pack: 1,
        }),
        _ => Err(Error::Native(
            "No image preview projection for this model".into(),
        )),
    }
}

fn projected_rgb(denoised: &Tensor, projection: &Projection) -> Result<Tensor, Error> {
    let shape = denoised.size();
    let channels = projection.weights.len() as i64;
    let pack = projection.pack;
    if shape.len() != 4
        || shape[0] != 1
        || shape[1] != channels * pack * pack
        || shape[2] <= 0
        || shape[3] <= 0
        || shape[2] > i64::from(u32::MAX) / pack
        || shape[3] > i64::from(u32::MAX) / pack
    {
        return Err(Error::Native("Invalid image preview latent layout".into()));
    }
    let (height, width) = (shape[2] * pack, shape[3] * pack);
    let input = denoised.f_get(0)?.f_to_kind(Kind::Float)?;
    let input = if pack == 2 {
        // Native channel = d*4 + (output_y%2)*2 + output_x%2.
        input
            .f_reshape([channels, 2, 2, shape[2], shape[3]])?
            .f_permute([0, 3, 1, 4, 2])?
            .f_reshape([channels, height, width])?
    } else {
        input
    };
    let coefficients: Vec<f32> = projection.weights.iter().flatten().copied().collect();
    let coefficients = Tensor::f_from_slice(&coefficients)?
        .f_view([channels, 3, 1, 1])?
        .f_to_device(denoised.device())?;
    let mut rgb = Tensor::f_zeros([3, height, width], (Kind::Float, denoised.device()))?;
    // Preserve native's ascending-channel FP32 multiply/add order. A GEMM may
    // use a different accumulation or TF32 policy and move a byte threshold.
    for channel in 0..channels {
        let contribution = input
            .f_get(channel)?
            .f_unsqueeze(0)?
            .f_mul(&coefficients.f_get(channel)?)?;
        rgb = rgb.f_add(&contribution)?;
    }
    let bias = Tensor::f_from_slice(&projection.bias)?
        .f_view([3, 1, 1])?
        .f_to_device(denoised.device())?;
    Ok(rgb
        .f_add(&bias)?
        .f_mul_scalar(0.5)?
        .f_add_scalar(0.5)?
        .f_clamp(0., 1.)?
        .f_mul_scalar(255.)?
        .f_to_kind(Kind::Uint8)?
        .f_permute([1, 2, 0])?)
}

fn dimensions(width: u32, height: u32, maximum: u32) -> (u32, u32) {
    let maximum = maximum.max(1);
    if width <= maximum && height <= maximum {
        return (width, height);
    }
    let ratio = (f64::from(maximum) / f64::from(width)).min(f64::from(maximum) / f64::from(height));
    (
        (f64::from(width) * ratio).round().max(1.) as u32,
        (f64::from(height) * ratio).round().max(1.) as u32,
    )
}

// image 0.25's native-preview Triangle resize: vertical FP32 pass, horizontal
// FP32 pass, then nearest integer. Pixels stay on the input device throughout.
fn triangle_weights(source: i64, target: i64) -> Vec<Vec<(i64, f32)>> {
    let ratio = source as f32 / target as f32;
    let support = ratio.max(1.);
    (0..target)
        .map(|position| {
            let center = (position as f32 + 0.5) * ratio;
            let left = ((center - support).floor() as i64).clamp(0, source - 1);
            let right = ((center + support).ceil() as i64).clamp(left + 1, source);
            let center = center - 0.5;
            let mut weights: Vec<_> = (left..right)
                .map(|index| {
                    let distance = ((index as f32 - center) / support).abs();
                    (index, if distance < 1. { 1. - distance } else { 0. })
                })
                .collect();
            let sum: f32 = weights.iter().map(|(_, weight)| weight).sum();
            for (_, weight) in &mut weights {
                *weight /= sum;
            }
            weights
        })
        .collect()
}
fn resize_axis(input: &Tensor, axis: usize, target: i64) -> Result<Tensor, Error> {
    let source = input.size()[axis];
    let weights = triangle_weights(source, target);
    let taps = weights.iter().map(Vec::len).max().unwrap();
    let mut shape = input.size();
    shape[axis] = target;
    let mut output = Tensor::f_zeros(&shape, (Kind::Float, input.device()))?;
    let mut weight_shape = [1_i64; 3];
    weight_shape[axis] = target;
    for tap in 0..taps {
        let entries: Vec<_> = weights
            .iter()
            .map(|row| row.get(tap).copied().unwrap_or((0, 0.)))
            .collect();
        let indices =
            Tensor::f_from_slice(&entries.iter().map(|(index, _)| *index).collect::<Vec<_>>())?
                .f_to_device(input.device())?;
        let values =
            Tensor::f_from_slice(&entries.iter().map(|(_, value)| *value).collect::<Vec<_>>())?
                .f_to_device(input.device())?
                .f_view(weight_shape)?;
        output = output.f_add(
            &input
                .f_index_select(axis as i64, &indices)?
                .f_mul(&values)?,
        )?;
    }
    Ok(output)
}

pub(super) fn render(
    model: &str,
    denoised: &Tensor,
    max_resolution: u32,
) -> Result<(u32, u32, Vec<u8>), Error> {
    let _grad = tch::no_grad_guard();
    let rgb = projected_rgb(denoised, &projection(model)?)?;
    let (width, height) = (rgb.size()[1] as u32, rgb.size()[0] as u32);
    let (target_width, target_height) = dimensions(width, height, max_resolution);
    let rgb = if (width, height) == (target_width, target_height) {
        rgb
    } else {
        let vertical = resize_axis(&rgb.f_to_kind(Kind::Float)?, 0, i64::from(target_height))?;
        resize_axis(&vertical, 1, i64::from(target_width))?
            .f_clamp(0., 255.)?
            .f_add_scalar(0.5)?
            .f_floor()?
            .f_to_kind(Kind::Uint8)?
    };
    let cpu = rgb.f_contiguous()?.f_to_device(Device::Cpu)?;
    let mut bytes = vec![0_u8; cpu.numel()];
    let length = bytes.len();
    cpu.f_copy_data_u8(&mut bytes, length)?;
    Ok((target_width, target_height, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent scalar form of native latent-preview.h's image loop. It
    // indexes packed source channels directly; it never uses Tensor reshaping.
    fn native_scalar(values: &[f32], h: usize, w: usize, p: &Projection) -> Vec<u8> {
        let pack = p.pack as usize;
        let (height, width) = (h * pack, w * pack);
        let mut bytes = Vec::with_capacity(height * width * 3);
        for y in 0..height {
            for x in 0..width {
                let offset = (y % pack) * pack + x % pack;
                let mut rgb = [0_f32; 3];
                for (channel, weights) in p.weights.iter().enumerate() {
                    let channel = channel * pack * pack + offset;
                    let value = values[(channel * h + y / pack) * w + x / pack];
                    for (channel, weight) in rgb.iter_mut().zip(weights) {
                        *channel += value * weight;
                    }
                }
                for (value, bias) in rgb.into_iter().zip(p.bias) {
                    bytes.push((((value + bias) * 0.5 + 0.5).clamp(0., 1.) * 255.) as u8);
                }
            }
        }
        bytes
    }
    fn values(channels: usize, h: usize, w: usize) -> Vec<f32> {
        (0..channels * h * w)
            .map(|i| ((i * 37 % 101) as f32 - 50.) / 4.)
            .collect()
    }
    #[test]
    fn qwen_and_moss_projection_match_native_scalar_pack_bytecast_and_do_not_mutate() {
        for model in ["qwen-image-2.1-int8", "moss-image-sft-int8"] {
            let p = projection(model).unwrap();
            let (h, w) = (3, 5);
            let channels = p.weights.len() * p.pack as usize * p.pack as usize;
            let data = values(channels, h, w);
            for dtype in [Kind::Float, Kind::BFloat16] {
                let input = Tensor::from_slice(&data)
                    .view([1, channels as i64, h as i64, w as i64])
                    .to_kind(dtype);
                let before = input.copy();
                let (width, height, actual) = render(model, &input, 100).unwrap();
                assert_eq!(
                    (width, height),
                    (w as u32 * p.pack as u32, h as u32 * p.pack as u32)
                );
                assert_eq!(actual, native_scalar(&data, h, w, &p));
                assert!(input.equal(&before));
                assert_eq!(input.kind(), dtype);
            }
        }
    }
    #[test]
    fn source_bias_and_signed_channel_extremes_use_native_clamp_and_truncation() {
        for model in ["qwen-image-2.1-bf16", "moss-image-sft-int8"] {
            let p = projection(model).unwrap();
            let channels = p.weights.len() * p.pack as usize * p.pack as usize;
            let (width, height, zero) = render(
                model,
                &Tensor::zeros([1, channels as i64, 1, 1], (Kind::Float, Device::Cpu)),
                100,
            )
            .unwrap();
            let expected: Vec<_> = p
                .bias
                .iter()
                .map(|b| ((b * 0.5 + 0.5) * 255.) as u8)
                .collect();
            assert_eq!(zero, expected.repeat((width * height) as usize));
            for sign in [-1_f32, 1.] {
                let mut data = vec![0.; channels];
                for entry in data.iter_mut().take((p.pack * p.pack) as usize) {
                    *entry = sign * 1_000_000.;
                }
                let input = Tensor::from_slice(&data).view([1, channels as i64, 1, 1]);
                let (_, _, actual) = render(model, &input, 100).unwrap();
                assert_eq!(actual, native_scalar(&data, 1, 1, &p));
                assert!(actual.iter().all(|byte| *byte == 0 || *byte == 255));
            }
        }
    }
    #[test]
    fn bounded_resolution_preserves_aspect_does_not_upscale_and_reduces_before_cpu_copy() {
        for model in ["qwen-image-2.1-turbo", "moss-image-sft-int8"] {
            let p = projection(model).unwrap();
            let channels = p.weights.len() as i64 * p.pack * p.pack;
            let input = Tensor::zeros([1, channels, 3, 9], (Kind::Float, Device::Cpu));
            for edge in [0, 1, 4, 7, 100] {
                let (width, height, bytes) = render(model, &input, edge).unwrap();
                assert_eq!(
                    (width, height),
                    dimensions(9 * p.pack as u32, 3 * p.pack as u32, edge)
                );
                assert!(width.max(height) <= edge.max(1));
                assert_eq!(bytes.len(), width as usize * height as usize * 3);
                let expected: Vec<_> = p
                    .bias
                    .iter()
                    .map(|b| ((b * 0.5 + 0.5) * 255.) as u8)
                    .collect();
                assert_eq!(bytes, expected.repeat((width * height) as usize));
            }
        }
    }
    #[test]
    fn triangle_resize_matches_native_adapters_two_pass_antialias_and_half_up_cast() {
        let mut data = vec![];
        for row in 0..2 {
            for column in 0..4 {
                let value = (row * 40 + column * 40) as f32;
                data.extend([value, 255. - value, 128.]);
            }
        }
        let input = Tensor::from_slice(&data).view([2, 4, 3]);
        let vertical = resize_axis(&input, 0, 1).unwrap();
        let output = resize_axis(&vertical, 1, 2)
            .unwrap()
            .clamp(0., 255.)
            .f_add_scalar(0.5)
            .unwrap()
            .floor()
            .to_kind(Kind::Uint8)
            .contiguous();
        let mut bytes = vec![0_u8; output.numel()];
        let len = bytes.len();
        output.copy_data_u8(&mut bytes, len);
        // Vertical weights 1/2,1/2. Horizontal 3/7,3/7,1/7 and 1/7,3/7,3/7.
        assert_eq!(bytes, [49, 206, 128, 111, 144, 128]);
    }
    #[test]
    fn wrong_latent_layout_or_unknown_projection_is_a_preview_error() {
        let input = Tensor::zeros([1, 64, 2, 2], (Kind::Float, Device::Cpu));
        assert!(render("unknown", &input, 512).is_err());
        assert!(render("moss-image-sft-int8", &input, 512).is_err());
        assert!(render("qwen-image-2.1-int8", &input.squeeze(), 512).is_err());
    }
    #[test]
    #[ignore = "requires own CPU Python-renderer oracle, no ComfyUI/model/GPU"]
    fn qwen_worker_renderer_matches_python_rgb_and_noninteger_triangle_resize() {
        let path = std::env::var("IMAGINE_IMAGE_PREVIEW_TEST_ORACLE").unwrap();
        let oracle: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for case in oracle.as_array().unwrap() {
            let (h, w) = (
                case["height"].as_i64().unwrap(),
                case["width"].as_i64().unwrap(),
            );
            let input =
                (Tensor::arange(64 * h * w, (Kind::Float, Device::Cpu)).remainder(37) - 18.) / 7.;
            let input = input.view([1, 64, h, w]).to_kind(if case["bf16"] == true {
                Kind::BFloat16
            } else {
                Kind::Float
            });
            let (width, height, bytes) = render(
                "qwen-image-2.1-int8",
                &input,
                case["edge"].as_u64().unwrap() as u32,
            )
            .unwrap();
            assert_eq!(
                (width, height),
                (
                    case["output_width"].as_u64().unwrap() as u32,
                    case["output_height"].as_u64().unwrap() as u32
                )
            );
            assert_eq!(
                bytes,
                serde_json::from_value::<Vec<u8>>(case["pixels"].clone()).unwrap()
            );
        }
    }
}
