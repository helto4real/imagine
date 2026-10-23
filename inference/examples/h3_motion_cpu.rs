//! Pure CPU oracle output: no Runtime, model weights or CUDA initialization.
use imagine_inference::{
    context_windows,
    execution_value::Value,
    h3_product::{
        motion,
        protocol::{self as p, continuation as c},
    },
    models::minimax_h3::{sampling as h3_sampling, streams::NestedTensor},
    native::Error,
    sampling::model_sampling::{Config, Flavor, ModelSampling},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tch::{Device, Kind, Tensor};
fn values(t: &Tensor) -> Vec<f32> {
    Vec::<f32>::try_from(&t.f_contiguous().unwrap().f_view([-1]).unwrap()).unwrap()
}
fn main() -> Result<(), Error> {
    tch::set_num_threads(1);
    let shape = p::Shape {
        width: 32,
        height: 32,
        frames: 39,
        fps: 24,
    };
    let video = Tensor::f_arange(24 * shape.video_latent_t() * 4, (Kind::Float, Device::Cpu))?
        .f_view([1, 24, shape.video_latent_t(), 2, 2])?
        .f_div_scalar(100.)?;
    let audio = Tensor::f_arange(64 * shape.audio_latent_t(), (Kind::Float, Device::Cpu))?
        .f_view([1, 32, 2, shape.audio_latent_t()])?
        .f_div_scalar(1000.)?;
    let artifact = motion::capture(&video, &audio, shape, shape, 0)?;
    let schedule = ModelSampling {
        flavor: Flavor::AudioVideo,
        config: Config {
            shift: 12.,
            audio_shift: Some(3.),
            ..Config::default()
        },
        sigmas: Tensor::new(),
        log_sigmas: None,
        num_timesteps: 0,
        cosine_s: None,
        initial_alpha: None,
        skip_steps: 1,
    };
    let mut cases = vec![];
    let mut audio_trims = vec![];
    for method in [c::Method::MotionContext, c::Method::MotionContextExp] {
        let visible = p::Shape {
            frames: 124,
            ..shape
        };
        let (sampled, head, _) = c::runtime(visible, method).unwrap();
        let input = (0..sampled.native_audio_samples() * 2)
            .map(|i| (i as f32 - 200_000.) / 1_048_576.)
            .collect();
        let trimmed = c::trim_audio(sampled, visible, head, input).unwrap();
        let bytes: Vec<u8> = trimmed.iter().flat_map(|v| v.to_le_bytes()).collect();
        audio_trims.push(json!({"method":method,"sampled_frames":sampled.frames,"visible_frames":visible.frames,"head_frames":head,"native_samples":sampled.native_audio_samples(),"output_samples":trimmed.len()/2,"sha256":format!("{:x}",Sha256::digest(&bytes))}));
        for include_audio in [false, true] {
            let input = c::Input {
                method,
                include_audio,
                descriptor: artifact.descriptor.clone(),
                sha256: "a".repeat(64),
            };
            let context = motion::decode(&input, &artifact.bytes)?;
            let (sampled, _, _) = c::runtime(shape, method).unwrap();
            let v = Tensor::f_zeros(
                [1, 24, sampled.video_latent_t(), 2, 2],
                (Kind::Float, Device::Cpu),
            )?;
            let a = Tensor::f_zeros(
                [1, 32, 2, sampled.audio_latent_t()],
                (Kind::Float, Device::Cpu),
            )?;
            let mask = if method == c::Method::MotionContextExp {
                Some(motion::prefix(&v, &a, &context, include_audio)?)
            } else {
                None
            };
            let mask_probe = if let Some(mask) = &mask {
                let (latent, shapes) = NestedTensor::new(vec![
                    Value::tensor(v.shallow_clone()),
                    Value::tensor(a.f_mul_scalar(4.)?),
                ])
                .pack()?;
                let latent = latent.as_tensor()?;
                let range = Tensor::f_arange(latent.numel() as i64, (Kind::Float, Device::Cpu))?
                    .f_view(latent.size().as_slice())?
                    .f_div_scalar(1000.)?;
                let noise = range.f_sin()?;
                let x = range.f_cos()?;
                let masks = h3_sampling::denoise_masks(mask, &shapes)?;
                let mut injected = vec![];
                for sigma in [1f32, 0.5, 0.01, 0.] {
                    let output = h3_sampling::scale_latent_inpaint(
                        &schedule,
                        &shapes,
                        &Tensor::f_from_slice(&[sigma])?,
                        &noise,
                        &latent,
                        Some(&x),
                        Some(mask),
                    )?;
                    let streams = context_windows::unpack_latents(&Value::tensor(output), &shapes)?;
                    injected.push(json!({"sigma":sigma,"video":values(&streams[0].as_tensor()?),"audio":values(&streams[1].as_tensor()?)}));
                }
                Some(
                    json!({"denoise_video":masks[0].as_ref().map(values),"denoise_audio":masks[1].as_ref().map(values),"injected":injected}),
                )
            } else {
                None
            };
            let n =
                3 + if method == c::Method::MotionContext {
                    7 + if include_audio { 74 } else { 0 }
                } else {
                    0
                } + sampled.audio_latent_t() * 2
                    + sampled.video_latent_t();
            let q = Tensor::f_arange(n * 2, (Kind::Float, Device::Cpu))?
                .f_view([1, 1, n, 2])?
                .f_div_scalar(1000.)?;
            let k = q.f_flip([2])?;
            let val = q.f_sin()?;
            let out = motion::Window::new(&input, sampled, 3).run(&q, &k, &val, |q, k, v| {
                Ok(q.f_matmul(&k.f_transpose(-2, -1)?)?
                    .f_div_scalar(2f64.sqrt())?
                    .f_softmax(-1, Kind::Float)?
                    .f_matmul(v)?
                    .f_transpose(1, 2)?
                    .f_contiguous()?
                    .f_view([1, -1, 2])?)
            })?;
            cases.push(json!({"method":method,"include_audio":include_audio,"n":n,"output":values(&out),"prefix_video":values(&v),"prefix_audio":values(&a),"mask":mask.as_ref().map(values),"mask_probe":mask_probe}));
        }
    }
    println!(
        "{}",
        json!({"descriptor":artifact.descriptor,"context":artifact.bytes.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>(),"cases":cases,"audio_trims":audio_trims})
    );
    Ok(())
}
