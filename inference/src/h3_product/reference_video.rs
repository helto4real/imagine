//! Director temporal selection and MiniMax reference-video canvas/presentation.
use super::protocol::{Shape, references::Video};
use crate::{models::qwen_image21::Pillow, native::Error};
use std::path::Path;
use tch::{Device, Kind, Tensor};
fn error(text: &str) -> Error {
    Error::Native(text.into())
}
pub fn canvas(width: u32, height: u32) -> (u32, u32) {
    let ratio = width as f64 / height as f64;
    let (mut w, mut h) = if ratio >= 1. {
        (768. * ratio, 768.)
    } else {
        (768., 768. / ratio)
    };
    if w * h > 768. * 1344. {
        let scale = (768. * 1344. / (w * h)).sqrt();
        w *= scale;
        h *= scale;
    }
    let mut cw = ((w / 32.).round_ties_even() * 32.).max(32.) as u32;
    let mut ch = ((h / 32.).round_ties_even() * 32.).max(32.) as u32;
    if u64::from(width) * u64::from(height) < u64::from(cw) * u64::from(ch) {
        cw = ((width as f64 / 32.).round_ties_even() * 32.).max(32.) as u32;
        ch = ((height as f64 / 32.).round_ties_even() * 32.).max(32.) as u32;
    }
    (cw, ch)
}
pub fn prepare(video: &Video, shape: Shape, pillow_path: &Path) -> Result<Tensor, Error> {
    let mut frames = Tensor::f_from_slice(&video.rgb)?
        .f_view([
            i64::from(video.frames),
            i64::from(video.height),
            i64::from(video.width),
            3,
        ])?
        .f_to_kind(Kind::Float)?
        .f_div_scalar(255.)?;
    let source_fps = video.source_fps.max(0.001);
    if video.frames > 1 && (source_fps - 24.).abs() > 1e-6 {
        let count = ((video.frames as f64 / source_fps) * 24.)
            .round_ties_even()
            .max(1.) as i64;
        let indices = Tensor::f_linspace(
            0.,
            f64::from(video.frames - 1),
            count,
            (Kind::Float, Device::Cpu),
        )?
        .f_round()?
        .f_to_kind(Kind::Int64)?;
        frames = frames.f_index_select(0, &indices)?;
    }
    if let Some(tail) = video.tail_frames {
        let mut count = frames.size()[0]
            .min(i64::from(shape.frames))
            .min(i64::from(tail));
        if count < 5 {
            return Err(error(
                "MiniMax H3 continuation needs at least 5 tail frames",
            ));
        }
        count -= (count - 5) % 17;
        frames = frames.f_narrow(0, frames.size()[0] - count, count)?;
    }
    let count = frames.size()[0];
    let (width, height) = canvas(video.width, video.height);
    let pillow = unsafe { Pillow::open(pillow_path) }
        .map_err(|_| error("Reference video Pillow unavailable"))?;
    let mut output = Vec::with_capacity(count as usize * width as usize * height as usize * 3);
    for i in 0..count {
        let frame = frames.f_select(0, i)?.f_contiguous()?;
        let mut values = vec![0f32; frame.numel()];
        let length = values.len();
        frame.f_copy_data(&mut values, length)?;
        let rgb = values
            .into_iter()
            .map(|v| (v * 255.).clamp(0., 255.) as u8)
            .collect::<Vec<_>>();
        let resized = pillow
            .resize_rgb(
                &rgb,
                video.width as i32,
                video.height as i32,
                3,
                width as i32,
                height as i32,
            )
            .map_err(|_| error("Reference video resize failed"))?;
        output.extend(resized.into_iter().map(|v| v as f32 / 255.));
    }
    let mut selected = count.min(i64::from(shape.frames));
    if selected < 5 {
        return Err(error("MiniMax H3 reference videos need at least 5 frames"));
    }
    selected -= (selected - 5) % 17;
    // Comfy's Lanczos returns contiguous NCHW, then _resize moves channels last.
    Ok(Tensor::f_from_slice(&output)?
        .f_view([count, i64::from(height), i64::from(width), 3])?
        .f_permute([0, 3, 1, 2])?
        .f_contiguous()?
        .f_permute([0, 2, 3, 1])?
        .f_narrow(0, 0, selected)?)
}
pub fn qwen(frames: &Tensor) -> Result<Tensor, Error> {
    Ok(frames.f_index_select(
        0,
        &Tensor::f_arange_start_step(0, frames.size()[0], 12, (Kind::Int64, Device::Cpu))?,
    )?)
}
pub fn vae_input(frames: &Tensor) -> Result<Tensor, Error> {
    super::image::process_input(&frames.f_permute([3, 0, 1, 2])?.f_unsqueeze(0)?, Kind::Half)
}
