//! Original MiniMax video-block preprocessing and typed AV latent operations.
use crate::{
    execution_value::Value,
    native::{self, Error, Kitchen},
};
use tch::{Kind, Tensor};

fn gap(operation: &str) -> Error {
    Error::BindingGap {
        backend: "minimax_h3".into(),
        operation: operation.into(),
    }
}

/// Typed image spans after embedding expansion, in original request order.
/// Non-image entries must be filtered before reading their index or size.
pub fn token_tags(seq_len: i64, image_spans: &[(i64, i64)]) -> Result<Tensor, Error> {
    let tags = Tensor::f_ones([seq_len], (Kind::Int64, tch::Device::Cpu))?;
    for &(index, size) in image_spans {
        let start = (i128::from(index) - 1).max(0).min(i128::from(i64::MAX)) as i64;
        let end = (i128::from(index) + i128::from(size) + 1)
            .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
        let _ = native::context_slice(&tags, 0, start, end)?.f_fill_(0)?;
    }
    Ok(tags)
}

/// Defaults of comfy.text_encoders.minimax.process_video_block; two frames
/// fill the temporal patch. No repeated first-image shortcut is used.
pub fn video_block(frames: &Tensor, native: &Kitchen) -> Result<(Tensor, Tensor), Error> {
    let shape = frames.size();
    if shape.len() != 4 {
        return Err(gap("video block requires typed THWC input"));
    }
    let (height, width) = (shape[1], shape[2]);
    let factor = 32;
    let mut h = (height as f64 / factor as f64).round_ties_even() as i64 * factor;
    let mut w = (width as f64 / factor as f64).round_ties_even() as i64 * factor;
    if h * w > 12_845_056 {
        let beta = ((height * width) as f64 / 12_845_056.0).sqrt();
        h = factor.max((height as f64 / beta / factor as f64).floor() as i64 * factor);
        w = factor.max((width as f64 / beta / factor as f64).floor() as i64 * factor);
    } else if h * w < 3136 {
        let beta = (3136.0 / (height * width) as f64).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as i64 * factor;
        w = (width as f64 * beta / factor as f64).ceil() as i64 * factor;
    }
    let images =
        frames
            .f_permute([0, 3, 1, 2])?
            .f_upsample_bilinear2d([h, w], false, None, None)?;
    let mean = Tensor::f_from_slice(&[0.5_f64; 3])?
        .f_to_device(images.device())?
        .f_to_kind(native.default_dtype()?)?
        .f_view([1, 3, 1, 1])?;
    let std = Tensor::f_from_slice(&[0.5_f64; 3])?
        .f_to_device(images.device())?
        .f_to_kind(native.default_dtype()?)?
        .f_view([1, 3, 1, 1])?;
    let images = images.f_sub(&mean)?.f_div(&std)?;
    let (gh, gw) = (h / 16, w / 16);
    let flat = images
        .f_reshape([1, 2, 3, gh / 2, 2, 16, gw / 2, 2, 16])?
        .f_permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?
        .f_reshape([gh * gw, 1536])?;
    let grid = Tensor::f_from_slice(&[1_i64, gh, gw])?
        .f_to_device(frames.device())?
        .f_unsqueeze(0)?;
    Ok((flat, grid))
}

/// Typed stream payload of MiniMaxH3AV.fix_empty_latent. The surrounding
/// NestedTensor object's mutable-list semantics are not represented by this array.
pub fn empty_streams(latent: &Tensor) -> Result<[Value; 2], Error> {
    let video = native::clone_tensor(&native::context_slice(latent, 1, 0, 24)?)?;
    let shape = video.size();
    let frames = *shape
        .get(2)
        .ok_or_else(|| gap("empty AV latent requires temporal axis"))?;
    let count: i64 = (0..frames).map(|i| if i % 5 == 0 { 1 } else { 4 }).sum();
    let audio_t = (count as f64 * (5.0 / 3.0)).round_ties_even() as i64;
    let audio = latent.f_new_zeros(
        [latent.size()[0], 32, 2, audio_t],
        (latent.kind(), latent.device()),
    )?;
    Ok([Value::tensor(video), Value::tensor(audio)])
}

/// Packed branch of MiniMaxH3._scale_audio_slice. `video_shape` is the
/// first original stream shape. Scale=1 returns the original tensor object.
pub fn scale_packed_audio(
    latent: &Value,
    video_shape: &[i64],
    scale: f64,
    native: &Kitchen,
) -> Result<Value, Error> {
    if scale == 1.0 {
        return Ok(latent.clone());
    }
    let cut = video_shape
        .iter()
        .skip(1)
        .try_fold(1_i64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| gap("packed video shape product exceeds native index"))?;
    let value = native::clone_tensor(&latent.as_tensor()?)?;
    let slice = Value::tensor(native::context_slice(
        &value,
        value.dim() as i64 - 1,
        cut,
        i64::MAX,
    )?);
    // Generated tch scalar temporaries panic in Drop before collecting dtype
    // errors. The existing boxed boundary calls that same mul_.Scalar safely.
    unsafe {
        crate::boxed_aten::call(
            native,
            "aten::mul_",
            "Scalar",
            &[slice, Value::Float(scale)],
            &vec![],
        )?;
    }
    Ok(Value::tensor(value))
}
