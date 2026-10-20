//! H3's latent-format operations around the existing shared flow sampler.
//! No learned denoiser is embedded here. Its typed packed and nested boundaries
//! retain the original multiplication/division by scale_factor=1.0 as well as
//! the separate audio-scale clone; those operations are not optimized away.
use super::{preprocessing, streams::NestedTensor};
use crate::{
    execution_value::Value,
    native::{Error, Kitchen},
};
pub fn process_packed_in(
    latent: &Value,
    video_shape: &[i64],
    audio_scale: f64,
    native: &Kitchen,
) -> Result<Value, Error> {
    let value = Value::tensor(latent.as_tensor()?.f_mul_scalar(1.0)?);
    preprocessing::scale_packed_audio(&value, video_shape, audio_scale, native)
}
pub fn process_packed_out(
    latent: &Value,
    video_shape: &[i64],
    audio_scale: f64,
    native: &Kitchen,
) -> Result<Value, Error> {
    let value = preprocessing::scale_packed_audio(
        latent,
        video_shape,
        inverse_audio_scale(audio_scale)?,
        native,
    )?;
    Ok(Value::tensor(value.as_tensor()?.f_div_scalar(1.0)?))
}
pub fn process_streams_in(latent: &NestedTensor, audio_scale: f64) -> Result<NestedTensor, Error> {
    let mut values = vec![];
    for value in latent.unbind().borrow().iter() {
        values.push(Value::tensor(value.as_tensor()?.f_mul_scalar(1.0)?));
    }
    NestedTensor::new(values).scale_audio(audio_scale)
}
pub fn process_streams_out(latent: &NestedTensor, audio_scale: f64) -> Result<NestedTensor, Error> {
    let scaled = latent.scale_audio(inverse_audio_scale(audio_scale)?)?;
    let mut values = vec![];
    for value in scaled.unbind().borrow().iter() {
        values.push(Value::tensor(value.as_tensor()?.f_div_scalar(1.0)?));
    }
    Ok(NestedTensor::new(values))
}

fn inverse_audio_scale(scale: f64) -> Result<f64, Error> {
    if scale == 0.0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "float division by zero".into(),
        });
    }
    Ok(1.0 / scale)
}
