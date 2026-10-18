//! Exact ATen operation order from Kitchen's eager quantization paths.
use crate::native::{Error, Kitchen};
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use tch::{Kind, Tensor};
type LookupCache = Mutex<HashMap<(tch::Device, Option<Kind>), Tensor>>;
static E2M1_LUT_CACHE: LazyLock<LookupCache> = LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn dequantize_nvfp4(
    payload: &Tensor,
    global_scale: &Tensor,
    block_scales: &Tensor,
    output_kind: Kind,
    hi_first: bool,
) -> Result<Tensor, tch::TchError> {
    dequantize_nvfp4_optional(
        payload,
        global_scale,
        block_scales,
        Some(output_kind),
        || Ok(hi_first),
        Tensor::f_mul,
    )
}
pub(crate) fn dequantize_nvfp4_optional<E: From<tch::TchError>>(
    payload: &Tensor,
    global_scale: &Tensor,
    block_scales: &Tensor,
    output_kind: Option<Kind>,
    hi_first: impl FnOnce() -> Result<bool, E>,
    multiply: impl Fn(&Tensor, &Tensor) -> Result<Tensor, E>,
) -> Result<Tensor, E> {
    let lut = {
        let mut cache = E2M1_LUT_CACHE.lock().unwrap();
        let key = (payload.device(), output_kind);
        if let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(key) {
            let table = Tensor::f_from_slice(&[
                0_f32, 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
            ])?
            .f_unsqueeze(1)?
            .f_to_device_(
                payload.device(),
                output_kind.unwrap_or(Kind::Float),
                false,
                false,
            )?;
            entry.insert(table);
        }
        cache[&key].shallow_clone()
    };
    let lo = payload.f_bitwise_and(15)?;
    let hi = payload.f_bitwise_right_shift_tensor_scalar(4)?;
    let mut shape = payload.size();
    *shape
        .last_mut()
        .ok_or_else(|| tch::TchError::Shape("NVFP4 packed payload needs a dimension".into()))? = -1;
    let nibbles = if hi_first()? { [hi, lo] } else { [lo, hi] };
    let unpacked = Tensor::f_stack(&nibbles, -1)?.f_view(shape.as_slice())?;
    let out = Tensor::f_embedding(&lut, &unpacked.f_to_kind(Kind::Int)?, -1, false, false)?
        .f_squeeze_dim(-1)?;
    let shape = out.size();
    let blocked = out.f_reshape([shape[0], -1, 16])?;
    let scales = crate::block_scales::from_blocked(block_scales, shape[0], shape[1] / 16)?;
    let cast = |t: &Tensor| match output_kind {
        Some(kind) => t.f_to_kind(kind),
        None => Ok(t.shallow_clone()),
    };
    let scale = multiply(&cast(global_scale)?, &cast(&scales)?)?;
    cast(&multiply(&blocked, &scale.f_unsqueeze(-1)?)?.f_view(shape.as_slice())?).map_err(E::from)
}

pub fn quantize_fp8(x: &Tensor, scale: &Tensor, kind: Kind) -> Result<Tensor, tch::TchError> {
    let limit = match kind {
        Kind::Float8e4m3fn => 448.0,
        Kind::Float8e5m2 => 57344.0,
        _ => return Err(tch::TchError::Kind("Expected E4M3FN or E5M2".into())),
    };
    // Python Tensor.__rtruediv__: reciprocal(scale) * 1.0, then cast to x.dtype.
    let reciprocal = scale
        .f_reciprocal()?
        .f_mul_scalar(1.0)?
        .f_to_kind(x.kind())?;
    let mut temp = x.f_mul(&reciprocal)?;
    let _ = temp.f_clamp_(-limit, limit)?;
    temp.f_to_kind(kind)
}

pub fn dequantize_fp8(x: &Tensor, scale: &Tensor, kind: Kind) -> Result<Tensor, tch::TchError> {
    x.f_to_kind(kind)?.f_mul(&scale.f_to_kind(kind)?)
}

fn int8_scale_for_math(scale: &Tensor, x: &Tensor) -> Result<Tensor, Error> {
    let scale = scale.f_to_device(x.device())?.f_to_kind(x.kind())?;
    let tiny = match x.kind() {
        Kind::Half => 6.103515625e-5,
        Kind::BFloat16 | Kind::Float => f32::MIN_POSITIVE as f64,
        Kind::Double => f64::MIN_POSITIVE,
        _ => {
            return Err(Error::Native(
                "INT8 quantization expects floating input".into(),
            ));
        }
    };
    Ok(scale
        .f_full_like(tiny)?
        .f_where_self(&scale.f_eq(0.)?, &scale)?)
}
fn round_int8(native: &Kitchen, mut scaled: Tensor, seed: Option<u64>) -> Result<Tensor, Error> {
    if let Some(seed) = seed {
        let noise = native
            .generator(scaled.device(), seed)?
            .rand(&scaled.size(), scaled.kind())?;
        let _ = scaled.f_add_(&noise)?;
        let _ = scaled.f_floor_()?;
    } else {
        let _ = scaled.f_round_()?;
    }
    Ok(scaled.f_clamp_(-128., 127.)?.f_to_kind(Kind::Int8)?)
}
/// None corresponds to the reference's None/"recalculate" scale. Scalar inputs
/// are converted by the caller to a device float32 tensor before this operation.
pub fn quantize_int8_tensorwise(
    native: &Kitchen,
    x: &Tensor,
    scale: Option<&Tensor>,
    seed: Option<i64>,
) -> Result<(Tensor, Tensor), Error> {
    quantize_int8_tensorwise_with_seed(native, x, scale, || {
        Ok(seed.filter(|s| *s > 0).map(|s| s as u64))
    })
}
pub(crate) fn quantize_int8_tensorwise_with_seed(
    native: &Kitchen,
    x: &Tensor,
    scale: Option<&Tensor>,
    seed: impl FnOnce() -> Result<Option<u64>, Error>,
) -> Result<(Tensor, Tensor), Error> {
    let scale = match scale {
        None => x
            .f_abs()?
            .f_max()?
            .f_to_kind(Kind::Float)?
            .f_div_scalar(127.)?
            .f_clamp_min(1e-30)?,
        Some(scale) => scale.f_to_device(x.device())?.f_to_kind(Kind::Float)?,
    };
    let q = round_int8(native, x.f_div(&int8_scale_for_math(&scale, x)?)?, seed()?)?;
    Ok((q, scale))
}
pub fn quantize_int8_rowwise(
    native: &Kitchen,
    x: &Tensor,
    seed: Option<i64>,
) -> Result<(Tensor, Tensor), Error> {
    quantize_int8_rowwise_with_seed(native, x, || Ok(seed.filter(|s| *s > 0).map(|s| s as u64)))
}
pub(crate) fn quantize_int8_rowwise_with_seed(
    native: &Kitchen,
    x: &Tensor,
    seed: impl FnOnce() -> Result<Option<u64>, Error>,
) -> Result<(Tensor, Tensor), Error> {
    let scale = native
        .amax_last(&x.f_abs()?)?
        .f_to_kind(Kind::Float)?
        .f_div_scalar(127.)?
        .f_clamp_min(1e-30)?;
    let q = round_int8(native, x.f_div(&int8_scale_for_math(&scale, x)?)?, seed()?)?;
    Ok((q, scale))
}
pub fn dequantize_int8(x: &Tensor, scale: &Tensor) -> Result<Tensor, tch::TchError> {
    x.f_to_kind(Kind::Float)?.f_mul(scale)
}

pub fn quantize_mxfp8(native: &Kitchen, x: &Tensor, pad: bool) -> Result<(Tensor, Tensor), Error> {
    use crate::{block_scales, native::ShellDType};
    let (rows, cols) = x.size2()?;
    let input = if pad {
        let padded_rows = ((rows + 31) / 32) * 32;
        let padded_cols = ((cols + 31) / 32) * 32;
        if (rows, cols) != (padded_rows, padded_cols) {
            x.f_constant_pad_nd([0, padded_cols - cols, 0, padded_rows - rows])?
        } else {
            x.shallow_clone()
        }
    } else {
        if cols % 32 != 0 {
            return Err(Error::Native("K dimension must be divisible by 32".into()));
        }
        x.shallow_clone()
    };
    let (rows, cols) = input.size2()?;
    let blocked = input.f_reshape([rows, cols / 32, 32])?;
    let max_abs = blocked.f_abs()?.f_amax([-1], false)?;
    let needed = max_abs
        .f_to_kind(Kind::Float)?
        .f_div_scalar(448.)?
        .f_clamp_min(2_f64.powi(-127))?;
    let biased = needed
        .f_log2()?
        .f_ceil()?
        .f_to_kind(Kind::Int)?
        .f_add_scalar(127)?
        .f_clamp(0, 254)?;
    let scales = biased.f_to_kind(Kind::Uint8)?;
    let scales_f32 = block_scales::e8m0_to_f32(&scales)?;
    let zero_mask = max_abs.f_eq(0)?;
    let scales_f32 = scales_f32
        .f_ones_like()?
        .f_where_self(&zero_mask, &scales_f32)?;
    let scaled = blocked
        .f_to_kind(Kind::Float)?
        .f_div(&scales_f32.f_unsqueeze(-1)?)?;
    let scaled = scaled
        .f_zeros_like()?
        .f_where_self(&zero_mask.f_unsqueeze(-1)?, &scaled)?;
    let payload = scaled
        .f_clamp(-448., 448.)?
        .f_reshape([rows, cols])?
        .f_to_kind(Kind::Float8e4m3fn)?;
    let scales = scales.f_zeros_like()?.f_where_self(&zero_mask, &scales)?;
    let tiled = block_scales::to_blocked(&scales, false)?;
    Ok((payload, native.view_shell_dtype(&tiled, ShellDType::E8M0)?))
}

pub fn dequantize_mxfp8(
    payload: &Tensor,
    scales: &Tensor,
    output_kind: Kind,
) -> Result<Tensor, tch::TchError> {
    use crate::block_scales;
    let (rows, cols) = payload.size2()?;
    let unpacked = block_scales::from_blocked(&scales.f_view_dtype(Kind::Uint8)?, rows, cols / 32)?;
    let scales = block_scales::e8m0_to_f32(&unpacked)?;
    payload
        .f_to_kind(Kind::Float)?
        .f_reshape([rows, cols / 32, 32])?
        .f_mul(&scales.f_unsqueeze(-1)?)?
        .f_reshape([rows, cols])?
        .f_to_kind(output_kind)
}
