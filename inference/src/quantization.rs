//! CUDA allocation and launch policies from the pinned Kitchen backend.
use crate::native::{Error, KernelArg, Kitchen, ShellDType};
use tch::{Device, Kind, Tensor};

fn stream(native: &Kitchen, x: &Tensor) -> Result<i64, Error> {
    match x.device() {
        Device::Cuda(index) => Ok(native.current_stream(index)? as i64),
        _ => Err(Error::Native("CUDA quantizer requires CUDA input".into())),
    }
}
fn round_up(value: i64, factor: i64) -> i64 {
    ((value + factor - 1) / factor) * factor
}
/// CUDA backend's FP8 dequantizer, without selecting a different backend.
/// # Safety
/// x must be a contiguous E4M3FN/E5M2 CUDA payload and scale a contiguous
/// float32 CUDA scalar on its device, ready on its current stream. The original
/// native wrapper does not validate dtype/scale buffers or their capacities.
pub unsafe fn dequantize_fp8(
    native: &Kitchen,
    x: &Tensor,
    scale: &Tensor,
    kind: Kind,
) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    if scale.numel() != 1 {
        return Err(Error::Native("Scale must be a scalar tensor".into()));
    }
    let input_code = match x.kind() {
        Kind::Float8e4m3fn => 5,
        Kind::Float8e5m2 => 6,
        _ => return Err(Error::Native("Expected E4M3FN or E5M2".into())),
    };
    let stream = stream(native, x)?;
    let output = Tensor::f_empty(x.size(), (kind, x.device()))?;
    let raw = x.f_view_dtype(Kind::Uint8)?;
    unsafe {
        native.call_unchecked(
            "dequantize_per_tensor_fp8",
            &[
                T(&raw),
                T(scale),
                T(&output),
                Int(input_code),
                Int(dtype_code(kind)?),
                Int(x.numel() as i64),
                Int(stream),
            ],
        )?;
    }
    Ok(output)
}
fn matrix_shape(x: &Tensor, alignment: i64, padded: bool) -> Result<(i64, i64), Error> {
    if x.dim() != 2 || !x.is_contiguous() {
        return Err(Error::Native(
            "Expected contiguous two-dimensional input".into(),
        ));
    }
    let shape = x.size();
    if padded {
        Ok((round_up(shape[0], alignment), round_up(shape[1], alignment)))
    } else if shape[0] % alignment == 0 && shape[1] % alignment == 0 {
        Ok((shape[0], shape[1]))
    } else {
        Err(Error::Native(format!(
            "Dimensions must be divisible by {alignment}"
        )))
    }
}
/// # Safety
/// Scale must be a contiguous float32 CUDA scalar on x's device. Both inputs
/// must be ready on the active ATen stream. This is the native launch contract.
pub unsafe fn nvfp4(
    native: &Kitchen,
    x: &Tensor,
    scale: &Tensor,
    epsilon: f64,
    padded: bool,
    hi_first: bool,
) -> Result<(Tensor, Tensor), Error> {
    unsafe { nvfp4_with_hi(native, x, scale, epsilon, padded, || Ok(hi_first)) }
}
/// # Safety
/// Original allocation, native operand and asynchronous stream contracts apply.
pub unsafe fn nvfp4_with_hi(
    native: &Kitchen,
    x: &Tensor,
    scale: &Tensor,
    epsilon: f64,
    padded: bool,
    hi_first: impl FnOnce() -> Result<bool, Error>,
) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Float, Int, Tensor as T};
    let (rows, cols) = matrix_shape(x, 16, padded)?;
    let stream = stream(native, x)?;
    let payload = Tensor::f_empty([rows, cols / 2], (Kind::Uint8, x.device()))?;
    let scales = Tensor::f_zeros(
        [round_up(rows, 128), round_up(cols / 16, 4)],
        (Kind::Uint8, x.device()),
    )?;
    let scale = if scale.dim() == 0 {
        scale.f_reshape([1])?
    } else {
        scale.shallow_clone()
    };
    // Original wrapper checks scalar dtype/rank/device; all buffers remain live
    // on the active ATen stream, including their asynchronous allocations.
    unsafe {
        native.call_unchecked(
            "quantize_nvfp4",
            &[
                T(x),
                T(&scale),
                T(&payload),
                T(&scales),
                Float(epsilon),
                Bool(padded),
                Bool(hi_first()?),
                Int(stream),
            ],
        )?;
    }
    Ok((payload, scales.f_view_dtype(Kind::Float8e4m3fn)?))
}
/// # Safety
/// Payload and tiled scales must be a matching NVFP4 pair. Scale must be a
/// contiguous float32 scalar. All inputs must share a CUDA device and be ready
/// on its active ATen stream. Native wrappers do not validate buffer capacities.
pub unsafe fn dequantize_nvfp4(
    native: &Kitchen,
    payload: &Tensor,
    scale: &Tensor,
    scales: &Tensor,
    kind: Kind,
    hi_first: bool,
) -> Result<Tensor, Error> {
    unsafe { dequantize_nvfp4_with_hi(native, payload, scale, scales, kind, || Ok(hi_first)) }
}
/// # Safety
/// Original allocation, native operand and asynchronous stream contracts apply.
pub unsafe fn dequantize_nvfp4_with_hi(
    native: &Kitchen,
    payload: &Tensor,
    scale: &Tensor,
    scales: &Tensor,
    kind: Kind,
    hi_first: impl FnOnce() -> Result<bool, Error>,
) -> Result<Tensor, Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    if !payload.is_contiguous() || payload.dim() != 2 || payload.kind() != Kind::Uint8 {
        return Err(Error::Native(
            "Expected contiguous two-dimensional packed FP4 bytes".into(),
        ));
    }
    let code = dtype_code(kind)?;
    let stream = stream(native, payload)?;
    let shape = payload.size();
    let output = Tensor::f_empty([shape[0], shape[1] * 2], (kind, payload.device()))?;
    let scales = scales.f_view_dtype(Kind::Uint8)?;
    let scale = if scale.dim() == 0 {
        scale.f_reshape([1])?
    } else {
        scale.shallow_clone()
    };
    // Validation is also required below at the native wrapper boundary.
    unsafe {
        native.call_unchecked(
            "dequantize_nvfp4",
            &[
                T(payload),
                T(&scale),
                T(&scales),
                T(&output),
                Int(code),
                Bool(hi_first()?),
                Int(stream),
            ],
        )?;
    }
    Ok(output)
}
pub fn mxfp8(native: &Kitchen, x: &Tensor, padded: bool) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let (rows, cols) = matrix_shape(x, 32, padded)?;
    let stream = stream(native, x)?;
    let payload = Tensor::f_empty([rows, cols], (Kind::Float8e4m3fn, x.device()))?;
    let scales = Tensor::f_zeros(
        [round_up(rows, 128), round_up(cols / 32, 4)],
        (Kind::Uint8, x.device()),
    )?;
    unsafe {
        native.call_unchecked(
            "quantize_mxfp8",
            &[T(x), T(&payload), T(&scales), Bool(padded), Int(stream)],
        )?;
    }
    Ok((payload, native.view_shell_dtype(&scales, ShellDType::E8M0)?))
}
pub fn int8_rowwise(
    native: &Kitchen,
    x: &Tensor,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    int8_rowwise_with_seed(native, x, || Ok((seed.enabled, seed.bits as i64)))
}
pub(crate) fn int8_rowwise_with_seed(
    native: &Kitchen,
    x: &Tensor,
    seed: impl FnOnce() -> Result<(bool, i64), Error>,
) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let shape = x.size();
    let width = *shape
        .last()
        .ok_or_else(|| Error::Native("Rowwise quantization needs a dimension".into()))?;
    let input = x.f_reshape([-1, width])?.f_contiguous()?;
    let output = Tensor::f_empty(input.size(), (Kind::Int8, input.device()))?;
    let scales = Tensor::f_empty([input.size()[0], 1], (Kind::Float, input.device()))?;
    let stream = stream(native, x)?;
    let (stochastic, seed) = seed()?;
    unsafe {
        native.call_unchecked(
            "quantize_int8_rowwise",
            &[
                T(&input),
                T(&output),
                T(&scales),
                Bool(stochastic),
                Int(seed),
                Int(stream),
            ],
        )?;
    }
    let mut scale_shape = shape.clone();
    *scale_shape.last_mut().unwrap() = 1;
    Ok((output.f_reshape(shape)?, scales.f_reshape(scale_shape)?))
}
#[derive(Clone, Copy)]
pub enum Int4Rows {
    Plain,
    ConvRot(i64),
    ConvRotToInt8(i64),
}
/// Exact three CUDA activation-quantizer wrappers, retaining packed versus
/// INT8 fallback storage, original validation order and Philox seed policy.
pub fn int4_rowwise(
    native: &Kitchen,
    x: &Tensor,
    path: Int4Rows,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    int4_rowwise_with_seed(native, x, path, |_| Ok(seed))
}
pub(crate) fn int4_rowwise_with_seed(
    native: &Kitchen,
    x: &Tensor,
    path: Int4Rows,
    seed: impl FnOnce(&str) -> Result<crate::rng_seed::NativeSeed, Error>,
) -> Result<(Tensor, Tensor), Error> {
    use KernelArg::{Bool, Int, Tensor as T};
    let shape = x.size();
    let width = *shape.last().ok_or_else(|| Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })?;
    let input = x.f_reshape([-1, width])?.f_contiguous()?;
    let invalid = |message: String| Error::ReferenceException {
        class: "ValueError",
        message,
    };
    match path {
        Int4Rows::Plain if width % 64 != 0 => {
            return Err(invalid(format!(
                "INT4 rowwise quantization requires K divisible by 64, got {width}"
            )));
        }
        Int4Rows::ConvRot(group) => {
            if ![16, 64, 256].contains(&group) {
                return Err(invalid(format!(
                    "INT4 ConvRot fused quantization requires group_size 16, 64, or 256, got {group}"
                )));
            }
            if width % group != 0 {
                return Err(invalid(format!(
                    "INT4 ConvRot fused quantization requires K divisible by group_size {group}, got {width}"
                )));
            }
        }
        Int4Rows::ConvRotToInt8(group) => {
            if group != 256 {
                return Err(invalid(format!(
                    "INT4 fallback INT8 activation quantization requires group_size 256, got {group}"
                )));
            }
            if width % group != 0 {
                return Err(invalid(format!(
                    "INT4 ConvRot fallback quantization requires K divisible by {group}, got {width}"
                )));
            }
        }
        _ => {}
    }
    let unpacked = matches!(path, Int4Rows::ConvRotToInt8(_));
    let output = Tensor::f_empty(
        [input.size()[0], if unpacked { width } else { width / 2 }],
        (Kind::Int8, x.device()),
    )?;
    let scales = Tensor::f_empty([input.size()[0], 1], (Kind::Float, x.device()))?;
    let stream = stream(native, x)?;
    let mut args = vec![T(&input), T(&output), T(&scales)];
    let name = match path {
        Int4Rows::Plain => "quantize_int4_rowwise",
        Int4Rows::ConvRot(group) => {
            args.push(Int(group));
            "quantize_int4_rowwise_convrot64"
        }
        Int4Rows::ConvRotToInt8(group) => {
            args.push(Int(group));
            "quantize_int4_rowwise_convrot64_to_int8"
        }
    };
    let seed = seed(name)?;
    args.extend([Bool(seed.enabled), Int(seed.bits as i64), Int(stream)]);
    // Exact wrapper validates dtype/ranks, K/group, output and scale capacities
    // before launch; all allocations and reshaped inputs remain live on stream.
    unsafe {
        native.call_unchecked(name, &args)?;
    }
    let mut packed_shape = shape.clone();
    let mut scale_shape = shape;
    *packed_shape.last_mut().unwrap() = if unpacked { width } else { width / 2 };
    *scale_shape.last_mut().unwrap() = 1;
    Ok((
        output.f_reshape(packed_shape)?,
        scales.f_reshape(scale_shape)?,
    ))
}
pub fn dtype_code(kind: Kind) -> Result<i64, Error> {
    match kind {
        Kind::Float => Ok(0),
        Kind::Half => Ok(1),
        Kind::BFloat16 => Ok(2),
        _ => Err(Error::Native(
            "Expected float32, float16 or bfloat16".into(),
        )),
    }
}
