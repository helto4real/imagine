//! Frozen CUDA ConvRot W4A4 weight quantization/dequantization wrappers.
use crate::{
    convrot, int8_utils,
    native::{Error, KernelArg, Kitchen},
    packing,
    quantization::{self, Int4Rows},
};
use tch::{Device, Kind, Tensor};

fn value_error(message: String) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message,
    }
}
fn tuple(shape: &[i64]) -> String {
    let contents = shape
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    format!("({contents}{})", if shape.len() == 1 { "," } else { "" })
}
fn validate_group(width: i64, group: i64) -> Result<(), Error> {
    if group == 0 {
        return Err(Error::ReferenceException {
            class: "ZeroDivisionError",
            message: "integer modulo by zero".into(),
        });
    }
    if width % group != 0 {
        return Err(value_error(format!(
            "in_features {width} not divisible by convrot_groupsize {group}"
        )));
    }
    Ok(())
}

/// Exact original INT4 fused shared-memory calculation, separate from INT8's.
pub fn fused_shared_memory_bytes(rows: i64, width: i64, group: i64, dtype_bytes: i64) -> i64 {
    if [16, 64].contains(&group) {
        let threads = if width > 4096 { 512 } else { 128 };
        let groups = threads / (group / 4);
        return (width + groups * 2 * group) * dtype_bytes;
    }
    let (threads, scratch) = if rows != 1 && width <= 4096 {
        (256, 2)
    } else if rows == 1 {
        (512, 2)
    } else if width == 15360 {
        (640, 1)
    } else {
        (1024, 2)
    };
    (width + (threads / 64) * scratch * 256) * dtype_bytes
}

pub(crate) fn fused_fits(native: &Kitchen, x: &Tensor, group: i64) -> Result<bool, Error> {
    match x.device() {
        Device::Cuda(device) if [16, 64, 256].contains(&group) => Ok(fused_shared_memory_bytes(
            x.size()[0],
            x.size()[1],
            group,
            x.kind().elt_size_in_bytes() as i64,
        )
            < convrot::max_dynamic_shared_memory(native, device)?),
        _ => Ok(true),
    }
}

/// # Safety
/// CUDA input must satisfy the original selected launch contract. Presence of
/// the original native entrypoints is fixed by the locked 69-entrypoint build.
pub unsafe fn quantize_weight(
    native: &Kitchen,
    weight: &Tensor,
    group: i64,
    quant_group: i64,
    seed: impl Into<crate::rng_seed::NativeSeed>,
) -> Result<(Tensor, Tensor), Error> {
    let seed = seed.into();
    unsafe { quantize_weight_with_seed(native, weight, group, quant_group, |_| Ok(seed)) }
}
/// # Safety
/// Retains the original selected wrapper contract with deferred seed conversion.
pub(crate) unsafe fn quantize_weight_with_seed(
    native: &Kitchen,
    weight: &Tensor,
    group: i64,
    quant_group: i64,
    seed: impl FnOnce(&str) -> Result<crate::rng_seed::NativeSeed, Error>,
) -> Result<(Tensor, Tensor), Error> {
    if quant_group != 64 {
        return Err(value_error(
            "int4 MMA kernel requires quant_group_size 64".into(),
        ));
    }
    if weight.dim() != 2 {
        return Err(value_error(format!(
            "ConvRot W4A4 expects a 2D tensor, got shape {}",
            tuple(&weight.size())
        )));
    }
    validate_group(weight.size()[1], group)?;
    let weight = weight.f_contiguous()?;
    let (q, scale) = if [16, 64, 256].contains(&group) && fused_fits(native, &weight, group)? {
        quantization::int4_rowwise_with_seed(native, &weight, Int4Rows::ConvRot(group), seed)?
    } else {
        let h = int8_utils::hadamard(group, weight.device(), weight.kind())?;
        let rotated = int8_utils::rotate_weight(&weight, &h, group)?.f_contiguous()?;
        quantization::int4_rowwise_with_seed(native, &rotated, Int4Rows::Plain, seed)?
    };
    Ok((q, scale.f_reshape([-1])?))
}

/// # Safety
/// Payload/scales must meet the selected original wrapper's dtype, capacity
/// and stream contract. Native shape/scale checks are retained.
pub unsafe fn dequantize_weight(
    native: &Kitchen,
    q: &Tensor,
    scales: &Tensor,
    group: i64,
    quant_group: i64,
    kind: Kind,
) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    if quant_group != 64 {
        return Err(value_error(
            "int4 MMA kernel requires quant_group_size 64".into(),
        ));
    }
    if q.dim() != 2 {
        return Err(value_error(format!(
            "ConvRot W4A4 dequant expects 2D qdata, got shape {}",
            tuple(&q.size())
        )));
    }
    let width = q.size()[1] * 2;
    validate_group(width, group)?;
    if [16, 64, 256].contains(&group) {
        let q = q.f_contiguous()?;
        let scales = scales
            .f_to_device_(q.device(), Kind::Float, false, false)?
            .f_reshape([-1])?
            .f_contiguous()?;
        let output = Tensor::f_empty([q.size()[0], width], (kind, q.device()))?;
        let device = match q.device() {
            Device::Cuda(device) => device,
            _ => return Err(Error::Native("CUDA ConvRot requires CUDA input".into())),
        };
        let stream = native.current_stream(device)? as i64;
        unsafe {
            native.call_unchecked(
                "dequantize_int4_convrot64",
                &[T(&q), T(&scales), T(&output), Int(group), Int(stream)],
            )?;
        }
        return Ok(output);
    }
    let unpacked = packing::unpack_int4_row_major(q, true)?.f_to_kind(Kind::Float)?;
    let scales = scales
        .f_to_device_(q.device(), Kind::Float, false, false)?
        .f_reshape([-1, 1])?;
    let rotated = unpacked.f_mul(&scales)?;
    let h = int8_utils::hadamard(group, q.device(), Kind::Float)?;
    Ok(int8_utils::rotate_weight(&rotated.f_to_kind(Kind::Float)?, &h, group)?.f_to_kind(kind)?)
}

/// Original `prepare_int4_weight_for_int8_linear`, retaining native unpack.
/// # Safety
/// q must be a signed packed INT4 two-dimensional CUDA tensor on active stream.
pub unsafe fn prepare_int8_weight(native: &Kitchen, q: &Tensor) -> Result<Tensor, Error> {
    use KernelArg::{Int, Tensor as T};
    if q.dim() != 2 {
        return Err(value_error(
            "prepared INT4 fallback weight expects a 2D tensor".into(),
        ));
    }
    let q = q.f_contiguous()?;
    let output = Tensor::f_empty([q.size()[0], q.size()[1] * 2], (Kind::Int8, q.device()))?;
    let device = match q.device() {
        Device::Cuda(device) => device,
        _ => return Err(Error::Native("CUDA ConvRot requires CUDA input".into())),
    };
    let stream = native.current_stream(device)? as i64;
    unsafe {
        native.call_unchecked("unpack_int4_to_int8", &[T(&q), T(&output), Int(stream)])?;
    }
    Ok(output)
}
