//! W4A8/W6A8 code planes from the frozen eager backend, in ATen order.
pub mod mma;
use crate::native::Error;
use tch::{Kind, Tensor};

/// Internal helper: unsigned int32 [N,K] codes with the layout's valid K.
/// The original helper treats the non-four-bit case as the six-bit plane.
pub fn pack_codes(unsigned: &Tensor, bits: i64) -> Result<Tensor, Error> {
    let k = unsigned.size()[1];
    let even = unsigned.f_slice(1, 0, k, 2)?.f_bitwise_and(15)?;
    let odd = unsigned
        .f_slice(1, 1, k, 2)?
        .f_bitwise_and(15)?
        .f_bitwise_left_shift_tensor_scalar(4)?;
    let low = even.f_bitwise_or_tensor(&odd)?.f_to_kind(Kind::Int8)?;
    if bits == 4 {
        return Ok(low.f_contiguous()?);
    }
    let mut hi = unsigned
        .f_slice(1, 0, k, 4)?
        .f_bitwise_right_shift_tensor_scalar(4)?
        .f_bitwise_and(3)?;
    for j in 1..4 {
        let term = unsigned
            .f_slice(1, j, k, 4)?
            .f_bitwise_right_shift_tensor_scalar(4)?
            .f_bitwise_and(3)?
            .f_bitwise_left_shift_tensor_scalar(2 * j)?;
        let _ = hi.f_bitwise_or_tensor_(&term)?;
    }
    Ok(Tensor::f_cat(&[low, hi.f_to_kind(Kind::Int8)?], 1)?.f_contiguous()?)
}

/// Internal inverse for valid int8 [N,K*bits/8] storage and its logical K.
pub fn unpack_codes(qdata: &Tensor, k: i64, bits: i64) -> Result<Tensor, Error> {
    let n = qdata.size()[0];
    let packed = qdata.f_view_dtype(Kind::Uint8)?.f_to_kind(Kind::Int)?;
    let low = packed.f_slice(1, 0, k / 2, 1)?;
    let mut codes = Tensor::f_empty([n, k], (Kind::Int, qdata.device()))?;
    codes
        .f_slice(1, 0, k, 2)?
        .f_copy_(&low.f_bitwise_and(15)?)?;
    codes.f_slice(1, 1, k, 2)?.f_copy_(
        &low.f_bitwise_right_shift_tensor_scalar(4)?
            .f_bitwise_and(15)?,
    )?;
    if bits == 6 {
        let shifts = Tensor::f_from_slice(&[0_i32, 2, 4, 6])?.f_to_device(qdata.device())?;
        let hi = packed
            .f_slice(1, k / 2, packed.size()[1], 1)?
            .f_unsqueeze(-1)?
            .f_bitwise_right_shift(&shifts)?
            .f_bitwise_and(3)?;
        let _ = codes.f_bitwise_or_tensor_(
            &hi.f_reshape([n, k])?
                .f_bitwise_left_shift_tensor_scalar(4)?,
        )?;
    }
    Ok(codes)
}

fn value_error(message: String) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message,
    }
}

pub fn check_six_bit_layout(k: i64, group: i64) -> Result<(), Error> {
    if k % 32 != 0 || group < 16 || group % 16 != 0 {
        return Err(value_error(format!(
            "6-bit storage needs K divisible by 32 and group_size a multiple of 16, got K={k}, group_size={group}"
        )));
    }
    Ok(())
}

/// Infer packed bit depth from the two-dimensional payload and scale grid.
/// Callers retain the reference's preceding rank/type validation contract.
pub fn geometry(qdata: &Tensor, scales: &Tensor, group: i64) -> Result<(i64, i64, i64), Error> {
    let shape = qdata.size();
    let n = shape[0];
    let cols = shape[1];
    let k = scales.size()[1] * group;
    let bits = if k > 0 { cols * 8 / k } else { 0 };
    if k <= 0 || ![4, 6].contains(&bits) || k * bits != cols * 8 {
        return Err(value_error(format!(
            "packed row width {cols} does not match K={k} at 4 or 6 bits"
        )));
    }
    if bits == 6 {
        check_six_bit_layout(k, group)?;
    }
    Ok((n, k, bits))
}

/// Original SVDQuant row-major signed/unsigned nibble codec. This storage
/// codec preserves all nibble patterns, including signed -8.
pub fn unpack_int4_row_major(packed: &Tensor, signed: bool) -> Result<Tensor, Error> {
    let x = packed.f_to_kind(Kind::Int)?;
    let lo = x.f_bitwise_and(15)?;
    let hi = x
        .f_bitwise_right_shift_tensor_scalar(4)?
        .f_bitwise_and(15)?;
    let (lo, hi) = if signed {
        (
            lo.f_sub_scalar(16)?.f_where_self(&lo.f_ge(8)?, &lo)?,
            hi.f_sub_scalar(16)?.f_where_self(&hi.f_ge(8)?, &hi)?,
        )
    } else {
        (lo, hi)
    };
    let mut shape = packed.size();
    let _ = shape.pop();
    shape.push(-1);
    Ok(Tensor::f_stack(&[lo, hi], -1)?
        .f_reshape(shape.as_slice())?
        .f_to_kind(Kind::Int8)?)
}

pub fn pack_int4_row_major(values: &Tensor) -> Result<Tensor, Error> {
    let shape = values.size();
    let width = *shape.last().ok_or_else(|| Error::ReferenceException {
        class: "IndexError",
        message: "tuple index out of range".into(),
    })?;
    if width % 2 != 0 {
        return Err(value_error(format!("last dim must be even, got {width}")));
    }
    let lo = values
        .f_slice(-1, 0, width, 2)?
        .f_to_kind(Kind::Int)?
        .f_bitwise_and(15)?;
    let hi = values
        .f_slice(-1, 1, width, 2)?
        .f_to_kind(Kind::Int)?
        .f_bitwise_and(15)?;
    Ok(lo
        .f_bitwise_or_tensor(&hi.f_bitwise_left_shift_tensor_scalar(4)?)?
        .f_to_kind(Kind::Int8)?)
}
