//! Original w4a8_packed.py record codec, preserving ATen allocation and view order.
use crate::native::Error;
use tch::{Kind, Tensor};
fn invalid(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "ValueError",
        message: message.into(),
    }
}
pub fn record_bytes(bits: i64) -> i64 {
    512 * bits / 8 + 32
}
pub fn bit_depth(count: i64, n: i64, k: i64) -> Result<i64, Error> {
    for bits in [4, 6] {
        if count == n * k * record_bytes(bits) / 512 {
            return Ok(bits);
        }
    }
    Err(invalid(format!(
        "MMA packed weight must be a contiguous int8 [{}] or [{}] tensor",
        n * k * 9 / 16,
        n * k * 13 / 16
    )))
}
fn fragment_inverse(fragments: &Tensor, cols: i64) -> Result<Tensor, Error> {
    let shape = fragments.size();
    let (tiles, rows, half) = (shape[0], shape[1], cols / 2);
    let codes = Tensor::f_empty(
        [tiles, rows, 16, cols],
        (fragments.kind(), fragments.device()),
    )?;
    for (i, start, col) in [(0, 0, 0), (1, 8, 0), (2, 0, half), (3, 8, half)] {
        codes
            .f_narrow(2, start, 8)?
            .f_narrow(3, col, half)?
            .f_copy_(
                &fragments
                    .f_select(4, i)?
                    .f_reshape([tiles, rows, 8, half])?,
            )?;
    }
    Ok(codes)
}
pub fn unpack(
    packed: &Tensor,
    n: i64,
    k: i64,
    scale_kind: Kind,
) -> Result<(Tensor, Tensor), Error> {
    if packed.dim() != 1 || packed.kind() != Kind::Int8 || !packed.is_contiguous() {
        return Err(invalid(
            "MMA packed weight must be a contiguous 1D int8 tensor",
        ));
    }
    if n % 16 != 0 || k % 256 != 0 {
        return Err(invalid(
            "MMA packed weight requires N % 16 == 0 and K % 256 == 0",
        ));
    }
    let bits = bit_depth(packed.numel() as i64, n, k)?;
    let bytes = record_bytes(bits);
    let code_bytes = bytes - 32;
    let (tiles, rows) = (n / 16, k / 32);
    let records = packed
        .f_view_dtype(Kind::Uint8)?
        .f_view([tiles, rows, bytes])?;
    let qdata = if bits == 4 {
        let fragments = records
            .f_narrow(2, 0, code_bytes)?
            .f_view([tiles, rows, 8, 4, 4, 2])?;
        fragment_inverse(&fragments, 16)?
            .f_permute([0, 2, 1, 3])?
            .f_reshape([n, k / 2])?
    } else {
        let low = records
            .f_narrow(2, 0, 256)?
            .f_reshape([tiles, rows, 32, 8])?;
        let high = records
            .f_narrow(2, 256, code_bytes - 256)?
            .f_reshape([tiles, rows, 32, 4])?;
        let lane_bytes = Tensor::f_cat(&[low, high], 3)?
            .f_to_kind(Kind::Int)?
            .f_view([tiles, rows, 8, 4, 4, 3])?;
        let words = lane_bytes
            .f_select(5, 0)?
            .f_bitwise_or_tensor(
                &lane_bytes
                    .f_select(5, 1)?
                    .f_bitwise_left_shift_tensor_scalar(8)?,
            )?
            .f_bitwise_or_tensor(
                &lane_bytes
                    .f_select(5, 2)?
                    .f_bitwise_left_shift_tensor_scalar(16)?,
            )?;
        let shifts = Tensor::f_from_slice(&[0_i32, 6, 12, 18])?.f_to_device(packed.device())?;
        let fragments = words
            .f_unsqueeze(-1)?
            .f_bitwise_right_shift(&shifts)?
            .f_bitwise_and(63)?;
        let codes = fragment_inverse(&fragments, 32)?
            .f_permute([0, 2, 1, 3])?
            .f_reshape([n, k])?;
        super::pack_codes(&codes, bits)?.f_view_dtype(Kind::Uint8)?
    };
    let fragments = records
        .f_narrow(2, code_bytes, 32)?
        .f_view([tiles, rows, 8, 4])?;
    let scale_tiles = Tensor::f_empty([tiles, rows, 16, 2], (Kind::Uint8, packed.device()))?;
    for (i, start, col) in [(0, 0, 0), (1, 8, 0), (2, 0, 1), (3, 8, 1)] {
        scale_tiles
            .f_narrow(2, start, 8)?
            .f_select(3, col)?
            .f_copy_(&fragments.f_select(3, i)?)?;
    }
    let scales = scale_tiles
        .f_permute([0, 2, 1, 3])?
        .f_reshape([n, k / 16])?;
    Ok((
        qdata.f_view_dtype(Kind::Int8)?,
        scales.f_view_dtype(scale_kind)?,
    ))
}
