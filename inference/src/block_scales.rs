//! Exact cuBLAS scale tiling and E8M0 decoding from Kitchen float_utils.py.
use tch::{Kind, TchError, Tensor};

pub fn to_blocked(input: &Tensor, flatten: bool) -> Result<Tensor, TchError> {
    let (rows, cols) = input.size2()?;
    let row_blocks = (rows + 127) / 128;
    let col_blocks = (cols + 3) / 4;
    let padded_rows = row_blocks * 128;
    let padded_cols = col_blocks * 4;
    let padded = if (rows, cols) == (padded_rows, padded_cols) {
        input.shallow_clone()
    } else {
        let output = Tensor::f_zeros([padded_rows, padded_cols], (input.kind(), input.device()))?;
        output
            .f_slice(0, 0, rows, 1)?
            .f_slice(1, 0, cols, 1)?
            .f_copy_(input)?;
        output
    };
    let rearranged = padded
        .f_view([row_blocks, 128, col_blocks, 4])?
        .f_permute([0, 2, 1, 3])?
        .f_reshape([-1, 4, 32, 4])?
        .f_transpose(1, 2)?
        .f_reshape([-1, 32, 16])?;
    if flatten {
        rearranged.f_flatten(0, -1)
    } else {
        rearranged.f_reshape([padded_rows, padded_cols])
    }
}

pub fn from_blocked(blocked: &Tensor, rows: i64, cols: i64) -> Result<Tensor, TchError> {
    let row_blocks = (rows + 127) / 128;
    let col_blocks = (cols + 3) / 4;
    blocked
        .f_reshape([-1, 32, 16])?
        .f_reshape([-1, 32, 4, 4])?
        .f_transpose(1, 2)?
        .f_reshape([row_blocks, col_blocks, 4, 32, 4])?
        .f_reshape([row_blocks, col_blocks, 128, 4])?
        .f_permute([0, 2, 1, 3])?
        .f_reshape([row_blocks * 128, col_blocks * 4])?
        .f_slice(0, 0, rows, 1)?
        .f_slice(1, 0, cols, 1)
}

pub fn e8m0_to_f32(input: &Tensor) -> Result<Tensor, TchError> {
    if input.kind() != Kind::Uint8 {
        return Err(TchError::Kind("Input must be uint8".into()));
    }
    let biased = input.f_to_kind(Kind::Int)?;
    let result = biased.f_bitwise_left_shift_tensor_scalar(23)?;
    result
        .f_zeros_like()?
        .f_where_self(&biased.f_eq(0)?, &result)?
        .f_view_dtype(Kind::Float)
}
