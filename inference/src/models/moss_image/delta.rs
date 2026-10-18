//! Qwen3.5's original chunked gated DeltaNet prefill, including FP32 state.
use crate::native::{Error, Kitchen};
use tch::{Device, Kind, Tensor};

fn normalize(x: &Tensor) -> Result<Tensor, Error> {
    let denom = x
        .f_norm_scalaropt_dim(2., [-1], true)?
        .f_clamp_min(1e-12)?
        .f_expand_as(x)?;
    Ok(x.f_div(&denom)?)
}
#[expect(
    clippy::too_many_arguments,
    reason = "Preserve original DeltaNet operands and state options"
)]
pub fn moss_gated_delta_rule(
    native: &Kitchen,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    chunk: i64,
    initial: Option<&Tensor>,
    output_final_state: bool,
) -> Result<(Tensor, Option<Tensor>), Error> {
    let dtype = query.kind();
    let trans = |x: &Tensor| x.f_transpose(1, 2)?.f_contiguous()?.f_to_kind(Kind::Float);
    let query = trans(&normalize(query)?)?;
    let key = trans(&normalize(key)?)?;
    let value = trans(value)?;
    let beta = trans(beta)?;
    let g = trans(g)?;
    let s = key.size();
    let (batch, heads, seq, kdim) = (s[0], s[1], s[2], s[3]);
    let vdim = value.size()[3];
    let pad = (chunk - seq % chunk) % chunk;
    let total = seq + pad;
    let query = query.f_constant_pad_nd([0, 0, 0, pad])?;
    let key = key.f_constant_pad_nd([0, 0, 0, pad])?;
    let value = value.f_constant_pad_nd([0, 0, 0, pad])?;
    let beta = beta.f_constant_pad_nd([0, pad])?;
    let g = g.f_constant_pad_nd([0, pad])?;
    let query = query.f_mul_scalar(1. / (query.size()[3] as f64).sqrt())?;
    let vb = value.f_mul(&beta.f_unsqueeze(-1)?)?;
    let kb = key.f_mul(&beta.f_unsqueeze(-1)?)?;
    let reshape = |x: &Tensor| x.f_reshape([batch, heads, -1, chunk, x.size()[3]]);
    let query = reshape(&query)?;
    let key = reshape(&key)?;
    let _value = reshape(&value)?;
    let kb = reshape(&kb)?;
    let vb = reshape(&vb)?;
    let g = g.f_reshape([batch, heads, -1, chunk])?;
    let mask = Tensor::f_ones([chunk, chunk], (Kind::Bool, query.device()))?.f_triu(0)?;
    let g = g.f_cumsum(-1, None)?;
    let decay = g
        .f_unsqueeze(-1)?
        .f_sub(&g.f_unsqueeze(-2)?)?
        .f_tril(0)?
        .f_exp()?
        .f_to_kind(Kind::Float)?
        .f_tril(0)?;
    let attn = kb
        .f_matmul(&key.f_transpose(-1, -2)?)?
        .f_mul(&decay)?
        .f_masked_fill(&mask, 0.)?
        .f_neg()?;
    for i in 1..chunk {
        let row = native.clone_tensor(&attn.f_select(-2, i)?.f_narrow(-1, 0, i)?)?;
        let sub = native.clone_tensor(&attn.f_narrow(-2, 0, i)?.f_narrow(-1, 0, i)?)?;
        let update =
            row.f_unsqueeze(-1)?
                .f_mul(&sub)?
                .f_sum_dim_intlist([-2].as_slice(), false, None)?;
        attn.f_select(-2, i)?
            .f_narrow(-1, 0, i)?
            .f_copy_(&row.f_add(&update)?)?;
    }
    let attn = attn.f_add(&Tensor::f_eye(chunk, (attn.kind(), attn.device()))?)?;
    let value = attn.f_matmul(&vb)?;
    let kcum = attn.f_matmul(&kb.f_mul(&g.f_exp()?.f_unsqueeze(-1)?)?)?;
    let mut state = match initial {
        Some(x) => x.f_to_device_(value.device(), value.kind(), false, false)?,
        None => Tensor::f_zeros(
            [batch, heads, kdim, vdim],
            (native.default_dtype()?, Device::Cpu),
        )?
        .f_to_device_(value.device(), value.kind(), false, false)?,
    };
    let output = value.f_zeros_like()?;
    let mask = Tensor::f_ones([chunk, chunk], (Kind::Bool, query.device()))?.f_triu(1)?;
    for i in 0..total / chunk {
        let q = query.f_select(2, i)?;
        let k = key.f_select(2, i)?;
        let v = value.f_select(2, i)?;
        let attn = q
            .f_matmul(&k.f_transpose(-1, -2)?)?
            .f_mul(&decay.f_select(2, i)?)?
            .f_masked_fill_(&mask, 0.)?;
        let vp = kcum.f_select(2, i)?.f_matmul(&state)?;
        let vn = v.f_sub(&vp)?;
        let gi = g.f_select(2, i)?;
        let inter = q.f_mul(&gi.f_unsqueeze(-1)?.f_exp()?)?.f_matmul(&state)?;
        output
            .f_select(2, i)?
            .f_copy_(&inter.f_add(&attn.f_matmul(&vn)?)?)?;
        let last = gi.f_select(-1, chunk - 1)?;
        state = state
            .f_mul(&last.f_unsqueeze(-1)?.f_unsqueeze(-1)?.f_exp()?)?
            .f_add(
                &k.f_mul(&last.f_unsqueeze(-1)?.f_sub(&gi)?.f_exp()?.f_unsqueeze(-1)?)?
                    .f_transpose(-1, -2)?
                    .f_matmul(&vn)?,
            )?;
    }
    let output = output
        .f_reshape([batch, heads, -1, vdim])?
        .f_slice(2, 0, seq, 1)?
        .f_transpose(1, 2)?
        .f_contiguous()?
        .f_to_kind(dtype)?;
    Ok((
        output,
        if output_final_state {
            Some(state)
        } else {
            None
        },
    ))
}
