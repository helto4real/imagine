//! Original portable Kitchen INT8 ConvRot and embedding operation order.
use crate::{
    eager, int8_utils,
    native::{Error, Kitchen},
};
use tch::{Kind, Tensor};

pub fn rotate_weight(weight: &Tensor, group: i64) -> Result<Tensor, Error> {
    let h = int8_utils::hadamard(group, weight.device(), weight.kind())?;
    int8_utils::rotate_weight(weight, &h, group)
}

pub fn quantize_weight(
    native: &Kitchen,
    weight: &Tensor,
    group: i64,
    seed: Option<i64>,
) -> Result<(Tensor, Tensor), Error> {
    eager::quantize_int8_rowwise(native, &rotate_weight(weight, group)?, seed)
}

pub fn quantize_activation(
    native: &Kitchen,
    x: &Tensor,
    h: &Tensor,
    group: i64,
    seed: Option<i64>,
) -> Result<(Tensor, Tensor), Error> {
    let rotated = int8_utils::rotate_activation(x, h, group)?;
    eager::quantize_int8_rowwise(native, &rotated, seed)
}

pub fn dequantize_weight(
    q: &Tensor,
    scale: &Tensor,
    group: i64,
    kind: Kind,
) -> Result<Tensor, Error> {
    let h = int8_utils::hadamard(group, q.device(), Kind::Float)?;
    let decoded = eager::dequantize_int8(q, scale)?;
    Ok(int8_utils::rotate_weight(&decoded, &h, group)?.f_to_kind(kind)?)
}

/// Original eager embedding: gather before dequantizing and before rotation.
/// No complete floating copy of the quantized vocabulary table is created.
pub fn embedding(
    native: &Kitchen,
    q: &Tensor,
    scale: &Tensor,
    indices: &Tensor,
    group: i64,
    kind: Kind,
) -> Result<Tensor, Error> {
    let x = native.embedding(q, indices)?.f_to_kind(Kind::Float)?;
    let scale = if scale.dim() >= 2 {
        native.embedding(scale, indices)?.f_to_kind(Kind::Float)?
    } else {
        scale.f_to_kind(Kind::Float)?
    };
    let x = x.f_mul(&scale)?;
    let x = if group > 0 {
        let h = int8_utils::hadamard(group, q.device(), Kind::Float)?;
        let shape = x.size();
        let grouped = x.f_reshape([-1, shape[shape.len() - 1]])?;
        int8_utils::rotate_weight(&grouped, &h, group)?.f_reshape(shape.as_slice())?
    } else {
        x
    };
    Ok(x.f_to_kind(kind)?)
}
