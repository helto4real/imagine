//! Original eager INT8 accumulation/padding, linear scaling and activations.
use crate::{
    convrot::InputActivation,
    eager, int8_utils,
    native::{Error, Kitchen},
};
use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};
use tch::{Device, Kind, Tensor};

fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn tuple(shape: &[i64]) -> String {
    format!(
        "({}{})",
        shape
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(", "),
        if shape.len() == 1 { "," } else { "" }
    )
}
fn last_dimension(x: &Tensor) -> Result<i64, Error> {
    x.size()
        .last()
        .copied()
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))
}

pub fn apply_input(x: &Tensor, activation: &InputActivation<'_>) -> Result<Tensor, Error> {
    Ok(match activation {
        InputActivation::None => x.shallow_clone(),
        InputActivation::GeluTanh => x.f_gelu("tanh")?,
        InputActivation::SwiGlu => {
            let parts = x.f_chunk(2, -1)?;
            if parts.len() != 2 {
                return Err(exception(
                    "ValueError",
                    format!(
                        "not enough values to unpack (expected 2, got {})",
                        parts.len()
                    ),
                ));
            }
            parts[0].f_silu()?.f_mul_(&parts[1])?
        }
        InputActivation::RmsNormDefault { weight } => {
            let weight = weight.ok_or_else(|| {
                exception("ValueError", "input_act 'rms_norm' requires act_weight")
            })?;
            x.f_rms_norm(
                [last_dimension(x)?],
                Some(weight.f_to_kind(x.kind())?),
                None,
            )?
        }
        InputActivation::RmsNorm { weight, epsilon } => {
            let weight = weight.ok_or_else(|| {
                exception("ValueError", "input_act 'rms_norm' requires act_weight")
            })?;
            x.f_rms_norm(
                [last_dimension(x)?],
                Some(weight.f_to_kind(x.kind())?),
                Some(*epsilon),
            )?
        }
    })
}

pub fn apply_residual(
    out: &Tensor,
    residual: Option<&Tensor>,
    scale: Option<&Tensor>,
) -> Result<Tensor, Error> {
    let Some(residual) = residual else {
        return Ok(out.shallow_clone());
    };
    let scale = scale.ok_or_else(|| exception("ValueError", "residual requires residual_scale"))?;
    Ok(residual
        .f_to_kind(out.kind())?
        .f_addcmul(out, &scale.f_to_kind(out.kind())?)?)
}

static TURING_CACHE: LazyLock<Mutex<HashMap<usize, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
fn n_alignment(native: &Kitchen, x: &Tensor) -> i64 {
    let Device::Cuda(index) = x.device() else {
        return 8;
    };
    let mut cache = TURING_CACHE.lock().unwrap();
    let turing = *cache.entry(index).or_insert_with(|| {
        native
            .device_properties(index)
            .map(|p| (p.major, p.minor) == (7, 5))
            .unwrap_or(false)
    });
    if turing { 32 } else { 8 }
}
fn round_up(n: i64, alignment: i64) -> i64 {
    ((n + alignment - 1) / alignment) * alignment
}

/// Original `_int8_matmul_accumulate` for its internal 2D INT8 contract.
/// The frozen Torch has no torch.int8_mm; its original `_int_mm` branch is used.
fn accumulate(native: &Kitchen, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    let (m, n, k) = (a.size()[0], b.size()[1], a.size()[1]);
    if m == 0 || k == 0 {
        return Ok(a.f_internal_int_mm(b)?);
    }
    let mut a = a.shallow_clone();
    let mut b = b.shallow_clone();
    let padded_m = if matches!(a.device(), Device::Cuda(_)) {
        round_up(m.max(32), 32)
    } else {
        m
    };
    if padded_m != m {
        let padding = Tensor::f_zeros([padded_m - m, k], (a.kind(), a.device()))?;
        a = Tensor::f_cat(&[a, padding], 0)?;
    }
    let padded_k = round_up(k, 8);
    if padded_k != k {
        let a_padding = Tensor::f_zeros([a.size()[0], padded_k - k], (a.kind(), a.device()))?;
        let b_padding = Tensor::f_zeros([padded_k - k, b.size()[1]], (b.kind(), b.device()))?;
        a = Tensor::f_cat(&[a, a_padding], 1)?;
        b = Tensor::f_cat(&[b, b_padding], 0)?;
    }
    let padded_n = round_up(n, n_alignment(native, &a));
    if padded_n != n {
        let padding = Tensor::f_zeros([b.size()[0], padded_n - n], (b.kind(), b.device()))?;
        b = Tensor::f_cat(&[b, padding], 1)?;
    }
    let result = a.f_internal_int_mm(&b)?;
    if result.size()[0] != m || result.size()[1] != n {
        Ok(result.f_slice(0, 0, m, 1)?.f_slice(1, 0, n, 1)?)
    } else {
        Ok(result)
    }
}

pub fn mm(native: &Kitchen, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    if a.kind() != Kind::Int8 || b.kind() != Kind::Int8 || a.dim() != 2 || b.dim() != 2 {
        return Err(exception("AssertionError", ""));
    }
    if a.size()[1] != b.size()[0] {
        return Err(exception(
            "AssertionError",
            format!("K mismatch: {} vs {}", a.size()[1], b.size()[0]),
        ));
    }
    accumulate(native, a, b)
}

pub struct Linear<'a> {
    pub x: &'a Tensor,
    pub weight: &'a Tensor,
    pub weight_scale: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub output_kind: Kind,
    pub convrot: bool,
    pub convrot_group: i64,
    pub input_activation: InputActivation<'a>,
    pub residual: Option<&'a Tensor>,
    pub residual_scale: Option<&'a Tensor>,
}

/// Portable INT8 linear in the exact original eager operation/chunk order.
/// Its 2D weight contract is supplied by the original registry's constraints.
pub fn linear(native: &Kitchen, request: &Linear<'_>) -> Result<Tensor, Error> {
    let mut x = apply_input(request.x, &request.input_activation)?;
    let width = last_dimension(&x)?;
    let weight_width = last_dimension(request.weight)?;
    if width != weight_width {
        return Err(exception(
            "ValueError",
            format!("Input and weight inner dimensions must match, got {width} and {weight_width}"),
        ));
    }
    let weight = request.weight.f_to_device(x.device())?.f_contiguous()?;
    let weight_scale = request
        .weight_scale
        .f_to_device_(x.device(), Kind::Float, false, false)?
        .f_reshape([-1])?;
    if weight_scale.numel() != 1 && weight_scale.numel() != weight.size()[0] as usize {
        return Err(exception(
            "ValueError",
            format!(
                "INT8 weight scale must be scalar or per-output-channel, got {} for weight shape {}",
                tuple(&weight_scale.size()),
                tuple(&weight.size())
            ),
        ));
    }
    if request.convrot {
        let group = request.convrot_group;
        if group == 0 {
            return Err(exception("ZeroDivisionError", "integer modulo by zero"));
        }
        if width % group != 0 {
            return Err(exception(
                "ValueError",
                format!("ConvRot group size {group} does not divide input features {width}"),
            ));
        }
        let h = int8_utils::hadamard(group, x.device(), x.kind())?;
        x = int8_utils::rotate_activation(&x, &h, group)?;
    }
    let shape = x.size();
    let x_2d = x.f_reshape([-1, width])?;
    let (x8, x_scale) = eager::quantize_int8_rowwise(native, &x_2d, None)?;
    let result = accumulate(native, &x8, &weight.f_transpose(0, 1)?.f_contiguous()?)?;
    let (m, n) = (result.size()[0], result.size()[1]);
    if n == 0 {
        return Err(exception(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let chunk_size = m.min(256 * 1024 * 1024 / (n * 4)).max(1);
    let weight_scale = weight_scale.f_reshape([1, -1])?;
    let mut scaled = Vec::new();
    let mut i = 0;
    while i < m {
        let end = (i + chunk_size).min(m);
        let chunk = result.f_slice(0, i, end, 1)?.f_to_kind(Kind::Float)?;
        let scale = x_scale
            .f_slice(0, i, end, 1)?
            .f_to_device_(chunk.device(), Kind::Float, false, false)?
            .f_mul(&weight_scale)?;
        scaled.push(chunk.f_mul(&scale)?.f_to_kind(request.output_kind)?);
        i += chunk_size;
    }
    let mut result = native.cat(&scaled, 0)?;
    if let Some(bias) = request.bias {
        result = result.f_add(
            &bias
                .f_to_device_(result.device(), result.kind(), false, false)?
                .f_reshape([1, -1])?,
        )?;
    }
    let mut output_shape = shape[..shape.len() - 1].to_vec();
    output_shape.push(weight.size()[0]);
    apply_residual(
        &result.f_reshape(output_shape.as_slice())?,
        request.residual,
        request.residual_scale,
    )
}

pub fn fp16_linear(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    residual: Option<&Tensor>,
    scale: Option<&Tensor>,
) -> Result<Tensor, Error> {
    apply_residual(&x.f_linear(weight, bias)?, residual, scale)
}
