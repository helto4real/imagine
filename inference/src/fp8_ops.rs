//! Kitchen tensor/fp8.py linear/mm/addmm paths, including original fallback.
//!
//! Dequantization is supplied by the caller's actual layout/backend dispatch.
//! This is not a replacement for the pending general tensor-subclass dispatcher.
use crate::{
    layouts,
    native::{Error, Kitchen, ScaledMm, ScaledMmFailure},
    storage::QuantizedStorage,
};
use tch::{Kind, Tensor};

#[derive(Clone, Copy)]
pub enum Operand<'a> {
    Plain(&'a Tensor),
    Quantized(&'a QuantizedStorage),
}
pub enum Value {
    Plain(Tensor),
    Quantized(QuantizedStorage),
}
#[derive(Debug, PartialEq, Eq)]
pub enum Path {
    ScaledMm,
    Dequantized,
}
pub struct Output {
    pub value: Value,
    pub path: Path,
}
impl Output {
    fn plain(tensor: Tensor, path: Path) -> Self {
        Self {
            value: Value::Plain(tensor),
            path,
        }
    }
}
impl Operand<'_> {
    fn decode(
        self,
        dequantize: &mut impl FnMut(&QuantizedStorage) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error> {
        match self {
            Self::Plain(tensor) => Ok(tensor.shallow_clone()),
            Self::Quantized(q) => dequantize(q),
        }
    }
}
fn scale(q: &QuantizedStorage) -> Result<&Tensor, Error> {
    q.params
        .tensors
        .get("scale")
        .ok_or_else(|| Error::Native("Missing layout tensor field scale".into()))
}
fn attempt(native: &Kitchen, request: ScaledMm<'_>) -> Result<Option<Tensor>, Error> {
    // The native helper keeps contiguous inside its RuntimeError catch.
    match native.scaled_mm_fp8(request) {
        Ok(tensor) => Ok(Some(tensor)),
        Err(ScaledMmFailure::Runtime(_)) => Ok(None),
        Err(ScaledMmFailure::Uncaught(error)) => Err(error),
    }
}
/// None = omitted out_dtype; Some(None) = explicit None; Some(Some(k)) = k.
pub fn linear(
    native: &Kitchen,
    input: Operand<'_>,
    weight: Operand<'_>,
    bias: Option<&Tensor>,
    out_dtype: Option<Option<Kind>>,
    dequantize: impl FnMut(&QuantizedStorage) -> Result<Tensor, Error>,
) -> Result<Output, Error> {
    linear_with_bias(
        native,
        input,
        weight,
        bias.map(Operand::Plain),
        out_dtype,
        dequantize,
    )
}
pub(crate) fn linear_with_bias(
    native: &Kitchen,
    input: Operand<'_>,
    weight: Operand<'_>,
    bias: Option<Operand<'_>>,
    out_dtype: Option<Option<Kind>>,
    mut dequantize: impl FnMut(&QuantizedStorage) -> Result<Tensor, Error>,
) -> Result<Output, Error> {
    if let (Operand::Quantized(a), Operand::Quantized(b)) = (input, weight) {
        // In the original this transpose happens before the catch block.
        let transposed = b.payload.f_tr()?;
        let sa = scale(a)?;
        let sb = scale(b)?;
        let fast_bias = bias.map(|b| b.decode(&mut dequantize)).transpose()?;
        if let Some(tensor) = attempt(
            native,
            ScaledMm {
                input: &a.payload,
                weight: &transposed,
                scale_a: sa,
                scale_b: sb,
                bias: fast_bias.as_ref(),
                output_kind: out_dtype.unwrap_or(Some(a.params.original_kind)),
            },
        )? {
            let value = if matches!(tensor.kind(), Kind::Float8e4m3fn | Kind::Float8e5m2) {
                let shape = tensor.size();
                Value::Quantized(layouts::fp8_parts(
                    tensor,
                    sa.f_mul(sb)?,
                    a.params.original_kind,
                    shape,
                )?)
            } else {
                Value::Plain(tensor)
            };
            return Ok(Output {
                value,
                path: Path::ScaledMm,
            });
        }
    }
    // Reference fallback uses original logical dtypes, not out_dtype.
    let a = input.decode(&mut dequantize)?;
    let b = weight.decode(&mut dequantize)?;
    let bias = bias.map(|v| v.decode(&mut dequantize)).transpose()?;
    Ok(Output::plain(
        a.f_linear(&b, bias.as_ref())?,
        Path::Dequantized,
    ))
}
pub fn mm(
    native: &Kitchen,
    a: Operand<'_>,
    b: Operand<'_>,
    out_dtype: Option<Option<Kind>>,
    mut dequantize: impl FnMut(&QuantizedStorage) -> Result<Tensor, Error>,
) -> Result<Output, Error> {
    if let (Operand::Quantized(qa), Operand::Quantized(qb)) = (a, b)
        && let Some(tensor) = attempt(
            native,
            ScaledMm {
                input: &qa.payload,
                weight: &qb.payload,
                scale_a: scale(qa)?,
                scale_b: scale(qb)?,
                bias: None,
                output_kind: out_dtype.unwrap_or(Some(qa.params.original_kind)),
            },
        )?
    {
        return Ok(Output::plain(tensor, Path::ScaledMm));
    }
    let a = a.decode(&mut dequantize)?;
    let b = b.decode(&mut dequantize)?;
    Ok(Output::plain(a.f_mm(&b)?, Path::Dequantized))
}
pub fn addmm(
    native: &Kitchen,
    bias: &Tensor,
    a: Operand<'_>,
    b: Operand<'_>,
    out_dtype: Option<Option<Kind>>,
    dequantize: impl FnMut(&QuantizedStorage) -> Result<Tensor, Error>,
) -> Result<Output, Error> {
    addmm_with_bias(native, Operand::Plain(bias), a, b, out_dtype, dequantize)
}
pub(crate) fn addmm_with_bias(
    native: &Kitchen,
    bias: Operand<'_>,
    a: Operand<'_>,
    b: Operand<'_>,
    out_dtype: Option<Option<Kind>>,
    mut dequantize: impl FnMut(&QuantizedStorage) -> Result<Tensor, Error>,
) -> Result<Output, Error> {
    if let (Operand::Quantized(qa), Operand::Quantized(qb)) = (a, b) {
        let fast_bias = bias.decode(&mut dequantize)?;
        if let Some(tensor) = attempt(
            native,
            ScaledMm {
                input: &qa.payload,
                weight: &qb.payload,
                scale_a: scale(qa)?,
                scale_b: scale(qb)?,
                bias: Some(&fast_bias),
                output_kind: out_dtype.unwrap_or(Some(qa.params.original_kind)),
            },
        )? {
            return Ok(Output::plain(tensor, Path::ScaledMm));
        }
    }
    let bias = bias.decode(&mut dequantize)?;
    let a = a.decode(&mut dequantize)?;
    let b = b.decode(&mut dequantize)?;
    Ok(Output::plain(bias.f_addmm(&a, &b)?, Path::Dequantized))
}
