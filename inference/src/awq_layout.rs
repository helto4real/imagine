//! Frozen AWQ layout handlers. General tensor-subclass/backend binding is separate.
use crate::{
    awq, layouts,
    native::{Error, Kitchen},
    storage::{LayoutParams, QuantizedStorage, ScalarField},
};
use std::collections::BTreeMap;
use tch::{Kind, Tensor};

pub const MIN_SM_VERSION: Option<(i64, i64)> = None;
pub const QUANTIZES_INPUT: bool = false;

fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}

pub fn parts(
    payload: Tensor,
    scale: Tensor,
    zeros: Tensor,
    original_kind: Kind,
    original_shape: Vec<i64>,
    group_size: i64,
    transposed: bool,
) -> QuantizedStorage {
    // Params validation is deliberately a no-op: neither tensor is coerced.
    QuantizedStorage::from_validated_parts(
        payload,
        "TensorCoreAWQW4A16Layout".into(),
        LayoutParams {
            original_kind,
            original_shape,
            tensors: BTreeMap::from([("scale".into(), scale), ("zeros".into(), zeros)]),
            fields: BTreeMap::from([
                ("group_size".into(), ScalarField::Integer(group_size)),
                ("transposed".into(), ScalarField::Bool(transposed)),
            ]),
            field_order: [
                "scale",
                "orig_dtype",
                "orig_shape",
                "zeros",
                "group_size",
                "transposed",
            ]
            .map(str::to_owned)
            .to_vec(),
        },
    )
}

pub fn quantize(_tensor: &Tensor) -> Result<QuantizedStorage, Error> {
    Err(exception(
        "NotImplementedError",
        "AWQ W4A16 requires offline calibration. Load pre-quantized tensors via `from_state_dict` instead.",
    ))
}

fn field<'a>(params: &'a LayoutParams, name: &str) -> Result<&'a Tensor, Error> {
    params.tensors.get(name).ok_or_else(|| {
        exception(
            "AttributeError",
            format!("'Params' object has no attribute '{name}'"),
        )
    })
}

fn group(params: &LayoutParams) -> i64 {
    match params.fields.get("group_size") {
        Some(ScalarField::Integer(n)) => *n,
        None => awq::DEFAULT_GROUP,
        _ => unreachable!("typed validated AWQ group parameter"),
    }
}
fn transposed(params: &LayoutParams) -> Result<bool, Error> {
    match params.fields.get("transposed") {
        Some(ScalarField::Bool(value)) => Ok(*value),
        None => Err(exception(
            "AttributeError",
            "'Params' object has no attribute 'transposed'",
        )),
        _ => unreachable!("typed validated logical transpose flag"),
    }
}

pub fn plain_tensors(q: &QuantizedStorage) -> Result<(&Tensor, &Tensor, &Tensor), Error> {
    Ok((
        &q.payload,
        field(&q.params, "scale")?,
        field(&q.params, "zeros")?,
    ))
}
pub fn state_dict<'a>(
    q: &'a QuantizedStorage,
    prefix: &str,
) -> Result<Vec<(String, &'a Tensor)>, Error> {
    let (payload, scale, zeros) = plain_tensors(q)?;
    Ok(vec![
        (prefix.into(), payload),
        (format!("{prefix}_scale"), scale),
        (format!("{prefix}_zeros"), zeros),
    ])
}

pub fn transpose(q: &QuantizedStorage) -> Result<QuantizedStorage, Error> {
    let shape = &q.params.original_shape;
    // Reference evaluates shape[1] and shape[0] before toggling the flag.
    if shape.len() < 2 {
        return Err(exception("IndexError", "tuple index out of range"));
    }
    let mut params = q.params.shallow_copy();
    params.original_shape = vec![shape[1], shape[0]];
    params.fields.insert(
        "transposed".into(),
        ScalarField::Bool(!transposed(&q.params)?),
    );
    Ok(QuantizedStorage::from_validated_parts(
        q.payload.shallow_clone(),
        "TensorCoreAWQW4A16Layout".into(),
        params,
    ))
}

/// Original layout dequantizer, before common crop/logical transpose handling.
pub fn dequantize_raw(payload: &Tensor, params: &LayoutParams) -> Result<Tensor, Error> {
    let shape = payload.size();
    if shape.len() != 2 {
        return Err(exception(
            "ValueError",
            if shape.len() > 2 {
                "too many values to unpack (expected 2)".into()
            } else {
                format!(
                    "not enough values to unpack (expected 2, got {})",
                    shape.len()
                )
            },
        ));
    }
    let (n, k) = (shape[0], shape[1] * 2);
    let g = group(params);
    let q = payload.f_to_kind(Kind::Int)?;
    let lo = q.f_bitwise_and(15)?.f_to_kind(Kind::Int)?;
    let hi = q
        .f_bitwise_right_shift_tensor_scalar(4)?
        .f_bitwise_and(15)?
        .f_to_kind(Kind::Int)?;
    let unsigned = Tensor::f_empty([n, k], (Kind::Int, payload.device()))?;
    unsigned.f_slice(1, 0, k, 2)?.f_copy_(&lo)?;
    unsigned.f_slice(1, 1, k, 2)?.f_copy_(&hi)?;
    let scales = field(params, "scale")?.f_tr()?.f_unsqueeze(-1)?;
    let zeros = field(params, "zeros")?.f_tr()?.f_unsqueeze(-1)?;
    if g == 0 {
        return Err(exception(
            "ZeroDivisionError",
            "integer division or modulo by zero",
        ));
    }
    let quotient = k / g;
    let quotient = if k % g != 0 && g < 0 {
        quotient - 1
    } else {
        quotient
    };
    let groups = unsigned
        .f_view_([n, quotient, g].as_slice())?
        .f_to_kind(params.original_kind)?;
    Ok(groups
        .f_sub_scalar(8.0)?
        .f_mul(&scales)?
        .f_add(&zeros)?
        .f_view_([n, k].as_slice())?)
}

pub fn dequantize(q: &QuantizedStorage) -> Result<Tensor, Error> {
    layouts::dequantize_with(q, dequantize_raw)
}

#[derive(Clone, Copy)]
pub enum Operand<'a> {
    Plain(&'a Tensor),
    Quantized(&'a QuantizedStorage),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Eager,
    Cuda,
}
pub type Implementation<'a> = dyn FnMut(&awq::Linear<'_>) -> Result<Tensor, Error> + 'a;

/// Typed AWQ handlers. A general dispatcher supplies other layouts' actual
/// dequantizers; this does not replace their behavior with AWQ dequantization.
pub struct Dispatch<'a, F> {
    pub native: &'a Kitchen,
    pub backend: Backend,
    /// General registry binding at the original backend-entry point.
    /// None retains the direct typed helper used by primitive tests.
    pub implementation: Option<&'a mut Implementation<'a>>,
    pub dequantize_other: F,
    /// Optional metadata-only development trace; ordinary dispatch installs none.
    pub observer: Option<&'a mut dyn FnMut(&str)>,
}
impl<F: FnMut(&QuantizedStorage) -> Result<Tensor, Error>> Dispatch<'_, F> {
    fn decode(&mut self, value: Operand<'_>) -> Result<Tensor, Error> {
        if let Operand::Quantized(q) = value
            && let Some(observer) = self.observer.as_mut()
        {
            observer(&format!("dequantize:{}", q.layout));
        }
        match value {
            Operand::Plain(t) => Ok(t.shallow_clone()),
            Operand::Quantized(q) if q.layout == "TensorCoreAWQW4A16Layout" => dequantize(q),
            Operand::Quantized(q) => (self.dequantize_other)(q),
        }
    }

    fn resolve_rhs(q: &QuantizedStorage) -> Result<(), Error> {
        if !transposed(&q.params)? {
            return Err(exception(
                "RuntimeError",
                "AWQ W4A16 GEMM expects the RHS to be W.T (stored W). Use F.linear(x, W) or mm(x, W.t()).",
            ));
        }
        Ok(())
    }

    unsafe fn forward(
        &mut self,
        x: &Tensor,
        weight: &QuantizedStorage,
        bias: Option<Operand<'_>>,
    ) -> Result<Tensor, Error> {
        if let Some(observer) = self.observer.as_mut() {
            observer("forward");
        }
        let (qweight, wscales, wzeros) = plain_tensors(weight)?;
        let group_size = group(&weight.params);
        // ck.gemv_awq_w4a16 enters a torch custom op. A quantized bias
        // triggers its generic dequantize_args fallback before backend entry.
        let bias = bias.map(|value| self.decode(value)).transpose()?;
        let request = awq::Linear {
            x,
            qweight,
            wscales,
            wzeros,
            bias: bias.as_ref(),
            group_size,
        };
        if let Some(implementation) = self.implementation.as_mut() {
            return implementation(&request);
        }
        match self.backend {
            Backend::Eager => awq::eager(&request),
            Backend::Cuda => unsafe { awq::cuda(self.native, &request) },
        }
    }

    /// # Safety
    /// Selected CUDA forward operands must satisfy awq::cuda's native contract.
    /// Quantized operands have the validated typed layout parameters above.
    pub unsafe fn linear(
        &mut self,
        input: Operand<'_>,
        weight: Operand<'_>,
        bias: Option<Operand<'_>>,
    ) -> Result<Tensor, Error> {
        if let Operand::Plain(weight) = weight {
            let input = self.decode(input)?;
            let bias = bias.map(|b| self.decode(b)).transpose()?;
            return Ok(input.f_linear(weight, bias.as_ref())?);
        }
        let Operand::Quantized(weight) = weight else {
            unreachable!()
        };
        let input = self.decode(input)?;
        if transposed(&weight.params)? {
            let weight = self.decode(Operand::Quantized(weight))?;
            let bias = bias.map(|b| self.decode(b)).transpose()?;
            return Ok(input.f_linear(&weight, bias.as_ref())?);
        }
        unsafe { self.forward(&input, weight, bias) }
    }

    /// # Safety
    /// Same typed/native operand contract as linear. Original handler kwargs
    /// are ignored; the future generic binding must not forward extra options.
    pub unsafe fn mm(&mut self, a: Operand<'_>, b: Operand<'_>) -> Result<Tensor, Error> {
        if let Operand::Plain(b) = b {
            return Ok(self.decode(a)?.f_mm(b)?);
        }
        let Operand::Quantized(b) = b else {
            unreachable!()
        };
        let a = self.decode(a)?;
        Self::resolve_rhs(b)?;
        unsafe { self.forward(&a, b, None) }
    }

    /// # Safety
    /// Same contract as linear. Like the original handler, alpha/beta and other
    /// handler kwargs are not forwarded, including on its dequantized fallback.
    pub unsafe fn addmm(
        &mut self,
        bias: Operand<'_>,
        a: Operand<'_>,
        b: Operand<'_>,
    ) -> Result<Tensor, Error> {
        if let Operand::Plain(b) = b {
            // dequantize_args traverses the original tuple (bias, a, b).
            let bias = self.decode(bias)?;
            let a = self.decode(a)?;
            return Ok(bias.f_addmm(&a, b)?);
        }
        let Operand::Quantized(b) = b else {
            unreachable!()
        };
        let a = self.decode(a)?;
        Self::resolve_rhs(b)?;
        unsafe { self.forward(&a, b, Some(bias)) }
    }
}
