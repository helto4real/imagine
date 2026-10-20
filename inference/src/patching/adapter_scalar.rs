//! Source scalar/tensor coefficients shared by merged and bypass adapters.
use super::{Adapter, exception};
use crate::{execution_value::Value, native::Error};
use tch::Tensor;

pub(super) enum Scale {
    Scalar(f64),
    Tensor(Tensor),
    /// OFT/BOFT read alpha before try, but validate it inside the constraint
    /// comparison. Keep this value unevaluated until that original operation.
    Constraint(Value),
}
#[derive(Clone, Copy)]
pub(super) enum Rank {
    Integer(i128),
    Float(f64),
}
impl Rank {
    pub(super) fn from_value(value: &Value) -> Option<Self> {
        Some(match value {
            Value::Int(value) => Self::Integer(i128::from(*value)),
            Value::UInt(value) => Self::Integer(i128::from(*value)),
            Value::Bool(value) => Self::Integer(i128::from(*value)),
            Value::Float(value) => Self::Float(*value),
            _ => return None,
        })
    }
}
fn constraint(value: &Value) -> Result<Scale, Error> {
    if let Some(value) = numeric(value) {
        return Ok(Scale::Scalar(value));
    }
    if matches!(value, Value::Tensor(_)) {
        return Ok(Scale::Tensor(value.as_tensor()?));
    }
    Err(exception(
        "TypeError",
        format!(
            "'>' not supported between instances of '{}' and 'int'",
            super::options::Options::from_value(value.clone()).typename()
        ),
    ))
}
impl Scale {
    pub(super) fn scaled(&self, factor: f64) -> Result<Self, Error> {
        Ok(match self {
            Self::Scalar(value) => Self::Scalar(factor * value),
            Self::Tensor(value) => Self::Tensor(value.f_mul_scalar(factor)?),
            Self::Constraint(value) => return constraint(value)?.scaled(factor),
        })
    }
    pub(super) fn multiply_left(&self, value: &Tensor) -> Result<Tensor, Error> {
        Ok(match self {
            Self::Scalar(scale) => value.f_mul_scalar(*scale)?,
            Self::Tensor(scale) => scale.f_mul(value)?,
            Self::Constraint(alpha) => return constraint(alpha)?.multiply_left(value),
        })
    }
    pub(super) fn multiply_in_place(&self, value: &mut Tensor) -> Result<(), Error> {
        match self {
            Self::Scalar(scale) => {
                let _ = value.f_mul_scalar_(*scale)?;
            }
            Self::Tensor(scale) => {
                let _ = value.f_mul_(scale)?;
            }
            Self::Constraint(alpha) => return constraint(alpha)?.multiply_in_place(value),
        }
        Ok(())
    }

    pub(super) fn multiply(&self, value: &Tensor) -> Result<Tensor, Error> {
        Ok(match self {
            Self::Scalar(scale) => value.f_mul_scalar(*scale)?,
            Self::Tensor(scale) => value.f_mul(scale)?,
            Self::Constraint(alpha) => return constraint(alpha)?.multiply(value),
        })
    }
    pub(super) fn positive(&self) -> Result<bool, Error> {
        match self {
            Self::Scalar(value) => Ok(*value > 0.),
            Self::Tensor(value) => Ok(value.f_gt(0.)?.f_is_nonzero()?),
            Self::Constraint(value) => constraint(value)?.positive(),
        }
    }
    pub(super) fn less_than(&self, value: &Tensor) -> Result<bool, Error> {
        Ok(match self {
            Self::Scalar(alpha) => value.f_gt(*alpha)?,
            Self::Tensor(alpha) => value.f_gt_tensor(alpha)?,
            Self::Constraint(alpha) => return constraint(alpha)?.less_than(value),
        }
        .f_is_nonzero()?)
    }
}
pub(super) fn numeric(value: &Value) -> Option<f64> {
    match value {
        Value::Float(n) => Some(*n),
        Value::Int(n) => Some(*n as f64),
        Value::UInt(n) => Some(*n as f64),
        Value::Bool(n) => Some(u8::from(*n) as f64),
        _ => None,
    }
}
pub(super) fn division_type_error(alpha: &Value, denominator: &str) -> Error {
    exception(
        "TypeError",
        format!(
            "unsupported operand type(s) for /: '{}' and '{denominator}'",
            super::options::Options::from_value(alpha.clone()).typename()
        ),
    )
}
pub(super) fn scale(
    adapter: &Adapter,
    index: usize,
    rank: Option<Rank>,
    multiplier: f64,
) -> Result<Scale, Error> {
    Ok(match quotient(adapter, index, rank)? {
        Scale::Scalar(value) => Scale::Scalar(value * multiplier),
        Scale::Tensor(value) => Scale::Tensor(value.f_mul_scalar(multiplier)?),
        Scale::Constraint(_) => {
            unreachable!("quotient returns a resolved scalar/tensor coefficient")
        }
    })
}
pub(super) fn quotient(
    adapter: &Adapter,
    index: usize,
    rank: Option<Rank>,
) -> Result<Scale, Error> {
    let alpha = adapter
        .weights
        .get(index)
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    if matches!(alpha, Value::None) {
        return Ok(Scale::Scalar(1.));
    }
    let rank = rank.ok_or_else(|| {
        exception(
            "UnboundLocalError",
            "cannot access local variable 'rank' where it is not associated with a value",
        )
    })?;
    if let Value::Tensor(_) = alpha {
        let alpha = alpha.as_tensor()?;
        return Ok(Scale::Tensor(match rank {
            Rank::Integer(rank) => {
                alpha.f_div_scalar(i64::try_from(rank).map_err(|_| Error::BindingGap {
                    backend: "adapter scalar".into(),
                    operation: "tensor divisor outside source shape int64".into(),
                })?)?
            }
            Rank::Float(rank) => alpha.f_div_scalar(rank)?,
        }));
    }
    let scalar = numeric(alpha).ok_or_else(|| division_type_error(alpha, "int"))?;
    if let (Some(Rank::Integer(alpha)), Rank::Integer(rank)) = (Rank::from_value(alpha), rank) {
        return Ok(Scale::Scalar(crate::python_numbers::integer_ratio(
            alpha, rank,
        )?));
    }
    let floating_rank = match rank {
        Rank::Integer(rank) => rank as f64,
        Rank::Float(rank) => rank,
    };
    if floating_rank == 0. {
        return Err(exception(
            "ZeroDivisionError",
            if matches!(alpha, Value::Float(_)) {
                "float division by zero"
            } else {
                "division by zero"
            },
        ));
    }
    Ok(Scale::Scalar(scalar / floating_rank))
}
