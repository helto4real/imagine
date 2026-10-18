//! Original W4/W6 Python seed value, kept unconverted until its source use.
use super::*;
#[derive(Clone)]
pub(super) enum Seed {
    Raw(Value),
    Offset(i128),
}
impl Seed {
    pub(super) fn zero() -> Self {
        Self::Raw(Value::Int(0))
    }
    pub(super) fn positive(&self, rt: &Runtime) -> Result<bool, Error> {
        match self {
            Self::Offset(n) => Ok(*n > 0),
            Self::Raw(Value::None) => Err(type_error(
                "'>' not supported between instances of 'NoneType' and 'int'",
            )),
            Self::Raw(value) => crate::backend_triton_rowwise::positive(rt, value),
        }
    }
    pub(super) fn integer(&self, rt: &Runtime) -> Result<i128, Error> {
        match self {
            Self::Offset(n) => Ok(*n),
            Self::Raw(v) => crate::python_numbers::integer_value(rt, v),
        }
    }
    pub(super) fn offset(&self, rt: &Runtime, start: i64) -> Result<Self, Error> {
        if !self.positive(rt)? {
            return Ok(Self::zero());
        }
        Ok(match self {
            Self::Offset(n) => Self::Offset(*n + i128::from(start)),
            Self::Raw(Value::Int(n)) => Self::Offset(i128::from(*n) + i128::from(start)),
            Self::Raw(Value::UInt(n)) => Self::Offset(i128::from(*n) + i128::from(start)),
            Self::Raw(Value::Bool(n)) => Self::Offset(i128::from(*n) + i128::from(start)),
            Self::Raw(Value::Float(n)) => Self::Raw(Value::Float(*n + start as f64)),
            Self::Raw(value @ (Value::Tensor(_) | Value::Quantized(_))) => Self::Raw(unsafe {
                rt.op(
                    "aten::add.Scalar",
                    &[value.clone(), Value::Int(start)],
                    &vec![],
                )?
            }),
            _ => unreachable!("Only numeric source seeds pass the positive test"),
        })
    }
    pub(super) fn generator(&self, rt: &Runtime) -> Result<u64, Error> {
        self.integer(rt)?
            .try_into()
            .map_err(|_| Error::ReferenceException {
                class: "RuntimeError",
                message: "Overflow when unpacking long".into(),
            })
    }
    pub(super) fn native(&self, rt: &Runtime, operation: &str) -> Result<(bool, i64), Error> {
        let tensor = matches!(self, Self::Raw(Value::Tensor(_) | Value::Quantized(_)));
        let enabled = if let Self::Raw(value @ (Value::Tensor(_) | Value::Quantized(_))) = self {
            // Python evaluates the comparison object, then int(seed), before
            // nanobind attempts to convert its strict bool parameter.
            unsafe { rt.op("aten::gt.Scalar", &[value.clone(), Value::Int(0)], &vec![])? };
            false
        } else {
            self.positive(rt)?
        };
        let seed = self.integer(rt)?;
        let seed = u64::try_from(seed);
        if tensor || seed.is_err() {
            return Err(super::native_seed_error(operation, tensor));
        }
        Ok((enabled, seed.unwrap() as i64))
    }
}
