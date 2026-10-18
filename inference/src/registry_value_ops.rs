//! Lazy Python operations used by the original Sol validation predicates.
use crate::{
    execution_value::Value,
    registry::{Argument, Failure},
    registry_source::SourceArgument,
};
use std::cmp::Ordering;

pub(crate) struct Operand<'a, 'rt> {
    value: Value,
    context: Option<&'a SourceArgument<'rt>>,
}
impl<'a, 'rt> Operand<'a, 'rt> {
    pub(crate) fn argument(value: &'a Argument<'rt>) -> Result<Self, Failure> {
        let context = if let Argument::Source(source) = value {
            Some(source)
        } else {
            None
        };
        let value = match value {
            Argument::Source(source) => source.value.clone(),
            Argument::None => Value::None,
            Argument::BoolValue(n) => Value::Bool(*n),
            Argument::Integer(n) => Value::Int(*n),
            Argument::Unsigned(n) => Value::UInt(*n),
            Argument::FloatValue(n) => Value::Float(*n),
            Argument::Text(s) => Value::Text(s.clone()),
            Argument::DType(k) => Value::DType(*k),
            Argument::Sequence(v) => Value::List(v.iter().map(|n| Value::Int(*n)).collect()),
            Argument::BoolSequence(v) => Value::List(v.iter().map(|n| Value::Bool(*n)).collect()),
            _ => {
                return Err(Failure::raised(
                    "RuntimeError",
                    "Value-dependent validation requires an owned source value",
                ));
            }
        };
        Ok(Self { value, context })
    }
    fn with(&self, value: Value) -> Self {
        Self {
            value,
            context: self.context,
        }
    }
    fn op(&self, op: &str, args: &[Value]) -> Result<Value, Failure> {
        self.context
            .expect("Tensor operands retain their source context")
            .op(op, args)
    }
    pub(crate) fn truthy(&self) -> Result<bool, Failure> {
        Ok(match &self.value {
            Value::None => false,
            Value::Bool(n) => *n,
            Value::Int(n) => *n != 0,
            Value::UInt(n) => *n != 0,
            Value::Float(n) => *n != 0.,
            Value::Text(s) => !s.is_empty(),
            Value::List(v) | Value::Tuple(v) => !v.is_empty(),
            Value::Dict(v) => !v.is_empty(),
            Value::Tensor(_) | Value::Quantized(_) => match self.op(
                "aten::is_nonzero.default",
                std::slice::from_ref(&self.value),
            )? {
                Value::Bool(b) => b,
                _ => unreachable!(),
            },
            Value::DType(_) | Value::Device(_) => true,
        })
    }
    pub(crate) fn display(&self) -> Result<String, Failure> {
        if let Value::Text(text) = &self.value {
            return Ok(text.clone());
        }
        if let Some(context) = self.context {
            if let Value::Tensor(_) = &self.value {
                // Tensor.__format__('') formats item() for a plain scalar Tensor.
                let t = self
                    .value
                    .as_tensor()
                    .map_err(crate::registry_source::failure)?;
                if t.dim() == 0 {
                    let item = self.op("aten::item.default", std::slice::from_ref(&self.value))?;
                    return crate::python_value::display(&item)
                        .map_err(crate::registry_source::failure);
                }
            }
            return crate::python_value::repr_with_native(&self.value, &context.native)
                .map_err(crate::registry_source::failure);
        }
        crate::python_value::display(&self.value).map_err(crate::registry_source::failure)
    }
    pub(crate) fn len(&self) -> Result<usize, Failure> {
        if let Some(context) = self.context {
            return context.len();
        }
        match &self.value {
            Value::Tuple(v) | Value::List(v) => Ok(v.len()),
            Value::Text(s) => Ok(s.chars().count()),
            _ => Err(Failure::raised(
                "TypeError",
                format!(
                    "object of type '{}' has no len()",
                    crate::backend_triton_fp8::typename(&self.value)
                ),
            )),
        }
    }
    pub(crate) fn items(&self) -> Result<Vec<Self>, Failure> {
        let items = if let Some(context) = self.context {
            context.items()?
        } else {
            match &self.value {
                Value::Tuple(v) | Value::List(v) => v.clone(),
                Value::Text(s) => s.chars().map(|c| Value::Text(c.to_string())).collect(),
                _ => unreachable!("Only iterable source values reach unpacking"),
            }
        };
        Ok(items.into_iter().map(|value| self.with(value)).collect())
    }
    pub(crate) fn list_repr(&self) -> Result<String, Failure> {
        let value = Value::List(self.items()?.into_iter().map(|v| v.value).collect());
        match self.context {
            Some(c) => crate::python_value::repr_with_native(&value, &c.native),
            None => crate::python_value::repr(&value),
        }
        .map_err(crate::registry_source::failure)
    }
    /// Original operand direction is retained in both dispatch and TypeError.
    pub(crate) fn lt(&self, right: &Self) -> Result<bool, Failure> {
        let a = &self.value;
        let b = &right.value;
        let tensor = |v: &Value| matches!(v, Value::Tensor(_) | Value::Quantized(_));
        if tensor(a) || tensor(b) {
            let (ctx, op, args) = if tensor(a) {
                (
                    self,
                    if tensor(b) {
                        "aten::lt.Tensor"
                    } else {
                        "aten::lt.Scalar"
                    },
                    vec![a.clone(), b.clone()],
                )
            } else {
                (right, "aten::gt.Scalar", vec![b.clone(), a.clone()])
            };
            let value = ctx.op(op, &args)?;
            return ctx.with(value).truthy();
        }
        if let (Value::Text(a), Value::Text(b)) = (a, b) {
            return Ok(a < b);
        }
        let int = |v: &Value| match v {
            Value::Int(n) => Some(i128::from(*n)),
            Value::UInt(n) => Some(i128::from(*n)),
            Value::Bool(n) => Some(i128::from(*n)),
            _ => None,
        };
        let ordering = match (int(a), int(b), a, b) {
            (Some(a), Some(b), _, _) => Some(a.cmp(&b)),
            (Some(a), _, _, Value::Float(b)) => int_float_cmp(a, *b),
            (_, Some(b), Value::Float(a), _) => int_float_cmp(b, *a).map(Ordering::reverse),
            (_, _, Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            _ => {
                return Err(Failure::raised(
                    "TypeError",
                    format!(
                        "'<' not supported between instances of '{}' and '{}'",
                        crate::backend_triton_fp8::typename(a),
                        crate::backend_triton_fp8::typename(b)
                    ),
                ));
            }
        };
        Ok(ordering == Some(Ordering::Less))
    }
    pub(crate) fn gt(&self, right: &Self) -> Result<bool, Failure> {
        self_compare_gt(self, right)
    }
    pub(crate) fn integer(n: i64) -> Self {
        Self {
            value: Value::Int(n),
            context: None,
        }
    }
    pub(crate) fn float(n: f64) -> Self {
        Self {
            value: Value::Float(n),
            context: None,
        }
    }
    pub(crate) fn remainder64(&self) -> Result<bool, Failure> {
        Ok(match &self.value {
            Value::Int(n) => n % 64 != 0,
            Value::UInt(n) => n % 64 != 0,
            Value::Bool(b) => *b,
            Value::Float(n) => n % 64. != 0.,
            Value::Tensor(_) | Value::Quantized(_) => {
                return self
                    .with(self.op(
                        "aten::remainder.Scalar",
                        &[self.value.clone(), Value::Int(64)],
                    )?)
                    .truthy();
            }
            // Earlier comparisons reject all other source types.
            _ => unreachable!("Only numeric operands pass the token budget comparisons"),
        })
    }
}
fn int_float_cmp(a: i128, b: f64) -> Option<Ordering> {
    if b.is_nan() {
        return None;
    }
    if b >= i128::MAX as f64 {
        return Some(Ordering::Less);
    }
    if b < i128::MIN as f64 {
        return Some(Ordering::Greater);
    }
    let truncated = b as i128;
    Some(a.cmp(&truncated).then_with(|| {
        if b.fract() > 0. {
            Ordering::Less
        } else if b.fract() < 0. {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }))
}

fn self_compare_gt(a: &Operand, b: &Operand) -> Result<bool, Failure> {
    b.lt(a).map_err(|failure| {
        if failure.raised.as_deref() == Some("TypeError")
            && failure
                .reason
                .starts_with("'<' not supported between instances of ")
        {
            Failure::raised(
                "TypeError",
                format!(
                    "'>' not supported between instances of '{}' and '{}'",
                    crate::backend_triton_fp8::typename(&a.value),
                    crate::backend_triton_fp8::typename(&b.value)
                ),
            )
        } else {
            failure
        }
    })
}
