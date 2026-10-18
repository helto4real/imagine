//! Python host arithmetic retained for dynamic eager-NA geometry values.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
static IDENTITIES: AtomicU64 = AtomicU64::new(1);
#[derive(Clone)]
pub(super) enum Number {
    Int(i128),
    Float(f64, u64),
    Tensor(Value),
    Other(Value),
}
impl Number {
    pub(super) fn float(n: f64) -> Self {
        Self::Float(n, IDENTITIES.fetch_add(1, Ordering::Relaxed))
    }
    pub(super) fn from_value(v: &Value) -> Self {
        match v {
            Value::Int(n) => Self::Int(i128::from(*n)),
            Value::UInt(n) => Self::Int(i128::from(*n)),
            Value::Bool(n) => Self::Int(i128::from(*n)),
            Value::Float(n) => Self::float(*n),
            Value::Tensor(_) | Value::Quantized(_) => Self::Tensor(v.clone()),
            _ => Self::Other(v.clone()),
        }
    }
    pub(super) fn value(&self) -> Result<Value, Error> {
        Ok(match self {
            Self::Int(n) => {
                if let Ok(n) = i64::try_from(*n) {
                    Value::Int(n)
                } else if let Ok(n) = u64::try_from(*n) {
                    Value::UInt(n)
                } else {
                    return Err(error("OverflowError", "int too big to convert"));
                }
            }
            Self::Float(n, _) => Value::Float(*n),
            Self::Tensor(v) | Self::Other(v) => v.clone(),
        })
    }
    fn fail(f: crate::registry::Failure) -> Error {
        Error::ReferenceException {
            class: match f.raised.as_deref() {
                Some("TypeError") => "TypeError",
                Some("ValueError") => "ValueError",
                Some("IndexError") => "IndexError",
                Some("OverflowError") => "OverflowError",
                _ => "RuntimeError",
            },
            message: f.reason,
        }
    }
    pub(super) fn lt(&self, rt: &Runtime, rhs: &Self) -> Result<bool, Error> {
        if let (Self::Int(a), Self::Int(b)) = (self, rhs) {
            return Ok(a < b);
        }
        // The source-value comparator keeps Python int/float ordering exact and
        // dispatches Tensor comparisons through the original boxed ATen op.
        let a = self.value()?.metadata_for_call(rt)?;
        let b = rhs.value()?.metadata_for_call(rt)?;
        crate::registry_value_ops::Operand::argument(&a)
            .map_err(Self::fail)?
            .lt(&crate::registry_value_ops::Operand::argument(&b).map_err(Self::fail)?)
            .map_err(Self::fail)
    }
    pub(super) fn min(&self, rt: &Runtime, rhs: &Self) -> Result<Self, Error> {
        Ok(if rhs.lt(rt, self)? {
            rhs.clone()
        } else {
            self.clone()
        })
    }
    pub(super) fn max(&self, rt: &Runtime, rhs: &Self) -> Result<Self, Error> {
        Ok(if self.lt(rt, rhs)? {
            rhs.clone()
        } else {
            self.clone()
        })
    }
    fn typename(&self) -> &'static str {
        match self {
            Self::Int(_) => "int",
            Self::Float(..) => "float",
            Self::Tensor(v) | Self::Other(v) => crate::backend_triton_fp8::typename(v),
        }
    }
    pub(super) fn binary(&self, rt: &Runtime, rhs: &Self, op: &str) -> Result<Self, Error> {
        if let (Self::Int(a), Self::Int(b)) = (self, rhs) {
            return Ok(match op {
                "add" => Self::Int(a + b),
                "sub" => Self::Int(a - b),
                "mul" => Self::Int(a * b),
                "floordiv" => {
                    if *b == 0 {
                        return Err(error(
                            "ZeroDivisionError",
                            "integer division or modulo by zero",
                        ));
                    }
                    Self::Int(a.div_euclid(*b) - i128::from(*b < 0 && a % *b != 0))
                }
                "div" => {
                    if *b == 0 {
                        return Err(error("ZeroDivisionError", "division by zero"));
                    }
                    Self::float(*a as f64 / *b as f64)
                }
                _ => unreachable!(),
            });
        }
        if matches!(self, Self::Tensor(_)) || matches!(rhs, Self::Tensor(_)) {
            let a = self.value()?;
            let b = rhs.value()?;
            let both = matches!(self, Self::Tensor(_)) && matches!(rhs, Self::Tensor(_));
            let (name, args) = if matches!(self, Self::Tensor(_)) {
                (
                    format!(
                        "aten::{}.{}",
                        if op == "floordiv" { "floor_divide" } else { op },
                        if both { "Tensor" } else { "Scalar" }
                    ),
                    vec![a, b],
                )
            } else {
                match op {
                    "add" | "mul" => (format!("aten::{op}.Scalar"), vec![b, a]),
                    "sub" => ("aten::rsub.Scalar".into(), vec![b, a]),
                    "div" => {
                        let inverse = unsafe { rt.op("aten::reciprocal.default", &[b], &vec![])? };
                        return Ok(Self::from_value(&unsafe {
                            rt.op("aten::mul.Scalar", &[inverse, a], &vec![])?
                        }));
                    }
                    _ => unreachable!("NA only floors the kernel on the left"),
                }
            };
            return Ok(Self::from_value(&unsafe { rt.op(&name, &args, &vec![])? }));
        }
        let float = |v: &Self| match v {
            Self::Int(n) => Some(*n as f64),
            Self::Float(n, _) => Some(*n),
            _ => None,
        };
        if let (Some(a), Some(b)) = (float(self), float(rhs)) {
            return Ok(Self::float(match op {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "div" => {
                    if b == 0. {
                        return Err(error("ZeroDivisionError", "float division by zero"));
                    }
                    a / b
                }
                "floordiv" => {
                    if b == 0. {
                        return Err(error("ZeroDivisionError", "float floor division by zero"));
                    }
                    let mut modulo = a % b;
                    let mut div = (a - modulo) / b;
                    if modulo != 0. && (b < 0.) != (modulo < 0.) {
                        modulo += b;
                        div -= 1.;
                    }
                    let _ = modulo;
                    if div != 0. {
                        let floor = div.floor();
                        if div - floor > 0.5 { floor + 1. } else { floor }
                    } else {
                        0_f64.copysign(a / b)
                    }
                }
                _ => unreachable!(),
            }));
        }
        Err(type_error(format!(
            "unsupported operand type(s) for {}: '{}' and '{}'",
            match op {
                "add" => "+",
                "sub" => "-",
                "mul" => "*",
                "div" => "/",
                "floordiv" => "//",
                _ => unreachable!(),
            },
            self.typename(),
            rhs.typename()
        )))
    }
    pub(super) fn index(&self) -> Result<i64, Error> {
        match self {
            Self::Int(n) => Ok((*n).clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64),
            Self::Tensor(v) => {
                let t = v.as_tensor()?;
                if t.numel() != 1
                    || !matches!(
                        t.kind(),
                        Kind::Bool
                            | Kind::Int8
                            | Kind::Uint8
                            | Kind::Int16
                            | Kind::Int
                            | Kind::Int64
                    )
                {
                    return Err(type_error(
                        "only integer tensors of a single element can be converted to an index",
                    ));
                }
                Ok(t.f_reshape([])?.f_int64_value(&[])?)
            }
            _ => Err(type_error(
                "slice indices must be integers or None or have an __index__ method",
            )),
        }
    }
    pub(super) fn key(&self) -> String {
        match self {
            Self::Int(n) => format!("n:{n}"),
            Self::Float(n, id) if n.is_nan() => format!("nan:{id}"),
            Self::Float(n, _) if n.is_finite() && n.fract() == 0. => {
                format!("n:{:.0}", if *n == 0. { 0. } else { *n })
            }
            Self::Float(n, _) => format!("f:{:016x}", n.to_bits()),
            Self::Tensor(Value::Tensor(t)) => format!("t:{:p}", std::sync::Arc::as_ptr(t)),
            Self::Tensor(Value::Quantized(t)) => format!("q:{:p}", std::sync::Arc::as_ptr(t)),
            _ => unreachable!("Numeric geometry key"),
        }
    }
}
pub(super) fn array(rt: &Runtime, values: &[Number], device: tch::Device) -> Result<Tensor, Error> {
    let mut kind = None;
    for value in values {
        let next = match value {
            Number::Int(_) => Kind::Int64,
            Number::Float(..) => rt.native.default_dtype()?,
            Number::Tensor(v) => v.as_tensor()?.kind(),
            _ => unreachable!("Numeric bounds"),
        };
        kind = Some(if let Some(k) = kind {
            crate::boxed_aten::promote_types(&rt.native, k, next)?
        } else {
            next
        });
    }
    let kind = kind.unwrap_or(rt.native.default_dtype()?);
    let tensors = values
        .iter()
        .map(|n| -> Result<Tensor, Error> {
            Ok(match n {
                Number::Int(n) => Tensor::f_from_slice(&[i64::try_from(*n)
                    .map_err(|_| error("RuntimeError", "Overflow when unpacking long"))?])?,
                Number::Float(n, _) => Tensor::f_from_slice(&[*n])?,
                Number::Tensor(v) => {
                    let t = v.as_tensor()?;
                    if matches!(
                        t.kind(),
                        Kind::Bool
                            | Kind::Int8
                            | Kind::Uint8
                            | Kind::Int16
                            | Kind::Int
                            | Kind::Int64
                    ) {
                        Tensor::f_from_slice(&[t.f_int64_value(&[])?])?
                    } else {
                        Tensor::f_from_slice(&[rt.native.item_double(&t)?])?
                    }
                }
                _ => unreachable!(),
            }
            .f_to_kind(kind)?
            .f_reshape([])?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tensors.is_empty() {
        return Ok(Tensor::f_empty([0], (kind, device))?);
    }
    Ok(Tensor::f_stack(&tensors, 0)?.f_to_device(device)?)
}
