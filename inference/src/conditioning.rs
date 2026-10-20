//! Shared COND values and area/multiplier preparation from comfy.conds/samplers.
use crate::patching::options::Options;
use crate::{execution_value::Value, native::Error};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Regular,
    NoiseShape,
    CrossAttention,
    Constant,
    List,
}
#[derive(Clone)]
pub struct Condition {
    pub kind: Kind,
    data: Rc<RefCell<Options>>,
}
fn exception(class: &'static str, message: &str) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
pub fn repeat_to_batch_size(t: &Tensor, batch: i64, dim: usize) -> Result<Tensor, Error> {
    let shape = t.size();
    let n = *shape
        .get(dim)
        .ok_or_else(|| exception("IndexError", "tuple index out of range"))?;
    if n > batch {
        return Ok(t.f_narrow(dim as i64, 0, batch)?);
    }
    if n < batch {
        if n == 0 {
            return Err(exception("ZeroDivisionError", "division by zero"));
        }
        let mut repeats = vec![1; t.dim()];
        repeats[dim] = (batch as f64 / n as f64).ceil() as i64;
        return Ok(t.f_repeat(&repeats)?.f_narrow(dim as i64, 0, batch)?);
    }
    Ok(t.shallow_clone())
}
pub fn equal_value(a: &Value, b: &Value, warning: &mut dyn FnMut(&str)) -> Result<Value, Error> {
    use Value::*;
    if matches!(a, Tensor(_)) != matches!(b, Tensor(_)) {
        let (tensor, other) = if matches!(a, Tensor(_)) {
            (a.as_tensor()?, b)
        } else {
            (b.as_tensor()?, a)
        };
        let result = match other {
            Int(n) => Some(tensor.f_eq(*n)),
            UInt(n) if *n <= i64::MAX as u64 => Some(tensor.f_eq(*n as i64)),
            UInt(_) => {
                warning("comparison issue with COND");
                return Ok(Bool(false));
            }
            Float(n) => Some(tensor.f_eq(*n)),
            Bool(n) => Some(tensor.f_eq(i64::from(*n))),
            _ => std::option::Option::None,
        };
        return Ok(match result {
            Some(Ok(tensor)) => Value::tensor(tensor),
            Some(Err(_)) => {
                warning("comparison issue with COND");
                Bool(false)
            }
            std::option::Option::None => Bool(false),
        });
    }
    is_equal_inner(a, b, warning).map(Bool)
}
fn equal_truth(value: Value) -> Result<bool, Error> {
    match value {
        Value::Bool(value) => Ok(value),
        Value::Tensor(_) => Ok(value.as_tensor()?.f_is_nonzero()?),
        _ => Err(Error::Native("Invalid COND equality result".into())),
    }
}
pub fn is_equal(a: &Value, b: &Value) -> Result<bool, Error> {
    equal_truth(equal_value(a, b, &mut |_| {})?)
}
fn is_equal_inner(a: &Value, b: &Value, warning: &mut dyn FnMut(&str)) -> Result<bool, Error> {
    use Value::*;
    Ok(match (a, b) {
        (Tensor(_), Tensor(_)) => a.as_tensor()?.f_equal(&b.as_tensor()?)?,
        (NamedTuple(a), NamedTuple(b)) => {
            std::ptr::eq(a.class, b.class) && a.fields.len() == b.fields.len() && {
                let mut equal = true;
                for ((_, a), (_, b)) in a.fields.iter().zip(&b.fields) {
                    if !equal_truth(equal_value(a, b, warning)?)? {
                        equal = false;
                        break;
                    }
                }
                equal
            }
        }
        (Dict(a), Dict(b)) => {
            a.len() == b.len() && {
                let mut equal = true;
                for (key, value) in a {
                    equal &= if let Some((_, other)) = b.iter().find(|(k, _)| k == key) {
                        equal_truth(equal_value(value, other, warning)?)?
                    } else {
                        false
                    };
                    if !equal {
                        break;
                    }
                }
                equal
            }
        }
        (List(a), List(b)) | (Tuple(a), Tuple(b)) => {
            let mut equal = a.len() == b.len();
            if equal {
                for (a, b) in a.iter().zip(b) {
                    if !equal_truth(equal_value(a, b, warning)?)? {
                        equal = false;
                        break;
                    }
                }
            }
            equal
        }
        (None, None) => true,
        (Text(a), Text(b)) => a == b,
        (DType(a), DType(b)) => a == b,
        (Device(a), Device(b)) => a == b,
        (Bool(a), Bool(b)) => a == b,
        (Int(a), Int(b)) => a == b,
        (UInt(a), UInt(b)) => a == b,
        (Int(a), UInt(b)) | (UInt(b), Int(a)) => *a >= 0 && *a as u64 == *b,
        (Float(a), Float(b)) => a == b,
        (Int(a), Float(b)) | (Float(b), Int(a)) => {
            b.is_finite()
                && b.fract() == 0.
                && *b >= i64::MIN as f64
                && *b < 9223372036854775808.
                && *a == *b as i64
        }
        (UInt(a), Float(b)) | (Float(b), UInt(a)) => {
            b.is_finite()
                && b.fract() == 0.
                && *b >= 0.
                && *b < 18446744073709551616.
                && *a == *b as u64
        }
        (Bool(a), Int(b)) | (Int(b), Bool(a)) => i64::from(*a) == *b,
        (Bool(a), UInt(b)) | (UInt(b), Bool(a)) => u64::from(*a) == *b,
        (Bool(a), Float(b)) | (Float(b), Bool(a)) => f64::from(u8::from(*a)) == *b,
        _ => false,
    })
}
fn lcm(a: i64, b: i64) -> i64 {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    if x == 0 { 0 } else { (a / x * b).abs() }
}
impl Condition {
    pub fn new(kind: Kind, value: Value) -> Self {
        Self::from_options(kind, Options::from_value(value))
    }
    pub fn from_options(kind: Kind, data: Options) -> Self {
        Self {
            kind,
            data: Rc::new(RefCell::new(data)),
        }
    }
    pub fn copy_with(&self, value: Value) -> Self {
        Self::new(self.kind, value)
    }
    pub fn copy_with_options(&self, data: Options) -> Self {
        Self::from_options(self.kind, data)
    }
    /// Clones of a COND handle refer to the same original wrapper object.
    pub fn same_object(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
    pub fn set_value(&self, data: Options) {
        let old = std::mem::replace(&mut *self.data.borrow_mut(), data);
        drop(old);
    }
    pub fn source_value(&self) -> Options {
        self.data.borrow().clone()
    }
    pub fn value(&self) -> Result<Value, Error> {
        self.data.borrow().snapshot()
    }
    pub fn tensor(kind: Kind, value: Tensor) -> Self {
        Self::new(kind, Value::tensor(value))
    }
    fn tensors(&self) -> Result<Vec<Tensor>, Error> {
        let Value::List(values) = self.value()? else {
            return Err(exception(
                "TypeError",
                "Expected a list of conditioning tensors",
            ));
        };
        values.iter().map(Value::as_tensor).collect()
    }
    pub fn process(&self, batch: i64, area: Option<&[i64]>) -> Result<Self, Error> {
        let data = match self.kind {
            Kind::Constant => self.source_value(),
            Kind::List => {
                let Options::List(values) = self.source_value() else {
                    return Err(exception(
                        "TypeError",
                        "Expected a list of conditioning tensors",
                    ));
                };
                let out = values
                    .borrow()
                    .iter()
                    .map(|value| {
                        let tensor = value.snapshot()?.as_tensor()?;
                        if tensor.size()[0] == batch {
                            Ok(value.clone())
                        } else {
                            Ok(Options::from_value(Value::tensor(repeat_to_batch_size(
                                &tensor, batch, 0,
                            )?)))
                        }
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                Options::List(Rc::new(RefCell::new(out)))
            }
            _ => {
                let mut tensor = self.value()?.as_tensor()?;
                let cropped = self.kind == Kind::NoiseShape && area.is_some_and(|a| !a.is_empty());
                if self.kind == Kind::NoiseShape
                    && let Some(area) = area
                {
                    let n = area.len() / 2;
                    for i in 0..n {
                        tensor = tensor.f_narrow(i as i64 + 2, area[i + n], area[i])?;
                    }
                }
                if !cropped && tensor.size()[0] == batch {
                    self.source_value()
                } else {
                    Options::from_value(Value::tensor(repeat_to_batch_size(&tensor, batch, 0)?))
                }
            }
        };
        Ok(Self::from_options(self.kind, data))
    }
    pub fn concat_options(&self, others: &[&Self]) -> Result<Options, Error> {
        if self.kind == Kind::Constant {
            Ok(self.source_value())
        } else {
            Ok(Options::from_value(self.concat(others)?))
        }
    }
    pub fn can_concat(&self, other: &Self, warning: &mut dyn FnMut(&str)) -> Result<bool, Error> {
        if self.kind == Kind::Constant {
            return is_equal(&self.value()?, &other.value()?);
        }
        if self.kind == Kind::List {
            let a = self.tensors()?;
            let b = other.tensors()?;
            return Ok(a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.size() == b.size()));
        }
        let a = self.value()?.as_tensor()?;
        let b = other.value()?.as_tensor()?;
        let (s1, s2) = (a.size(), b.size());
        if s1 != s2 {
            if self.kind != Kind::CrossAttention {
                return Ok(false);
            }
            if s1.first() != s2.first() || s1.get(2) != s2.get(2) {
                return Ok(false);
            }
            let m = s1[1].min(s2[1]);
            if m == 0 {
                return Err(exception(
                    "ZeroDivisionError",
                    "integer division or modulo by zero",
                ));
            }
            if lcm(s1[1], s2[1]) / m > 4 {
                return Ok(false);
            }
        }
        if a.device() != b.device() {
            warning(if self.kind == Kind::CrossAttention {
                "WARNING: conds not on same device: skipping concat."
            } else {
                "WARNING: conds not on same device, skipping concat."
            });
            return Ok(false);
        }
        Ok(true)
    }
    pub fn concat(&self, others: &[&Self]) -> Result<Value, Error> {
        if self.kind == Kind::Constant {
            return self.value();
        }
        if self.kind == Kind::List {
            let source = self.tensors()?;
            let others = others
                .iter()
                .map(|c| c.tensors())
                .collect::<Result<Vec<_>, _>>()?;
            let mut out = Vec::new();
            for (i, tensor) in source.iter().enumerate() {
                let mut tensors = vec![tensor.shallow_clone()];
                for other in &others {
                    tensors.push(
                        other
                            .get(i)
                            .ok_or_else(|| exception("IndexError", "list index out of range"))?
                            .shallow_clone(),
                    );
                }
                out.push(Value::tensor(Tensor::f_cat(&tensors, 0)?));
            }
            return Ok(Value::List(out));
        }
        let mut tensors = vec![self.value()?.as_tensor()?];
        for c in others {
            tensors.push(c.value()?.as_tensor()?);
        }
        if self.kind == Kind::CrossAttention {
            let max = tensors
                .iter()
                .fold(tensors[0].size()[1], |m, t| lcm(m, t.size()[1]));
            for t in &mut tensors {
                if t.size()[1] < max {
                    if t.size()[1] == 0 {
                        return Err(exception(
                            "ZeroDivisionError",
                            "integer division or modulo by zero",
                        ));
                    }
                    *t = t.f_repeat([1, max / t.size()[1], 1])?;
                }
            }
        }
        Ok(Value::tensor(Tensor::f_cat(&tensors, 0)?))
    }
    pub fn size(&self) -> Result<Value, Error> {
        if self.kind == Kind::Constant {
            return Ok(Value::List(vec![Value::Int(1)]));
        }
        if self.kind != Kind::List {
            return Ok(Value::List(
                self.value()?
                    .as_tensor()?
                    .size()
                    .into_iter()
                    .map(Value::Int)
                    .collect(),
            ));
        }
        let tensors = self.tensors()?;
        let mut elements = 0;
        let mut channels = Value::Int(1);
        for t in tensors {
            let size = t.size();
            elements += size.iter().product::<i64>();
            channels = if size.len() > 1 {
                Value::Int(size[1])
            } else {
                Value::tensor(t)
            };
        }
        let last = match &channels {
            Value::Int(c) => {
                if *c == 0 {
                    return Err(exception(
                        "ZeroDivisionError",
                        "integer division or modulo by zero",
                    ));
                }
                Value::Int(elements.div_euclid(*c))
            }
            _ => {
                let t = channels.as_tensor()?;
                Value::tensor(
                    Tensor::f_full([], elements, (t.kind(), t.device()))?.f_floor_divide(&t)?,
                )
            }
        };
        Ok(Value::List(vec![Value::Int(1), channels, last]))
    }
}

pub fn add_area_dims(area: &[i64], dims: usize) -> Vec<i64> {
    let mut area = area.to_vec();
    while area.len() / 2 < dims {
        let n = area.len() / 2;
        area.insert(n, 0);
        area.insert(0, 2_147_483_648);
    }
    area
}
/// Tensor-only part of get_area_and_mult. Hook/control/GLIGEN ownership is
/// supplied by the calling conditioning executor, not discarded here.
pub struct AreaCondition {
    pub area: Option<Vec<i64>>,
    pub strength: f64,
    pub mask: Option<Tensor>,
    pub mask_strength: f64,
    pub timestep_start: Option<f64>,
    pub timestep_end: Option<f64>,
}
pub struct PreparedArea {
    pub input: Tensor,
    pub multiplier: Tensor,
    pub area: Option<Vec<i64>>,
}
impl AreaCondition {
    pub fn prepare(
        &self,
        input: &Tensor,
        timestep: &Tensor,
    ) -> Result<Option<PreparedArea>, Error> {
        let time = timestep.f_get(0)?.double_value(&[]);
        if self.timestep_start.is_some_and(|start| time > start)
            || self.timestep_end.is_some_and(|end| time < end)
        {
            return Ok(None);
        }
        let dims = input.dim() - 2;
        let mut area = self.area.as_ref().map(|a| add_area_dims(a, dims));
        if let Some(a) = &mut area
            && a.len() / 2 > dims
        {
            *a = a[..dims]
                .iter()
                .chain(&a[a.len() / 2..a.len() / 2 + dims])
                .copied()
                .collect();
        }
        let mut x = input.shallow_clone();
        if let Some(a) = &mut area {
            for i in 0..dims {
                a[i] = (x.size()[i + 2] - a[dims + i]).min(a[i]);
                x = x.f_narrow(i as i64 + 2, a[dims + i], a[i])?;
            }
        }
        let mask = if let Some(mask) = &self.mask {
            let mut mask = mask.f_narrow(0, 0, x.size()[0].min(mask.size()[0]))?;
            if let Some(a) = &area {
                for i in 0..dims {
                    mask = mask.f_narrow(i as i64 + 1, a[dims + i], a[i])?;
                }
            }
            mask = mask.f_mul_scalar(self.mask_strength)?;
            if mask.size()[0] == 0 {
                return Err(exception(
                    "ZeroDivisionError",
                    "integer division or modulo by zero",
                ));
            }
            let mut repeats = vec![x.size()[0].div_euclid(mask.size()[0]), x.size()[1]];
            repeats.extend(vec![1; mask.dim() - 1]);
            mask.f_unsqueeze(1)?.f_repeat(&repeats)?
        } else {
            x.f_ones_like()?
        };
        let multiplier = mask.f_mul_scalar(self.strength)?;
        if self.mask.is_none()
            && let Some(a) = &area
        {
            for i in 0..dims {
                let rr = 8.min(multiplier.size()[i + 2].div_euclid(4));
                // Python __imul__ dispatches mul_.Tensor with a CPU scalar.
                // mul_.Scalar rounds its scalar to BF16 early on CPU and is
                // observably different, despite having the same schema math.
                if a[dims + i] != 0 {
                    for t in 0..rr {
                        let scalar = Tensor::from((1. / rr as f64) * (t + 1) as f64);
                        let _ = multiplier.f_narrow(i as i64 + 2, t, 1)?.f_mul_(&scalar)?;
                    }
                }
                if a[i] + a[dims + i] < input.size()[i + 2] {
                    for t in 0..rr {
                        let scalar = Tensor::from((1. / rr as f64) * (t + 1) as f64);
                        let _ = multiplier
                            .f_narrow(i as i64 + 2, a[i] - 1 - t, 1)?
                            .f_mul_(&scalar)?;
                    }
                }
            }
        }
        Ok(Some(PreparedArea {
            input: x,
            multiplier,
            area,
        }))
    }
}
