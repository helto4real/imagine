//! Shared latent shape containers. Source-backed views retain their actual
//! option list/tuple rather than copying it at conditioning extraction.
use super::{WindowList, error, gap};
use crate::{
    execution_value::{TORCH_SIZE_CLASS, Value as V},
    native::Error,
    patching::options::{MetadataObject, MetadataValue, Options},
};
use std::{any::Any, cell::RefCell, rc::Rc};
fn typed<T: Any>(value: &dyn Any) -> Option<&T> {
    value.downcast_ref().or_else(|| MetadataObject::get(value))
}
#[derive(Clone)]
pub enum LatentShape {
    Size(Rc<Vec<i64>>),
    Tuple(Rc<Vec<i64>>),
    List(WindowList<i64>),
    Source(Options),
}
fn integer(value: &Options) -> Result<i64, Error> {
    match value {
        Options::Atom(V::Int(v)) => Ok(*v),
        Options::Atom(V::Bool(v)) => Ok(i64::from(*v)),
        Options::Atom(V::UInt(v)) => {
            i64::try_from(*v).map_err(|_| gap("latent shape integer exceeds i64"))
        }
        _ => Err(gap("latent shape requires its original integral binding")),
    }
}
fn sequence(value: &Options) -> Result<Vec<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok((**v).clone()),
        _ => Err(gap("latent shape requires its original sequence binding")),
    }
}
fn position(index: i64, len: usize, kind: &str) -> Result<usize, Error> {
    let index = if index < 0 {
        len as i128 + i128::from(index)
    } else {
        i128::from(index)
    };
    usize::try_from(index)
        .ok()
        .filter(|i| *i < len)
        .ok_or_else(|| error("IndexError", format!("{kind} index out of range")))
}
impl LatentShape {
    pub fn as_options(&self) -> Options {
        match self {
            Self::Source(value) => value.clone(),
            Self::Tuple(value) if value.is_empty() => Options::Tuple(Rc::new(vec![])),
            _ => MetadataObject::wrap(self.clone()),
        }
    }
    pub fn from_options(value: Options) -> Result<Self, Error> {
        match value {
            v @ (Options::List(_) | Options::Tuple(_)) => Ok(Self::Source(v)),
            Options::Object(v) => typed::<Self>(v.as_ref())
                .cloned()
                .ok_or_else(|| gap("latent shape requires its original object binding")),
            _ => Err(gap("latent shape requires its original sequence binding")),
        }
    }
    pub fn class_name(&self) -> &str {
        match self {
            Self::Size(_) => "Size",
            Self::Tuple(_) => "tuple",
            Self::List(_) => "list",
            Self::Source(Options::Tuple(_)) => "tuple",
            Self::Source(_) => "list",
        }
    }
    pub fn values(&self) -> Result<Vec<i64>, Error> {
        match self {
            Self::Size(v) | Self::Tuple(v) => Ok((**v).clone()),
            Self::List(v) => Ok(v.borrow().clone()),
            Self::Source(v) => sequence(v)?.iter().map(integer).collect(),
        }
    }
    pub fn dimension(&self, index: i64) -> Result<i64, Error> {
        match self {
            Self::Source(v) => {
                let values = sequence(v)?;
                let index = position(index, values.len(), self.class_name())?;
                integer(&values[index])
            }
            _ => {
                let values = self.values()?;
                let index = position(
                    index,
                    values.len(),
                    if matches!(self, Self::Size(_)) {
                        "tuple"
                    } else {
                        self.class_name()
                    },
                )?;
                Ok(values[index])
            }
        }
    }
    pub fn same_object(&self, other: &Self) -> bool {
        let empty = |value: &Self| {
            matches!(value,Self::Tuple(v)if v.is_empty())
                || matches!(value,Self::Source(Options::Tuple(v))if v.is_empty())
        };
        if empty(self) && empty(other) {
            return true;
        }
        match (self, other) {
            (Self::Size(a), Self::Size(b)) | (Self::Tuple(a), Self::Tuple(b)) => Rc::ptr_eq(a, b),
            (Self::List(a), Self::List(b)) => Rc::ptr_eq(a, b),
            (Self::Source(a), Self::Source(b)) => a.same_object(b),
            _ => false,
        }
    }
}
#[derive(Clone)]
pub enum LatentShapes {
    Native(WindowList<LatentShape>),
    Source(Options),
}
impl From<WindowList<LatentShape>> for LatentShapes {
    fn from(v: WindowList<LatentShape>) -> Self {
        Self::Native(v)
    }
}
impl LatentShapes {
    pub fn as_options(&self) -> Options {
        match self {
            Self::Source(value) => value.clone(),
            _ => MetadataObject::wrap(self.clone()),
        }
    }
    pub fn new(values: Vec<LatentShape>) -> Self {
        Self::Native(Rc::new(RefCell::new(values)))
    }
    pub fn from_options(value: &Options) -> Result<Self, Error> {
        match value {
            Options::List(_) | Options::Tuple(_) => Ok(Self::Source(value.clone())),
            Options::Object(v) if typed::<Self>(v.as_ref()).is_some() => {
                Ok(Self::Source(value.clone()))
            }
            Options::Object(_) => Err(gap("latent shapes require their original object binding")),
            _ => Err(gap("latent shapes require their original sequence binding")),
        }
    }
    pub fn len(&self) -> Result<usize, Error> {
        match self {
            Self::Native(v) => Ok(v.borrow().len()),
            Self::Source(Options::List(v)) => Ok(v.borrow().len()),
            Self::Source(Options::Tuple(v)) => Ok(v.len()),
            Self::Source(Options::Object(v)) => typed::<Self>(v.as_ref())
                .ok_or_else(|| gap("latent shapes require their original object binding"))?
                .len(),
            Self::Source(_) => Err(gap("latent shapes require their original sequence binding")),
        }
    }
    pub fn is_empty(&self) -> Result<bool, Error> {
        self.len().map(|v| v == 0)
    }
    pub fn next(&self, index: usize) -> Result<Option<LatentShape>, Error> {
        match self {
            Self::Native(v) => Ok(v.borrow().get(index).cloned()),
            Self::Source(Options::List(v)) => v
                .borrow()
                .get(index)
                .cloned()
                .map(LatentShape::from_options)
                .transpose(),
            Self::Source(Options::Tuple(v)) => v
                .get(index)
                .cloned()
                .map(LatentShape::from_options)
                .transpose(),
            Self::Source(Options::Object(v)) => typed::<Self>(v.as_ref())
                .ok_or_else(|| gap("latent shapes require their original object binding"))?
                .next(index),
            Self::Source(_) => Err(gap("latent shapes require their original sequence binding")),
        }
    }
    pub fn get(&self, index: usize) -> Result<LatentShape, Error> {
        self.next(index)?.ok_or_else(|| {
            error(
                "IndexError",
                if matches!(self, Self::Source(Options::Tuple(_))) {
                    "tuple index out of range"
                } else {
                    "list index out of range"
                },
            )
        })
    }
    pub fn entries(&self) -> Result<Vec<LatentShape>, Error> {
        let mut values = vec![];
        while let Some(value) = self.next(values.len())? {
            values.push(value);
        }
        Ok(values)
    }
    pub fn same_object(&self, other: &Self) -> bool {
        if let Self::Source(Options::Object(v)) = self
            && let Some(value) = typed::<Self>(v.as_ref())
        {
            return value.same_object(other);
        }
        if let Self::Source(Options::Object(v)) = other
            && let Some(value) = typed::<Self>(v.as_ref())
        {
            return self.same_object(value);
        }
        match (self, other) {
            (Self::Native(a), Self::Native(b)) => Rc::ptr_eq(a, b),
            (Self::Source(a), Self::Source(b)) => a.same_object(b),
            _ => false,
        }
    }
}
impl MetadataValue for LatentShape {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &str {
        if matches!(self, Self::Size(_)) {
            "torch.Size"
        } else {
            self.class_name()
        }
    }
    fn truth(&self) -> Result<bool, Error> {
        match self {
            Self::Size(v) | Self::Tuple(v) => Ok(!v.is_empty()),
            Self::List(v) => Ok(!v.borrow().is_empty()),
            Self::Source(Options::List(v)) => Ok(!v.borrow().is_empty()),
            Self::Source(Options::Tuple(v)) => Ok(!v.is_empty()),
            Self::Source(_) => Err(gap("latent shape truth requires its source binding")),
        }
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        typed::<Self>(other).is_some_and(|v| self.same_object(v))
    }
    fn snapshot(&self) -> Result<V, Error> {
        Ok(match self {
            Self::Size(v) => V::named_tuple(
                &TORCH_SIZE_CLASS,
                v.iter().map(|n| ("", V::Int(*n))).collect(),
            ),
            Self::Tuple(v) => V::Tuple(v.iter().map(|n| V::Int(*n)).collect()),
            Self::List(v) => V::List(v.borrow().iter().map(|n| V::Int(*n)).collect()),
            Self::Source(v) => v.snapshot()?,
        })
    }
}
impl MetadataValue for LatentShapes {
    fn truth(&self) -> Result<bool, Error> {
        self.len().map(|n| n != 0)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &str {
        match self {
            Self::Source(v) => v.typename(),
            Self::Native(_) => "list",
        }
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        typed::<Self>(other).is_some_and(|v| self.same_object(v))
    }
    fn snapshot(&self) -> Result<V, Error> {
        match self {
            Self::Native(v) => Ok(V::List(
                v.borrow()
                    .iter()
                    .map(MetadataValue::snapshot)
                    .collect::<Result<_, _>>()?,
            )),
            Self::Source(v) => v.snapshot(),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_size_metadata_representation_matches_torch_tuple_subtype() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../fixtures/shared-context-shape-metadata.json"
        ))
        .unwrap();
        for case in fixture["representations"].as_array().unwrap() {
            let dimensions = case["dims"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let value = LatentShape::Size(Rc::new(dimensions)).as_options();
            let error = crate::context_windows::ConditionAttributes::cond(
                &crate::context_windows::BuiltinConditionAttributes,
                &value,
            )
            .err()
            .unwrap();
            match error {
                Error::ReferenceException { class, message } => {
                    assert_eq!(class, "AttributeError");
                    assert_eq!(message, case["attribute_error"].as_str().unwrap());
                }
                error => panic!("unexpected Size attribute error {error}"),
            }
            assert_eq!(
                crate::python_value::repr(&value.snapshot().unwrap()).unwrap(),
                case["repr"].as_str().unwrap()
            );
        }
    }
}
