//! Source live sequence iterators shared by model-owner operations.
use super::{
    exception,
    options::{GeneralMapping, KeysIterator, Options},
};
use crate::{execution_value::Value as V, native::Error};
pub(crate) enum SequenceIterator {
    List(Options, usize),
    Keys(KeysIterator),
    Text(Vec<char>, usize),
}
impl SequenceIterator {
    pub(crate) fn new(value: &Options) -> Result<Self, Error> {
        Self::for_operation(value, "model injection sequence")
    }
    pub(crate) fn for_operation(value: &Options, operation: &str) -> Result<Self, Error> {
        Ok(match value {
            Options::List(_) | Options::Tuple(_) => Self::List(value.clone(), 0),
            _ if GeneralMapping::get(value).is_some() => Self::Keys(value.keys_iterator()?),
            Options::Dict(_) => Self::Keys(value.keys_iterator()?),
            Options::Atom(V::Text(v)) => Self::Text(v.chars().collect(), 0),
            Options::Object(_)
            | Options::Atom(V::Tensor(_) | V::Quantized(_) | V::NamedTuple(_)) => {
                return Err(super::gap(format!(
                    "{operation} requires its original iterator binding"
                )));
            }
            _ => {
                return Err(exception(
                    "TypeError",
                    format!("'{}' object is not iterable", value.typename()),
                ));
            }
        })
    }
    pub(crate) fn next(&mut self) -> Result<Option<Options>, Error> {
        Ok(match self {
            Self::List(value, pos) => {
                let out = match value {
                    Options::List(v) => v.borrow().get(*pos).cloned(),
                    Options::Tuple(v) => v.get(*pos).cloned(),
                    _ => unreachable!(),
                };
                if out.is_some() {
                    *pos += 1;
                }
                out
            }
            Self::Keys(v) => v.next_value()?,
            Self::Text(values, pos) => {
                let out = values
                    .get(*pos)
                    .map(|v| Options::Atom(V::Text(v.to_string())));
                if out.is_some() {
                    *pos += 1;
                }
                out
            }
        })
    }
}
