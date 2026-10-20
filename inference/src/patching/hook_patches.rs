//! Ordered hook-patch containers from ModelPatcher, separate from weight I/O.
//! Hook references, lists and tuples retain their original sharing semantics.
use super::{exception, hooks::HookGroup, options::Options};
use crate::{execution_value::Value as V, native::Error};
use std::{cell::RefCell, rc::Rc};

pub fn empty_dict() -> Options {
    Options::dictionary(vec![])
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn iterable(value: &Options) -> Result<Vec<Options>, Error> {
    match value {
        Options::List(v) => Ok(v.borrow().clone()),
        Options::Tuple(v) => Ok(v.as_ref().clone()),
        Options::Dict(v) => Ok(v
            .borrow()
            .iter()
            .map(|(k, _)| Options::Atom(V::Text(k.clone())))
            .collect()),
        Options::Atom(V::Text(v)) => Ok(v
            .chars()
            .map(|c| Options::Atom(V::Text(c.to_string())))
            .collect()),
        _ => Err(exception(
            "TypeError",
            format!("'{}' object is not iterable", value.typename()),
        )),
    }
}
fn slice(value: &Options) -> Result<Options, Error> {
    match value {
        Options::List(v) => Ok(list(v.borrow().clone())),
        Options::Tuple(_) => Ok(value.clone()),
        Options::Atom(V::Text(_)) => Ok(value.clone()),
        // Python 3.12 slices are hashable; dict[:] reaches its missing key.
        Options::Dict(_) => Err(exception("KeyError", "slice(None, None, None)")),
        _ => Err(exception(
            "TypeError",
            format!("'{}' object is not subscriptable", value.typename()),
        )),
    }
}
thread_local! {static EMPTY_TUPLE:Options=Options::Tuple(Rc::new(vec![]));}
fn tuple_from(value: &Options) -> Result<Options, Error> {
    if matches!(value, Options::Tuple(_)) {
        return Ok(value.clone());
    }
    let values = iterable(value)?;
    Ok(if values.is_empty() {
        EMPTY_TUPLE.with(Clone::clone)
    } else {
        Options::Tuple(Rc::new(values))
    })
}
#[derive(Default)]
pub struct HookPatches {
    pub entries: Vec<(Rc<()>, Options)>,
}
impl HookPatches {
    pub fn get(&self, reference: &Rc<()>) -> Option<Options> {
        self.entries
            .iter()
            .find(|(key, _)| Rc::ptr_eq(key, reference))
            .map(|(_, v)| v.clone())
    }
    pub fn set(&mut self, reference: Rc<()>, values: Options) {
        if let Some((_, v)) = self
            .entries
            .iter_mut()
            .find(|(key, _)| Rc::ptr_eq(key, &reference))
        {
            *v = values;
        } else {
            self.entries.push((reference, values));
        }
    }
    pub fn clone_source(&self, copy_tuples: bool) -> Result<Self, Error> {
        let mut out = Self::default();
        for (reference, values) in &self.entries {
            let cloned = empty_dict();
            out.set(reference.clone(), cloned.clone());
            for (key, patches) in values.dict_items()? {
                let new = slice(&patches)?;
                cloned.set(&key, new.clone())?;
                if copy_tuples {
                    let length = iterable(&new)?.len();
                    for index in 0..length {
                        let value = tuple_from(&iterable(&new)?[index])?;
                        let Options::List(values) = &new else {
                            return Err(exception(
                                "TypeError",
                                format!(
                                    "'{}' object does not support item assignment",
                                    new.typename()
                                ),
                            ));
                        };
                        values.borrow_mut()[index] = value;
                    }
                }
            }
        }
        Ok(out)
    }
    pub fn combined(&self, hooks: Option<&HookGroup>) -> Result<Options, Error> {
        let out = empty_dict();
        if let Some(hooks) = hooks {
            for handle in &hooks.hooks {
                let hook = handle.borrow();
                let strength = hook.keyframes.borrow().strength();
                let values = self.get(&hook.reference).unwrap_or_else(empty_dict);
                for (key, patches) in values.dict_items()? {
                    let current = out.get(&key)?.unwrap_or_else(|| list(vec![]));
                    let Options::List(current_values) = &current else {
                        unreachable!()
                    };
                    if super::super::sampling::guidance::is_close_one(strength) {
                        current_values.borrow_mut().extend(iterable(&patches)?);
                    } else {
                        for patch in iterable(&patches)? {
                            let mut values = iterable(&patch)?;
                            let first = values.first_mut().ok_or_else(|| {
                                exception("IndexError", "list index out of range")
                            })?;
                            *first = scale(first, strength)?;
                            current_values
                                .borrow_mut()
                                .push(Options::Tuple(Rc::new(values)));
                        }
                    }
                    out.set(&key, current)?;
                }
            }
        }
        Ok(out)
    }
}
fn scale(value: &Options, strength: f64) -> Result<Options, Error> {
    let out = match value {
        Options::Atom(V::Bool(v)) => V::Float(f64::from(u8::from(*v)) * strength),
        Options::Atom(V::Int(v)) => V::Float(*v as f64 * strength),
        Options::Atom(V::UInt(v)) => V::Float(*v as f64 * strength),
        Options::Atom(V::Float(v)) => V::Float(*v * strength),
        Options::Atom(V::Tensor(tensor)) => {
            // Python's augmented assignment mutates the shared scalar Tensor.
            // Copying the patch list does not copy its children.
            let _ = tensor.lock().unwrap().tensor.f_mul_scalar_(strength)?;
            return Ok(value.clone());
        }
        Options::List(_) | Options::Tuple(_) | Options::Atom(V::Text(_)) => {
            return Err(exception(
                "TypeError",
                "can't multiply sequence by non-int of type 'float'",
            ));
        }
        _ => {
            return Err(exception(
                "TypeError",
                format!(
                    "unsupported operand type(s) for *=: '{}' and 'float'",
                    value.typename()
                ),
            ));
        }
    };
    Ok(Options::Atom(out))
}
/// Source cache budget arithmetic. Strict `>` is intentional; an exact fit
/// refuses caching and does not decrement the counter.
pub struct MemoryCounter {
    pub value: i128,
    pub minimum: i128,
}
impl MemoryCounter {
    pub fn is_useable(&self, used: i128) -> bool {
        self.value - used > self.minimum
    }
    pub fn decrement(&mut self, used: i128) {
        self.value -= used;
    }
    pub fn use_bytes(&mut self, used: i128) -> bool {
        if self.is_useable(used) {
            self.decrement(used);
            true
        } else {
            false
        }
    }
}
