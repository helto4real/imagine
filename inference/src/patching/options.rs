//! Request-owned option containers. Dict/list aliases survive source-faithful
//! shallow copies, independently of the owned Value trees at the ATen ABI.
use crate::{execution_value::Value, native::Error};
use std::{any::Any, cell::RefCell, rc::Rc};

#[path = "options_dict.rs"]
mod dictionary;
pub use dictionary::{Dictionary, ItemsIterator, KeysIterator, ValuesIterator};
#[path = "options_general_dict.rs"]
mod general_dictionary;
pub use general_dictionary::{
    DictionaryKeyBindings, DictionaryLayout, GeneralDictionary, GeneralDictionaryIterator,
};
#[path = "options_dict_iterator.rs"]
mod dictionary_iterator;
#[path = "options_mapping.rs"]
mod mapping;
pub use mapping::{GeneralMapping, MappingKeyBindings};
#[path = "options_text.rs"]
mod source_text;
pub use source_text::SourceText;
#[path = "options_metadata.rs"]
mod metadata;
pub(crate) use metadata::{MetadataObject, MetadataValue};
pub type Dict = Rc<RefCell<Dictionary>>;
pub type List = Rc<RefCell<Vec<Options>>>;
#[derive(Clone)]
pub enum Options {
    Atom(Value),
    Dict(Dict),
    List(List),
    Tuple(Rc<Vec<Options>>),
    Object(Rc<dyn Any>),
}
impl Options {
    pub(crate) fn dictionary_method(&self, name: &str) -> Result<(), Error> {
        if self.is_dictionary() {
            Ok(())
        } else {
            Err(self.missing(name))
        }
    }
    pub fn is_dictionary(&self) -> bool {
        matches!(self, Self::Dict(_)) || GeneralMapping::get(self).is_some()
    }
    /// Typed item access retains general keys and their hash/equality protocols.
    pub fn item(&self, key: &Self) -> Result<Self, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return match mapping.dictionary.get(key, mapping.bindings.as_ref())? {
                Some(value) => Ok(value),
                None => Err(mapping.bindings.missing_key(key)?),
            };
        }
        if let Self::Dict(_) = self {
            if let Self::Atom(Value::Text(key)) = key {
                return self.get(key)?.ok_or_else(|| {
                    super::exception(
                        "KeyError",
                        crate::triton_specialization::python_str_repr(key),
                    )
                });
            }
            return Err(Error::BindingGap {
                backend: "patch_options".into(),
                operation: "general key requires the native general dictionary representation"
                    .into(),
            });
        }
        Err(super::gap(
            "item access requires its original typed consumer",
        ))
    }
    pub fn set_item(&self, key: Self, value: Self) -> Result<(), Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping
                .dictionary
                .set(key, value, mapping.bindings.as_ref());
        }
        if let Self::Atom(Value::Text(key)) = key {
            return self.set(&key, value);
        }
        Err(Error::BindingGap {
            backend: "patch_options".into(),
            operation: "general key assignment requires its native dictionary representation"
                .into(),
        })
    }
    pub fn get_item_optional(&self, key: &Self) -> Result<Option<Self>, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.dictionary.get(key, mapping.bindings.as_ref());
        }
        if !matches!(self, Self::Dict(_)) {
            return Err(self.missing("get"));
        }
        let Self::Atom(Value::Text(key)) = key else {
            return Err(super::gap(
                "raw dictionary keys require a general dictionary",
            ));
        };
        self.get(key)
    }
    pub fn setdefault_item(&self, key: &Self, default: &Self) -> Result<Self, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping
                .dictionary
                .setdefault(key, default, mapping.bindings.as_ref());
        }
        let Self::Dict(dict) = self else {
            return Err(self.missing("setdefault"));
        };
        let Self::Atom(Value::Text(key)) = key else {
            return Err(super::gap(
                "raw dictionary keys require a general dictionary",
            ));
        };
        let existing = dict
            .borrow()
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone());
        if let Some(existing) = existing {
            return Ok(existing);
        }
        drop(dict.borrow_mut().set(key, default.clone()));
        Ok(default.clone())
    }
    pub fn contains_item(&self, key: &Self) -> Result<bool, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.dictionary.contains(key, mapping.bindings.as_ref());
        }
        Ok(self.get_item_optional(key)?.is_some())
    }
    /// dict.pop(key), with no default. Keep its empty-map hash short circuit.
    pub fn pop_item(&self, key: &Self) -> Result<Self, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return match mapping.dictionary.pop(key, mapping.bindings.as_ref())? {
                Some(value) => Ok(value),
                None => Err(mapping.bindings.missing_key(key)?),
            };
        }
        let Self::Atom(Value::Text(key)) = key else {
            return Err(super::gap(
                "raw dictionary keys require a general dictionary",
            ));
        };
        self.pop(key)?.ok_or_else(|| {
            super::exception(
                "KeyError",
                crate::triton_specialization::python_str_repr(key),
            )
        })
    }
    pub(crate) fn metadata_truth(&self) -> Option<Result<bool, Error>> {
        match self {
            Self::Object(v) => v
                .downcast_ref::<MetadataObject>()
                .map(MetadataObject::truth),
            _ => None,
        }
    }
    /// A new dictionary populated from ordered pairs, like dict(iterable).
    pub fn dictionary(values: Vec<(String, Self)>) -> Self {
        Self::Dict(Rc::new(RefCell::new(Dictionary::from_pairs(values))))
    }
    pub fn items_iterator(&self) -> Result<ItemsIterator, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return Ok(ItemsIterator::general(mapping.dictionary.clone()));
        }
        let Self::Dict(dict) = self else {
            return Err(self.missing("items"));
        };
        Ok(ItemsIterator::new(dict.clone()))
    }
    pub fn keys_iterator(&self) -> Result<KeysIterator, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return Ok(KeysIterator::general(mapping.dictionary.clone()));
        }
        let Self::Dict(dict) = self else {
            return Err(self.missing("keys"));
        };
        Ok(KeysIterator::new(dict.clone()))
    }
    pub fn values_iterator(&self) -> Result<ValuesIterator, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return Ok(ValuesIterator::general(mapping.dictionary.clone()));
        }
        let Self::Dict(dict) = self else {
            return Err(self.missing("values"));
        };
        Ok(ValuesIterator::new(dict.clone()))
    }
    pub fn clear_dict(&self) -> Result<(), Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            mapping.dictionary.clear();
            return Ok(());
        }
        let Self::Dict(dict) = self else {
            return Err(self.missing("clear"));
        };
        let old = std::mem::take(&mut *dict.borrow_mut());
        drop(old);
        Ok(())
    }
    pub fn from_value(value: Value) -> Self {
        match value {
            Value::Dict(values) => Self::Dict(Rc::new(RefCell::new(
                values
                    .into_iter()
                    .map(|(k, v)| (k, Self::from_value(v)))
                    .collect(),
            ))),
            Value::List(values) => Self::List(Rc::new(RefCell::new(
                values.into_iter().map(Self::from_value).collect(),
            ))),
            Value::Tuple(values) => {
                Self::Tuple(Rc::new(values.into_iter().map(Self::from_value).collect()))
            }
            atom => Self::Atom(atom),
        }
    }
    pub fn snapshot(&self) -> Result<Value, Error> {
        Ok(match self {
            Self::Atom(value) => value.clone(),
            Self::Dict(values) => Value::Dict(
                values
                    .borrow()
                    .iter()
                    .map(|(k, v)| Ok((k.clone(), v.snapshot()?)))
                    .collect::<Result<_, Error>>()?,
            ),
            Self::List(values) => Value::List(
                values
                    .borrow()
                    .iter()
                    .map(Self::snapshot)
                    .collect::<Result<_, _>>()?,
            ),
            Self::Tuple(values) => Value::Tuple(
                values
                    .iter()
                    .map(Self::snapshot)
                    .collect::<Result<_, _>>()?,
            ),
            Self::Object(value) => match value.downcast_ref::<MetadataObject>() {
                Some(value) => value.snapshot()?,
                None => {
                    return Err(Error::BindingGap {
                        backend: "patch_options".into(),
                        operation: "opaque object requires its typed Rust consumer".into(),
                    });
                }
            },
        })
    }
    pub fn same_object(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dict(a), Self::Dict(b)) => Rc::ptr_eq(a, b),
            (Self::List(a), Self::List(b)) => Rc::ptr_eq(a, b),
            (Self::Tuple(a), Self::Tuple(b)) => Rc::ptr_eq(a, b) || (a.is_empty() && b.is_empty()),
            (Self::Object(a), Self::Object(b)) => {
                Rc::ptr_eq(a, b)
                    || a.downcast_ref::<MetadataObject>()
                        .is_some_and(|v| v.same_object(b.as_ref()))
                    || b.downcast_ref::<MetadataObject>()
                        .is_some_and(|v| v.same_object(a.as_ref()))
            }
            (Self::Atom(a), Self::Atom(b)) => a.same_object(b),
            _ => false,
        }
    }
    pub(crate) fn typename(&self) -> &str {
        match self {
            Self::Dict(_) => "dict",
            Self::List(_) => "list",
            Self::Tuple(_) => "tuple",
            Self::Object(v) => v
                .downcast_ref::<MetadataObject>()
                .map_or("object", MetadataObject::type_name),
            Self::Atom(Value::None) => "NoneType",
            Self::Atom(Value::Bool(_)) => "bool",
            Self::Atom(Value::Int(_) | Value::UInt(_)) => "int",
            Self::Atom(Value::Float(_)) => "float",
            Self::Atom(Value::Text(_)) => "str",
            Self::Atom(Value::Tensor(_) | Value::Quantized(_)) => "Tensor",
            _ => "object",
        }
    }
    fn missing(&self, attr: &str) -> Error {
        Error::ReferenceException {
            class: "AttributeError",
            message: format!("'{}' object has no attribute '{attr}'", self.typename()),
        }
    }
    pub fn get(&self, key: &str) -> Result<Option<Self>, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.get_text(key);
        }
        let Self::Dict(dict) = self else {
            return Err(self.missing("get"));
        };
        Ok(dict
            .borrow()
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone()))
    }
    pub fn set(&self, key: &str, value: Self) -> Result<(), Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.set_text(key, value);
        }
        let Self::Dict(dict) = self else {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: format!(
                    "'{}' object does not support item assignment",
                    self.typename()
                ),
            });
        };
        let old = dict.borrow_mut().set(key, value);
        drop(old);
        Ok(())
    }
    pub(crate) fn dict_items(&self) -> Result<Vec<(String, Self)>, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.text_items();
        }
        let Self::Dict(values) = self else {
            return Err(self.missing("items"));
        };
        Ok(values.borrow().snapshot())
    }
    /// dict.copy keeps every child container/object by identity.
    pub fn shallow_copy_dict(&self) -> Result<Self, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.copy();
        }
        let Self::Dict(values) = self else {
            return Err(self.missing("copy"));
        };
        Ok(Self::Dict(Rc::new(RefCell::new(
            values.borrow().source_copy(),
        ))))
    }
    pub fn pop(&self, key: &str) -> Result<Option<Self>, Error> {
        if let Some(mapping) = GeneralMapping::get(self) {
            return mapping.pop_text(key);
        }
        let Self::Dict(values) = self else {
            return Err(self.missing("pop"));
        };
        Ok(values.borrow_mut().pop(key))
    }
}
pub fn copy_nested_dicts(input: &Options) -> Result<Options, Error> {
    // Source copies the entire dictionary first, then traverses the live
    // original items. Reconstructing from an entries snapshot loses its copy
    // layout, arbitrary keys and mutation/error ordering.
    let output = if let Options::List(values) = input {
        // list.copy succeeds before the source subsequently rejects .items().
        Options::List(Rc::new(RefCell::new(values.borrow().clone())))
    } else {
        input.shallow_copy_dict()?
    };
    let items = input.items_iterator()?;
    let mut key = None;
    let mut value = None;
    while let Some(item) = items.next_item()? {
        let Options::Tuple(values) = item else {
            unreachable!("dictionary item pair")
        };
        drop(key.replace(values[0].clone()));
        drop(value.replace(values[1].clone()));
        drop(values);
        let current = value.as_ref().unwrap();
        if current.is_dictionary() {
            let copied = copy_nested_dicts(current)?;
            output.set_item(key.as_ref().unwrap().clone(), copied)?;
        } else if let Options::List(values) = current {
            let copied = Options::List(Rc::new(RefCell::new(values.borrow().clone())));
            output.set_item(key.as_ref().unwrap().clone(), copied)?;
        }
    }
    Ok(output)
}
pub fn merge_nested_dicts(
    left: &Options,
    right: &Options,
    copy_left: bool,
) -> Result<Options, Error> {
    let output = if copy_left {
        copy_nested_dicts(left)?
    } else {
        left.clone()
    };
    for (key, value) in right.dict_items()? {
        match value {
            Options::Dict(_) => {
                let current = if let Some(v) = output.get(&key)? {
                    v
                } else {
                    let v = Options::dictionary(vec![]);
                    output.set(&key, v.clone())?;
                    v
                };
                output.set(&key, merge_nested_dicts(&current, &value, true)?)?;
            }
            Options::List(ref incoming) => {
                let incoming = incoming.borrow().clone();
                let current = if let Some(v) = output.get(&key)? {
                    v
                } else {
                    let v = Options::List(Rc::new(RefCell::new(Vec::new())));
                    output.set(&key, v.clone())?;
                    v
                };
                let Options::List(list) = &current else {
                    return Err(current.missing("extend"));
                };
                list.borrow_mut().extend(incoming);
            }
            _ => output.set(&key, value)?,
        }
    }
    Ok(output)
}
