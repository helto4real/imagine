//! Typed native general-dictionary objects in the shared Options graph.
use super::{DictionaryKeyBindings, GeneralDictionary, MetadataObject, MetadataValue, Options};
use crate::{execution_value::Value, native::Error};
use std::{any::Any, rc::Rc};

pub trait MappingKeyBindings: DictionaryKeyBindings {
    /// Actual source text object for a string-key operation, including its
    /// host identity. No value-equality substitute for source object identity.
    fn text_key(&self, text: &str) -> Result<Options, Error>;
    /// Raw Unicode contents for string-only Rust/ATen consumers. General typed
    /// consumers keep the original key object through the items iterator.
    fn text_contents(&self, key: &Options) -> Result<Option<String>, Error>;
    /// Original KeyError formatting, including the native key repr protocol.
    fn missing_key(&self, key: &Options) -> Result<Error, Error>;
}
pub struct GeneralMapping {
    pub dictionary: GeneralDictionary,
    pub bindings: Rc<dyn MappingKeyBindings>,
}
impl GeneralMapping {
    pub fn empty(bindings: Rc<dyn MappingKeyBindings>) -> Options {
        Self::wrap(GeneralDictionary::new(), bindings)
    }
    pub fn wrap(dictionary: GeneralDictionary, bindings: Rc<dyn MappingKeyBindings>) -> Options {
        MetadataObject::wrap(Self {
            dictionary,
            bindings,
        })
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(object) = value else {
            return None;
        };
        MetadataObject::get(object.as_ref())
    }
    pub fn get_text(&self, text: &str) -> Result<Option<Options>, Error> {
        self.dictionary
            .get(&self.bindings.text_key(text)?, self.bindings.as_ref())
    }
    pub fn set_text(&self, text: &str, value: Options) -> Result<(), Error> {
        self.dictionary
            .set(self.bindings.text_key(text)?, value, self.bindings.as_ref())
    }
    pub fn pop_text(&self, text: &str) -> Result<Option<Options>, Error> {
        self.dictionary
            .pop(&self.bindings.text_key(text)?, self.bindings.as_ref())
    }
    pub fn copy(&self) -> Result<Options, Error> {
        Ok(Self::wrap(
            self.dictionary.shallow_copy(self.bindings.as_ref())?,
            self.bindings.clone(),
        ))
    }
    pub fn text_items(&self) -> Result<Vec<(String, Options)>, Error> {
        self.dictionary.entries().into_iter().map(|(key,value)| {
            self.bindings.text_contents(&key)?.map(|key|(key,value)).ok_or_else(|| Error::BindingGap {
                backend:"patch_options".into(),
                operation:"general dictionary key requires its typed consumer, not string conversion".into(),
            })
        }).collect()
    }
}
impl MetadataValue for GeneralMapping {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Ok(Value::Dict(
            self.text_items()?
                .into_iter()
                .map(|(key, value)| Ok((key, value.snapshot()?)))
                .collect::<Result<_, Error>>()?,
        ))
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| self.dictionary.same_object(&other.dictionary))
    }
    fn type_name(&self) -> &str {
        "dict"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(!self.dictionary.is_empty())
    }
}
