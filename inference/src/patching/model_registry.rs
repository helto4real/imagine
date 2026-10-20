//! ModelPatcher registry methods over the canonical instance fields. Views
//! retain the field store, never a second registry or snapshots of its values.
use super::{
    attributes::{AttributeSlot, AttributeStore},
    options::Options,
    options_iterator::SequenceIterator,
};
use crate::{execution_value::Value as V, native::Error};
use std::{cell::RefCell, rc::Rc};

pub trait RegistryBindings {
    /// Original new dictionary/key protocols, not an inherited hash policy.
    fn empty_dictionary(&self) -> Result<Options, Error>;
}
pub struct ModelRegistries {
    pub callbacks: Registry,
    pub wrappers: Registry,
}
impl ModelRegistries {
    pub(crate) fn bind(fields: &AttributeStore) -> Self {
        Self {
            callbacks: Registry(fields.slot("callbacks")),
            wrappers: Registry(fields.slot("wrappers")),
        }
    }
}
pub struct Registry(AttributeSlot);
fn list() -> Options {
    Options::List(Rc::new(RefCell::new(vec![])))
}
impl Registry {
    pub fn add(
        &self,
        kind: &Options,
        value: Options,
        bindings: &dyn RegistryBindings,
    ) -> Result<(), Error> {
        self.add_with_key(kind, &Options::Atom(V::None), value, bindings)
    }
    pub fn add_with_key(
        &self,
        kind: &Options,
        key: &Options,
        value: Options,
        bindings: &dyn RegistryBindings,
    ) -> Result<(), Error> {
        let root = self.0.get()?;
        root.dictionary_method("setdefault")?;
        let default = bindings.empty_dictionary()?;
        let groups = root.setdefault_item(kind, &default);
        drop(default);
        drop(root);
        let groups = groups?;
        groups.dictionary_method("setdefault")?;
        let default = list();
        let group = groups.setdefault_item(key, &default);
        drop(default);
        drop(groups);
        let group = group?;
        let Options::List(items) = &group else {
            return Err(super::exception(
                "AttributeError",
                format!("'{}' object has no attribute 'append'", group.typename()),
            ));
        };
        items.borrow_mut().push(value);
        Ok(())
    }
    pub fn remove_with_key(
        &self,
        kind: &Options,
        key: &Options,
        bindings: &dyn RegistryBindings,
    ) -> Result<(), Error> {
        let root = self.0.get()?;
        root.dictionary_method("get")?;
        let default = bindings.empty_dictionary()?;
        let groups = root
            .get_item_optional(kind)?
            .unwrap_or_else(|| default.clone());
        drop(default);
        drop(root);
        if groups.contains_item(key)? {
            drop(groups.pop_item(key)?);
        }
        Ok(())
    }
    pub fn get(
        &self,
        kind: &Options,
        key: &Options,
        bindings: &dyn RegistryBindings,
    ) -> Result<Options, Error> {
        let root = self.0.get()?;
        root.dictionary_method("get")?;
        let default = bindings.empty_dictionary()?;
        let groups = root
            .get_item_optional(kind)?
            .unwrap_or_else(|| default.clone());
        drop(default);
        drop(root);
        groups.dictionary_method("get")?;
        let default = list();
        let group = groups
            .get_item_optional(key)?
            .unwrap_or_else(|| default.clone());
        drop(default);
        drop(groups);
        Ok(group)
    }
    pub fn all(&self, kind: &Options, bindings: &dyn RegistryBindings) -> Result<Options, Error> {
        let output = list();
        let root = self.0.get()?;
        root.dictionary_method("get")?;
        let default = bindings.empty_dictionary()?;
        let groups = root
            .get_item_optional(kind)?
            .unwrap_or_else(|| default.clone());
        drop(default);
        drop(root);
        let values = groups.values_iterator();
        drop(groups);
        let values = values?;
        let mut last = None;
        while let Some(group) = values.next_value()? {
            drop(last.replace(group));
            let mut items =
                SequenceIterator::for_operation(last.as_ref().unwrap(), "registry list extension")?;
            while let Some(item) = items.next()? {
                let Options::List(out) = &output else {
                    unreachable!()
                };
                out.borrow_mut().push(item);
            }
        }
        Ok(output)
    }
}
