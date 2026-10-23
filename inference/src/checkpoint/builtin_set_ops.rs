//! Exact builtin set/frozenset constructors for native scalar containers.
//! Unknown classes retain a required iterator binding, never a snapshot/coercion.
use crate::{
    execution_value::Value,
    patching::options::{MetadataObject, MetadataValue},
};
use crate::{
    native::Error,
    patching::options::{
        FrozenSet, GeneralMapping, GeneralSet, MappingKeyBindings, Options, SourceByteArray,
        SourceBytes, SourceNone, SourceNumber, SourceText,
    },
};
use std::{any::Any, rc::Rc};
/// Process-owned exact builtin classes, distinct from registry aliases.
pub struct SetClass {
    frozen: bool,
}
thread_local! {
 static SET_CLASS: Options = MetadataObject::wrap(SetClass { frozen: false });
 static FROZEN_CLASS: Options = MetadataObject::wrap(SetClass { frozen: true });
}
impl SetClass {
    pub fn value(frozen: bool) -> Options {
        if frozen {
            FROZEN_CLASS.with(Clone::clone)
        } else {
            SET_CLASS.with(Clone::clone)
        }
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        MetadataObject::get(value.as_ref())
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    pub fn representation(&self) -> &'static str {
        if self.frozen {
            "<class 'frozenset'>"
        } else {
            "<class 'set'>"
        }
    }
    pub fn call_positional(
        &self,
        args: &[Options],
        keys: Rc<dyn MappingKeyBindings>,
        update_iterable: impl FnMut(&GeneralSet, &Options) -> Result<(), Error>,
    ) -> Result<Options, Error> {
        construct(self.frozen, args, keys, update_iterable)
    }
}
impl MetadataValue for SetClass {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Err(Error::BindingGap {
            backend: "builtin_set_class".into(),
            operation: "class object requires its typed caller, not a scalar snapshot".into(),
        })
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| std::ptr::eq(self, other))
    }
    fn type_name(&self) -> &str {
        "type"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(true)
    }
}
fn type_error(message: String) -> Error {
    Error::ReferenceException {
        class: "TypeError",
        message,
    }
}
pub trait IteratorBindings {
    fn iter(&mut self, source: &Options) -> Result<Options, Error>;
    /// Actual PyIter_Next boundary, including StopIteration class handling.
    fn next(&mut self, iterator: &Options) -> Result<Option<Options>, Error>;
}
/// Generic set_update_internal path. No length hint/presizing, and an add
/// failure releases the iterator before the currently owned key.
pub fn update_from_protocol(
    target: &GeneralSet,
    source: &Options,
    b: &mut dyn IteratorBindings,
) -> Result<(), Error> {
    let iterator = b.iter(source)?;
    loop {
        let next = b.next(&iterator);
        let key = match next {
            Ok(Some(key)) => key,
            Ok(None) => {
                drop(iterator);
                return Ok(());
            }
            Err(error) => {
                drop(iterator);
                return Err(error);
            }
        };
        let result = target.add(&key);
        if let Err(error) = result {
            drop(iterator);
            drop(key);
            return Err(error);
        }
        drop(key);
    }
}
/// Return false only when the actual source iterator protocol is still needed.
/// Hash callbacks may mutate lists/bytearrays; acquire each next value live.
pub fn update_builtin(target: &GeneralSet, source: &Options) -> Result<bool, Error> {
    if target.merge_set_object(source)? {
        return Ok(true);
    }
    if let Some(source) = GeneralMapping::get(source) {
        target.merge_dictionary(&source.dictionary)?;
        return Ok(true);
    }
    if let Options::List(values) = source {
        let mut index = 0;
        loop {
            let value = values.borrow().get(index).cloned();
            let Some(value) = value else { break };
            index += 1;
            target.add(&value)?;
        }
        return Ok(true);
    }
    if let Options::Tuple(values) = source {
        for value in values.iter() {
            target.add(value)?;
        }
        return Ok(true);
    }
    if let Some(text) = SourceText::get(source) {
        for point in text.codepoints() {
            target.add(&SourceText::decoded_codepoints(vec![*point])?)?;
        }
        return Ok(true);
    }
    if let Some(bytes) = SourceBytes::get(source) {
        for byte in bytes.bytes() {
            target.add(&SourceNumber::from_i64(i64::from(*byte)))?;
        }
        return Ok(true);
    }
    if let Some(bytes) = SourceByteArray::get(source) {
        let mut index = 0;
        loop {
            let value = bytes.bytes().get(index).copied();
            let Some(value) = value else { break };
            index += 1;
            target.add(&SourceNumber::from_i64(i64::from(value)))?;
        }
        return Ok(true);
    }
    if SourceNone::get(source).is_some()
        || SourceNumber::get(source).is_some()
        || super::dtype_objects::SourceDtype::get(source).is_some()
        || SetClass::get(source).is_some()
        || super::builtin_scalar_ops::ComplexClass::get(source).is_some()
    {
        return Err(type_error(format!(
            "'{}' object is not iterable",
            source.typename()
        )));
    }
    Ok(false)
}
/// Positional argument expansion/keyword validation happens at the original
/// caller boundary. The fallback must execute the actual iterator protocol.
pub fn construct(
    frozen: bool,
    args: &[Options],
    keys: Rc<dyn MappingKeyBindings>,
    mut update_iterable: impl FnMut(&GeneralSet, &Options) -> Result<(), Error>,
) -> Result<Options, Error> {
    let name = if frozen { "frozenset" } else { "set" };
    if args.len() > 1 {
        return Err(type_error(format!(
            "{name} expected at most 1 argument, got {}",
            args.len()
        )));
    }
    if frozen
        && let Some(source) = args.first()
        && FrozenSet::get(source).is_some()
    {
        return Ok(source.clone());
    }
    let result = GeneralSet::new(keys);
    if let Some(source) = args.first()
        && !update_builtin(&result, source)?
    {
        update_iterable(&result, source)?;
    }
    Ok(if frozen {
        FrozenSet::from_new_table(result).into_options()
    } else {
        result.into_options()
    })
}
