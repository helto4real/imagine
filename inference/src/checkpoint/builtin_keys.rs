//! Concrete source-owned scalar/Unicode/tuple keys for native checkpoint maps.
//! Non-admitted classes keep their required key protocol instead of a fallback.
use super::{
    builtin_scalar_ops::ComplexClass, builtin_set_ops::SetClass, dtype_objects::SourceDtype,
};
use crate::{
    native::Error,
    patching::options::{
        DictionaryKeyBindings, FrozenSet, GeneralSet, MappingKeyBindings, Options, SourceByteArray,
        SourceBytes, SourceNone, SourceNumber, SourceText,
    },
    python_hash::HashState,
};
use std::{cell::Cell, rc::Rc};
#[derive(Clone)]
pub struct BuiltinKeys {
    pub hash_state: HashState,
    /// Shared interpreter configuration supplied by the actual caller.
    pub decimal_digit_limit: Rc<Cell<usize>>,
}
fn gap(operation: &str) -> Error {
    Error::BindingGap {
        backend: "checkpoint_builtin_keys".into(),
        operation: operation.into(),
    }
}
impl BuiltinKeys {
    fn tuple_equal(&self, a: &[Options], b: &[Options]) -> Result<bool, Error> {
        for (a, b) in a.iter().zip(b) {
            if self.identity_token(a) != self.identity_token(b) && !self.equal(a, b)? {
                return Ok(false);
            }
        }
        Ok(a.len() == b.len())
    }
}
impl DictionaryKeyBindings for BuiltinKeys {
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        if let Some(class) = SetClass::get(key) {
            return Ok(crate::python_hash::identity_hash_address(
                class.identity_token() as usize,
            ));
        }
        if let Some(class) = ComplexClass::get(key) {
            return Ok(crate::python_hash::identity_hash_address(
                class.identity_token() as usize,
            ));
        }
        if let Some(set) = FrozenSet::get(key) {
            return Ok(set.hash());
        }
        if crate::patching::options::GeneralSet::get(key).is_some() {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "unhashable type: 'set'".into(),
            });
        }
        if let Some(dtype) = SourceDtype::get(key) {
            return Ok(dtype.hash());
        }
        if let Some(bytes) = SourceBytes::get(key) {
            return Ok(self.hash_state.hash_bytes(bytes.bytes()));
        }
        if SourceByteArray::get(key).is_some() {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "unhashable type: 'bytearray'".into(),
            });
        }
        if let Some(n) = SourceNumber::get(key) {
            return Ok(n.hash());
        }
        if let Some(n) = SourceNone::get(key) {
            return Ok(n.hash());
        }
        if let Some(t) = SourceText::get(key) {
            return Ok(self.hash_state.hash_unicode(t.codepoints()));
        }
        if let Options::Tuple(v) = key {
            return Ok(crate::python_hash::hash_tuple(
                &v.iter()
                    .map(|v| self.hash(v))
                    .collect::<Result<Vec<_>, _>>()?,
            ));
        }
        if matches!(key, Options::List(_)) || key.is_dictionary() {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: format!("unhashable type: '{}'", key.typename()),
            });
        }
        Err(gap("unmaterialized key requires its original key protocol"))
    }
    fn identity_token(&self, key: &Options) -> u64 {
        if let Some(class) = SetClass::get(key) {
            return class.identity_token();
        }
        if let Some(class) = ComplexClass::get(key) {
            return class.identity_token();
        }
        if let Some(set) = FrozenSet::get(key) {
            return set.identity_token();
        }
        if let Some(set) = GeneralSet::get(key) {
            return set.identity_token();
        }
        if let Some(dtype) = SourceDtype::get(key) {
            return dtype.identity_token();
        }
        if let Some(bytes) = SourceBytes::get(key) {
            bytes.identity_token()
        } else if let Some(bytes) = SourceByteArray::get(key) {
            bytes.identity_token()
        } else if let Some(n) = SourceNumber::get(key) {
            n.identity_token()
        } else if let Some(n) = SourceNone::get(key) {
            n.identity_token()
        } else if let Some(t) = SourceText::get(key) {
            t.identity_token()
        } else if let Options::Tuple(v) = key {
            Rc::as_ptr(v) as usize as u64
        } else if let Options::Object(v) = key {
            Rc::as_ptr(v) as *const () as usize as u64
        } else {
            0
        }
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        if let Some(equal) = crate::patching::options::set_equal(a, b)? {
            return Ok(equal);
        }
        if let (Some(a), Some(b)) = (SourceBytes::get(a), SourceBytes::get(b)) {
            return Ok(a.bytes() == b.bytes());
        }
        if let (Some(a), Some(b)) = (SourceByteArray::get(a), SourceByteArray::get(b)) {
            return Ok(*a.bytes() == *b.bytes());
        }
        if let (Some(a), Some(b)) = (SourceBytes::get(a), SourceByteArray::get(b)) {
            return Ok(a.bytes() == &*b.bytes());
        }
        if let (Some(a), Some(b)) = (SourceByteArray::get(a), SourceBytes::get(b)) {
            return Ok(&*a.bytes() == b.bytes());
        }
        let admitted = |v: &Options| {
            SourceDtype::get(v).is_some()
                || SourceBytes::get(v).is_some()
                || SourceByteArray::get(v).is_some()
                || SourceNumber::get(v).is_some()
                || SourceNone::get(v).is_some()
                || SourceText::get(v).is_some()
                || FrozenSet::get(v).is_some()
                || GeneralSet::get(v).is_some()
                || SetClass::get(v).is_some()
                || ComplexClass::get(v).is_some()
                || matches!(v, Options::Tuple(_))
        };
        if !admitted(a) || !admitted(b) {
            return Err(gap("non-builtin equality requires its native protocol"));
        }
        if SetClass::get(a).is_some()
            || ComplexClass::get(a).is_some()
            || SetClass::get(b).is_some()
            || ComplexClass::get(b).is_some()
        {
            return Ok(self.identity_token(a) == self.identity_token(b));
        }
        if let (Some(a), Some(b)) = (SourceDtype::get(a), SourceDtype::get(b)) {
            return Ok(std::ptr::eq(a, b));
        }
        if let (Some(a), Some(b)) = (SourceNumber::get(a), SourceNumber::get(b)) {
            return Ok(a.equal(b));
        }
        if SourceNone::get(a).is_some() || SourceNone::get(b).is_some() {
            return Ok(SourceNone::get(a).is_some() && SourceNone::get(b).is_some());
        }
        if let (Some(a), Some(b)) = (SourceText::get(a), SourceText::get(b)) {
            return Ok(a.codepoints() == b.codepoints());
        }
        if let (Options::Tuple(a), Options::Tuple(b)) = (a, b) {
            return self.tuple_equal(a, b);
        }
        // Exact admitted builtin classes return NotImplemented across these types;
        // the other exact builtin then also returns NotImplemented, yielding False.
        Ok(false)
    }
    fn exact_unicode(&self, key: &Options) -> bool {
        SourceText::get(key).is_some()
    }
}
impl MappingKeyBindings for BuiltinKeys {
    fn text_key(&self, text: &str) -> Result<Options, Error> {
        Ok(SourceText::from_utf8(text))
    }
    fn text_contents(&self, key: &Options) -> Result<Option<String>, Error> {
        match SourceText::get(key) {
            Some(t) => t
                .utf8()
                .map(Some)
                .ok_or_else(|| gap("surrogate key requires typed Unicode consumer")),
            None => Ok(None),
        }
    }
    fn missing_key(&self, key: &Options) -> Result<Error, Error> {
        Ok(Error::KeyError(crate::source_key_error::KeyError::new(
            key.clone(),
            Rc::new(self.clone()),
        )))
    }
}
impl crate::source_key_error::KeyRepresentation for BuiltinKeys {
    fn repr(&self, key: &Options) -> Result<String, Error> {
        if let Some(class) = SetClass::get(key) {
            return Ok(class.representation().into());
        }
        if let Some(class) = ComplexClass::get(key) {
            return Ok(class.representation().into());
        }
        if GeneralSet::get(key).is_some_and(|s| s.is_empty()) {
            return Ok("set()".into());
        }
        if FrozenSet::get(key).is_some_and(|s| s.is_empty()) {
            return Ok("frozenset()".into());
        }
        if let Some(dtype) = SourceDtype::get(key) {
            Ok(dtype.representation())
        } else if let Some(bytes) = SourceBytes::get(key) {
            Ok(bytes.repr())
        } else if let Some(bytes) = SourceByteArray::get(key) {
            Ok(bytes.repr())
        } else if let Some(n) = SourceNumber::get(key) {
            n.repr_with_limit(self.decimal_digit_limit.get())
        } else if SourceNone::get(key).is_some() {
            Ok("None".to_string())
        } else if let Some(t) = SourceText::get(key) {
            Ok(crate::triton_specialization::python_codepoint_str_repr(
                t.codepoints(),
            ))
        } else {
            Err(gap(
                "compound key repr requires its original object/recursion binding",
            ))
        }
    }
}
