//! Compose actual immutable scalar owners and builtin maps with the already
//! source-faithful weights-only VM. Global/tensor/storage operations stay with
//! their explicit materializer; this does not supply placeholder model objects.
use super::{builtin_keys::BuiltinKeys, weights_only::Bindings};
use crate::{
    native::Error,
    patching::options::{
        GeneralMapping, GeneralSet, Options, SourceByteArray, SourceBytes, SourceNone,
        SourceNumber, SourceText,
    },
    python_hash::HashState,
};
use std::rc::Rc;
pub struct ScalarBindings<B> {
    pub inner: B,
    pub keys: Rc<BuiltinKeys>,
}
impl<B: Bindings> ScalarBindings<B> {
    pub fn new(inner: B, hash_state: HashState) -> Self {
        Self {
            inner,
            keys: Rc::new(BuiltinKeys {
                hash_state,
                decimal_digit_limit: Rc::new(std::cell::Cell::new(4300)),
            }),
        }
    }
}
impl<B: Bindings> Bindings for ScalarBindings<B> {
    fn none(&mut self) -> Result<Options, Error> {
        Ok(SourceNone::value())
    }
    fn boolean(&mut self, v: bool) -> Result<Options, Error> {
        Ok(SourceNumber::boolean(v))
    }
    fn integer(&mut self, v: i64) -> Result<Options, Error> {
        Ok(SourceNumber::from_i64(v))
    }
    fn float(&mut self, v: f64) -> Result<Options, Error> {
        Ok(SourceNumber::float(v))
    }
    fn long_integer(&mut self, v: &[u8]) -> Result<Options, Error> {
        Ok(SourceNumber::integer(v))
    }
    fn empty_dictionary(&mut self) -> Result<Options, Error> {
        Ok(GeneralMapping::empty(self.keys.clone()))
    }
    fn global(&mut self, module: &[u8], name: &[u8]) -> Result<Options, Error> {
        self.inner.global(module, name)
    }
    fn new_object(&mut self, class: &Options, args: &Options) -> Result<Options, Error> {
        self.inner.new_object(class, args)
    }
    fn reduce(&mut self, function: &Options, args: &Options) -> Result<Options, Error> {
        self.inner.reduce(function, args)
    }
    fn build(&mut self, instance: &Options, state: &Options) -> Result<(), Error> {
        self.inner.build(instance, state)
    }
    fn set_item(&mut self, target: &Options, key: &Options, value: Options) -> Result<(), Error> {
        if GeneralMapping::get(target).is_some() {
            target.set_item(key.clone(), value)
        } else {
            self.inner.set_item(target, key, value)
        }
    }
    fn is_set_item_target(&mut self, target: &Options) -> Result<bool, Error> {
        if GeneralMapping::get(target).is_some() {
            Ok(true)
        } else {
            self.inner.is_set_item_target(target)
        }
    }
    fn type_repr(&mut self, value: &Options) -> Result<String, Error> {
        if SourceNumber::get(value).is_some()
            || SourceNone::get(value).is_some()
            || SourceText::get(value).is_some()
            || SourceBytes::get(value).is_some()
            || SourceByteArray::get(value).is_some()
            || super::dtype_objects::SourceDtype::get(value).is_some()
            || GeneralSet::get(value).is_some()
            || crate::patching::options::FrozenSet::get(value).is_some()
        {
            Ok(format!("<class '{}'>", value.typename()))
        } else {
            self.inner.type_repr(value)
        }
    }
    fn empty_set(&mut self) -> Result<Options, Error> {
        Ok(GeneralSet::new(self.keys.clone()).into_options())
    }
    fn unicode(&mut self, bytes: &[u8]) -> Result<Options, Error> {
        self.inner.unicode(bytes)
    }
    fn short_string(&mut self, bytes: &[u8], encoding: &str) -> Result<Options, Error> {
        if encoding == "bytes" {
            Ok(SourceBytes::from_data(bytes))
        } else {
            self.inner.short_string(bytes, encoding)
        }
    }
    fn persistent_load(&mut self, pid: &Options) -> Result<Options, Error> {
        self.inner.persistent_load(pid)
    }
    fn is_exact_integer(&mut self, value: &Options) -> Result<bool, Error> {
        if let Some(n) = SourceNumber::get(value) {
            Ok(n.is_exact_integer())
        } else {
            self.inner.is_exact_integer(value)
        }
    }
    fn persistent_tag_is_storage(&mut self, value: &Options) -> Result<bool, Error> {
        if let Some(bytes) = SourceBytes::get(value) {
            let text = crate::python_unicode_codec::ascii(bytes.bytes())?;
            Ok(crate::patching::options::SourceText::get(&text)
                .unwrap()
                .codepoints()
                == [115, 116, 111, 114, 97, 103, 101])
        } else if let Some(text) = SourceText::get(value) {
            Ok(text.codepoints() == [115, 116, 111, 114, 97, 103, 101])
        } else {
            self.inner.persistent_tag_is_storage(value)
        }
    }
    fn warning(&mut self, message: &str) -> Result<(), Error> {
        self.inner.warning(message)
    }
}
