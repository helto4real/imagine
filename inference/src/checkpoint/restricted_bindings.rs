//! Install original GLOBAL/REDUCE policy around a real native materializer.
//! This adapter does not manufacture a partial default allowlist or translate
//! unknown globals into approximate objects. All other VM operations delegate.
use super::{
    weights_only::Bindings as Vm, weights_only_globals, weights_only_newobj, weights_only_reduce,
};
use crate::{
    native::Error,
    patching::options::{Options, SourceText},
};

pub struct RestrictedBindings<B> {
    pub inner: B,
}
impl<B: Vm + weights_only_newobj::Bindings> Vm for RestrictedBindings<B> {
    fn none(&mut self) -> Result<Options, Error> {
        self.inner.none()
    }
    fn boolean(&mut self, value: bool) -> Result<Options, Error> {
        self.inner.boolean(value)
    }
    fn integer(&mut self, value: i64) -> Result<Options, Error> {
        self.inner.integer(value)
    }
    fn float(&mut self, value: f64) -> Result<Options, Error> {
        self.inner.float(value)
    }
    fn empty_dictionary(&mut self) -> Result<Options, Error> {
        self.inner.empty_dictionary()
    }
    fn global(&mut self, module: &[u8], name: &[u8]) -> Result<Options, Error> {
        let module = crate::python_unicode_codec::utf8(module, false)?;
        let name = crate::python_unicode_codec::utf8(name, false)?;
        weights_only_globals::resolve(
            &SourceText::get(&module).unwrap().utf8().unwrap(),
            &SourceText::get(&name).unwrap().utf8().unwrap(),
            &mut self.inner,
        )
    }
    fn new_object(&mut self, class: &Options, args: &Options) -> Result<Options, Error> {
        weights_only_newobj::new_object(class, args, &mut self.inner)
    }
    fn reduce(&mut self, function: &Options, args: &Options) -> Result<Options, Error> {
        weights_only_reduce::reduce(function, args, &mut self.inner)
    }
    fn build(&mut self, instance: &Options, state: &Options) -> Result<(), Error> {
        self.inner.build(instance, state)
    }
    fn set_item(&mut self, target: &Options, key: &Options, value: Options) -> Result<(), Error> {
        self.inner.set_item(target, key, value)
    }
    fn is_set_item_target(&mut self, target: &Options) -> Result<bool, Error> {
        self.inner.is_set_item_target(target)
    }
    fn type_repr(&mut self, value: &Options) -> Result<String, Error> {
        self.inner.type_repr(value)
    }
    fn empty_set(&mut self) -> Result<Options, Error> {
        self.inner.empty_set()
    }
    fn unicode(&mut self, bytes: &[u8]) -> Result<Options, Error> {
        self.inner.unicode(bytes)
    }
    fn short_string(&mut self, bytes: &[u8], encoding: &str) -> Result<Options, Error> {
        self.inner.short_string(bytes, encoding)
    }
    fn long_integer(&mut self, bytes: &[u8]) -> Result<Options, Error> {
        self.inner.long_integer(bytes)
    }
    fn persistent_load(&mut self, pid: &Options) -> Result<Options, Error> {
        self.inner.persistent_load(pid)
    }
    fn is_exact_integer(&mut self, value: &Options) -> Result<bool, Error> {
        self.inner.is_exact_integer(value)
    }
    fn persistent_tag_is_storage(&mut self, value: &Options) -> Result<bool, Error> {
        self.inner.persistent_tag_is_storage(value)
    }
    fn warning(&mut self, message: &str) -> Result<(), Error> {
        self.inner.warning(message)
    }
}
