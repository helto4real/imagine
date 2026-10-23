//! Original uncached _get_user_allowed_globals body. Tuple-subclass indexing,
//! length and unpack are separate source protocols; keys are not normalized.
use super::default_registry;
use crate::{
    native::Error,
    patching::options::{Options, SourceText},
    source_exception::Frame,
};
use std::rc::Rc;
pub trait Bindings: default_registry::Bindings {
    fn is_tuple(&mut self, value: &Options) -> Result<bool, Error>;
    fn length(&mut self, value: &Options) -> Result<usize, Error>;
    fn tuple_item(&mut self, value: &Options, index: usize) -> Result<Options, Error>;
    fn is_exact_string(&mut self, value: &Options) -> Result<bool, Error>;
    fn type_object(&mut self, value: &Options) -> Result<Options, Error>;
    fn unpack_pair(&mut self, value: &Options) -> Result<(Options, Options), Error>;
}
#[derive(Default)]
struct Locals {
    rc: Option<Options>,
    f: Option<Options>,
    name: Option<Options>,
    module: Option<Options>,
}
fn type_message(value: &Options) -> Result<String, Error> {
    SourceText::get(value)
        .and_then(SourceText::utf8)
        .ok_or_else(|| Error::BindingGap {
            backend: "user_registry".into(),
            operation: "non-UTF8 exception text requires its typed source exception consumer"
                .into(),
        })
}
fn body(b: &mut dyn Bindings, locals: &mut Locals) -> Result<Options, Error> {
    let rc = b.literal_map(vec![])?;
    locals.rc = Some(rc.clone());
    let marked = b.load("_marked_safe_globals_set")?;
    let iterator = b.iter(&marked)?;
    drop(marked);
    while let Some(value) = b.next(&iterator)? {
        drop(locals.f.replace(value));
        let f = locals.f.as_ref().unwrap();
        if b.is_tuple(f)? {
            if b.length(f)? != 2 {
                let length = b.length(f)?;
                return Err(Error::ReferenceException {
                    class: "ValueError",
                    message: format!(
                        "Expected tuple of length 2 (global, str of callable full path), but got tuple of length: {length}"
                    ),
                });
            }
            let item = b.tuple_item(f, 1)?;
            let exact = b.is_exact_string(&item)?;
            drop(item);
            if !exact {
                let item = b.tuple_item(f, 1)?;
                let ty = b.type_object(&item)?;
                drop(item);
                let text = b.format(&ty)?;
                drop(ty);
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: format!(
                        "Expected second item in tuple to be str of callable full path, but got: {}",
                        type_message(&text)?
                    ),
                });
            }
            let (function, name) = b.unpack_pair(f)?;
            drop(locals.f.replace(function));
            drop(locals.name.replace(name));
            b.set(
                &rc,
                locals.name.as_ref().unwrap().clone(),
                locals.f.as_ref().unwrap().clone(),
            )?;
        } else {
            let module = b.attribute(f, "__module__")?;
            let name = b.attribute(f, "__qualname__")?;
            drop(locals.module.replace(module));
            drop(locals.name.replace(name));
            let module = b.format(locals.module.as_ref().unwrap())?;
            let name = b.format(locals.name.as_ref().unwrap())?;
            let key = default_registry::join(&module, &name)?;
            b.set(&rc, key, locals.f.as_ref().unwrap().clone())?;
        }
    }
    Ok(rc)
}
pub fn build(bindings: &mut dyn Bindings) -> Result<Options, Error> {
    let mut locals = Locals::default();
    body(bindings, &mut locals).map_err(|error| {
        let frame = Frame::new("_get_user_allowed_globals");
        for (name, value) in [
            ("rc", locals.rc),
            ("f", locals.f),
            ("name", locals.name),
            ("module", locals.module),
        ] {
            if let Some(value) = value {
                frame.set(name, Rc::new(value));
            }
        }
        frame.finish();
        error.retain_frame(frame)
    })
}
