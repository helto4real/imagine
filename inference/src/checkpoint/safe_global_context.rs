//! Unmodified _safe_globals method bodies over the actual native object/caller
//! protocols. Helpers are captured before reading self.safe_globals.
use crate::{
    native::Error,
    patching::options::{Options, SourceNone},
    source_exception::Frame,
};
use std::rc::Rc;
pub trait Bindings {
    fn helper(&mut self, name: &str) -> Result<Options, Error>;
    fn attribute(&mut self, object: &Options, name: &str) -> Result<Options, Error>;
    fn set_attribute(&mut self, object: &Options, name: &str, value: Options) -> Result<(), Error>;
    fn call(&mut self, function: &Options, args: &[Options]) -> Result<Options, Error>;
}
fn retain(error: Error, name: &str, locals: &[(&str, &Options)]) -> Error {
    let frame = Frame::new(name);
    for (name, value) in locals {
        frame.set(*name, Rc::new((*value).clone()));
    }
    frame.finish();
    error.retain_frame(frame)
}
pub fn initialize(
    object: &Options,
    args: &Options,
    b: &mut dyn Bindings,
) -> Result<Options, Error> {
    b.set_attribute(object, "safe_globals", args.clone())
        .map(|()| SourceNone::value())
        .map_err(|error| {
            retain(
                error,
                "__init__",
                &[("self", object), ("safe_globals", args)],
            )
        })
}
fn invoke(object: &Options, name: &str, b: &mut dyn Bindings) -> Result<Options, Error> {
    let function = b.helper(name)?;
    let args = match b.attribute(object, "safe_globals") {
        Ok(args) => args,
        Err(error) => {
            drop(function);
            return Err(error);
        }
    };
    let result = b.call(&function, std::slice::from_ref(&args));
    drop(function);
    drop(args);
    // The returned helper value is discarded; it does not become the with target.
    drop(result?);
    Ok(SourceNone::value())
}
pub fn enter(object: &Options, b: &mut dyn Bindings) -> Result<Options, Error> {
    invoke(object, "_add_safe_globals", b)
        .map_err(|error| retain(error, "__enter__", &[("self", object)]))
}
pub fn exit(
    object: &Options,
    kind: &Options,
    value: &Options,
    tb: &Options,
    b: &mut dyn Bindings,
) -> Result<Options, Error> {
    invoke(object, "_remove_safe_globals", b).map_err(|error| {
        retain(
            error,
            "__exit__",
            &[
                ("self", object),
                ("type", kind),
                ("value", value),
                ("tb", tb),
            ],
        )
    })
}
