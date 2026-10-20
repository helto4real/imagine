//! Original base/dynamic owner lifecycle bodies. Bindings are bound to the
//! actual canonical owner, including during its destructor when no new strong
//! owner handle can be manufactured.
use super::{model_patcher::PatcherState, options::Options, options_iterator::SequenceIterator};
use crate::{execution_value::Value as V, native::Error};

pub trait LifecycleBindings {
    fn eject_model(&self) -> Result<(), Error>;
    fn model_patches_to(&self, device: &Options) -> Result<(), Error>;
    fn unpatch_model(&self, device: &Options, unpatch_weights: &Options) -> Result<(), Error>;
    fn model_patches_cleanup(&self) -> Result<(), Error>;
    fn clean_hooks(&self) -> Result<(), Error>;
    fn unpin_all_weights(&self) -> Result<(), Error>;
    fn detach(&self, unpatch_all: bool) -> Result<(), Error>;
    fn attribute(&self, object: &Options, name: &str) -> Result<Options, Error>;
    fn set_current_patcher(&self, model: &Options, present: bool) -> Result<(), Error>;
    fn callbacks(&self, kind: &str) -> Result<Options, Error>;
    fn callback(&self, callback: &Options, unpatch_all: Option<&Options>) -> Result<(), Error>;
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn contains_self_identity(&self, collection: &Options) -> Result<bool, Error>;
    fn discard_self_identity(&self, collection: &Options) -> Result<(), Error>;
}
fn none() -> Options {
    Options::Atom(V::None)
}
fn is_none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
fn attribute_or_none(result: Result<Options, Error>) -> Result<Options, Error> {
    match result {
        Err(caught) if caught.is_reference("AttributeError") => Ok(none()),
        value => value,
    }
}
fn has_current_patcher(
    state: &PatcherState,
    bindings: &dyn LifecycleBindings,
) -> Result<bool, Error> {
    let model = state.fields.get("model")?;
    let result = match bindings.attribute(&model, "current_patcher") {
        Ok(value) => {
            drop(value);
            Ok(true)
        }
        Err(caught) if caught.is_reference("AttributeError") => Ok(false),
        Err(error) => Err(error),
    };
    drop(model);
    result
}
fn callbacks(
    kind: &str,
    unpatch_all: Option<&Options>,
    bindings: &dyn LifecycleBindings,
) -> Result<(), Error> {
    let mut callbacks =
        SequenceIterator::for_operation(&bindings.callbacks(kind)?, "patcher lifecycle callbacks")?;
    let mut last = None;
    while let Some(callback) = callbacks.next()? {
        drop(last.replace(callback));
        bindings.callback(last.as_ref().unwrap(), unpatch_all)?;
    }
    Ok(())
}
pub fn detach(
    state: &PatcherState,
    unpatch_all: &Options,
    bindings: &dyn LifecycleBindings,
) -> Result<Options, Error> {
    bindings.eject_model()?;
    bindings.model_patches_to(&state.fields.get("offload_device")?)?;
    if bindings.truth(unpatch_all)? {
        bindings.unpatch_model(&state.fields.get("offload_device")?, unpatch_all)?;
    }
    // The final callback local survives the final self.model lookup.
    let mut callbacks = SequenceIterator::for_operation(
        &bindings.callbacks(super::extensions::ON_DETACH)?,
        "patcher detach callbacks",
    )?;
    let mut last = None;
    while let Some(callback) = callbacks.next()? {
        drop(last.replace(callback));
        bindings.callback(last.as_ref().unwrap(), Some(unpatch_all))?;
    }
    state.fields.get("model")
}
pub fn cleanup(state: &PatcherState, bindings: &dyn LifecycleBindings) -> Result<(), Error> {
    bindings.model_patches_cleanup()?;
    bindings.clean_hooks()?;
    if has_current_patcher(state, bindings)? {
        bindings.set_current_patcher(&state.fields.get("model")?, false)?;
    }
    callbacks("on_cleanup", None, bindings)
}
pub fn pre_run(state: &PatcherState, bindings: &dyn LifecycleBindings) -> Result<(), Error> {
    if has_current_patcher(state, bindings)? {
        bindings.set_current_patcher(&state.fields.get("model")?, true)?;
    }
    callbacks("on_pre_run", None, bindings)
}
pub fn finalize_base(bindings: &dyn LifecycleBindings) -> Result<(), Error> {
    bindings.unpin_all_weights()?;
    bindings.detach(false)
}
pub fn finalize_dynamic(
    state: &PatcherState,
    bindings: &dyn LifecycleBindings,
) -> Result<(), Error> {
    let model = attribute_or_none(state.fields.get("model"))?;
    let patchers = attribute_or_none(bindings.attribute(&model, "dynamic_patchers"))?;
    if is_none(&patchers) || !bindings.contains_self_identity(&patchers)? {
        return Ok(());
    }
    bindings.discard_self_identity(&patchers)?;
    let result = (|| {
        if !bindings.truth(&patchers)? {
            bindings.unpin_all_weights()?;
        }
        Ok(())
    })();
    // Source finally always runs after the discard, even if truth/unpin fails;
    // a detach failure replaces the previous exception.
    crate::source_exception::finally(result, || bindings.detach(false))
}
