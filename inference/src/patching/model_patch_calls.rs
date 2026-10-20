//! Original ModelPatcher patch traversal shared by loading and cleanup.
//! Attribute lookup, calls and list extension use required native protocols;
//! no callback is replaced with a default or executed through Python.
use super::{model_patcher::PatcherState, options::Options, options_iterator::SequenceIterator};
use crate::{execution_value::Value as V, native::Error};
use std::{cell::RefCell, rc::Rc};

pub trait PatchCallBindings {
    fn name(&self, name: &str) -> Options;
    fn contains(&self, container: &Options, key: &str) -> Result<bool, Error>;
    fn item(&self, container: &Options, key: &Options) -> Result<Options, Error>;
    fn set_item(&self, container: &Options, key: &Options, value: Options) -> Result<(), Error>;
    fn length(&self, container: &Options) -> Result<usize, Error>;
    fn attribute(&self, object: &Options, name: &Options) -> Result<Options, Error>;
    fn call(
        &self,
        callable: &Options,
        positional: &[Options],
        keywords: Option<&Options>,
    ) -> Result<Options, Error>;
    /// Original list += iterable, including partial extension before failure.
    fn extend(&self, list: &Options, iterable: &Options) -> Result<(), Error>;
}

enum Operation<'a> {
    To(&'a Options),
    Models,
    Call(&'a Options),
}
#[derive(Default)]
struct Locals {
    patches: Option<Options>,
    name: Option<Options>,
    patch_list: Option<Options>,
    index: Option<Options>,
    key: Option<Options>,
    wrapper: Option<Options>,
}
fn assign(slot: &mut Option<Options>, value: Options) {
    drop(slot.replace(value));
}
fn has_attribute(
    receiver: Options,
    name: &Options,
    bindings: &dyn PatchCallBindings,
) -> Result<bool, Error> {
    let outcome = match bindings.attribute(&receiver, name) {
        Ok(attribute) => {
            drop(attribute);
            Ok(true)
        }
        Err(caught) if caught.is_reference("AttributeError") => Ok(false),
        Err(error) => Err(error),
    };
    drop(receiver);
    outcome
}
fn invoke(
    receiver: Options,
    method: &Options,
    operation: &Operation<'_>,
    models: &Option<Options>,
    bindings: &dyn PatchCallBindings,
) -> Result<Option<Options>, Error> {
    let callable = bindings.attribute(&receiver, method)?;
    drop(receiver);
    let result = match operation {
        Operation::To(device) => bindings.call(&callable, &[(*device).clone()], None),
        Operation::Models => bindings.call(&callable, &[], None),
        Operation::Call(arguments) => bindings.call(&callable, &[], Some(arguments)),
    };
    drop(callable);
    let result = result?;
    match operation {
        Operation::To(_) => Ok(Some(result)),
        Operation::Models => {
            bindings.extend(models.as_ref().unwrap(), &result)?;
            Ok(None)
        }
        Operation::Call(_) => Ok(None),
    }
}
fn traverse(
    state: &PatcherState,
    method: &Options,
    operation: Operation<'_>,
    bindings: &dyn PatchCallBindings,
) -> Result<Option<Options>, Error> {
    let options = state.fields.get("model_options")?;
    let transformer = bindings.item(&options, &bindings.name("transformer_options"))?;
    drop(options);
    // Allocated after the original transformer_options lookup.
    let models = matches!(operation, Operation::Models)
        .then(|| Options::List(Rc::new(RefCell::new(vec![]))));
    let mut locals = Locals::default();
    if bindings.contains(&transformer, "patches")? {
        assign(
            &mut locals.patches,
            bindings.item(&transformer, &bindings.name("patches"))?,
        );
        let mut names =
            SequenceIterator::for_operation(locals.patches.as_ref().unwrap(), "model patches")?;
        while let Some(name) = names.next()? {
            assign(&mut locals.name, name);
            assign(
                &mut locals.patch_list,
                bindings.item(
                    locals.patches.as_ref().unwrap(),
                    locals.name.as_ref().unwrap(),
                )?,
            );
            // range(len(...)) captures length, but each subsequent subscript
            // reads the live list again, including after descriptor callbacks.
            let count = bindings.length(locals.patch_list.as_ref().unwrap())?;
            for i in 0..count {
                assign(&mut locals.index, Options::Atom(V::Int(i as i64)));
                let list = locals.patch_list.as_ref().unwrap();
                let key = locals.index.as_ref().unwrap();
                if has_attribute(bindings.item(list, key)?, method, bindings)? {
                    let result = invoke(
                        bindings.item(list, key)?,
                        method,
                        &operation,
                        &models,
                        bindings,
                    )?;
                    if let Some(result) = result {
                        bindings.set_item(list, key, result)?;
                    }
                }
            }
        }
    }
    if bindings.contains(&transformer, "patches_replace")? {
        assign(
            &mut locals.patches,
            bindings.item(&transformer, &bindings.name("patches_replace"))?,
        );
        let mut names = SequenceIterator::for_operation(
            locals.patches.as_ref().unwrap(),
            "replacement patches",
        )?;
        while let Some(name) = names.next()? {
            assign(&mut locals.name, name);
            assign(
                &mut locals.patch_list,
                bindings.item(
                    locals.patches.as_ref().unwrap(),
                    locals.name.as_ref().unwrap(),
                )?,
            );
            let mut keys = SequenceIterator::for_operation(
                locals.patch_list.as_ref().unwrap(),
                "replacement patch group",
            )?;
            while let Some(key) = keys.next()? {
                assign(&mut locals.key, key);
                let list = locals.patch_list.as_ref().unwrap();
                let key = locals.key.as_ref().unwrap();
                if has_attribute(bindings.item(list, key)?, method, bindings)? {
                    let result = invoke(
                        bindings.item(list, key)?,
                        method,
                        &operation,
                        &models,
                        bindings,
                    )?;
                    if let Some(result) = result {
                        bindings.set_item(list, key, result)?;
                    }
                }
            }
        }
    }
    if bindings.contains(
        &state.fields.get("model_options")?,
        "model_function_wrapper",
    )? {
        assign(
            &mut locals.wrapper,
            bindings.item(
                &state.fields.get("model_options")?,
                &bindings.name("model_function_wrapper"),
            )?,
        );
        if has_attribute(locals.wrapper.as_ref().unwrap().clone(), method, bindings)? {
            let result = invoke(
                locals.wrapper.as_ref().unwrap().clone(),
                method,
                &operation,
                &models,
                bindings,
            )?;
            if let Some(result) = result {
                bindings.set_item(
                    &state.fields.get("model_options")?,
                    &bindings.name("model_function_wrapper"),
                    result,
                )?;
            }
        }
    }
    Ok(models)
}
pub fn model_patches_to(
    state: &PatcherState,
    device: &Options,
    bindings: &dyn PatchCallBindings,
) -> Result<(), Error> {
    traverse(state, &bindings.name("to"), Operation::To(device), bindings)?;
    Ok(())
}
pub fn model_patches_models(
    state: &PatcherState,
    bindings: &dyn PatchCallBindings,
) -> Result<Options, Error> {
    Ok(
        traverse(state, &bindings.name("models"), Operation::Models, bindings)?
            .expect("source models list"),
    )
}
/// The caller supplies the actual source function_name/default keyword object;
/// this layer does not allocate a replacement for a shared default argument.
pub fn model_patches_call_function(
    state: &PatcherState,
    function_name: &Options,
    arguments: &Options,
    bindings: &dyn PatchCallBindings,
) -> Result<(), Error> {
    traverse(state, function_name, Operation::Call(arguments), bindings)?;
    Ok(())
}
