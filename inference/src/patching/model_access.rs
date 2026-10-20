//! Source ModelPatcher attribute/size/clone queries. All model attributes,
//! arithmetic and comparisons use required native protocols.
use super::{
    model_patcher::PatcherState,
    options::{GeneralMapping, Options},
    options_iterator::SequenceIterator,
};
use crate::{execution_value::Value as V, native::Error};
use std::rc::Rc;
pub trait AccessBindings {
    fn name(&self, name: &str) -> Result<Options, Error>;
    fn attribute(&self, object: &Options, name: &Options) -> Result<Options, Error>;
    fn split_dots(&self, name: &Options) -> Result<Options, Error>;
    fn positive(&self, size: &Options) -> Result<bool, Error>;
    fn module_size(&self, model: &Options) -> Result<Options, Error>;
    fn call_get_dtype(&self, method: &Options) -> Result<Options, Error>;
    fn equal(&self, left: &Options, right: &Options) -> Result<bool, Error>;
    fn not_equal(&self, left: &Options, right: &Options) -> Result<bool, Error>;
    fn length(&self, value: &Options) -> Result<usize, Error>;
    fn warning(&self, message: &str) -> Result<(), Error>;
}
fn attribute(object: Options, name: &str, bindings: &dyn AccessBindings) -> Result<Options, Error> {
    bindings.attribute(&object, &bindings.name(name)?)
}
pub fn model_size(state: &PatcherState, bindings: &dyn AccessBindings) -> Result<Options, Error> {
    if bindings.positive(&state.fields.get("size")?)? {
        return state.fields.get("size");
    }
    let size = bindings.module_size(&state.fields.get("model")?)?;
    state.fields.set("size", size);
    state.fields.get("size")
}
pub fn loaded_size(state: &PatcherState, bindings: &dyn AccessBindings) -> Result<Options, Error> {
    attribute(
        state.fields.get("model")?,
        "model_loaded_weight_memory",
        bindings,
    )
}
pub fn lowvram_patch_counter(
    state: &PatcherState,
    bindings: &dyn AccessBindings,
) -> Result<Options, Error> {
    attribute(
        state.fields.get("model")?,
        "lowvram_patch_counter",
        bindings,
    )
}
pub fn get_clone_model_override(state: &PatcherState) -> Result<Options, Error> {
    let model = state.fields.get("model")?;
    let backups = [
        "backup",
        "backup_buffers",
        "object_patches_backup",
        "pinned",
    ]
    .into_iter()
    .map(|name| state.fields.get(name))
    .collect::<Result<Vec<_>, _>>()?;
    Ok(Options::Tuple(Rc::new(vec![
        model,
        Options::Tuple(Rc::new(backups)),
    ])))
}
pub fn get_attr(
    mut object: Options,
    name: &Options,
    bindings: &dyn AccessBindings,
) -> Result<Options, Error> {
    let parts = bindings.split_dots(name)?;
    let mut parts = SequenceIterator::for_operation(&parts, "model attribute path")?;
    let mut last = None;
    while let Some(name) = parts.next()? {
        drop(last.replace(name));
        object = bindings.attribute(&object, last.as_ref().unwrap())?;
    }
    Ok(object)
}
pub fn get_model_object(
    state: &PatcherState,
    name: &Options,
    bindings: &dyn AccessBindings,
) -> Result<Options, Error> {
    if state.fields.get("object_patches")?.contains_item(name)? {
        return state.fields.get("object_patches")?.item(name);
    }
    if state
        .fields
        .get("object_patches_backup")?
        .contains_item(name)?
    {
        return state.fields.get("object_patches_backup")?.item(name);
    }
    get_attr(state.fields.get("model")?, name, bindings)
}
pub fn model_dtype(state: &PatcherState, bindings: &dyn AccessBindings) -> Result<Options, Error> {
    let method = bindings.name("get_dtype")?;
    let model = state.fields.get("model")?;
    let found = match bindings.attribute(&model, &method) {
        Ok(value) => {
            drop(value);
            true
        }
        Err(caught) if caught.is_reference("AttributeError") => false,
        Err(error) => return Err(error),
    };
    drop(model);
    if !found {
        return Ok(Options::Atom(V::None));
    }
    let model = state.fields.get("model")?;
    let callable = bindings.attribute(&model, &method);
    drop(model);
    bindings.call_get_dtype(&callable?)
}
pub fn is_clone(
    state: &PatcherState,
    other: &Options,
    bindings: &dyn AccessBindings,
) -> Result<bool, Error> {
    let name = bindings.name("model")?;
    match bindings.attribute(other, &name) {
        Ok(value) => drop(value),
        Err(caught) if caught.is_reference("AttributeError") => return Ok(false),
        Err(error) => return Err(error),
    }
    let model = state.fields.get("model")?;
    Ok(model.same_object(&bindings.attribute(other, &name)?))
}
struct KeyView(Options);
fn keys(value: Options) -> Result<KeyView, Error> {
    value.dictionary_method("keys")?;
    Ok(KeyView(value))
}
fn keys_differ(left: KeyView, right: KeyView) -> Result<bool, Error> {
    let (left, right) = (left.0, right.0);
    fn len(value: &Options) -> usize {
        if let Some(mapping) = GeneralMapping::get(value) {
            mapping.dictionary.len()
        } else {
            let Options::Dict(value) = value else {
                unreachable!()
            };
            value.borrow().len()
        }
    }
    if len(&left) != len(&right) {
        return Ok(true);
    }
    let keys = left.keys_iterator()?;
    while let Some(key) = keys.next_value()? {
        if !right.contains_item(&key)? {
            return Ok(true);
        }
    }
    Ok(false)
}
pub fn clone_has_same_weights(
    state: &PatcherState,
    clone: &Options,
    allow_multigpu: bool,
    bindings: &dyn AccessBindings,
) -> Result<Options, Error> {
    let field = |name: &str| attribute(clone.clone(), name, bindings);
    let boolean = |v| Options::Atom(V::Bool(v));
    if allow_multigpu {
        if bindings.not_equal(
            &state.fields.get("clone_base_uuid")?,
            &field("clone_base_uuid")?,
        )? {
            return Ok(boolean(false));
        }
    } else if !is_clone(state, clone, bindings)? {
        return Ok(boolean(false));
    }
    for name in ["current_hooks", "forced_hooks"] {
        if bindings.not_equal(&state.fields.get(name)?, &field(name)?)? {
            return Ok(boolean(false));
        }
    }
    for name in ["hook_patches", "attachments", "additional_models"] {
        if keys_differ(keys(state.fields.get(name)?)?, keys(field(name)?)?)? {
            return Ok(boolean(false));
        }
    }
    let mut last = None;
    for name in ["callbacks", "wrappers"] {
        let keys = state.fields.get(name)?.keys_iterator()?;
        while let Some(key) = keys.next_value()? {
            drop(last.replace(key));
            let key = last.as_ref().unwrap();
            let left = bindings.length(&state.fields.get(name)?.item(key)?)?;
            let right = bindings.length(&field(name)?.item(key)?)?;
            if left != right {
                return Ok(boolean(false));
            }
        }
    }
    if keys_differ(
        keys(state.fields.get("injections")?)?,
        keys(field("injections")?)?,
    )? {
        return Ok(boolean(false));
    }
    if bindings.length(&state.fields.get("patches")?)? == 0
        && bindings.length(&field("patches")?)? == 0
    {
        return Ok(boolean(true));
    }
    if bindings.equal(&state.fields.get("patches_uuid")?, &field("patches_uuid")?)? {
        if bindings.length(&state.fields.get("patches")?)? != bindings.length(&field("patches")?)? {
            bindings.warning(
                "WARNING: something went wrong, same patch uuid but different length of patches.",
            )?;
        } else {
            return Ok(boolean(true));
        }
    }
    Ok(Options::Atom(V::None))
}
