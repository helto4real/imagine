//! Original sampler cast_to_load_options/cast_transformer_options. One hasattr
//! guards all casts for a slot; later casts read back its replaced value.
use crate::{
    execution_value::Value as V,
    native::Error,
    patching::{
        model_patch_calls::PatchCallBindings, options::Options, options_iterator::SequenceIterator,
    },
};
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
#[derive(Default)]
struct Locals {
    patches: Option<Options>,
    name: Option<Options>,
    patch_list: Option<Options>,
    index: Option<Options>,
    key: Option<Options>,
    cast: Option<Options>,
    registry: Option<Options>,
    group: Option<Options>,
    list: Option<Options>,
}
fn assign(slot: &mut Option<Options>, value: Options) {
    drop(slot.replace(value));
}
fn cast_slot(
    container: &Options,
    key: &Options,
    casts: &[Options],
    cast: &mut Option<Options>,
    bindings: &dyn PatchCallBindings,
) -> Result<(), Error> {
    let name = bindings.name("to");
    let receiver = bindings.item(container, key)?;
    let has_attribute = match bindings.attribute(&receiver, &name) {
        Ok(attribute) => {
            drop(attribute);
            true
        }
        Err(caught) if caught.is_reference("AttributeError") => false,
        Err(error) => return Err(error),
    };
    drop(receiver);
    if has_attribute {
        for value in casts {
            assign(cast, value.clone());
            let receiver = bindings.item(container, key)?;
            let callable = bindings.attribute(&receiver, &name)?;
            drop(receiver);
            let value = bindings.call(
                &callable,
                std::slice::from_ref(cast.as_ref().unwrap()),
                None,
            );
            drop(callable);
            bindings.set_item(container, key, value?)?;
        }
    }
    Ok(())
}
pub fn cast_transformer_options(
    transformer: &Options,
    device: &Options,
    dtype: &Options,
    bindings: &dyn PatchCallBindings,
) -> Result<(), Error> {
    let mut casts = vec![];
    if !none(device) {
        casts.push(device.clone());
    }
    if !none(dtype) {
        casts.push(dtype.clone());
    }
    if casts.is_empty() {
        return Ok(());
    }
    let mut locals = Locals::default();
    for (name, replacement) in [("patches", false), ("patches_replace", true)] {
        if !bindings.contains(transformer, name)? {
            continue;
        }
        assign(
            &mut locals.patches,
            bindings.item(transformer, &bindings.name(name))?,
        );
        let mut names = SequenceIterator::for_operation(
            locals.patches.as_ref().unwrap(),
            "transformer cast patches",
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
            if replacement {
                let mut keys = SequenceIterator::for_operation(
                    locals.patch_list.as_ref().unwrap(),
                    "replacement cast patches",
                )?;
                while let Some(key) = keys.next()? {
                    assign(&mut locals.key, key);
                    cast_slot(
                        locals.patch_list.as_ref().unwrap(),
                        locals.key.as_ref().unwrap(),
                        &casts,
                        &mut locals.cast,
                        bindings,
                    )?;
                }
            } else {
                let count = bindings.length(locals.patch_list.as_ref().unwrap())?;
                for i in 0..count {
                    assign(&mut locals.index, Options::Atom(V::Int(i as i64)));
                    cast_slot(
                        locals.patch_list.as_ref().unwrap(),
                        locals.index.as_ref().unwrap(),
                        &casts,
                        &mut locals.cast,
                        bindings,
                    )?;
                }
            }
        }
    }
    for name in ["wrappers", "callbacks"] {
        if !bindings.contains(transformer, name)? {
            continue;
        }
        assign(
            &mut locals.registry,
            bindings.item(transformer, &bindings.name(name))?,
        );
        let groups = locals.registry.as_ref().unwrap().values_iterator()?;
        while let Some(group) = groups.next_value()? {
            assign(&mut locals.group, group);
            let lists = locals.group.as_ref().unwrap().values_iterator()?;
            while let Some(list) = lists.next_value()? {
                assign(&mut locals.list, list);
                let count = bindings.length(locals.list.as_ref().unwrap())?;
                for i in 0..count {
                    assign(&mut locals.index, Options::Atom(V::Int(i as i64)));
                    cast_slot(
                        locals.list.as_ref().unwrap(),
                        locals.index.as_ref().unwrap(),
                        &casts,
                        &mut locals.cast,
                        bindings,
                    )?;
                }
            }
        }
    }
    Ok(())
}
pub fn cast_to_load_options(
    options: &Options,
    device: &Options,
    dtype: &Options,
    bindings: &dyn PatchCallBindings,
) -> Result<(), Error> {
    if none(options) {
        return Ok(());
    }
    let transformer = options
        .get("to_load_options")?
        .unwrap_or(Options::Atom(V::None));
    if none(&transformer) {
        return Ok(());
    }
    cast_transformer_options(&transformer, device, dtype, bindings)
}
