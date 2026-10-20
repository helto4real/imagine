//! Original multigpu deepclone body. Fresh loader factories and recursive
//! child dispatch are required native bindings; a shared model is never used
//! as a substitute for newly loaded pristine weights.
use super::{
    exception, model_patcher::PatcherState, options::Options, options_iterator::SequenceIterator,
};
use crate::{execution_value::Value as V, native::Error};
use std::rc::Rc;

pub trait DeepCloneBindings {
    fn state<'a>(&self, owner: &'a Options) -> Result<&'a PatcherState, Error>;
    fn model_class_name(&self, model: &Options) -> Result<String, Error>;
    fn format_device(&self, device: &Options) -> Result<String, Error>;
    fn info(&self, message: &str) -> Result<(), Error>;
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn unload_model_and_clones(&self, owner: &Options) -> Result<(), Error>;
    /// The original call supplies positional factory arguments and no
    /// disable_dynamic keyword (unlike the force-deepcopy clone branch).
    fn factory(&self, callable: &Options, arguments: &Options) -> Result<Options, Error>;
    fn index(&self, value: &Options, key: &Options) -> Result<Options, Error>;
    fn length(&self, value: &Options) -> Result<usize, Error>;
    fn empty_dictionary(&self) -> Result<Options, Error>;
    fn empty_pinned_set(&self) -> Result<Options, Error>;
    fn clone_with_override(
        &self,
        owner: &Options,
        model_override: &Options,
    ) -> Result<Options, Error>;
    fn attribute(&self, owner: &Options, name: &str) -> Result<Options, Error>;
    fn register_load_device(&self, callable: &Options, device: &Options) -> Result<(), Error>;
    fn cache_contains(&self, cache: &Options, key: &Options) -> Result<bool, Error>;
    fn deepclone_child(
        &self,
        child: &Options,
        device: &Options,
        cache: &Options,
    ) -> Result<Options, Error>;
    fn set_index(&self, container: &Options, key: &Options, value: Options) -> Result<(), Error>;
    fn callbacks(&self, owner: &Options) -> Result<Options, Error>;
    fn callback(&self, callback: &Options, owner: &Options, target: &Options) -> Result<(), Error>;
}
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
fn integer(value: i64) -> Options {
    Options::Atom(V::Int(value))
}
#[derive(Default)]
struct Locals {
    temporary: Option<Options>,
    model_override: Option<Options>,
    target: Option<Options>,
    key: Option<Options>,
    model_list: Option<Options>,
    index: Option<Options>,
    additional: Option<Options>,
    callback: Option<Options>,
}
fn assign(slot: &mut Option<Options>, value: Options) {
    drop(slot.replace(value));
}
pub fn deepclone_multigpu(
    owner: &Options,
    new_load_device: &Options,
    models_cache: &Options,
    bindings: &dyn DeepCloneBindings,
) -> Result<Options, Error> {
    let source = bindings.state(owner)?;
    let class_name = bindings.model_class_name(&source.fields.get("model")?)?;
    let device = if bindings.truth(new_load_device)? {
        new_load_device.clone()
    } else {
        source.fields.get("load_device")?
    };
    let message = format!(
        "Creating deepclone of {class_name} for {}.",
        bindings.format_device(&device)?
    );
    drop(device);
    bindings.info(&message)?;
    if none(&source.fields.get("cached_patcher_init")?) {
        let class_name = bindings.model_class_name(&source.fields.get("model")?)?;
        return Err(exception(
            "RuntimeError",
            format!(
                "Cannot create multigpu deepclone of {class_name}: the loader that produced this model does not support multigpu (cached_patcher_init is not initialized). Use a core loader (CheckpointLoaderSimple, UNETLoader, CLIPLoader/DualCLIPLoader, VAELoader), or have the custom loader register a cached_patcher_init factory."
            ),
        ));
    }
    bindings.unload_model_and_clones(owner)?;
    let mut locals = Locals::default();
    let callable = bindings.index(&source.fields.get("cached_patcher_init")?, &integer(0))?;
    let arguments = bindings.index(&source.fields.get("cached_patcher_init")?, &integer(1))?;
    let temporary = bindings.factory(&callable, &arguments)?;
    drop(arguments);
    drop(callable);
    assign(&mut locals.temporary, temporary);
    if bindings.length(&source.fields.get("cached_patcher_init")?)? > 2 {
        let key = bindings.index(&source.fields.get("cached_patcher_init")?, &integer(2))?;
        let temporary = bindings.index(locals.temporary.as_ref().unwrap(), &key)?;
        assign(&mut locals.temporary, temporary);
    }
    let model = bindings.attribute(locals.temporary.as_ref().unwrap(), "model")?;
    let backups = Options::Tuple(Rc::new(vec![
        bindings.empty_dictionary()?,
        bindings.empty_dictionary()?,
        bindings.empty_dictionary()?,
        bindings.empty_pinned_set()?,
    ]));
    assign(
        &mut locals.model_override,
        Options::Tuple(Rc::new(vec![model, backups])),
    );
    assign(
        &mut locals.target,
        bindings.clone_with_override(owner, locals.model_override.as_ref().unwrap())?,
    );
    let target_owner = locals.target.as_ref().unwrap();
    let target = bindings.state(target_owner)?;
    target
        .fields
        .set("hook_backup", bindings.empty_dictionary()?);
    if !none(new_load_device) {
        target.fields.set("load_device", new_load_device.clone());
    }
    let has_register = match bindings.attribute(target_owner, "register_load_device") {
        Ok(attribute) => {
            drop(attribute);
            true
        }
        Err(caught) if caught.is_reference("AttributeError") => false,
        Err(error) => return Err(error),
    };
    if has_register {
        let callable = bindings.attribute(target_owner, "register_load_device")?;
        let device = target.fields.get("load_device")?;
        let result = bindings.register_load_device(&callable, &device);
        drop(device);
        drop(callable);
        result?;
    }
    target.relationships.remove_additional_models("multigpu")?;
    let cache = if none(models_cache) {
        bindings.empty_dictionary()?
    } else {
        models_cache.clone()
    };
    let items = target.fields.get("additional_models")?.items_iterator()?;
    while let Some(item) = items.next_item()? {
        let Options::Tuple(values) = item else {
            unreachable!("dictionary item pair")
        };
        assign(&mut locals.key, values[0].clone());
        assign(&mut locals.model_list, values[1].clone());
        drop(values);
        let count = bindings.length(locals.model_list.as_ref().unwrap())?;
        for i in 0..count {
            assign(&mut locals.index, integer(i as i64));
            let group = bindings.index(
                &target.fields.get("additional_models")?,
                locals.key.as_ref().unwrap(),
            )?;
            let child = bindings.index(&group, locals.index.as_ref().unwrap())?;
            drop(group);
            assign(&mut locals.additional, child);
            let child = locals.additional.as_ref().unwrap();
            let key = bindings.attribute(child, "clone_base_uuid")?;
            let missing = !bindings.cache_contains(&cache, &key)?;
            drop(key);
            if missing {
                // No early placeholder: source inserts only after the child's
                // recursive call returns, retaining its original cycle/error behavior.
                let cloned = bindings.deepclone_child(child, new_load_device, &cache)?;
                let key = bindings.attribute(child, "clone_base_uuid")?;
                cache.set_item(key, cloned)?;
            }
            let key = bindings.attribute(child, "clone_base_uuid")?;
            let cloned = bindings.index(&cache, &key)?;
            drop(key);
            let group = bindings.index(
                &target.fields.get("additional_models")?,
                locals.key.as_ref().unwrap(),
            )?;
            bindings.set_index(&group, locals.index.as_ref().unwrap(), cloned)?;
        }
    }
    drop(items);
    let mut callbacks = SequenceIterator::for_operation(
        &bindings.callbacks(owner)?,
        "multigpu deepclone callbacks",
    )?;
    while let Some(callback) = callbacks.next()? {
        assign(&mut locals.callback, callback);
        bindings.callback(locals.callback.as_ref().unwrap(), owner, target_owner)?;
    }
    Ok(target_owner.clone())
}
