//! Original whole ModelPatcher.clone control flow. The canonical owner/factory,
//! model methods and native object protocols are required bindings; this does
//! not construct a second patcher identity or execute Python.
use super::{
    deepcopy::{DeepCopyBindings, deepcopy_list_dict},
    exception,
    model_patcher::{Initialization, PatcherState},
    options::Options,
    options_iterator::SequenceIterator,
    relationships::{RelationshipCloneBindings, RelationshipCloneLocals},
};
use crate::{execution_value::Value as V, native::Error};
use std::rc::Rc;

pub struct CloneArguments {
    pub disable_dynamic: Options,
    pub model_override: Options,
    pub force_deepcopy: Options,
}
impl Default for CloneArguments {
    fn default() -> Self {
        Self {
            disable_dynamic: Options::Atom(V::Bool(false)),
            model_override: Options::Atom(V::None),
            force_deepcopy: Options::Atom(V::Bool(false)),
        }
    }
}
pub trait CloneBindings: RelationshipCloneBindings + DeepCopyBindings {
    fn state<'a>(&self, owner: &'a Options) -> Result<&'a PatcherState, Error>;
    fn patcher_class(&self, owner: &Options) -> Result<Options, Error>;
    fn base_patcher_class(&self) -> Result<Options, Error>;
    fn is_dynamic(&self, owner: &Options) -> Result<Options, Error>;
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn index(&self, value: &Options, key: &Options) -> Result<Options, Error>;
    fn length(&self, value: &Options) -> Result<usize, Error>;
    fn unpack_backups(&self, value: &Options) -> Result<[Options; 4], Error>;
    fn factory(
        &self,
        callable: &Options,
        arguments: &Options,
        disable_dynamic: bool,
    ) -> Result<Options, Error>;
    fn construct(&self, class: &Options, arguments: &Initialization) -> Result<Options, Error>;
    fn model_size(&self, owner: &Options) -> Result<Options, Error>;
    fn model_override(&self, owner: &Options) -> Result<Options, Error>;
    fn slice_copy(&self, value: &Options) -> Result<Options, Error>;
    fn shallow_copy(&self, value: &Options) -> Result<Options, Error>;
    fn hook_patches_clone(&self, value: &Options) -> Result<Options, Error>;
    fn hook_group_clone(&self, value: &Options) -> Result<Options, Error>;
    fn clone_callbacks(&self, owner: &Options) -> Result<Options, Error>;
    fn call_callback(
        &self,
        callback: &Options,
        owner: &Options,
        target: &Options,
    ) -> Result<(), Error>;
}
fn none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
fn integer(value: i64) -> Options {
    Options::Atom(V::Int(value))
}

/// Explicit original fast locals. Loop assignments replace their old owners;
/// the last loop values remain alive until the complete clone frame exits.
#[derive(Default)]
struct Locals {
    class: Option<Options>,
    temporary: Option<Options>,
    target: Option<Options>,
    key: Option<Options>,
    callbacks: Option<Options>,
    inner_key: Option<Options>,
    inner_callbacks: Option<Options>,
    wrappers: Option<Options>,
    inner_wrappers: Option<Options>,
    injection: Option<Options>,
    group: Option<Options>,
    callback: Option<Options>,
}
fn assign(slot: &mut Option<Options>, value: Options) {
    drop(slot.replace(value));
}
fn pair(value: Options) -> (Options, Options) {
    let Options::Tuple(items) = value else {
        unreachable!("native dictionary items")
    };
    (items[0].clone(), items[1].clone())
}
pub fn clone_patcher(
    owner: &Options,
    arguments: &CloneArguments,
    bindings: &dyn CloneBindings,
) -> Result<Options, Error> {
    let source = bindings.state(owner)?;
    let mut locals = Locals::default();
    assign(&mut locals.class, bindings.patcher_class(owner)?);
    let mut model_override = arguments.model_override.clone();
    if (bindings.truth(&bindings.is_dynamic(owner)?)?
        && bindings.truth(&arguments.disable_dynamic)?)
        || bindings.truth(&arguments.force_deepcopy)?
    {
        if bindings.truth(&bindings.is_dynamic(owner)?)?
            && bindings.truth(&arguments.disable_dynamic)?
        {
            assign(&mut locals.class, bindings.base_patcher_class()?);
        }
        if none(&model_override) {
            if none(&source.fields.get("cached_patcher_init")?) {
                return Err(exception(
                    "RuntimeError",
                    "Cannot create non-dynamic delegate: cached_patcher_init is not initialized.",
                ));
            }
            let callable =
                bindings.index(&source.fields.get("cached_patcher_init")?, &integer(0))?;
            let args = bindings.index(&source.fields.get("cached_patcher_init")?, &integer(1))?;
            let temporary = bindings.factory(&callable, &args, true)?;
            drop(args);
            drop(callable);
            assign(&mut locals.temporary, temporary);
            if bindings.length(&source.fields.get("cached_patcher_init")?)? > 2 {
                let key =
                    bindings.index(&source.fields.get("cached_patcher_init")?, &integer(2))?;
                let value = bindings.index(locals.temporary.as_ref().unwrap(), &key)?;
                assign(&mut locals.temporary, value);
            }
            model_override = bindings.model_override(locals.temporary.as_ref().unwrap())?;
        }
    }
    if none(&model_override) {
        model_override = bindings.model_override(owner)?;
    }
    let args = Initialization {
        model: bindings.index(&model_override, &integer(0))?,
        load_device: source.fields.get("load_device")?,
        offload_device: source.fields.get("offload_device")?,
        size: bindings.model_size(owner)?,
        weight_inplace_update: source.fields.get("weight_inplace_update")?,
        fast_disk: source.fields.get("fast_disk")?,
    };
    let target = bindings.construct(locals.class.as_ref().unwrap(), &args)?;
    drop(args);
    assign(&mut locals.target, target);
    let target_owner = locals.target.as_ref().unwrap();
    let target = bindings.state(target_owner)?;
    target.fields.set("patches", bindings.empty_dictionary()?);
    let keys = source.fields.get("patches")?.keys_iterator()?;
    while let Some(key) = keys.next_value()? {
        assign(&mut locals.key, key);
        let value = bindings.slice_copy(
            &source
                .fields
                .get("patches")?
                .item(locals.key.as_ref().unwrap())?,
        )?;
        target
            .fields
            .get("patches")?
            .set_item(locals.key.as_ref().unwrap().clone(), value)?;
    }
    drop(keys);
    target
        .fields
        .set("patches_uuid", source.fields.get("patches_uuid")?);
    for name in ["object_patches", "weight_wrapper_patches"] {
        target
            .fields
            .set(name, bindings.shallow_copy(&source.fields.get(name)?)?);
    }
    target.fields.set(
        "model_options",
        deepcopy_list_dict(&source.fields.get("model_options")?, None, bindings)?,
    );
    target.fields.set("parent", owner.clone());
    target.fields.set(
        "force_cast_weights",
        source.fields.get("force_cast_weights")?,
    );
    let backups = bindings.index(&model_override, &integer(1))?;
    let [backup, buffers, objects, pinned] = bindings.unpack_backups(&backups)?;
    drop(backups);
    for (name, value) in [
        ("backup", backup),
        ("backup_buffers", buffers),
        ("object_patches_backup", objects),
        ("pinned", pinned),
    ] {
        target.fields.set(name, value);
    }
    let mut relationship_locals = RelationshipCloneLocals {
        key: locals.key.take(),
        group: locals.callbacks.take(),
    };
    let relationship_result = source.relationships.clone_members_into_with_locals(
        &target.relationships,
        bindings,
        &mut relationship_locals,
    );
    locals.key = relationship_locals.key;
    locals.callbacks = relationship_locals.group;
    relationship_result?;
    for name in ["callbacks", "wrappers"] {
        let items = source.fields.get(name)?.items_iterator()?;
        while let Some(item) = items.next_item()? {
            let (key, value) = pair(item);
            assign(&mut locals.key, key);
            let slot = if name == "callbacks" {
                &mut locals.callbacks
            } else {
                &mut locals.wrappers
            };
            assign(slot, value);
            let value = bindings.empty_dictionary()?;
            target
                .fields
                .get(name)?
                .set_item(locals.key.as_ref().unwrap().clone(), value)?;
            let inner = slot.as_ref().unwrap().items_iterator()?;
            while let Some(item) = inner.next_item()? {
                let (key, value) = pair(item);
                assign(&mut locals.inner_key, key);
                let slot = if name == "callbacks" {
                    &mut locals.inner_callbacks
                } else {
                    &mut locals.inner_wrappers
                };
                assign(slot, value);
                let value = bindings.shallow_copy(slot.as_ref().unwrap())?;
                target
                    .fields
                    .get(name)?
                    .item(locals.key.as_ref().unwrap())?
                    .set_item(locals.inner_key.as_ref().unwrap().clone(), value)?;
            }
        }
    }
    target
        .fields
        .set("is_injected", source.fields.get("is_injected")?);
    target
        .fields
        .set("skip_injection", source.fields.get("skip_injection")?);
    let items = source.fields.get("injections")?.items_iterator()?;
    while let Some(item) = items.next_item()? {
        let (key, value) = pair(item);
        assign(&mut locals.key, key);
        assign(&mut locals.injection, value);
        let value = bindings.shallow_copy(locals.injection.as_ref().unwrap())?;
        target
            .fields
            .get("injections")?
            .set_item(locals.key.as_ref().unwrap().clone(), value)?;
    }
    drop(items);
    target.fields.set(
        "hook_patches",
        bindings.hook_patches_clone(&source.fields.get("hook_patches")?)?,
    );
    let backup = if bindings.truth(&source.fields.get("hook_patches_backup")?)? {
        bindings.hook_patches_clone(&source.fields.get("hook_patches_backup")?)?
    } else {
        source.fields.get("hook_patches_backup")?
    };
    target.fields.set("hook_patches_backup", backup);
    let groups = source.fields.get("cached_hook_patches")?.keys_iterator()?;
    while let Some(group) = groups.next_value()? {
        assign(&mut locals.group, group);
        let value = bindings.empty_dictionary()?;
        target
            .fields
            .get("cached_hook_patches")?
            .set_item(locals.group.as_ref().unwrap().clone(), value)?;
        let keys = source
            .fields
            .get("cached_hook_patches")?
            .item(locals.group.as_ref().unwrap())?
            .keys_iterator()?;
        while let Some(key) = keys.next_value()? {
            assign(&mut locals.key, key);
            let value = source
                .fields
                .get("cached_hook_patches")?
                .item(locals.group.as_ref().unwrap())?
                .item(locals.key.as_ref().unwrap())?;
            target
                .fields
                .get("cached_hook_patches")?
                .item(locals.group.as_ref().unwrap())?
                .set_item(locals.key.as_ref().unwrap().clone(), value)?;
        }
    }
    drop(groups);
    target
        .fields
        .set("hook_backup", source.fields.get("hook_backup")?);
    for name in ["current_hooks", "forced_hooks"] {
        let value = if bindings.truth(&source.fields.get(name)?)? {
            bindings.hook_group_clone(&source.fields.get(name)?)?
        } else {
            source.fields.get(name)?
        };
        target.fields.set(name, value);
    }
    for name in [
        "is_clip",
        "hook_mode",
        "cached_patcher_init",
        "is_multigpu_base_clone",
        "clone_base_uuid",
    ] {
        target.fields.set(name, source.fields.get(name)?);
    }
    let callbacks = bindings.clone_callbacks(owner)?;
    let mut callbacks = SequenceIterator::for_operation(&callbacks, "model clone callbacks")?;
    while let Some(callback) = callbacks.next()? {
        assign(&mut locals.callback, callback);
        bindings.call_callback(locals.callback.as_ref().unwrap(), owner, target_owner)?;
    }
    drop(callbacks);
    Ok(target_owner.clone())
}

impl PatcherState {
    pub fn get_clone_model_override(&self) -> Result<Options, Error> {
        Ok(Options::Tuple(Rc::new(vec![
            self.fields.get("model")?,
            Options::Tuple(Rc::new(
                [
                    "backup",
                    "backup_buffers",
                    "object_patches_backup",
                    "pinned",
                ]
                .iter()
                .map(|name| self.fields.get(name))
                .collect::<Result<_, Error>>()?,
            )),
        ])))
    }
}
