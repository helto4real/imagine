//! Original sampler_helpers.get_hooks_from_cond, with required native control
//! attribute/hash/recursion bindings. See the bounded D-05 control-order record.
pub use crate::python_hash::{identity_hash_address, object_identity_hash};
use crate::source_set::Table as IdentitySet;
use crate::{
    execution_value::Value as V,
    native::Error,
    patching::{
        exception,
        hooks::{HookGroup, HookHandle},
        options::Options,
    },
};
use std::{cell::RefCell, rc::Rc};

pub type Group = Rc<RefCell<HookGroup>>;
pub trait ControlHookBindings {
    /// Native ControlBase owners use object_identity_hash on their canonical Rc.
    /// A controlled oracle binding may replay captured original object hashes.
    fn control_hash(&self, control: &Options) -> Result<i64, Error>;
    /// Compare canonical control owners, not temporary Options/trait wrappers.
    /// ControlBase inherits object identity equality; overrides need their own
    /// original rich-comparison binding and are not silently treated as identity.
    fn same_control(&self, a: &Options, b: &Options) -> bool;
    fn extra_hooks(&self, control: &Options) -> Result<Option<Group>, Error>;
    fn previous_controlnet(&self, control: &Options) -> Result<Option<Options>, Error>;
    /// The common execution context owns the source recursion budget. These
    /// required bindings prevent a cycle becoming an unbounded native recursion.
    fn enter_control_recursion(&self) -> Result<(), Error>;
    fn leave_control_recursion(&self);
    fn source_locals(&self, _: &ControlHookLocals, _: &str) -> Result<(), Error> {
        Ok(())
    }
}
#[derive(Default)]
pub struct ControlHookLocals {
    pub record: Option<Options>,
    pub hook: Option<HookHandle>,
    pub base_control: Option<Options>,
    pub hook_groups: Vec<Option<Group>>,
    pub extra_hooks: Option<Group>,
    pub recursive_controls: Vec<Options>,
    controls: IdentitySet<Options>,
}
impl ControlHookLocals {
    pub fn control_order(&self) -> impl Iterator<Item = &Options> {
        self.controls.iter()
    }
    pub fn set_capacity(&self) -> usize {
        self.controls.capacity()
    }
}
fn group(value: &Options) -> Result<Group, Error> {
    if let Options::Object(v) = value
        && let Ok(group) = v.clone().downcast::<RefCell<HookGroup>>()
    {
        return Ok(group);
    }
    Err(exception(
        "AttributeError",
        format!("'{}' object has no attribute 'hooks'", value.typename()),
    ))
}
fn add_group(full: &Group, source: &Group, locals: &mut ControlHookLocals) {
    let mut i = 0;
    // The source list is live, including when full and source are the same group.
    loop {
        let Some(hook) = source.borrow().hooks.get(i).cloned() else {
            break;
        };
        i += 1;
        locals.hook = Some(hook.clone());
        full.borrow_mut().add(hook);
    }
}
fn collect_chain(
    control: Options,
    bindings: &dyn ControlHookBindings,
    locals: &mut ControlHookLocals,
) -> Result<(), Error> {
    bindings.enter_control_recursion()?;
    locals.recursive_controls.push(control.clone());
    let outcome = (|| {
        bindings.source_locals(locals, "control")?;
        // Preserve the source's two reads of each non-None attribute. Custom
        // native control descriptors may change or fail between the reads.
        if bindings.extra_hooks(&control)?.is_some() {
            locals.hook_groups.push(bindings.extra_hooks(&control)?);
        }
        if bindings.previous_controlnet(&control)?.is_some() {
            let previous = bindings
                .previous_controlnet(&control)?
                .unwrap_or(Options::Atom(V::None));
            collect_chain(previous, bindings, locals)?;
        }
        Ok(())
    })();
    locals.recursive_controls.pop();
    bindings.leave_control_recursion();
    outcome
}
pub fn get_hooks_from_cond(
    conditions: &Options,
    full_hooks: Group,
    bindings: &dyn ControlHookBindings,
) -> Result<Group, Error> {
    let mut locals = ControlHookLocals::default();
    let outcome = body(conditions, &full_hooks, bindings, &mut locals);
    bindings.source_locals(&locals, if outcome.is_ok() { "return" } else { "error" })?;
    outcome.map(|()| full_hooks)
}
fn body(
    conditions: &Options,
    full: &Group,
    bindings: &dyn ControlHookBindings,
    locals: &mut ControlHookLocals,
) -> Result<(), Error> {
    let mut controls = Vec::new();
    let mut i = 0;
    loop {
        let record = match conditions {
            Options::List(v) => v.borrow().get(i).cloned(),
            Options::Tuple(v) => v.get(i).cloned(),
            _ => {
                return Err(Error::BindingGap {
                    backend: "get_hooks_from_cond".into(),
                    operation: "conditioning iterator requires its original native binding".into(),
                });
            }
        };
        let Some(record) = record else { break };
        i += 1;
        locals.record = Some(record.clone());
        if let Some(hooks) = record.get("hooks")? {
            add_group(full, &group(&hooks)?, locals);
        }
        if let Some(control) = record.get("control")? {
            controls.push(control);
        }
    }
    // set(list), not set.update(set) or a pre-sized Rust collection. Source
    // hashes every occurrence, including duplicates, after all direct hooks.
    for control in &controls {
        let hash = bindings.control_hash(control)?;
        locals
            .controls
            .add(hash, control.clone(), |a, b| bindings.same_control(a, b));
    }
    drop(controls);
    // Retain the set, and visit its original ascending slot order.
    // A snapshot of borrowed handles is not used to choose a traversal policy.
    let mut slot = 0;
    loop {
        let Some(control) = locals.controls.next(&mut slot).cloned() else {
            break;
        };
        locals.base_control = Some(control.clone());
        collect_chain(control, bindings, locals)?;
    }
    locals.extra_hooks = HookGroup::combine_all(&locals.hook_groups, 0)?;
    if let Some(extra) = locals.extra_hooks.clone() {
        add_group(full, &extra, locals);
    }
    Ok(())
}
