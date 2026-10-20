//! ModelPatcher injection/ejection bodies. A shared component of the model
//! owner, with required foreign injection/callback effects and live iterators.
use super::{
    ejection::EjectionOwner,
    extensions::OrderedRegistry,
    options::{Options, ValuesIterator},
    options_iterator::SequenceIterator,
};
use crate::{execution_value::Value as V, native::Error};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
pub trait InjectionBindings {
    fn inject(&self, injection: &Options, owner: &Options) -> Result<(), Error>;
    fn eject(&self, injection: &Options, owner: &Options) -> Result<(), Error>;
    fn callback(&self, callback: &Options, owner: &Options) -> Result<(), Error>;
    fn source_locals(&self, _: &InjectionLocals, _: &str) -> Result<(), Error> {
        Ok(())
    }
}
pub struct InjectionState {
    pub is_injected: Cell<bool>,
    pub skip_injection: Cell<bool>,
    injections: RefCell<Options>,
    pub callbacks: Rc<RefCell<OrderedRegistry<Options>>>,
}
#[derive(Default)]
pub struct InjectionLocals {
    pub injecting: bool,
    pub group: Option<Options>,
    pub injection: Option<Options>,
    pub callback: Option<Rc<Options>>,
    outer: Option<ValuesIterator>,
    inner: Option<SequenceIterator>,
}
#[derive(Clone, Copy)]
enum Action {
    Inject,
    Eject,
}
/// Native protocols for the canonical patcher's actual fields. Registry
/// collection uses those fields directly, not a separate OrderedRegistry.
pub trait SourceInjectionBindings:
    InjectionBindings + super::model_registry::RegistryBindings
{
    fn truth(&self, value: &Options) -> Result<bool, Error>;
    fn callback_kind(&self, name: &str) -> Result<Options, Error>;
}
pub fn inject_model(
    state: &super::model_patcher::PatcherState,
    owner: &Options,
    bindings: &dyn SourceInjectionBindings,
) -> Result<(), Error> {
    source_run(state, owner, bindings, Action::Inject)
}
pub fn eject_model(
    state: &super::model_patcher::PatcherState,
    owner: &Options,
    bindings: &dyn SourceInjectionBindings,
) -> Result<(), Error> {
    source_run(state, owner, bindings, Action::Eject)
}
fn source_run(
    state: &super::model_patcher::PatcherState,
    owner: &Options,
    bindings: &dyn SourceInjectionBindings,
    action: Action,
) -> Result<(), Error> {
    let injecting = matches!(action, Action::Inject);
    let mut locals = InjectionLocals {
        injecting,
        ..InjectionLocals::default()
    };
    let outcome = (|| {
        let active = bindings.truth(&state.fields.get("is_injected")?)?;
        if if injecting {
            active || bindings.truth(&state.fields.get("skip_injection")?)?
        } else {
            !active
        } {
            return Ok(());
        }
        locals.outer = Some(state.fields.get("injections")?.values_iterator()?);
        while let Some(group) = locals.outer.as_ref().unwrap().next_value()? {
            drop(locals.group.replace(group));
            locals.inner = Some(SequenceIterator::new(locals.group.as_ref().unwrap())?);
            while let Some(injection) = locals.inner.as_mut().unwrap().next()? {
                drop(locals.injection.replace(injection));
                bindings.source_locals(&locals, if injecting { "inject" } else { "eject" })?;
                let injection = locals.injection.as_ref().unwrap();
                if injecting {
                    bindings.inject(injection, owner)?;
                    state
                        .fields
                        .set("is_injected", Options::Atom(V::Bool(true)));
                } else {
                    bindings.eject(injection, owner)?;
                }
            }
            locals.inner = None;
        }
        locals.outer = None;
        if !injecting {
            state
                .fields
                .set("is_injected", Options::Atom(V::Bool(false)));
        }
        if !injecting || bindings.truth(&state.fields.get("is_injected")?)? {
            let kind = bindings.callback_kind(if injecting {
                "on_inject_model"
            } else {
                "on_eject_model"
            })?;
            let callbacks = state.registries.callbacks.all(&kind, bindings);
            drop(kind);
            let callbacks = callbacks?;
            let mut items = SequenceIterator::new(&callbacks)?;
            drop(callbacks);
            while let Some(callback) = items.next()? {
                drop(locals.callback.replace(Rc::new(callback)));
                bindings.source_locals(&locals, "callback")?;
                bindings.callback(locals.callback.as_ref().unwrap(), owner)?;
            }
        }
        Ok(())
    })();
    bindings.source_locals(&locals, if outcome.is_ok() { "return" } else { "error" })?;
    outcome
}
impl InjectionState {
    pub fn new(callbacks: Rc<RefCell<OrderedRegistry<Options>>>) -> Self {
        Self {
            is_injected: Cell::new(false),
            skip_injection: Cell::new(false),
            injections: RefCell::new(Options::dictionary(vec![])),
            callbacks,
        }
    }
    pub fn injections(&self) -> Options {
        self.injections.borrow().clone()
    }
    pub fn replace_injections(&self, value: Options) {
        let old = std::mem::replace(&mut *self.injections.borrow_mut(), value);
        drop(old);
    }
    pub fn set_injections(&self, key: &str, value: Options) -> Result<(), Error> {
        self.injections().set(key, value)
    }
    pub fn remove_injections(&self, key: &str) -> Result<(), Error> {
        drop(self.injections().pop(key)?);
        Ok(())
    }
    pub fn get_injections(&self, key: &str) -> Result<Options, Error> {
        Ok(self
            .injections()
            .get(key)?
            .unwrap_or(Options::Atom(V::None)))
    }
    pub fn inject_model(
        &self,
        owner: &Options,
        bindings: &dyn InjectionBindings,
    ) -> Result<(), Error> {
        self.run(owner, bindings, Action::Inject)
    }
    pub fn eject_model(
        &self,
        owner: &Options,
        bindings: &dyn InjectionBindings,
    ) -> Result<(), Error> {
        self.run(owner, bindings, Action::Eject)
    }
    fn run(
        &self,
        owner: &Options,
        bindings: &dyn InjectionBindings,
        action: Action,
    ) -> Result<(), Error> {
        let mut locals = InjectionLocals {
            injecting: matches!(action, Action::Inject),
            ..InjectionLocals::default()
        };
        let outcome = self.body(owner, bindings, action, &mut locals);
        bindings.source_locals(&locals, if outcome.is_ok() { "return" } else { "error" })?;
        outcome
    }
    fn body(
        &self,
        owner: &Options,
        bindings: &dyn InjectionBindings,
        action: Action,
        locals: &mut InjectionLocals,
    ) -> Result<(), Error> {
        let injecting = matches!(action, Action::Inject);
        if if injecting {
            self.is_injected.get() || self.skip_injection.get()
        } else {
            !self.is_injected.get()
        } {
            return Ok(());
        }
        let mapping = self.injections();
        if matches!(mapping, Options::Object(_)) {
            return Err(super::gap(
                "model injection mapping requires its original values binding",
            ));
        }
        locals.outer = Some(mapping.values_iterator()?);
        drop(mapping);
        while let Some(group) = locals.outer.as_ref().unwrap().next_value()? {
            locals.group = Some(group.clone());
            locals.inner = Some(SequenceIterator::new(&group)?);
            while let Some(injection) = locals.inner.as_mut().unwrap().next()? {
                locals.injection = Some(injection.clone());
                bindings.source_locals(locals, if injecting { "inject" } else { "eject" })?;
                if injecting {
                    bindings.inject(&injection, owner)?;
                    self.is_injected.set(true);
                } else {
                    bindings.eject(&injection, owner)?;
                }
            }
            locals.inner = None;
        }
        locals.outer = None;
        if !injecting {
            self.is_injected.set(false);
        }
        if !injecting || self.is_injected.get() {
            let callbacks = self.callbacks.borrow().all(if injecting {
                "on_inject_model"
            } else {
                "on_eject_model"
            });
            for callback in &callbacks {
                let old = locals.callback.replace(callback.clone());
                drop(old);
                bindings.source_locals(locals, "callback")?;
                bindings.callback(callback, owner)?;
            }
            drop(callbacks);
        }
        Ok(())
    }
}
pub struct BoundInjections {
    pub state: Rc<InjectionState>,
    pub owner: Options,
    pub bindings: Rc<dyn InjectionBindings>,
}
impl EjectionOwner for BoundInjections {
    fn is_injected(&self) -> bool {
        self.state.is_injected.get()
    }
    fn skip_injection(&self) -> bool {
        self.state.skip_injection.get()
    }
    fn set_skip_injection(&self, value: bool) {
        self.state.skip_injection.set(value);
    }
    fn eject_model(&self) -> Result<(), Error> {
        self.state.eject_model(&self.owner, self.bindings.as_ref())
    }
    fn inject_model(&self) -> Result<(), Error> {
        self.state.inject_model(&self.owner, self.bindings.as_ref())
    }
}
