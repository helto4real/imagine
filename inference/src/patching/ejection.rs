//! AutoPatcherEjector entry/exit ordering from the frozen ModelPatcher source.
//! Explicit fallible exit preserves partial state and exception precedence;
//! Drop cannot represent this contract because it cannot return an error.
use crate::native::Error;
use std::{cell::Cell, rc::Rc};
pub trait EjectionOwner {
    fn is_injected(&self) -> bool;
    fn skip_injection(&self) -> bool;
    fn set_skip_injection(&self, value: bool);
    fn eject_model(&self) -> Result<(), Error>;
    fn inject_model(&self) -> Result<(), Error>;
}
pub struct AutoPatcherEjector<M: EjectionOwner + ?Sized> {
    pub model: Rc<M>,
    pub was_injected: Cell<bool>,
    pub prev_skip_injection: Cell<bool>,
    pub skip_and_inject_on_exit_only: bool,
}
impl<M: EjectionOwner + ?Sized> AutoPatcherEjector<M> {
    pub fn new(model: Rc<M>, skip_and_inject_on_exit_only: bool) -> Self {
        Self {
            model,
            was_injected: Cell::new(false),
            prev_skip_injection: Cell::new(false),
            skip_and_inject_on_exit_only,
        }
    }
    pub fn enter(&self) -> Result<(), Error> {
        self.was_injected.set(false);
        self.prev_skip_injection.set(self.model.skip_injection());
        if self.skip_and_inject_on_exit_only {
            self.model.set_skip_injection(true);
        }
        if self.model.is_injected() {
            self.model.eject_model()?;
            self.was_injected.set(true);
        }
        Ok(())
    }
    pub fn exit(&self) -> Result<(), Error> {
        if self.skip_and_inject_on_exit_only {
            self.model
                .set_skip_injection(self.prev_skip_injection.get());
            self.model.inject_model()?;
        }
        if self.was_injected.get() && !self.model.skip_injection() {
            self.model.inject_model()?;
        }
        self.model
            .set_skip_injection(self.prev_skip_injection.get());
        Ok(())
    }
    /// Exactly the fallible `with` protocol: failed entry has no exit; failed
    /// exit replaces the body result; no unconditional flag restoration occurs.
    pub fn run<T>(&self, body: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        self.enter()?;
        let result = body();
        crate::source_exception::finally(result, || self.exit())
    }
}
