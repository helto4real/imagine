//! Owned source exceptions and traceback frame roots. No Python runtime.
//!
//! A traceback keeps completed frames alive. `frame.clear()` releases fast
//! locals, while a previously exposed f_locals dictionary keeps its entries
//! until the next f_locals synchronization (or an explicit dictionary edit).
use crate::native::Error;
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::Rc,
};

pub type Root = Rc<dyn Any>;

#[derive(Clone, Default)]
pub struct Locals(Rc<RefCell<Vec<(String, Root)>>>);
impl Locals {
    pub fn names(&self) -> Vec<String> {
        self.0
            .borrow()
            .iter()
            .map(|(name, _)| name.clone())
            .collect()
    }
    pub fn get(&self, name: &str) -> Option<Root> {
        self.0
            .borrow()
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.clone())
    }
    pub fn set(&self, name: impl Into<String>, value: Root) {
        let name = name.into();
        let old = {
            let mut values = self.0.borrow_mut();
            if let Some((_, slot)) = values.iter_mut().find(|(key, _)| key == &name) {
                Some(std::mem::replace(slot, value))
            } else {
                values.push((name, value));
                None
            }
        };
        drop(old);
    }
    pub fn remove(&self, name: &str) {
        let old = {
            let mut values = self.0.borrow_mut();
            values
                .iter()
                .position(|(key, _)| key == name)
                .map(|index| values.remove(index))
        };
        drop(old);
    }
    pub fn clear(&self) {
        let old = std::mem::take(&mut *self.0.borrow_mut());
        drop(old);
    }
}

pub struct Frame {
    pub function: String,
    active: Cell<bool>,
    fast: Locals,
    names: RefCell<Vec<String>>,
    exposed: Locals,
}
impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("function", &self.function)
            .field("active", &self.active.get())
            .finish_non_exhaustive()
    }
}
impl Frame {
    pub fn new(function: impl Into<String>) -> Rc<Self> {
        Rc::new(Self {
            function: function.into(),
            active: Cell::new(true),
            fast: Locals::default(),
            names: RefCell::new(Vec::new()),
            exposed: Locals::default(),
        })
    }
    pub fn set(&self, name: impl Into<String>, value: Root) {
        let name = name.into();
        if !self.names.borrow().contains(&name) {
            self.names.borrow_mut().push(name.clone());
        }
        self.fast.set(name, value);
    }
    pub fn delete(&self, name: &str) {
        self.fast.remove(name);
    }
    pub fn finish(&self) {
        self.active.set(false);
    }
    pub fn locals(&self) -> Locals {
        let names = self.names.borrow().clone();
        for name in names {
            if let Some(value) = self.fast.get(&name) {
                self.exposed.set(name, value);
            } else {
                self.exposed.remove(&name);
            }
        }
        self.exposed.clone()
    }
    pub fn clear(&self) -> Result<(), Error> {
        if self.active.get() {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: "cannot clear an executing frame".into(),
            });
        }
        self.fast.clear();
        Ok(())
    }
}

pub struct Exception {
    source: Error,
    frames: RefCell<Vec<Rc<Frame>>>,
    context: RefCell<Option<Rc<Exception>>>,
    cause: RefCell<Option<Rc<Exception>>>,
    suppress_context: Cell<bool>,
}
impl std::fmt::Debug for Exception {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Context and cause may form legal source cycles.
        f.debug_struct("Exception")
            .field("source", &self.source)
            .field("frames", &self.frames)
            .finish_non_exhaustive()
    }
}
pub enum ExceptionLink {
    None,
    Exception(Rc<Exception>),
    Other,
}
impl Exception {
    pub fn source(&self) -> &Error {
        &self.source
    }
    /// Outer-to-inner order, matching traceback.tb_next traversal.
    pub fn frames(&self) -> Vec<Rc<Frame>> {
        self.frames.borrow().clone()
    }
    pub fn clear_traceback(&self) {
        let old = std::mem::take(&mut *self.frames.borrow_mut());
        drop(old);
    }
    pub fn context(&self) -> Option<Rc<Self>> {
        self.context.borrow().clone()
    }
    pub fn cause(&self) -> Option<Rc<Self>> {
        self.cause.borrow().clone()
    }
    pub fn suppress_context(&self) -> bool {
        self.suppress_context.get()
    }
    pub fn set_context(&self, value: ExceptionLink) -> Result<(), Error> {
        let value = match value {
            ExceptionLink::None => None,
            ExceptionLink::Exception(value) => Some(value),
            ExceptionLink::Other => {
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: "exception context must be None or derive from BaseException".into(),
                });
            }
        };
        let old = std::mem::replace(&mut *self.context.borrow_mut(), value);
        drop(old);
        Ok(())
    }
    pub fn set_cause(&self, value: ExceptionLink) -> Result<(), Error> {
        let value = match value {
            ExceptionLink::None => None,
            ExceptionLink::Exception(value) => Some(value),
            ExceptionLink::Other => {
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: "exception cause must be None or derive from BaseException".into(),
                });
            }
        };
        self.suppress_context.set(true);
        let old = std::mem::replace(&mut *self.cause.borrow_mut(), value);
        drop(old);
        Ok(())
    }
    pub fn set_suppress_context(&self, value: Option<bool>) -> Result<(), Error> {
        let Some(value) = value else {
            return Err(Error::ReferenceException {
                class: "TypeError",
                message: "attribute value type must be bool".into(),
            });
        };
        self.suppress_context.set(value);
        Ok(())
    }
}
impl Error {
    pub fn is_reference(&self, class: &str) -> bool {
        match self.root() {
            Self::ReferenceException { class: actual, .. } => *actual == class,
            Self::KeyError(_) => class == "KeyError",
            _ => false,
        }
    }
    pub fn root(&self) -> &Self {
        match self {
            Self::Retained(value) => value.source().root(),
            value => value,
        }
    }
    pub fn retain_frame(self, frame: Rc<Frame>) -> Self {
        let exception = self.into_exception();
        exception.frames.borrow_mut().insert(0, frame);
        Self::Retained(exception)
    }
    pub fn into_exception(self) -> Rc<Exception> {
        match self {
            Self::Retained(value) => value,
            source => Rc::new(Exception {
                source,
                frames: RefCell::new(Vec::new()),
                context: RefCell::new(None),
                cause: RefCell::new(None),
                suppress_context: Cell::new(false),
            }),
        }
    }
    /// CPython's automatic exception context when a new error escapes a
    /// handler/finally. Break a context link back to the raised error first.
    pub fn during_handling(self, current: Error) -> Self {
        let raised = self.into_exception();
        let current = current.into_exception();
        if !Rc::ptr_eq(&raised, &current) {
            let mut cursor = Some(current.clone());
            let mut seen = Vec::new();
            while let Some(value) = cursor {
                if seen.iter().any(|prior| Rc::ptr_eq(prior, &value)) {
                    break;
                }
                seen.push(value.clone());
                let next = value.context();
                if next.as_ref().is_some_and(|next| Rc::ptr_eq(next, &raised)) {
                    let old = value.context.borrow_mut().take();
                    drop(old);
                    break;
                }
                cursor = next;
            }
            let old = raised.context.replace(Some(current));
            drop(old);
        }
        Self::Retained(raised)
    }
    pub fn exception(&self) -> Option<Rc<Exception>> {
        if let Self::Retained(value) = self {
            Some(value.clone())
        } else {
            None
        }
    }
}

/// Execute source finally/with exit while retaining the active body error.
/// An exit error replaces its value and retains it as __context__.
pub fn finally<T>(
    body: Result<T, Error>,
    cleanup: impl FnOnce() -> Result<(), Error>,
) -> Result<T, Error> {
    match (cleanup(), body) {
        (Ok(()), result) => result,
        (Err(error), Err(current)) => Err(error.during_handling(current)),
        (Err(error), Ok(_)) => Err(error),
    }
}
