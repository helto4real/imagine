//! Native zero-argument weak finalizers used by LoadedModel. Active callbacks
//! belong to the inference registry, not to caller-held finalizer handles.
//! The process-exit/GC controller is a separate binding, not a Python runtime.
use crate::{execution_value::Value as V, native::Error, patching::options::Options};
use std::{
    cell::{Cell, RefCell},
    ops::Deref,
    rc::{Rc, Weak},
};

pub type Callback<'a> = Rc<dyn Fn() -> Result<Options, Error> + 'a>;
pub type Reporter<'a> = Rc<dyn Fn(Error) + 'a>;
pub trait Target<'a> {
    fn weak_callbacks(&self) -> &WeakCallbacks<'a>;
}
pub type Handle<'a> = Rc<dyn Target<'a> + 'a>;

/// Keep this field before the data: weak callbacks run while object fields
/// remain alive, after the Rc's weak references have become unupgradeable.
pub struct ModelObject<'a, T> {
    callbacks: WeakCallbacks<'a>,
    pub data: T,
}
impl<'a, T> ModelObject<'a, T> {
    pub fn with_data(data: T) -> Rc<Self> {
        Rc::new(Self {
            callbacks: WeakCallbacks(RefCell::new(vec![])),
            data,
        })
    }
}
impl<T> Deref for ModelObject<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.data
    }
}
impl<'a, T> Target<'a> for ModelObject<'a, T> {
    fn weak_callbacks(&self) -> &WeakCallbacks<'a> {
        &self.callbacks
    }
}
impl Default for WeakCallbacks<'_> {
    fn default() -> Self {
        Self::new()
    }
}
impl WeakCallbacks<'_> {
    pub fn new() -> Self {
        Self(RefCell::new(vec![]))
    }
}
pub struct WeakCallbacks<'a>(RefCell<Vec<Weak<Finalizer<'a>>>>);
pub struct Registry<'a> {
    entries: RefCell<Vec<Rc<Finalizer<'a>>>>,
    shutdown: Cell<bool>,
    reporter: Reporter<'a>,
}
impl<'a> Registry<'a> {
    pub fn new(reporter: Reporter<'a>) -> Rc<Self> {
        Rc::new(Self {
            entries: RefCell::new(vec![]),
            shutdown: Cell::new(false),
            reporter,
        })
    }
    pub fn register(
        self: &Rc<Self>,
        target: &Handle<'a>,
        callback: Callback<'a>,
    ) -> Rc<Finalizer<'a>> {
        let finalizer = Rc::new(Finalizer {
            registry: Rc::downgrade(self),
            target: Rc::downgrade(target),
            callback: RefCell::new(Some(callback)),
            atexit: Cell::new(true),
        });
        self.entries.borrow_mut().push(finalizer.clone());
        target
            .weak_callbacks()
            .0
            .borrow_mut()
            .push(Rc::downgrade(&finalizer));
        finalizer
    }
    /// The enclosing process lifecycle supplies this transition. This is not
    /// weakref.finalize._exitfunc or an implementation of its GC/atexit loop.
    pub fn mark_shutdown(&self) {
        self.shutdown.set(true);
    }
    fn pop(&self, finalizer: &Finalizer<'a>) -> Option<Callback<'a>> {
        let removed = {
            let mut entries = self.entries.borrow_mut();
            let index = entries
                .iter()
                .position(|entry| std::ptr::eq(entry.as_ref(), finalizer))?;
            entries.remove(index)
        };
        let callback = finalizer.callback.borrow_mut().take();
        drop(removed);
        callback
    }
}
pub struct Finalizer<'a> {
    registry: Weak<Registry<'a>>,
    target: Weak<dyn Target<'a> + 'a>,
    callback: RefCell<Option<Callback<'a>>>,
    atexit: Cell<bool>,
}
/// Source detach's callable is released before its retained target. LoadedModel
/// discards the returned tuple; only its zero-argument bound-call form is used.
pub struct Detached<'a> {
    pub callback: Callback<'a>,
    pub target: Handle<'a>,
}
impl<'a> Finalizer<'a> {
    pub fn alive(&self) -> bool {
        self.registry.upgrade().is_some() && self.callback.borrow().is_some()
    }
    pub fn atexit(&self) -> bool {
        self.alive() && self.atexit.get()
    }
    pub fn set_atexit(&self, value: bool) {
        if self.alive() {
            self.atexit.set(value);
        }
    }
    pub fn invoke(&self) -> Result<Options, Error> {
        let Some(registry) = self.registry.upgrade() else {
            return Ok(Options::Atom(V::None));
        };
        let callback = registry.pop(self);
        if !registry.shutdown.get()
            && let Some(callback) = callback
        {
            return callback();
        }
        Ok(Options::Atom(V::None))
    }
    pub fn detach(&self) -> Option<Detached<'a>> {
        let registry = self.registry.upgrade()?;
        let target = self.target.upgrade()?;
        let callback = registry.pop(self)?;
        Some(Detached { callback, target })
    }
}
impl Drop for WeakCallbacks<'_> {
    fn drop(&mut self) {
        // CPython's target weakref list invokes the newest callback first.
        // Release every registry/list borrow before calling foreign code.
        let callbacks = std::mem::take(self.0.get_mut());
        for finalizer in callbacks.into_iter().rev().filter_map(|f| f.upgrade()) {
            if let Some(registry) = finalizer.registry.upgrade()
                && let Some(callback) = registry.pop(&finalizer)
                && !registry.shutdown.get()
            {
                // Retain the source info's callable through unraisable reporting.
                if let Err(error) = callback() {
                    (registry.reporter)(error);
                }
            }
        }
    }
}
