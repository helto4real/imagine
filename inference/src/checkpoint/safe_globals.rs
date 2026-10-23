//! Concrete exact-set store underlying the original Torch safe-global helpers.
//! Capture the old receiver before the caller constructs the temporary set;
//! reentrant callbacks may replace the global while that construction runs.
use crate::{
    native::Error,
    patching::options::{GeneralSet, MappingKeyBindings, Options},
    source_exception::Frame,
};
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};
#[derive(Clone)]
pub struct SafeGlobals {
    current: Rc<RefCell<GeneralSet>>,
    keys: Rc<dyn MappingKeyBindings>,
}
pub struct WeakSafeGlobals {
    current: Weak<RefCell<GeneralSet>>,
    keys: Weak<dyn MappingKeyBindings>,
}
impl WeakSafeGlobals {
    pub fn upgrade(&self) -> Option<SafeGlobals> {
        Some(SafeGlobals {
            current: self.current.upgrade()?,
            keys: self.keys.upgrade()?,
        })
    }
}
impl SafeGlobals {
    pub fn new(keys: Rc<dyn MappingKeyBindings>) -> Self {
        Self {
            current: Rc::new(RefCell::new(GeneralSet::new(keys.clone()))),
            keys,
        }
    }
    pub fn current(&self) -> GeneralSet {
        self.current.borrow().clone()
    }
    pub fn downgrade(&self) -> WeakSafeGlobals {
        WeakSafeGlobals {
            current: Rc::downgrade(&self.current),
            keys: Rc::downgrade(&self.keys),
        }
    }
    fn publish(&self, value: GeneralSet) {
        let old = self.current.replace(value);
        drop(old);
    }
    pub fn clear(&self) {
        self.publish(GeneralSet::new(self.keys.clone()));
    }
    pub fn get(&self) -> Result<Vec<Options>, Error> {
        let iterator = self.current().iterator();
        let mut values = vec![];
        while let Some(value) = iterator.next()? {
            values.push(value);
        }
        Ok(values)
    }
    fn update(
        &self,
        args: &Options,
        remove: bool,
        construct: impl FnOnce(&Options) -> Result<GeneralSet, Error>,
    ) -> Result<(), Error> {
        let receiver = self.current();
        let result: Result<(), Error> = (|| {
            let temporary = construct(args)?;
            let result = if remove {
                receiver.difference_set(&temporary)
            } else {
                receiver.union_set(&temporary)
            };
            drop(receiver);
            drop(temporary);
            let result = result?;
            self.publish(result);
            Ok(())
        })();
        result.map_err(|error| {
            let frame = Frame::new(if remove {
                "_remove_safe_globals"
            } else {
                "_add_safe_globals"
            });
            frame.set(
                if remove {
                    "globals_to_remove"
                } else {
                    "safe_globals"
                },
                Rc::new(args.clone()),
            );
            frame.finish();
            error.retain_frame(frame)
        })
    }
    pub fn add(
        &self,
        args: &Options,
        construct: impl FnOnce(&Options) -> Result<GeneralSet, Error>,
    ) -> Result<(), Error> {
        self.update(args, false, construct)
    }
    pub fn remove(
        &self,
        args: &Options,
        construct: impl FnOnce(&Options) -> Result<GeneralSet, Error>,
    ) -> Result<(), Error> {
        self.update(args, true, construct)
    }
}
