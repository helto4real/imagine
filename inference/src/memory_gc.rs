//! The original cleanup_models_gc body over a live registration list.
//! Full collection is a required native binding; it is never a no-op fallback.
use crate::{native::Error, source_exception::Frame};
use std::rc::Rc;

pub trait Bindings {
    type Entry: Clone + 'static;
    type Model;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn entry(&self, index: usize) -> Result<Self::Entry, Error>;
    fn is_dead(&self, entry: &Self::Entry) -> Result<bool, Error>;
    fn real_model(&self, entry: &Self::Entry) -> Result<Self::Model, Error>;
    fn class_name(&self, model: &Self::Model) -> Result<String, Error>;
    fn info(&self, message: &str) -> Result<(), Error>;
    fn warning(&self, message: &str) -> Result<(), Error>;
    fn collect(&self) -> Result<(), Error>;
    fn empty_cache(&self) -> Result<(), Error>;
}

pub fn cleanup<B: Bindings>(bindings: &B) -> Result<(), Error> {
    let mut do_gc = false;
    let mut current = None;
    let mut index = None;
    let result = (|| {
        for i in 0..bindings.len() {
            index = Some(i);
            drop(current.replace(bindings.entry(i)?));
            let cur = current.as_ref().unwrap();
            if bindings.is_dead(cur)? {
                let model = bindings.real_model(cur)?;
                let name = bindings.class_name(&model);
                drop(model);
                bindings.info(&format!("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.",name?))?;
                do_gc = true;
                break;
            }
        }
        if do_gc {
            bindings.collect()?;
            bindings.empty_cache()?;
            // Source takes a fresh len after both calls. Reentrant callbacks
            // may replace/remove/append entries, but each range is fixed.
            for i in 0..bindings.len() {
                index = Some(i);
                drop(current.replace(bindings.entry(i)?));
                let cur = current.as_ref().unwrap();
                if bindings.is_dead(cur)? {
                    let model = bindings.real_model(cur)?;
                    let name = bindings.class_name(&model);
                    drop(model);
                    bindings.warning(&format!("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.",name?))?;
                }
            }
        }
        Ok(())
    })();
    result.map_err(|error: Error| {
        if matches!(error.root(), Error::BindingGap { .. }) {
            return error;
        }
        let frame = Frame::new("cleanup_models_gc");
        frame.set("do_gc", Rc::new(do_gc));
        if let Some(i) = index {
            frame.set("i", Rc::new(i));
        }
        if let Some(cur) = current {
            frame.set("cur", Rc::new(cur));
        }
        frame.finish();
        error.retain_frame(frame)
    })
}

/// Native coordinator services. Class metadata belongs to the actual model;
/// collection and diagnostics keep the enclosing runtime's policies.
pub trait Services {
    fn class_name(
        &self,
        model: &crate::memory_finalizers::Handle<'static>,
    ) -> Result<String, Error>;
    fn info(&self, message: &str) -> Result<(), Error>;
    fn warning(&self, message: &str) -> Result<(), Error>;
    fn collect(&self) -> Result<(), Error>;
    fn empty_cache(&self) -> Result<(), Error>;
}
pub fn cleanup_native(
    loaded: &super::RegistrationView<'static>,
    services: &dyn Services,
) -> Result<(), Error> {
    struct Native<'a> {
        loaded: &'a super::RegistrationView<'static>,
        services: &'a dyn Services,
    }
    impl Bindings for Native<'_> {
        type Entry = Rc<super::LoadedModel<'static>>;
        type Model = Option<crate::memory_finalizers::Handle<'static>>;
        fn len(&self) -> usize {
            self.loaded.len()
        }
        fn entry(&self, index: usize) -> Result<Self::Entry, Error> {
            self.loaded
                .0
                .borrow()
                .get(index)
                .cloned()
                .ok_or_else(|| Error::ReferenceException {
                    class: "IndexError",
                    message: "list index out of range".into(),
                })
        }
        fn is_dead(&self, entry: &Self::Entry) -> Result<bool, Error> {
            entry.is_dead()
        }
        fn real_model(&self, entry: &Self::Entry) -> Result<Self::Model, Error> {
            let weak = entry
                .real_model()
                .ok_or_else(|| Error::ReferenceException {
                    class: "TypeError",
                    message: "'NoneType' object is not callable".into(),
                })?;
            Ok(weak.upgrade())
        }
        fn class_name(&self, model: &Self::Model) -> Result<String, Error> {
            match model {
                Some(model) => self.services.class_name(model),
                None => Ok("NoneType".into()),
            }
        }
        fn info(&self, message: &str) -> Result<(), Error> {
            self.services.info(message)
        }
        fn warning(&self, message: &str) -> Result<(), Error> {
            self.services.warning(message)
        }
        fn collect(&self) -> Result<(), Error> {
            self.services.collect()
        }
        fn empty_cache(&self) -> Result<(), Error> {
            self.services.empty_cache()
        }
    }
    cleanup(&Native { loaded, services })
}
