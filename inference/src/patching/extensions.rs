//! Ordered callback/wrapper contracts from comfy.patcher_extension.
use crate::{execution_value::Value, native::Error};
use std::{any::Any, cell::RefCell, rc::Rc};

pub const ON_DETACH: &str = "on_detach_after";
pub const CALLBACKS: [&str; 12] = [
    "on_clone",
    "on_deepclone_multigpu",
    "on_match_multigpu_clones",
    "on_load_after",
    ON_DETACH,
    "on_cleanup",
    "on_pre_run",
    "on_prepare_state",
    "on_apply_hooks",
    "on_register_all_hook_patches",
    "on_inject_model",
    "on_eject_model",
];
pub const WRAPPERS: [&str; 7] = [
    "outer_sample",
    "prepare_sampling",
    "sampler_sample",
    "predict_noise",
    "calc_cond_batch",
    "apply_model",
    "diffusion_model",
];

/// Vecs preserve Python dict insertion order. Copying a registry copies its
/// containers and retains the same callback objects and per-key ordering.
pub type CallbackList<C> = Rc<RefCell<Vec<Rc<C>>>>;
type KeyedCallbacks<C> = Vec<(Option<String>, CallbackList<C>)>;
pub struct OrderedRegistry<C> {
    entries: Vec<(String, KeyedCallbacks<C>)>,
}
impl<C> Default for OrderedRegistry<C> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}
impl<C> Clone for OrderedRegistry<C> {
    fn clone(&self) -> Self {
        Self {
            entries: self
                .entries
                .iter()
                .map(|(kind, groups)| {
                    (
                        kind.clone(),
                        groups
                            .iter()
                            .map(|(key, list)| {
                                (key.clone(), Rc::new(RefCell::new(list.borrow().clone())))
                            })
                            .collect(),
                    )
                })
                .collect(),
        }
    }
}
impl<C> OrderedRegistry<C> {
    pub fn add(&mut self, call_type: &str, key: Option<&str>, callback: Rc<C>) {
        let index = if let Some(i) = self.entries.iter().position(|(k, _)| k == call_type) {
            i
        } else {
            self.entries.push((call_type.into(), Vec::new()));
            self.entries.len() - 1
        };
        let groups = &mut self.entries[index].1;
        let index = if let Some(i) = groups.iter().position(|(k, _)| k.as_deref() == key) {
            i
        } else {
            groups.push((key.map(str::to_owned), Rc::new(RefCell::new(Vec::new()))));
            groups.len() - 1
        };
        groups[index].1.borrow_mut().push(callback);
    }
    /// ModelPatcher.get_callbacks/get_wrappers return the actual existing list.
    /// Each absent lookup returns a fresh empty list without inserting a key.
    pub fn list(&self, call_type: &str, key: Option<&str>) -> CallbackList<C> {
        self.entries
            .iter()
            .find(|(kind, _)| kind == call_type)
            .and_then(|(_, groups)| groups.iter().find(|(name, _)| name.as_deref() == key))
            .map_or_else(|| Rc::new(RefCell::new(vec![])), |(_, list)| list.clone())
    }
    pub fn remove(&mut self, call_type: &str, key: Option<&str>) {
        if let Some((_, groups)) = self.entries.iter_mut().find(|(kind, _)| kind == call_type)
            && let Some(index) = groups.iter().position(|(name, _)| name.as_deref() == key)
        {
            groups.remove(index);
        }
    }
    pub fn with_key(&self, call_type: &str, key: Option<&str>) -> Vec<Rc<C>> {
        self.entries
            .iter()
            .find(|(k, _)| k == call_type)
            .and_then(|(_, groups)| groups.iter().find(|(k, _)| k.as_deref() == key))
            .map_or_else(Vec::new, |(_, list)| list.borrow().clone())
    }
    pub fn all(&self, call_type: &str) -> Vec<Rc<C>> {
        self.entries
            .iter()
            .find(|(k, _)| k == call_type)
            .map_or_else(Vec::new, |(_, groups)| {
                groups
                    .iter()
                    .flat_map(|(_, list)| list.borrow().clone())
                    .collect()
            })
    }
}
pub type Arguments<A = Value> = Vec<(String, A)>;
pub type Original<A = Value> = Rc<dyn Fn(&[A], &Arguments<A>) -> Result<A, Error>>;
pub type Wrapper<A = Value> =
    Rc<dyn Fn(&WrapperExecutor<A>, &[A], &Arguments<A>) -> Result<A, Error>>;
pub struct WrapperExecutor<A: Clone = Value> {
    pub original: Original<A>,
    pub class_obj: Option<Rc<dyn Any>>,
    pub wrappers: RefCell<Vec<Wrapper<A>>>,
    pub idx: usize,
    pub is_last: bool,
}
impl<A: Clone + 'static> WrapperExecutor<A> {
    pub fn new(
        original: Original<A>,
        class_obj: Option<Rc<dyn Any>>,
        wrappers: &[Wrapper<A>],
        idx: usize,
    ) -> Self {
        Self {
            original,
            class_obj,
            wrappers: RefCell::new(wrappers.to_vec()),
            idx,
            is_last: idx == wrappers.len(),
        }
    }
    pub fn execute(&self, args: &[A], kwargs: &Arguments<A>) -> Result<A, Error> {
        let args = args.to_vec();
        let kwargs = kwargs.clone();
        if self.is_last {
            return (self.original)(&args, &kwargs);
        }
        let wrapper = self
            .wrappers
            .borrow()
            .get(self.idx)
            .cloned()
            .ok_or_else(|| Error::ReferenceException {
                class: "IndexError",
                message: "list index out of range".into(),
            })?;
        wrapper(self, &args, &kwargs)
    }
    /// Source __call__ advances to a fresh executor; execute starts at this one.
    pub fn next(&self, args: &[A], kwargs: &Arguments<A>) -> Result<A, Error> {
        let idx = self.idx + 1;
        let wrappers = self.wrappers.borrow();
        if idx > wrappers.len() {
            return Err(Error::ReferenceException {
                class: "Exception",
                message: "Wrapper idx exceeded available wrappers; something went very wrong."
                    .into(),
            });
        }
        let next = Self::new(
            self.original.clone(),
            self.class_obj.clone(),
            &wrappers,
            idx,
        );
        // The copied executor runs after the parent's list access completed.
        // Its wrappers may re-enter and mutate the parent through another alias.
        drop(wrappers);
        next.execute(args, kwargs)
    }
}
