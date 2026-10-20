//! Frozen model_management.LoadedModel and the shared VRAM-policy coordinator.
//! Linux/Nvidia NORMAL_VRAM remains the default. Registrations are weak; model storage belongs to patchers and
//! execution users. Device queries and patcher actions stay explicit so bounded
//! original-code oracles can exercise pressure without allocating model weights.
use crate::native::Error;
use std::{
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
use tch::Device;
#[path = "memory_policy.rs"]
pub mod policy;
#[path = "memory_source.rs"]
pub mod source;
use policy::{MemoryPolicy, VramState};
#[path = "memory_dtype.rs"]
pub mod dtype;
#[path = "memory_exceptions.rs"]
pub mod exceptions;
#[path = "memory_gc.rs"]
pub mod gc;
#[path = "memory_pinning.rs"]
pub mod pinning;

#[derive(Clone, Copy, Debug)]
pub enum PatchTarget {
    Device(Device),
    Dtype(Option<tch::Kind>),
}

/// Concrete legacy native owners may bind their source-class destructor here.
/// Canonical source owners use OwnerBindings.finalize instead.
pub trait NativeLifecycle {
    fn finalize(&self) -> Result<(), Error>;
    fn unraisable(&self, error: Error);
}
pub trait Model<'a> {
    fn native_lifecycle(&self) -> Option<&dyn NativeLifecycle> {
        None
    }
    fn model_dtype(&self) -> Result<Option<tch::Kind>, Error>;
    fn model_patches_to(&self, target: PatchTarget) -> Result<(), Error>;
    fn storage(&self) -> crate::memory_finalizers::Handle<'a>;
    fn load_device(&self) -> Device;
    fn offload_device(&self) -> Result<Device, Error>;
    fn current_device(&self) -> Result<Device, Error>;
    fn size(&self) -> Result<f64, Error>;
    fn loaded_size(&self) -> Result<f64, Error>;
    fn loaded_ram_size(&self) -> Result<f64, Error>;
    fn partially_load(&self, device: Device, extra: f64, force_patch: bool) -> Result<f64, Error>;
    fn partially_unload(&self, device: Device, amount: f64) -> Result<f64, Error>;
    fn detach(&self, unpatch: bool) -> Result<(), Error>;
    fn dynamic(&self) -> Result<bool, Error> {
        Ok(false)
    }
    fn patch_selection(&self) -> Option<(u64, crate::patching::PatchMap, String)> {
        None
    }
}

/// A clone patcher has a distinct identity, a strong parent and shared model
/// storage. Cloning this handle itself is ordinary reference retention.
pub struct Patcher<'a> {
    pub model: Box<dyn Model<'a> + 'a>,
    parent: Option<Handle<'a>>,
    additional: Vec<Handle<'a>>,
    parent_finalizers: RefCell<Vec<Rc<LoadedModel<'a>>>>,
    source: Option<source::SourceOwner<'a>>,
}
pub type Handle<'a> = Rc<Patcher<'a>>;
impl<'a> Patcher<'a> {
    pub fn new(model: impl Model<'a> + 'a) -> Handle<'a> {
        Rc::new(Self {
            model: Box::new(model),
            parent: None,
            additional: vec![],
            parent_finalizers: RefCell::new(vec![]),
            source: None,
        })
    }
    pub fn with_parent(model: impl Model<'a> + 'a, parent: &Handle<'a>) -> Handle<'a> {
        Rc::new(Self {
            model: Box::new(model),
            parent: Some(parent.clone()),
            additional: vec![],
            parent_finalizers: RefCell::new(vec![]),
            source: None,
        })
    }
    pub fn with_additional(model: impl Model<'a> + 'a, additional: Vec<Handle<'a>>) -> Handle<'a> {
        Rc::new(Self {
            model: Box::new(model),
            parent: None,
            additional,
            parent_finalizers: RefCell::new(vec![]),
            source: None,
        })
    }
    pub fn source_state(&self) -> Option<&crate::patching::model_patcher::PatcherState> {
        self.source.as_ref().map(|source| &source.state)
    }
    pub fn source_bindings(&self) -> Option<&(dyn source::OwnerBindings<'a> + 'a)> {
        self.source.as_ref().map(|source| source.bindings.as_ref())
    }
    fn parent_handle(&self) -> Result<Option<Handle<'a>>, Error> {
        if let Some(source) = &self.source {
            source.bindings.parent(&source.state)
        } else {
            Ok(self.parent.clone())
        }
    }
    fn load_device(&self) -> Result<Device, Error> {
        if let Some(source) = &self.source {
            source.bindings.load_device(&source.state)
        } else {
            Ok(self.model.load_device())
        }
    }
    fn patch_models(&self) -> Result<Vec<Handle<'a>>, Error> {
        if let Some(source) = &self.source {
            source.bindings.model_patches_models(&source.state)
        } else {
            Ok(self.additional.clone())
        }
    }
}
impl Patcher<'static> {
    /// One canonical owned model identity with its complete source state. The
    /// supplied backend must read a clone of state.fields when it needs source
    /// attributes. No second parent/dependency copy is installed here.
    pub fn from_source(
        model: impl Model<'static> + 'static,
        state: crate::patching::model_patcher::PatcherState,
        bindings: Rc<dyn source::OwnerBindings<'static>>,
    ) -> Handle<'static> {
        Rc::new(Self {
            model: Box::new(model),
            parent: None,
            additional: vec![],
            parent_finalizers: RefCell::new(vec![]),
            source: Some(source::SourceOwner { state, bindings }),
        })
    }
}
impl Drop for Patcher<'_> {
    fn drop(&mut self) {
        if let Some(source) = &self.source
            && let Err(error) = source.bindings.finalize(self)
        {
            source.bindings.unraisable(error);
        } else if self.source.is_none()
            && let Some(lifecycle) = self.model.native_lifecycle()
            && let Err(error) = lifecycle.finalize()
        {
            lifecycle.unraisable(error);
        }
        // weakref.finalize runs while the dying patcher's strong parent field
        // still exists. Transfer registrations now, even if no manager will
        // subsequently read them. No borrow is held over a model getter.
        let finalizers = std::mem::take(self.parent_finalizers.get_mut());
        // Source weakref callbacks run newest first on the dying object.
        for registration in finalizers.into_iter().rev() {
            registration.switch_parent();
        }
    }
}
/// Native loaded-model registration. Model/parent/storage references are weak.
/// Active parent finalizers retain this registration, as weakref.finalize's
/// original registry retains its bound callback, without retaining the patcher.
pub struct LoadedModel<'a> {
    patcher: RefCell<Weak<Patcher<'a>>>,
    parent: RefCell<Option<Weak<Patcher<'a>>>>,
    real: RefCell<Option<Weak<dyn crate::memory_finalizers::Target<'a> + 'a>>>,
    model_finalizer: RefCell<Option<Rc<crate::memory_finalizers::Finalizer<'a>>>>,
    used: Cell<bool>,
    device: Cell<Option<Device>>,
    patcher_finalizer_assigned: Cell<bool>,
}
impl<'a> LoadedModel<'a> {
    pub fn real_model(&self) -> Option<Weak<dyn crate::memory_finalizers::Target<'a> + 'a>> {
        self.real.borrow().clone()
    }
    pub fn model_finalizer(&self) -> Option<Rc<crate::memory_finalizers::Finalizer<'a>>> {
        self.model_finalizer.borrow().clone()
    }
    pub fn model_load(
        &self,
        lowvram_model_memory: f64,
        force_patch_weights: bool,
        finalizers: &Rc<crate::memory_finalizers::Registry<'a>>,
        registrations: &RegistrationView<'a>,
    ) -> Result<crate::memory_finalizers::Handle<'a>, Error> {
        {
            let model = self.query_owner("model_patches_to")?;
            model
                .model
                .model_patches_to(PatchTarget::Device(self.device()?))?;
        }
        {
            let model = self.query_owner("model_patches_to")?;
            let dtype = self.query_owner("model_dtype")?.model.model_dtype()?;
            model.model.model_patches_to(PatchTarget::Dtype(dtype))?;
        }
        let use_more = if lowvram_model_memory == 0. {
            1e32
        } else {
            lowvram_model_memory
        };
        self.model_use_more_vram(use_more, force_patch_weights)?;
        let real = self.query_owner("model")?.model.storage();
        *self.real.borrow_mut() = Some(Rc::downgrade(&real));
        let registrations = Rc::downgrade(&registrations.0);
        let callback: crate::memory_finalizers::Callback<'a> = Rc::new(move || {
            if let Some(registrations) = registrations.upgrade() {
                cleanup_registrations(&registrations)?;
            }
            Ok(crate::patching::options::Options::Atom(
                crate::execution_value::Value::None,
            ))
        });
        let finalizer = finalizers.register(&real, callback);
        finalizer.set_atexit(false);
        drop(self.model_finalizer.replace(Some(finalizer)));
        Ok(real)
    }
    pub fn same_model(&self, other: &Self) -> bool {
        match (self.patcher(), other.patcher()) {
            (Some(left), Some(right)) => Rc::ptr_eq(&left, &right),
            (None, None) => true,
            _ => false,
        }
    }
    fn detach_real_finalizer(&self) -> Result<(), Error> {
        let finalizer =
            self.model_finalizer
                .borrow()
                .clone()
                .ok_or_else(|| Error::ReferenceException {
                    class: "AttributeError",
                    message: "'NoneType' object has no attribute 'detach'".into(),
                })?;
        drop(finalizer.detach());
        Ok(())
    }
    fn query_owner(&self, method: &str) -> Result<Handle<'a>, Error> {
        self.patcher().ok_or_else(|| Error::ReferenceException {
            class: "AttributeError",
            message: format!("'NoneType' object has no attribute '{method}'"),
        })
    }
    pub fn model_memory(&self) -> Result<f64, Error> {
        self.query_owner("model_size")?.model.size()
    }
    pub fn model_loaded_memory(&self) -> Result<f64, Error> {
        self.query_owner("loaded_size")?.model.loaded_size()
    }
    pub fn model_offloaded_memory(&self) -> Result<f64, Error> {
        Ok(self.model_memory()? - self.model_loaded_memory()?)
    }
    pub fn model_memory_required(&self, device: Device) -> Result<f64, Error> {
        // Original tests current_loaded_device before querying model_size.
        // Model getters may mutate the size/loaded-size used by this budget.
        let current = self
            .query_owner("current_loaded_device")?
            .model
            .current_device()?;
        if device == current {
            self.model_offloaded_memory()
        } else {
            self.model_memory()
        }
    }
    pub fn model_use_more_vram(
        &self,
        extra_memory: f64,
        force_patch_weights: bool,
    ) -> Result<f64, Error> {
        let model = self.query_owner("partially_load")?;
        model
            .model
            .partially_load(self.device()?, extra_memory, force_patch_weights)
    }
    pub fn model_unload(
        &self,
        memory_to_free: Option<f64>,
        unpatch_weights: bool,
    ) -> Result<bool, Error> {
        if let Some(memory) = memory_to_free
            && memory < self.model_loaded_memory()?
        {
            let freed = {
                let model = self.query_owner("partially_unload")?;
                let device = self.query_owner("offload_device")?.model.offload_device()?;
                model.model.partially_unload(device, memory)?
            };
            if freed >= memory {
                return Ok(false);
            }
        }
        self.query_owner("detach")?.model.detach(unpatch_weights)?;
        self.detach_real_finalizer()?;
        *self.model_finalizer.borrow_mut() = None;
        *self.real.borrow_mut() = None;
        Ok(true)
    }
    pub fn is_dead(&self) -> Result<bool, Error> {
        let real = self
            .real
            .borrow()
            .clone()
            .ok_or_else(|| Error::ReferenceException {
                class: "TypeError",
                message: "'NoneType' object is not callable".into(),
            })?;
        let real_alive = real.upgrade().is_some();
        Ok(real_alive && self.patcher().is_none())
    }
    pub fn new(p: &Handle<'a>) -> Result<Rc<Self>, Error> {
        let out = Rc::new(Self {
            patcher: RefCell::new(Rc::downgrade(p)),
            parent: RefCell::new(None),
            real: RefCell::new(None),
            model_finalizer: RefCell::new(None),
            used: Cell::new(true),
            device: Cell::new(None),
            patcher_finalizer_assigned: Cell::new(false),
        });
        out.set_model(p)?;
        out.device.set(Some(p.load_device()?));
        out.patcher_finalizer_assigned.set(true);
        Ok(out)
    }
    fn set_model(self: &Rc<Self>, model: &Handle<'a>) -> Result<(), Error> {
        *self.patcher.borrow_mut() = Rc::downgrade(model);
        // _set_model reads parent separately for the condition and weakref.
        // With no parent it leaves any previous _parent_model untouched.
        if model.parent_handle()?.is_some() {
            let parent = model
                .parent_handle()?
                .ok_or_else(|| Error::ReferenceException {
                    class: "TypeError",
                    message: "cannot create weak reference to 'NoneType' object".into(),
                })?;
            *self.parent.borrow_mut() = Some(Rc::downgrade(&parent));
            drop(parent);
            model.parent_finalizers.borrow_mut().push(self.clone());
            self.patcher_finalizer_assigned.set(true);
        }
        Ok(())
    }
    fn switch_parent(self: &Rc<Self>) {
        let parent = self.parent.borrow().as_ref().and_then(Weak::upgrade);
        if let Some(parent) = parent {
            let result = self.set_model(&parent).and_then(|()| {
                self.device.set(Some(parent.load_device()?));
                Ok(())
            });
            if let Err(error) = result {
                // Only a source binding can fail here; legacy parents use an
                // ordinary infallible strong field. Preserve the already-set
                // weak model and every field updated before the failing query.
                parent
                    .source
                    .as_ref()
                    .expect("fallible source owner query")
                    .bindings
                    .unraisable(error);
            }
        }
    }
    pub fn patcher(&self) -> Option<Handle<'a>> {
        self.patcher.borrow().upgrade()
    }
    pub fn device(&self) -> Result<Device, Error> {
        self.device.get().ok_or_else(|| Error::ReferenceException {
            class: "AttributeError",
            message: "'LoadedModel' object has no attribute 'device'".into(),
        })
    }
}

impl Drop for LoadedModel<'_> {
    fn drop(&mut self) {
        // Source __del__ also runs after a failed __init__. Before the first
        // finalizer assignment its attribute access itself is unraisable.
        if !self.patcher_finalizer_assigned.get() {
            let reporter = self
                .patcher
                .get_mut()
                .upgrade()
                .and_then(|owner| owner.source.as_ref().map(|source| source.bindings.clone()));
            if let Some(reporter) = reporter {
                reporter.unraisable(Error::ReferenceException {
                    class: "AttributeError",
                    message: "'LoadedModel' object has no attribute '_patcher_finalizer'".into(),
                });
            }
        }
    }
}

pub trait MemoryServices {
    /// Comfy get_free_memory's total-free and unused Torch-reserved bytes.
    fn free(&mut self, device: Device) -> Result<(f64, f64), Error>;
    fn empty_cache(&mut self) -> Result<(), Error>;
    fn ensure_pins(&mut self, bytes: f64) -> Result<(), Error>;
}

type Registrations<'a> = Rc<RefCell<Vec<Rc<LoadedModel<'a>>>>>;
/// A live view of the original registration list, without borrowing Manager
/// across model destruction or manufacturing another model/manager identity.
#[derive(Clone)]
pub struct RegistrationView<'a>(Registrations<'a>);
impl<'a> RegistrationView<'a> {
    /// Bind an existing source list identity; this does not copy its entries.
    pub fn from_shared(loaded: Rc<RefCell<Vec<Rc<LoadedModel<'a>>>>>) -> Self {
        Self(loaded)
    }
    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
    pub fn entries(&self) -> Vec<Rc<LoadedModel<'a>>> {
        self.0.borrow().clone()
    }
}
/// Original helpers over a live native loaded-model list. Read the next index
/// only after the model call: callbacks can mutate this same list. Other source
/// iterable classes require their own iterator binding, not a copied Vec.
pub fn use_more_memory<'a>(
    mut extra_memory: f64,
    loaded_models: &RegistrationView<'a>,
    device: Device,
) -> Result<(), Error> {
    let mut index = 0;
    let mut current = None;
    loop {
        let next = loaded_models.0.borrow().get(index).cloned();
        let Some(next) = next else { break };
        index += 1;
        drop(current.replace(next));
        let model = current.as_ref().unwrap();
        if model.device()? == device {
            extra_memory -= model.model_use_more_vram(extra_memory, false)?;
            if extra_memory <= 0. {
                break;
            }
        }
    }
    Ok(())
}
pub fn offloaded_memory<'a>(
    loaded_models: &RegistrationView<'a>,
    device: Device,
) -> Result<f64, Error> {
    let mut offloaded_mem = 0.;
    let mut index = 0;
    let mut current = None;
    loop {
        let next = loaded_models.0.borrow().get(index).cloned();
        let Some(next) = next else { break };
        index += 1;
        drop(current.replace(next));
        let model = current.as_ref().unwrap();
        if model.device()? == device {
            offloaded_mem += model.model_offloaded_memory()?;
        }
    }
    Ok(offloaded_mem)
}
pub struct Manager<'a> {
    loaded: Registrations<'a>,
    finalizers: Rc<crate::memory_finalizers::Registry<'a>>,
    policy: MemoryPolicy,
}
fn cleanup_registrations(loaded: &Registrations<'_>) -> Result<(), Error> {
    let count = loaded.borrow().len();
    let mut removed = vec![];
    for index in 0..count {
        let entry =
            loaded
                .borrow()
                .get(index)
                .cloned()
                .ok_or_else(|| Error::ReferenceException {
                    class: "IndexError",
                    message: "list index out of range".into(),
                })?;
        let real = entry
            .real
            .borrow()
            .clone()
            .ok_or_else(|| Error::ReferenceException {
                class: "TypeError",
                message: "'NoneType' object is not callable".into(),
            })?;
        if real.upgrade().is_none() {
            removed.push(index);
        }
    }
    for index in removed.into_iter().rev() {
        let removed = {
            let mut entries = loaded.borrow_mut();
            if index >= entries.len() {
                return Err(Error::ReferenceException {
                    class: "IndexError",
                    message: "pop index out of range".into(),
                });
            }
            entries.remove(index)
        };
        drop(removed);
    }
    Ok(())
}
impl Default for Manager<'_> {
    fn default() -> Self {
        Self::with_finalizers(
            MemoryPolicy::default(),
            crate::memory_finalizers::Registry::new(Rc::new(|error| eprintln!("{error}"))),
        )
    }
}
impl<'a> Manager<'a> {
    pub fn registration_view(&self) -> RegistrationView<'a> {
        RegistrationView(self.loaded.clone())
    }
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_policy(policy: MemoryPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }
    /// Native global weak-finalizer/error context for an inference environment.
    pub fn with_finalizers(
        policy: MemoryPolicy,
        finalizers: Rc<crate::memory_finalizers::Registry<'a>>,
    ) -> Self {
        Self {
            loaded: Rc::new(RefCell::new(vec![])),
            finalizers,
            policy,
        }
    }
    fn entry(&self, index: usize) -> Result<Rc<LoadedModel<'a>>, Error> {
        self.loaded
            .borrow()
            .get(index)
            .cloned()
            .ok_or_else(|| Error::ReferenceException {
                class: "IndexError",
                message: "list index out of range".into(),
            })
    }
    fn pop(&self, index: usize) -> Result<Rc<LoadedModel<'a>>, Error> {
        let mut entries = self.loaded.borrow_mut();
        if index >= entries.len() {
            return Err(Error::ReferenceException {
                class: "IndexError",
                message: "pop index out of range".into(),
            });
        }
        Ok(entries.remove(index))
    }
    pub fn cleanup(&mut self) -> Result<(), Error> {
        cleanup_registrations(&self.loaded)
    }
    pub fn loaded(&mut self, only_used: bool) -> Vec<Option<Handle<'a>>> {
        let mut output = vec![];
        let mut index = 0;
        let mut last = None;
        loop {
            let entry = self.loaded.borrow().get(index).cloned();
            let Some(entry) = entry else { break };
            index += 1;
            drop(last.replace(entry));
            let entry = last.as_ref().unwrap();
            if !only_used || entry.used.get() {
                output.push(entry.patcher());
            }
        }
        output
    }
    pub fn registrations(&self) -> usize {
        self.loaded.borrow().len()
    }
    pub fn states(&mut self) -> Vec<(Option<Handle<'a>>, bool, Device)> {
        self.loaded
            .borrow()
            .iter()
            .map(|e| {
                (
                    e.patcher(),
                    e.used.get(),
                    e.device()
                        .expect("registered model has an initialized device"),
                )
            })
            .collect()
    }
    /// Original free_memory ordering: most offloaded, least retained, smallest,
    /// then original list index. Partial success keeps its registration.
    pub fn free_memory(
        &mut self,
        required: f64,
        device: Option<Device>,
        keep: &[Rc<LoadedModel<'a>>],
        dynamic: bool,
        pins: f64,
        services: &mut dyn MemoryServices,
    ) -> Result<Vec<Rc<LoadedModel<'a>>>, Error> {
        self.cleanup()?;
        let mut candidates = vec![];
        let count = self.loaded.borrow().len();
        for i in (0..count).rev() {
            let e = self.entry(i)?;
            if let Some(d) = device
                && d != e.device()?
            {
                continue;
            }
            if let Some(p) = e.patcher() {
                if keep.iter().any(|k| Rc::ptr_eq(k, &e) || k.same_model(&e)) {
                    continue;
                }
                candidates.push((
                    -(p.model.size()? - p.model.loaded_size()?),
                    Rc::strong_count(&p),
                    p.model.size()?,
                    i,
                ));
                e.used.set(false);
            }
        }
        candidates.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.cmp(&b.1))
                .then(a.2.total_cmp(&b.2))
                .then(a.3.cmp(&b.3))
        });
        let mut required = required;
        let mut removed = vec![];
        for (_, _, _, i) in candidates {
            let entry = self.entry(i)?;
            let Some(p) = entry.patcher() else {
                continue;
            };
            // device=None under smart memory intentionally marks inactive but
            // requests zero bytes; unload_all_models passes a concrete device.
            let mut amount = 1e32;
            if !self.policy.disable_smart_memory || device.is_none() {
                amount = if let Some(d) = device {
                    required - services.free(d)?.0
                } else {
                    0.
                };
                if p.model.dynamic()? && dynamic {
                    required -= p.model.loaded_size()?;
                    amount = 0.;
                }
            }
            drop(p);
            if amount > 0. && entry.model_unload(Some(amount), true)? {
                removed.push(i);
            }
        }
        removed.sort_by_key(|i| std::cmp::Reverse(*i));
        let mut unloaded = vec![];
        for i in removed {
            unloaded.push(self.pop(i)?);
        }
        if !dynamic && pins > 0. {
            services.ensure_pins(pins)?;
        }
        if !unloaded.is_empty() {
            services.empty_cache()?;
        } else if let Some(d) = device
            && self.policy.vram_state != VramState::HighVram
        {
            let (total, torch) = services.free(d)?;
            if torch > total * 0.25 {
                services.empty_cache()?;
            }
        }
        Ok(unloaded)
    }
    pub fn load(
        &mut self,
        requested: &[Handle<'a>],
        memory: f64,
        minimum: Option<f64>,
        force_patch: bool,
        force_full: bool,
        services: &mut dyn MemoryServices,
    ) -> Result<(), Error> {
        self.cleanup()?;
        let (extra, minimum) = self.policy.requirements(memory, minimum);
        let mut models = vec![];
        for p in requested {
            let dependencies = p.patch_models()?;
            for model in std::iter::once(p).chain(dependencies.iter()) {
                if !models.iter().any(|m| Rc::ptr_eq(m, model)) {
                    models.push(model.clone());
                }
            }
        }
        models.reverse();
        let mut dynamic = true;
        let mut selected = vec![];
        let mut loaded_model = None;
        for p in &models {
            // Source does not short-circuit this loop. Each dynamic query is
            // followed by a fresh LoadedModel, even when an existing registry
            // entry will be reused. Its weak parent finalizer still exists.
            if !p.model.dynamic()? {
                dynamic = false;
            }
            drop(loaded_model.replace(LoadedModel::new(p)?));
            let found = self
                .loaded
                .borrow()
                .iter()
                .find(|e| e.patcher().is_some_and(|x| Rc::ptr_eq(&x, p)))
                .cloned();
            let e = if let Some(found) = found {
                found
            } else {
                loaded_model.as_ref().unwrap().clone()
            };
            e.used.set(true);
            selected.push(e);
        }
        // is_clone includes self. Preserve weights via detach(false), remove
        // registrations before asking the shared manager for space.
        for (p, entry) in models.iter().zip(&selected) {
            drop(loaded_model.replace(entry.clone()));
            let storage = p.model.storage();
            let count = self.loaded.borrow().len();
            for i in (0..count).rev() {
                let candidate = self.entry(i)?.patcher();
                if let Some(c) = candidate
                    && Rc::ptr_eq(&storage, &c.model.storage())
                {
                    let removed = self.pop(i)?;
                    c.model.detach(false)?;
                    removed.detach_real_finalizer()?;
                }
            }
        }
        let mut devices: Vec<(Device, f64, f64)> = vec![];
        for (p, entry) in models.iter().zip(&selected) {
            drop(loaded_model.replace(entry.clone()));
            let d = entry.device()?;
            let required = entry.model_memory_required(d)?;
            if let Some((_, sum, pins)) = devices.iter_mut().find(|x| x.0 == d) {
                *sum += required;
                if !p.model.dynamic()? {
                    *pins += p.model.size()?;
                }
            } else {
                devices.push((
                    d,
                    required,
                    if p.model.dynamic()? {
                        0.
                    } else {
                        p.model.size()?
                    },
                ));
            }
        }
        for &(d, bytes, pins) in &devices {
            if d != Device::Cpu {
                self.free_memory(bytes * 1.1 + extra, Some(d), &[], dynamic, pins, services)?;
            }
        }
        for &(d, _, _) in &devices {
            if d != Device::Cpu && services.free(d)?.0 < minimum {
                self.free_memory(minimum, Some(d), &[], dynamic, 0., services)?;
            }
        }
        for (p, entry) in models.iter().zip(selected) {
            drop(loaded_model.replace(entry.clone()));
            let d = p.load_device()?;
            let mut budget = if self
                .policy
                .uses_lowvram_budget(d == Device::Cpu, force_full)
            {
                let loaded = entry.model_loaded_memory()?;
                self.policy
                    .lowvram_budget(services.free(d)?.0, loaded, minimum)
            } else {
                0.
            };
            if d != Device::Cpu && self.policy.vram_state == VramState::NoVram {
                budget = 0.1;
            }
            drop(entry.model_load(
                budget,
                force_patch,
                &self.finalizers,
                &self.registration_view(),
            )?);
            // These source queries execute even when detail logging is off.
            // A failure must occur before the registration is published.
            let vram_used = if d == Device::Cpu {
                0.
            } else {
                entry.model_loaded_memory()?
            };
            let _ram_used = if p.model.dynamic()? {
                p.model.loaded_ram_size()?
            } else {
                entry.model_memory()? - vram_used
            };
            self.loaded.borrow_mut().insert(0, entry);
        }
        Ok(())
    }
    pub fn unload_all(
        &mut self,
        device: Device,
        services: &mut dyn MemoryServices,
    ) -> Result<(), Error> {
        self.free_memory(1e30, Some(device), &[], false, 0., services)?;
        Ok(())
    }
}
