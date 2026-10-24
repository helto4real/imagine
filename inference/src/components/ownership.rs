//! Static component patchers sharing physical parameters, backups, pins and sampler.
//! A patch selection belongs to the patcher; installed weights belong to storage.
use super::{StaticOffload, host_available_memory, prepare_component_patched_modules};
use crate::{
    execution::Runtime,
    memory::{Handle, MemoryServices, Model, Patcher},
    native::{Error, Kitchen},
    patching::PatchOwner,
};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::Arc,
};
use tch::{Device, Kind};

pub struct StaticModel<'a, B: super::ModelBuffers> {
    // Weak callbacks precede model fields, matching module weakref teardown.
    callbacks: crate::memory_finalizers::WeakCallbacks<'a>,
    pub owner: RefCell<StaticOffload<B>>,
    rt: ModelRuntime<'a>,
    // BaseModel.get_dtype(), distinct from manual_cast_dtype. Component
    // wrappers without get_dtype return None through ModelPatcher.
    source_dtype: Option<Kind>,
    installed: Cell<Option<u64>>,
    next_selection: Cell<u64>,
    device: Cell<Device>,
    original: Rc<Selection>,
}
enum ModelRuntime<'a> {
    Borrowed(&'a Runtime),
    Owned(Arc<Runtime>),
}
impl ModelRuntime<'_> {
    fn get(&self) -> &Runtime {
        match self {
            Self::Borrowed(runtime) => runtime,
            Self::Owned(runtime) => runtime,
        }
    }
}
struct Selection {
    id: u64,
    patches: crate::patching::PatchMap,
    prefix: String,
}
struct StaticPatcher<'a, B: super::ModelBuffers> {
    shared: Rc<StaticModel<'a, B>>,
    selection: Rc<Selection>,
}
impl<'a, B: super::ModelBuffers + 'a> StaticModel<'a, B> {
    pub fn new(owner: StaticOffload<B>, rt: &'a Runtime, source_dtype: Option<Kind>) -> Rc<Self> {
        Self::from_runtime(owner, ModelRuntime::Borrowed(rt), source_dtype)
    }
    fn from_runtime(
        owner: StaticOffload<B>,
        rt: ModelRuntime<'a>,
        source_dtype: Option<Kind>,
    ) -> Rc<Self> {
        let original = Rc::new(Selection {
            id: 0,
            patches: owner.patch_owner.patches.clone(),
            prefix: owner.patch_owner.key_prefix.clone(),
        });
        Rc::new(Self {
            callbacks: crate::memory_finalizers::WeakCallbacks::new(),
            owner: RefCell::new(owner),
            rt,
            source_dtype,
            installed: Cell::new(None),
            next_selection: Cell::new(1),
            device: Cell::new(Device::Cpu),
            original,
        })
    }
    pub fn patcher(self: &Rc<Self>) -> Handle<'a> {
        self.owner.borrow_mut().bind_native_patcher_lifecycle();
        Patcher::new(StaticPatcher {
            shared: self.clone(),
            selection: self.original.clone(),
        })
    }
    /// Equivalent static ModelPatcher.clone with unchanged patch UUID.
    pub fn clone_patcher(self: &Rc<Self>, parent: &Handle<'a>) -> Result<Handle<'a>, Error> {
        let physical: crate::memory_finalizers::Handle<'a> = self.clone();
        if !Rc::ptr_eq(&parent.model.storage(), &physical) {
            return Err(Error::Native(
                "Clone parent belongs to another physical model".into(),
            ));
        }
        let (id, patches, prefix) = parent
            .model
            .patch_selection()
            .ok_or_else(|| Error::Native("Missing static patch configuration".into()))?;
        Ok(Patcher::with_parent(
            StaticPatcher {
                shared: self.clone(),
                selection: Rc::new(Selection {
                    id,
                    patches,
                    prefix,
                }),
            },
            parent,
        ))
    }
    /// A changed patch set gets a new identity, exactly as add_patches changes
    /// patches_uuid. Merely retaining a handle does not change patch identity.
    pub fn with_patches(
        self: &Rc<Self>,
        parent: &Handle<'a>,
        patches: PatchOwner,
    ) -> Result<Handle<'a>, Error> {
        let physical: crate::memory_finalizers::Handle<'a> = self.clone();
        if !Rc::ptr_eq(&parent.model.storage(), &physical) {
            return Err(Error::Native(
                "Patch parent belongs to another physical model".into(),
            ));
        }
        let id = self.next_selection.get();
        self.next_selection.set(id + 1);
        Ok(Patcher::with_parent(
            StaticPatcher {
                shared: self.clone(),
                selection: Rc::new(Selection {
                    id,
                    patches: patches.patches,
                    prefix: patches.key_prefix,
                }),
            },
            parent,
        ))
    }
}
impl<'a, B: super::ModelBuffers + 'a> StaticModel<'a, B> {
    /// Owned shared runtime for canonical native owners stored in Options.
    /// No borrowed lifetime is extended and no model or parameter is copied.
    /// The borrowed constructor retains its scoped runtime ownership.
    pub fn new_owned(
        owner: StaticOffload<B>,
        rt: Arc<Runtime>,
        source_dtype: Option<Kind>,
    ) -> Rc<Self> {
        Self::from_runtime(owner, ModelRuntime::Owned(rt), source_dtype)
    }
}
impl<'a, B: super::ModelBuffers + 'a> crate::memory_finalizers::Target<'a> for StaticModel<'a, B> {
    fn weak_callbacks(&self) -> &crate::memory_finalizers::WeakCallbacks<'a> {
        &self.callbacks
    }
}
impl<'a, B: super::ModelBuffers + 'a> Model<'a> for StaticPatcher<'a, B> {
    fn native_lifecycle(&self) -> Option<&dyn crate::memory::NativeLifecycle> {
        Some(self)
    }
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn model_dtype(&self) -> Result<Option<Kind>, Error> {
        Ok(self.shared.source_dtype)
    }
    fn model_patches_to(&self, _target: crate::memory::PatchTarget) -> Result<(), Error> {
        // This admitted Stage 2 profile has no transformer option patch,
        // model-function wrapper or weight-wrapper carriers. General source
        // owners must bind model_patch_calls over their actual option fields.
        Ok(())
    }
    fn patch_selection(&self) -> Option<(u64, crate::patching::PatchMap, String)> {
        Some((
            self.selection.id,
            self.selection.patches.clone(),
            self.selection.prefix.clone(),
        ))
    }
    fn storage(&self) -> crate::memory_finalizers::Handle<'a> {
        self.shared.clone()
    }
    fn offload_device(&self) -> Result<Device, Error> {
        Ok(Device::Cpu)
    }
    fn load_device(&self) -> Device {
        Device::Cuda(0)
    }
    fn current_device(&self) -> Result<Device, Error> {
        Ok(self.shared.device.get())
    }
    fn size(&self) -> Result<f64, Error> {
        Ok(self.shared.owner.borrow().model_size() as f64)
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok(self.shared.owner.borrow().placement.loaded_bytes as f64)
    }
    fn partially_load(&self, device: Device, mut extra: f64, force: bool) -> Result<f64, Error> {
        if device != Device::Cuda(0) {
            return Err(Error::BindingGap {
                backend: "static_qwen_memory".into(),
                operation: "Unbound static load target".into(),
            });
        }
        let mut owner = self.shared.owner.borrow_mut();
        let used = owner.placement.loaded_bytes;
        let unpatch = self
            .shared
            .installed
            .get()
            .is_some_and(|id| id != self.selection.id || force);
        if unpatch {
            owner.unload()?;
            self.shared.installed.set(None);
            self.shared.device.set(Device::Cpu);
            extra += used as f64;
        }
        owner.patch_owner.patches = self.selection.patches.clone();
        owner
            .patch_owner
            .key_prefix
            .clone_from(&self.selection.prefix);
        owner.modules = prepare_component_patched_modules(
            &owner.parameters,
            &owner.patch_owner,
            owner.modules.clone(),
            owner.cast_dtype,
        )?;
        let negative = extra < 0. && !unpatch;
        let current = owner.placement.loaded_bytes;
        let result = if unpatch {
            owner.load_after_unpatch(self.shared.rt.get(), extra, force)
        } else {
            owner.partially_load_with_runtime(self.shared.rt.get(), extra, force)
        };
        if let Err(e) = result {
            self.shared.installed.set(None);
            self.shared.device.set(Device::Cpu);
            return Err(e);
        }
        if !negative {
            self.shared.installed.set(Some(self.selection.id));
            self.shared.device.set(Device::Cuda(0));
        }
        Ok(if negative {
            0.
        } else {
            (owner.placement.loaded_bytes - current) as f64
        })
    }
    fn partially_unload(&self, device: Device, amount: f64) -> Result<f64, Error> {
        if device != Device::Cpu {
            return Err(Error::BindingGap {
                backend: "static_qwen_memory".into(),
                operation: "Unbound static offload target".into(),
            });
        }
        let mut owner = self.shared.owner.borrow_mut();
        let used = owner.placement.loaded_bytes;
        owner.partially_load_with_runtime(self.shared.rt.get(), -amount, false)?;
        Ok((used - owner.placement.loaded_bytes) as f64)
    }
    fn detach(&self, unpatch: bool) -> Result<(), Error> {
        if unpatch {
            self.shared.owner.borrow_mut().unload()?;
            self.shared.installed.set(None);
            self.shared.device.set(Device::Cpu);
        }
        Ok(())
    }
}

impl<'a, B: super::ModelBuffers + 'a> crate::memory::NativeLifecycle for StaticPatcher<'a, B> {
    fn finalize(&self) -> Result<(), Error> {
        self.shared.owner.borrow_mut().unpin_all_weights()?;
        Model::detach(self, false)
    }
    fn unraisable(&self, error: Error) {
        eprintln!("Exception ignored in ModelPatcher.__del__: {error}");
    }
}

/// Shared source registration accounting for the admitted Linux static owners.
/// The registry owns pointer/size pairs only. Tensor owners perform unpinning.
#[derive(Clone)]
pub struct HostPins(Rc<PinState>);
struct PinState {
    native: Kitchen,
    registry: crate::memory::pinning::PinRegistry,
}
struct NativePins<'a> {
    state: &'a PinState,
}
impl crate::memory::pinning::PinBindings for NativePins<'_> {
    type Tensor = tch::Tensor;
    fn allowed_type(&mut self, _: &tch::Tensor) -> Result<bool, Error> {
        Ok(true)
    }
    fn is_cpu(&mut self, t: &tch::Tensor) -> Result<bool, Error> {
        Ok(t.device() == Device::Cpu)
    }
    fn is_pinned(&mut self, t: &tch::Tensor) -> Result<bool, Error> {
        Ok(t.f_is_pinned(Device::Cuda(0))?)
    }
    fn is_contiguous(&mut self, t: &tch::Tensor) -> Result<bool, Error> {
        Ok(t.is_contiguous())
    }
    fn nbytes(&mut self, t: &tch::Tensor) -> Result<usize, Error> {
        Ok(t.numel() * t.kind().elt_size_in_bytes())
    }
    fn extra_ram_release(&mut self) -> Result<(), Error> {
        // No graph RAM-cache release callback is installed by this admitted
        // synchronous profile, matching extra_ram_release_callback=None.
        Ok(())
    }
    fn ensure_registerable(&mut self, size: usize, _evict_active: bool) -> Result<bool, Error> {
        // Static patchers are excluded by models_for_pin_eviction. With no
        // dynamic owners there is no candidate to change this shortfall.
        Ok((self
            .state
            .registry
            .bytes()
            .checked_add(size)
            .ok_or_else(|| Error::BindingGap {
                backend: "host_registration".into(),
                operation: "registration budget requires arbitrary precision source arithmetic"
                    .into(),
            })? as f64)
            <= self.state.registry.limit.get())
    }
    fn pointer(&mut self, t: &tch::Tensor) -> Result<usize, Error> {
        Ok(t.data_ptr() as usize)
    }
    fn register(&mut self, pointer: usize, size: usize) -> Result<bool, Error> {
        // The pin body retains its tensor argument throughout the operation.
        unsafe { self.state.native.host_registration_raw(pointer, size, true) }
    }
    fn unregister(&mut self, pointer: usize) -> Result<bool, Error> {
        // The unpin body retains its tensor argument and live allocation.
        unsafe { self.state.native.host_registration_raw(pointer, 0, false) }
    }
    fn warning(&mut self, message: &'static str) -> Result<(), Error> {
        eprintln!("WARNING:root:{message}");
        Ok(())
    }
    fn discard_async_error(&mut self) -> Result<(), Error> {
        crate::memory::exceptions::discard_native_cuda_async_error(
            &self.state.native,
            Device::Cuda(0),
        )
    }
}
impl HostPins {
    pub fn new(native: Kitchen) -> Result<Self, Error> {
        Self::with_limit(native, super::host_pin_limit()?)
    }
    pub fn with_limit(native: Kitchen, limit: f64) -> Result<Self, Error> {
        Ok(Self(Rc::new(PinState {
            native,
            registry: crate::memory::pinning::PinRegistry::new(limit),
        })))
    }
    pub fn bytes(&self) -> usize {
        self.0.registry.bytes()
    }
    pub fn pin(&self, t: &tch::Tensor) -> Result<bool, Error> {
        self.0
            .registry
            .pin(t, true, &mut NativePins { state: &self.0 })
    }
    pub fn unpin(&self, t: &tch::Tensor) -> Result<bool, Error> {
        self.0.registry.unpin(t, &mut NativePins { state: &self.0 })
    }
}
pub struct StaticMemoryServices {
    native: Kitchen,
    pub pins: HostPins,
}
impl StaticMemoryServices {
    pub fn new(native: Kitchen, pins: HostPins) -> Self {
        Self { native, pins }
    }
}
impl MemoryServices for StaticMemoryServices {
    fn free(&mut self, device: Device) -> Result<(f64, f64), Error> {
        if device == Device::Cpu {
            return Ok((host_available_memory()? as f64, 0.));
        }
        let Device::Cuda(index) = device else {
            return Err(Error::BindingGap {
                backend: "static_memory".into(),
                operation: "Non Linux CUDA/CPU device".into(),
            });
        };
        let [free, _, active, reserved] = self.native.cuda_memory(index)?;
        Ok((
            (free + reserved - active) as f64,
            (reserved - active) as f64,
        ))
    }
    fn empty_cache(&mut self) -> Result<(), Error> {
        self.native.cuda_soft_empty_cache(0)
    }
    fn ensure_pins(&mut self, _bytes: f64) -> Result<(), Error> {
        // Original callers ignore both boolean results. With static patchers
        // no dynamic owner is eligible for RAM or registration eviction.
        Ok(())
    }
}
