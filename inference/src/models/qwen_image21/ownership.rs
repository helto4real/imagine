//! Static Qwen patchers sharing physical parameters, backups, pins and sampler.
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
};
use tch::Device;

pub struct StaticModel<'a, B: super::ModelBuffers = super::FluxSampling> {
    pub owner: RefCell<StaticOffload<B>>,
    rt: &'a Runtime,
    token: Rc<()>,
    installed: Cell<Option<u64>>,
    next_selection: Cell<u64>,
    device: Cell<Device>,
    original: Rc<Selection>,
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
    pub fn new(owner: StaticOffload<B>, rt: &'a Runtime) -> Rc<Self> {
        let original = Rc::new(Selection {
            id: 0,
            patches: owner.patch_owner.patches.clone(),
            prefix: owner.patch_owner.key_prefix.clone(),
        });
        Rc::new(Self {
            owner: RefCell::new(owner),
            rt,
            token: Rc::new(()),
            installed: Cell::new(None),
            next_selection: Cell::new(1),
            device: Cell::new(Device::Cpu),
            original,
        })
    }
    pub fn patcher(self: &Rc<Self>) -> Handle<'a> {
        Patcher::new(StaticPatcher {
            shared: self.clone(),
            selection: self.original.clone(),
        })
    }
    /// Equivalent static ModelPatcher.clone with unchanged patch UUID.
    pub fn clone_patcher(self: &Rc<Self>, parent: &Handle<'a>) -> Result<Handle<'a>, Error> {
        if !Rc::ptr_eq(&parent.model.storage(), &self.token) {
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
        if !Rc::ptr_eq(&parent.model.storage(), &self.token) {
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
impl<B: super::ModelBuffers> Model for StaticPatcher<'_, B> {
    fn patch_selection(&self) -> Option<(u64, crate::patching::PatchMap, String)> {
        Some((
            self.selection.id,
            self.selection.patches.clone(),
            self.selection.prefix.clone(),
        ))
    }
    fn storage(&self) -> Rc<()> {
        self.shared.token.clone()
    }
    fn load_device(&self) -> Device {
        Device::Cuda(0)
    }
    fn current_device(&self) -> Device {
        self.shared.device.get()
    }
    fn size(&self) -> f64 {
        self.shared.owner.borrow().model_size() as f64
    }
    fn loaded_size(&self) -> f64 {
        self.shared.owner.borrow().placement.loaded_bytes as f64
    }
    fn partially_load(&self, mut extra: f64, force: bool) -> Result<f64, Error> {
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
            owner.load_after_unpatch(self.shared.rt, extra, force)
        } else {
            owner.partially_load_with_runtime(self.shared.rt, extra, force)
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
    fn partially_unload(&self, amount: f64) -> Result<f64, Error> {
        let mut owner = self.shared.owner.borrow_mut();
        let used = owner.placement.loaded_bytes;
        owner.partially_load_with_runtime(self.shared.rt, -amount, false)?;
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

/// Shared Linux static registrations. Static ModelPatcher is not an eligible
/// dynamic pin-eviction owner; exceeding its cap returns false, as upstream.
/// The registry retains tensors until successful unregister (also on errors).
#[derive(Clone)]
pub struct HostPins(Rc<RefCell<PinState>>);
struct PinState {
    native: Kitchen,
    limit: f64,
    pins: std::collections::BTreeMap<usize, tch::Tensor>,
}
impl HostPins {
    pub fn new(native: Kitchen) -> Result<Self, Error> {
        Self::with_limit(native, super::host_pin_limit()?)
    }
    pub fn with_limit(native: Kitchen, limit: f64) -> Result<Self, Error> {
        Ok(Self(Rc::new(RefCell::new(PinState {
            native,
            limit,
            pins: Default::default(),
        }))))
    }
    pub fn bytes(&self) -> usize {
        self.0
            .borrow()
            .pins
            .values()
            .map(|t| t.numel() * t.kind().elt_size_in_bytes())
            .sum()
    }
    pub fn pin(&self, t: &tch::Tensor) -> Result<bool, Error> {
        if self.0.borrow().limit <= 0. {
            return Ok(false);
        }
        if t.device() != Device::Cpu
            || !t.is_contiguous()
            || t.data_ptr().is_null()
            || t.f_is_pinned(Device::Cuda(0))?
        {
            return Ok(false);
        }
        let bytes = t.numel() * t.kind().elt_size_in_bytes();
        let total = self.bytes();
        let mut s = self.0.borrow_mut();
        if s.limit <= 0. || (total + bytes) as f64 > s.limit {
            return Ok(false);
        }
        if s.native.host_registration(t, true)? {
            s.pins.insert(t.data_ptr() as usize, t.shallow_clone());
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub fn unpin(&self, t: &tch::Tensor) -> Result<bool, Error> {
        let mut s = self.0.borrow_mut();
        let ptr = t.data_ptr() as usize;
        if s.limit <= 0.
            || t.device() != Device::Cpu
            || !s.pins.get(&ptr).is_some_and(|p| {
                p.numel() * p.kind().elt_size_in_bytes() == t.numel() * t.kind().elt_size_in_bytes()
            })
        {
            return Ok(false);
        }
        if s.native.host_registration(t, false)? {
            s.pins.remove(&ptr);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
impl Drop for PinState {
    fn drop(&mut self) {
        if !self.pins.is_empty() {
            let _ = self.native.synchronize_device(0);
        }
        for t in self.pins.values() {
            let _ = self.native.host_registration(t, false);
        }
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
