//! Frozen model_management.LoadedModel and the Linux/NORMAL_VRAM smart-memory
//! coordinator. Registrations are weak; model storage belongs to patchers and
//! execution users. Device queries and patcher actions stay explicit so bounded
//! original-code oracles can exercise pressure without allocating model weights.
use crate::native::Error;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};
use tch::Device;

pub trait Model {
    fn storage(&self) -> Rc<()>;
    fn load_device(&self) -> Device;
    fn current_device(&self) -> Device;
    fn size(&self) -> f64;
    fn loaded_size(&self) -> f64;
    fn partially_load(&self, extra: f64, force_patch: bool) -> Result<f64, Error>;
    fn partially_unload(&self, amount: f64) -> Result<f64, Error>;
    fn detach(&self, unpatch: bool) -> Result<(), Error>;
    fn dynamic(&self) -> bool {
        false
    }
    fn patch_selection(&self) -> Option<(u64, crate::patching::PatchMap, String)> {
        None
    }
}

/// A clone patcher has a distinct identity, a strong parent and shared model
/// storage. Cloning this handle itself is ordinary reference retention.
pub struct Patcher<'a> {
    pub model: Box<dyn Model + 'a>,
    parent: Option<Handle<'a>>,
    additional: Vec<Handle<'a>>,
}
pub type Handle<'a> = Rc<Patcher<'a>>;
impl<'a> Patcher<'a> {
    pub fn new(model: impl Model + 'a) -> Handle<'a> {
        Rc::new(Self {
            model: Box::new(model),
            parent: None,
            additional: vec![],
        })
    }
    pub fn with_parent(model: impl Model + 'a, parent: &Handle<'a>) -> Handle<'a> {
        Rc::new(Self {
            model: Box::new(model),
            parent: Some(parent.clone()),
            additional: vec![],
        })
    }
    pub fn with_additional(model: impl Model + 'a, additional: Vec<Handle<'a>>) -> Handle<'a> {
        Rc::new(Self {
            model: Box::new(model),
            parent: None,
            additional,
        })
    }
}
struct Loaded<'a> {
    patcher: Weak<Patcher<'a>>,
    parent: Option<Weak<Patcher<'a>>>,
    real: Weak<()>,
    used: bool,
    device: Device,
}
impl<'a> Loaded<'a> {
    fn new(p: &Handle<'a>) -> Self {
        Self {
            patcher: Rc::downgrade(p),
            parent: p.parent.as_ref().map(Rc::downgrade),
            real: Weak::new(),
            used: true,
            device: p.model.load_device(),
        }
    }
    fn patcher(&mut self) -> Option<Handle<'a>> {
        if let Some(p) = self.patcher.upgrade() {
            return Some(p);
        }
        let parent = self.parent.as_ref()?.upgrade()?;
        self.patcher = Rc::downgrade(&parent);
        self.parent = parent.parent.as_ref().map(Rc::downgrade);
        self.device = parent.model.load_device();
        Some(parent)
    }
}

pub trait MemoryServices {
    /// Comfy get_free_memory's total-free and unused Torch-reserved bytes.
    fn free(&mut self, device: Device) -> Result<(f64, f64), Error>;
    fn empty_cache(&mut self) -> Result<(), Error>;
    fn ensure_pins(&mut self, bytes: f64) -> Result<(), Error>;
}

#[derive(Default)]
pub struct Manager<'a> {
    loaded: Vec<Rc<RefCell<Loaded<'a>>>>,
}
impl<'a> Manager<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cleanup(&mut self) {
        for e in &self.loaded {
            e.borrow_mut().patcher();
        }
        self.loaded.retain(|e| e.borrow().real.strong_count() > 0);
    }
    pub fn loaded(&mut self, only_used: bool) -> Vec<Handle<'a>> {
        self.loaded
            .iter()
            .filter_map(|e| {
                let mut e = e.borrow_mut();
                if only_used && !e.used {
                    None
                } else {
                    e.patcher()
                }
            })
            .collect()
    }
    pub fn registrations(&self) -> usize {
        self.loaded.len()
    }
    pub fn states(&mut self) -> Vec<(Option<Handle<'a>>, bool, Device)> {
        self.loaded
            .iter()
            .map(|e| {
                let mut e = e.borrow_mut();
                (e.patcher(), e.used, e.device)
            })
            .collect()
    }
    /// Original free_memory ordering: most offloaded, least retained, smallest,
    /// then original list index. Partial success keeps its registration.
    pub fn free_memory(
        &mut self,
        required: f64,
        device: Option<Device>,
        keep: &[Handle<'a>],
        dynamic: bool,
        pins: f64,
        services: &mut dyn MemoryServices,
    ) -> Result<Vec<Handle<'a>>, Error> {
        self.cleanup();
        let mut candidates = vec![];
        for (i, entry) in self.loaded.iter().enumerate().rev() {
            let mut e = entry.borrow_mut();
            if device.is_some_and(|d| d != e.device) {
                continue;
            }
            if let Some(p) = e.patcher() {
                if keep.iter().any(|k| Rc::ptr_eq(k, &p)) {
                    continue;
                }
                e.used = false;
                candidates.push((
                    -(p.model.size() - p.model.loaded_size()),
                    Rc::strong_count(&p),
                    p.model.size(),
                    i,
                ));
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
            let mut entry = self.loaded[i].borrow_mut();
            let Some(p) = entry.patcher() else {
                continue;
            };
            // device=None under smart memory intentionally marks inactive but
            // requests zero bytes; unload_all_models passes a concrete device.
            let mut amount = if let Some(d) = device {
                required - services.free(d)?.0
            } else {
                0.
            };
            if p.model.dynamic() && dynamic {
                required -= p.model.loaded_size();
                amount = 0.;
            }
            if amount > 0. {
                let partial =
                    amount < p.model.loaded_size() && p.model.partially_unload(amount)? >= amount;
                if !partial {
                    p.model.detach(true)?;
                    entry.real = Weak::new();
                    removed.push((i, p));
                }
            }
        }
        removed.sort_by_key(|(i, _)| std::cmp::Reverse(*i));
        for (i, _) in &removed {
            self.loaded.remove(*i);
        }
        if !dynamic && pins > 0. {
            services.ensure_pins(pins)?;
        }
        if !removed.is_empty() {
            services.empty_cache()?;
        } else if let Some(d) = device {
            let (total, torch) = services.free(d)?;
            if torch > total * 0.25 {
                services.empty_cache()?;
            }
        }
        Ok(removed.into_iter().map(|(_, p)| p).collect())
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
        self.cleanup();
        let inference = crate::models::qwen_image21::minimum_inference_memory();
        let reserve = crate::models::qwen_image21::LINUX_RESERVED;
        let extra = inference.max(memory + reserve);
        let minimum = minimum.map_or(extra, |m| inference.max(m + reserve));
        let mut models = vec![];
        for p in requested {
            for model in std::iter::once(p).chain(p.additional.iter()) {
                if !models.iter().any(|m| Rc::ptr_eq(m, model)) {
                    models.push(model.clone());
                }
            }
        }
        models.reverse();
        let dynamic = models.iter().all(|p| p.model.dynamic());
        let mut selected = vec![];
        for p in &models {
            let found = self
                .loaded
                .iter()
                .find(|e| e.borrow_mut().patcher().is_some_and(|x| Rc::ptr_eq(&x, p)))
                .cloned();
            let e = found.unwrap_or_else(|| Rc::new(RefCell::new(Loaded::new(p))));
            e.borrow_mut().used = true;
            selected.push(e);
        }
        // is_clone includes self. Preserve weights via detach(false), remove
        // registrations before asking the shared manager for space.
        for p in &models {
            let storage = p.model.storage();
            for i in (0..self.loaded.len()).rev() {
                let candidate = self.loaded[i].borrow_mut().patcher();
                if let Some(c) = candidate
                    && Rc::ptr_eq(&storage, &c.model.storage())
                {
                    self.loaded.remove(i);
                    c.model.detach(false)?;
                }
            }
        }
        let mut devices: Vec<(Device, f64, f64)> = vec![];
        for p in &models {
            let d = p.model.load_device();
            let required = p.model.size()
                - if d == p.model.current_device() {
                    p.model.loaded_size()
                } else {
                    0.
                };
            if let Some((_, sum, pins)) = devices.iter_mut().find(|x| x.0 == d) {
                *sum += required;
                if !p.model.dynamic() {
                    *pins += p.model.size();
                }
            } else {
                devices.push((
                    d,
                    required,
                    if p.model.dynamic() {
                        0.
                    } else {
                        p.model.size()
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
            let d = p.model.load_device();
            let budget = if d == Device::Cpu || force_full {
                0.
            } else {
                crate::models::qwen_image21::static_load_budget(
                    services.free(d)?.0,
                    p.model.loaded_size(),
                    Some(minimum - reserve),
                )
            };
            p.model
                .partially_load(if budget == 0. { 1e32 } else { budget }, force_patch)?;
            entry.borrow_mut().real = Rc::downgrade(&p.model.storage());
            self.loaded.insert(0, entry);
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
