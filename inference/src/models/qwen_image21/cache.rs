//! Frozen Qwen select_prefix_cache and PoseBranchCache slot/quantization policy.
//! Memory accounting, pinning and stream transfers are explicit shared services.
use super::{CacheSelection, PrefixKv};
use crate::{execution::Runtime, execution_value::Value, native::Error};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

pub struct CacheOptions {
    pub device: String,
    pub dtype: String,
}
impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            device: "auto".into(),
            dtype: "default".into(),
        }
    }
}
/// The original offload stream's pending cast. synchronize() is a host wait
/// (select/free), while wait() synchronizes the consumer stream (take).
pub trait PendingCacheTensor {
    fn synchronize(&mut self) -> Result<(), Error>;
    fn wait(self: Box<Self>, device: Device) -> Result<Tensor, Error>;
}
/// Shared memory/transfer services preserve the source's inactive-eviction,
/// pin budget and two staging buffers per (shape,cast dtype), indexed by i%2.
pub trait CacheServices {
    fn patcher_free_memory(&mut self, device: Device) -> Result<f64, Error>;
    fn free_memory(&mut self, device: Device) -> Result<f64, Error>;
    fn ensure_pin_budget(&mut self, bytes: usize, evict_active: bool) -> Result<bool, Error>;
    fn pin(&mut self, t: &Tensor, evict_active: bool) -> Result<bool, Error>;
    fn unpin(&mut self, t: &Tensor) -> Result<(), Error>;
    fn prefetch(
        &mut self,
        t: &Tensor,
        block: usize,
        cast_dtype: Option<Kind>,
        device: Device,
    ) -> Result<Box<dyn PendingCacheTensor>, Error>;
    fn clear_staging(&mut self) -> Result<(), Error>;
}
struct Entry {
    payload: Tensor,
    quantized: Option<Value>,
    shape: Vec<i64>,
}
struct Slot {
    key: Tensor,
    blocks: BTreeMap<usize, Entry>,
    pinned: Vec<Tensor>,
}
impl Slot {
    fn bytes(&self) -> usize {
        self.blocks
            .values()
            .map(|e| e.payload.numel() * e.payload.kind().elt_size_in_bytes())
            .sum()
    }
}
pub struct PrefixCache<'a, S: CacheServices> {
    pub options: CacheOptions,
    pub services: S,
    rt: &'a Runtime,
    store: Option<Device>,
    dtype: String,
    slots: Vec<Slot>,
    selected: Option<usize>,
    pending: BTreeMap<usize, Box<dyn PendingCacheTensor>>,
}
impl<'a, S: CacheServices> PrefixCache<'a, S> {
    pub fn new(rt: &'a Runtime, options: CacheOptions, services: S) -> Self {
        Self {
            options,
            services,
            rt,
            store: None,
            dtype: "default".into(),
            slots: vec![],
            selected: None,
            pending: BTreeMap::new(),
        }
    }
    pub fn store_device(&self) -> Option<Device> {
        self.store
    }
    pub fn memory_bytes(&self) -> usize {
        self.slots.iter().map(Slot::bytes).sum()
    }
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }
    fn drain_pending(&mut self) -> Result<(), Error> {
        for p in self.pending.values_mut() {
            p.synchronize()?;
        }
        self.pending.clear();
        Ok(())
    }
    fn free_slot(&mut self, slot: Slot) -> Result<(), Error> {
        self.drain_pending()?;
        for tensor in slot.pinned {
            self.services.unpin(&tensor)?;
        }
        Ok(())
    }
    fn find(&mut self, key: &Tensor) -> Result<bool, Error> {
        self.drain_pending()?;
        for i in 0..self.slots.len() {
            let s = &self.slots[i];
            if s.key.size() == key.size() && s.key.f_equal(&key.f_to_device(s.key.device())?)? {
                let slot = self.slots.remove(i);
                self.slots.push(slot);
                self.selected = Some(self.slots.len() - 1);
                return Ok(true);
            }
        }
        self.selected = None;
        Ok(false)
    }
    fn create(&mut self, key: &Tensor) -> Result<(), Error> {
        // select(create=True) performs a second pending drain and key search.
        if self.find(key)? {
            return Ok(());
        }
        let estimate = self.slots.iter().map(Slot::bytes).max().unwrap_or(0) as f64 * 1.5;
        let store = self.store.unwrap();
        while !self.slots.is_empty() && self.services.free_memory(store)? < estimate {
            let slot = self.slots.remove(0);
            self.free_slot(slot)?;
        }
        self.slots.push(Slot {
            key: self.rt.native.clone_tensor(key)?.f_to_device(store)?,
            blocks: BTreeMap::new(),
            pinned: vec![],
        });
        self.selected = Some(self.slots.len() - 1);
        Ok(())
    }
}
impl<S: CacheServices> PrefixKv for PrefixCache<'_, S> {
    fn select(
        &mut self,
        key: &Tensor,
        mut bytes: usize,
        blocks: usize,
        device: Device,
    ) -> Result<CacheSelection, Error> {
        if self.options.device == "off" {
            return Ok(CacheSelection::Disabled);
        }
        if self.store.is_some() && self.find(key)? {
            return Ok(
                if self.slots[self.selected.unwrap()].blocks.len() == blocks {
                    CacheSelection::Hit
                } else {
                    CacheSelection::Fill
                },
            );
        }
        bytes /= match self.options.dtype.as_str() {
            "int8" => 2,
            "int4" => 4,
            _ => 1,
        };
        if self.store.is_none() {
            let store = match self.options.device.as_str() {
                "auto" => {
                    if self.services.patcher_free_memory(device)? > 4. * bytes as f64 {
                        device
                    } else if self.services.ensure_pin_budget(bytes, false)? {
                        Device::Cpu
                    } else {
                        return Ok(CacheSelection::Disabled);
                    }
                }
                "gpu" => device,
                _ => Device::Cpu,
            };
            self.store = Some(store);
            self.dtype = self.options.dtype.clone();
        }
        let room = if self.store == Some(device) {
            self.services.patcher_free_memory(device)? > 2. * bytes as f64
        } else {
            self.services.ensure_pin_budget(bytes, false)?
        };
        if !room {
            return Ok(CacheSelection::Disabled);
        }
        self.create(key)?;
        Ok(CacheSelection::Fill)
    }
    fn put(&mut self, index: usize, t: Tensor) -> Result<(), Error> {
        let shape = t.size();
        let store = self.store.unwrap();
        let (payload, quantized) = if matches!(self.dtype.as_str(), "int8" | "int4") {
            let width = *shape.last().unwrap();
            let mut group = 256;
            while group > 4 && width % group != 0 {
                group /= 4;
            }
            let layout = if self.dtype == "int4" {
                "TensorCoreConvRotW4A4Layout"
            } else {
                "TensorWiseINT8Layout"
            };
            let mut kwargs = vec![("convrot_groupsize".into(), Value::Int(group))];
            if self.dtype == "int8" {
                kwargs.extend([
                    ("is_weight".into(), Value::Bool(true)),
                    ("per_channel".into(), Value::Bool(true)),
                    ("convrot".into(), Value::Bool(true)),
                ]);
            }
            // Exact original quantizer; its default stochastic_rounding is kept.
            let q = unsafe {
                self.rt
                    .from_float(&t.f_reshape([-1, width])?, layout, &kwargs)?
            };
            let payload = q
                .payload
                .f_to_device_(store, q.payload.kind(), false, true)?;
            let mut params = q.params.shallow_copy();
            for v in params.tensors.values_mut() {
                *v = v.f_to_device(store)?;
            }
            let moved = q.with_parts(&self.rt.native, payload.shallow_clone(), params);
            (payload, Some(Value::quantized(moved)))
        } else {
            (t.f_to_device_(store, t.kind(), false, true)?, None)
        };
        let pinned = self.services.pin(&payload, false)?;
        let slot = &mut self.slots[self.selected.unwrap()];
        if pinned {
            slot.pinned.push(payload.shallow_clone());
        }
        slot.blocks.insert(
            index,
            Entry {
                payload,
                quantized,
                shape,
            },
        );
        Ok(())
    }
    fn prefetch(&mut self, index: usize, device: Device, dtype: Kind) -> Result<(), Error> {
        let slot = &self.slots[self.selected.unwrap()];
        let Some(entry) = slot.blocks.get(&index) else {
            return Ok(());
        };
        if self.pending.contains_key(&index) {
            return Ok(());
        }
        let cast = if entry.quantized.is_some() {
            None
        } else {
            Some(dtype)
        };
        self.pending.insert(
            index,
            self.services
                .prefetch(&entry.payload, index, cast, device)?,
        );
        Ok(())
    }
    fn take(
        &mut self,
        index: usize,
        device: Device,
        dtype: Kind,
        batch: i64,
    ) -> Result<Tensor, Error> {
        if !self.pending.contains_key(&index) {
            self.prefetch(index, device, dtype)?;
        }
        let mut t = self
            .pending
            .remove(&index)
            .ok_or_else(|| Error::Native("Missing selected prefix cache block".into()))?
            .wait(device)?;
        let entry = &self.slots[self.selected.unwrap()].blocks[&index];
        if let Some(value) = &entry.quantized {
            let q = value.as_quantized()?;
            let mut params = q.params.shallow_copy();
            for v in params.tensors.values_mut() {
                *v = v.f_to_device(t.device())?;
            }
            let q = q.with_parts(&self.rt.native, t, params);
            t = unsafe { self.rt.decode(&q)? }
                .f_reshape(entry.shape.as_slice())?
                .f_to_kind(dtype)?;
        }
        // comfy.utils.repeat_to_batch_size: truncate, otherwise ceil-repeat.
        if t.size()[0] > batch {
            t = t.f_slice(0, 0, batch, 1)?;
        } else if t.size()[0] < batch {
            let mut repeats = vec![1; t.dim()];
            repeats[0] = (batch + t.size()[0] - 1) / t.size()[0];
            t = t.f_repeat(repeats.as_slice())?.f_slice(0, 0, batch, 1)?;
        }
        Ok(t)
    }
    fn free(&mut self) -> Result<(), Error> {
        for slot in std::mem::take(&mut self.slots) {
            self.free_slot(slot)?;
        }
        self.selected = None;
        self.store = None;
        self.pending.clear();
        self.services.clear_staging()?;
        Ok(())
    }
}
