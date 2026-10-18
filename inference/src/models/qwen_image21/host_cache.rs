//! Original host prefix cache services for synchronous static execution.
//! Model/cache registrations can share HostPins; dynamic pin eviction and graph
//! RAM-cache callbacks have separate bindings.
use super::{
    CacheServices, PendingCacheTensor, ResidentCacheServices, host_available_memory,
    host_pin_limit, single_owner_pin_budget,
};
use crate::native::{Error, Kitchen};
use std::collections::BTreeMap;
use tch::{Device, Kind, Tensor};

pub struct StaticCacheServices {
    resident: ResidentCacheServices,
    pin_limit: f64,
    other_pinned_bytes: usize,
    shared_pins: Option<super::HostPins>,
    pinned: BTreeMap<usize, Tensor>,
    staging: BTreeMap<(Vec<i64>, String), [Tensor; 2]>,
}
struct StaticPending(Tensor);
impl PendingCacheTensor for StaticPending {
    fn synchronize(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn wait(self: Box<Self>, _device: Device) -> Result<Tensor, Error> {
        Ok(self.0)
    }
}
impl StaticCacheServices {
    /// Legacy explicit single-owner accounting. The shared coordinator uses
    /// with_shared_pins instead of freezing another owner's byte count.
    pub fn new(native: Kitchen, other_pinned_bytes: usize) -> Result<Self, Error> {
        Ok(Self {
            resident: ResidentCacheServices { native },
            pin_limit: host_pin_limit()?,
            other_pinned_bytes,
            shared_pins: None,
            pinned: BTreeMap::new(),
            staging: BTreeMap::new(),
        })
    }
    pub fn with_shared_pins(native: Kitchen, pins: super::HostPins) -> Result<Self, Error> {
        let mut services = Self::new(native, 0)?;
        services.shared_pins = Some(pins);
        Ok(services)
    }
    pub fn pinned_bytes(&self) -> usize {
        self.pinned
            .values()
            .map(|t| t.numel() * t.kind().elt_size_in_bytes())
            .sum()
    }
    pub fn staging_buffers(&self) -> usize {
        self.staging.len() * 2
    }
}
impl CacheServices for StaticCacheServices {
    fn patcher_free_memory(&mut self, device: Device) -> Result<f64, Error> {
        self.free_memory(device)
    }
    fn free_memory(&mut self, device: Device) -> Result<f64, Error> {
        if device == Device::Cpu {
            Ok(host_available_memory()? as f64)
        } else {
            self.resident.free_memory(device)
        }
    }
    fn ensure_pin_budget(&mut self, bytes: usize, _evict_active: bool) -> Result<bool, Error> {
        // Original ensure_pin_budget is a RAM-pressure check, distinct from
        // ensure_pin_registerable. Static owners are not eviction candidates.
        let room = single_owner_pin_budget(bytes, host_available_memory()?);
        Ok(room)
    }
    fn pin(&mut self, t: &Tensor, _evict_active: bool) -> Result<bool, Error> {
        if let Some(pins) = &self.shared_pins {
            let registered = pins.pin(t)?;
            if registered {
                self.pinned.insert(t.data_ptr() as usize, t.shallow_clone());
            }
            return Ok(registered);
        }
        if self.pin_limit <= 0. || t.device() != Device::Cpu || !t.is_contiguous() {
            return Ok(false);
        }
        let bytes = t.numel() * t.kind().elt_size_in_bytes();
        if (self.other_pinned_bytes + self.pinned_bytes() + bytes) as f64 > self.pin_limit {
            return Ok(false);
        }
        if self.resident.native.host_registration(t, true)? {
            self.pinned.insert(t.data_ptr() as usize, t.shallow_clone());
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn unpin(&mut self, t: &Tensor) -> Result<(), Error> {
        let ptr = t.data_ptr() as usize;
        if let Some(original) = self.pinned.get(&ptr)
            && original.numel() * original.kind().elt_size_in_bytes()
                == t.numel() * t.kind().elt_size_in_bytes()
            && if let Some(pins) = &self.shared_pins {
                pins.unpin(t)?
            } else {
                self.resident.native.host_registration(t, false)?
            }
        {
            self.pinned.remove(&ptr);
        }
        Ok(())
    }
    fn prefetch(
        &mut self,
        t: &Tensor,
        block: usize,
        kind: Option<Kind>,
        device: Device,
    ) -> Result<Box<dyn PendingCacheTensor>, Error> {
        if t.device() == device {
            return self.resident.prefetch(t, block, kind, device);
        }
        let kind = kind.unwrap_or(t.kind());
        let key = (t.size(), format!("{kind:?}"));
        if !self.staging.contains_key(&key) {
            self.staging.insert(
                key.clone(),
                [
                    Tensor::f_empty(t.size(), (kind, device))?,
                    Tensor::f_empty(t.size(), (kind, device))?,
                ],
            );
        }
        let buffer = &mut self.staging.get_mut(&key).unwrap()[block % 2];
        self.resident.native.copy_tensor(buffer, t, true)?;
        // Original NUM_STREAMS=0: current stream copy and no pending event.
        Ok(Box::new(StaticPending(buffer.shallow_clone())))
    }
    fn clear_staging(&mut self) -> Result<(), Error> {
        self.staging.clear();
        Ok(())
    }
}
impl Drop for StaticCacheServices {
    fn drop(&mut self) {
        // Error/cancellation fallback: keep buffers alive through pending copies
        // before releasing host registrations. Normal PrefixCache.free unpins.
        if !self.pinned.is_empty() {
            let _ = self.resident.native.synchronize_device(0);
            for tensor in self.pinned.values() {
                if let Some(pins) = &self.shared_pins {
                    let _ = pins.unpin(tensor);
                } else {
                    let _ = self.resident.native.host_registration(tensor, false);
                }
            }
        }
    }
}
