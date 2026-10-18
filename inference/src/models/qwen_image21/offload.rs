//! Frozen static ModelPatcher, synchronous BF16 weights, no patches/vbars.
//! Placement arithmetic is independently tested against the original methods.
use crate::{
    checkpoint::PreparedParameters,
    execution_value::Value,
    native::{Error, Kitchen},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};
use tch::{Device, Kind, Tensor};

pub const LINUX_RESERVED: f64 = (400 * 1024 * 1024) as f64;
pub fn minimum_inference_memory() -> f64 {
    0.8 * (1024_u64.pow(3) as f64) + LINUX_RESERVED
}
/// load_models_gpu: NORMAL_VRAM/Nvidia, smart memory, no reserve override.
pub fn static_load_budget(free: f64, loaded: f64, minimum: Option<f64>) -> f64 {
    let minimum = minimum
        .map(|m| minimum_inference_memory().max(m + LINUX_RESERVED))
        .unwrap_or(minimum_inference_memory());
    let current = free + loaded;
    let budget = 0_f64
        .max(current - minimum)
        .max(0_f64.min(current - minimum_inference_memory()))
        - loaded;
    if budget == 0. { 0.1 } else { budget }
}
/// BaseModel.memory_required, QwenImage21 factor=6, BF16 + PyTorch flash SDPA,
/// no reference latents. The batch doubled by estimate_memory is explicit.
pub fn qwen_sampling_memory(batch: i64, height: i64, width: i64) -> f64 {
    ((batch * height * width) as f64 * 2. * 0.01 * 6.) * (1024 * 1024) as f64
}
#[derive(Clone, Deserialize, Serialize)]
pub struct OffloadModule {
    pub name: String,
    pub module_bytes: u64,
    pub offload_bytes: u64,
    pub cast_weights: bool,
    pub parameters: Vec<String>,
}
#[derive(Default, Clone, Deserialize, Serialize)]
pub struct StaticPlacement {
    pub resident: BTreeSet<String>,
    pub loaded_bytes: u64,
    pub buffer_bytes: u64,
}
fn sorted(modules: &[OffloadModule], reverse: bool) -> Vec<&OffloadModule> {
    let mut list = modules.iter().collect::<Vec<_>>();
    list.sort_by(|a, b| {
        (a.offload_bytes, a.module_bytes, &a.name).cmp(&(b.offload_bytes, b.module_bytes, &b.name))
    });
    if reverse {
        list.reverse();
    }
    list
}
impl StaticPlacement {
    pub fn reload(&self, modules: &[OffloadModule], extra: f64) -> Self {
        if extra < 0. {
            let mut next = self.clone();
            next.unload(modules, -extra);
            return next;
        }
        let total: u64 = modules.iter().map(|m| m.module_bytes).sum();
        if self.loaded_bytes == total {
            return self.clone();
        }
        let budget = self.loaded_bytes as f64 + extra;
        let mut next = Self::load(modules, budget, budget > total as f64);
        // Original load() does not move already-patched CUDA parameters back
        // to CPU merely because a new fit calculation skips their module.
        // Preserve those objects independently of its recomputed byte counter.
        next.resident.extend(self.resident.iter().cloned());
        next
    }
    /// Original NUM_STREAMS=0 branch. Existing resident entries are patched only
    /// once; callers perform original partial-unload before extending a load.
    pub fn load(modules: &[OffloadModule], budget: f64, full: bool) -> Self {
        let mut out = Self::default();
        for m in sorted(modules, true) {
            let potential = out.buffer_bytes.max(m.offload_bytes);
            let fits = ((out.loaded_bytes + m.module_bytes + potential) as f64) < budget;
            if !full && m.cast_weights && !fits {
                out.buffer_bytes = potential;
            } else if full || fits {
                out.resident.insert(m.name.clone());
                out.loaded_bytes += m.module_bytes;
            } else {
                out.buffer_bytes = potential;
            }
        }
        out
    }
    pub fn unload(&mut self, modules: &[OffloadModule], amount: f64) -> u64 {
        let original_buffer = self.buffer_bytes;
        let mut freed = 0;
        for m in sorted(modules, false) {
            if amount + self.buffer_bytes as f64 - (original_buffer as f64) < freed as f64 {
                break;
            }
            if self.resident.remove(&m.name) {
                freed += m.module_bytes;
                self.buffer_bytes = self.buffer_bytes.max(m.offload_bytes);
            }
        }
        self.loaded_bytes -= freed;
        freed
    }
}
fn gap(s: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "static_bf16_offload".into(),
        operation: s.into(),
    }
}
pub fn bf16_dit_modules() -> Result<Vec<OffloadModule>, Error> {
    serde_json::from_str(include_str!(
        "../../../resources/qwen-image21-offload-modules.json"
    ))
    .map_err(|e| Error::Native(e.to_string()))
}
/// Linux host/cgroup totals used for static cudaHostRegister's original cap.
/// Only aggregate memory files and this process's cgroup ancestry are read.
pub fn host_pin_limit() -> Result<f64, Error> {
    let raw = fs::read_to_string("/proc/meminfo").map_err(|e| Error::Native(e.to_string()))?;
    let total = raw
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemTotal:")
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<u64>().ok())
        })
        .ok_or_else(|| gap("MemTotal unavailable"))?
        * 1024;
    let mut ram = total;
    let mut limited = false;
    let mut dirs = Vec::new();
    if let Ok(raw) = fs::read_to_string("/proc/self/cgroup") {
        for line in raw.lines() {
            let parts = line.splitn(3, ':').collect::<Vec<_>>();
            if parts.len() != 3 {
                continue;
            }
            let root = if parts[1].is_empty() {
                "/sys/fs/cgroup"
            } else if parts[1].split(',').any(|s| s == "memory") {
                "/sys/fs/cgroup/memory"
            } else {
                continue;
            };
            let mut p = PathBuf::from(root);
            dirs.push(p.clone());
            for part in parts[2].split('/').filter(|s| !s.is_empty()) {
                p.push(part);
                dirs.push(p.clone());
            }
        }
    }
    if dirs.is_empty() {
        dirs.extend([
            PathBuf::from("/sys/fs/cgroup"),
            PathBuf::from("/sys/fs/cgroup/memory"),
        ]);
    }
    for dir in dirs {
        for name in ["memory.max", "memory.limit_in_bytes"] {
            if let Ok(n) = fs::read_to_string(dir.join(name))
                .unwrap_or_default()
                .trim()
                .parse::<u64>()
            {
                if n > 0 && n < total {
                    ram = ram.min(n);
                    limited = true;
                }
                break;
            }
        }
    }
    let mut swap = 0_u64;
    if !limited && let Ok(raw) = fs::read_to_string("/proc/swaps") {
        for line in raw.lines().skip(1) {
            let p = line.split_whitespace().collect::<Vec<_>>();
            if p.len() == 5
                && !PathBuf::from(p[0])
                    .canonicalize()
                    .unwrap_or_else(|_| PathBuf::from(p[0]))
                    .file_name()
                    .is_some_and(|s| s.to_string_lossy().starts_with("zram"))
            {
                swap += p[2].parse::<u64>().unwrap_or(0) * 1024;
            }
        }
    }
    let ram = ram as f64;
    Ok((ram * 0.4).max(
        (ram * 0.9)
            .min(ram - 4. * 1024_f64.powi(3))
            .min(ram + swap as f64 - 16. * 1024_f64.powi(3)),
    ))
}
pub struct StaticOffload {
    pub parameters: PreparedParameters,
    pub modules: Vec<OffloadModule>,
    pub placement: StaticPlacement,
    pins: BTreeMap<String, Tensor>,
    native: Kitchen,
    pin_limit: f64,
    pub pinned_bytes: u64,
}
impl StaticOffload {
    pub fn new(parameters: PreparedParameters, native: Kitchen) -> Result<Self, Error> {
        let modules = bf16_dit_modules()?;
        let names = modules
            .iter()
            .flat_map(|m| m.parameters.iter().cloned())
            .collect::<BTreeSet<_>>();
        if names != parameters.keys().cloned().collect() {
            return Err(gap(
                "alternate constructor/metadata outside selected BF16 static binding",
            ));
        }
        for (n, v) in &parameters.values {
            let t = v.as_tensor()?;
            if t.kind() != Kind::BFloat16 || t.device() != Device::Cpu {
                return Err(gap(format!("Expected CPU BF16 weight {n}")));
            }
        }
        Ok(Self {
            parameters,
            modules,
            placement: StaticPlacement::default(),
            pins: BTreeMap::new(),
            pin_limit: host_pin_limit()?,
            pinned_bytes: 0,
            native,
        })
    }
    pub fn free_memory(&self) -> Result<f64, Error> {
        let [free, _, active, reserved] = self.native.cuda_memory(0)?;
        Ok((free + reserved - active) as f64)
    }
    pub fn apply(&mut self, next: StaticPlacement) -> Result<(), Error> {
        // Source load_completely.sort(reverse=True): size/name descending;
        // here offload_bytes==module_bytes for the selected plain BF16 model.
        for m in sorted(&self.modules, true) {
            let target = if next.resident.contains(&m.name) {
                Device::Cuda(0)
            } else {
                Device::Cpu
            };
            for name in &m.parameters {
                if target == Device::Cuda(0)
                    && let Some(t) = self.pins.remove(name)
                    && self.native.host_registration(&t, false)?
                {
                    self.pinned_bytes -= (t.numel() * t.kind().elt_size_in_bytes()) as u64;
                }
                let t = self.parameters[name].as_tensor()?;
                if t.device() != target {
                    self.parameters
                        .values
                        .insert(name.clone(), Value::tensor(t.f_to_device(target)?));
                }
            }
            if target == Device::Cuda(0) && !self.placement.resident.contains(&m.name) {
                self.native.synchronize_device(0)?;
            }
        }
        // Original offloaded traversal follows descending _load_list order.
        for m in sorted(&self.modules, true) {
            if !next.resident.contains(&m.name)
                && (m.cast_weights || self.placement.resident.contains(&m.name))
            {
                for name in &m.parameters {
                    if self.pins.contains_key(name) {
                        continue;
                    }
                    let t = self.parameters[name].as_tensor()?;
                    let bytes = (t.numel() * t.kind().elt_size_in_bytes()) as u64;
                    if self.pinned_bytes as f64 + bytes as f64 <= self.pin_limit
                        && self.native.host_registration(&t, true)?
                    {
                        self.pinned_bytes += bytes;
                        self.pins.insert(name.clone(), t);
                    }
                }
            }
        }
        self.placement = next;
        Ok(())
    }
    /// The original removes same-model/clone registrations with detach(false)
    /// before free_memory. In this isolated single-model owner there is then
    /// no registered model to evict; only the original allocator-cache check.
    pub fn load_for(&mut self, minimum: Option<f64>, requested: f64) -> Result<(), Error> {
        let extra = minimum_inference_memory().max(requested + LINUX_RESERVED);
        let minimum = minimum.or(Some(extra - LINUX_RESERVED));
        {
            let [free, _, active, reserved] = self.native.cuda_memory(0)?;
            if (reserved - active) as f64 > (free + reserved - active) as f64 * 0.25 {
                self.native.cuda_soft_empty_cache(0)?;
            }
        }
        let loaded = self.placement.loaded_bytes as f64;
        let budget = static_load_budget(self.free_memory()?, loaded, minimum);
        let next = self.placement.reload(&self.modules, budget);
        self.apply(next)
    }
}
impl Drop for StaticOffload {
    fn drop(&mut self) {
        let _ = self.native.synchronize_device(0);
        for t in self.pins.values() {
            let _ = self.native.host_registration(t, false);
        }
    }
}
