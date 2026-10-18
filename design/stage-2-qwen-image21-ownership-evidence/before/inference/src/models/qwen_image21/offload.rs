//! Frozen static ModelPatcher, synchronous BF16/INT8 weights and optional LoRA patches.
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
/// Original estimate_memory + QwenImage.extra_conds_shapes. Each entry is one
/// conditioning's ordered reference tensor shapes, including positive/negative.
/// CFG=1's later denoising shortcut does not remove them from this estimate.
pub fn qwen_sampling_memory_with_references(
    noise_shape: &[i64],
    conditions: &[Vec<Vec<i64>>],
) -> (f64, f64) {
    let area = noise_shape[0] * noise_shape[2..].iter().product::<i64>();
    let reference_areas = conditions
        .iter()
        .map(|refs| refs.iter().map(|s| s.iter().product::<i64>()).sum::<i64>() / 64)
        .collect::<Vec<_>>();
    let bytes = |a: i64| (a as f64 * 2. * 0.01 * 6.) * (1024 * 1024) as f64;
    (
        bytes(area * 2 + reference_areas.iter().sum::<i64>()),
        bytes(area + reference_areas.iter().copied().max().unwrap_or(0)),
    )
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
    // Original recomputed accounting can become negative after a subsequent
    // partial unload. Physical residency and this counter are distinct state.
    pub loaded_bytes: i64,
    pub buffer_bytes: u64,
    /// Low-vram mode and ordered patch passes, including still-resident weights.
    #[serde(default)]
    pub lowvram_passes: BTreeMap<String, usize>,
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
/// Original load_completely.sort(reverse=True), then separate patch/move loops.
pub enum RegularLoadAction<'a> {
    Patch(&'a OffloadModule),
    Move(&'a OffloadModule),
}
pub fn regular_load_actions<'a>(
    modules: &'a [OffloadModule],
    resident: &BTreeSet<String>,
) -> Vec<RegularLoadAction<'a>> {
    let mut selected = modules
        .iter()
        .filter(|m| resident.contains(&m.name))
        .collect::<Vec<_>>();
    selected.sort_by(|a, b| (b.module_bytes, &b.name).cmp(&(a.module_bytes, &a.name)));
    selected
        .iter()
        .map(|m| RegularLoadAction::Patch(m))
        .chain(selected.iter().map(|m| RegularLoadAction::Move(m)))
        .collect()
}
impl StaticPlacement {
    pub fn reload(&self, modules: &[OffloadModule], extra: f64) -> Self {
        if extra < 0. {
            let mut next = self.clone();
            next.unload(modules, -extra);
            return next;
        }
        let total: u64 = modules.iter().map(|m| m.module_bytes).sum();
        if self.loaded_bytes == total as i64 {
            return self.clone();
        }
        let budget = self.loaded_bytes as f64 + extra;
        let mut next = Self::load(modules, budget, budget > total as f64);
        // Original load() does not move already-patched CUDA parameters back
        // to CPU merely because a new fit calculation skips their module.
        // Preserve those objects independently of its recomputed byte counter.
        next.resident.extend(self.resident.iter().cloned());
        for (name, passes) in &mut next.lowvram_passes {
            if let Some(previous) = self.lowvram_passes.get(name) {
                // load() skips an already-lowvram module, retaining functions.
                *passes = *previous;
            }
        }
        next
    }
    /// Original NUM_STREAMS=0 branch. Existing resident entries are patched only
    /// once; callers perform original partial-unload before extending a load.
    pub fn load(modules: &[OffloadModule], budget: f64, full: bool) -> Self {
        let mut out = Self::default();
        for m in sorted(modules, true) {
            let potential = out.buffer_bytes.max(m.offload_bytes);
            let fits =
                ((out.loaded_bytes + m.module_bytes as i64 + potential as i64) as f64) < budget;
            if !full && m.cast_weights && !fits {
                out.buffer_bytes = potential;
                out.lowvram_passes.insert(m.name.clone(), 1);
            } else if full || fits {
                out.resident.insert(m.name.clone());
                out.loaded_bytes += m.module_bytes as i64;
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
                if m.cast_weights {
                    *self.lowvram_passes.entry(m.name.clone()).or_default() += 1;
                }
            }
        }
        self.loaded_bytes -= freed as i64;
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
fn host_cgroup_dirs() -> Vec<PathBuf> {
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
    dirs
}
/// Original available aggregation over observed cgroups, excluding inactive files.
pub fn host_available_from_cgroups(
    total: u64,
    available: u64,
    groups: &[(i128, Option<(i128, i128)>)],
) -> u64 {
    let mut available = available as i128;
    for &(limit, usage) in groups {
        if limit > 0 && limit < (total as i128) {
            let used = usage
                .map(|(bytes, inactive)| (bytes - inactive).max(0))
                .unwrap_or(0);
            available = available.min(limit - used);
        }
    }
    available.max(0) as u64
}
/// Original graph-free ensure_pin_budget with no eligible inactive owners.
pub fn single_owner_pin_budget(size: usize, available: u64) -> bool {
    size as u128 + 2048 * 1024 * 1024 <= available as u128
}
/// Selected Linux system_memory.virtual_memory_available branch: MemAvailable
/// and the current process's complete cgroup lineage, no process inventory.
pub fn host_available_memory() -> Result<u64, Error> {
    let raw = fs::read_to_string("/proc/meminfo").map_err(|e| Error::Native(e.to_string()))?;
    let mem = raw
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((
                name,
                value.split_whitespace().next()?.parse::<u64>().ok()? * 1024,
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let total = *mem
        .get("MemTotal")
        .ok_or_else(|| gap("MemTotal unavailable"))?;
    let available = *mem
        .get("MemAvailable")
        .ok_or_else(|| gap("Legacy Linux psutil MemAvailable fallback"))?;
    let mut groups = Vec::new();
    let read = |path: PathBuf| fs::read_to_string(path).ok()?.trim().parse::<i128>().ok();
    for dir in host_cgroup_dirs() {
        let limit = ["memory.max", "memory.limit_in_bytes"]
            .iter()
            .find_map(|n| read(dir.join(n)).filter(|n| *n > 0));
        if let Some(limit) = limit.filter(|n| *n > 0 && *n < total as i128) {
            let usage = read(dir.join("memory.current"))
                .map(|n| (n, "inactive_file"))
                .or_else(|| {
                    read(dir.join("memory.usage_in_bytes")).map(|n| (n, "total_inactive_file"))
                });
            let usage = usage.map(|(usage, key)| {
                let stat = fs::read_to_string(dir.join("memory.stat")).unwrap_or_default();
                let inactive = stat
                    .lines()
                    .find_map(|l| {
                        let p = l.split_whitespace().collect::<Vec<_>>();
                        if p.len() == 2 && p[0] == key {
                            p[1].parse::<i128>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                (usage, inactive)
            });
            groups.push((limit, usage));
        }
    }
    Ok(host_available_from_cgroups(total, available, &groups))
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
    let dirs = host_cgroup_dirs();
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
/// Original _load_list/low_vram_patch_estimate_vram extra math workspace.
pub fn patched_module_offload_bytes(base: u64, patched_elements: &[u64], kind: Kind) -> u64 {
    base + patched_elements.iter().sum::<u64>() * kind.elt_size_in_bytes() as u64 * 2
}
fn parameter_shape_kind(value: &Value) -> Result<(Vec<i64>, Kind, bool), Error> {
    match value {
        Value::Quantized(_) => {
            let q = value.as_quantized()?;
            Ok((q.params.original_shape, q.params.original_kind, true))
        }
        _ => {
            let t = value.as_tensor()?;
            Ok((t.size(), t.kind(), false))
        }
    }
}
fn parameter_to_device(native: &Kitchen, value: &Value, device: Device) -> Result<Value, Error> {
    match value {
        Value::Quantized(_) => Ok(Value::quantized(
            value
                .as_quantized()?
                .to(native, Some(device), None, false)?
                .as_ref()
                .shallow_parts(),
        )),
        _ => Ok(Value::tensor(value.as_tensor()?.f_to_device(device)?)),
    }
}
// Module._apply re-wraps moved ordinary Parameters with their previous grad
// flag even inside no_grad. Backup tensor.to intentionally has other semantics.
fn move_parameter(native: &Kitchen, value: &Value, device: Device) -> Result<Value, Error> {
    let moved = parameter_to_device(native, value, device)?;
    if let Value::Tensor(_) = value {
        let original = value.as_tensor()?;
        return Ok(Value::tensor(
            moved
                .as_tensor()?
                .f_set_requires_grad(original.requires_grad())?,
        ));
    }
    Ok(moved)
}
/// Original module_size(state_dict) includes serialized scales and comfy_quant
/// JSON, while pinning uses QuantizedTensor.nbytes (packed payload only).
/// Templates retain the source module traversal and cast_weights attributes.
pub fn prepare_offload_modules(
    parameters: &PreparedParameters,
    mut modules: Vec<OffloadModule>,
) -> Result<Vec<OffloadModule>, Error> {
    use crate::storage::ScalarField;
    let names = modules
        .iter()
        .flat_map(|m| m.parameters.iter().cloned())
        .collect::<BTreeSet<_>>();
    if names != parameters.keys().cloned().collect() {
        return Err(gap(
            "alternate constructor/metadata outside selected static binding",
        ));
    }
    for module in &mut modules {
        module.module_bytes = 0;
        let mut extra = 0;
        if parameters
            .linear_policies
            .get(&module.name)
            .is_some_and(|p| p.pre_quant_scale.is_some() || p.input_scale.is_some())
        {
            return Err(gap("Static offload extra registered scale parameters"));
        }
        for name in &module.parameters {
            let value = &parameters[name];
            let (shape, kind, quantized) = parameter_shape_kind(value)?;
            if kind != Kind::BFloat16 {
                return Err(gap(format!("Expected BF16 logical weight {name}")));
            }
            if quantized {
                let q = value.as_quantized()?;
                if q.layout != "TensorWiseINT8Layout" || q.payload.device() != Device::Cpu {
                    return Err(gap(format!("Expected CPU tensorwise INT8 weight {name}")));
                }
                let mut conf = serde_json::json!({"format":"int8_tensorwise"});
                if parameters
                    .linear_policies
                    .get(&module.name)
                    .is_some_and(|p| p.full_precision_config)
                {
                    conf["full_precision_matrix_mult"] = true.into();
                }
                if q.params.fields.get("convrot") == Some(&ScalarField::Bool(true)) {
                    conf["convrot"] = true.into();
                    conf["convrot_groupsize"] = match q.params.fields.get("convrot_groupsize") {
                        Some(ScalarField::Integer(n)) => (*n).into(),
                        _ => 256.into(),
                    };
                }
                // Python json.dumps adds one space after every colon/comma.
                let json_bytes = serde_json::to_vec(&conf)
                    .map_err(|e| Error::Native(e.to_string()))?
                    .len()
                    + 2 * conf.as_object().unwrap().len()
                    - 1;
                module.module_bytes += json_bytes as u64;
                for (_, t) in crate::layouts::tensorwise_state_dict(&q, "weight")? {
                    module.module_bytes += (t.numel() * t.kind().elt_size_in_bytes()) as u64;
                }
            } else {
                let t = value.as_tensor()?;
                if t.device() != Device::Cpu {
                    return Err(gap(format!("Expected CPU weight {name}")));
                }
                module.module_bytes += (t.numel() * t.kind().elt_size_in_bytes()) as u64;
            }
            if module.cast_weights && (quantized || kind != Kind::BFloat16) {
                extra += shape.iter().product::<i64>() as u64 * 2;
            }
        }
        module.offload_bytes = module.module_bytes + extra;
    }
    Ok(modules)
}
/// Original _load_list patch workspace replaces the unpatched cast workspace.
pub fn prepare_patched_offload_modules(
    parameters: &PreparedParameters,
    owner: &crate::patching::PatchOwner,
    mut modules: Vec<OffloadModule>,
) -> Result<Vec<OffloadModule>, Error> {
    for module in &mut modules {
        if module.cast_weights {
            module.offload_bytes = module.module_bytes;
            for name in &module.parameters {
                let value = &parameters[name];
                let (shape, kind, quantized) = parameter_shape_kind(value)?;
                let elements = shape.iter().product::<i64>() as u64;
                if let Some(patches) = owner.patches.get(name) {
                    if crate::patching::calculate_shape_from_shape(patches, shape.clone())? != shape
                    {
                        return Err(gap("Static offload forced shape-changing patch lifecycle"));
                    }
                    module.offload_bytes = patched_module_offload_bytes(
                        module.offload_bytes,
                        &[elements],
                        Kind::BFloat16,
                    );
                } else if quantized || kind != Kind::BFloat16 {
                    module.offload_bytes += elements * 2;
                }
            }
        }
    }
    Ok(modules)
}
pub struct StaticOffload {
    pub parameters: PreparedParameters,
    pub patch_owner: crate::patching::PatchOwner,
    pub modules: Vec<OffloadModule>,
    pub placement: StaticPlacement,
    pub sampling: super::FluxSampling,
    /// Original ModelPatcher caches module_size before patch/load transitions.
    model_size: u64,
    pub model_lowvram: bool,
    pins: BTreeMap<String, Tensor>,
    native: Kitchen,
    pin_limit: f64,
    pub pinned_bytes: u64,
}
impl StaticOffload {
    pub fn new(parameters: PreparedParameters, native: Kitchen) -> Result<Self, Error> {
        Self::from_modules(
            parameters,
            native,
            bf16_dit_modules()?,
            super::FluxSampling::qwen21()?,
        )
    }
    /// Explicit source module inventory, also used by bounded lifecycle oracles.
    pub fn from_modules(
        parameters: PreparedParameters,
        native: Kitchen,
        modules: Vec<OffloadModule>,
        sampling: super::FluxSampling,
    ) -> Result<Self, Error> {
        let modules = prepare_offload_modules(&parameters, modules)?;
        let model_size =
            modules.iter().map(|m| m.module_bytes).sum::<u64>() + sampling.buffer_bytes();
        Ok(Self {
            parameters,
            patch_owner: crate::patching::PatchOwner::new(),
            modules,
            placement: StaticPlacement::default(),
            sampling,
            model_size,
            model_lowvram: false,
            pins: BTreeMap::new(),
            pin_limit: host_pin_limit()?,
            pinned_bytes: 0,
            native,
        })
    }
    pub fn model_size(&self) -> u64 {
        self.model_size
    }
    /// Original partially_load on the selected single owner, without hooks/clones.
    pub fn partially_load(&mut self, extra_memory: f64) -> Result<(), Error> {
        self.partially_load_impl(extra_memory, None)
    }
    fn partially_load_impl(
        &mut self,
        extra: f64,
        rt: Option<&crate::execution::Runtime>,
    ) -> Result<(), Error> {
        if extra < 0. {
            let mut next = self.placement.clone();
            next.unload(&self.modules, -extra);
            self.apply_impl(next, rt)?;
            self.model_lowvram = true;
            // partially_unload moves selected leaf modules only, not buffers.
            return Ok(());
        }
        if !self.model_lowvram && self.placement.loaded_bytes > 0 {
            return Ok(());
        }
        let budget = self.placement.loaded_bytes as f64 + extra;
        let full = budget > self.model_size as f64;
        let mut next = StaticPlacement::load(&self.modules, budget, full);
        next.resident
            .extend(self.placement.resident.iter().cloned());
        for (name, passes) in &mut next.lowvram_passes {
            if let Some(previous) = self.placement.lowvram_passes.get(name) {
                *passes = *previous;
            }
        }
        let lowvram = !next.lowvram_passes.is_empty();
        let result = (|| {
            self.apply_impl(next, rt)?;
            if full && !lowvram {
                self.sampling.to_device(Device::Cuda(0))?;
                self.placement.loaded_bytes = self.model_size as i64;
            }
            self.model_lowvram = lowvram;
            Ok(())
        })();
        if result.is_err() {
            let _ = self.unload();
        }
        result
    }
    /// Original unpatch_model(offload_device, unpatch_weights=True).
    pub fn unload(&mut self) -> Result<(), Error> {
        for (_, t) in std::mem::take(&mut self.pins) {
            self.native.host_registration(&t, false)?;
        }
        self.pinned_bytes = 0;
        self.patch_owner.restore(&mut self.parameters)?;
        for value in self.parameters.values.values_mut() {
            *value = move_parameter(&self.native, value, Device::Cpu)?;
        }
        self.sampling.to_device(Device::Cpu)?;
        self.placement = StaticPlacement::default();
        self.model_lowvram = false;
        Ok(())
    }
    /// Explicit diagnostic replay must carry buffer placement and low-VRAM state.
    pub fn replay_state(
        &mut self,
        rt: &crate::execution::Runtime,
        placement: StaticPlacement,
        sampling_device: Device,
        lowvram: bool,
    ) -> Result<(), Error> {
        self.apply_impl(placement, Some(rt))?;
        self.sampling.to_device(sampling_device)?;
        self.model_lowvram = lowvram;
        Ok(())
    }
    pub fn with_patches(mut self, owner: crate::patching::PatchOwner) -> Result<Self, Error> {
        self.modules = prepare_patched_offload_modules(
            &self.parameters,
            &owner,
            std::mem::take(&mut self.modules),
        )?;
        self.patch_owner = owner;
        Ok(self)
    }
    pub fn apply_patched(
        &mut self,
        rt: &crate::execution::Runtime,
        next: StaticPlacement,
    ) -> Result<(), Error> {
        self.apply_impl(next, Some(rt))
    }
    pub fn free_memory(&self) -> Result<f64, Error> {
        let [free, _, active, reserved] = self.native.cuda_memory(0)?;
        Ok((free + reserved - active) as f64)
    }
    pub fn apply(&mut self, next: StaticPlacement) -> Result<(), Error> {
        self.apply_impl(next, None)
    }
    fn apply_impl(
        &mut self,
        next: StaticPlacement,
        rt: Option<&crate::execution::Runtime>,
    ) -> Result<(), Error> {
        if !self.patch_owner.patches.is_empty() && rt.is_none() {
            return Err(gap("Patched static owner requires its original runtime"));
        }
        // partial_unload walks ascending _load_list order, restores originals,
        // moves a complete module, then pins it before considering the next one.
        let removed = sorted(&self.modules, false)
            .into_iter()
            .filter(|m| {
                self.placement.resident.contains(&m.name) && !next.resident.contains(&m.name)
            })
            .map(|m| m.parameters.clone())
            .collect::<Vec<_>>();
        for names in removed {
            for name in &names {
                if let Some(base) = self.patch_owner.backup.remove(name) {
                    self.parameters.values.insert(name.clone(), base);
                }
                let moved = move_parameter(&self.native, &self.parameters[name], Device::Cpu)?;
                self.parameters.values.insert(name.clone(), moved);
            }
            for name in &names {
                self.pin_parameter(name)?;
            }
        }
        // load configures lazy functions before patching any resident weights.
        for m in &self.modules {
            for name in &m.parameters {
                self.patch_owner.bind_lowvram(
                    &mut self.parameters,
                    name,
                    next.lowvram_passes.get(&m.name).copied().unwrap_or(0),
                );
            }
        }
        for action in regular_load_actions(&self.modules, &next.resident) {
            match action {
                RegularLoadAction::Patch(m) => {
                    if self.placement.resident.contains(&m.name) {
                        continue;
                    }
                    for name in &m.parameters {
                        if let Some(t) = self.pins.remove(name)
                            && self.native.host_registration(&t, false)?
                        {
                            self.pinned_bytes -= (t.numel() * t.kind().elt_size_in_bytes()) as u64;
                        }
                        if let Some(patches) = self.patch_owner.patches.get(name)
                            && !self.patch_owner.backup.contains_key(name)
                        {
                            let base = self.parameters[name].clone();
                            let patched = unsafe {
                                crate::patching::patch_weight(
                                    rt.unwrap(),
                                    &base,
                                    patches,
                                    &format!("{}{name}", self.patch_owner.key_prefix),
                                    Device::Cuda(0),
                                    Kind::Half,
                                    &mut self.patch_owner.diagnostics,
                                )?
                            };
                            let saved = parameter_to_device(&self.native, &base, Device::Cpu)?;
                            self.patch_owner.backup.insert(name.clone(), saved);
                            self.parameters.values.insert(name.clone(), patched);
                        }
                    }
                    self.native.synchronize_device(0)?;
                }
                RegularLoadAction::Move(m) => {
                    for name in &m.parameters {
                        let moved =
                            move_parameter(&self.native, &self.parameters[name], Device::Cuda(0))?;
                        self.parameters.values.insert(name.clone(), moved);
                    }
                }
            }
        }
        let offloaded = sorted(&self.modules, true)
            .into_iter()
            .filter(|m| {
                !next.resident.contains(&m.name)
                    && (m.cast_weights || self.placement.resident.contains(&m.name))
            })
            .map(|m| m.parameters.clone())
            .collect::<Vec<_>>();
        for names in offloaded {
            for name in &names {
                self.pin_parameter(name)?;
            }
        }
        self.placement = next;
        Ok(())
    }
    fn pin_parameter(&mut self, name: &str) -> Result<(), Error> {
        if self.pins.contains_key(name) {
            return Ok(());
        }
        // QuantizedTensor.nbytes/data_ptr describe only its packed payload.
        let t = match &self.parameters[name] {
            Value::Quantized(_) => self.parameters[name].as_quantized()?.payload,
            value => value.as_tensor()?,
        };
        let bytes = (t.numel() * t.kind().elt_size_in_bytes()) as u64;
        if self.pinned_bytes as f64 + bytes as f64 <= self.pin_limit
            && self.native.host_registration(&t, true)?
        {
            self.pinned_bytes += bytes;
            self.pins.insert(name.to_owned(), t);
        }
        Ok(())
    }
    /// The original removes same-model/clone registrations with detach(false)
    /// before free_memory. In this isolated single-model owner there is then
    /// no registered model to evict; only the original allocator-cache check.
    pub fn load_for(&mut self, minimum: Option<f64>, requested: f64) -> Result<(), Error> {
        self.load_for_impl(minimum, requested, None)
    }
    pub fn load_for_patched(
        &mut self,
        rt: &crate::execution::Runtime,
        minimum: Option<f64>,
        requested: f64,
    ) -> Result<(), Error> {
        self.load_for_impl(minimum, requested, Some(rt))
    }
    fn load_for_impl(
        &mut self,
        minimum: Option<f64>,
        requested: f64,
        rt: Option<&crate::execution::Runtime>,
    ) -> Result<(), Error> {
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
        self.partially_load_impl(budget, rt)
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
