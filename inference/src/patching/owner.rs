use super::*;
use crate::{checkpoint::PreparedParameters, execution::Runtime};
use tch::Device;
/// Frozen static ModelPatcher.patch_weight_to_device, keeping original logical
/// quantized dtype through the pre-dequantization cast and final requantization.
/// # Safety
/// Runtime kernels and source value storage/stream contracts must hold.
pub unsafe fn patch_weight(
    rt: &Runtime,
    base: &Value,
    patches: &[Patch],
    key: &str,
    device: Device,
    compute: Kind,
    d: &mut Diagnostics,
) -> Result<Value, Error> {
    if patches.is_empty() {
        return Ok(base.clone());
    }
    let (mut temp, original_kind) = match base {
        Value::Quantized(_) => {
            let q = base.as_quantized()?;
            let kind = q.params.original_kind;
            let q = q.to(&rt.native, Some(device), Some(compute), true)?;
            let temp = unsafe { rt.decode(q.as_ref())? };
            (temp, kind)
        }
        _ => {
            let t = base.as_tensor()?;
            let kind = t.kind();
            (cast_to_device(&rt.native, &t, device, compute, true)?, kind)
        }
    };
    temp = calculate_weight(&rt.native, patches, temp, key, Kind::Float, None, d)?;
    if let Value::Quantized(_) = base {
        let q = base.as_quantized()?;
        let mut out = unsafe {
            rt.requantize_from_float(
                &q,
                &temp,
                &vec![
                    ("scale".into(), Value::Text("recalculate".into())),
                    (
                        "stochastic_rounding".into(),
                        Value::UInt(string_to_seed(key)),
                    ),
                    ("inplace_ops".into(), Value::Bool(true)),
                ],
            )?
        };
        out.params.original_kind = original_kind;
        Ok(Value::quantized(out))
    } else {
        Ok(Value::tensor(temp.f_to_kind(original_kind)?))
    }
}
pub struct PatchOwner {
    pub patches: PatchMap,
    pub key_prefix: String,
    pub(crate) backup: BTreeMap<String, Value>,
    pub diagnostics: Diagnostics,
}
impl Default for PatchOwner {
    fn default() -> Self {
        Self::new()
    }
}
impl PatchOwner {
    pub fn new() -> Self {
        Self {
            patches: BTreeMap::new(),
            key_prefix: String::new(),
            backup: BTreeMap::new(),
            diagnostics: Diagnostics::default(),
        }
    }
    pub fn add(
        &mut self,
        state: &PreparedParameters,
        loaded: &Loaded,
        strength: f64,
    ) -> Vec<Target> {
        let mut accepted = vec![];
        for (target, value) in &loaded.patches {
            if state.contains_key(&target.key) && !matches!(value, PatchValue::Config(_)) {
                let mut p = Patch::new(value.clone(), strength);
                p.offset = target.offset;
                self.patches.entry(target.key.clone()).or_default().push(p);
                accepted.push(target.clone());
            }
        }
        accepted
    }
    pub fn restore(&mut self, state: &mut PreparedParameters) -> Result<(), Error> {
        for (key, value) in std::mem::take(&mut self.backup) {
            // set_attr_param wraps the backup as Parameter(requires_grad=False).
            let value = match value {
                Value::Tensor(_) => Value::tensor(value.as_tensor()?.f_detach()?),
                _ => value,
            };
            state.values.insert(key, value);
        }
        state.lowvram_patches.clear();
        Ok(())
    }
    pub fn clear(&mut self, state: &mut PreparedParameters) -> Result<(), Error> {
        self.restore(state)?;
        self.patches.clear();
        Ok(())
    }
    /// Original LowVramPatch functions may remain on a resident patched weight,
    /// and partial unload appends rather than replaces an existing function.
    pub fn bind_lowvram(&self, state: &mut PreparedParameters, key: &str, passes: usize) {
        state.lowvram_patches.remove(key);
        if passes > 0
            && let Some(patches) = self.patches.get(key)
        {
            state.lowvram_patches.insert(
                key.to_owned(),
                (0..passes).flat_map(|_| patches.iter().cloned()).collect(),
            );
        }
    }
    /// # Safety
    /// Same runtime/storage/stream requirements as patch_weight.
    pub unsafe fn apply(
        &mut self,
        rt: &Runtime,
        state: &mut PreparedParameters,
        device: Device,
        compute: Kind,
    ) -> Result<(), Error> {
        self.restore(state)?;
        for (key, patches) in &self.patches {
            let base = state
                .get(key)
                .ok_or_else(|| exception("KeyError", key))?
                .clone();
            // Calculate before replacing: failures cannot destroy this original.
            let patched = unsafe {
                patch_weight(
                    rt,
                    &base,
                    patches,
                    &format!("{}{key}", self.key_prefix),
                    device,
                    compute,
                    &mut self.diagnostics,
                )?
            };
            let backup = match &base {
                Value::Quantized(_) => Value::quantized(
                    base.as_quantized()?
                        .to(&rt.native, Some(Device::Cpu), None, false)?
                        .as_ref()
                        .shallow_parts(),
                ),
                _ => {
                    let t = base.as_tensor()?;
                    if t.device() == Device::Cpu {
                        base.clone()
                    } else {
                        Value::tensor(t.f_to_device(Device::Cpu)?)
                    }
                }
            };
            self.backup.insert(key.clone(), backup);
            state.values.insert(key.clone(), patched);
        }
        Ok(())
    }
}
