//! Frozen model_management coordinator policy, independent of model families.
//! Values are supplied by the native environment/argument bindings; this module
//! neither probes services nor changes the reference's selected configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VramState {
    Disabled,
    NoVram,
    LowVram,
    NormalVram,
    HighVram,
    Shared,
}
#[derive(Clone, Copy, Debug)]
pub struct MemoryPolicy {
    pub vram_state: VramState,
    pub lowvram_available: bool,
    pub disable_smart_memory: bool,
    pub reserved: f64,
    pub minimum_weight_ratio: f64,
}
impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            vram_state: VramState::NormalVram,
            lowvram_available: true,
            disable_smart_memory: false,
            reserved: (400 * 1024 * 1024) as f64,
            minimum_weight_ratio: 0.,
        }
    }
}
// Python min/max retain the first argument on ties or unordered comparisons.
fn max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}
fn min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}
impl MemoryPolicy {
    pub fn minimum_inference_memory(self) -> f64 {
        (1024_u64.pow(3) as f64) * 0.8 + self.reserved
    }
    pub fn requirements(self, memory: f64, minimum: Option<f64>) -> (f64, f64) {
        let inference = self.minimum_inference_memory();
        let extra = max(inference, memory + self.reserved);
        (
            extra,
            minimum.map_or(extra, |value| max(inference, value + self.reserved)),
        )
    }
    pub fn uses_lowvram_budget(self, cpu: bool, force_full: bool) -> bool {
        !cpu && self.lowvram_available
            && matches!(self.vram_state, VramState::LowVram | VramState::NormalVram)
            && !force_full
    }
    /// Called only at the original low-VRAM branch after its free-memory probe.
    pub fn lowvram_budget(self, free: f64, loaded: f64, minimum: f64) -> f64 {
        let current = free + loaded;
        let budget = max(
            max(0., current - minimum),
            min(
                current * self.minimum_weight_ratio,
                current - self.minimum_inference_memory(),
            ),
        ) - loaded;
        if budget == 0. { 0.1 } else { budget }
    }
}
