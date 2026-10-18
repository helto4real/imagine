//! Native seed bits and enablement are independent, as in the Python wrappers.
#[derive(Clone, Copy)]
pub struct NativeSeed {
    pub enabled: bool,
    pub bits: u64,
}
impl From<Option<i64>> for NativeSeed {
    fn from(value: Option<i64>) -> Self {
        Self {
            enabled: value.is_some_and(|v| v > 0),
            bits: value.unwrap_or(0) as u64,
        }
    }
}
