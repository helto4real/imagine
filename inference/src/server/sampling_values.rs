//! Numeric domain shared by the worker admission and its CPU-only regression.
pub fn valid(steps: i64, cfg: f64) -> bool {
    steps > 0 && cfg.is_finite()
}
