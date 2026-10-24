//! Compatibility exports; generic ownership belongs to components.
pub use crate::components::{HostPins, StaticMemoryServices};
pub type StaticModel<'a, B = super::FluxSampling> = crate::components::StaticModel<'a, B>;
