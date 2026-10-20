//! One-request H3 product worker boundary. The frozen replay/core remain controls.
pub mod deployment;
mod native;
#[path = "../../../shared/h3_protocol.rs"]
pub mod protocol;
pub mod wire;
pub use native::run;
