//! Explicit CPU-only transport/deployment tests. No inference/tch/CUDA linkage.
#[path = "../../../src/h3_product/deployment.rs"]
pub mod deployment;
#[path = "../../../../shared/h3_protocol.rs"]
pub mod protocol;
#[path = "../../../src/h3_product/wire.rs"]
pub mod wire;
