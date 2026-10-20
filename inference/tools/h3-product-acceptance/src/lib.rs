//! CPU development supervisor only. The admitted GPU child is the real worker.
#[path = "../../../../src/coordination.rs"]
pub mod coordination;
#[path = "../../../src/h3_product/deployment.rs"]
pub mod deployment;
#[path = "../../../../shared/h3_protocol.rs"]
pub mod protocol;
