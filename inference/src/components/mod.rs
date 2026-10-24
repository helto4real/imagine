//! Shared component preparation, ownership and synchronous memory mechanics.
//! Model adapters supply schemas, inventories, budgets and lifecycle ordering.
mod checkpoint;
mod inventory;
pub mod offload;
mod ownership;
pub use checkpoint::*;
pub use inventory::*;
pub use offload::*;
pub use ownership::*;
