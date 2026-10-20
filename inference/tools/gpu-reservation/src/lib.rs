//! Development-only reservation controller, sharing the actual product protocol.
#[path = "../../../../src/coordination.rs"]
pub mod coordination;
#[path = "../../gpu_reserved_job.rs"]
mod gpu_reserved_job;
pub use gpu_reserved_job::{Report, Request, run, validate_gpu_worker};

mod bwrap_job;
pub use bwrap_job::{Admission, NamespaceReport, run_bwrap};
