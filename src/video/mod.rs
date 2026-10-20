//! Model-neutral video authoring and explicit local project/library storage.
//! No inference or queue admission.
pub mod contract;
pub mod editing;
pub mod planning;
pub mod project;

mod bundle;
pub mod library;
pub mod persistence;

pub mod http;
pub mod job;
