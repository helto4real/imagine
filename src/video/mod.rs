//! Model-neutral authoring, frozen scene admission and explicit project/library storage.
pub mod contract;
pub mod editing;
pub mod planning;
pub mod project;
pub(crate) mod rgb_resize;
pub(crate) mod text;
mod text_unicode;

pub(crate) mod bundle;
pub mod library;
pub mod persistence;

pub mod http;
pub mod job;

pub mod image;

pub mod output;

pub mod adapters;
pub mod api;
pub mod artifacts;
pub mod assembly;
pub mod authoring;
pub mod cast;
pub mod execution;
pub mod media;
pub(crate) mod reference_inputs;
pub mod reference_video;
pub mod review;
pub mod runs;
pub mod service;
pub mod takes;
