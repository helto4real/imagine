//! Frozen, CPU-only montage. Media descriptors stay private; public plans contain identities.
pub mod dsp;
mod freeze;
pub(crate) mod media;
mod processor;
pub(crate) mod publication;
mod resize;
pub use freeze::{Frozen, freeze};
pub use imagine_video_contract::assembly::*;
pub use processor::{
    PreparedMedia, fit_frame as fit_rgb_frame, prepare_media, quantize_rgb_frame, render,
};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
#[derive(Clone, Default)]
pub struct Progress(pub Arc<Mutex<State>>);
#[derive(Clone, Default, serde::Serialize)]
pub struct State {
    pub phase: String,
    pub completed: u64,
    pub total: u64,
}
impl Progress {
    pub fn set(&self, phase: &str, completed: u64, total: u64) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = State {
            phase: phase.into(),
            completed,
            total,
        };
    }
    pub fn read(&self) -> State {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}
pub struct Entry {
    pub frozen: Frozen,
    pub progress: Progress,
    pub job: Option<String>,
    pub rendered: Rendered,
}
/// Completed compressed output retained only until publication succeeds or the session ends.
#[derive(Clone, Default)]
pub struct Rendered(Arc<Mutex<Option<Arc<Vec<u8>>>>>);
impl Rendered {
    pub fn get(&self) -> Option<Arc<Vec<u8>>> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub fn set(&self, bytes: Arc<Vec<u8>>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(bytes);
    }
    pub fn clear(&self) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = None;
    }
}
pub fn cancelled(flag: &AtomicBool) -> bool {
    flag.load(std::sync::atomic::Ordering::Acquire)
}
