//! Native callbacks at the frozen Triton module-load and launch boundaries.
use crate::{
    native::Error,
    triton_file_cache::{CacheText, JsonPath},
};
use std::sync::{Arc, RwLock};
pub struct LoadMetadata<'a> {
    pub module: Option<u64>,
    pub function: Option<u64>,
    pub name: &'a str,
    pub metadata_group: &'a [(CacheText, JsonPath)],
    pub digest: &'a str,
}
#[derive(Clone, Debug)]
pub struct LaunchMetadata {
    pub name: String,
    pub function: u64,
    pub stream: u64,
}
pub type LoadHook = Arc<dyn Fn(&LoadMetadata<'_>) -> Result<(), Error> + Send + Sync>;
pub type LaunchHook = Arc<dyn Fn(Option<&LaunchMetadata>) -> Result<(), Error> + Send + Sync>;
#[derive(Clone, Default)]
pub struct Hooks {
    pub load_start: Option<LoadHook>,
    pub load_end: Option<LoadHook>,
    pub launch_enter: Option<LaunchHook>,
    pub launch_exit: Option<LaunchHook>,
}
#[derive(Default)]
pub struct HookState(RwLock<Hooks>);
impl HookState {
    pub fn replace(&self, hooks: Hooks) {
        *self.0.write().unwrap() = hooks;
    }
    pub(crate) fn snapshot(&self) -> Hooks {
        self.0.read().unwrap().clone()
    }
}
