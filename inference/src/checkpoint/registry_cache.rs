//! functools.lru_cache(maxsize=1) for a no-argument registry factory. Release
//! native borrows before calling the factory so recursion/clear are observable.
use crate::{native::Error, patching::options::Options};
use std::{cell::RefCell, rc::Rc};
#[derive(Default)]
struct State {
    value: Option<Options>,
    hits: usize,
    misses: usize,
}
#[derive(Clone, Default)]
pub struct RegistryCache(Rc<RefCell<State>>);
impl RegistryCache {
    pub fn get_or_build(
        &self,
        build: impl FnOnce() -> Result<Options, Error>,
    ) -> Result<Options, Error> {
        {
            let mut state = self.0.borrow_mut();
            if let Some(value) = state.value.clone() {
                state.hits += 1;
                return Ok(value);
            }
            state.misses += 1;
        }
        let result = build()?;
        let mut state = self.0.borrow_mut();
        // A recursive call may already have cached this same argument key.
        // Source returns the outer result while retaining that inner cache.
        if state.value.is_none() {
            state.value = Some(result.clone());
        }
        Ok(result)
    }
    pub fn clear(&self) {
        let old = std::mem::take(&mut *self.0.borrow_mut());
        drop(old);
    }
    /// hits, misses, maxsize, currsize; matches the source cache_info tuple.
    pub fn info(&self) -> [usize; 4] {
        let state = self.0.borrow();
        [
            state.hits,
            state.misses,
            1,
            usize::from(state.value.is_some()),
        ]
    }
}
