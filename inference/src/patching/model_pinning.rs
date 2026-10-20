//! Source ModelPatcher pin/unpin bodies. Set and memory operations belong to
//! their actual native providers; no ordered-map substitute is assumed here.
use super::{model_patcher::PatcherState, options::Options, options_iterator::SequenceIterator};
use crate::native::Error;
pub trait PinningBindings {
    fn get_key_weight(&self, model: &Options, key: &Options) -> Result<[Options; 3], Error>;
    fn pin_memory(&self, weight: &Options) -> Result<bool, Error>;
    fn unpin_memory(&self, weight: &Options) -> Result<Options, Error>;
    fn contains(&self, set: &Options, key: &Options) -> Result<bool, Error>;
    fn add(&self, set: &Options, key: &Options) -> Result<(), Error>;
    fn remove(&self, set: &Options, key: &Options) -> Result<(), Error>;
    fn list_snapshot(&self, value: &Options) -> Result<Options, Error>;
    fn unpin_weight(&self, state: &PatcherState, key: &Options) -> Result<Options, Error>;
}
pub fn pin_weight_to_device(
    state: &PatcherState,
    key: &Options,
    bindings: &dyn PinningBindings,
) -> Result<(), Error> {
    // Array fields retain weight,set_func,convert_func in source local order.
    let values = bindings.get_key_weight(&state.fields.get("model")?, key)?;
    if bindings.pin_memory(&values[0])? {
        bindings.add(&state.fields.get("pinned")?, key)?;
    }
    Ok(())
}
pub fn unpin_weight(
    state: &PatcherState,
    key: &Options,
    bindings: &dyn PinningBindings,
) -> Result<(), Error> {
    if bindings.contains(&state.fields.get("pinned")?, key)? {
        let values = bindings.get_key_weight(&state.fields.get("model")?, key)?;
        drop(bindings.unpin_memory(&values[0])?);
        bindings.remove(&state.fields.get("pinned")?, key)?;
    }
    Ok(())
}
pub fn unpin_all_weights(
    state: &PatcherState,
    bindings: &dyn PinningBindings,
) -> Result<(), Error> {
    struct Frame {
        iterator: Option<SequenceIterator>,
        key: Option<Options>,
    }
    let mut frame = Frame {
        iterator: Some(SequenceIterator::for_operation(
            &bindings.list_snapshot(&state.fields.get("pinned")?)?,
            "pinned key snapshot",
        )?),
        key: None,
    };
    loop {
        let Some(key) = frame.iterator.as_mut().unwrap().next()? else {
            drop(frame.iterator.take());
            break;
        };
        drop(frame.key.replace(key));
        drop(bindings.unpin_weight(state, frame.key.as_ref().unwrap())?);
    }
    Ok(())
}
