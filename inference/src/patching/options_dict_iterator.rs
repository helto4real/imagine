//! Shared original dictionary iterator state for string and general-key tables.
use super::Options;
use crate::{execution_value::Value, native::Error};
use std::{cell::RefCell, rc::Rc};

#[derive(Clone, Copy)]
pub(super) enum IteratorKind {
    Items,
    Keys,
    Values,
}
pub(super) trait DictionaryEntries: Clone {
    fn used(&self) -> usize;
    fn next_position(&self, position: usize) -> Option<usize>;
    fn key_at(&self, position: usize) -> Options;
    fn value_at(&self, position: usize) -> Options;
}
struct IteratorState<D> {
    dictionary: Option<D>,
    expected_used: usize,
    failed_size: bool,
    position: usize,
    remaining: usize,
    cached: Option<Rc<Vec<Options>>>,
    kind: IteratorKind,
}
pub(super) struct SourceDictionaryIterator<D> {
    state: RefCell<IteratorState<D>>,
}
impl<D: DictionaryEntries> SourceDictionaryIterator<D> {
    pub(super) fn new(dictionary: D, kind: IteratorKind) -> Self {
        let used = dictionary.used();
        Self {
            state: RefCell::new(IteratorState {
                dictionary: Some(dictionary),
                expected_used: used,
                failed_size: false,
                position: 0,
                remaining: used,
                cached: matches!(kind, IteratorKind::Items)
                    .then(|| Rc::new(vec![Options::Atom(Value::None), Options::Atom(Value::None)])),
                kind,
            }),
        }
    }
    pub(super) fn length_hint(&self) -> usize {
        let state = self.state.borrow();
        state
            .dictionary
            .as_ref()
            .filter(|d| !state.failed_size && d.used() == state.expected_used)
            .map_or(0, |_| state.remaining)
    }
    pub(super) fn next(&self) -> Result<Option<Options>, Error> {
        let mut state = self.state.borrow_mut();
        let Some(owner) = state.dictionary.clone() else {
            return Ok(None);
        };
        if state.failed_size || owner.used() != state.expected_used {
            state.failed_size = true;
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: "dictionary changed size during iteration".into(),
            });
        }
        let position = owner.next_position(state.position);
        if position.is_none() || state.remaining == 0 {
            let old = state.dictionary.take();
            drop(state);
            drop(owner);
            drop(old);
            return if position.is_some() {
                Err(Error::ReferenceException {
                    class: "RuntimeError",
                    message: "dictionary keys changed during iteration".into(),
                })
            } else {
                Ok(None)
            };
        }
        let position = position.unwrap();
        let value = (!matches!(state.kind, IteratorKind::Keys)).then(|| owner.value_at(position));
        let key = (!matches!(state.kind, IteratorKind::Values)).then(|| owner.key_at(position));
        state.position = position + 1;
        state.remaining -= 1;
        let (result, old) = if matches!(state.kind, IteratorKind::Items) {
            let pair = vec![key.unwrap(), value.unwrap()];
            let cached = state.cached.as_mut().unwrap();
            if let Some(value) = Rc::get_mut(cached) {
                let old = std::mem::replace(value, pair);
                (Options::Tuple(cached.clone()), Some(old))
            } else {
                (Options::Tuple(Rc::new(pair)), None)
            }
        } else if matches!(state.kind, IteratorKind::Values) {
            (value.unwrap(), None)
        } else {
            (key.unwrap(), None)
        };
        // Increment the emitted tuple before releasing previous entries. Reentry
        // from any released value must see an unlocked iterator/dictionary.
        drop(state);
        drop(owner);
        drop(old);
        Ok(Some(result))
    }
}
