//! Ordered plain-string dictionaries with the locked CPython iterator state.
//! No Python runtime, hashing policy replacement or snapshot items iterator.
use super::Options;
use crate::{execution_value::Value, native::Error};
use std::{cell::RefCell, ops::Deref, rc::Rc};

#[derive(Clone, Default)]
pub struct Dictionary {
    entries: Vec<(String, Options)>,
    slots: Vec<Option<String>>,
    capacity: usize,
    usable: usize,
}
impl Deref for Dictionary {
    type Target = Vec<(String, Options)>;
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}
pub(super) fn keysize(minimum: usize) -> usize {
    // calculate_log2_keysize on the locked LP64 platform. Preserve its OR,
    // including presizing boundaries; max(8).next_power_of_two differs there.
    let value = (minimum | 8).saturating_sub(1) | 7;
    1usize
        .checked_shl(usize::BITS - value.leading_zeros())
        .expect("source dictionary key capacity exceeds native usize")
}
impl Dictionary {
    pub fn from_pairs(values: impl IntoIterator<Item = (String, Options)>) -> Self {
        let mut out = Self::default();
        for (key, value) in values {
            drop(out.set(&key, value));
        }
        out
    }
    pub fn presized(values: Vec<(String, Options)>) -> Self {
        let mut out = Self::default();
        if values.len() > 5 {
            if values.len() > ((1 << 17) * 2) / 3 {
                out.capacity = 1 << 17;
                out.usable = (out.capacity * 2) / 3;
            } else {
                out.resize_min((values.len() * 3).div_ceil(2));
            }
        }
        for (key, value) in values {
            drop(out.set(&key, value));
        }
        out
    }
    fn resize_min(&mut self, minimum: usize) {
        self.capacity = keysize(minimum);
        self.slots = self.entries.iter().map(|(k, _)| Some(k.clone())).collect();
        self.usable = (self.capacity * 2) / 3 - self.entries.len();
    }
    pub(super) fn set(&mut self, key: &str, value: Options) -> Option<Options> {
        if let Some((_, old)) = self.entries.iter_mut().find(|(k, _)| k == key) {
            return Some(std::mem::replace(old, value));
        }
        if self.usable == 0 {
            self.resize_min(self.entries.len() * 3);
        }
        self.slots.push(Some(key.into()));
        self.entries.push((key.into(), value));
        self.usable -= 1;
        None
    }
    pub(super) fn pop(&mut self, key: &str) -> Option<Options> {
        let index = self.entries.iter().position(|(k, _)| k == key)?;
        let slot = self
            .slots
            .iter()
            .position(|k| k.as_deref() == Some(key))
            .unwrap();
        self.slots[slot] = None;
        Some(self.entries.remove(index).1)
    }
    pub(super) fn snapshot(&self) -> Vec<(String, Options)> {
        self.entries.clone()
    }
    pub(super) fn source_copy(&self) -> Self {
        if self.entries.is_empty() {
            Self::default()
        } else if self.entries.len() >= (self.slots.len() * 2) / 3 {
            self.clone()
        } else {
            // PyDict_Copy falls back to dict_merge, whose initial resize is
            // estimate_log2_keysize(used), without NewPresized's small guard.
            let mut out = Self::default();
            out.resize_min((self.entries.len() * 3).div_ceil(2));
            for (key, value) in &self.entries {
                drop(out.set(key, value.clone()));
            }
            out
        }
    }
}
impl FromIterator<(String, Options)> for Dictionary {
    fn from_iter<I: IntoIterator<Item = (String, Options)>>(iter: I) -> Self {
        Self::from_pairs(iter)
    }
}
impl IntoIterator for Dictionary {
    type Item = (String, Options);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}
use super::dictionary_iterator::{DictionaryEntries, IteratorKind, SourceDictionaryIterator};
impl DictionaryEntries for Rc<RefCell<Dictionary>> {
    fn used(&self) -> usize {
        self.borrow().len()
    }
    fn next_position(&self, mut position: usize) -> Option<usize> {
        let dict = self.borrow();
        while position < dict.slots.len() && dict.slots[position].is_none() {
            position += 1;
        }
        (position < dict.slots.len()).then_some(position)
    }
    fn key_at(&self, position: usize) -> Options {
        Options::Atom(Value::Text(
            self.borrow().slots[position].as_ref().unwrap().clone(),
        ))
    }
    fn value_at(&self, position: usize) -> Options {
        let dict = self.borrow();
        let key = dict.slots[position].as_ref().unwrap();
        dict.entries
            .iter()
            .find(|(k, _)| k == key)
            .unwrap()
            .1
            .clone()
    }
}
enum IteratorStorage {
    Text(SourceDictionaryIterator<Rc<RefCell<Dictionary>>>),
    General(SourceDictionaryIterator<super::GeneralDictionary>),
}
pub struct ItemsIterator {
    inner: IteratorStorage,
}
impl ItemsIterator {
    pub(super) fn new(dictionary: Rc<RefCell<Dictionary>>) -> Self {
        Self::with_kind(dictionary, IteratorKind::Items)
    }
    fn with_kind(dictionary: Rc<RefCell<Dictionary>>, kind: IteratorKind) -> Self {
        Self {
            inner: IteratorStorage::Text(SourceDictionaryIterator::new(dictionary, kind)),
        }
    }
    pub(super) fn general(dictionary: super::GeneralDictionary) -> Self {
        Self::with_general(dictionary, IteratorKind::Items)
    }
    fn with_general(dictionary: super::GeneralDictionary, kind: IteratorKind) -> Self {
        Self {
            inner: IteratorStorage::General(SourceDictionaryIterator::new(dictionary, kind)),
        }
    }
    pub fn length_hint(&self) -> usize {
        match &self.inner {
            IteratorStorage::Text(v) => v.length_hint(),
            IteratorStorage::General(v) => v.length_hint(),
        }
    }
    pub fn next_item(&self) -> Result<Option<Options>, Error> {
        match &self.inner {
            IteratorStorage::Text(v) => v.next(),
            IteratorStorage::General(v) => v.next(),
        }
    }
}
pub struct KeysIterator {
    inner: ItemsIterator,
}
impl KeysIterator {
    pub(super) fn new(dictionary: Rc<RefCell<Dictionary>>) -> Self {
        Self {
            inner: ItemsIterator::with_kind(dictionary, IteratorKind::Keys),
        }
    }
    pub fn length_hint(&self) -> usize {
        self.inner.length_hint()
    }
    pub(super) fn general(dictionary: super::GeneralDictionary) -> Self {
        Self {
            inner: ItemsIterator::with_general(dictionary, IteratorKind::Keys),
        }
    }
    pub fn next_value(&self) -> Result<Option<Options>, Error> {
        self.inner.next_item()
    }
    pub fn next_key(&self) -> Result<Option<String>, Error> {
        self.next_value()?
            .map(|value| {
                if let Options::Atom(Value::Text(key)) = value {
                    return Ok(key);
                }
                match value.snapshot()? {
                    Value::Text(key) => Ok(key),
                    _ => Err(Error::BindingGap {
                        backend: "patch_options".into(),
                        operation: "non-string dictionary key requires its typed consumer".into(),
                    }),
                }
            })
            .transpose()
    }
}
pub struct ValuesIterator {
    inner: ItemsIterator,
}
impl ValuesIterator {
    pub(super) fn general(dictionary: super::GeneralDictionary) -> Self {
        Self {
            inner: ItemsIterator::with_general(dictionary, IteratorKind::Values),
        }
    }
    pub(super) fn new(dictionary: Rc<RefCell<Dictionary>>) -> Self {
        Self {
            inner: ItemsIterator::with_kind(dictionary, IteratorKind::Values),
        }
    }
    pub fn length_hint(&self) -> usize {
        self.inner.length_hint()
    }
    pub fn next_value(&self) -> Result<Option<Options>, Error> {
        self.inner.next_item()
    }
}
