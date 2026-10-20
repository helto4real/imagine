//! Combined, arbitrary-key CPython 3.12.13 dictionaries for native owner data.
//! Key protocols are required native bindings, not Rust Hash/Eq replacements.
use super::{
    Options,
    dictionary::keysize,
    dictionary_iterator::{DictionaryEntries, IteratorKind, SourceDictionaryIterator},
};
use crate::native::Error;
use std::{cell::RefCell, rc::Rc};

pub trait DictionaryKeyBindings {
    fn hash(&self, key: &Options) -> Result<i64, Error>;
    /// Pure canonical object identity, without retaining the object. This must
    /// not use value equality or invoke a user protocol.
    fn identity_token(&self, key: &Options) -> u64;
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error>;
    fn exact_unicode(&self, key: &Options) -> bool;
}
#[derive(Clone)]
struct Entry {
    key: Options,
    value: Options,
    hash: i64,
}
struct Data {
    // Release entries before their old table allocation on clear/deallocation.
    entries: Vec<Option<Entry>>,
    indices: Vec<isize>,
    used: usize,
    usable: usize,
    unicode: bool,
}
impl Default for Data {
    fn default() -> Self {
        Self {
            entries: vec![],
            indices: vec![],
            used: 0,
            usable: 0,
            unicode: true,
        }
    }
}
#[derive(Clone, Default)]
pub struct GeneralDictionary(Rc<RefCell<Data>>);
pub struct DictionaryLayout {
    pub capacity: usize,
    pub used: usize,
    pub usable: usize,
    pub entry_count: usize,
    pub unicode: bool,
    pub indices: Vec<isize>,
}
fn memory_error() -> Error {
    Error::ReferenceException {
        class: "MemoryError",
        message: String::new(),
    }
}
fn empty_slot(indices: &[isize], hash: i64) -> usize {
    let mask = indices.len() - 1;
    let mut perturb = hash as usize;
    let mut index = perturb & mask;
    while indices[index] >= 0 {
        perturb >>= 5;
        index = index.wrapping_mul(5).wrapping_add(perturb).wrapping_add(1) & mask;
    }
    index
}
fn resize(data: &mut Data, capacity: usize, unicode: bool) -> Result<(), Error> {
    let mut indices = vec![];
    indices
        .try_reserve_exact(capacity)
        .map_err(|_| memory_error())?;
    indices.resize(capacity, -1);
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(capacity * 2 / 3)
        .map_err(|_| memory_error())?;
    // New storage is allocated before releasing the old table; lookup compares
    // its allocation identity after a reentrant rich comparison, as source does.
    for entry in std::mem::take(&mut data.entries).into_iter().flatten() {
        let index = empty_slot(&indices, entry.hash);
        indices[index] = entries.len() as isize;
        entries.push(Some(entry));
    }
    data.entries = entries;
    data.indices = indices;
    data.usable = capacity * 2 / 3 - data.used;
    data.unicode &= unicode;
    Ok(())
}
impl GeneralDictionary {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn presized(minimum_used: usize) -> Result<Self, Error> {
        if minimum_used <= 5 {
            return Ok(Self::new());
        }
        let capacity = if minimum_used > ((1 << 17) * 2) / 3 {
            1 << 17
        } else {
            keysize(
                minimum_used
                    .checked_mul(3)
                    .ok_or_else(memory_error)?
                    .div_ceil(2),
            )
        };
        let mut data = Data::default();
        // _PyDict_NewPresized requests a general-key table. In particular its
        // first non-Unicode key must not trigger Unicode-table conversion and
        // discard the capacity selected above.
        resize(&mut data, capacity, false)?;
        Ok(Self(Rc::new(RefCell::new(data))))
    }
    pub fn same_object(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
    pub fn len(&self) -> usize {
        self.0.borrow().used
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn entries(&self) -> Vec<(Options, Options)> {
        self.0
            .borrow()
            .entries
            .iter()
            .flatten()
            .map(|entry| (entry.key.clone(), entry.value.clone()))
            .collect()
    }
    pub fn layout(&self) -> DictionaryLayout {
        let data = self.0.borrow();
        DictionaryLayout {
            capacity: data.indices.len().max(1),
            used: data.used,
            usable: data.usable,
            entry_count: data.entries.len(),
            unicode: data.unicode,
            indices: if data.indices.is_empty() {
                vec![-1]
            } else {
                data.indices.clone()
            },
        }
    }
    fn hash(key: &Options, bindings: &dyn DictionaryKeyBindings) -> Result<i64, Error> {
        let hash = bindings.hash(key)?;
        if hash == -1 {
            return Err(Error::BindingGap {
                backend: "dictionary_keys".into(),
                operation: "native key hash returned the reserved error sentinel".into(),
            });
        }
        Ok(hash)
    }
    fn lookup(
        &self,
        key: &Options,
        hash: i64,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<Option<(usize, usize)>, Error> {
        'restart: loop {
            let (mask, allocation) = {
                let data = self.0.borrow();
                if data.indices.is_empty() {
                    return Ok(None);
                }
                (data.indices.len() - 1, data.indices.as_ptr() as usize)
            };
            let mut perturb = hash as usize;
            let mut index = perturb & mask;
            loop {
                let candidate = {
                    let data = self.0.borrow();
                    let position = data.indices[index];
                    if position == -1 {
                        return Ok(None);
                    }
                    if position < 0 {
                        None
                    } else {
                        let entry = data.entries[position as usize].as_ref().unwrap();
                        Some((position as usize, entry.key.clone(), entry.hash))
                    }
                };
                if let Some((position, candidate, candidate_hash)) = candidate {
                    let identity = bindings.identity_token(&candidate);
                    if identity == bindings.identity_token(key) {
                        return Ok(Some((position, index)));
                    }
                    if candidate_hash == hash {
                        let equal = bindings.equal(&candidate, key);
                        // The source releases its temporary key before testing
                        // table/key identity; its finalizer may mutate the dict.
                        drop(candidate);
                        let equal = equal?;
                        let unchanged = {
                            let data = self.0.borrow();
                            data.indices.as_ptr() as usize == allocation
                                && data
                                    .entries
                                    .get(position)
                                    .and_then(Option::as_ref)
                                    .is_some_and(|entry| {
                                        bindings.identity_token(&entry.key) == identity
                                    })
                        };
                        if !unchanged {
                            continue 'restart;
                        }
                        if equal {
                            return Ok(Some((position, index)));
                        }
                    }
                }
                perturb >>= 5;
                index = index.wrapping_mul(5).wrapping_add(perturb).wrapping_add(1) & mask;
            }
        }
    }
    pub fn get(
        &self,
        key: &Options,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<Option<Options>, Error> {
        let hash = Self::hash(key, bindings)?;
        let position = self.lookup(key, hash, bindings)?;
        Ok(position.map(|(position, _)| {
            self.0.borrow().entries[position]
                .as_ref()
                .unwrap()
                .value
                .clone()
        }))
    }
    pub fn set(
        &self,
        key: Options,
        value: Options,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<(), Error> {
        let hash = match Self::hash(&key, bindings) {
            Ok(hash) => hash,
            Err(error) => {
                drop(key);
                drop(value);
                return Err(error);
            }
        };
        self.set_hashed(key, value, hash, bindings)
    }
    /// dict.setdefault hashes once and performs one rich-key lookup. A get
    /// followed by set would repeat observable protocols on a missing key.
    pub fn setdefault(
        &self,
        key: &Options,
        default: &Options,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<Options, Error> {
        let hash = Self::hash(key, bindings)?;
        {
            let mut data = self.0.borrow_mut();
            if data.indices.is_empty() {
                resize(&mut data, 8, bindings.exact_unicode(key))?;
            } else if data.unicode && !bindings.exact_unicode(key) {
                let capacity = keysize(data.used.checked_mul(3).ok_or_else(memory_error)?);
                resize(&mut data, capacity, false)?;
            }
        }
        if let Some((position, _)) = self.lookup(key, hash, bindings)? {
            return Ok(self.0.borrow().entries[position]
                .as_ref()
                .unwrap()
                .value
                .clone());
        }
        let mut data = self.0.borrow_mut();
        if data.usable == 0 {
            let capacity = keysize(data.used.checked_mul(3).ok_or_else(memory_error)?);
            resize(&mut data, capacity, true)?;
        }
        let slot = empty_slot(&data.indices, hash);
        data.indices[slot] = data.entries.len() as isize;
        data.entries.push(Some(Entry {
            key: key.clone(),
            value: default.clone(),
            hash,
        }));
        data.used += 1;
        data.usable -= 1;
        Ok(default.clone())
    }
    fn set_hashed(
        &self,
        key: Options,
        value: Options,
        hash: i64,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<(), Error> {
        let initially_empty = self.0.borrow().indices.is_empty();
        let mut key = Some(key);
        let mut value = Some(value);
        let outcome = (|| {
            {
                let mut data = self.0.borrow_mut();
                if data.indices.is_empty() {
                    resize(&mut data, 8, bindings.exact_unicode(key.as_ref().unwrap()))?;
                } else if data.unicode && !bindings.exact_unicode(key.as_ref().unwrap()) {
                    let capacity = keysize(data.used.checked_mul(3).ok_or_else(memory_error)?);
                    resize(&mut data, capacity, false)?;
                }
            }
            if let Some((position, _)) = self.lookup(key.as_ref().unwrap(), hash, bindings)? {
                let old = {
                    let mut data = self.0.borrow_mut();
                    let entry = data.entries[position].as_mut().unwrap();
                    std::mem::replace(&mut entry.value, value.take().unwrap())
                };
                drop(old);
                drop(key.take());
            } else {
                let mut data = self.0.borrow_mut();
                if data.usable == 0 {
                    let capacity = keysize(data.used.checked_mul(3).ok_or_else(memory_error)?);
                    resize(&mut data, capacity, true)?;
                }
                let slot = empty_slot(&data.indices, hash);
                data.indices[slot] = data.entries.len() as isize;
                data.entries.push(Some(Entry {
                    key: key.take().unwrap(),
                    value: value.take().unwrap(),
                    hash,
                }));
                data.used += 1;
                data.usable -= 1;
            }
            Ok(())
        })();
        if initially_empty {
            drop(key);
            drop(value);
        } else {
            drop(value);
            drop(key);
        }
        outcome
    }
    pub fn pop(
        &self,
        key: &Options,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<Option<Options>, Error> {
        // dict.pop checks ma_used before hashing its key, unlike get/contains.
        if self.is_empty() {
            return Ok(None);
        }
        let hash = Self::hash(key, bindings)?;
        let Some((position, slot)) = self.lookup(key, hash, bindings)? else {
            return Ok(None);
        };
        let entry = {
            let mut data = self.0.borrow_mut();
            data.indices[slot] = -2;
            data.used -= 1;
            data.entries[position].take().unwrap()
        };
        drop(entry.key);
        Ok(Some(entry.value))
    }
    pub fn clear(&self) {
        let old = std::mem::take(&mut *self.0.borrow_mut());
        drop(old);
    }
    pub fn contains(
        &self,
        key: &Options,
        bindings: &dyn DictionaryKeyBindings,
    ) -> Result<bool, Error> {
        let hash = Self::hash(key, bindings)?;
        Ok(self.lookup(key, hash, bindings)?.is_some())
    }
    /// Exact combined-dict copy. Fast copy retains holes without protocols;
    /// compacting copy reuses cached hashes but may call rich key equality.
    pub fn shallow_copy(&self, bindings: &dyn DictionaryKeyBindings) -> Result<Self, Error> {
        let data = self.0.borrow();
        if data.used == 0 {
            return Ok(Self::new());
        }
        if data.used >= (data.entries.len() * 2) / 3 {
            let mut copied = Data {
                entries: vec![],
                indices: vec![],
                used: data.used,
                usable: data.usable,
                unicode: data.unicode,
            };
            copied
                .entries
                .try_reserve_exact(data.entries.len())
                .map_err(|_| memory_error())?;
            copied.entries.extend(data.entries.iter().cloned());
            copied
                .indices
                .try_reserve_exact(data.indices.len())
                .map_err(|_| memory_error())?;
            copied.indices.extend_from_slice(&data.indices);
            return Ok(Self(Rc::new(RefCell::new(copied))));
        }
        let original_count = data.entries.len();
        let minimum = data
            .used
            .checked_mul(3)
            .ok_or_else(memory_error)?
            .div_ceil(2);
        let unicode = data.unicode;
        drop(data);
        let mut copied = Data::default();
        resize(&mut copied, keysize(minimum), unicode)?;
        let copied = Self(Rc::new(RefCell::new(copied)));
        let mut position = 0;
        loop {
            let entry = {
                let data = self.0.borrow();
                while position < data.entries.len() && data.entries[position].is_none() {
                    position += 1;
                }
                let entry = data.entries.get(position).and_then(Option::as_ref).cloned();
                position += usize::from(entry.is_some());
                entry
            };
            let Some(entry) = entry else {
                return Ok(copied);
            };
            let result =
                copied.set_hashed(entry.key.clone(), entry.value.clone(), entry.hash, bindings);
            drop(entry.value);
            drop(entry.key);
            result?;
            if self.0.borrow().entries.len() != original_count {
                return Err(Error::ReferenceException {
                    class: "RuntimeError",
                    message: "dict mutated during update".into(),
                });
            }
        }
    }
}

impl DictionaryEntries for GeneralDictionary {
    fn used(&self) -> usize {
        self.len()
    }
    fn next_position(&self, mut position: usize) -> Option<usize> {
        let data = self.0.borrow();
        while position < data.entries.len() && data.entries[position].is_none() {
            position += 1;
        }
        (position < data.entries.len()).then_some(position)
    }
    fn key_at(&self, position: usize) -> Options {
        self.0.borrow().entries[position]
            .as_ref()
            .unwrap()
            .key
            .clone()
    }
    fn value_at(&self, position: usize) -> Options {
        self.0.borrow().entries[position]
            .as_ref()
            .unwrap()
            .value
            .clone()
    }
}
pub struct GeneralDictionaryIterator {
    inner: SourceDictionaryIterator<GeneralDictionary>,
}
impl GeneralDictionaryIterator {
    pub fn length_hint(&self) -> usize {
        self.inner.length_hint()
    }
    pub fn next(&self) -> Result<Option<Options>, Error> {
        self.inner.next()
    }
}
impl GeneralDictionary {
    pub fn items_iterator(&self) -> GeneralDictionaryIterator {
        GeneralDictionaryIterator {
            inner: SourceDictionaryIterator::new(self.clone(), IteratorKind::Items),
        }
    }
    pub fn keys_iterator(&self) -> GeneralDictionaryIterator {
        GeneralDictionaryIterator {
            inner: SourceDictionaryIterator::new(self.clone(), IteratorKind::Keys),
        }
    }
    pub fn values_iterator(&self) -> GeneralDictionaryIterator {
        GeneralDictionaryIterator {
            inner: SourceDictionaryIterator::new(self.clone(), IteratorKind::Values),
        }
    }
}
