//! Mutable CPython set table with fallible/reentrant native key protocols.
//! Keep add's positive-comparison short circuit distinct from lookup's restart.
use super::{MappingKeyBindings, MetadataObject, MetadataValue, Options, SourceText};
use crate::{execution_value::Value, native::Error};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::{Rc, Weak},
};
enum Slot {
    Empty,
    Dummy,
    Active { key: Options, hash: i64 },
}
struct Data {
    small: [Slot; 8],
    heap: Option<Vec<Slot>>,
    used: usize,
    fill: usize,
    finger: usize,
}
impl Default for Data {
    fn default() -> Self {
        Self {
            small: std::array::from_fn(|_| Slot::Empty),
            heap: None,
            used: 0,
            fill: 0,
            finger: 0,
        }
    }
}
impl Data {
    fn slots(&self) -> &[Slot] {
        self.heap.as_deref().unwrap_or(&self.small)
    }
    fn slots_mut(&mut self) -> &mut [Slot] {
        self.heap.as_deref_mut().unwrap_or(&mut self.small)
    }
    fn table(&self) -> usize {
        self.heap.as_ref().map_or(0, |v| v.as_ptr() as usize)
    }
    fn clean_insert(&mut self, key: Options, hash: i64) {
        let mask = self.slots().len() - 1;
        let mut perturb = hash as usize;
        let mut index = perturb & mask;
        loop {
            let probes = if index + 9 <= mask { 9 } else { 0 };
            for slot in index..=index + probes {
                if matches!(self.slots()[slot], Slot::Empty) {
                    self.slots_mut()[slot] = Slot::Active { key, hash };
                    return;
                }
            }
            perturb >>= 5;
            index = index.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
        }
    }
    fn resize(&mut self, minimum: usize) -> Result<(), Error> {
        let mut size = 8_usize;
        while size <= minimum {
            size = size.checked_mul(2).ok_or_else(memory)?;
        }
        if size == 8 && self.heap.is_none() && self.fill == self.used {
            return Ok(());
        }
        let heap = if size > 8 {
            let mut values = Vec::new();
            values.try_reserve_exact(size).map_err(|_| memory())?;
            values.resize_with(size, || Slot::Empty);
            Some(values)
        } else {
            None
        };
        let old_heap = std::mem::replace(&mut self.heap, heap);
        let old_small = std::mem::replace(&mut self.small, std::array::from_fn(|_| Slot::Empty));
        self.fill = self.used;
        if let Some(old) = old_heap {
            for slot in old {
                if let Slot::Active { key, hash } = slot {
                    self.clean_insert(key, hash);
                }
            }
        } else {
            for slot in old_small {
                if let Slot::Active { key, hash } = slot {
                    self.clean_insert(key, hash);
                }
            }
        }
        Ok(())
    }
}
fn memory() -> Error {
    Error::ReferenceException {
        class: "MemoryError",
        message: String::new(),
    }
}
#[derive(Clone)]
pub struct GeneralSet {
    data: Rc<RefCell<Data>>,
    bindings: Rc<dyn MappingKeyBindings>,
}
/// Exact immutable set owner. The private table is never published as mutable.
#[derive(Clone)]
pub struct FrozenSet {
    owner: Rc<FrozenData>,
}
struct FrozenData {
    table: GeneralSet,
    hash: Cell<i64>,
}
impl FrozenSet {
    pub(crate) fn from_new_table(table: GeneralSet) -> Self {
        Self {
            owner: Rc::new(FrozenData {
                table,
                hash: Cell::new(-1),
            }),
        }
    }
    pub fn from_set(source: &GeneralSet) -> Result<Self, Error> {
        Ok(Self {
            owner: Rc::new(FrozenData {
                table: source.copy_set()?,
                hash: Cell::new(-1),
            }),
        })
    }
    pub fn into_options(self) -> Options {
        MetadataObject::wrap(self)
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        MetadataObject::get(value.as_ref())
    }
    pub fn copy(&self) -> Self {
        self.clone()
    }
    pub fn same_object(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.owner, &other.owner)
    }
    pub fn identity_token(&self) -> u64 {
        Rc::as_ptr(&self.owner) as usize as u64
    }
    pub fn len(&self) -> usize {
        self.owner.table.len()
    }
    pub fn is_empty(&self) -> bool {
        self.owner.table.is_empty()
    }
    pub fn capacity(&self) -> usize {
        self.owner.table.capacity()
    }
    pub fn entries(&self) -> Vec<(Options, i64)> {
        self.owner.table.entries()
    }
    pub fn iterator(&self) -> GeneralSetIterator {
        let iterator = self.owner.table.iterator();
        iterator.state.borrow_mut().frozen_owner = Some(self.clone());
        iterator
    }
    pub fn contains(&self, key: &Options) -> Result<bool, Error> {
        self.owner.table.contains(key)
    }
    pub fn hash(&self) -> i64 {
        if self.owner.hash.get() != -1 {
            return self.owner.hash.get();
        }
        fn shuffle(h: u64) -> u64 {
            ((h ^ 89869747) ^ (h << 16)).wrapping_mul(3644798167)
        }
        let data = self.owner.table.data.borrow();
        let mut hash = 0_u64;
        for slot in data.slots() {
            let h = match slot {
                Slot::Empty => 0,
                Slot::Dummy => -1,
                Slot::Active { hash, .. } => *hash,
            };
            hash ^= shuffle(h as u64);
        }
        if data.slots().len().wrapping_sub(data.fill) & 1 != 0 {
            hash ^= shuffle(0);
        }
        if data.fill.wrapping_sub(data.used) & 1 != 0 {
            hash ^= shuffle(u64::MAX);
        }
        hash ^= (data.used as u64).wrapping_add(1).wrapping_mul(1927868237);
        hash ^= (hash >> 11) ^ (hash >> 25);
        hash = hash.wrapping_mul(69069).wrapping_add(907133923);
        if hash == u64::MAX {
            hash = 590923713;
        }
        self.owner.hash.set(hash as i64);
        hash as i64
    }
}
fn set_view(value: &Options) -> Option<(&GeneralSet, i64)> {
    if let Some(set) = GeneralSet::get(value) {
        Some((set, -1))
    } else {
        FrozenSet::get(value).map(|set| (&set.owner.table, set.owner.hash.get()))
    }
}
/// PyAnySet rich equality; None is the source NotImplemented result.
pub fn set_equal(left: &Options, right: &Options) -> Result<Option<bool>, Error> {
    let (Some((left, lh)), Some((right, rh))) = (set_view(left), set_view(right)) else {
        return Ok(None);
    };
    if left.len() != right.len() || (lh != -1 && rh != -1 && lh != rh) {
        return Ok(Some(false));
    }
    let mut position = 0;
    while let Some((key, hash)) = left.next_cached(&mut position) {
        if !matches!(right.probe(&key, hash, false)?, Probe::Found(_)) {
            return Ok(Some(false));
        }
    }
    Ok(Some(true))
}
pub struct WeakGeneralSet {
    data: Weak<RefCell<Data>>,
    bindings: Weak<dyn MappingKeyBindings>,
}
impl WeakGeneralSet {
    pub fn upgrade(&self) -> Option<GeneralSet> {
        Some(GeneralSet {
            data: self.data.upgrade()?,
            bindings: self.bindings.upgrade()?,
        })
    }
}
enum Probe {
    Found(usize),
    Vacant { slot: usize, dummy: bool },
}
impl GeneralSet {
    pub fn merge_set_object(&self, source: &Options) -> Result<bool, Error> {
        let Some((source, _)) = set_view(source) else {
            return Ok(false);
        };
        self.merge_set(source)?;
        Ok(true)
    }
    pub fn merge_dictionary(&self, source: &super::GeneralDictionary) -> Result<(), Error> {
        {
            let mut data = self.data.borrow_mut();
            let size = source.len();
            if data
                .fill
                .checked_add(size)
                .and_then(|n| n.checked_mul(5))
                .ok_or_else(memory)?
                >= (data.slots().len() - 1) * 3
            {
                let minimum = data
                    .used
                    .checked_add(size)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(memory)?;
                data.resize(minimum)?;
            }
        }
        let mut position = 0;
        while let Some((key, hash)) = source.next_key_hash(&mut position) {
            self.add_cached(&key, hash)?;
        }
        Ok(())
    }
    pub fn new(bindings: Rc<dyn MappingKeyBindings>) -> Self {
        Self {
            data: Rc::new(RefCell::new(Data::default())),
            bindings,
        }
    }
    pub fn into_options(self) -> Options {
        MetadataObject::wrap(self)
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        MetadataObject::get(value.as_ref())
    }
    pub fn len(&self) -> usize {
        self.data.borrow().used
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn capacity(&self) -> usize {
        self.data.borrow().slots().len()
    }
    pub fn fill(&self) -> usize {
        self.data.borrow().fill
    }
    pub fn same_object(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
    pub fn identity_token(&self) -> u64 {
        Rc::as_ptr(&self.data) as usize as u64
    }
    pub fn downgrade(&self) -> WeakGeneralSet {
        WeakGeneralSet {
            data: Rc::downgrade(&self.data),
            bindings: Rc::downgrade(&self.bindings),
        }
    }
    pub fn entries(&self) -> Vec<(Options, i64)> {
        self.data
            .borrow()
            .slots()
            .iter()
            .filter_map(|slot| {
                if let Slot::Active { key, hash } = slot {
                    Some((key.clone(), *hash))
                } else {
                    None
                }
            })
            .collect()
    }
    /// Internal set_next walks the live table without the public iterator's
    /// used-count guard. C set operators use this exact traversal.
    fn next_cached(&self, position: &mut usize) -> Option<(Options, i64)> {
        let data = self.data.borrow();
        while *position < data.slots().len() {
            let current = *position;
            *position += 1;
            if let Slot::Active { key, hash } = &data.slots()[current] {
                return Some((key.clone(), *hash));
            }
        }
        None
    }
    pub fn merge_set(&self, other: &Self) -> Result<(), Error> {
        if self.same_object(other) || other.is_empty() {
            return Ok(());
        }
        {
            let mut data = self.data.borrow_mut();
            let other_used = other.len();
            let mask = data.slots().len() - 1;
            if data
                .fill
                .checked_add(other_used)
                .and_then(|n| n.checked_mul(5))
                .ok_or_else(memory)?
                >= mask * 3
            {
                let minimum = data
                    .used
                    .checked_add(other_used)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(memory)?;
                data.resize(minimum)?;
            }
            let source = other.data.borrow();
            if data.fill == 0 {
                if data.slots().len() == source.slots().len() && source.fill == source.used {
                    for (target, source) in data.slots_mut().iter_mut().zip(source.slots()) {
                        if let Slot::Active { key, hash } = source {
                            *target = Slot::Active {
                                key: key.clone(),
                                hash: *hash,
                            };
                        }
                    }
                    data.fill = source.fill;
                    data.used = source.used;
                    return Ok(());
                }
                data.fill = source.used;
                data.used = source.used;
                for slot in source.slots() {
                    if let Slot::Active { key, hash } = slot {
                        data.clean_insert(key.clone(), *hash);
                    }
                }
                return Ok(());
            }
        }
        let mut position = 0;
        while let Some((key, hash)) = other.next_cached(&mut position) {
            self.add_cached(&key, hash)?;
        }
        Ok(())
    }
    pub fn copy_set(&self) -> Result<Self, Error> {
        let result = Self::new(self.bindings.clone());
        result.merge_set(self)?;
        Ok(result)
    }
    pub fn union_set(&self, other: &Self) -> Result<Self, Error> {
        let result = self.copy_set()?;
        if !self.same_object(other) {
            result.merge_set(other)?;
        }
        Ok(result)
    }
    fn discard_cached(&self, key: &Options, hash: i64) -> Result<bool, Error> {
        let Probe::Found(slot) = self.probe(key, hash, false)? else {
            return Ok(false);
        };
        let old = {
            let mut data = self.data.borrow_mut();
            data.used -= 1;
            std::mem::replace(&mut data.slots_mut()[slot], Slot::Dummy)
        };
        drop(old);
        Ok(true)
    }
    pub fn intersection_set(&self, other: &Self) -> Result<Self, Error> {
        if self.same_object(other) {
            return self.copy_set();
        }
        let result = Self::new(self.bindings.clone());
        let (mut probe, mut source) = (self, other);
        if other.len() > self.len() {
            (probe, source) = (other, self);
        }
        let mut current = None;
        let outcome: Result<(), Error> = (|| {
            let mut position = 0;
            while let Some((key, hash)) = source.next_cached(&mut position) {
                current = Some(key);
                let key = current.as_ref().unwrap();
                if matches!(probe.probe(key, hash, false)?, Probe::Found(_)) {
                    result.add_cached(key, hash)?;
                }
                drop(current.take());
            }
            Ok(())
        })();
        match outcome {
            Ok(()) => Ok(result),
            Err(error) => {
                drop(result);
                drop(current);
                Err(error)
            }
        }
    }
    pub fn difference_update_set(&self, other: &Self) -> Result<(), Error> {
        if self.same_object(other) {
            self.clear();
            return Ok(());
        }
        let source = if (other.len() >> 3) > self.len() {
            self.intersection_set(other)?
        } else {
            other.clone()
        };
        let mut current = None;
        let outcome: Result<(), Error> = (|| {
            let mut position = 0;
            while let Some((key, hash)) = source.next_cached(&mut position) {
                current = Some(key);
                self.discard_cached(current.as_ref().unwrap(), hash)?;
                drop(current.take());
            }
            Ok(())
        })();
        drop(source);
        drop(current);
        outcome?;
        let mut data = self.data.borrow_mut();
        let mask = data.slots().len() - 1;
        if data.fill.wrapping_sub(data.used) > mask / 4 {
            let minimum = data
                .used
                .checked_mul(if data.used > 50_000 { 2 } else { 4 })
                .ok_or_else(memory)?;
            data.resize(minimum)?;
        }
        Ok(())
    }
    pub fn difference_set(&self, other: &Self) -> Result<Self, Error> {
        if (self.len() >> 2) > other.len() {
            let result = self.copy_set()?;
            result.difference_update_set(other)?;
            return Ok(result);
        }
        let result = Self::new(self.bindings.clone());
        let mut current = None;
        let outcome: Result<(), Error> = (|| {
            let mut position = 0;
            while let Some((key, hash)) = self.next_cached(&mut position) {
                current = Some(key);
                let key = current.as_ref().unwrap();
                if !matches!(other.probe(key, hash, false)?, Probe::Found(_)) {
                    result.add_cached(key, hash)?;
                }
                drop(current.take());
            }
            Ok(())
        })();
        match outcome {
            Ok(()) => Ok(result),
            Err(error) => {
                drop(result);
                drop(current);
                Err(error)
            }
        }
    }
    fn hash(&self, key: &Options) -> Result<i64, Error> {
        let hash = self.bindings.hash(key)?;
        if hash == -1 {
            return Err(Error::BindingGap {
                backend: "set_keys".into(),
                operation: "key hash returned the reserved error sentinel".into(),
            });
        }
        Ok(hash)
    }
    fn probe(&self, key: &Options, hash: i64, add: bool) -> Result<Probe, Error> {
        'restart: loop {
            let (mask, allocation) = {
                let data = self.data.borrow();
                (data.slots().len() - 1, data.table())
            };
            let mut perturb = hash as usize;
            let mut index = perturb & mask;
            let mut free = None;
            loop {
                let probes = if index + 9 <= mask { 9 } else { 0 };
                for position in index..=index + probes {
                    let candidate = {
                        let data = self.data.borrow();
                        match &data.slots()[position] {
                            Slot::Empty => {
                                return Ok(Probe::Vacant {
                                    slot: free.unwrap_or(position),
                                    dummy: free.is_some(),
                                });
                            }
                            Slot::Dummy => {
                                free = Some(position);
                                None
                            }
                            Slot::Active { key, hash: old } if *old == hash => Some(key.clone()),
                            _ => None,
                        }
                    };
                    if let Some(candidate) = candidate {
                        let identity = self.bindings.identity_token(&candidate);
                        if identity == self.bindings.identity_token(key) {
                            return Ok(Probe::Found(position));
                        }
                        if let (Some(a), Some(b)) =
                            (SourceText::get(&candidate), SourceText::get(key))
                            && a.codepoints() == b.codepoints()
                        {
                            return Ok(Probe::Found(position));
                        }
                        let equal = self.bindings.equal(&candidate, key);
                        drop(candidate);
                        let equal = equal?;
                        // set_add_entry returns immediately for a positive cmp,
                        // even if that cmp changed the table. set_lookkey does not.
                        if add && equal {
                            return Ok(Probe::Found(position));
                        }
                        let unchanged = {
                            let data = self.data.borrow();
                            data.table() == allocation
                                && matches!(&data.slots()[position],Slot::Active {key,..}if self.bindings.identity_token(key)==identity)
                        };
                        if !unchanged {
                            continue 'restart;
                        }
                        if equal {
                            return Ok(Probe::Found(position));
                        }
                    }
                }
                perturb >>= 5;
                index = index.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
            }
        }
    }
    pub fn add(&self, key: &Options) -> Result<(), Error> {
        let hash = self.hash(key)?;
        self.add_cached(key, hash)
    }
    fn add_cached(&self, key: &Options, hash: i64) -> Result<(), Error> {
        let owned = key.clone();
        let probe = self.probe(&owned, hash, true)?;
        if let Probe::Vacant { slot, dummy } = probe {
            let mut data = self.data.borrow_mut();
            let old = std::mem::replace(
                &mut data.slots_mut()[slot],
                Slot::Active { key: owned, hash },
            );
            // Source keeps its previously observed dummy pointer across user
            // comparisons. If a callback filled it, the overwritten C reference
            // is leaked rather than decref'd. Preserve that rare source effect.
            if matches!(old, Slot::Active { .. }) {
                std::mem::forget(old);
            }
            data.used += 1;
            if !dummy {
                data.fill += 1;
                let mask = data.slots().len() - 1;
                if data.fill * 5 >= mask * 3 {
                    let minimum = data
                        .used
                        .checked_mul(if data.used > 50_000 { 2 } else { 4 })
                        .ok_or_else(memory)?;
                    data.resize(minimum)?;
                }
            }
        }
        Ok(())
    }
    /// set_contains_key kernel; the mutable-set/frozenset wrapper is separate.
    pub fn contains_key(&self, key: &Options) -> Result<bool, Error> {
        Ok(matches!(
            self.probe(key, self.hash(key)?, false)?,
            Probe::Found(_)
        ))
    }
    /// set_discard_key kernel, without its mutable-set argument wrapper.
    pub fn discard_key(&self, key: &Options) -> Result<bool, Error> {
        let hash = self.hash(key)?;
        self.discard_cached(key, hash)
    }
    pub fn remove_key(&self, key: &Options) -> Result<(), Error> {
        if self.discard_key(key)? {
            Ok(())
        } else {
            Err(self.bindings.missing_key(key)?)
        }
    }
    fn mutable_argument<T>(
        &self,
        key: &Options,
        operation: impl Fn(&Options) -> Result<T, Error>,
    ) -> Result<T, Error> {
        match operation(key) {
            Err(Error::ReferenceException {
                class: "TypeError", ..
            }) if GeneralSet::get(key).is_some() => {
                let temporary = FrozenSet::from_set(GeneralSet::get(key).unwrap())?.into_options();
                operation(&temporary)
            }
            result => result,
        }
    }
    /// Python's public wrapper retries an unhashable mutable set as a frozenset.
    pub fn contains(&self, key: &Options) -> Result<bool, Error> {
        self.mutable_argument(key, |key| self.contains_key(key))
    }
    pub fn discard(&self, key: &Options) -> Result<bool, Error> {
        self.mutable_argument(key, |key| self.discard_key(key))
    }
    pub fn remove(&self, key: &Options) -> Result<(), Error> {
        if self.discard(key)? {
            Ok(())
        } else {
            Err(self.bindings.missing_key(key)?)
        }
    }
    pub fn clear(&self) {
        let old = {
            let mut data = self.data.borrow_mut();
            let old = std::mem::take(&mut *data);
            data.finger = old.finger;
            old
        };
        drop(old);
    }
    pub fn pop(&self) -> Result<Options, Error> {
        let mut data = self.data.borrow_mut();
        if data.used == 0 {
            drop(data);
            let key = self.bindings.text_key("pop from an empty set")?;
            return Err(self.bindings.missing_key(&key)?);
        }
        let mask = data.slots().len() - 1;
        let mut position = data.finger & mask;
        while !matches!(data.slots()[position], Slot::Active { .. }) {
            position = (position + 1) & mask;
        }
        let Slot::Active { key, .. } =
            std::mem::replace(&mut data.slots_mut()[position], Slot::Dummy)
        else {
            unreachable!()
        };
        data.used -= 1;
        data.finger = position + 1;
        Ok(key)
    }
    pub fn is_subset_set(&self, other: &Self) -> Result<bool, Error> {
        if self.len() > other.len() {
            return Ok(false);
        }
        let mut position = 0;
        while let Some((key, hash)) = self.next_cached(&mut position) {
            if !matches!(other.probe(&key, hash, false)?, Probe::Found(_)) {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn is_disjoint_set(&self, other: &Self) -> Result<bool, Error> {
        if self.same_object(other) {
            return Ok(self.is_empty());
        }
        let (probe, source) = if other.len() > self.len() {
            (other, self)
        } else {
            (self, other)
        };
        let mut position = 0;
        while let Some((key, hash)) = source.next_cached(&mut position) {
            if matches!(probe.probe(&key, hash, false)?, Probe::Found(_)) {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn symmetric_difference_update_set(&self, other: &Self) -> Result<(), Error> {
        if self.same_object(other) {
            self.clear();
            return Ok(());
        }
        let mut position = 0;
        while let Some((key, hash)) = other.next_cached(&mut position) {
            if !self.discard_cached(&key, hash)? {
                self.add_cached(&key, hash)?;
            }
        }
        Ok(())
    }
    pub fn symmetric_difference_set(&self, other: &Self) -> Result<Self, Error> {
        let result = other.copy_set()?;
        result.symmetric_difference_update_set(self)?;
        Ok(result)
    }
    pub fn intersection_update_set(&self, other: &Self) -> Result<(), Error> {
        let result = self.intersection_set(other)?;
        if !self.same_object(&result) {
            let mut left = self.data.borrow_mut();
            let mut right = result.data.borrow_mut();
            let (left_finger, right_finger) = (left.finger, right.finger);
            std::mem::swap(&mut *left, &mut *right);
            left.finger = left_finger;
            right.finger = right_finger;
        }
        drop(result);
        Ok(())
    }
    pub fn iterator(&self) -> GeneralSetIterator {
        GeneralSetIterator {
            state: RefCell::new(IteratorState {
                owner: Some(self.clone()),
                used: self.len(),
                failed: false,
                position: 0,
                remaining: self.len() as isize,
                frozen_owner: None,
            }),
        }
    }
}
struct IteratorState {
    owner: Option<GeneralSet>,
    used: usize,
    failed: bool,
    position: usize,
    remaining: isize,
    frozen_owner: Option<FrozenSet>,
}
pub struct GeneralSetIterator {
    state: RefCell<IteratorState>,
}
impl GeneralSetIterator {
    pub fn length_hint(&self) -> isize {
        let state = self.state.borrow();
        if state
            .owner
            .as_ref()
            .is_some_and(|set| set.len() == state.used)
            && !state.failed
        {
            state.remaining
        } else {
            0
        }
    }
    pub fn next(&self) -> Result<Option<Options>, Error> {
        let mut state = self.state.borrow_mut();
        let Some(owner) = state.owner.clone() else {
            return Ok(None);
        };
        if state.failed || owner.len() != state.used {
            state.failed = true;
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: "Set changed size during iteration".into(),
            });
        }
        let value = {
            let data = owner.data.borrow();
            let mut found = None;
            while state.position < data.slots().len() {
                let position = state.position;
                state.position += 1;
                if let Slot::Active { key, .. } = &data.slots()[position] {
                    found = Some(key.clone());
                    break;
                }
            }
            found
        };
        if value.is_some() {
            state.remaining -= 1;
            return Ok(value);
        }
        let old = state.owner.take();
        let frozen = state.frozen_owner.take();
        drop(state);
        drop(owner);
        drop(old);
        drop(frozen);
        Ok(None)
    }
}
impl MetadataValue for GeneralSet {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Err(Error::BindingGap {
            backend: "native_set".into(),
            operation: "set requires its original typed consumer, not a sequence conversion".into(),
        })
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| self.same_object(other))
    }
    fn type_name(&self) -> &str {
        "set"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(!self.is_empty())
    }
}
impl MetadataValue for FrozenSet {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Err(Error::BindingGap {
            backend: "native_frozenset".into(),
            operation: "frozenset requires its original typed consumer, not a sequence conversion"
                .into(),
        })
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| self.same_object(other))
    }
    fn type_name(&self) -> &str {
        "frozenset"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(!self.is_empty())
    }
}
