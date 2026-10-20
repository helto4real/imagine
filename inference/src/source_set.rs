//! CPython 3.12.13 hash tables for exact keys shared by native consumers.
//! Keys supply their original hash/equality; no Rust HashSet policy is used.
struct Entry<T> {
    hash: i64,
    key: T,
}
pub(crate) struct Table<T> {
    slots: Vec<Option<Entry<T>>>,
    used: usize,
    fill: usize,
    dummy: Vec<bool>,
}
impl<T> Default for Table<T> {
    fn default() -> Self {
        Self {
            slots: (0..8).map(|_| None).collect(),
            used: 0,
            fill: 0,
            dummy: vec![false; 8],
        }
    }
}
impl<T> Table<T> {
    pub fn contains(&self, hash: i64, key: &T, same: impl Fn(&T, &T) -> bool) -> bool {
        let mask = self.slots.len() - 1;
        let mut perturb = hash as usize;
        let mut i = perturb & mask;
        loop {
            let probes = if i + 9 <= mask { 9 } else { 0 };
            for slot in i..=i + probes {
                match &self.slots[slot] {
                    None if !self.dummy[slot] => return false,
                    None => {}
                    Some(old) if old.hash == hash && same(&old.key, key) => return true,
                    Some(_) => {}
                }
            }
            perturb >>= 5;
            i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
        }
    }
    pub fn add(&mut self, hash: i64, key: T, same: impl Fn(&T, &T) -> bool) {
        let mask = self.slots.len() - 1;
        let mut perturb = hash as usize;
        let mut i = perturb & mask;
        let mut free = None;
        loop {
            let probes = if i + 9 <= mask { 9 } else { 0 };
            for slot in i..=i + probes {
                match &self.slots[slot] {
                    Some(old) if old.hash == hash && same(&old.key, &key) => return,
                    Some(_) => {}
                    None if self.dummy[slot] => free = Some(slot),
                    None => {
                        let target = free.unwrap_or(slot);
                        self.slots[target] = Some(Entry { hash, key });
                        self.dummy[target] = false;
                        self.used += 1;
                        if free.is_none() {
                            self.fill += 1;
                            if self.fill * 5 >= mask * 3 {
                                self.resize();
                            }
                        }
                        return;
                    }
                }
            }
            perturb >>= 5;
            i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
        }
    }
    /// Exact-key set_discard_entry. Publishes the dummy before releasing the key.
    /// Callers with user equality/reentrancy require the general object protocol.
    pub fn discard(&mut self, hash: i64, key: &T, same: impl Fn(&T, &T) -> bool) -> bool {
        let mask = self.slots.len() - 1;
        let mut perturb = hash as usize;
        let mut i = perturb & mask;
        loop {
            let probes = if i + 9 <= mask { 9 } else { 0 };
            for slot in i..=i + probes {
                match &self.slots[slot] {
                    None if !self.dummy[slot] => return false,
                    Some(old) if old.hash == hash && same(&old.key, key) => {
                        self.dummy[slot] = true;
                        self.used -= 1;
                        drop(self.slots[slot].take());
                        return true;
                    }
                    _ => {}
                }
            }
            perturb >>= 5;
            i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
        }
    }
    fn resize(&mut self) {
        // set_table_resize: strictly greater than minused; reinsertion traverses
        // old table slots, not input order. The 50,000 switch is source policy.
        let minimum = self.used * if self.used > 50_000 { 2 } else { 4 };
        self.resize_min(minimum);
    }
    fn resize_min(&mut self, minimum: usize) {
        let mut size = 8;
        while size <= minimum {
            size *= 2;
        }
        let old = std::mem::replace(&mut self.slots, (0..size).map(|_| None).collect());
        self.dummy = vec![false; size];
        self.fill = self.used;
        let mask = size - 1;
        for entry in old.into_iter().flatten() {
            let mut perturb = entry.hash as usize;
            let mut i = perturb & mask;
            'probe: loop {
                let probes = if i + 9 <= mask { 9 } else { 0 };
                for slot in i..=i + probes {
                    if self.slots[slot].is_none() {
                        self.slots[slot] = Some(entry);
                        break 'probe;
                    }
                }
                perturb >>= 5;
                i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
            }
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().flatten().map(|v| &v.key)
    }
    pub fn entries(&self) -> impl Iterator<Item = (i64, &T)> {
        self.slots.iter().flatten().map(|v| (v.hash, &v.key))
    }
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    pub fn next(&self, position: &mut usize) -> Option<&T> {
        while *position < self.slots.len() {
            let current = *position;
            *position += 1;
            if let Some(entry) = &self.slots[current] {
                return Some(&entry.key);
            }
        }
        None
    }
}
impl<T: Clone> Table<T> {
    /// BUILD_SET 0; SET_UPDATE of an immutable, dummy-free table. This is the
    /// empty-destination branch of set_merge, not repeated ordinary set.add.
    pub fn from_frozen(other: &Self) -> Self {
        assert_eq!(
            other.fill, other.used,
            "from_frozen requires a dummy-free table"
        );
        let mut out = Self::default();
        if other.used == 0 {
            return out;
        }
        if other.used * 5 >= (out.slots.len() - 1) * 3 {
            out.resize_min(other.used * 2);
        }
        if out.slots.len() == other.slots.len() {
            out.slots = other
                .slots
                .iter()
                .map(|v| {
                    v.as_ref().map(|v| Entry {
                        hash: v.hash,
                        key: v.key.clone(),
                    })
                })
                .collect();
        } else {
            let mask = out.slots.len() - 1;
            for (hash, key) in other.entries() {
                let mut perturb = hash as usize;
                let mut i = perturb & mask;
                'probe: loop {
                    let probes = if i + 9 <= mask { 9 } else { 0 };
                    for slot in i..=i + probes {
                        if out.slots[slot].is_none() {
                            out.slots[slot] = Some(Entry {
                                hash,
                                key: key.clone(),
                            });
                            break 'probe;
                        }
                    }
                    perturb >>= 5;
                    i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
                }
            }
        }
        out.used = other.used;
        out.fill = other.used;
        out
    }
}
