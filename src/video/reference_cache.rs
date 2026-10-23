//! RAM-only encoding cache core. The encoding owner supplies keys, payloads and actual sizes.
//! A cache instance must live at the real encoder seam; this is never a thumbnail counter.
use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
#[derive(Clone, Debug, Serialize)]
pub struct Status {
    pub available: bool,
    pub mode: &'static str,
    pub limit_bytes: u64,
    pub used_bytes: u64,
    pub entries: usize,
    pub busy: usize,
    pub generation: u64,
    pub hits: u64,
    pub misses: u64,
}
impl Status {
    pub fn unavailable() -> Self {
        Self {
            available: false,
            mode: "session",
            limit_bytes: 0,
            used_bytes: 0,
            entries: 0,
            busy: 0,
            generation: 0,
            hits: 0,
            misses: 0,
        }
    }
}
/// Identity follows encoder/preprocessing/content, never a semantic asset ID alone.
/// The owner includes tensor shape, sampling/window/crop policy and encoder compatibility
/// in `preprocessing`; relinking stable asset IDs must therefore invalidate old encodings.
pub fn encoding_key(kind: &str, encoder: &[u8], preprocessing: &[u8], content: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for part in [
        b"imagine-reference-encoding-v1".as_slice(),
        kind.as_bytes(),
        encoder,
        preprocessing,
        content,
    ] {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}
struct Entry<T> {
    value: Arc<T>,
    bytes: u64,
    touched: u64,
}
#[derive(Clone)]
pub struct Ticket {
    key: String,
    generation: u64,
    serial: u64,
}
pub struct Cache<T> {
    values: BTreeMap<String, Entry<T>>,
    pending: BTreeMap<String, u64>,
    limit: u64,
    used: u64,
    clock: u64,
    generation: u64,
    enabled: bool,
    hits: u64,
    misses: u64,
}
impl<T> Cache<T> {
    pub fn new(limit: u64) -> Self {
        Self {
            values: BTreeMap::new(),
            pending: BTreeMap::new(),
            limit,
            used: 0,
            clock: 0,
            generation: 0,
            enabled: true,
            hits: 0,
            misses: 0,
        }
    }
    pub fn status(&self) -> Status {
        Status {
            available: true,
            mode: if self.enabled { "session" } else { "off" },
            limit_bytes: self.limit,
            used_bytes: self.used,
            entries: self.values.len(),
            busy: self.pending.len(),
            generation: self.generation,
            hits: self.hits,
            misses: self.misses,
        }
    }
    pub fn get(&mut self, key: &str) -> Option<Arc<T>> {
        self.clock = self.clock.wrapping_add(1);
        let Some(e) = self.values.get_mut(key) else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        self.hits = self.hits.saturating_add(1);
        e.touched = self.clock;
        Some(e.value.clone())
    }
    pub fn begin(&mut self, key: String) -> Option<Ticket> {
        if !self.enabled || self.limit == 0 || self.pending.contains_key(&key) {
            return None;
        }
        self.clock = self.clock.wrapping_add(1);
        self.pending.insert(key.clone(), self.clock);
        Some(Ticket {
            key,
            generation: self.generation,
            serial: self.clock,
        })
    }
    pub fn abort(&mut self, ticket: &Ticket) {
        if self.current(ticket) {
            self.pending.remove(&ticket.key);
        }
    }
    fn current(&self, ticket: &Ticket) -> bool {
        ticket.generation == self.generation
            && self.pending.get(&ticket.key) == Some(&ticket.serial)
    }
    pub fn finish(&mut self, ticket: Ticket, value: Arc<T>, bytes: u64) -> bool {
        if !self.current(&ticket) {
            return false;
        }
        self.pending.remove(&ticket.key);
        if !self.enabled || bytes == 0 || bytes > self.limit {
            return false;
        }
        if let Some(old) = self.values.remove(&ticket.key) {
            self.used -= old.bytes;
        }
        while self.used > self.limit - bytes {
            self.evict();
        }
        self.clock = self.clock.wrapping_add(1);
        self.used += bytes;
        self.values.insert(
            ticket.key,
            Entry {
                value,
                bytes,
                touched: self.clock,
            },
        );
        true
    }
    fn evict(&mut self) {
        if let Some(key) = self
            .values
            .iter()
            .min_by_key(|(_, v)| v.touched)
            .map(|(k, _)| k.clone())
        {
            self.used -= self.values.remove(&key).unwrap().bytes;
        }
    }
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.values.clear();
        self.pending.clear();
        self.used = 0;
    }
    pub fn configure(&mut self, limit: u64, enabled: bool) {
        // Pending encodings belong to the previous policy and cannot repopulate after Apply.
        self.generation = self.generation.wrapping_add(1);
        self.pending.clear();
        self.limit = limit;
        self.enabled = enabled;
        if !enabled {
            self.values.clear();
            self.used = 0;
        } else {
            while self.used > limit {
                self.evict();
            }
        }
    }
}
/// In-process bridge only. A separate inference process needs the coordinated wire seam.
pub trait Control: Send + Sync {
    fn status(&self) -> Result<Status, &'static str>;
    fn configure(&self, limit: u64, enabled: bool) -> Result<Status, &'static str>;
    fn clear(&self) -> Result<Status, &'static str>;
}
impl<T: Send + Sync> Control for Mutex<Cache<T>> {
    fn status(&self) -> Result<Status, &'static str> {
        Ok(self
            .lock()
            .map_err(|_| "reference_cache_unavailable")?
            .status())
    }
    fn configure(&self, limit: u64, enabled: bool) -> Result<Status, &'static str> {
        let mut cache = self.lock().map_err(|_| "reference_cache_unavailable")?;
        cache.configure(limit, enabled);
        Ok(cache.status())
    }
    fn clear(&self) -> Result<Status, &'static str> {
        let mut cache = self.lock().map_err(|_| "reference_cache_unavailable")?;
        cache.clear();
        Ok(cache.status())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn insert(cache: &mut Cache<u8>, key: &str, bytes: u64) {
        let ticket = cache.begin(key.into()).unwrap();
        assert!(cache.finish(ticket, Arc::new(7), bytes));
    }
    #[test]
    fn reference_cache_identity_changes_with_content_encoder_and_preprocessing() {
        let original = encoding_key(
            "image",
            b"vae-1",
            b"shape:8x8;quality:standard",
            b"old pixels",
        );
        assert_eq!(
            original,
            encoding_key(
                "image",
                b"vae-1",
                b"shape:8x8;quality:standard",
                b"old pixels"
            )
        );
        assert_ne!(
            original,
            encoding_key(
                "image",
                b"vae-1",
                b"shape:8x8;quality:standard",
                b"relinked pixels"
            )
        );
        assert_ne!(
            original,
            encoding_key(
                "image",
                b"vae-2",
                b"shape:8x8;quality:standard",
                b"old pixels"
            )
        );
        assert_ne!(
            original,
            encoding_key(
                "image",
                b"vae-1",
                b"shape:16x4;quality:standard",
                b"old pixels"
            )
        );
    }
    #[test]
    fn reference_cache_lru_budget_and_active_lease() {
        let mut cache = Cache::new(10);
        insert(&mut cache, "a", 4);
        insert(&mut cache, "b", 4);
        let lease = cache.get("a").unwrap();
        insert(&mut cache, "c", 4);
        assert!(cache.get("b").is_none());
        assert!(cache.get("a").is_some());
        assert_eq!(cache.status().used_bytes, 8);
        cache.configure(4, true);
        assert_eq!(cache.status().used_bytes, 4);
        cache.clear();
        assert_eq!(*lease, 7);
        assert_eq!(cache.status().used_bytes, 0);
    }
    #[test]
    fn reference_cache_clear_and_policy_discard_inflight() {
        let mut c = Cache::new(8);
        let t = c.begin("a".into()).unwrap();
        assert!(c.begin("a".into()).is_none());
        c.clear();
        assert!(!c.finish(t, Arc::new(1), 4));
        let t = c.begin("a".into()).unwrap();
        c.configure(8, false);
        assert!(!c.finish(t, Arc::new(1), 4));
        assert!(c.begin("a".into()).is_none());
        c.configure(2, true);
        let t = c.begin("a".into()).unwrap();
        assert!(!c.finish(t, Arc::new(1), 3));
        assert_eq!(c.status().entries, 0);
    }
    #[test]
    fn reference_cache_abort_does_not_cancel_newer_ticket() {
        let mut c = Cache::<u8>::new(8);
        let old = c.begin("a".into()).unwrap();
        c.clear();
        let new = c.begin("a".into()).unwrap();
        c.abort(&old);
        assert_eq!(c.status().busy, 1);
        assert!(c.finish(new, Arc::new(1), 4));
    }
}
