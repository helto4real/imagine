//! Concrete mutable set for exact Unicode parameter names. Hash state belongs
//! to the execution owner; it is never initialized from the inference RNG.
use super::options::{Options, SourceText};
use crate::{execution_value::Value, native::Error, python_hash::HashState, source_set::Table};
use std::{cell::RefCell, rc::Rc};

pub struct PinnedKeySet {
    hashes: HashState,
    keys: RefCell<Table<TextKey>>,
}
struct TextKey {
    points: Vec<u32>,
    object: Options,
}
impl PartialEq for TextKey {
    fn eq(&self, other: &Self) -> bool {
        self.points == other.points
    }
}
fn points(key: &Options) -> Result<Vec<u32>, Error> {
    if let Some(key) = SourceText::get(key) {
        return Ok(key.codepoints().to_vec());
    }
    if let Options::Atom(Value::Text(key)) = key {
        return Ok(key.chars().map(u32::from).collect());
    }
    Err(Error::BindingGap {
        backend: "pinned_key_set".into(),
        operation: "non-exact-Unicode set key requires its source hash/equality protocol".into(),
    })
}
impl PinnedKeySet {
    pub fn new(hashes: HashState) -> Self {
        Self {
            hashes,
            keys: RefCell::new(Table::default()),
        }
    }
    pub fn into_options(self) -> Options {
        Options::Object(Rc::new(self))
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        value.downcast_ref()
    }
    pub fn contains(&self, key: &Options) -> Result<bool, Error> {
        let key = TextKey {
            points: points(key)?,
            object: key.clone(),
        };
        Ok(self
            .keys
            .borrow()
            .contains(self.hashes.hash_unicode(&key.points), &key, PartialEq::eq))
    }
    pub fn add(&self, key: &Options) -> Result<(), Error> {
        let key = TextKey {
            points: points(key)?,
            object: key.clone(),
        };
        self.keys
            .borrow_mut()
            .add(self.hashes.hash_unicode(&key.points), key, PartialEq::eq);
        Ok(())
    }
    pub fn discard(&self, key: &Options) -> Result<bool, Error> {
        let key = TextKey {
            points: points(key)?,
            object: key.clone(),
        };
        Ok(self.keys.borrow_mut().discard(
            self.hashes.hash_unicode(&key.points),
            &key,
            PartialEq::eq,
        ))
    }
    pub fn snapshot(&self) -> Result<Options, Error> {
        let keys = self
            .keys
            .borrow()
            .iter()
            .map(|key| key.object.clone())
            .collect::<Vec<_>>();
        Ok(Options::List(Rc::new(RefCell::new(keys))))
    }
}
