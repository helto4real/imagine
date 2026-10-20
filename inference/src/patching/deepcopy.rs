//! comfy.utils.deepcopy_list_dict, with original post-order memo insertion.
//! The shared execution owner supplies object/key identity and recursion policy.
use super::{gap, options::GeneralMapping, options::ItemsIterator, options::Options};
use crate::{execution_value::Value as V, native::Error};
use std::{cell::RefCell, rc::Rc};

/// Original plain id-keyed memo. Values retain their source object aliases;
/// insertion order survives replacements and partial failures.
#[derive(Default)]
pub struct CopyMemo {
    entries: Vec<(u64, Options)>,
}
impl CopyMemo {
    pub fn entries(&self) -> &[(u64, Options)] {
        &self.entries
    }
    pub fn get(&self, identity: u64) -> Option<Options> {
        self.entries
            .iter()
            .find(|(id, _)| *id == identity)
            .map(|(_, value)| value.clone())
    }
    pub fn set(&mut self, identity: u64, value: Options) {
        if let Some((_, old)) = self.entries.iter_mut().find(|(id, _)| *id == identity) {
            let old = std::mem::replace(old, value);
            drop(old);
        } else {
            self.entries.push((identity, value));
        }
    }
}

pub trait DeepCopyBindings {
    /// Python id, including primitive objects: value equality is insufficient.
    fn object_id(&self, value: &Options) -> Result<u64, Error>;
    /// The dictionary ABI stores plain string keys. Recover the actual owner
    /// of this key before calling the source helper on it, including memo hits.
    fn key_object(&self, dictionary: &Options, key: &str) -> Result<Options, Error>;
    fn enter_recursion(&self) -> Result<(), Error>;
    fn leave_recursion(&self);
}

enum Children {
    List(usize),
    Dict(ItemsIterator),
    None,
}
enum Pending {
    None,
    List,
    Key { value: Options },
    Value { key: Options },
}
struct Frame {
    input: Options,
    identity: u64,
    result: Options,
    children: Children,
    pending: Pending,
}

/// Plain dict/list body, lowered onto explicit frames like CPython 3.12's
/// interpreter loop. No early memo placeholder, tuple traversal, deepcopy of
/// tensors/callables, cycle repair, or replacement recursion limit is added.
pub fn deepcopy_list_dict(
    input: &Options,
    memo: Option<&mut CopyMemo>,
    bindings: &dyn DeepCopyBindings,
) -> Result<Options, Error> {
    let mut fresh = CopyMemo::default();
    let memo = memo.unwrap_or(&mut fresh);
    let mut frames = Vec::<Frame>::new();
    let mut active = 0usize;
    let outcome = (|| {
        let mut next = Some(input.clone());
        let mut returned = None;
        loop {
            if let Some(value) = next.take() {
                bindings.enter_recursion()?;
                active += 1;
                let identity = bindings.object_id(&value)?;
                if let Some(hit) = memo.get(identity) {
                    bindings.leave_recursion();
                    active -= 1;
                    returned = Some(hit);
                } else {
                    let (result, children) = match &value {
                        _ if GeneralMapping::get(&value).is_some() => (
                            GeneralMapping::empty(
                                GeneralMapping::get(&value).unwrap().bindings.clone(),
                            ),
                            Children::Dict(value.items_iterator()?),
                        ),
                        Options::Dict(_) => (
                            Options::dictionary(vec![]),
                            Children::Dict(value.items_iterator()?),
                        ),
                        Options::List(_) => (
                            Options::List(Rc::new(RefCell::new(vec![]))),
                            Children::List(0),
                        ),
                        _ => (value.clone(), Children::None),
                    };
                    frames.push(Frame {
                        input: value,
                        identity,
                        result,
                        children,
                        pending: Pending::None,
                    });
                }
            }
            if let Some(value) = returned.take() {
                let Some(parent) = frames.last_mut() else {
                    return Ok(value);
                };
                match std::mem::replace(&mut parent.pending, Pending::None) {
                    Pending::List => {
                        let Options::List(output) = &parent.result else {
                            unreachable!()
                        };
                        output.borrow_mut().push(value);
                    }
                    Pending::Key { value: child } => {
                        parent.pending = Pending::Value { key: value };
                        next = Some(child);
                        continue;
                    }
                    Pending::Value { key } if GeneralMapping::get(&parent.result).is_some() => {
                        let mapping = GeneralMapping::get(&parent.result).unwrap();
                        mapping
                            .dictionary
                            .set(key, value, mapping.bindings.as_ref())?;
                    }
                    Pending::Value { key } => match &key {
                        Options::Atom(V::Text(key)) => parent.result.set(key, value)?,
                        Options::List(_) | Options::Dict(_) => {
                            return Err(super::exception(
                                "TypeError",
                                format!("unhashable type: '{}'", key.typename()),
                            ));
                        }
                        _ => {
                            return Err(gap(
                                "deepcopy memo substituted a key requiring the general dictionary key binding",
                            ));
                        }
                    },
                    Pending::None => unreachable!("source child return"),
                }
            }
            let frame = frames.last_mut().expect("source active call");
            match &mut frame.children {
                Children::List(position) => {
                    let Options::List(input) = &frame.input else {
                        unreachable!()
                    };
                    let child = input.borrow().get(*position).cloned();
                    if let Some(child) = child {
                        *position += 1;
                        frame.pending = Pending::List;
                        next = Some(child);
                        continue;
                    }
                }
                Children::Dict(items) => {
                    if let Some(pair) = items.next_item()? {
                        let Options::Tuple(pair) = pair else {
                            unreachable!()
                        };
                        next = Some(if GeneralMapping::get(&frame.input).is_some() {
                            pair[0].clone()
                        } else {
                            let Options::Atom(V::Text(key)) = &pair[0] else {
                                unreachable!()
                            };
                            bindings.key_object(&frame.input, key)?
                        });
                        frame.pending = Pending::Key {
                            value: pair[1].clone(),
                        };
                        continue;
                    }
                }
                Children::None => (),
            }
            let frame = frames.pop().unwrap();
            memo.set(frame.identity, frame.result.clone());
            returned = Some(frame.result);
            bindings.leave_recursion();
            active -= 1;
        }
    })();
    // This balances the required context on every error, including failures
    // before a frame could be created. Retained tracebacks remain owner work.
    while active != 0 {
        bindings.leave_recursion();
        active -= 1;
    }
    outcome
}
