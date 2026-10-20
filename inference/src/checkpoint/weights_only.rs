//! Locked torch._weights_only_unpickler stack machine. No executable Python.
//! Tensor reconstruction, approved global lookup and typed object operations
//! are explicit native bindings, not unrestricted pickle/JIT fallbacks.
use crate::{execution_value::Value, native::Error, patching::options::Options};
use std::{collections::HashMap, rc::Rc};

pub trait Bindings {
    fn global(&mut self, module: &[u8], name: &[u8]) -> Result<Options, Error>;
    fn new_object(&mut self, class: &Options, args: &Options) -> Result<Options, Error>;
    fn reduce(&mut self, function: &Options, args: &Options) -> Result<Options, Error>;
    fn build(&mut self, instance: &Options, state: &Options) -> Result<(), Error>;
    fn set_item(&mut self, target: &Options, key: &Options, value: Options) -> Result<(), Error>;
    fn is_set_item_target(&mut self, target: &Options) -> Result<bool, Error>;
    fn type_repr(&mut self, value: &Options) -> Result<String, Error>;
    fn empty_set(&mut self) -> Result<Options, Error>;
    fn unicode(&mut self, bytes: &[u8]) -> Result<Options, Error>;
    fn short_string(&mut self, bytes: &[u8], encoding: &str) -> Result<Options, Error>;
    fn long_integer(&mut self, bytes: &[u8]) -> Result<Options, Error>;
    fn persistent_load(&mut self, pid: &Options) -> Result<Options, Error>;
    fn is_exact_integer(&mut self, value: &Options) -> Result<bool, Error>;
    fn persistent_tag_is_storage(&mut self, value: &Options) -> Result<bool, Error>;
    fn warning(&mut self, message: &str) -> Result<(), Error>;
}
fn exception(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn unpickling(message: impl Into<String>) -> Error {
    exception("UnpicklingError", message)
}
fn atom(value: Value) -> Options {
    Options::Atom(value)
}
#[derive(Default)]
pub struct Memo {
    entries: Vec<(u32, Options)>,
    indices: HashMap<u32, usize>,
}
impl Memo {
    pub fn get(&self, index: &u32) -> Option<&Options> {
        self.indices.get(index).map(|i| &self.entries[*i].1)
    }
    pub fn insert(&mut self, index: u32, value: Options) {
        if let Some(i) = self.indices.get(&index) {
            self.entries[*i].1 = value;
        } else {
            self.indices.insert(index, self.entries.len());
            self.entries.push((index, value));
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &Options)> {
        self.entries.iter().map(|(key, value)| (key, value))
    }
}
/// Fields persist after failure. load resets stacks but preserves memo/proto,
/// as a second source load on the same Unpickler does.
pub struct Unpickler {
    pub memo: Memo,
    pub proto: i32,
    pub stack: Options,
    pub metastack: Vec<Options>,
    pub encoding: String,
}
impl Unpickler {
    pub fn new(encoding: impl Into<String>) -> Self {
        Self {
            memo: Memo::default(),
            proto: -1,
            stack: Options::List(Default::default()),
            metastack: vec![],
            encoding: encoding.into(),
        }
    }
    fn push(&self, value: Options) {
        let Options::List(stack) = &self.stack else {
            unreachable!("native stack representation")
        };
        stack.borrow_mut().push(value);
    }
    fn pop(&self) -> Result<Options, Error> {
        let Options::List(stack) = &self.stack else {
            unreachable!()
        };
        stack
            .borrow_mut()
            .pop()
            .ok_or_else(|| exception("IndexError", "pop from empty list"))
    }
    fn last(&self, back: usize) -> Result<Options, Error> {
        let Options::List(stack) = &self.stack else {
            unreachable!()
        };
        let stack = stack.borrow();
        stack
            .len()
            .checked_sub(back)
            .and_then(|index| stack.get(index))
            .cloned()
            .ok_or_else(|| exception("IndexError", "list index out of range"))
    }
    fn replace_last(&self, value: Options) -> Result<(), Error> {
        let Options::List(stack) = &self.stack else {
            unreachable!()
        };
        let mut stack = stack.borrow_mut();
        let slot = stack
            .last_mut()
            .ok_or_else(|| exception("IndexError", "list assignment index out of range"))?;
        *slot = value;
        Ok(())
    }
    fn pop_mark(&mut self) -> Result<Options, Error> {
        let previous = self
            .metastack
            .pop()
            .ok_or_else(|| exception("IndexError", "pop from empty list"))?;
        Ok(std::mem::replace(&mut self.stack, previous))
    }
    fn tuple_from_mark(&mut self) -> Result<(), Error> {
        let items = self.pop_mark()?;
        let Options::List(items) = items else {
            unreachable!()
        };
        self.push(Options::Tuple(Rc::new(items.borrow().clone())));
        Ok(())
    }
    fn tuple_n(&self, n: usize) -> Result<(), Error> {
        let values = (1..=n)
            .rev()
            .map(|i| self.last(i))
            .collect::<Result<Vec<_>, _>>()?;
        let Options::List(stack) = &self.stack else {
            unreachable!()
        };
        let mut stack = stack.borrow_mut();
        let start = stack.len() - n;
        stack.truncate(start);
        stack.push(Options::Tuple(Rc::new(values)));
        Ok(())
    }
    fn check_set_item(&self, bindings: &mut dyn Bindings, opcode: &str) -> Result<Options, Error> {
        let target = self.last(1)?;
        if !bindings.is_set_item_target(&target)? {
            return Err(unpickling(format!(
                "Can only {opcode} for dict, collections.OrderedDict, collections.Counter, but got {}",
                bindings.type_repr(&target)?
            )));
        }
        Ok(target)
    }
    /// A native byte stream is the admitted file representation. Its position
    /// remains observable after failures; no retry or rewind is introduced.
    pub fn load(
        &mut self,
        input: &mut ByteStream<'_>,
        bindings: &mut dyn Bindings,
    ) -> Result<Options, Error> {
        self.metastack.clear();
        self.stack = Options::List(Default::default());
        loop {
            let opcode = input
                .read(1)
                .first()
                .copied()
                .ok_or_else(|| exception("EOFError", ""))?;
            match opcode {
                b'c' => {
                    // readline()[:-1] discards the final byte even without LF.
                    let module = input.line_without_last();
                    let name = input.line_without_last();
                    self.push(bindings.global(module, name)?);
                }
                0x81 => {
                    let args = self.pop()?;
                    let class = self.pop()?;
                    self.push(bindings.new_object(&class, &args)?);
                }
                b'R' => {
                    let args = self.pop()?;
                    let function = self.last(1)?;
                    self.replace_last(bindings.reduce(&function, &args)?)?;
                }
                b'b' => {
                    let state = self.pop()?;
                    let instance = self.last(1)?;
                    bindings.build(&instance, &state)?;
                }
                b'a' => {
                    let item = self.pop()?;
                    let target = self.last(1)?;
                    let Options::List(values) = target else {
                        return Err(unpickling(format!(
                            "Can only append to lists, but got {}",
                            bindings.type_repr(&target)?
                        )));
                    };
                    values.borrow_mut().push(item);
                }
                b'e' => {
                    let items = self.pop_mark()?;
                    let target = self.last(1)?;
                    let Options::List(values) = target else {
                        return Err(unpickling(format!(
                            "Can only extend lists, but got {}",
                            bindings.type_repr(&target)?
                        )));
                    };
                    let Options::List(items) = items else {
                        unreachable!()
                    };
                    let items = items.borrow().clone();
                    values.borrow_mut().extend(items);
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let target = self.check_set_item(bindings, "SETITEM")?;
                    bindings.set_item(&target, &key, value)?;
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    let target = self.check_set_item(bindings, "SETITEMS")?;
                    let Options::List(items) = items else {
                        unreachable!()
                    };
                    let items = items.borrow().clone();
                    for index in (0..items.len()).step_by(2) {
                        let value = items
                            .get(index + 1)
                            .ok_or_else(|| exception("IndexError", "list index out of range"))?;
                        bindings.set_item(&target, &items[index], value.clone())?;
                    }
                }
                b'(' => {
                    self.metastack.push(std::mem::replace(
                        &mut self.stack,
                        Options::List(Default::default()),
                    ));
                }
                b't' => self.tuple_from_mark()?,
                0x85 => {
                    let value = self.last(1)?;
                    self.replace_last(Options::Tuple(Rc::new(vec![value])))?;
                }
                0x86 => self.tuple_n(2)?,
                0x87 => self.tuple_n(3)?,
                b'N' => self.push(atom(Value::None)),
                0x89 => self.push(atom(Value::Bool(false))),
                0x88 => self.push(atom(Value::Bool(true))),
                b')' => self.push(Options::Tuple(Rc::new(vec![]))),
                b']' => self.push(Options::List(Default::default())),
                b'}' => self.push(Options::dictionary(vec![])),
                0x8f => self.push(bindings.empty_set()?),
                b'J' => {
                    let bytes = input.fixed::<4>()?;
                    self.push(atom(Value::Int(i32::from_le_bytes(bytes).into())));
                }
                b'K' => {
                    let value = input.byte()?;
                    self.push(atom(Value::Int(value.into())));
                }
                b'M' => {
                    let bytes = input.fixed::<2>()?;
                    self.push(atom(Value::Int(u16::from_le_bytes(bytes).into())));
                }
                b'G' => {
                    let bytes = input.fixed::<8>()?;
                    self.push(atom(Value::Float(f64::from_be_bytes(bytes))));
                }
                b'X' => {
                    let length = u32::from_le_bytes(input.fixed::<4>()?) as usize;
                    self.push(bindings.unicode(input.read(length))?);
                }
                b'U' => {
                    let length = input.byte()? as usize;
                    self.push(bindings.short_string(input.read(length), &self.encoding)?);
                }
                b'Q' => {
                    let pid = self.pop()?;
                    if let Options::Tuple(values) = &pid {
                        if let Some(tag) = values.first()
                            && !bindings.persistent_tag_is_storage(tag)?
                        {
                            return Err(unpickling(format!(
                                "Only persistent_load of storage is allowed, but got {}",
                                bindings.type_repr(tag)?
                            )));
                        }
                    } else if !bindings.is_exact_integer(&pid)? {
                        return Err(unpickling(format!(
                            "persistent_load id must be tuple or int, but got {}",
                            bindings.type_repr(&pid)?
                        )));
                    }
                    self.push(bindings.persistent_load(&pid)?);
                }
                b'h' | b'j' => {
                    let index = if opcode == b'h' {
                        input.byte()?.into()
                    } else {
                        u32::from_le_bytes(input.fixed::<4>()?)
                    };
                    self.push(
                        self.memo
                            .get(&index)
                            .cloned()
                            .ok_or_else(|| exception("KeyError", index.to_string()))?,
                    );
                }
                b'q' | b'r' => {
                    let index = if opcode == b'q' {
                        input.byte()?.into()
                    } else {
                        u32::from_le_bytes(input.fixed::<4>()?)
                    };
                    let value = self.last(1)?;
                    self.memo.insert(index, value);
                }
                0x8a => {
                    let length = input.byte()? as usize;
                    self.push(bindings.long_integer(input.read(length))?);
                }
                0x80 => {
                    self.proto = input.byte()?.into();
                    if self.proto != 2 {
                        bindings.warning(&format!("Detected pickle protocol {} in the checkpoint, which was not the default pickle protocol used by `torch.load` (2). The weights_only Unpickler might not support all instructions implemented by this protocol, please file an issue for adding support if you encounter this.",self.proto))?;
                    }
                }
                b'.' => return self.pop(),
                other => return Err(unpickling(format!("Unsupported operand {other}"))),
            }
        }
    }
}
pub struct ByteStream<'a> {
    bytes: &'a [u8],
    pub position: usize,
}
impl<'a> ByteStream<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn read(&mut self, n: usize) -> &'a [u8] {
        let end = self.position.saturating_add(n).min(self.bytes.len());
        let output = &self.bytes[self.position..end];
        self.position = end;
        output
    }
    fn byte(&mut self) -> Result<u8, Error> {
        self.read(1)
            .first()
            .copied()
            .ok_or_else(|| exception("IndexError", "index out of range"))
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.read(N)
            .try_into()
            .map_err(|_| exception("error", format!("unpack requires a buffer of {N} bytes")))
    }
    fn line_without_last(&mut self) -> &'a [u8] {
        let length = self.bytes[self.position..]
            .iter()
            .position(|b| *b == b'\n')
            .map_or(self.bytes.len() - self.position, |n| n + 1);
        let line = self.read(length);
        &line[..line.len().saturating_sub(1)]
    }
}
