//! Exact bytes payload owners and mutable bytearray payloads for checkpoint
//! objects. Constructors distinguish non-NULL copy/cached characters from a
//! fresh allocated buffer. General buffer exports and raw constructor coercion
//! remain caller bindings; snapshots never turn bytes into a Python list/string.
use super::{MetadataObject, MetadataValue, Options};
use crate::{execution_value::Value, native::Error};
use std::{
    any::Any,
    cell::{Ref, RefCell},
};
pub struct SourceBytes {
    data: Vec<u8>,
}
pub struct SourceByteArray {
    data: RefCell<Vec<u8>>,
}
thread_local! {
 static EMPTY:Options=MetadataObject::wrap(SourceBytes{data:vec![]});
 static CHARS:Vec<Options>=(0_u8..=255).map(|b|MetadataObject::wrap(SourceBytes{data:vec![b]})).collect();
}
impl SourceBytes {
    /// PyBytes_FromStringAndSize(non_null_data, size).
    pub fn from_data(data: &[u8]) -> Options {
        match data {
            [] => EMPTY.with(Clone::clone),
            [b] => CHARS.with(|v| v[usize::from(*b)].clone()),
            _ => MetadataObject::wrap(Self {
                data: data.to_vec(),
            }),
        }
    }
    /// An already-filled fresh buffer created by the NULL-data allocation path.
    pub fn from_allocated(data: Vec<u8>) -> Options {
        if data.is_empty() {
            EMPTY.with(Clone::clone)
        } else {
            MetadataObject::wrap(Self { data })
        }
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(v) = value else {
            return None;
        };
        MetadataObject::get(v.as_ref())
    }
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    pub fn repr(&self) -> String {
        repr_bytes(&self.data, false)
    }
}
impl SourceByteArray {
    pub fn from_data(data: Vec<u8>) -> Options {
        MetadataObject::wrap(Self {
            data: RefCell::new(data),
        })
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(v) = value else {
            return None;
        };
        MetadataObject::get(v.as_ref())
    }
    pub fn bytes(&self) -> Ref<'_, [u8]> {
        Ref::map(self.data.borrow(), Vec::as_slice)
    }
    /// Typed owned-buffer update after source validation/export checks. It does
    /// not implement a raw bytearray slice/index/buffer-protocol operation.
    pub fn replace_owned_buffer(&self, data: Vec<u8>) {
        let old = self.data.replace(data);
        drop(old);
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
    pub fn repr(&self) -> String {
        format!("bytearray({})", repr_bytes(&self.data.borrow(), true))
    }
}
fn gap(kind: &str) -> Error {
    Error::BindingGap {
        backend: "byte_metadata".into(),
        operation: format!("{kind} requires its typed native consumer"),
    }
}
impl MetadataValue for SourceBytes {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Err(gap("bytes"))
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|v| std::ptr::eq(self, v))
    }
    fn type_name(&self) -> &str {
        "bytes"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(!self.data.is_empty())
    }
}
impl MetadataValue for SourceByteArray {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        Err(gap("bytearray"))
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|v| std::ptr::eq(self, v))
    }
    fn type_name(&self) -> &str {
        "bytearray"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(!self.data.borrow().is_empty())
    }
}
fn repr_bytes(bytes: &[u8], bytearray: bool) -> String {
    use std::fmt::Write;
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut text = String::from("b");
    text.push(char::from(quote));
    for &byte in bytes {
        match byte {
            b'\\' => text.push_str("\\\\"),
            b'\t' => text.push_str("\\t"),
            b'\n' => text.push_str("\\n"),
            b'\r' => text.push_str("\\r"),
            b if b == if bytearray { b'\'' } else { quote } => {
                text.push('\\');
                text.push(char::from(b));
            }
            b if !(32..127).contains(&b) => write!(text, "\\x{b:02x}").unwrap(),
            b => text.push(char::from(b)),
        }
    }
    text.push(char::from(quote));
    text
}
