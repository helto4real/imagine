//! Native immutable Unicode objects with actual host identity. Code points,
//! including lone surrogates, are retained without UTF-8 replacement.
use super::{MetadataObject, MetadataValue, Options};
use crate::{execution_value::Value, native::Error};
use std::any::Any;

pub struct SourceText {
    points: Vec<u32>,
}
thread_local! {
    static EMPTY: Options = MetadataObject::wrap(SourceText { points: vec![] });
}
impl SourceText {
    /// Allocate a nonempty ordinary string object. Callers retain/clone its
    /// Options handle for source aliases, or implement their source interning
    /// recipe separately. Empty exact strings use the source singleton.
    pub fn from_utf8(text: &str) -> Options {
        if text.is_empty() {
            return EMPTY.with(Clone::clone);
        }
        MetadataObject::wrap(Self {
            points: text.chars().map(u32::from).collect(),
        })
    }
    pub fn from_codepoints(points: Vec<u32>) -> Result<Options, Error> {
        if points.iter().any(|value| *value > 0x10ffff) {
            return Err(Error::Native(
                "Unicode object contains an out-of-range code point".into(),
            ));
        }
        if points.is_empty() {
            return Ok(EMPTY.with(Clone::clone));
        }
        Ok(MetadataObject::wrap(Self { points }))
    }
    pub fn get(value: &Options) -> Option<&Self> {
        let Options::Object(value) = value else {
            return None;
        };
        MetadataObject::get(value.as_ref())
    }
    pub fn codepoints(&self) -> &[u32] {
        &self.points
    }
    pub fn utf8(&self) -> Option<String> {
        self.points.iter().map(|v| char::from_u32(*v)).collect()
    }
    pub fn identity_token(&self) -> u64 {
        self as *const Self as usize as u64
    }
}
impl MetadataValue for SourceText {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn snapshot(&self) -> Result<Value, Error> {
        self.utf8().map(Value::Text).ok_or_else(||Error::BindingGap {
            backend:"unicode_metadata".into(),
            operation:"lone surrogate requires the typed Unicode consumer at the native value boundary".into(),
        })
    }
    fn same_object(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<Self>().is_some_and(|other| {
            std::ptr::eq(self, other) || (self.points.is_empty() && other.points.is_empty())
        })
    }
    fn type_name(&self) -> &str {
        "str"
    }
    fn truth(&self) -> Result<bool, Error> {
        Ok(!self.points.is_empty())
    }
}
