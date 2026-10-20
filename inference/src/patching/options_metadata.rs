//! Registered shared metadata objects. The object and value boundaries are
//! distinct: aliases retain native owners, while snapshots preserve value type.
use super::Options;
use crate::{execution_value::Value, native::Error};
use std::{any::Any, rc::Rc};
pub(crate) trait MetadataValue: Any {
    fn as_any(&self) -> &dyn Any;
    fn snapshot(&self) -> Result<Value, Error>;
    fn same_object(&self, other: &dyn Any) -> bool;
    fn type_name(&self) -> &str;
    fn truth(&self) -> Result<bool, Error>;
}
pub(crate) struct MetadataObject(Box<dyn MetadataValue>);
impl MetadataObject {
    pub(crate) fn wrap(value: impl MetadataValue) -> Options {
        Options::Object(Rc::new(Self(Box::new(value))))
    }
    pub(crate) fn get<T: Any>(object: &dyn Any) -> Option<&T> {
        object.downcast_ref::<Self>()?.0.as_any().downcast_ref()
    }
    pub(crate) fn snapshot(&self) -> Result<Value, Error> {
        self.0.snapshot()
    }
    pub(crate) fn same_object(&self, other: &dyn Any) -> bool {
        self.0
            .same_object(other.downcast_ref::<Self>().map_or(other, |v| v.0.as_any()))
    }
    pub(crate) fn type_name(&self) -> &str {
        self.0.type_name()
    }
    pub(crate) fn truth(&self) -> Result<bool, Error> {
        self.0.truth()
    }
}
