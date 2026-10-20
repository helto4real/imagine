//! Ordinary native ModelPatcher instance fields. Owner components share one
//! field store; retaining a view does not manufacture another model identity.
use super::{exception, options::Options};
use crate::native::Error;
use std::{cell::RefCell, rc::Rc};

struct Field {
    name: String,
    value: Option<Options>,
}
#[derive(Default)]
struct Data {
    fields: Vec<Field>,
    assigned: Vec<usize>,
}
struct Fields {
    type_name: String,
    data: RefCell<Data>,
}
#[derive(Clone)]
pub(crate) struct AttributeSlot {
    owner: Rc<Fields>,
    index: usize,
}
impl AttributeSlot {
    fn missing(&self) -> Error {
        exception(
            "AttributeError",
            format!(
                "'{}' object has no attribute '{}'",
                self.owner.type_name,
                self.owner.data.borrow().fields[self.index].name
            ),
        )
    }
    pub(crate) fn get(&self) -> Result<Options, Error> {
        let value = self.owner.data.borrow().fields[self.index].value.clone();
        value.ok_or_else(|| self.missing())
    }
    pub(crate) fn set(&self, value: Options) {
        let old = {
            let mut data = self.owner.data.borrow_mut();
            if data.fields[self.index].value.is_none() {
                data.assigned.push(self.index);
            }
            data.fields[self.index].value.replace(value)
        };
        // Publish the new field and release the borrow before old-value code.
        drop(old);
    }
    fn delete(&self) -> Result<(), Error> {
        let old = {
            let mut data = self.owner.data.borrow_mut();
            if data.fields[self.index].value.is_none() {
                drop(data);
                return Err(self.missing());
            }
            data.assigned.retain(|index| *index != self.index);
            data.fields[self.index].value.take()
        };
        drop(old);
        Ok(())
    }
}

/// A field-store view, not a second source owner or a shallow field/value copy.
/// Slots created before initialization remain unassigned until source SET_ATTR.
#[derive(Clone)]
pub struct AttributeStore(Rc<Fields>);
impl AttributeStore {
    pub fn new(type_name: &str) -> Self {
        Self(Rc::new(Fields {
            type_name: type_name.into(),
            data: RefCell::new(Data::default()),
        }))
    }
    pub(crate) fn slot(&self, name: &str) -> AttributeSlot {
        let mut data = self.0.data.borrow_mut();
        let index = if let Some(index) = data.fields.iter().position(|field| field.name == name) {
            index
        } else {
            let index = data.fields.len();
            data.fields.push(Field {
                name: name.into(),
                value: None,
            });
            index
        };
        AttributeSlot {
            owner: self.0.clone(),
            index,
        }
    }
    pub fn get(&self, name: &str) -> Result<Options, Error> {
        self.slot(name).get()
    }
    pub fn set(&self, name: &str, value: Options) {
        self.slot(name).set(value);
    }
    pub fn delete(&self, name: &str) -> Result<(), Error> {
        self.slot(name).delete()
    }
    /// Current source __dict__ insertion order, excluding unassigned slots.
    pub fn entries(&self) -> Vec<(String, Options)> {
        let data = self.0.data.borrow();
        data.assigned
            .iter()
            .map(|index| {
                let field = &data.fields[*index];
                (
                    field.name.clone(),
                    field.value.as_ref().expect("assigned native field").clone(),
                )
            })
            .collect()
    }
}
