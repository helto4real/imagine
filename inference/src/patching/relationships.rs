//! ModelPatcher attachments and additional-model relationships. These fields
//! belong to the shared model owner; model calls/identity/recursion stay bound.
use super::{
    attributes::{AttributeSlot, AttributeStore},
    exception, gap,
    options::Options,
    options_iterator::SequenceIterator,
};
use crate::{execution_value::Value as V, native::Error, source_set::Table};
use std::{cell::RefCell, rc::Rc};
fn empty_dict() -> Options {
    Options::dictionary(vec![])
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn append(values: &Options, value: Options) {
    let Options::List(values) = values else {
        unreachable!("source new list")
    };
    values.borrow_mut().push(value);
}
pub struct ModelRelationships {
    attachments: AttributeSlot,
    additional_models: AttributeSlot,
}
#[derive(Default)]
pub struct RelationshipCloneLocals {
    pub key: Option<Options>,
    pub group: Option<Options>,
}
impl Default for ModelRelationships {
    fn default() -> Self {
        Self::new()
    }
}
impl ModelRelationships {
    pub fn new() -> Self {
        let fields = AttributeStore::new("ModelPatcher");
        let out = Self::bind(&fields);
        out.replace_attachments(empty_dict());
        out.replace_additional_models(empty_dict());
        out
    }
    pub(crate) fn bind(fields: &AttributeStore) -> Self {
        Self {
            attachments: fields.slot("attachments"),
            additional_models: fields.slot("additional_models"),
        }
    }
    pub fn attachments(&self) -> Result<Options, Error> {
        self.attachments.get()
    }
    pub fn additional_models(&self) -> Result<Options, Error> {
        self.additional_models.get()
    }
    pub fn replace_attachments(&self, value: Options) {
        self.attachments.set(value);
    }
    pub fn replace_additional_models(&self, value: Options) {
        self.additional_models.set(value);
    }
    pub fn set_attachments(&self, key: &str, value: Options) -> Result<(), Error> {
        self.attachments()?.set(key, value)
    }
    pub fn remove_attachments(&self, key: &str) -> Result<(), Error> {
        drop(self.attachments()?.pop(key)?);
        Ok(())
    }
    pub fn get_attachment(&self, key: &str) -> Result<Options, Error> {
        Ok(self
            .attachments()?
            .get(key)?
            .unwrap_or(Options::Atom(V::None)))
    }
    pub fn set_additional_models(&self, key: &str, value: Options) -> Result<(), Error> {
        self.additional_models()?.set(key, value)
    }
    pub fn remove_additional_models(&self, key: &str) -> Result<(), Error> {
        drop(self.additional_models()?.pop(key)?);
        Ok(())
    }
    pub fn get_additional_models_with_key(&self, key: &str) -> Result<Options, Error> {
        Ok(self
            .additional_models()?
            .get(key)?
            .unwrap_or_else(|| list(vec![])))
    }
    pub fn get_additional_models(&self) -> Result<Options, Error> {
        let out = list(vec![]);
        let values = self.additional_models()?.values_iterator()?;
        while let Some(group) = values.next_value()? {
            let mut items = SequenceIterator::for_operation(&group, "additional model sequence")?;
            while let Some(item) = items.next()? {
                append(&out, item);
            }
        }
        Ok(out)
    }
    /// The attachments/additional-model portion of ModelPatcher.clone. The
    /// whole clone owner must invoke this at its original position; it is not
    /// a substitute for constructor, callbacks, hook/backup or parent copying.
    pub fn clone_members_into(
        &self,
        target: &Self,
        bindings: &dyn RelationshipCloneBindings,
    ) -> Result<(), Error> {
        self.clone_members_into_with_locals(
            target,
            bindings,
            &mut RelationshipCloneLocals::default(),
        )
    }
    pub fn clone_members_into_with_locals(
        &self,
        target: &Self,
        bindings: &dyn RelationshipCloneBindings,
        locals: &mut RelationshipCloneLocals,
    ) -> Result<(), Error> {
        target.replace_attachments(bindings.empty_dictionary()?);
        let keys = self.attachments()?.keys_iterator()?;
        while let Some(key) = keys.next_value()? {
            drop(locals.key.replace(key));
            let key = locals.key.as_ref().unwrap();
            let value = mapping_item(&self.attachments()?, key)?;
            let attribute = bindings.clone_attribute(&value);
            let has_method = match attribute {
                Ok(attribute) => {
                    drop(attribute);
                    true
                }
                Err(caught) if caught.is_reference("AttributeError") => false,
                Err(error) => return Err(error),
            };
            // hasattr releases its descriptor result before the call's owner
            // argument. A descriptor can replace the root's last owner.
            drop(value);
            let value = if has_method {
                // hasattr does not retain/reuse the descriptor's first result.
                let value = mapping_item(&self.attachments()?, key)?;
                let method = bindings.clone_attribute(&value);
                drop(value);
                let method = method?;
                let result = bindings.call_clone_attribute(&method);
                drop(method);
                result?
            } else {
                mapping_item(&self.attachments()?, key)?
            };
            target.attachments()?.set_item(key.clone(), value)?;
        }
        drop(keys);
        let items = self.additional_models()?.items_iterator()?;
        while let Some(pair) = items.next_item()? {
            let Options::Tuple(pair) = pair else {
                unreachable!()
            };
            let key = pair[0].clone();
            let group = pair[1].clone();
            drop(pair);
            drop(locals.key.replace(key));
            drop(locals.group.replace(group));
            let output = list(vec![]);
            let mut models = SequenceIterator::for_operation(
                locals.group.as_ref().unwrap(),
                "model clone additional group",
            )?;
            let mut current_model = None;
            while let Some(model) = models.next()? {
                drop(current_model.replace(model));
                append(
                    &output,
                    bindings.clone_model(current_model.as_ref().unwrap())?,
                );
            }
            drop(current_model);
            drop(models);
            // Evaluate the entire comprehension before loading target's root.
            target
                .additional_models()?
                .set_item(locals.key.as_ref().unwrap().clone(), output)?;
        }
        Ok(())
    }
}
pub trait RelationshipCloneBindings {
    fn empty_dictionary(&self) -> Result<Options, Error>;
    /// Perform the actual descriptor/attribute access, including AttributeError.
    fn clone_attribute(&self, attachment: &Options) -> Result<Options, Error>;
    fn call_clone_attribute(&self, method: &Options) -> Result<Options, Error>;
    /// Includes clone attribute lookup, binding and call for an additional owner.
    fn clone_model(&self, model: &Options) -> Result<Options, Error>;
}
fn mapping_item(mapping: &Options, key: &Options) -> Result<Options, Error> {
    if !mapping.is_dictionary() {
        return Err(gap(
            "model clone relationship root requires its mapping binding",
        ));
    }
    mapping.item(key)
}
pub trait AdditionalModelBindings {
    fn additional_models(&self, model: &Options) -> Result<Options, Error>;
    fn model_hash(&self, model: &Options) -> Result<i64, Error>;
    /// Canonical ModelPatcher objects inherit identity equality. Native wrappers
    /// must resolve to that owner, as in the control-owner binding.
    fn same_model(&self, a: &Options, b: &Options) -> bool;
    fn enter_model_recursion(&self) -> Result<(), Error>;
    fn leave_model_recursion(&self);
    fn source_locals(&self, _: &AdditionalModelLocals, _: &str) -> Result<(), Error> {
        Ok(())
    }
}
pub struct AdditionalModelFrame {
    pub previous: Options,
    pub next: Options,
    pub model: Option<Options>,
    pub candidates: Option<Options>,
    pub candidate: Option<Options>,
}
#[derive(Default)]
pub struct AdditionalModelLocals {
    pub all_models: Option<Options>,
    pub result: Option<Options>,
    pub frames: Vec<AdditionalModelFrame>,
    cache: Table<Options>,
}
impl AdditionalModelLocals {
    pub fn cached_models(&self) -> impl Iterator<Item = &Options> {
        self.cache.iter()
    }
}
pub fn get_nested_additional_models(
    owner: &Options,
    bindings: &dyn AdditionalModelBindings,
) -> Result<Options, Error> {
    let mut locals = AdditionalModelLocals::default();
    let outcome = (|| {
        bindings.source_locals(&locals, "root")?;
        let models = bindings.additional_models(owner)?;
        locals.all_models = Some(models.clone());
        let mut items = SequenceIterator::for_operation(&models, "additional model root sequence")?;
        while let Some(model) = items.next()? {
            let hash = bindings.model_hash(&model)?;
            locals
                .cache
                .add(hash, model, |a, b| bindings.same_model(a, b));
        }
        drop(items);
        cascade(models, bindings, &mut locals)
    })();
    if let Ok(result) = &outcome {
        locals.result = Some(result.clone());
    }
    bindings.source_locals(&locals, if outcome.is_ok() { "return" } else { "error" })?;
    outcome
}
// Python 3.12 evaluates these Python frames without recursive native calls.
// Retain the original frame/recursion-accounting and concatenate-on-return
// order on an explicit native frame stack, not the Rust thread stack.
fn cascade(
    mut previous: Options,
    bindings: &dyn AdditionalModelBindings,
    locals: &mut AdditionalModelLocals,
) -> Result<Options, Error> {
    let initial_depth = locals.frames.len();
    let outcome = (|| {
        loop {
            bindings.enter_model_recursion()?;
            locals.frames.push(AdditionalModelFrame {
                previous: previous.clone(),
                next: list(vec![]),
                model: None,
                candidates: None,
                candidate: None,
            });
            let index = locals.frames.len() - 1;
            evaluate_level(&previous, index, bindings, locals)?;
            let next = locals.frames[index].next.clone();
            let empty = if let Options::List(values) = &next {
                values.borrow().is_empty()
            } else {
                unreachable!()
            };
            if empty {
                drop(next);
                let mut result = previous;
                loop {
                    drop(locals.frames.pop().unwrap());
                    bindings.leave_model_recursion();
                    if locals.frames.len() == initial_depth {
                        return Ok(result);
                    }
                    result = concatenate(&locals.frames.last().unwrap().previous, &result)?;
                }
            }
            previous = next;
        }
    })();
    while locals.frames.len() > initial_depth {
        drop(locals.frames.pop().unwrap());
        bindings.leave_model_recursion();
    }
    outcome
}
fn evaluate_level(
    previous: &Options,
    index: usize,
    bindings: &dyn AdditionalModelBindings,
    locals: &mut AdditionalModelLocals,
) -> Result<(), Error> {
    let mut models = SequenceIterator::for_operation(previous, "additional model cascade")?;
    while let Some(model) = models.next()? {
        let old = locals.frames[index].model.replace(model.clone());
        drop(old);
        bindings.source_locals(locals, "model")?;
        let candidates = bindings.additional_models(&model)?;
        let old = locals.frames[index].candidates.replace(candidates.clone());
        drop(old);
        let mut candidates =
            SequenceIterator::for_operation(&candidates, "additional model candidates")?;
        while let Some(candidate) = candidates.next()? {
            let old = locals.frames[index].candidate.replace(candidate.clone());
            drop(old);
            let hash = bindings.model_hash(&candidate)?;
            if !locals
                .cache
                .contains(hash, &candidate, |a, b| bindings.same_model(a, b))
            {
                append(&locals.frames[index].next, candidate.clone());
                // Source membership and add hash independently, after append.
                let hash = bindings.model_hash(&candidate)?;
                locals
                    .cache
                    .add(hash, candidate, |a, b| bindings.same_model(a, b));
            }
        }
    }
    Ok(())
}
fn concatenate(previous: &Options, more: &Options) -> Result<Options, Error> {
    match (previous, more) {
        (Options::List(a), Options::List(b)) => {
            let mut out = a.borrow().clone();
            out.extend(b.borrow().iter().cloned());
            Ok(list(out))
        }
        (Options::Tuple(a), Options::Tuple(b)) => {
            let mut out = (**a).clone();
            out.extend(b.iter().cloned());
            Ok(Options::Tuple(Rc::new(out)))
        }
        (Options::List(_), other) => Err(exception(
            "TypeError",
            format!(
                "can only concatenate list (not \"{}\") to list",
                other.typename()
            ),
        )),
        (Options::Tuple(_), other) => Err(exception(
            "TypeError",
            format!(
                "can only concatenate tuple (not \"{}\") to tuple",
                other.typename()
            ),
        )),
        _ => Err(gap(
            "model cascade concatenation requires original native value binding",
        )),
    }
}
