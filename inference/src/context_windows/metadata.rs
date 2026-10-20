//! Source-ordered conditioning metadata lookups. Attribute access remains a
//! required typed binding, including repeated descriptor reads and exceptions.
use super::{Handler, error, gap};
use crate::{
    conditioning::Condition,
    execution_value::Value as V,
    native::Error,
    patching::options::{MetadataObject, Options},
};

pub trait ConditionAttributes {
    fn cond(&self, value: &Options) -> Result<Options, Error>;
}
pub struct BuiltinConditionAttributes;
impl ConditionAttributes for BuiltinConditionAttributes {
    fn cond(&self, value: &Options) -> Result<Options, Error> {
        if let Options::Object(object) = value {
            if object.is::<MetadataObject>() {
                return Err(error(
                    "AttributeError",
                    format!("'{}' object has no attribute 'cond'", value.typename()),
                ));
            }
            return object
                .downcast_ref::<Condition>()
                .map(Condition::source_value)
                .ok_or_else(|| {
                    gap("context metadata requires the object's original attribute binding")
                });
        }
        Err(error(
            "AttributeError",
            format!("'{}' object has no attribute 'cond'", value.typename()),
        ))
    }
}
pub(super) fn next(value: &Options, position: &mut usize) -> Result<Option<Options>, Error> {
    let value = match value {
        Options::List(v) => v.borrow().get(*position).cloned(),
        Options::Tuple(v) => v.get(*position).cloned(),
        _ => {
            return Err(error(
                "TypeError",
                format!("'{}' object is not iterable", value.typename()),
            ));
        }
    };
    if value.is_some() {
        *position += 1;
    }
    Ok(value)
}
fn is_none(value: &Options) -> bool {
    matches!(value, Options::Atom(V::None))
}
pub(super) fn guide_count(entries: &Options) -> Result<i64, Error> {
    if is_none(entries) {
        return Ok(0);
    }
    let mut position = 0;
    let mut count = 0i64;
    while let Some(entry) = next(entries, &mut position)? {
        if !matches!(entry, Options::Dict(_)) {
            return Err(gap(
                "guide metadata requires its original entry subscription binding",
            ));
        }
        let shape = entry
            .get("latent_shape")?
            .ok_or_else(|| error("KeyError", "'latent_shape'"))?;
        let amount = super::LatentShape::from_options(shape)?.dimension(0)?;
        count = count
            .checked_add(amount)
            .ok_or_else(|| gap("context guide count exceeds i64"))?;
    }
    Ok(count)
}
fn truth(value: &Options) -> Result<bool, Error> {
    if let Some(value) = value.metadata_truth() {
        return value;
    }
    Ok(match value {
        Options::Atom(V::None) => false,
        Options::Atom(V::Bool(v)) => *v,
        Options::Atom(V::Int(v)) => *v != 0,
        Options::Atom(V::UInt(v)) => *v != 0,
        Options::Atom(V::Float(v)) => *v != 0.,
        Options::Atom(V::Text(v)) => !v.is_empty(),
        Options::Atom(v @ V::Tensor(_)) => v.as_tensor()?.f_is_nonzero()?,
        Options::Dict(v) => !v.borrow().is_empty(),
        Options::List(v) => !v.borrow().is_empty(),
        Options::Tuple(v) => !v.is_empty(),
        Options::Object(_) => {
            return Err(gap(
                "context metadata truth requires the object's original truth binding",
            ));
        }
        _ => {
            return Err(gap(
                "context metadata truth requires its original value binding",
            ));
        }
    })
}
#[derive(Clone, Copy)]
enum Field {
    Shapes,
    Guides,
    Keyframes,
}
fn shapes_field(model: &Options) -> Result<Option<Options>, Error> {
    let key = "latent_shapes";
    match model {
        Options::Dict(_) => model.get(key),
        Options::List(_) | Options::Tuple(_) => {
            let mut position = 0;
            while let Some(value) = next(model, &mut position)? {
                match value {
                    Options::Atom(V::Text(value)) if value == key => {
                        return Err(error(
                            "TypeError",
                            format!(
                                "{} indices must be integers or slices, not str",
                                model.typename()
                            ),
                        ));
                    }
                    Options::Object(_) | Options::Atom(V::Tensor(_) | V::Quantized(_)) => {
                        return Err(gap(
                            "context shape-key membership requires the value's original equality binding",
                        ));
                    }
                    _ => {}
                }
            }
            Ok(None)
        }
        Options::Atom(V::Text(value)) => {
            if value.contains(key) {
                Err(error(
                    "TypeError",
                    "string indices must be integers, not 'str'",
                ))
            } else {
                Ok(None)
            }
        }
        Options::Atom(V::None | V::Bool(_) | V::Int(_) | V::UInt(_) | V::Float(_)) => Err(error(
            "TypeError",
            format!("argument of type '{}' is not iterable", model.typename()),
        )),
        _ => Err(gap(
            "context shape-key membership requires its original object binding",
        )),
    }
}
fn lookup(
    input: &Options,
    attributes: &dyn ConditionAttributes,
    field: Field,
) -> Result<Options, Error> {
    let mut outer = 0;
    while let Some(row) = next(input, &mut outer)? {
        if is_none(&row) {
            continue;
        }
        let mut inner = 0;
        while let Some(entry) = next(&row, &mut inner)? {
            let model = entry
                .get("model_conds")?
                .unwrap_or_else(|| Options::dictionary(vec![]));
            let key = match field {
                Field::Shapes => "latent_shapes",
                Field::Guides => "guide_attention_entries",
                Field::Keyframes => "keyframe_idxs",
            };
            let Some(value) = (if matches!(field, Field::Shapes) {
                shapes_field(&model)
            } else {
                model.get(key)
            })?
            else {
                continue;
            };
            if matches!(field, Field::Shapes) {
                return attributes.cond(&value);
            }
            if is_none(&value) {
                continue;
            }
            match attributes.cond(&value) {
                Err(caught) if caught.is_reference("AttributeError") => continue,
                Err(e) => return Err(e),
                Ok(v) => drop(v), // hasattr discards the successful attribute value.
            }
            let predicate = attributes.cond(&value)?;
            let use_value = if matches!(field, Field::Guides) {
                truth(&predicate)?
            } else {
                !is_none(&predicate)
            };
            drop(predicate);
            if use_value {
                return attributes.cond(&value);
            }
        }
    }
    Ok(Options::Atom(V::None))
}
impl Handler {
    pub fn get_latent_shapes(
        input: &Options,
        attributes: &dyn ConditionAttributes,
    ) -> Result<Options, Error> {
        lookup(input, attributes, Field::Shapes)
    }
    pub fn get_guide_entries(
        input: &Options,
        attributes: &dyn ConditionAttributes,
    ) -> Result<Options, Error> {
        lookup(input, attributes, Field::Guides)
    }
    pub fn get_keyframe_indices(
        input: &Options,
        attributes: &dyn ConditionAttributes,
    ) -> Result<Options, Error> {
        lookup(input, attributes, Field::Keyframes)
    }
}
