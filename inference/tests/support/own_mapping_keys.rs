//! Native protocols for own metadata fixtures. This is not a production key
//! provider or an inference hash-state initialization policy.
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::{
        DictionaryKeyBindings, GeneralMapping, MappingKeyBindings, Options, SourceText,
    },
    python_hash::{HashState, hash_tuple, object_identity_hash},
};
use std::rc::Rc;

pub struct OwnMappingKeys(pub HashState);
impl OwnMappingKeys {
    pub fn seeded() -> Rc<Self> {
        Rc::new(Self(HashState::from_seed(0)))
    }
    pub fn empty(self: &Rc<Self>) -> Options {
        GeneralMapping::empty(self.clone())
    }
}
impl DictionaryKeyBindings for OwnMappingKeys {
    fn hash(&self, value: &Options) -> Result<i64, Error> {
        if let Some(text) = SourceText::get(value) {
            return Ok(self.0.hash_unicode(text.codepoints()));
        }
        Ok(match value {
            Options::Atom(V::None) => 4_238_894_112,
            Options::Atom(V::Bool(v)) => i64::from(*v),
            Options::Atom(V::Int(v)) => {
                if *v == -1 {
                    -2
                } else {
                    *v
                }
            }
            Options::Tuple(v) => hash_tuple(
                &v.iter()
                    .map(|v| self.hash(v))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Options::Object(v) if GeneralMapping::get(value).is_none() => object_identity_hash(v),
            _ => {
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: format!(
                        "unhashable type: '{}'",
                        if matches!(value, Options::List(_)) {
                            "list"
                        } else {
                            "dict"
                        }
                    ),
                });
            }
        })
    }
    fn identity_token(&self, value: &Options) -> u64 {
        match value {
            Options::Object(v) => Rc::as_ptr(v) as *const () as usize as u64,
            Options::Tuple(v) => Rc::as_ptr(v) as usize as u64,
            Options::Atom(V::None) => 1,
            Options::Atom(V::Bool(v)) => 2 + u64::from(*v),
            // No fixture uses distinct integer objects as keys. A non-object
            // integer cannot silently substitute for native primitive identity.
            _ => panic!("own mapping fixture requires object-backed key identity"),
        }
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        if let (Some(a), Some(b)) = (SourceText::get(a), SourceText::get(b)) {
            return Ok(a.codepoints() == b.codepoints());
        }
        Ok(match (a, b) {
            (Options::Atom(V::None), Options::Atom(V::None)) => true,
            (Options::Tuple(a), Options::Tuple(b)) if a.len() == b.len() => {
                for (a, b) in a.iter().zip(b.iter()) {
                    if !a.same_object(b) && !self.equal(a, b)? {
                        return Ok(false);
                    }
                }
                true
            }
            _ => a.same_object(b),
        })
    }
    fn exact_unicode(&self, value: &Options) -> bool {
        SourceText::get(value).is_some()
    }
}
impl MappingKeyBindings for OwnMappingKeys {
    fn text_key(&self, value: &str) -> Result<Options, Error> {
        Ok(SourceText::from_utf8(value))
    }
    fn text_contents(&self, value: &Options) -> Result<Option<String>, Error> {
        SourceText::get(value).map_or(Ok(None), |v| {
            v.utf8().map(Some).ok_or_else(|| Error::BindingGap {
                backend: "own_mapping_fixture".into(),
                operation: "raw surrogate key requires typed consumer".into(),
            })
        })
    }
    fn missing_key(&self, key: &Options) -> Result<Error, Error> {
        let message = if let Some(text) = SourceText::get(key) {
            imagine_inference::triton_specialization::python_str_repr(
                &text.utf8().expect("own utf8 missing key"),
            )
        } else if matches!(key, Options::Atom(V::None)) {
            "None".into()
        } else {
            panic!("own missing-key fixture needs explicit repr protocol")
        };
        Ok(Error::ReferenceException {
            class: "KeyError",
            message,
        })
    }
}
