//! Concrete KeyError args ownership and fallible, deferred __str__ protocol.
//! Raising KeyError does not call repr(key); formatting can itself raise later.
use crate::{native::Error, patching::options::Options};
use std::{cell::RefCell, rc::Rc};
pub trait KeyRepresentation {
    fn repr(&self, key: &Options) -> Result<String, Error>;
}
pub struct KeyError {
    args: RefCell<Options>,
    representation: Rc<dyn KeyRepresentation>,
}
impl std::fmt::Debug for KeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyError").finish_non_exhaustive()
    }
}
impl KeyError {
    pub fn new(key: Options, representation: Rc<dyn KeyRepresentation>) -> Rc<Self> {
        Rc::new(Self {
            args: RefCell::new(Options::Tuple(Rc::new(vec![key]))),
            representation,
        })
    }
    pub fn arguments(&self) -> Options {
        self.args.borrow().clone()
    }
    /// Already-converted tuple from the BaseException.args setter. The public
    /// sequence conversion/caller frame remains a separate binding.
    pub fn replace_arguments(&self, args: Options) -> Result<(), Error> {
        if !matches!(args, Options::Tuple(_)) {
            return Err(Error::BindingGap {
                backend: "key_error".into(),
                operation: "args setter requires its source sequence-to-tuple conversion".into(),
            });
        }
        let old = self.args.replace(args);
        drop(old);
        Ok(())
    }
    pub fn message(&self) -> Result<String, Error> {
        let args = self.args.borrow().clone();
        let Options::Tuple(values) = &args else {
            unreachable!("source args tuple")
        };
        match values.len() {
            0 => Ok(String::new()),
            1 => self.representation.repr(&values[0]),
            _ => self.representation.repr(&args),
        }
    }
}
impl Error {
    /// Original str(exception), which can raise. Display is the diagnostic/
    /// uncaught-exception path and never panics if __str__ itself fails.
    pub fn python_message(&self) -> Result<String, Error> {
        match self {
            Self::KeyError(value) => value.message(),
            Self::Retained(value) => value.source().python_message(),
            _ => Ok(self.to_string()),
        }
    }
}
