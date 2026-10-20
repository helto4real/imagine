//! Original GLOBAL policy; a contains test and its later item read use distinct
//! live registry getter calls, as in the reference's cached/user dictionaries.
use crate::{execution_value::Value, native::Error, patching::options::Options};
type NamePair = (&'static str, &'static str);
type NameMapping = (NamePair, NamePair);
#[path = "weights_only_globals_data.rs"]
mod data;
pub trait Bindings {
    fn allowed(&mut self) -> Result<Options, Error>;
    fn user(&mut self) -> Result<Options, Error>;
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class: "UnpicklingError",
        message: message.into(),
    }
}
pub fn mapped_name<'a>(module: &'a str, name: &'a str) -> (&'a str, &'a str) {
    if let Some((_, replacement)) = data::NAMES.iter().find(|(old, _)| *old == (module, name)) {
        return *replacement;
    }
    (
        data::IMPORTS
            .iter()
            .find_map(|(old, new)| (*old == module).then_some(*new))
            .unwrap_or(module),
        name,
    )
}
pub fn resolve(module: &str, name: &str, bindings: &mut dyn Bindings) -> Result<Options, Error> {
    let (module, name) = mapped_name(module, name);
    let path = format!("{module}.{name}");
    if data::BLOCKED.contains(&module) {
        return Err(unsupported(format!(
            "Trying to load unsupported GLOBAL {path} whose module {module} is blocked."
        )));
    }
    let key = Options::Atom(Value::Text(path.clone()));
    let allowed = bindings.allowed()?;
    let present = allowed.contains_item(&key)?;
    drop(allowed);
    if present {
        return bindings.allowed()?.item(&key);
    }
    let user = bindings.user()?;
    let present = user.contains_item(&key)?;
    drop(user);
    if present {
        return bindings.user()?.item(&key);
    }
    if let Some((_, message)) = data::SPECIAL
        .iter()
        .find(|(paths, _)| paths.contains(&path.as_str()))
    {
        return Err(unsupported(*message));
    }
    let path = if let Some(rest) = path.strip_prefix("builtins.") {
        rest
    } else {
        &path
    };
    Err(unsupported(format!(
        "Unsupported global: GLOBAL {path} was not an allowed global by default. Please use `torch.serialization.add_safe_globals([{path}])` or the `torch.serialization.safe_globals([{path}])` context manager to allowlist this global if you trust this class/function."
    )))
}
