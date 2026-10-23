//! Original GLOBAL policy; a contains test and its later item read use distinct
//! live registry getter calls, as in the reference's cached/user dictionaries.
use crate::{
    execution_value::Value,
    native::Error,
    patching::options::{GeneralMapping, Options, SourceText},
};
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
    let key = SourceText::from_utf8(&path);
    // Preserve the typed Unicode key across both live getters. The older
    // string-only map is still a supported typed ABI consumer.
    let legacy_key = Options::Atom(Value::Text(path.clone()));
    let key_for = |dictionary: &Options| {
        if GeneralMapping::get(dictionary).is_some() {
            &key
        } else {
            &legacy_key
        }
    };
    let allowed = bindings.allowed()?;
    let present = allowed.contains_item(key_for(&allowed))?;
    drop(allowed);
    if present {
        let allowed = bindings.allowed()?;
        return allowed.item(key_for(&allowed));
    }
    let user = bindings.user()?;
    let present = user.contains_item(key_for(&user))?;
    drop(user);
    if present {
        let user = bindings.user()?;
        return user.item(key_for(&user));
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
