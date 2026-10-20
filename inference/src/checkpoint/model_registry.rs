//! Shared source-ordered model selection. A selected class must invoke its real
//! native constructor; descriptors never masquerade as constructed models.
use crate::{
    execution::Runtime, execution_value::Value as V, native::Error, patching::options::Options,
};
pub(super) struct ClassDef {
    pub name: &'static str,
    pub parent: Option<usize>,
    pub own_config: Option<Options>,
    pub own_required: Option<Vec<String>>,
}
pub struct ModelConfigRegistry {
    classes: Vec<ClassDef>,
    order: Vec<usize>,
}
/// Original constructor and logging effects are required native bindings.
/// Implementations must construct the selected class or return an explicit
/// binding gap; choosing a different class is not an admissible fallback.
pub trait ModelConfigBindings {
    type Model;
    fn construct(&mut self, class: &'static str, config: &Options) -> Result<Self::Model, Error>;
    fn no_match(&mut self, message: &str) -> Result<(), Error>;
}
impl Default for ModelConfigRegistry {
    fn default() -> Self {
        Self::frozen()
    }
}
impl ModelConfigRegistry {
    pub fn frozen() -> Self {
        Self {
            classes: super::registry_data::classes(),
            order: super::registry_data::ORDER.to_vec(),
        }
    }
    pub fn names(&self) -> Vec<&'static str> {
        self.order.iter().map(|i| self.classes[*i].name).collect()
    }
    fn index(&self, name: &str) -> Result<usize, Error> {
        self.classes
            .iter()
            .position(|c| c.name == name)
            .ok_or_else(|| Error::BindingGap {
                backend: "model_config".into(),
                operation: format!("unknown frozen class {name}"),
            })
    }
    fn config(&self, mut index: usize) -> Options {
        loop {
            let class = &self.classes[index];
            if let Some(v) = &class.own_config {
                return v.clone();
            }
            index = class.parent.expect("generated base has config");
        }
    }
    fn required(&self, mut index: usize) -> &[String] {
        loop {
            let class = &self.classes[index];
            if let Some(v) = &class.own_required {
                return v;
            }
            index = class.parent.expect("generated base has required keys");
        }
    }
    /// The original class dictionary object; inherited dictionaries remain shared.
    pub fn match_config(&self, name: &str) -> Result<Options, Error> {
        Ok(self.config(self.index(name)?))
    }
    pub fn matches(
        &self,
        rt: &Runtime,
        name: &str,
        config: &Options,
        state: Option<&Options>,
        prefix: &str,
    ) -> Result<bool, Error> {
        let index = self.index(name)?;
        for (key, expected) in self.config(index).dict_items()? {
            let Some(value) = config.get(&key)? else {
                return Ok(false);
            };
            if not_equal(rt, &expected, &value)? {
                return Ok(false);
            }
        }
        if let Some(state) = state {
            for key in self.required(index) {
                // Every frozen template contains either no replacement field or
                // exactly the plain positional field. Reject new format syntax.
                let key = key.replace("{}", prefix);
                if state.get(&key)?.is_none() {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    pub fn select(
        &self,
        rt: &Runtime,
        config: &Options,
        state: Option<&Options>,
        prefix: &str,
    ) -> Result<Option<&'static str>, Error> {
        for index in &self.order {
            let name = self.classes[*index].name;
            if self.matches(rt, name, config, state, prefix)? {
                return Ok(Some(name));
            }
        }
        Ok(None)
    }
    /// `construct` binds the selected original class, including subclass/base
    /// initialization and latent-format ownership. No constructor is synthesized
    /// from match data and unsupported classes are never skipped.
    pub fn construct<B: ModelConfigBindings>(
        &self,
        rt: &Runtime,
        config: &Options,
        state: Option<&Options>,
        prefix: &str,
        bindings: &mut B,
    ) -> Result<Option<B::Model>, Error> {
        if let Some(name) = self.select(rt, config, state, prefix)? {
            return bindings.construct(name, config).map(Some);
        }
        let message = format!(
            "no match {}",
            crate::python_value::repr_with_native(&config.snapshot()?, &rt.native)?
        );
        bindings.no_match(&message)?;
        Ok(None)
    }
}
fn not_equal(rt: &Runtime, a: &Options, b: &Options) -> Result<bool, Error> {
    if matches!(b, Options::Atom(V::Tensor(_) | V::Quantized(_)))
        && matches!(
            a,
            Options::Atom(V::Bool(_) | V::Int(_) | V::UInt(_) | V::Float(_))
        )
    {
        // Tensor's ordinary Python rich comparison accepts bool as a scalar;
        // the boxed number schema rejects an IValue Bool before dispatch.
        let scalar = match a.snapshot()? {
            V::Bool(value) => V::Int(i64::from(value)),
            value => value,
        };
        let ne = unsafe { rt.op("aten::ne.Scalar", &[b.snapshot()?, scalar], &vec![]) }?;
        let truth = unsafe { rt.op("aten::is_nonzero.default", &[ne], &vec![]) }?;
        return Ok(matches!(truth, V::Bool(true)));
    }
    match (a, b) {
        (Options::List(a), Options::List(b)) => {
            if std::rc::Rc::ptr_eq(a, b) {
                return Ok(false);
            }
            let a = a.borrow();
            let b = b.borrow();
            if a.len() != b.len() {
                return Ok(true);
            }
            for (a, b) in a.iter().zip(b.iter()) {
                if not_equal(rt, a, b)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        (Options::Tuple(a), Options::Tuple(b)) => {
            if std::rc::Rc::ptr_eq(a, b) {
                return Ok(false);
            }
            if a.len() != b.len() {
                return Ok(true);
            }
            for (a, b) in a.iter().zip(b.iter()) {
                if not_equal(rt, a, b)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Ok(!crate::conditioning::is_equal(
            &a.snapshot()?,
            &b.snapshot()?,
        )?),
    }
}
