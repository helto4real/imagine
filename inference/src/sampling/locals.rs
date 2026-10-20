//! Own source function locals until their original reassignment/function return.
//! Shallow handles retain original storage; this never copies tensor payloads.
use super::solvers::{Execution, Prediction};
use crate::native::Error;
use std::collections::BTreeMap;
use tch::Tensor;

/// Tensor roots held by torch.utils._contextlib.decorate_context's args/kwargs
/// for the entire decorated invocation. Declare before the no-grad guard so
/// roots also survive restoration of the caller's grad mode.
#[must_use]
pub(crate) struct ArgumentRoots {
    tensors: Vec<Tensor>,
}
impl ArgumentRoots {
    pub(crate) fn new(arguments: &[&Tensor]) -> Self {
        Self {
            tensors: arguments
                .iter()
                .map(|value| value.shallow_clone())
                .collect(),
        }
    }
}
impl Drop for ArgumentRoots {
    fn drop(&mut self) {
        while self.tensors.pop().is_some() {}
    }
}

#[derive(Default)]
pub struct SourceLocals {
    values: BTreeMap<String, Tensor>,
    function: Option<&'static str>,
    arguments: Vec<Tensor>,
}
impl SourceLocals {
    pub fn named(function: &'static str) -> Self {
        Self {
            function: Some(function),
            ..Self::default()
        }
    }
    pub fn decorated(mut self, arguments: &[&Tensor]) -> Self {
        self.arguments = arguments
            .iter()
            .map(|value| value.shallow_clone())
            .collect();
        self
    }
    /// Copy handles only when an exception escapes. The resulting traceback
    /// owns the last published source locals after the Rust stack unwinds.
    pub fn retain_error(&self, error: Error) -> Error {
        let Some(function) = self.function else {
            return error;
        };
        if matches!(error.root(), Error::BindingGap { .. }) {
            return error;
        }
        let frame = crate::source_exception::Frame::new(function);
        for (name, value) in self.tensors() {
            frame.set(name, std::rc::Rc::new(value.shallow_clone()));
        }
        frame.finish();
        let error = error.retain_frame(frame);
        if self.arguments.is_empty() {
            return error;
        }
        let decorator = crate::source_exception::Frame::new("decorate_context");
        decorator.set(
            "args",
            std::rc::Rc::new(
                self.arguments
                    .iter()
                    .map(Tensor::shallow_clone)
                    .collect::<Vec<_>>(),
            ),
        );
        decorator.finish();
        error.retain_frame(decorator)
    }
    pub fn keep(&mut self, name: impl Into<String>, tensor: Tensor) -> Tensor {
        self.values.insert(name.into(), tensor.shallow_clone());
        tensor
    }
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Tensor)> {
        self.values
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }
    pub fn remove(&mut self, name: &str) {
        self.values.remove(name);
    }
    pub fn keep_list(&mut self, name: &str, tensors: &[Tensor]) {
        let prefix = format!("{name}[");
        self.values.retain(|key, _| !key.starts_with(&prefix));
        for (index, tensor) in tensors.iter().enumerate() {
            let _ = self.keep(format!("{name}[{index}]"), tensor.shallow_clone());
        }
    }
    pub fn denoise(
        &self,
        exec: &mut dyn Execution,
        x: &Tensor,
        sigma: &Tensor,
        cfg: bool,
    ) -> Result<Prediction, Error> {
        exec.source_locals(self, "model")?;
        exec.denoise(x, sigma, cfg)
    }
    pub fn callback(
        &self,
        exec: &mut dyn Execution,
        step: crate::models::qwen_image21::EulerStep<'_>,
    ) -> Result<(), Error> {
        exec.source_locals(self, "callback")?;
        exec.callback(step)
    }
    pub fn finish(&self, exec: &mut dyn Execution, value: Tensor) -> Result<Tensor, Error> {
        exec.source_locals(self, "return")?;
        Ok(value)
    }
}
