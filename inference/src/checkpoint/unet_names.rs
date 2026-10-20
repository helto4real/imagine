//! Original Diffusers UNet map construction and checkpoint selection. The
//! approved hash-state criterion is scoped to these three source set literals.
use super::{
    detector_values::{Context, list, text, tuple},
    unet_names_program as source,
};
pub use crate::python_hash::HashState;
use crate::{
    execution::Runtime, native::Error, patching::options::Options, python_hash::hash_tuple,
    source_set::Table,
};

pub struct UnetNames {
    pub attentions: Vec<String>,
    pub transformers: Vec<String>,
    pub basic: Vec<(String, String)>,
}
#[derive(Clone, Copy)]
pub enum ConstantOrigin {
    Source,
    Marshalled,
}
fn compiled_set<T: Clone + Eq>(values: impl IntoIterator<Item = (i64, T)>) -> Table<T> {
    // starunpack_helper folds source elements into a frozenset, then
    // merge_consts_recursive rebuilds it in table order before SET_UPDATE.
    // These locked literal strings contain dots and are not name-interned.
    let mut folded = Table::default();
    for (hash, value) in values {
        folded.add(hash, value, |a, b| a == b);
    }
    let mut merged = Table::default();
    for (hash, value) in folded.entries() {
        merged.add(hash, value.clone(), |a, b| a == b);
    }
    Table::from_frozen(&merged)
}
fn unmarshalled_set<T: Clone + Eq>(values: impl IntoIterator<Item = (i64, T)>) -> Table<T> {
    // marshal.read frozenset inserts serialized elements once, without the
    // source compiler's constant-merging rebuild. Runtime SET_UPDATE follows.
    let mut constant = Table::default();
    for (hash, value) in values {
        constant.add(hash, value, |a, b| a == b);
    }
    Table::from_frozen(&constant)
}
impl UnetNames {
    /// Original source-compilation path. Cached-code constant construction
    /// remains a separate binding; no existing Python cache is inspected here.
    pub fn from_source(state: &HashState) -> Self {
        Self::new(state, ConstantOrigin::Source)
    }
    /// Reconstruct the locked constant payload from original marshal element
    /// order. No Python bytecode, interpreter, or user's cache file is loaded.
    pub fn new(state: &HashState, origin: ConstantOrigin) -> Self {
        let strings = |values: &[&str]| {
            let values = values.iter().map(|v| (state.hash_text(v), (*v).to_owned()));
            let table = match origin {
                ConstantOrigin::Source => compiled_set(values),
                ConstantOrigin::Marshalled => unmarshalled_set(values),
            };
            table.iter().cloned().collect()
        };
        let (attentions, transformers, basic) = match origin {
            ConstantOrigin::Source => (
                source::UNET_MAP_ATTENTIONS,
                source::TRANSFORMER_BLOCKS,
                source::UNET_MAP_BASIC,
            ),
            ConstantOrigin::Marshalled => (
                source::UNET_MAP_ATTENTIONS_MARSHALLED,
                source::TRANSFORMER_BLOCKS_MARSHALLED,
                source::UNET_MAP_BASIC_MARSHALLED,
            ),
        };
        let basic = basic.iter().map(|(a, b)| {
            (
                hash_tuple(&[state.hash_text(a), state.hash_text(b)]),
                ((*a).into(), (*b).into()),
            )
        });
        let basic = match origin {
            ConstantOrigin::Source => compiled_set(basic),
            ConstantOrigin::Marshalled => unmarshalled_set(basic),
        };
        Self {
            attentions: strings(attentions),
            transformers: strings(transformers),
            basic: basic.iter().cloned().collect(),
        }
    }
    pub fn name_map(&self, runtime: &Runtime, config: &Options) -> Result<Options, Error> {
        let mut cx = Context::new(
            runtime,
            Options::dictionary(vec![]),
            "",
            Options::dictionary(vec![]),
        );
        cx.set("unet_config", config.clone());
        cx.set(
            "UNET_MAP_ATTENTIONS",
            list(self.attentions.iter().map(|v| text(v)).collect()),
        );
        cx.set(
            "TRANSFORMER_BLOCKS",
            list(self.transformers.iter().map(|v| text(v)).collect()),
        );
        cx.set(
            "UNET_MAP_BASIC",
            list(
                self.basic
                    .iter()
                    .map(|(a, b)| tuple(vec![text(a), text(b)]))
                    .collect(),
            ),
        );
        cx.set(
            "UNET_MAP_RESNET",
            Options::dictionary(
                source::UNET_MAP_RESNET
                    .iter()
                    .map(|(a, b)| ((*a).into(), text(b)))
                    .collect(),
            ),
        );
        source::run(cx)
    }
}
/// Exact original conversion loop. Warning effects are required and fallible;
/// partial pop/assignment state remains visible if the logging callback fails.
pub fn convert_checkpoint(
    name_map: &Options,
    state_dict: &Options,
    new_state_dict: &Options,
    warning: &mut dyn FnMut(&str) -> Result<(), Error>,
) -> Result<(), Error> {
    let keys = name_map.keys_iterator()?;
    while let Some(key) = keys.next_key()? {
        if state_dict.get(&key)?.is_some() {
            // Python evaluates the RHS pop before evaluating the assignment
            // target. This matters if these dictionaries alias one another.
            let value = state_dict
                .pop(&key)?
                .ok_or_else(|| Error::ReferenceException {
                    class: "KeyError",
                    message: crate::triton_specialization::python_str_repr(&key),
                })?;
            let target = name_map
                .get(&key)?
                .ok_or_else(|| Error::ReferenceException {
                    class: "KeyError",
                    message: crate::triton_specialization::python_str_repr(&key),
                })?;
            let crate::execution_value::Value::Text(target) = target.snapshot()? else {
                return Err(Error::BindingGap {
                    backend: "diffusers_unet".into(),
                    operation:
                        "non-string conversion target requires original hash/subscription binding"
                            .into(),
                });
            };
            new_state_dict.set(&target, value)?;
        } else {
            let target = name_map
                .get(&key)?
                .ok_or_else(|| Error::ReferenceException {
                    class: "KeyError",
                    message: crate::triton_specialization::python_str_repr(&key),
                })?;
            warning(&format!(
                "{} {}",
                crate::python_value::display(&target.snapshot()?)?,
                key
            ))?;
        }
    }
    Ok(())
}
