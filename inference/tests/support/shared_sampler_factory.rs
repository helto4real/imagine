use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::factory::{DpmWrapper, Factory, FunctionBindings, SamplerState},
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, rc::Rc};
struct Symbol(String);
struct Modules<'a> {
    case: &'a Value,
    events: Vec<Value>,
    symbols: BTreeMap<(String, String), Options>,
}
impl FunctionBindings for Modules<'_> {
    fn attribute(&mut self, module: &str, name: &str) -> Result<Options, Error> {
        let index = self.events.len();
        self.events.push(json!([module, name]));
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own factory lookup failure".into(),
            });
        }
        self.symbols
            .get(&(module.into(), name.into()))
            .cloned()
            .ok_or_else(|| Error::ReferenceException {
                class: "AttributeError",
                message: format!("own missing {module}.{name}"),
            })
    }
}
fn identity(value: &Options, roots: &mut Vec<Options>) -> usize {
    if let Some(index) = roots.iter().position(|old| old.same_object(value)) {
        return index;
    }
    roots.push(value.clone());
    roots.len() - 1
}
fn snapshot(
    value: &Options,
    functions: &mut Vec<Options>,
    dictionaries: &mut Vec<Options>,
) -> Value {
    match value {
        Options::Atom(V::None) => json!({"kind":"none"}),
        Options::Dict(values) => {
            let index = identity(value, dictionaries);
            let items: Vec<_> = values
                .borrow()
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        Options::Atom(V::Bool(v)) => json!(v),
                        Options::Atom(V::Int(v)) => json!(v),
                        _ => panic!("own dictionary atom"),
                    };
                    json!([key, value])
                })
                .collect();
            json!({"kind":"dict","identity":index,"items":items})
        }
        Options::Object(object) => {
            let name = if let Some(symbol) = object.downcast_ref::<Symbol>() {
                symbol.0.as_str()
            } else {
                object.downcast_ref::<DpmWrapper>().unwrap().name()
            };
            json!({"kind":"function","identity":identity(value,functions),"name":name})
        }
        _ => panic!("own factory value"),
    }
}
fn observe(
    state: &SamplerState,
    functions: &mut Vec<Options>,
    dictionaries: &mut Vec<Options>,
) -> Value {
    json!(
        state
            .fields
            .entries()
            .into_iter()
            .map(|(name, value)| json!([name, snapshot(&value, functions, dictionaries)]))
            .collect::<Vec<_>>()
    )
}
#[test]
fn original_sampler_factories_preserve_four_default_dictionaries_ddim_freshness_function_identity_explicit_none_reinitialization_and_lookup_errors()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-sampler-factory.json")).unwrap();
    assert_eq!(
        fixture["registered"],
        json!(&imagine_inference::sampling::plan::SAMPLERS[..])
    );
    for case in fixture["cases"].as_array().unwrap() {
        let factory = Factory::default();
        let view = factory.clone();
        let mut modules = Modules {
            case,
            events: vec![],
            symbols: BTreeMap::new(),
        };
        for name in fixture["module_functions"].as_array().unwrap() {
            let name = name.as_str().unwrap();
            modules.symbols.insert(
                ("k_diffusion_sampling".into(), name.into()),
                Options::Object(Rc::new(Symbol(name.into()))),
            );
        }
        for name in ["sample_unipc", "sample_unipc_bh2"] {
            modules.symbols.insert(
                ("uni_pc".into(), name.into()),
                Options::Object(Rc::new(Symbol(name.into()))),
            );
        }
        let (mut created, mut history, mut functions, mut dictionaries) =
            (vec![], vec![], vec![], vec![]);
        let shared_extra = Options::dictionary(vec![("explicit".into(), Options::Atom(V::Int(1)))]);
        let shared_inpaint =
            Options::dictionary(vec![("random".into(), Options::Atom(V::Bool(false)))]);
        let result = (|| -> Result<(), Error> {
            let names = [
                "euler",
                "heun",
                "uni_pc",
                "uni_pc_bh2",
                "ddim",
                "ddim",
                "dpm_fast",
                "dpm_fast",
                "dpm_adaptive",
                "dpm_adaptive",
            ]
            .into_iter()
            .chain(
                fixture["registered"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap()),
            )
            .chain(["ar_video"]);
            for (index, name) in names.enumerate() {
                let active = if index % 2 == 0 { &factory } else { &view };
                let state = active.sampler_object(name, &mut modules)?;
                history.push(observe(&state, &mut functions, &mut dictionaries));
                created.push(state);
                if created.len() == 1 {
                    created
                        .last()
                        .unwrap()
                        .fields
                        .get("extra_options")?
                        .set("shared-extra", Options::Atom(V::Int(7)))?;
                    created
                        .last()
                        .unwrap()
                        .fields
                        .get("inpaint_options")?
                        .set("shared-inpaint", Options::Atom(V::Bool(true)))?;
                }
                if created.len() == 3 {
                    created
                        .last()
                        .unwrap()
                        .fields
                        .get("extra_options")?
                        .set("constructor-extra", Options::Atom(V::Int(9)))?;
                    created
                        .last()
                        .unwrap()
                        .fields
                        .get("inpaint_options")?
                        .set("constructor-inpaint", Options::Atom(V::Bool(false)))?;
                }
            }
            let none = Options::Atom(V::None);
            for (extra, inpaint) in [
                (shared_extra.clone(), shared_inpaint.clone()),
                (shared_extra.clone(), shared_inpaint.clone()),
                (none.clone(), none.clone()),
                (Options::dictionary(vec![]), Options::dictionary(vec![])),
                (Options::dictionary(vec![]), none),
            ] {
                let state = factory.ksampler("euler", Some(extra), Some(inpaint), &mut modules)?;
                history.push(observe(&state, &mut functions, &mut dictionaries));
                created.push(state);
            }
            created[0].initialize(
                created[2].fields.get("sampler_function")?,
                shared_extra,
                shared_inpaint,
            );
            history.push(observe(&created[0], &mut functions, &mut dictionaries));
            factory.sampler_object("own-missing", &mut modules)?;
            Ok(())
        })();
        let error = match result {
            Ok(()) => Value::Null,
            Err(Error::ReferenceException { class, message }) => json!([class, message]),
            Err(error) => panic!("unbound factory {error:?}"),
        };
        assert_eq!(error, case["error"]);
        assert_eq!(json!(modules.events), case["events"]);
        assert_eq!(json!(history), case["history"]);
        let defaults = factory
            .default_dictionaries()
            .iter()
            .map(|value| snapshot(value, &mut functions, &mut dictionaries))
            .collect::<Vec<_>>();
        assert_eq!(json!(defaults), case["defaults"]);
    }
}
