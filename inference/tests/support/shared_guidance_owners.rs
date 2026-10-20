use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::{Error, WeakStorage},
    patching::options::Options,
    sampling::guidance::{
        self as g, CfgInput, Context, Evaluator, Hooks, PerpEvaluator, PerpPost, Predictions,
    },
};
use serde_json::{Value, json};
use tch::Tensor;
struct Exec<'a> {
    case: &'a Value,
    event: usize,
    probes: Vec<(String, WeakStorage)>,
    held: Option<Predictions>,
}
impl Exec<'_> {
    fn style(&self) -> &str {
        self.case["style"].as_str().unwrap()
    }
    fn track(&mut self, values: &[Tensor], label: &str) -> Result<(), Error> {
        for (index, value) in values.iter().enumerate() {
            self.probes
                .push((format!("{label}/{index}"), WeakStorage::new(value)?));
        }
        Ok(())
    }
    fn observe(&mut self, event: &str, fields: &[(&str, V)]) {
        let expected = &self.case["events"][self.event];
        self.event += 1;
        assert_eq!(expected["event"], event, "{}", self.case);
        assert_eq!(
            json!(self.probes.iter().map(|(name, _)| name).collect::<Vec<_>>()),
            expected["names"]
        );
        assert_eq!(
            json!(
                self.probes
                    .iter()
                    .map(|(_, weak)| !weak.expired())
                    .collect::<Vec<_>>()
            ),
            expected["alive"],
            "allocation lifetime {} {} {event}: {:?}",
            self.case["name"],
            self.case["style"],
            self.probes.iter().map(|(name, _)| name).collect::<Vec<_>>()
        );
        for (name, value) in fields {
            check(
                value,
                &expected[*name],
                device(self.case["device"].as_str().unwrap()),
                &format!("guidance owner {event} field {name}"),
            );
        }
    }
    fn post_body(&mut self, args: CfgInput<'_, f64>, denoised: Tensor) -> Result<Tensor, Error> {
        self.track(&[denoised.shallow_clone()], "post_input")?;
        self.observe(
            "post_enter",
            &[
                ("denoised", V::tensor(denoised.shallow_clone())),
                ("positive", V::tensor(args.cond_denoised.shallow_clone())),
                ("negative", V::tensor(args.uncond_denoised.shallow_clone())),
            ],
        );
        if self.style() == "post_clear" {
            self.held.as_ref().unwrap().borrow_mut().clear();
        }
        if self.style() == "post_replace" {
            let replacements = self
                .held
                .as_ref()
                .unwrap()
                .borrow()
                .iter()
                .map(|v| v.f_add_scalar(1.).map_err(Error::from))
                .collect::<Result<Vec<_>, _>>()?;
            self.track(&replacements, "replacement")?;
            *self.held.as_ref().unwrap().borrow_mut() =
                replacements.iter().map(Tensor::shallow_clone).collect();
            let out = denoised.f_mul_scalar(0.875)?.f_add_scalar(0.03125)?;
            self.track(&[out.shallow_clone()], "post_output")?;
            self.observe("post_exit", &[]);
            return Ok(out);
        }
        let out = denoised.f_mul_scalar(0.875)?.f_add_scalar(0.03125)?;
        self.track(&[out.shallow_clone()], "post_output")?;
        self.observe("post_exit", &[]);
        Ok(out)
    }
}
fn values(list: &Predictions) -> V {
    V::List(
        list.borrow()
            .iter()
            .map(|v| V::tensor(v.shallow_clone()))
            .collect(),
    )
}
impl Hooks<f64> for Exec<'_> {
    fn has_custom_cfg(&self) -> bool {
        self.style().starts_with("custom_")
    }
    fn custom_cfg(&mut self, args: CfgInput<'_, f64>) -> Result<Tensor, Error> {
        let cond = args.cond_noise.unwrap();
        let uncond = args.uncond_noise.unwrap();
        self.track(&[cond.shallow_clone(), uncond.shallow_clone()], "custom")?;
        self.observe(
            "custom_enter",
            &[
                ("positive", V::tensor(args.cond_denoised.shallow_clone())),
                ("negative", V::tensor(args.uncond_denoised.shallow_clone())),
            ],
        );
        Ok(uncond.f_add(&cond.f_sub(uncond)?.f_mul_scalar(args.context.scale * 0.5)?)?)
    }
    fn post_count(&self) -> usize {
        match self.style() {
            "post_clear" | "post_replace" | "pre_clear_post" | "custom_post" => 1,
            "pre_clear_post_two" => 2,
            _ => 0,
        }
    }
    fn post(
        &mut self,
        _index: usize,
        args: CfgInput<'_, f64>,
        denoised: Tensor,
    ) -> Result<Tensor, Error> {
        self.post_body(args, denoised)
    }
}
impl Evaluator<f64> for Exec<'_> {
    fn batch(
        &mut self,
        conditions: &[Option<&f64>],
        x: &Tensor,
        _time: &Tensor,
        _options: &Options,
    ) -> Result<Predictions, Error> {
        let out = g::predictions(
            conditions
                .iter()
                .map(|c| {
                    if let Some(c) = c {
                        Ok(x.f_mul_scalar(0.125)?.f_add_scalar(**c * 0.0625)?)
                    } else {
                        Ok(x.f_zeros_like()?)
                    }
                })
                .collect::<Result<Vec<_>, Error>>()?,
        );
        self.track(&out.borrow(), "batch")?;
        if matches!(self.style(), "post_clear" | "post_replace") {
            self.held = Some(out.clone());
        }
        self.observe(
            "batch",
            &[
                (
                    "conds",
                    V::List(
                        conditions
                            .iter()
                            .map(|v| v.map_or(V::None, |v| V::Float(*v)))
                            .collect(),
                    ),
                ),
                ("values", values(&out)),
            ],
        );
        Ok(out)
    }
    fn pre_count(&self) -> usize {
        usize::from(self.style().starts_with("pre_"))
    }
    fn pre(
        &mut self,
        _index: usize,
        _conditions: &[Option<&f64>],
        _context: &Context<'_, f64>,
        out: Predictions,
    ) -> Result<Predictions, Error> {
        self.observe("pre_enter", &[("values", values(&out))]);
        if self.style() == "pre_same" {
            self.observe("pre_exit", &[]);
            return Ok(out);
        }
        let mapped = out
            .borrow()
            .iter()
            .map(|v| v.f_mul_scalar(0.5).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        self.track(&mapped, "pre")?;
        if self.style().starts_with("pre_clear") {
            out.borrow_mut().clear();
        }
        let result = if self.style() == "pre_inplace" {
            *out.borrow_mut() = mapped.iter().map(Tensor::shallow_clone).collect();
            out
        } else {
            g::predictions(
                mapped
                    .iter()
                    .take(if self.style() == "pre_clear_short" {
                        1
                    } else {
                        mapped.len()
                    })
                    .map(Tensor::shallow_clone)
                    .collect(),
            )
        };
        self.observe("pre_exit", &[]);
        Ok(result)
    }
}
impl PerpEvaluator<f64> for Exec<'_> {
    fn post_perp(
        &mut self,
        _index: usize,
        args: PerpPost<'_, f64>,
        denoised: Tensor,
    ) -> Result<Tensor, Error> {
        self.post_body(args.cfg, denoised)
    }
}
#[test]
fn cfg_guiders_preserve_shared_prediction_lists_and_exact_weak_storage_lifetimes() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guidance-owners.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 176);
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let x = restore(&case["input"], dev).as_tensor().unwrap();
        let time = restore(&case["time"], dev).as_tensor().unwrap();
        let options = Options::from_value(V::Dict(vec![(
            "disable_cfg1_optimization".into(),
            V::Bool(true),
        )]));
        let mut exec = Exec {
            case,
            event: 0,
            probes: vec![],
            held: None,
        };
        let positive = 2.;
        let negative = -1.;
        let middle = 0.5;
        let empty = 0.25;
        let result = match case["name"].as_str().unwrap() {
            "sampling" => g::sampling(
                &mut exec,
                Context {
                    x: &x,
                    timestep: &time,
                    scale: 2.5,
                    options: &options,
                    cond: Some(&positive),
                    uncond: Some(&negative),
                },
                true,
            ),
            "perp" => g::perp(
                &mut exec,
                g::Perp {
                    x: &x,
                    timestep: &time,
                    options: &options,
                    positive: Some(&positive),
                    negative: Some(&negative),
                    empty: Some(&empty),
                    cfg: 2.5,
                    negative_scale: 0.5,
                },
            ),
            name => g::dual(
                &mut exec,
                g::Dual {
                    x: &x,
                    timestep: &time,
                    options: &options,
                    positive: Some(&positive),
                    middle: Some(&middle),
                    negative: Some(&negative),
                    cfg1: 2.5,
                    cfg2: 3.,
                    nested: name == "dual_nested",
                    disable_cfg1: true,
                },
            ),
        };
        let note = format!(
            "guidance storage case {index} {} {}",
            case["name"], case["style"]
        );
        if case["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected error {note}")),
                case,
            );
        } else {
            check(
                &V::tensor(result.unwrap_or_else(|e| panic!("{note}: {e}"))),
                &case["output"],
                dev,
                &note,
            );
        }
        exec.observe("after", &[]);
        assert_eq!(exec.event, case["events"].as_array().unwrap().len());
    }
}
