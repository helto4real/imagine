use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::guidance::{self as g, CfgInput, Context, Evaluator, Hooks, PerpEvaluator, PerpPost},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Tensor;
struct Exec {
    case: Value,
    event: usize,
    options: Options,
    pre: Option<Rc<RefCell<Vec<Options>>>>,
    post: Option<Rc<RefCell<Vec<Options>>>>,
}
impl Exec {
    fn next(&mut self, event: &str) -> Value {
        let expected = self.case["events"][self.event].clone();
        self.event += 1;
        assert_eq!(expected["event"], event, "{}", self.case["mode"]);
        expected
    }
    fn root_list(&self, name: &str) -> Rc<RefCell<Vec<Options>>> {
        match self.options.get(name).unwrap() {
            Some(Options::List(v)) => v,
            None => Rc::new(RefCell::new(vec![])),
            _ => panic!("Own callback list"),
        }
    }
    fn mutate(&self, name: &str, index: usize) {
        if index != 0 {
            return;
        }
        let mode = self.case["mode"].as_str().unwrap();
        if mode
            == format!(
                "{}_append",
                if name.contains("pre_") { "pre" } else { "post" }
            )
        {
            self.root_list(name)
                .borrow_mut()
                .push(Options::Atom(V::Int(1)));
        }
        if mode
            == format!(
                "{}_replace",
                if name.contains("pre_") { "pre" } else { "post" }
            )
        {
            self.options
                .set(
                    name,
                    Options::List(Rc::new(RefCell::new(vec![Options::Atom(V::Int(1))]))),
                )
                .unwrap();
        }
    }
    fn fields(&self, expected: &Value, context: &Context<'_, String>) {
        assert_eq!(expected["scale"], json!(context.scale));
        assert!(context.options.same_object(&self.options));
        assert!(expected["options_same"].as_bool().unwrap());
        assert!(expected["model_same"].as_bool().unwrap());
        check(
            &V::tensor(context.x.shallow_clone()),
            &expected["input"],
            context.x.device(),
            "Perp callback input",
        );
        check(
            &V::tensor(context.timestep.shallow_clone()),
            &expected["sigma"],
            context.timestep.device(),
            "Perp callback sigma",
        );
    }
}
impl Hooks<String> for Exec {
    fn has_custom_cfg(&self) -> bool {
        true
    }
    fn custom_cfg(&mut self, _: CfgInput<'_, String>) -> Result<Tensor, Error> {
        panic!("PerpNeg does not call the ordinary custom CFG")
    }
    fn begin_post(&mut self) -> Result<(), Error> {
        self.post = Some(self.root_list("sampler_post_cfg_function"));
        Ok(())
    }
    fn post_count(&self) -> usize {
        self.post.as_ref().unwrap().borrow().len()
    }
}
impl Evaluator<String> for Exec {
    fn batch(
        &mut self,
        conditions: &[Option<&String>],
        x: &Tensor,
        time: &Tensor,
        options: &Options,
    ) -> Result<g::Predictions, Error> {
        let expected = self.next("batch");
        assert_eq!(json!(conditions), expected["conds"]);
        assert!(options.same_object(&self.options));
        conditions
            .iter()
            .map(|c| {
                if c.is_none() || self.case["mode"] == "zeros" {
                    return Ok(x.f_zeros_like()?);
                }
                let offset = match c.unwrap().as_str() {
                    "p" => 0.5,
                    "n" => -0.5,
                    "e" => 0.25,
                    _ => unreachable!(),
                };
                Ok(x.f_mul_scalar(0.125)?
                    .f_add_scalar(offset)?
                    .f_add(&time.f_reshape([1, 1, 1, 1])?.f_mul_scalar(0.0625)?)?)
            })
            .collect::<Result<Vec<_>, Error>>()
            .map(g::predictions)
    }
    fn begin_pre(&mut self) -> Result<(), Error> {
        self.pre = Some(self.root_list("sampler_pre_cfg_function"));
        Ok(())
    }
    fn pre_count(&self) -> usize {
        self.pre.as_ref().unwrap().borrow().len()
    }
    fn pre(
        &mut self,
        index: usize,
        conditions: &[Option<&String>],
        context: &Context<'_, String>,
        out: g::Predictions,
    ) -> Result<g::Predictions, Error> {
        let V::Int(id) = self.pre.as_ref().unwrap().borrow()[index].snapshot()? else {
            panic!("Own callback ID")
        };
        let expected = self.next("pre");
        assert_eq!(json!(id), expected["index"]);
        assert_eq!(json!(conditions), expected["conds"]);
        self.fields(&expected, context);
        check(
            &V::List(
                out.borrow()
                    .iter()
                    .map(|t| V::tensor(t.shallow_clone()))
                    .collect(),
            ),
            &expected["output"],
            context.x.device(),
            "pre-CFG predictions",
        );
        for name in ["positive", "negative", "empty_denoised"] {
            check(
                &V::None,
                &expected[name],
                context.x.device(),
                "pre absent predictions",
            );
        }
        self.mutate("sampler_pre_cfg_function", id as usize);
        let mut values = out
            .borrow()
            .iter()
            .map(|v| {
                if id == 0 {
                    Ok(v.f_mul_scalar(0.5)?)
                } else {
                    Ok(v.f_add(&context.x.f_mul_scalar(0.03125)?)?)
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        if self.case["mode"] == "short" {
            values.truncate(2);
        } else if self.case["mode"] == "long" {
            values.push(values[0].shallow_clone());
        }
        Ok(g::predictions(values))
    }
}
impl PerpEvaluator<String> for Exec {
    fn post_perp(
        &mut self,
        index: usize,
        args: PerpPost<'_, String>,
        out: Tensor,
    ) -> Result<Tensor, Error> {
        let V::Int(id) = self.post.as_ref().unwrap().borrow()[index].snapshot()? else {
            panic!("Own callback ID")
        };
        let expected = self.next("post");
        assert_eq!(json!(id), expected["index"]);
        self.fields(&expected, &args.cfg.context);
        assert_eq!(json!(args.cfg.context.cond), expected["cond"]);
        assert_eq!(json!(args.cfg.context.uncond), expected["uncond"]);
        assert_eq!(json!(args.empty_cond), expected["empty"]);
        for (name, tensor) in [
            ("output", &out),
            ("positive", args.cfg.cond_denoised),
            ("negative", args.cfg.uncond_denoised),
            ("empty_denoised", args.empty_denoised),
        ] {
            check(
                &V::tensor(tensor.shallow_clone()),
                &expected[name],
                out.device(),
                "Perp post prediction",
            );
        }
        assert!(args.cfg.cond_noise.is_none() && args.cfg.uncond_noise.is_none());
        self.mutate("sampler_post_cfg_function", id as usize);
        Ok(if id == 0 {
            out.f_add(&args.empty_denoised.f_mul_scalar(0.125)?)?
        } else {
            out.f_mul_scalar(0.875)?
                .f_add(&args.empty_denoised.f_mul_scalar(0.03125)?)?
        })
    }
}
#[test]
fn perp_negative_guider_preserves_cfg1_branches_empty_callbacks_and_mutating_callback_lists() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-perp-guider.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 1056);
    let (positive, negative, empty) = ("p".to_string(), "n".to_string(), "e".to_string());
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let x = restore(&case["input"], dev).as_tensor().unwrap();
        let time = restore(&case["time"], dev).as_tensor().unwrap();
        let options = Options::from_value(V::Dict(vec![(
            "disable_cfg1_optimization".into(),
            restore(&case["disabled"], dev),
        )]));
        let mode = case["mode"].as_str().unwrap();
        for (name, present) in [
            (
                "sampler_pre_cfg_function",
                ["pre", "both", "pre_append", "pre_replace", "short", "long"].contains(&mode),
            ),
            (
                "sampler_post_cfg_function",
                ["post", "both", "post_append", "post_replace"].contains(&mode),
            ),
        ] {
            if present {
                options
                    .set(
                        name,
                        Options::List(Rc::new(RefCell::new(vec![Options::Atom(V::Int(0))]))),
                    )
                    .unwrap();
            }
        }
        let mut exec = Exec {
            case: case.clone(),
            event: 0,
            options: options.clone(),
            pre: None,
            post: None,
        };
        let result = g::perp(
            &mut exec,
            g::Perp {
                x: &x,
                timestep: &time,
                options: &options,
                positive: Some(&positive),
                negative: Some(&negative),
                empty: Some(&empty),
                cfg: case["cfg"].as_f64().unwrap(),
                negative_scale: case["negative_scale"].as_f64().unwrap(),
            },
        );
        if case["error"].is_string() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("Expected Perp error {index}")),
                case,
            )
        } else {
            check(
                &V::tensor(result.unwrap_or_else(|e| panic!("Perp {index}: {e}"))),
                &case["output"],
                dev,
                &format!("Perp {index}"),
            );
        }
        assert_eq!(
            exec.event,
            case["events"].as_array().unwrap().len(),
            "Perp {index}"
        );
        for (name, field) in [
            ("sampler_pre_cfg_function", "pre_ids"),
            ("sampler_post_cfg_function", "post_ids"),
        ] {
            let ids = exec
                .root_list(name)
                .borrow()
                .iter()
                .map(|v| match v.snapshot().unwrap() {
                    V::Int(n) => n,
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>();
            assert_eq!(json!(ids), case[field]);
        }
    }
}
