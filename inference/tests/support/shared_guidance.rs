use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::guidance::{self as g, CfgInput, Context, Evaluator, Hooks},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::{Kind, Tensor};
struct Exec<'a> {
    case: &'a Value,
    index: usize,
    mode: &'a str,
    callbacks: Rc<RefCell<Vec<usize>>>,
    active: Rc<RefCell<Vec<usize>>>,
}
impl Exec<'_> {
    fn next(&mut self, name: &str) -> &Value {
        let v = &self.case["events"][self.index];
        self.index += 1;
        assert_eq!(v["event"], name);
        v
    }
}
impl Hooks<f64> for Exec<'_> {
    fn begin_post(&mut self) -> Result<(), Error> {
        self.active = self.callbacks.clone();
        Ok(())
    }
    fn has_custom_cfg(&self) -> bool {
        self.mode == "custom"
    }
    fn custom_cfg(&mut self, args: CfgInput<'_, f64>) -> Result<Tensor, Error> {
        let v = self.next("custom");
        let cond = args.cond_noise.unwrap();
        let uncond = args.uncond_noise.unwrap();
        for (key, t) in [
            ("cond", cond),
            ("uncond", uncond),
            ("cond_denoised", args.cond_denoised),
            ("uncond_denoised", args.uncond_denoised),
        ] {
            check(&V::tensor(t.shallow_clone()), &v[key], t.device(), key);
        }
        Ok(uncond.f_add(&cond.f_sub(uncond)?.f_mul_scalar(args.context.scale * 0.5)?)?)
    }
    fn post_count(&self) -> usize {
        self.active.borrow().len()
    }
    fn post(
        &mut self,
        index: usize,
        args: CfgInput<'_, f64>,
        denoised: Tensor,
    ) -> Result<Tensor, Error> {
        let callback = self.active.borrow()[index];
        let v = self.next(if callback == 0 { "post0" } else { "post1" });
        for (key, t) in [
            ("denoised", &denoised),
            ("cond_denoised", args.cond_denoised),
            ("uncond_denoised", args.uncond_denoised),
        ] {
            check(&V::tensor(t.shallow_clone()), &v[key], t.device(), key);
        }
        assert_eq!(json!(args.context.cond), v["cond"]);
        assert_eq!(json!(args.context.uncond), v["uncond"]);
        if callback == 0 {
            if self.mode == "append_post" {
                self.callbacks.borrow_mut().push(1);
            }
            if self.mode == "replace_post" {
                self.callbacks = Rc::new(RefCell::new(Vec::new()));
            }
            Ok(denoised.f_mul_scalar(0.875)?.f_add_scalar(0.03125)?)
        } else {
            Ok(denoised.f_add_scalar(0.0625)?)
        }
    }
}
impl Evaluator<f64> for Exec<'_> {
    fn batch(
        &mut self,
        conditions: &[Option<&f64>],
        x: &Tensor,
        t: &Tensor,
        _opts: &Options,
    ) -> Result<g::Predictions, Error> {
        let v = self.next("batch");
        assert_eq!(json!(conditions), v["conds"]);
        check(
            &V::tensor(x.shallow_clone()),
            &v["input"],
            x.device(),
            "CFG batch input",
        );
        check(
            &V::tensor(t.shallow_clone()),
            &v["time"],
            x.device(),
            "CFG batch timestep",
        );
        conditions
            .iter()
            .map(|c| {
                if let Some(c) = c {
                    Ok(x.f_mul_scalar(0.125)?.f_add_scalar(**c * 0.0625)?)
                } else {
                    Ok(x.f_zeros_like()?)
                }
            })
            .collect::<Result<Vec<_>, Error>>()
            .map(g::predictions)
    }
    fn pre_count(&self) -> usize {
        usize::from(self.mode != "plain")
    }
    fn pre(
        &mut self,
        _index: usize,
        conditions: &[Option<&f64>],
        _context: &Context<'_, f64>,
        out: g::Predictions,
    ) -> Result<g::Predictions, Error> {
        let v = self.next("pre");
        assert_eq!(json!(conditions), v["conds"]);
        for (i, t) in out.borrow().iter().enumerate() {
            check(
                &V::tensor(t.shallow_clone()),
                &v["values"][i],
                t.device(),
                "pre-CFG input",
            );
        }
        out.borrow()
            .iter()
            .map(|t| Ok(t.f_mul_scalar(0.75)?))
            .collect::<Result<Vec<_>, Error>>()
            .map(g::predictions)
    }
}
#[test]
fn shared_cfg_dual_cfg_callbacks_mutation_and_cfg1_boundary_match() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-guidance.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let x = Tensor::arange(12, (Kind::Float, dev))
            .f_add_scalar(3)
            .unwrap()
            .f_remainder(29)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.40625)
            .unwrap()
            .reshape([2, 1, 2, 3])
            .to_kind(dtype);
        let time = Tensor::from_slice(&[0.75f32, 0.5]).to_device(dev);
        let mode = c["mode"].as_str().unwrap_or("plain");
        let callbacks = Rc::new(RefCell::new(if mode == "plain" {
            vec![]
        } else if mode == "append_post" {
            vec![0]
        } else {
            vec![0, 1]
        }));
        let mut exec = Exec {
            case: c,
            index: 0,
            mode,
            callbacks: callbacks.clone(),
            active: callbacks,
        };
        let options = Options::from_value(V::Dict(vec![(
            "disable_cfg1_optimization".into(),
            V::Bool(c["disable"].as_bool().unwrap()),
        )]));
        let positive = 2.;
        let middle = 0.5;
        let negative = -1.;
        let y = if c["name"] == "dual" {
            g::dual(
                &mut exec,
                g::Dual {
                    x: &x,
                    timestep: &time,
                    options: &options,
                    positive: Some(&positive),
                    middle: Some(&middle),
                    negative: Some(&negative),
                    cfg1: c["cfg1"].as_f64().unwrap(),
                    cfg2: c["cfg2"].as_f64().unwrap(),
                    nested: c["nested"].as_bool().unwrap(),
                    disable_cfg1: c["disable"].as_bool().unwrap(),
                },
            )
        } else {
            g::sampling(
                &mut exec,
                Context {
                    x: &x,
                    timestep: &time,
                    scale: c["scale"].as_f64().unwrap(),
                    options: &options,
                    cond: Some(&positive),
                    uncond: Some(&negative),
                },
                c["disable"].as_bool().unwrap(),
            )
        }
        .unwrap();
        check(&V::tensor(y), &c["output"], dev, &format!("Guidance {c}"));
        assert_eq!(exec.index, c["events"].as_array().unwrap().len());
    }
}
