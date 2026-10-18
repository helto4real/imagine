use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    models::qwen_image21 as q,
    native::{Error, Generator},
};
use serde_json::Value;
use tch::{Device, Kind, Tensor};
fn data(dev: Device, dtype: Kind, offset: i64) -> Tensor {
    Tensor::arange(2 * 64 * 2 * 3, (Kind::Float, dev))
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .reshape([2, 64, 2, 3])
        .to_kind(dtype)
}
struct Exec<'a> {
    fixture: &'a Value,
    dev: Device,
    step: usize,
    generator: Generator,
}
impl q::EulerExecution for Exec<'_> {
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor) -> Result<Tensor, Error> {
        let expected = &self.fixture["calls"][self.step];
        check(
            &V::tensor(x.shallow_clone()),
            &expected["x"],
            self.dev,
            "Euler model input",
        );
        check(
            &V::tensor(sigma.shallow_clone()),
            &expected["sigma"],
            self.dev,
            "Euler model sigma",
        );
        let output = x
            .f_mul_scalar(0.125)?
            .f_add(&sigma.f_view([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?;
        q::flow_denoised(sigma, &output, x)
    }
    fn noise_like(&mut self, x: &Tensor) -> Result<Tensor, Error> {
        self.generator.randn(&x.size(), x.kind())
    }
    fn callback(&mut self, s: q::EulerStep<'_>) -> Result<(), Error> {
        let expected = &self.fixture["callbacks"][self.step];
        assert_eq!(s.i, self.step as i64);
        for (key, t) in [
            ("x", s.x),
            ("sigma", s.sigma),
            ("sigma_hat", s.sigma_hat),
            ("denoised", s.denoised),
        ] {
            check(
                &V::tensor(t.shallow_clone()),
                &expected[key],
                self.dev,
                &format!("Euler callback {key}"),
            );
        }
        self.step += 1;
        Ok(())
    }
}
#[test]
fn qwen_image21_original_flux_const_schedules_noise_rng_and_euler_histories_match() {
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-sampling.json")).unwrap();
    for g in fixture["schedules"].as_array().unwrap() {
        let s = q::FluxSampling::new(g["shift"].as_f64().unwrap(), 10000).unwrap();
        check(
            &V::tensor(s.sigmas.shallow_clone()),
            &g["sigmas"],
            Device::Cpu,
            "full FLUX sigma buffer",
        );
        for c in g["cases"].as_array().unwrap() {
            let steps = c["steps"].as_i64().unwrap();
            let y = if c["method"] == "simple" {
                s.simple(steps as usize)
            } else {
                s.normal(steps, c["method"] == "sgm_uniform")
            }
            .unwrap();
            check(
                &V::tensor(y),
                &c["output"],
                Device::Cpu,
                &format!("{}/{}/{}", g["shift"], c["method"], steps),
            );
        }
    }
    for c in fixture["algebra"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["dtype"].as_str().unwrap());
        let x = data(dev, dtype, 3);
        let noise = data(dev, dtype, 5);
        let output = data(dev, dtype, 7);
        let sigma = Tensor::from_slice(&[0.85f32, 0.35]).to_device(dev);
        for (key, y) in [
            ("denoised", q::flow_denoised(&sigma, &output, &x)),
            ("noise", q::flow_noise_scaling(&sigma, &noise, &x, 1.)),
            ("inverse", q::flow_inverse_noise_scaling(&sigma, &x)),
            (
                "cfg",
                q::cfg_combine(&output, &noise, c["scale"].as_f64().unwrap()),
            ),
        ] {
            check(&V::tensor(y.unwrap()), &c[key], dev, key);
        }
    }
    for c in fixture["noises"].as_array().unwrap() {
        let mut genr = native
            .generator(Device::Cpu, c["seed"].as_u64().unwrap())
            .unwrap();
        let noise = genr
            .randn(&[2, 64, 2, 3], Kind::Float)
            .unwrap()
            .to_kind(kind(c["dtype"].as_str().unwrap()));
        check(
            &V::tensor(noise),
            &c["output"],
            Device::Cpu,
            "CPU prepare_noise",
        );
        check(
            &V::tensor(genr.state().unwrap()),
            &c["state"],
            Device::Cpu,
            "CPU prepare_noise generator state",
        );
    }
    for c in fixture["solvers"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["dtype"].as_str().unwrap());
        let mut e = Exec {
            fixture: c,
            dev,
            step: 0,
            generator: native.generator(dev, 17).unwrap(),
        };
        let sigmas = q::FluxSampling::qwen21()
            .unwrap()
            .simple(4)
            .unwrap()
            .to_device(dev);
        let y = q::sample_euler(
            data(dev, dtype, 3),
            &sigmas,
            q::EulerSettings {
                churn: c["churn"].as_f64().unwrap(),
                tmin: 0.2,
                tmax: 0.95,
                noise: 0.875,
            },
            &mut e,
        )
        .unwrap();
        check(&V::tensor(y), &c["output"], dev, "Euler final");
        check(
            &V::tensor(e.generator.state().unwrap()),
            &c["state"],
            Device::Cpu,
            "Euler noise state",
        );
        assert_eq!(e.step, 4);
    }
    for c in fixture["full_conditions"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["dtype"].as_str().unwrap());
        let x = data(dev, dtype, 3);
        let out = data(dev, dtype, 7);
        let flat = out.flatten(0, -1);
        for (i, v) in [(-0.), 0., f64::NEG_INFINITY, f64::INFINITY]
            .into_iter()
            .enumerate()
        {
            let _ = flat.get(i as i64).fill_(v);
        }
        check(
            &V::tensor(q::single_full_condition_cfg1(&x, &out).unwrap()),
            &c["output"],
            dev,
            "Original full-area CFG1 aggregation including signed zero and infinities",
        );
    }
    for (scale, expected) in [
        (1., true),
        (1. + 0.5e-9, true),
        (1. + 2e-9, false),
        (f64::INFINITY, false),
        (f64::NAN, false),
    ] {
        assert_eq!(q::cfg1_shortcut(scale, false), expected);
        assert!(!q::cfg1_shortcut(scale, true));
    }
}
