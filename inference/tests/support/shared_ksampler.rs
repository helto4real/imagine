use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    execution_value::Value as V,
    models::qwen_image21::{EulerSettings, EulerStep},
    native::{Error, global_rng_state},
    sampling::{
        ksampler::{self as k, MaskFunction},
        parameterization::{Parameterization, Prediction as Mode},
        solvers::{self as s, Execution, NoiseKind, Prediction},
        unipc,
    },
};
use serde_json::Value;
use std::{cell::Cell, rc::Rc};
use tch::{Device, Kind, Tensor};
fn data(dev: Device, kind: Kind, offset: i64) -> Tensor {
    Tensor::arange(12, (Kind::Float, dev))
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .reshape([2, 1, 2, 3])
        .to_kind(kind)
}
struct Exec<'a> {
    case: &'a Value,
    calls: usize,
    callbacks: usize,
    mask_set: bool,
    event: Rc<Cell<usize>>,
}
impl Execution for Exec<'_> {
    fn set_denoise_mask(&mut self, mask: Option<&Tensor>) -> Result<(), Error> {
        assert_eq!(mask.is_none(), self.case["mask_mode"] == "none");
        self.mask_set = true;
        Ok(())
    }
    fn denoise(&mut self, x: &Tensor, sigma: &Tensor, cfg: bool) -> Result<Prediction, Error> {
        assert!(self.mask_set);
        let v = &self.case["calls"][self.calls];
        check(
            &V::tensor(x.shallow_clone()),
            &v["x"],
            x.device(),
            "Inpaint model input",
        );
        check(
            &V::tensor(sigma.shallow_clone()),
            &v["sigma"],
            x.device(),
            "Inpaint sigma",
        );
        assert_eq!(cfg, v["cfg_pp"].as_bool().unwrap());
        self.calls += 1;
        Ok(Prediction {
            denoised: x
                .f_mul_scalar(0.125)?
                .f_add(&sigma.f_reshape([2, 1, 1, 1])?.f_mul_scalar(0.25)?)?,
            uncond_denoised: None,
            cond_denoised: None,
        })
    }
    fn noise(
        &mut self,
        _x: &Tensor,
        _s: &Tensor,
        _t: &Tensor,
        _kind: NoiseKind,
    ) -> Result<Tensor, Error> {
        panic!("Unexpected random draw in deterministic boundary fixture")
    }
    fn callback(&mut self, _step: EulerStep<'_>) -> Result<(), Error> {
        panic!("KSAMPLER must adapt the callback contract")
    }
    fn sample_callback(
        &mut self,
        i: i64,
        x0: &Tensor,
        x: &Tensor,
        total: i64,
    ) -> Result<(), Error> {
        let v = &self.case["callbacks"][self.callbacks];
        assert_eq!(i, v["i"].as_i64().unwrap());
        assert_eq!(total, v["total"].as_i64().unwrap());
        check(
            &V::tensor(x0.shallow_clone()),
            &v["denoised"],
            x.device(),
            "Inpaint callback denoised",
        );
        check(
            &V::tensor(x.shallow_clone()),
            &v["x"],
            x.device(),
            "Inpaint callback state",
        );
        self.callbacks += 1;
        Ok(())
    }
    fn scale_latent_inpaint(
        &mut self,
        p: Parameterization,
        x: &Tensor,
        sigma: &Tensor,
        noise: &Tensor,
        latent: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor, Error> {
        let v = &self.case["events"][self.event.get()];
        assert_eq!(v["event"], "scale");
        for (name, tensor) in [
            ("x", x),
            ("sigma", sigma),
            ("noise", noise),
            ("latent", latent),
            ("mask", mask),
        ] {
            check(
                &V::tensor(tensor.shallow_clone()),
                &v[name],
                x.device(),
                name,
            );
        }
        self.event.set(self.event.get() + 1);
        let mut shape = vec![sigma.size()[0]];
        shape.resize(noise.dim(), 1);
        p.noise_scaling(&sigma.f_reshape(shape.as_slice())?, noise, latent, false)
    }
}
struct Mask<'a> {
    case: &'a Value,
    event: Rc<Cell<usize>>,
}
impl MaskFunction for Mask<'_> {
    fn apply(
        &mut self,
        sigma: &Tensor,
        mask: &Tensor,
        _exec: &mut dyn Execution,
        sigmas: &Tensor,
    ) -> Result<Tensor, Error> {
        assert_eq!(sigmas.size(), [5]);
        let v = &self.case["events"][self.event.get()];
        assert_eq!(v["event"], "mask");
        check(
            &V::tensor(sigma.shallow_clone()),
            &v["sigma"],
            sigma.device(),
            "Mask callback sigma",
        );
        check(
            &V::tensor(mask.shallow_clone()),
            &v["mask"],
            mask.device(),
            "Mask callback input",
        );
        self.event.set(self.event.get() + 1);
        Ok(mask.f_mul_scalar(0.5)?.f_add_scalar(0.125)?)
    }
}
#[test]
fn whole_ksampler_ddim_mask_hooks_callbacks_and_global_rng_match() {
    let native = kitchen();
    let _guard = tch::no_grad_guard();
    let old = native.default_dtype().unwrap();
    native.set_default_dtype(Kind::Float).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-ksampler.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["kind"].as_str().unwrap());
        let x = data(dev, dtype, 3);
        let latent = data(dev, dtype, 5);
        let mask = if c["mask_mode"] == "none" {
            None
        } else {
            Some(data(dev, Kind::Float, 7).abs())
        };
        let sigmas = Tensor::from_slice(&[0.9f32, 0.6, 0.3, 0.1, 0.]).to_device(dev);
        let event = Rc::new(Cell::new(0));
        let mut exec = Exec {
            case: c,
            calls: 0,
            callbacks: 0,
            mask_set: false,
            event: event.clone(),
        };
        let mut function = Mask {
            case: c,
            event: event.clone(),
        };
        let mode = Parameterization {
            prediction: if c["flow"] == true {
                Mode::Const
            } else {
                Mode::Eps
            },
            sigma_data: 1.,
            noise_scale: 0.875,
        };
        tch::manual_seed(17);
        let name = c["name"].as_str().unwrap();
        let y = k::sample(
            &mut exec,
            &native,
            k::Inputs {
                sigmas: &sigmas,
                noise: &x,
                latent: &latent,
                mask: mask.as_ref(),
                random_inpaint: name == "ddim",
                seed: k::Seed::Integer(901),
                maximum_sigma: 1.,
                parameterization: mode,
                mask_function: if c["mask_mode"] == "function" {
                    Some(&mut function)
                } else {
                    None
                },
            },
            |e, x, sigmas| {
                if name.starts_with("uni_pc") {
                    unipc::sample(
                        e,
                        &native,
                        x,
                        sigmas,
                        if name.ends_with("bh2") { "bh2" } else { "bh1" },
                    )
                } else {
                    s::churn(
                        e,
                        x,
                        sigmas,
                        EulerSettings {
                            churn: 0.,
                            tmin: 0.,
                            tmax: f64::INFINITY,
                            noise: 1.,
                        },
                        s::ChurnSolver::Euler,
                    )
                }
            },
        )
        .unwrap();
        check(
            &V::tensor(y),
            &c["output"],
            dev,
            &format!("whole sampler {c}"),
        );
        check(&V::tensor(sigmas), &c["sigmas"], dev, "Post-sampler sigmas");
        check(
            &V::tensor(global_rng_state(Device::Cpu).unwrap()),
            &c["cpu_rng"],
            Device::Cpu,
            "KSAMPLER global CPU RNG",
        );
        check(
            &V::tensor(global_rng_state(Device::Cuda(0)).unwrap()),
            &c["cuda_rng"],
            Device::Cpu,
            "KSAMPLER global CUDA RNG",
        );
        assert_eq!(exec.calls, c["calls"].as_array().unwrap().len());
        assert_eq!(exec.callbacks, c["callbacks"].as_array().unwrap().len());
        assert_eq!(event.get(), c["events"].as_array().unwrap().len());
    }
    native.set_default_dtype(old).unwrap();
}
