use super::{
    binding_tests::{check, reference_error},
    device, kind, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::{Error, global_rng_state},
    patching::options::Options,
    sampling::ar_video::{self, Bindings, Inputs},
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::{Device, Kind, Tensor};
fn integer(v: i64) -> Options {
    Options::Atom(V::Int(v))
}
fn role(v: &str) -> Options {
    Options::Atom(V::Text(v.into()))
}
fn number(v: &Options) -> i64 {
    let Options::Atom(V::Int(v)) = v else {
        panic!("own int")
    };
    *v
}
fn end_number(v: &Options) -> i64 {
    match v {
        Options::Atom(V::Tensor(_)) => tensor(v).int64_value(&[]),
        _ => number(v),
    }
}
fn text(v: &Options) -> &str {
    let Options::Atom(V::Text(v)) = v else {
        panic!("own role")
    };
    v
}
fn data(shape: &[i64], device: Device, kind: Kind, offset: i64) -> Tensor {
    Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device))
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .reshape(shape)
        .to_kind(kind)
}
fn tensor(v: &Options) -> Tensor {
    let Options::Atom(v) = v else {
        panic!("own tensor")
    };
    v.as_tensor().unwrap()
}
fn cache(tag: i64, end: i64, tensor_end: bool) -> Options {
    let end = if tensor_end {
        Options::Atom(V::tensor(Tensor::from(end)))
    } else {
        integer(end)
    };
    Options::dictionary(vec![("end".into(), end), ("tag".into(), integer(tag))])
}

struct Execution<'a> {
    case: &'a Value,
    index: usize,
    kind: Kind,
    transformer: Options,
    caches: Option<Options>,
    source_device: Option<Options>,
    models: usize,
}
impl Execution<'_> {
    fn emit(&mut self, metadata: Value, tensors: &[(&str, &Tensor)]) -> Result<(), Error> {
        let expected = self.case["events"]
            .as_array()
            .unwrap()
            .get(self.index)
            .unwrap_or_else(|| panic!("unexpected AR event {metadata}"));
        let mut rest = expected.clone();
        for (key, value) in tensors {
            let stamp = rest
                .as_object_mut()
                .unwrap()
                .remove(*key)
                .expect("source tensor field");
            check(
                &V::tensor(value.shallow_clone()),
                &stamp,
                value.device(),
                &format!("AR {} {metadata}", self.index),
            );
        }
        assert_eq!(metadata, rest, "AR event {}", self.index);
        let index = self.index;
        self.index += 1;
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own AR-video failure".into(),
            });
        }
        Ok(())
    }
    fn ends(&self) -> Vec<i64> {
        let Some(Options::List(caches)) = &self.caches else {
            return vec![];
        };
        caches
            .borrow()
            .iter()
            .map(|cache| end_number(&cache.get("end").unwrap().unwrap()))
            .collect()
    }
}
impl Bindings for Execution<'_> {
    fn get(&mut self, object: &Options, key: &str, default: Options) -> Result<Options, Error> {
        Ok(object.get(key)?.unwrap_or(default))
    }
    fn set(&mut self, object: &Options, key: &str, value: Options) -> Result<(), Error> {
        if object.same_object(&self.transformer) && key == "ar_state" {
            self.emit(
                json!({"event":"state","start":number(&value.get("start_frame")?.unwrap())}),
                &[],
            )?;
        }
        object.set(key, value)
    }
    fn pop(&mut self, object: &Options, key: &str) -> Result<(), Error> {
        if object.same_object(&self.transformer) {
            self.emit(json!({"event":"pop"}), &[])?;
        }
        drop(object.pop(key)?);
        Ok(())
    }
    fn attribute(&mut self, object: &Options, name: &str) -> Result<Options, Error> {
        Ok(role(match (text(object), name) {
            ("model", "inner_model") => "outer",
            ("outer", "inner_model") => "inner",
            ("inner", "diffusion_model") => "causal",
            _ => panic!("own AR attribute"),
        }))
    }
    fn has_attribute(&mut self, object: &Options, name: &str) -> Result<bool, Error> {
        assert_eq!(text(object), "causal");
        Ok(match name {
            "init_kv_caches" => self.case["mode"] != "no-kv",
            "init_crossattn_caches" => self.case["mode"] != "no-cross",
            _ => panic!("own capability"),
        })
    }
    fn tensor_device(&mut self, tensor: &Tensor) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(tensor.device())))
    }
    fn method(&mut self, object: &Options, name: &str, args: &[Options]) -> Result<Options, Error> {
        match (text(object), name) {
            ("inner", "get_dtype") => {
                self.emit(json!({"event":"dtype"}), &[])?;
                Ok(Options::Atom(V::DType(self.kind)))
            }
            ("inner", "process_latent_in") => {
                let value = tensor(&args[0]);
                self.emit(json!({"event":"process"}), &[("value", &value)])?;
                Ok(Options::Atom(V::tensor(value.f_mul_scalar(0.5)?)))
            }
            ("causal", "init_kv_caches") => {
                let Options::Object(device) = &args[2] else {
                    panic!("source device identity")
                };
                let device = *device.downcast_ref::<Device>().unwrap();
                self.emit(json!({"event":"kv","batch":number(&args[0]),"length":number(&args[1]),"device":format!("{device:?}").to_lowercase().replace("cuda(0)","cuda:0"),"dtype":self.case["kind"]}),&[])?;
                self.source_device = Some(args[2].clone());
                let caches = Options::List(Rc::new(RefCell::new(vec![
                    cache(
                        0,
                        0,
                        matches!(
                            self.case["mode"].as_str(),
                            Some("tensor-cache" | "append-tensor")
                        ),
                    ),
                    cache(
                        1,
                        3,
                        matches!(
                            self.case["mode"].as_str(),
                            Some("tensor-cache" | "append-tensor")
                        ),
                    ),
                ])));
                self.caches = Some(caches.clone());
                Ok(caches)
            }
            ("causal", "init_crossattn_caches") => {
                assert!(
                    args[1].same_object(self.source_device.as_ref().unwrap()),
                    "the same source device object reaches both cache constructors"
                );
                let Options::Object(device) = &args[1] else {
                    panic!("source device")
                };
                let device = *device.downcast_ref::<Device>().unwrap();
                self.emit(json!({"event":"cross","batch":number(&args[0]),"device":format!("{device:?}").to_lowercase().replace("cuda(0)","cuda:0"),"dtype":self.case["kind"]}),&[])?;
                Ok(Options::List(Rc::new(RefCell::new(vec![
                    Options::dictionary(vec![("own".into(), Options::Atom(V::Bool(true)))]),
                ]))))
            }
            _ => panic!("own AR method {name}"),
        }
    }
    fn cast_latent(
        &mut self,
        value: &Options,
        device: Device,
        dtype: &Options,
    ) -> Result<Tensor, Error> {
        let Options::Atom(V::DType(dtype)) = dtype else {
            panic!("dtype")
        };
        Ok(tensor(value).f_to_device_(device, *dtype, false, false)?)
    }
    fn model(
        &mut self,
        model: &Options,
        x: &Tensor,
        sigma: &Tensor,
        args: &Options,
    ) -> Result<Tensor, Error> {
        assert_eq!(text(model), "model");
        let start = args
            .get("model_options")?
            .and_then(|options| options.get("transformer_options").unwrap())
            .and_then(|options| options.get("ar_state").unwrap())
            .map(|state| number(&state.get("start_frame").unwrap().unwrap()));
        self.emit(
            json!({"event":"model","start":start,"ends":self.ends()}),
            &[("value", x), ("sigma", sigma)],
        )?;
        let Some(Options::List(caches)) = &self.caches else {
            panic!("initialized KV")
        };
        if matches!(self.case["mode"].as_str(), Some("append" | "append-tensor"))
            && self.models == 0
        {
            caches
                .borrow_mut()
                .push(cache(2, 7, self.case["mode"] == "append-tensor"));
        }
        self.models += 1;
        for cache in caches.borrow().iter() {
            let end = cache.get("end")?.unwrap();
            let value = if matches!(end, Options::Atom(V::Tensor(_))) {
                Options::Atom(V::tensor(tensor(&end).f_add_scalar(x.size()[2] * 4)?))
            } else {
                integer(number(&end) + x.size()[2] * 4)
            };
            cache.set("end", value)?;
        }
        Ok(x.f_mul_scalar(0.125)?
            .f_add(&sigma.f_reshape([-1, 1, 1, 1, 1])?.f_mul_scalar(0.25)?)?)
    }
    fn callback(&mut self, callback: &Options, fields: Options) -> Result<Options, Error> {
        assert_eq!(text(callback), "callback");
        let value = tensor(&fields.get("x")?.unwrap());
        let sigma = tensor(&fields.get("sigma")?.unwrap());
        let hat = tensor(&fields.get("sigma_hat")?.unwrap());
        let denoised = tensor(&fields.get("denoised")?.unwrap());
        self.emit(
            json!({"event":"callback","index":number(&fields.get("i")?.unwrap())}),
            &[
                ("value", &value),
                ("sigma", &sigma),
                ("hat", &hat),
                ("denoised", &denoised),
            ],
        )?;
        Ok(Options::Atom(V::None))
    }
    fn reseed(&mut self, seed: &Options, block: i64, step: i64) -> Result<(), Error> {
        let value = match seed {
            Options::Atom(V::Int(v)) => *v as i128,
            Options::Atom(V::UInt(v)) => *v as i128,
            Options::Atom(V::None) => {
                return Err(Error::ReferenceException {
                    class: "TypeError",
                    message: "unsupported operand type(s) for +: 'NoneType' and 'int'".into(),
                });
            }
            _ => panic!("own seed"),
        };
        let value = value + block as i128 * 1000 + step as i128;
        if value < i64::MIN as i128 || value > u64::MAX as i128 {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "Overflow when unpacking long long".into(),
            });
        }
        tch::manual_seed(value as i64);
        Ok(())
    }
    fn cache_end(&mut self, cache: &Options) -> Result<Options, Error> {
        let end = cache.get("end")?.unwrap();
        let event = json!({"event":"cache-get","tag":number(&cache.get("tag")?.unwrap()),"value":end_number(&end)});
        if matches!(end, Options::Atom(V::Tensor(_))) {
            let value = tensor(&end);
            self.emit(event, &[("tensor", &value)])?;
        } else {
            self.emit(event, &[])?;
        }
        Ok(end)
    }
    fn inplace_subtract(&mut self, value: &Options, amount: i64) -> Result<Options, Error> {
        if matches!(value, Options::Atom(V::Tensor(_))) {
            let _ = tensor(value).f_sub_scalar_(amount)?;
            Ok(value.clone())
        } else {
            Ok(integer(number(value) - amount))
        }
    }
    fn set_cache_end(&mut self, cache: &Options, value: Options) -> Result<(), Error> {
        let mut event = json!({"event":"cache-set","tag":number(&cache.get("tag")?.unwrap()),"value":end_number(&value)});
        if matches!(value, Options::Atom(V::Tensor(_))) {
            let old = cache.get("end")?.unwrap();
            event["old_value"] = json!(end_number(&old));
            event["same_tensor"] = json!(old.same_object(&value));
            let tensor = tensor(&value);
            self.emit(event, &[("tensor", &tensor)])?;
        } else {
            self.emit(event, &[])?;
        }
        cache.set("end", value)
    }
}
#[test]
fn original_auxiliary_ar_video_preserves_block_cache_rollback_i2v_priming_seed_rng_callback_scaling_and_finally_error_boundaries()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-ar-video.json")).unwrap();
    let native = kitchen();
    native.cuda_memory(0).unwrap();
    let _grad = tch::no_grad_guard();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let device = device(case["device"].as_str().unwrap());
        let dtype = kind(case["kind"].as_str().unwrap());
        let frames = case["frames"].as_i64().unwrap();
        let shape = if case["mode"] == "rank" {
            vec![1, 1, 3, 3]
        } else {
            vec![1, 1, frames, 3, 3]
        };
        let x = data(&shape, device, dtype, 3);
        let transformer = Options::dictionary(vec![("ar_state".into(), role("old-state"))]);
        if let Some(initial) = case["initial"].as_i64() {
            transformer
                .set(
                    "ar_config",
                    Options::dictionary(vec![(
                        "initial_latent".into(),
                        Options::Atom(V::tensor(data(
                            &[1, 1, initial, 3, 3],
                            device,
                            Kind::Float,
                            5,
                        ))),
                    )]),
                )
                .unwrap();
        }
        let model_options =
            Options::dictionary(vec![("transformer_options".into(), transformer.clone())]);
        let mut extra_args = Options::dictionary(vec![("model_options".into(), model_options)]);
        if case["mode"] != "missing-seed" {
            let seed = if case["seed"].is_null() {
                Options::Atom(V::None)
            } else if let Some(v) = case["seed"].as_i64() {
                integer(v)
            } else {
                Options::Atom(V::UInt(case["seed"].as_u64().unwrap()))
            };
            extra_args.set("seed", seed).unwrap();
        }
        if case["mode"] == "no-args" {
            extra_args = Options::Atom(V::None);
        }
        let sigmas = match case["mode"].as_str().unwrap() {
            "one-sigma" => Tensor::from_slice(&[0.9f32]),
            "empty-sigmas" => Tensor::from_slice(&[] as &[f32]),
            "scalar-sigma" => Tensor::from(0.9f32),
            _ => Tensor::from_slice(&[0.9f32, 0.6, 0.2, 0.]),
        }
        .to_device(device);
        let mut execution = Execution {
            case,
            index: 0,
            kind: dtype,
            transformer: transformer.clone(),
            caches: None,
            source_device: None,
            models: 0,
        };
        let model = role("model");
        let callback = role("callback");
        tch::manual_seed(17);
        let result = ar_video::sample(
            &native,
            Inputs {
                model: &model,
                x: &x,
                sigmas: &sigmas,
                extra_args,
                callback: case["callback"].as_bool().unwrap().then_some(&callback),
                frames_per_block: case["block"].as_i64().unwrap(),
            },
            &mut execution,
        );
        if case["error"].is_object() {
            reference_error(
                result
                    .err()
                    .unwrap_or_else(|| panic!("expected AR error {index}")),
                &json!({"error":case["error"]["class_name"],"message":case["error"]["message"]}),
            );
        } else {
            check(
                &V::tensor(result.unwrap_or_else(|e| panic!("AR {index}: {e:?}"))),
                &case["output"],
                device,
                &format!("AR {index}"),
            );
        }
        assert_eq!(
            execution.index,
            case["events"].as_array().unwrap().len(),
            "all AR events {index}"
        );
        assert_eq!(json!(execution.ends()), case["ends"], "cache ends {index}");
        let state = transformer
            .get("ar_state")
            .unwrap()
            .map(|value| match value {
                Options::Atom(V::Text(value)) => json!(value),
                value => json!(number(&value.get("start_frame").unwrap().unwrap())),
            })
            .unwrap_or(Value::Null);
        assert_eq!(state, case["state"], "state {index}");
        check(
            &V::tensor(global_rng_state(Device::Cpu).unwrap()),
            &case["cpu_rng"],
            Device::Cpu,
            "AR CPU RNG",
        );
        check(
            &V::tensor(global_rng_state(Device::Cuda(0)).unwrap()),
            &case["cuda_rng"],
            Device::Cpu,
            "AR CUDA RNG state tensor",
        );
    }
}
