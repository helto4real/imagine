use super::{
    binding_tests::{check, restore},
    kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    models::moss_image as m,
    native::{Error, Kitchen, SdpaOptions},
};
use serde_json::{Value, json};
use tch::{Device, Tensor};
struct Services {
    native: Kitchen,
    mode: String,
    free: f64,
    calls: usize,
    events: Vec<Value>,
}
fn failure(oom: bool) -> Error {
    Error::ReferenceException {
        class: if oom {
            "OutOfMemoryError"
        } else {
            "ValueError"
        },
        message: if oom {
            "synthetic OOM"
        } else {
            "synthetic non-OOM"
        }
        .into(),
    }
}
impl m::FluxVaeAttentionServices for Services {
    fn sdpa(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
        self.events.push(json!(["sdpa", q.size()]));
        if self.mode == "sdpa_error" {
            return Err(failure(false));
        }
        if self.mode != "plain" {
            return Err(failure(true));
        }
        self.native.comfy_sdpa(
            q,
            k,
            v,
            SdpaOptions {
                mask: None,
                scale: None,
                enable_gqa: false,
                priority_wrapper: false,
            },
        )
    }
    fn bmm(&mut self, a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
        self.calls += 1;
        self.events.push(json!(["bmm", a.size(), b.size()]));
        if self.mode == "bmm_error" {
            return Err(failure(false));
        }
        if self.mode == "bmm_oom_always"
            || (self.mode == "bmm_oom_first" && self.calls == 1)
            || (self.mode == "bmm_oom_second" && self.calls == 2)
        {
            return Err(failure(true));
        }
        Ok(a.f_bmm(b)?)
    }
    fn free_memory(&mut self, _: Device) -> Result<f64, Error> {
        self.events.push(json!(["free_memory"]));
        Ok(self.free)
    }
    fn is_oom(&mut self, _: Device, error: &Error) -> Result<bool, Error> {
        let oom = matches!(
            error,
            Error::ReferenceException {
                class: "OutOfMemoryError",
                ..
            }
        );
        self.events.push(json!(["is_oom", oom]));
        Ok(oom)
    }
    fn empty_cache(&mut self, _: Device) -> Result<(), Error> {
        self.events.push(json!(["empty_cache", true]));
        Ok(())
    }
    fn warning(&mut self, message: &str) {
        self.events.push(json!(["warning", message]));
    }
}
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0)
    }
}
#[test]
fn moss_original_flux_vae_attention_memory_slicing_typed_failures_and_retry_lifetimes_match() {
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-vae-attention.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let args = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cpu).as_tensor().unwrap().shallow_clone())
            .collect::<Vec<_>>();
        let mut services = Services {
            native: native.clone(),
            mode: case["mode"].as_str().unwrap().into(),
            free: case["free"].as_f64().unwrap(),
            calls: 0,
            events: vec![],
        };
        let result =
            m::flux2_vae_attention_with_services(&args[0], &args[1], &args[2], &mut services);
        if let Some(class) = case["error"].as_str() {
            match result.unwrap_err() {
                Error::ReferenceException {
                    class: actual,
                    message,
                } => {
                    assert_eq!(actual, class);
                    assert_eq!(message, case["message"].as_str().unwrap());
                }
                other => panic!("unexpected error {other:?}"),
            }
        } else {
            check(
                &V::tensor(result.unwrap()),
                &case["output"],
                Device::Cpu,
                "Flux VAE attention",
            );
        }
        assert_eq!(
            json!(services.events),
            case["events"],
            "mode {}",
            case["mode"]
        );
    }
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 36);
}
