use imagine_inference::{
    memory::pinning::{PinBindings, PinRegistry},
    native::Error,
};
use serde_json::{Value, json};
#[test]
fn original_native_host_registration_matches_six_bounded_original_cuda_cases_without_preinitializing_torch_cuda()
 {
    use imagine_inference::models::qwen_image21::HostPins;
    use tch::{Device, Kind, Tensor};
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-pin-registry-cuda.json")).unwrap();
    let native = super::kitchen();
    for case in fixture["cases"].as_array().unwrap() {
        let pins = HostPins::with_limit(native.clone(), 65536.).unwrap();
        let kind = if case["dtype"] == "torch.bfloat16" {
            Kind::BFloat16
        } else {
            Kind::Float
        };
        let tensor = Tensor::ones([case["elements"].as_i64().unwrap()], (kind, Device::Cpu));
        let before = tensor.f_is_pinned(Device::Cuda(0)).unwrap();
        let pin = pins.pin(&tensor).unwrap();
        let during = tensor.f_is_pinned(Device::Cuda(0)).unwrap();
        let total_during = pins.bytes();
        let unpin = pins.unpin(&tensor).unwrap();
        let after = tensor.f_is_pinned(Device::Cuda(0)).unwrap();
        let total_after = pins.bytes();
        assert_eq!(
            json!({"dtype":case["dtype"],"elements":case["elements"],"bytes":tensor.numel()*kind.elt_size_in_bytes(),"before":before,"pin":pin,"during":during,"total_during":total_during,"unpin":unpin,"after":after,"total_after":total_after}),
            *case
        );
    }
}
struct Bindings {
    case: Value,
    events: Vec<Value>,
    setup: bool,
}
impl Bindings {
    fn emit(&mut self, event: Value) -> Result<(), Error> {
        if self.setup {
            return Ok(());
        }
        let index = self.events.len();
        self.events.push(event);
        if self.case["failure"].as_u64() == Some(index as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own registration failure".into(),
            });
        }
        Ok(())
    }
    fn boolean(&self, key: &str) -> bool {
        self.setup || self.case["config"][key].as_bool().unwrap()
    }
}
impl PinBindings for Bindings {
    type Tensor = ();
    fn allowed_type(&mut self, _: &()) -> Result<bool, Error> {
        Ok(self.boolean("allowed"))
    }
    fn is_cpu(&mut self, _: &()) -> Result<bool, Error> {
        self.emit(json!(["cpu"]))?;
        Ok(self.boolean("cpu"))
    }
    fn is_pinned(&mut self, _: &()) -> Result<bool, Error> {
        self.emit(json!(["pinned"]))?;
        Ok(!self.setup && self.boolean("pinned"))
    }
    fn is_contiguous(&mut self, _: &()) -> Result<bool, Error> {
        self.emit(json!(["contiguous"]))?;
        Ok(self.boolean("contiguous"))
    }
    fn nbytes(&mut self, _: &()) -> Result<usize, Error> {
        self.emit(json!(["bytes"]))?;
        Ok(if self.setup {
            32
        } else {
            self.case["config"]["size"].as_u64().unwrap() as usize
        })
    }
    fn extra_ram_release(&mut self) -> Result<(), Error> {
        self.emit(json!(["release"]))
    }
    fn ensure_registerable(&mut self, size: usize, evict: bool) -> Result<bool, Error> {
        self.emit(json!(["ensure", size, evict]))?;
        Ok(self.boolean("ensure"))
    }
    fn pointer(&mut self, _: &()) -> Result<usize, Error> {
        self.emit(json!(["pointer"]))?;
        Ok(if self.setup {
            7
        } else {
            self.case["config"]["pointer"].as_u64().unwrap() as usize
        })
    }
    fn register(&mut self, ptr: usize, size: usize) -> Result<bool, Error> {
        self.emit(json!(["register", ptr, size]))?;
        Ok(self.boolean("register"))
    }
    fn unregister(&mut self, ptr: usize) -> Result<bool, Error> {
        self.emit(json!(["unregister", ptr]))?;
        Ok(self.boolean("unregister"))
    }
    fn warning(&mut self, message: &'static str) -> Result<(), Error> {
        self.emit(json!(["warning", message]))
    }
    fn discard_async_error(&mut self) -> Result<(), Error> {
        self.emit(json!(["discard"]))
    }
}
#[test]
fn original_host_pin_registry_preserves_guards_pointer_size_accounting_warning_recovery_and_failure_prefixes_without_tensor_ownership()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-pin-registry.json")).unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let registry = PinRegistry::new(64.);
        let mut bindings = Bindings {
            case: case.clone(),
            events: vec![],
            setup: true,
        };
        if case["initial"] == true {
            assert!(registry.pin(&(), true, &mut bindings).unwrap());
        }
        registry
            .limit
            .set(case["config"]["limit"].as_f64().unwrap());
        bindings.setup = false;
        let result = if case["mode"] == "pin" {
            registry.pin(&(), case["evict"].as_bool().unwrap(), &mut bindings)
        } else {
            registry.unpin(&(), &mut bindings)
        };
        match result {
            Ok(value) => {
                assert!(case["error"].is_null(), "{index}");
                assert_eq!(json!(value), case["result"], "{index}");
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "{index}")
            }
            Err(error) => panic!("unbound {index}: {error:?}"),
        }
        assert_eq!(json!(bindings.events), case["events"], "events {index}");
        assert_eq!(json!(registry.bytes()), case["total"], "total {index}");
        let sizes: Vec<_> = [0, 7]
            .into_iter()
            .filter_map(|ptr| registry.registered_size(ptr).map(|size| (ptr, size)))
            .collect();
        assert_eq!(json!(sizes), case["sizes"], "sizes {index}");
    }
}
