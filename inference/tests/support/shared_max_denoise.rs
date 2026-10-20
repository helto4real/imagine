use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::options::Options,
    sampling::factory::{self, MaxDenoiseBindings},
};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
struct Data {
    case: Value,
    events: RefCell<Vec<Value>>,
    index: Cell<u64>,
    inner: Cell<u64>,
    sampling: Cell<u64>,
    sigma: Cell<u64>,
}
impl Data {
    fn emit(&self, event: Value) -> Result<(), Error> {
        self.events.borrow_mut().push(event);
        let index = self.index.get();
        self.index.set(index + 1);
        if self.case["failure"].as_u64() == Some(index) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own maximum-denoise failure".into(),
            });
        }
        Ok(())
    }
}
struct Float {
    label: &'static str,
    bits: u64,
    data: Rc<Data>,
}
impl Drop for Float {
    fn drop(&mut self) {
        self.data
            .events
            .borrow_mut()
            .push(json!(["drop", self.label]));
    }
}
struct Bindings(Rc<Data>);
fn role(value: &str) -> Options {
    Options::Atom(V::Text(value.into()))
}
impl MaxDenoiseBindings for Bindings {
    fn attribute(&mut self, object: &Options, name: &str) -> Result<Options, Error> {
        let Options::Atom(V::Text(label)) = object else {
            panic!("own attribute")
        };
        let inner = self.0.inner.get();
        let sampling = self.0.sampling.get();
        self.0.emit(json!(["attribute", label, name]))?;
        match name {
            "inner_model" => {
                if self.0.case["action"] == "replace-inner" {
                    self.0.inner.set(1);
                }
                Ok(role(&format!("inner{inner}")))
            }
            "model_sampling" => {
                if self.0.case["action"] == "replace-sampling" {
                    self.0.sampling.set(1);
                }
                Ok(role(&format!("sampling{sampling}")))
            }
            "sigma_max" => Ok(Options::Object(Rc::new(Float {
                label: "maximum",
                bits: if label == "sampling0" {
                    self.0.case["maximum"].as_u64().unwrap()
                } else {
                    2f64.to_bits()
                },
                data: self.0.clone(),
            }))),
            _ => panic!("own field"),
        }
    }
    fn first_sigma(&mut self, _: &Options) -> Result<Options, Error> {
        self.0.emit(json!(["index", 0]))?;
        Ok(Options::Object(Rc::new(Float {
            label: "sigma",
            bits: self.0.sigma.get(),
            data: self.0.clone(),
        })))
    }
    fn float(&mut self, value: &Options) -> Result<f64, Error> {
        let Options::Object(value) = value else {
            panic!("own float")
        };
        let value = value.downcast_ref::<Float>().unwrap();
        self.0.emit(json!(["float", value.label, value.bits]))?;
        if value.label == "maximum" && self.0.case["action"] == "replace-sigma" {
            self.0.sigma.set(2f64.to_bits());
        }
        Ok(f64::from_bits(value.bits))
    }
}
#[test]
fn original_maximum_denoise_preserves_nonfinite_subnormal_tolerance_boundaries_attribute_order_temporary_release_and_rebinding()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-max-denoise.json")).unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let data = Rc::new(Data {
            case: case.clone(),
            events: RefCell::new(vec![]),
            index: Cell::new(0),
            inner: Cell::new(0),
            sampling: Cell::new(0),
            sigma: Cell::new(case["sigma"].as_u64().unwrap()),
        });
        let result =
            factory::max_denoise(&role("model"), &role("sigmas"), &mut Bindings(data.clone()));
        match result {
            Ok(value) => {
                assert!(case["error"].is_null());
                assert_eq!(json!(value), case["result"], "{index}");
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "{index}")
            }
            Err(error) => panic!("unbound maximum denoise {error:?}"),
        }
        assert_eq!(
            json!(*data.events.borrow()),
            case["events"],
            "events {index}"
        );
        assert_eq!(json!(data.inner.get()), case["inner"]);
        assert_eq!(json!(data.sampling.get()), case["sampling"]);
        assert_eq!(json!(data.sigma.get()), case["current_sigma"]);
    }
}
