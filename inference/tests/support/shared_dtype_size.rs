use imagine_inference::{
    execution_value::Value as V,
    memory::dtype::{DtypeSizeBindings, dtype_size},
    native::Error,
    patching::options::Options,
};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
use tch::Kind;
fn native_kinds() -> [Kind; 20] {
    [
        Kind::Uint8,
        Kind::Int8,
        Kind::Int16,
        Kind::Int,
        Kind::Int64,
        Kind::Half,
        Kind::Float,
        Kind::Double,
        Kind::ComplexHalf,
        Kind::ComplexFloat,
        Kind::ComplexDouble,
        Kind::Bool,
        Kind::QInt8,
        Kind::QUInt8,
        Kind::QInt32,
        Kind::BFloat16,
        Kind::Float8e5m2,
        Kind::Float8e4m3fn,
        Kind::Float8e5m2fnuz,
        Kind::Float8e4m3fnuz,
    ]
}
#[test]
fn original_dtype_size_matches_all_twenty_locked_native_kinds_without_tensor_or_cuda_allocation() {
    use imagine_inference::memory::dtype::KindDtypeSizeBindings;
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dtype-size-kinds.json")).unwrap();
    assert_eq!(fixture["cuda_initialized"], false);
    assert_eq!(fixture["fixture_tensors"], 0);
    let kinds = native_kinds();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), kinds.len());
    for (kind, case) in kinds.into_iter().zip(cases) {
        assert_eq!(format!("{kind:?}"), case["kind"].as_str().unwrap());
        let Options::Atom(V::Int(size)) =
            dtype_size(&Options::Atom(V::DType(kind)), &KindDtypeSizeBindings).unwrap()
        else {
            panic!("original dtype returns an integer size")
        };
        assert_eq!(size, case["size"].as_i64().unwrap());
    }
    assert!(
        matches!(
            dtype_size(&Options::Object(Rc::new(())), &KindDtypeSizeBindings),
            Err(Error::BindingGap { .. })
        ),
        "unknown object requires its source protocols"
    );
}
struct Bindings {
    case: Value,
    events: RefCell<Vec<Value>>,
    sentinel: Options,
    gap: bool,
}
impl Bindings {
    fn emit(&self, event: Value) -> Result<(), Error> {
        let mut events = self.events.borrow_mut();
        events.push(event);
        if self.case["failure"].as_u64() == Some((events.len() - 1) as u64) {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own dtype-size failure".into(),
            });
        }
        Ok(())
    }
}
impl DtypeSizeBindings for Bindings {
    fn equals(&self, _: &Options, dtype: Kind) -> Result<bool, Error> {
        let index = match dtype {
            Kind::Half => 0,
            Kind::BFloat16 => 1,
            Kind::Float => 2,
            _ => panic!("own dtype constant"),
        };
        self.emit(json!(["eq", index]))?;
        self.emit(json!(["bool", index]))?;
        Ok(self.case["mask"].as_u64().unwrap() & (1 << index) != 0)
    }
    fn itemsize(&self, _: &Options) -> Result<Options, Error> {
        if self.gap {
            return Err(Error::BindingGap {
                backend: "own unbound attribute provider".into(),
                operation: "itemsize".into(),
            });
        }
        self.emit(json!(["itemsize"]))?;
        Ok(match self.case["attribute"].as_str().unwrap() {
            "eight" => Options::Atom(V::Int(8)),
            "negative" => Options::Atom(V::Int(-7)),
            "none" => Options::Atom(V::None),
            "false" => Options::Atom(V::Bool(false)),
            "object" => self.sentinel.clone(),
            class => {
                let class = match class {
                    "AttributeError" => "AttributeError",
                    "RuntimeError" => "RuntimeError",
                    "KeyboardInterrupt" => "KeyboardInterrupt",
                    "SystemExit" => "SystemExit",
                    "BaseException" => "BaseException",
                    _ => panic!("own class"),
                };
                return Err(Error::ReferenceException {
                    class,
                    message: "own itemsize failure".into(),
                });
            }
        })
    }
}
#[test]
fn original_dtype_size_preserves_equality_truth_short_circuits_attribute_identity_and_bare_exception_fallback_without_hiding_binding_gaps()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-dtype-size.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 370);
    let value = Options::Object(Rc::new("own dtype"));
    for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let bindings = Bindings {
            case: case.clone(),
            events: RefCell::new(vec![]),
            sentinel: Options::Object(Rc::new(())),
            gap: false,
        };
        match dtype_size(&value, &bindings) {
            Ok(output) => {
                assert!(case["error"].is_null(), "source error {i}");
                let result = match output {
                    Options::Atom(V::None) => Value::Null,
                    Options::Atom(V::Int(v)) => json!(v),
                    Options::Atom(V::Bool(v)) => json!(v),
                    object => {
                        assert!(object.same_object(&bindings.sentinel));
                        json!("same-object")
                    }
                };
                assert_eq!(result, case["output"], "output {i}");
            }
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!([class, message]), case["error"], "error {i}")
            }
            Err(error) => panic!("unbound dtype size {i}: {error:?}"),
        }
        assert_eq!(
            json!(*bindings.events.borrow()),
            case["events"],
            "events {i}"
        );
    }
    let bindings = Bindings {
        case: json!({"mask":0}),
        events: RefCell::new(vec![]),
        sentinel: Options::Object(Rc::new(())),
        gap: true,
    };
    assert!(
        matches!(dtype_size(&value, &bindings), Err(Error::BindingGap { .. })),
        "incomplete provider is not a source exception"
    );
}

#[test]
fn original_dtype_size_registered_type_metadata_preserves_twenty_kinds_two_shells_and_builtin_class_fallbacks()
 {
    use imagine_inference::{
        memory::dtype::RegisteredDtypeSizeBindings,
        native::ShellDType,
        registry::{AllowedType, ScalarType},
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/shared-dtype-size-registered.json"
    ))
    .unwrap();
    assert_eq!(fixture["cuda_initialized"], false);
    assert_eq!(fixture["fixture_tensors"], 0);
    let mut types = native_kinds()
        .into_iter()
        .map(|kind| (format!("{kind:?}"), AllowedType::DType(kind)))
        .collect::<Vec<_>>();
    types.extend([
        (
            "ShellE8M0".into(),
            AllowedType::ShellDType(ShellDType::E8M0),
        ),
        (
            "ShellPackedE2M1".into(),
            AllowedType::ShellDType(ShellDType::PackedE2M1),
        ),
        ("ScalarInt".into(), AllowedType::Scalar(ScalarType::Int)),
        ("ScalarFloat".into(), AllowedType::Scalar(ScalarType::Float)),
        ("ScalarBool".into(), AllowedType::Scalar(ScalarType::Bool)),
        ("ScalarStr".into(), AllowedType::Scalar(ScalarType::Str)),
        (
            "ScalarNoneType".into(),
            AllowedType::Scalar(ScalarType::NoneType),
        ),
    ]);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), types.len());
    for ((name, kind), case) in types.into_iter().zip(cases) {
        assert_eq!(name, case["kind"].as_str().unwrap());
        let value = Options::Object(Rc::new(kind));
        let Options::Atom(V::Int(size)) = dtype_size(&value, &RegisteredDtypeSizeBindings).unwrap()
        else {
            panic!("registered type has an integer source size")
        };
        assert_eq!(size, case["size"].as_i64().unwrap());
    }
    assert!(matches!(
        dtype_size(&Options::Object(Rc::new(())), &RegisteredDtypeSizeBindings),
        Err(Error::BindingGap { .. })
    ));
}
