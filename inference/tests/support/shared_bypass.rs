use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    execution_value::Value as V,
    native::Error,
    patching::{
        Adapter, Diagnostics,
        bypass::{self as b, Convolution, Settings},
        options::Options,
    },
};
use serde_json::{Value, json};
use tch::{Device, Tensor};
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
fn shape(value: Options) -> Vec<i64> {
    match value.snapshot().unwrap() {
        V::List(v) | V::Tuple(v) => v
            .into_iter()
            .map(|n| match n {
                V::Int(n) => n,
                _ => panic!("Own dimension"),
            })
            .collect(),
        _ => panic!("Own shape"),
    }
}
fn settings(info: &Value, multiplier: f64) -> Settings {
    let info = Options::from_value(restore(info, Device::Cpu));
    let V::Bool(conv) = info.get("is_conv").unwrap().unwrap().snapshot().unwrap() else {
        panic!("Conv flag")
    };
    let convolution = conv.then(|| {
        let integer = |key: &str| match info.get(key).unwrap().unwrap().snapshot().unwrap() {
            V::Int(n) => n,
            _ => panic!("Own integer"),
        };
        Convolution {
            dimension: integer("conv_dim") as usize,
            kernel_size: shape(info.get("kernel_size").unwrap().unwrap()),
            in_channels: match info
                .get("in_channels")
                .unwrap()
                .unwrap()
                .snapshot()
                .unwrap()
            {
                V::Int(n) => Some(n),
                V::None => None,
                _ => panic!("Own channels"),
            },
            stride: shape(info.get("stride").unwrap().unwrap()),
            padding: shape(info.get("padding").unwrap().unwrap()),
            dilation: shape(info.get("dilation").unwrap().unwrap()),
            groups: integer("groups"),
        }
    });
    Settings {
        multiplier,
        convolution,
    }
}
fn compare(output: Result<Tensor, Error>, case: &Value, dev: Device, note: &str) {
    if case["error"].is_string() {
        reference_error(
            output.err().unwrap_or_else(|| panic!("Expected {note}")),
            case,
        )
    } else {
        check(
            &V::tensor(output.unwrap_or_else(|e| panic!("{note}: {e}"))),
            &case["output"],
            dev,
            note,
        )
    }
}
fn original(x: &Tensor, w: &Tensor, settings: &Settings) -> Result<Tensor, Error> {
    let Some(c) = &settings.convolution else {
        return Ok(x.f_linear(w, None::<&Tensor>)?);
    };
    // The synthetic base uses the original physical channels even in the
    // intentionally unknown metadata profile.
    Ok(match c.dimension {
        1 => x.f_conv1d(w, None::<&Tensor>, &c.stride, &c.padding, [1], 1)?,
        2 => x.f_conv2d(w, None::<&Tensor>, &c.stride, &c.padding, [1, 1], 1)?,
        3 => x.f_conv3d(w, None::<&Tensor>, &c.stride, &c.padding, [1, 1, 1], 1)?,
        _ => unreachable!(),
    })
}
#[test]
fn inference_bypass_adapter_algebra_preserves_six_families_linear_convolutions_dtype_errors_and_base_calls()
 {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(1);
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-bypass.json")).unwrap();
    assert_eq!(fixture["inputs"].as_array().unwrap().len(), 696);
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 2088);
    run_fixture(&fixture);
}
#[test]
fn original_bypass_tensor_alpha_constraints_broadcasting_zero_ranks_dtypes_and_errors_match() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(1);
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-bypass-alpha.json")).unwrap();
    assert_eq!(fixture["inputs"].as_array().unwrap().len(), 2040);
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 4080);
    // The corresponding original oracle explicitly invokes cache_clear().
    b::clear_loha_warning_cache();
    run_fixture(&fixture);
}
fn run_fixture(fixture: &Value) {
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let input = &fixture["inputs"][case["input_index"].as_u64().unwrap() as usize];
        let dev = device(input["device"].as_str().unwrap());
        let V::Tuple(weights) = restore(&input["weights"], dev) else {
            panic!("Own source adapter tuple")
        };
        let adapter = Adapter {
            name: input["family"].as_str().unwrap().into(),
            weights,
        };
        let settings = settings(&input["info"], case["multiplier"].as_f64().unwrap());
        let x = restore(&input["input"], dev).as_tensor().unwrap();
        let base = restore(&input["base_out"], dev).as_tensor().unwrap();
        let w = restore(&input["base_weight"], dev).as_tensor().unwrap();
        let mut diagnostics = Diagnostics::default();
        let note = format!(
            "bypass {index} {} {} {} {}",
            input["family"], input["variant"], input["device"], input["kind"]
        );
        compare(
            b::additive(&adapter, &x, &base, &settings, &mut diagnostics),
            &case["h"],
            dev,
            &format!("{note} h"),
        );
        compare(
            b::transform(&adapter, &base, &settings),
            &case["g"],
            dev,
            &format!("{note} g"),
        );
        let mut calls = 0;
        let mut forward = |x: &Tensor| {
            let expected = &case["calls"][calls];
            calls += 1;
            check(
                &V::tensor(x.shallow_clone()),
                &expected["input"],
                dev,
                "source base-forward input",
            );
            check(
                &V::Tuple(vec![V::Int(7)]),
                &expected["args"],
                dev,
                "captured original args",
            );
            check(
                &V::Dict(vec![("own".into(), V::Bool(true))]),
                &expected["kwargs"],
                dev,
                "captured original kwargs",
            );
            original(x, &w, &settings)
        };
        compare(
            b::forward(&adapter, &x, &settings, &mut forward, &mut diagnostics),
            &case["forward"],
            dev,
            &format!("{note} forward"),
        );
        assert_eq!(calls, case["calls"].as_array().unwrap().len(), "{note}");
        assert_eq!(json!(diagnostics.warnings), case["warnings"], "{note}");
        assert!(diagnostics.errors.is_empty());
    }
}
