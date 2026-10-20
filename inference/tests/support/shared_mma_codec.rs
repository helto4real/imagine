use super::{
    binding_tests::{check, reference_error, restore},
    device, kind, kitchen,
};
use imagine_inference::{execution_value::Value as V, packing::mma};
use serde_json::Value;
#[test]
fn original_mma_packed_records_preserve_codes_scales_strides_empty_shapes_and_errors() {
    let _native = kitchen();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-mma-codec.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let input = restore(&c["input"], dev).as_tensor().unwrap();
        let output = mma::unpack(
            &input,
            c["n"].as_i64().unwrap(),
            c["k"].as_i64().unwrap(),
            kind(c["kind"].as_str().unwrap()),
        );
        if c["error"].is_string() {
            reference_error(
                output
                    .err()
                    .unwrap_or_else(|| panic!("Expected source rejection {}", c["label"])),
                c,
            );
        } else {
            let (q, s) = output.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            check(
                &V::Tuple(vec![V::tensor(q), V::tensor(s)]),
                &c["output"],
                dev,
                c["label"].as_str().unwrap(),
            );
        }
    }
}

#[test]
fn original_packed_and_canonical_w4a8_linear_preserve_activation_residual_and_allocation_context() {
    use imagine_inference::{
        execution::Runtime,
        kitchen_sol::{self, AllocationContext},
        native::Error,
    };
    use std::{
        collections::BTreeMap,
        sync::{Arc, Mutex},
    };
    use tch::{Kind, Tensor};
    struct Context(Arc<Mutex<Vec<&'static str>>>);
    impl AllocationContext for Context {
        fn enter(&self) -> Result<(), Error> {
            self.0.lock().unwrap().push("enter");
            Ok(())
        }
        fn exit(&self, _failure: Option<&Error>) -> Result<bool, Error> {
            self.0.lock().unwrap().push("exit");
            Ok(false)
        }
    }
    struct Restore(Option<Arc<dyn AllocationContext>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            kitchen_sol::set_allocation_context(self.0.take());
        }
    }
    let native = kitchen();
    struct Threads(i32);
    impl Drop for Threads {
        fn drop(&mut self) {
            tch::set_num_threads(self.0);
        }
    }
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(1); // Same explicit setting as the bounded Python oracle.
    let _guard = tch::no_grad_guard();
    let _restore = Restore(kitchen_sol::allocation_context());
    let events = Arc::new(Mutex::new(Vec::new()));
    kitchen_sol::set_allocation_context(Some(Arc::new(Context(events.clone()))));
    let rt = Runtime::frozen_partial(native).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-packed-linear.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        events.lock().unwrap().clear();
        let dev = device(c["device"].as_str().unwrap());
        let dtype = kind(c["dtype"].as_str().unwrap());
        let (m, n, k, bits) = (
            c["m"].as_i64().unwrap(),
            c["n"].as_i64().unwrap(),
            c["k"].as_i64().unwrap(),
            c["bits"].as_i64().unwrap(),
        );
        let bytes = mma::record_bytes(bits);
        let count = n * k * bytes / 512;
        let q = Tensor::arange(count, (Kind::Int, dev))
            .f_mul_scalar(37)
            .unwrap()
            .f_add_scalar(11)
            .unwrap()
            .f_remainder(256)
            .unwrap()
            .to_kind(Kind::Int8);
        let _ = q
            .view_dtype(Kind::Uint8)
            .view([n / 16, k / 32, bytes])
            .narrow(2, bytes - 32, 32)
            .fill_(40);
        let act = c["act"].as_str();
        let width = if matches!(act, Some("swiglu" | "situ")) {
            2
        } else {
            1
        };
        let x = Tensor::arange(m * k * width, (Kind::Float, dev))
            .f_remainder(17)
            .unwrap()
            .f_mul_scalar(0.03125)
            .unwrap()
            .f_sub_scalar(0.25)
            .unwrap()
            .reshape([m, k * width])
            .to_kind(dtype);
        let channel = Tensor::full([n], 0.03125, (Kind::Float, dev));
        let gamma = Tensor::full([k], 0.7, (Kind::Float, dev));
        let bias = (Tensor::arange(n, (Kind::Float, dev)) * 0.015625).to_kind(dtype);
        let packed = c["packed"].as_bool().unwrap();
        let (q, s) = if packed {
            (q, Tensor::empty([0], (Kind::Float8e4m3fn, dev)))
        } else {
            mma::unpack(&q, n, k, Kind::Float8e4m3fn).unwrap()
        };
        let args = vec![V::tensor(x), V::tensor(q), V::tensor(s), V::tensor(channel)];
        let mut kw = vec![
            ("bias".into(), V::tensor(bias)),
            ("out_dtype".into(), V::DType(dtype)),
            (
                "input_act".into(),
                act.map_or(V::None, |a| V::Text(a.into())),
            ),
            ("input_act_weight".into(), V::tensor(gamma)),
            ("input_act_eps".into(), V::Float(1e-6)),
        ];
        if c["residual"].as_bool().unwrap() {
            kw.extend([
                (
                    "residual".into(),
                    V::tensor(Tensor::full([m, n], 0.25, (dtype, dev))),
                ),
                (
                    "residual_scale".into(),
                    V::tensor(Tensor::full([n], 0.5, (dtype, dev))),
                ),
            ]);
        }
        let operation = if packed {
            "w4a8_packed_linear"
        } else {
            "w4a8_int8_linear"
        };
        let f = rt
            .backends
            .resolve(operation, c["backend"].as_str(), &BTreeMap::new())
            .unwrap();
        for _ in 0..if c["error"].is_string() { 1 } else { 2 } {
            let output = unsafe { f.call(&rt, &args, &kw) };
            if c["error"].is_string() {
                reference_error(
                    output
                        .err()
                        .unwrap_or_else(|| panic!("Expected source error {c}")),
                    c,
                );
            } else {
                check(
                    &output.unwrap_or_else(|e| panic!("{c}: {e}")),
                    &c["output"],
                    dev,
                    &format!("{c}"),
                );
            }
        }
        assert_eq!(
            serde_json::json!(*events.lock().unwrap()),
            c["allocation_events"],
            "{c}"
        );
    }
}
