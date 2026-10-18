use super::{
    binding_tests::{check, restore},
    kitchen,
};
use imagine_inference::{execution_value::Value as V, models::moss_image::moss_gated_delta_rule};
use serde_json::Value;
use tch::Device;
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
#[test]
fn moss_original_qwen35_chunked_delta_prefill_padding_dtype_and_recurrent_state_match() {
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/moss-delta.json")).unwrap();
    let mut count = 0;
    for case in fixture["cases"].as_array().unwrap() {
        let args = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, Device::Cpu).as_tensor().unwrap())
            .collect::<Vec<_>>();
        let dev = args[0].device();
        let initial = restore(&case["initial"], dev);
        let initial = if matches!(initial, V::None) {
            None
        } else {
            Some(initial.as_tensor().unwrap())
        };
        let (output, state) = moss_gated_delta_rule(
            &native,
            &args[0],
            &args[1],
            &args[2],
            &args[3],
            &args[4],
            64,
            initial.as_ref(),
            case["final"].as_bool().unwrap(),
        )
        .unwrap();
        check(
            &V::Tuple(vec![V::tensor(output), state.map_or(V::None, V::tensor)]),
            &case["output"],
            dev,
            case["label"].as_str().unwrap(),
        );
        count += 1;
    }
    assert_eq!(count, 54);
}
