use imagine_inference::native::Error;
use serde_json::Value;
use tch::{Device, Kind, Tensor};
#[test]
fn original_default_c10_runtime_errors_transport_exact_display_text_without_test_side_backtrace_normalization()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-runtime-errors.json")).unwrap();
    assert_eq!(fixture["cases"].as_array().unwrap().len(), 4);
    for case in fixture["cases"].as_array().unwrap() {
        let options = (Kind::Float, Device::Cpu);
        let error: Error = match case["name"].as_str().unwrap() {
            "view" => Tensor::ones([2, 3], options)
                .f_view([5])
                .expect_err("original view failure")
                .into(),
            "mm" => Tensor::ones([2, 3], options)
                .f_mm(&Tensor::ones([4, 2], options))
                .expect_err("original mm failure")
                .into(),
            "min" => Tensor::zeros([0], options)
                .f_min()
                .expect_err("original reduction failure")
                .into(),
            "truth" => Tensor::ones([2], options)
                .f_is_nonzero()
                .expect_err("original truth failure")
                .into(),
            _ => panic!("unknown original operation"),
        };
        assert!(
            error.is_reference("RuntimeError"),
            "typed source RuntimeError, not a generic TchError"
        );
        assert_eq!(
            error.to_string(),
            case["message"].as_str().unwrap(),
            "{case}"
        );
    }
}
