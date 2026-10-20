use imagine_inference::{execution_value::Value as V, native::Error, patching as p};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tch::{Device, Kind, Tensor};
#[test]
fn missing_native_patch_transform_binding_cannot_be_logged_as_a_successful_partially_patched_weight()
 {
    let library = super::kitchen();
    let _grad = tch::no_grad_guard();
    let options = (Kind::Float, Device::Cpu);
    let weight = Tensor::ones([4, 4], options);
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let mut patch = p::Patch::new(
        p::PatchValue::Adapter(p::Adapter {
            name: "lora".into(),
            weights: vec![
                V::tensor(Tensor::ones([4, 2], options)),
                V::tensor(Tensor::ones([2, 4], options)),
                V::None,
                V::None,
                V::None,
                V::None,
            ],
        }),
        1.,
    );
    patch.model_strength = 0.5;
    patch.function = Some(Arc::new(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
        Err(Error::BindingGap {
            backend: "required source callback".into(),
            operation: "not implemented".into(),
        })
    }));
    let mut diagnostics = p::Diagnostics::default();
    let error = p::calculate_weight(
        &library,
        &[patch],
        weight.shallow_clone(),
        "unit.weight",
        Kind::Float,
        None,
        &mut diagnostics,
    )
    .expect_err("a missing binding cannot be accepted as a successful patch");
    assert!(matches!(error.root(), Error::BindingGap { .. }));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(diagnostics.errors.is_empty());
    assert_eq!(
        super::bits(&weight),
        super::bits(&Tensor::full([4, 4], 0.5, options)),
        "preserve completed model-strength mutation, do not roll it back"
    );
}
