use super::binding_tests::{check, kwargs};
use super::*;
use imagine_inference::{execution::Runtime, execution_value::Value as V, int8_utils};
struct Restore {
    int8: Tensor,
    w4: Tensor,
    original_int8: Tensor,
    original_w4: Tensor,
}
impl Drop for Restore {
    fn drop(&mut self) {
        let _ = self.int8.f_copy_(&self.original_int8);
        let _ = self.w4.f_copy_(&self.original_w4);
    }
}
#[test]
fn w4a4_and_int8_hadamard_caches_preserve_original_separation_mutability_and_routing() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let data = fixtures();
    let cases = data["w4a4_cache_bindings"].as_array().unwrap();
    assert_eq!(cases.len(), 6);
    for c in cases {
        let dev = device(c["device"].as_str().unwrap());
        let dt = kind(c["dtype"].as_str().unwrap());
        {
            let mut guard = Restore {
                int8: int8_utils::hadamard(16, dev, dt).unwrap(),
                w4: int8_utils::eager_w4_hadamard(16, dev, dt).unwrap(),
                original_int8: Tensor::new(),
                original_w4: Tensor::new(),
            };
            guard.original_int8 = guard.int8.copy();
            guard.original_w4 = guard.w4.copy();
            assert_ne!(guard.int8.data_ptr(), guard.w4.data_ptr());
            check(
                &V::tensor(guard.int8.shallow_clone()),
                &c["int8_hadamard"],
                dev,
                "original INT8 cache",
            );
            check(
                &V::tensor(guard.w4.shallow_clone()),
                &c["w4_hadamard"],
                dev,
                "original W4 cache",
            );
            for (i, phase) in c["phases"].as_array().unwrap().iter().enumerate() {
                if i == 1 {
                    let _ = guard
                        .int8
                        .select(0, 0)
                        .select(0, 0)
                        .f_add_scalar_(0.5)
                        .unwrap();
                }
                if i == 2 {
                    let _ = guard
                        .w4
                        .select(0, 0)
                        .select(0, 0)
                        .f_add_scalar_(0.75)
                        .unwrap();
                }
                let hi = int8_utils::hadamard(16, dev, dt).unwrap();
                let hw = int8_utils::eager_w4_hadamard(16, dev, dt).unwrap();
                assert_eq!(hi.data_ptr(), guard.int8.data_ptr());
                assert_eq!(hw.data_ptr(), guard.w4.data_ptr());
                assert_ne!(hi.data_ptr(), hw.data_ptr());
                check(
                    &V::tensor(hi),
                    &phase["int8_hadamard"],
                    dev,
                    "INT8 mutable cache",
                );
                check(
                    &V::tensor(hw),
                    &phase["w4_hadamard"],
                    dev,
                    "W4 mutable cache",
                );
                for call in phase["calls"].as_array().unwrap() {
                    let k = kwargs(&call["kwargs"], dev);
                    let meta = k
                        .iter()
                        .map(|(n, v)| (n.clone(), v.metadata(&rt.native).unwrap()))
                        .collect();
                    let f = rt
                        .backends
                        .resolve(call["operation"].as_str().unwrap(), Some("eager"), &meta)
                        .unwrap();
                    let out = unsafe { f.call(&rt, &[], &k) }.unwrap();
                    check(
                        &out,
                        &call["output"],
                        dev,
                        "original mutable-cache operator output",
                    );
                    for ((key, v), expected) in
                        k.iter().zip(call["post_kwargs"].as_array().unwrap())
                    {
                        assert_eq!(key, expected[0].as_str().unwrap());
                        check(v, &expected[1], dev, "unchanged cache test operand");
                    }
                }
            }
        }
        check(
            &V::tensor(int8_utils::hadamard(16, dev, dt).unwrap()),
            &c["int8_hadamard"],
            dev,
            "restored INT8 cache",
        );
        check(
            &V::tensor(int8_utils::eager_w4_hadamard(16, dev, dt).unwrap()),
            &c["w4_hadamard"],
            dev,
            "restored W4 cache",
        );
    }
}
