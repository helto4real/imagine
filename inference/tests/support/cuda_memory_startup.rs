use super::kitchen;
use imagine_inference::{checkpoint::Aimdo, execution::Runtime, triton_compiler::Compiler};
use std::{path::Path, sync::Arc};
#[test]
fn cuda_memory_query_initializes_allocator_before_any_tensor_allocation() {
    let _directory = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = Arc::new(
        unsafe {
            Compiler::open(
                Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
                &own.join("cache"),
                &own.join("work"),
                native.clone(),
            )
        }
        .unwrap(),
    );
    let _runtime = Runtime::frozen(native.clone(), compiler).unwrap();
    let _aimdo =
        unsafe { Aimdo::load(Path::new(&std::env::var("IMAGINE_AIMDO_LIBRARY").unwrap())) }
            .unwrap();
    let [free, total, active, reserved] = native.cuda_memory(0).unwrap();
    assert!(total > 0 && free <= total && active <= reserved);
    let tensor = tch::Tensor::ones([4], (tch::Kind::Float, tch::Device::Cuda(0)));
    assert_eq!(tensor.sum(tch::Kind::Float).double_value(&[]), 4.);
    let [_, next_total, next_active, next_reserved] = native.cuda_memory(0).unwrap();
    assert_eq!(total, next_total);
    assert!(next_active > 0 && next_active <= next_reserved);
}
