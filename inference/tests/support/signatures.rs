use super::{fixtures, kitchen};
use imagine_inference::{execution::Runtime, execution_value::Value, triton_compiler::Compiler};
use std::{collections::BTreeMap, path::Path, sync::Arc};
#[test]
fn all_original_registered_callables_preserve_1004_argument_binding_failures_before_body_execution()
{
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    compiler.set_pre_run_guard(Some(Arc::new(|_, _, _| {
        panic!("Invalid argument binding must precede JIT execution")
    })));
    let rt = Runtime::frozen(native, compiler).unwrap();
    assert!(rt.backends.gaps().is_empty());
    assert_eq!(
        fixtures()["backend_signature_errors"]
            .as_array()
            .unwrap()
            .len(),
        1004
    );
    for c in fixtures()["backend_signature_errors"].as_array().unwrap() {
        let op = c["operation"].as_str().unwrap();
        let backend = c["backend"].as_str().unwrap();
        let args = vec![Value::None; c["args_count"].as_u64().unwrap() as usize];
        let kwargs = c["kwargs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| (v.as_str().unwrap().to_string(), Value::None))
            .collect::<Vec<_>>();
        let f = rt
            .backends
            .resolve(op, Some(backend), &BTreeMap::new())
            .unwrap();
        let error = unsafe { f.call(&rt, &args, &kwargs) }
            .err()
            .unwrap_or_else(|| panic!("Original binding failure: {c}"));
        super::binding_tests::reference_error(error, c);
    }
}
