use super::{fixtures, kitchen};
use imagine_inference::{
    boxed_aten,
    execution::Runtime,
    native::Error,
    triton_compiler::Compiler,
    triton_hooks::{Hooks, LaunchHook, LoadHook},
};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
};
use tch::Device;
#[test]
fn original_hook_storage_changes_preserve_late_pointer_extraction_views_and_stateful_tensor_ownership()
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
    let rt = Runtime::frozen(native.clone(), compiler.clone()).unwrap();
    for c in fixtures()["triton_object_arguments"].as_array().unwrap() {
        compiler.clear_specializations();
        let x = super::binding_tests::restore(&c["input"], Device::Cuda(0));
        let replacement = super::binding_tests::restore(&c["replacement"], Device::Cuda(0));
        let scale = super::binding_tests::restore(&c["scale"], Device::Cuda(0));
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        let input = x.clone();
        let n = native.clone();
        let stage = c["stage"].as_str().unwrap().to_string();
        let mutate: Arc<dyn Fn() -> Result<(), Error> + Send + Sync> = Arc::new(move || {
            let before = input.as_tensor()?.data_ptr();
            unsafe {
                boxed_aten::call(
                    &n,
                    "aten::set_",
                    "source_Tensor",
                    &[input.clone(), replacement.clone()],
                    &vec![],
                )?;
            }
            seen.lock().unwrap().push(serde_json::json!({"hook":stage,"pointer_changed":input.as_tensor()?.data_ptr()!=before}));
            Ok(())
        });
        let load = || -> LoadHook {
            let f = mutate.clone();
            Arc::new(move |_| f())
        };
        let launch = || -> LaunchHook {
            let f = mutate.clone();
            Arc::new(move |_| f())
        };
        compiler.hooks.replace(Hooks {
            load_start: (c["stage"] == "load_start").then(load),
            load_end: (c["stage"] == "load_end").then(load),
            launch_enter: (c["stage"] == "enter").then(launch),
            launch_exit: (c["stage"] == "exit").then(launch),
        });
        let function = rt
            .backends
            .resolve("quantize_per_tensor_fp8", Some("triton"), &BTreeMap::new())
            .unwrap();
        for step in c["steps"].as_array().unwrap() {
            events.lock().unwrap().clear();
            let output =
                unsafe { function.call(&rt, &[x.clone(), scale.clone()], &vec![]) }.unwrap();
            super::binding_tests::check(
                &output,
                &step["output"],
                Device::Cuda(0),
                c["label"].as_str().unwrap(),
            );
            super::binding_tests::check(
                &x,
                &step["input_after"],
                Device::Cuda(0),
                "Original stateful input",
            );
            assert_eq!(
                serde_json::json!(*events.lock().unwrap()),
                step["events"],
                "{}",
                c["label"]
            );
        }
        compiler.hooks.replace(Hooks::default());
    }
}
