use super::{fixtures, kitchen};
use imagine_inference::{
    native::Error,
    triton_compiler::Compiler,
    triton_driver::Scalar,
    triton_frontend::{Count, Element, Fp8Body, Fp8Specialization},
    triton_hooks::{Hooks, LaunchHook, LoadHook},
};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use tch::{Device, Kind, Tensor};
#[test]
fn original_kernel_hooks_preserve_order_metadata_repeated_failures_and_pointer_validation() {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let stream = native.current_stream(0).unwrap();
    for c in fixtures()["triton_kernel_hooks"].as_array().unwrap() {
        let compiler = unsafe {
            Compiler::open(
                Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
                &own.join("cache"),
                &own.join("work"),
                native.clone(),
            )
            .unwrap()
        };
        let s = Fp8Specialization {
            body: Fp8Body::Quantize,
            input: Element::F32,
            output: Element::E4M3,
            scale: Element::F32,
            block_size: 128,
            pointer_divisibility_16: [true; 3],
            count: Count::I32 {
                divisible_by_16: true,
            },
            source_path: compiler.source_path(),
        };
        let kernel = unsafe { compiler.jit_fp8(0, &s).unwrap() };
        let label = c["label"].as_str().unwrap().to_string();
        let events = Arc::new(Mutex::new(Vec::new()));
        let load = |which: &'static str| -> LoadHook {
            let label = label.clone();
            let events = events.clone();
            Arc::new(move |m| {
                events.lock().unwrap().push(serde_json::json!({"hook":which,"module_present":m.module.is_some(),"function_present":m.function.is_some(),"name":m.name,"group":m.metadata_group.iter().map(|(k,_)|k.to_utf8().unwrap()).collect::<Vec<_>>(),"digest":m.digest}));
                if label == format!("{which}_failure") {
                    Err(Error::ReferenceException {
                        class: "ValueError",
                        message: format!("own {which} failure"),
                    })
                } else {
                    Ok(())
                }
            })
        };
        let launch = |which: &'static str| -> LaunchHook {
            let label = label.clone();
            let events = events.clone();
            Arc::new(move |m| {
                events.lock().unwrap().push(serde_json::json!({"hook":which,"metadata":m.map(|m|serde_json::json!({"name":m.name,"function_present":m.function!=0,"stream_is_original":m.stream==stream}))}));
                if label == format!("{which}_failure") {
                    Err(Error::ReferenceException {
                        class: "ValueError",
                        message: format!("own {which} failure"),
                    })
                } else {
                    Ok(())
                }
            })
        };
        let enabled = label != "none" && label != "exit_only";
        compiler.hooks.replace(Hooks {
            load_start: enabled.then(|| load("start")),
            load_end: enabled.then(|| load("end")),
            launch_enter: enabled.then(|| launch("enter")),
            launch_exit: (label != "none").then(|| launch("exit")),
        });
        let x = Tensor::arange(
            16,
            (
                Kind::Float,
                if label == "cpu_pointer" {
                    Device::Cpu
                } else {
                    Device::Cuda(0)
                },
            ),
        );
        let out = Tensor::empty([16], (Kind::Float8e4m3fn, Device::Cuda(0)));
        let scale = Tensor::from_slice(&[0.7f32]).to_device(Device::Cuda(0));
        for step in c["steps"].as_array().unwrap() {
            events.lock().unwrap().clear();
            let mut args = [&x, &out, &scale]
                .map(|t| Scalar::TensorPointer(t.data_ptr() as u64))
                .into_iter()
                .collect::<Vec<_>>();
            args.extend([
                Scalar::F32(448.),
                Scalar::I32(16),
                Scalar::Pointer(0),
                Scalar::Pointer(0),
            ]);
            let result =
                unsafe { kernel.run([i32::from(label != "zero_grid"), 1, 1], stream, &mut args) };
            if step.get("error").is_some() {
                super::binding_tests::reference_error(result.err().unwrap(), step);
            } else {
                result.unwrap();
            }
            assert_eq!(
                serde_json::json!(*events.lock().unwrap()),
                step["events"],
                "{label}"
            );
            assert_eq!(
                kernel.driver_handle_presence(),
                (
                    step["module_present"].as_bool().unwrap(),
                    step["function_present"].as_bool().unwrap()
                ),
                "{label}"
            );
            assert_eq!(
                serde_json::json!(kernel.resources()),
                step["resources"],
                "{label}"
            );
        }
    }
}
