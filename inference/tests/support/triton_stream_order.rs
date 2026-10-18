use super::{fixtures, kitchen};
use imagine_inference::{
    execution::Runtime,
    execution_value::Value,
    native::{Error, StreamGuard},
    triton_compiler::Compiler,
    triton_hooks::Hooks,
};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::Path,
    sync::{Arc, Mutex},
};
use tch::{Device, Kind, Tensor};
thread_local! {static OWN_STREAM:RefCell<Option<StreamGuard>>=const{RefCell::new(None)};}
#[test]
fn original_jit_captures_device_and_stream_before_pre_run_hooks() {
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
    for c in fixtures()["triton_stream_order"].as_array().unwrap() {
        let original = native.current_stream(0).unwrap();
        let events = Arc::new(Mutex::new(Vec::new()));
        let seen = events.clone();
        let n = native.clone();
        compiler.set_pre_run_guard(Some(Arc::new(move|_,_,_|{seen.lock().unwrap().push(serde_json::json!({"hook":"pre","current_is_original":n.current_stream(0)?==original}));OWN_STREAM.with(|slot|{*slot.borrow_mut()=Some(n.stream_guard(0).unwrap());});Ok(())})));
        let seen = events.clone();
        let n = native.clone();
        compiler.hooks.replace(Hooks{launch_enter:Some(Arc::new(move|meta|{let meta=meta.unwrap();seen.lock().unwrap().push(serde_json::json!({"hook":"enter","captured_is_original":meta.stream==original,"current_is_other":n.current_stream(0)?!=original}));Err(Error::ReferenceException{class:"ValueError",message:"own stream boundary".into()})})),..Hooks::default()});
        let x = Value::tensor(Tensor::ones([1, 1, 1, 16], (Kind::Float, Device::Cuda(0))));
        let freq = Value::tensor(
            Tensor::eye(2, (Kind::Float, Device::Cuda(0)))
                .expand([1, 1, 1, 8, 2, 2], false)
                .contiguous(),
        );
        let scale = Value::tensor(Tensor::ones([16], (Kind::Float, Device::Cuda(0))));
        let args = match c["family"].as_str().unwrap() {
            "mxfp8" => vec![Value::tensor(Tensor::ones(
                [1, 32],
                (Kind::Float, Device::Cuda(0)),
            ))],
            "rms_rope" => vec![x, freq, scale],
            _ => vec![x, freq],
        };
        let f = rt
            .backends
            .resolve(
                c["operation"].as_str().unwrap(),
                Some("triton"),
                &BTreeMap::new(),
            )
            .unwrap();
        let result = unsafe { f.call(&rt, &args, &vec![]) };
        OWN_STREAM.with(|slot| drop(slot.borrow_mut().take()));
        compiler.set_pre_run_guard(None);
        compiler.hooks.replace(Hooks::default());
        super::binding_tests::reference_error(
            result.err().expect("Original controlled boundary"),
            c,
        );
        assert_eq!(serde_json::json!(*events.lock().unwrap()), c["events"]);
        assert_eq!(native.current_stream(0).unwrap(), original);
    }
}
