use super::{fixtures, kind, kitchen};
use imagine_inference::{execution::Runtime, native::Kitchen, triton_compiler::Compiler};
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tch::{Device, Kind};
pub(super) struct DefaultDtype(Kitchen, Kind);
impl DefaultDtype {
    pub(super) fn preserve(native: Kitchen) -> Self {
        let kind = native.default_dtype().unwrap();
        Self(native, kind)
    }
}
impl Drop for DefaultDtype {
    fn drop(&mut self) {
        self.0.set_default_dtype(self.1).unwrap();
    }
}
#[test]
fn none_output_dtype_preserves_all_original_floating_defaults_including_double() {
    let _root = super::triton_file_cache_tests::OwnDirectory::create();
    let own = std::env::current_dir().unwrap();
    let native = kitchen();
    let _restore = DefaultDtype::preserve(native.clone());
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE").unwrap()),
            &own.join("cache"),
            &own.join("work"),
            native.clone(),
        )
        .unwrap()
    });
    let rt = Runtime::frozen(native.clone(), compiler).unwrap();
    for c in fixtures()["triton_default_dtype"].as_array().unwrap() {
        native
            .set_default_dtype(kind(c["default_dtype"].as_str().unwrap()))
            .unwrap();
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| super::binding_tests::restore(v, Device::Cuda(0)))
            .collect::<Vec<_>>();
        let f = rt
            .backends
            .resolve("int8_linear", Some("triton"), &BTreeMap::new())
            .unwrap();
        for _ in 0..2 {
            let result = unsafe { f.call(&rt, &args, &vec![]) };
            if c.get("error").is_some() {
                super::binding_tests::reference_error(result.err().unwrap(), c);
            } else {
                super::binding_tests::check(
                    &result.unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                    &c["output"],
                    Device::Cuda(0),
                    c["label"].as_str().unwrap(),
                );
            }
        }
    }
}
