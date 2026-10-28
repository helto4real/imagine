//! Own timing of the Kitchen calls a Qwen-Image 2.1 DiT block makes, at its shapes,
//! against the bridge in IMAGINE_KITCHEN_LIBRARY. Synthetic tensors only.
use imagine_inference::{
    execution::Runtime, execution_value::Value as V, native::Kitchen, triton_compiler::Compiler,
};
use std::{path::Path, sync::Arc, time::Instant};
use tch::{Device, Kind, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scratch = std::env::args().nth(1).ok_or("scratch directory")?;
    let scratch = Path::new(&scratch);
    let native = unsafe { Kitchen::load(Path::new(&std::env::var("IMAGINE_KITCHEN_LIBRARY")?))? };
    let compiler = Arc::new(unsafe {
        Compiler::open(
            Path::new(&std::env::var("IMAGINE_TRITON_COMPILER_PROFILE")?),
            &scratch.join("cache"),
            &scratch.join("work"),
            native.clone(),
        )?
    });
    let rt = Runtime::frozen(native, compiler)?;
    let _grad = tch::no_grad_guard();
    let dev = Device::Cuda(0);
    let time = |label: &str,
                f: &mut dyn FnMut() -> Result<(), Box<dyn std::error::Error>>|
     -> Result<(), Box<dyn std::error::Error>> {
        for _ in 0..3 {
            f()?;
        }
        rt.native.synchronize_device(0)?;
        let start = Instant::now();
        let n = 20;
        for _ in 0..n {
            f()?;
        }
        rt.native.synchronize_device(0)?;
        println!(
            "{label:44} {:9.3} ms",
            start.elapsed().as_secs_f64() * 1000. / n as f64
        );
        Ok(())
    };
    for (m, k, n) in [
        (5600, 3072, 3072),
        (5600, 3072, 12288),
        (5600, 12288, 3072),
        (128, 3072, 3072),
        (128, 3072, 12288),
        (128, 12288, 3072),
        (1, 3072, 18432),
    ] {
        let x = Tensor::randn([1, m, k], (Kind::BFloat16, dev));
        let w = Tensor::randn([n, k], (Kind::BFloat16, dev)).f_mul_scalar(0.02)?;
        let q = unsafe {
            rt.from_float(
                &w,
                "TensorWiseINT8Layout",
                &vec![
                    ("is_weight".into(), V::Bool(true)),
                    ("per_channel".into(), V::Bool(true)),
                    ("convrot".into(), V::Bool(true)),
                    ("convrot_groupsize".into(), V::Int(256)),
                    ("stochastic_rounding".into(), V::Int(0)),
                ],
            )?
        };
        let payload = q.payload.shallow_clone();
        let scale = q.params.tensors["scale"].shallow_clone();
        let bias = Tensor::randn([n], (Kind::BFloat16, dev));
        time(&format!("int8_linear m={m} k={k} n={n}"), &mut || {
            unsafe {
                rt.call_kitchen(
                    "int8_linear",
                    &vec![
                        ("x".into(), V::tensor(x.shallow_clone())),
                        ("weight".into(), V::tensor(payload.shallow_clone())),
                        ("weight_scale".into(), V::tensor(scale.shallow_clone())),
                        ("bias".into(), V::tensor(bias.shallow_clone())),
                        ("out_dtype".into(), V::DType(Kind::BFloat16)),
                        ("convrot".into(), V::Bool(true)),
                        ("convrot_groupsize".into(), V::Int(256)),
                    ],
                )?
            };
            Ok(())
        })?;
    }
    let seq = 5728;
    let q = Tensor::randn([1, seq, 24, 128], (Kind::BFloat16, dev));
    let k = Tensor::randn([1, seq, 24, 128], (Kind::BFloat16, dev));
    let freqs = Tensor::randn([1, seq, 1, 64, 2, 2], (Kind::Float, dev));
    let w = Tensor::ones([128], (Kind::BFloat16, dev));
    time("rms_rope q,k [1,5728,24,128]", &mut || {
        unsafe {
            rt.call_kitchen(
                "rms_rope",
                &vec![
                    ("q".into(), V::tensor(q.shallow_clone())),
                    ("k".into(), V::tensor(k.shallow_clone())),
                    ("freqs_cis".into(), V::tensor(freqs.shallow_clone())),
                    ("q_scale".into(), V::tensor(w.shallow_clone())),
                    ("k_scale".into(), V::tensor(w.shallow_clone())),
                    ("epsilon".into(), V::Float(1e-6)),
                ],
            )?
        };
        Ok(())
    })?;
    time("apply_rope1 x [1,5728,24,128]", &mut || {
        unsafe {
            rt.call_kitchen(
                "apply_rope1",
                &vec![
                    ("x".into(), V::tensor(q.shallow_clone())),
                    ("freqs_cis".into(), V::tensor(freqs.shallow_clone())),
                ],
            )?
        };
        Ok(())
    })?;
    let x = Tensor::randn([1, 5600, 3072], (Kind::BFloat16, dev));
    let s = Tensor::randn([1, 1, 3072], (Kind::BFloat16, dev));
    time("adaln x [1,5600,3072]", &mut || {
        unsafe {
            rt.call_kitchen(
                "adaln",
                &vec![
                    ("x".into(), V::tensor(x.shallow_clone())),
                    ("scale".into(), V::tensor(s.shallow_clone())),
                    ("shift".into(), V::tensor(s.shallow_clone())),
                    ("eps".into(), V::Float(1e-6)),
                ],
            )?
        };
        Ok(())
    })?;
    Ok(())
}
