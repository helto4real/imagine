use super::{binding_tests::check, device, kind, kitchen};
use imagine_inference::{
    checkpoint::PreparedParameters, execution::Runtime, execution_value::Value as V,
    models::qwen_image21 as q, native::Error, triton_compiler::Compiler,
};
use serde_json::Value;
use std::{path::Path, sync::Arc};
use tch::{Device, Kind, Tensor};
struct Exec(usize);
impl q::VaeExecution for Exec {
    fn check_interrupt(&mut self) -> Result<(), Error> {
        self.0 += 1;
        Ok(())
    }
    fn attention(
        &mut self,
        rt: &Runtime,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Tensor, Error> {
        q::vae_pytorch_attention(rt, q, k, v)
    }
}
struct Threads(i32);
impl Drop for Threads {
    fn drop(&mut self) {
        tch::set_num_threads(self.0);
    }
}
fn data(shape: &[i64], dev: Device, dtype: Kind, offset: i64) -> Tensor {
    Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, dev))
        .f_add_scalar(offset)
        .unwrap()
        .f_remainder(29)
        .unwrap()
        .f_mul_scalar(0.03125)
        .unwrap()
        .f_sub_scalar(0.40625)
        .unwrap()
        .reshape(shape)
        .to_kind(dtype)
}
#[test]
fn qwen_image21_original_wan22_image_encode_decode_rgba_and_strips_match() {
    let _dir = super::triton_file_cache_tests::OwnDirectory::create();
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
    let rt = Runtime::frozen(native, compiler).unwrap();
    let _grad = tch::no_grad_guard();
    let _threads = Threads(tch::get_num_threads());
    tch::set_num_threads(2);
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-vae.json")).unwrap();
    let mut count = 0;
    for g in fixture["groups"].as_array().unwrap() {
        let dev = device(g["device"].as_str().unwrap());
        let dtype = kind(g["dtype"].as_str().unwrap());
        eprintln!("Original image VAE: {}/{}", g["device"], g["dtype"]);
        let mut state = PreparedParameters::default();
        for (name, f) in g["factories"].as_object().unwrap() {
            let shape = f["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_i64().unwrap())
                .collect::<Vec<_>>();
            let mut t = data(
                &shape,
                Device::Cpu,
                Kind::Float,
                f["offset"].as_i64().unwrap(),
            )
            .f_mul_scalar(0.0625)
            .unwrap();
            if f["gamma"].as_bool().unwrap() {
                t = t.f_add_scalar(1).unwrap();
            }
            state
                .values
                .insert(name.clone(), V::tensor(t.to_device(dev).to_kind(dtype)));
        }
        let model = q::ImageVae::new(
            q::ImageVaeConfig {
                dim: 4,
                dec_dim: 4,
                ..Default::default()
            },
            state,
        );
        for c in g["calls"].as_array().unwrap() {
            let shape = c["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect::<Vec<_>>();
            let x = data(&shape, dev, dtype, 7);
            let mut e = Exec(0);
            let out = match c["operation"].as_str().unwrap() {
                "encode" => model.encode(&rt, &x, &mut e),
                "decode" => model.decode(&rt, &x, &mut e),
                _ => unreachable!(),
            }
            .unwrap();
            check(
                &V::tensor(out.shallow_clone()),
                &c["output"],
                dev,
                &format!("{}/{}/{}", g["device"], g["dtype"], c["operation"]),
            );
            assert_eq!(e.0, c["operations"].as_u64().unwrap() as usize);
            if c.get("processed").is_some() {
                let mut pixels = out
                    .f_to_device_(Device::Cpu, Kind::Float, false, true)
                    .unwrap();
                q::vae_process_output(&mut pixels).unwrap();
                check(
                    &V::tensor(pixels.f_movedim_int(1, -1).unwrap()),
                    &c["processed"],
                    Device::Cpu,
                    "RGBA output processing",
                );
            }
            count += 1;
        }
    }
    assert_eq!(count, 18);
    for c in fixture["strips"].as_array().unwrap() {
        let shape = c["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect::<Vec<_>>();
        let scale = c["scale"].as_i64().unwrap();
        let x = data(&shape, Device::Cpu, Kind::Float, 3);
        let out = if c["add"].as_bool().unwrap() {
            let mut shape = shape.clone();
            let n = shape.len();
            shape[n - 2] *= scale;
            shape[n - 1] *= scale;
            Some(Tensor::full(&shape[..], 0.5, (Kind::Float, Device::Cpu)))
        } else {
            None
        };
        let ptr = out.as_ref().map(Tensor::data_ptr);
        let mut calls = vec![];
        let y = q::vae_strip_apply(&x, scale, c["halo"].as_i64().unwrap(), out, |s| {
            calls.push(s.size());
            let y = s.f_mul_scalar(0.125)?.f_add_scalar(0.25)?;
            Ok(if scale == 2 {
                let d = y.size();
                y.f_internal_upsample_nearest_exact2d([d[2] * 2, d[3] * 2], Some(2.), Some(2.))?
            } else {
                y
            })
        })
        .unwrap();
        assert_eq!(serde_json::json!(calls), c["calls"]);
        assert_eq!(Some(y.data_ptr()) == ptr, c["alias"].as_bool().unwrap());
        check(
            &V::tensor(y),
            &c["output"],
            Device::Cpu,
            "original VAE strips",
        );
    }
}
