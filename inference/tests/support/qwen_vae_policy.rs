use super::{binding_tests::check, bits, device, kind, kitchen};
use imagine_inference::{execution_value::Value as V, models::qwen_image21 as q, native::Error};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tch::{Device, Kind, Tensor};

fn stamp(x: &Tensor) -> Value {
    let dtype = match x.kind() {
        Kind::Float => "torch.float32",
        Kind::Double => "torch.float64",
        Kind::BFloat16 => "torch.bfloat16",
        _ => panic!("fixture dtype"),
    };
    json!({"type":"tensor", "device":if x.device()==Device::Cpu {"cpu"} else {"cuda:0"}, "dtype":dtype,
        "shape":x.size(), "strides":x.stride(), "requires_grad":x.requires_grad(),
        "payload_sha256":format!("{:x}",Sha256::digest(bits(x)))})
}
fn shape(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect()
}
fn data(shape: &[i64], device: Device, dtype: Kind, offset: i64) -> Tensor {
    (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, device)) + offset)
        .remainder(29)
        .g_mul_scalar(0.03125)
        .g_sub_scalar(0.40625)
        .reshape(shape)
        .to_kind(dtype)
}
fn resized(x: &Tensor, factor: f64) -> Result<Tensor, Error> {
    Ok(x.f_upsample_nearest2d(
        [
            (x.size()[2] as f64 * factor).floor() as i64,
            (x.size()[3] as f64 * factor).floor() as i64,
        ],
        Some(factor),
        Some(factor),
    )?)
}
#[test]
fn qwen_image21_original_tiled_scale_feather_order_default_dtype_and_batch_match() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let fixtures: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-vae-policy.json")).unwrap();
    for c in fixtures["tiles"].as_array().unwrap() {
        let x = data(
            &shape(&c["shape"]),
            device(c["device"].as_str().unwrap()),
            kind(c["dtype"].as_str().unwrap()),
            5,
        );
        let mut calls = Vec::new();
        let factor = c["factor"].as_f64().unwrap();
        let y = q::tiled_scale_image(
            &x,
            q::ImageScale {
                tiles: q::ImageTiles {
                    y: c["tile"][0].as_i64().unwrap(),
                    x: c["tile"][1].as_i64().unwrap(),
                    overlap: c["overlap"].as_i64().unwrap(),
                },
                factor,
                channels: 3,
                output_device: Device::Cpu,
                default_kind: kind(c["default_dtype"].as_str().unwrap()),
            },
            |x| {
                calls.push(stamp(x));
                Ok(resized(x, factor)?
                    .f_mul_scalar(0.125)?
                    .f_add_scalar((calls.len() % 7) as f64 * 0.015625)?)
            },
        )
        .unwrap();
        assert_eq!(json!(calls), c["calls"], "tile inputs {c}");
        check(
            &V::tensor(y),
            &c["output"],
            Device::Cpu,
            "original tiled scale",
        );
    }
}
struct Services<'a> {
    case: &'a Value,
    calls: usize,
    events: Vec<Value>,
}
impl q::ImageVaeServices for Services<'_> {
    fn load(&mut self, memory: f64) -> Result<(), Error> {
        self.events
            .push(json!({"event":"load","memory":memory as u64,"force":false}));
        if self.case["mode"] == "load-oom" {
            return Err(Error::ReferenceException {
                class: "OutOfMemoryError",
                message: "injected own bounded load failure".into(),
            });
        }
        Ok(())
    }
    fn free_memory(&mut self) -> Result<f64, Error> {
        self.events.push(json!({"event":"free"}));
        Ok(self.case["free_memory"].as_f64().unwrap())
    }
    fn empty_cache(&mut self) -> Result<(), Error> {
        self.events.push(json!({"event":"empty_cache"}));
        Ok(())
    }
    fn is_oom(&mut self, error: &Error) -> Result<bool, Error> {
        let Error::ReferenceException { class, .. } = error else {
            panic!("unexpected error: {error}")
        };
        self.events
            .push(json!({"event":"is_oom","class_name":class}));
        Ok(*class == "OutOfMemoryError")
    }
    fn call(&mut self, op: q::VaeOperation, x: &Tensor) -> Result<Tensor, Error> {
        self.calls += 1;
        self.events.push(json!({"event":"call","operation":if op==q::VaeOperation::Encode {"encode"} else {"decode"},"input":stamp(x)}));
        let mode = self.case["mode"].as_str().unwrap();
        if (self.calls == 1
            && ["call-oom", "runtime-oom-text", "interrupt", "non-oom"].contains(&mode))
            || (mode == "second-oom" && self.calls == 2)
        {
            let (class, message) = match mode {
                "call-oom" | "second-oom" => {
                    ("OutOfMemoryError", "injected own bounded call failure")
                }
                "runtime-oom-text" => ("RuntimeError", "out of memory"),
                "interrupt" => ("InterruptedError", "injected interrupt"),
                _ => ("ValueError", "injected invalid argument"),
            };
            return Err(Error::ReferenceException {
                class,
                message: message.into(),
            });
        }
        let y = if op == q::VaeOperation::Encode {
            x.f_narrow(1, 0, 1)?
                .f_avg_pool2d([16, 16], [16, 16], [0, 0], false, true, None)?
                .f_repeat([1, 64, 1, 1])?
        } else {
            resized(&x.f_narrow(1, 0, 4)?, 16.)?
        };
        Ok(y.f_mul_scalar(0.125)?
            .f_add_scalar((self.calls % 7) as f64 * 0.015625)?)
    }
}
#[test]
fn qwen_image21_original_image_vae_memory_batches_oom_retry_and_failure_classes_match() {
    let _native = kitchen();
    let _inference = imagine_inference::native::inference_mode(true).unwrap();
    let _grad = tch::no_grad_guard();
    let fixtures: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-vae-policy.json")).unwrap();
    for c in fixtures["policies"].as_array().unwrap() {
        eprintln!(
            "image VAE policy {}/{}/{}",
            c["device"], c["operation"], c["mode"]
        );
        let input = data(&shape(&c["shape"]), Device::Cpu, Kind::Float, 7);
        let mut services = Services {
            case: c,
            calls: 0,
            events: vec![],
        };
        let policy = q::ImageVaePolicy {
            device: device(c["device"].as_str().unwrap()),
            dtype: Kind::BFloat16,
            output_device: Device::Cpu,
            output_kind: Kind::Float,
            default_kind: Kind::Float,
            output_channels: 4,
            latent_channels: 64,
        };
        let op = if c["operation"] == "encode" {
            q::VaeOperation::Encode
        } else {
            q::VaeOperation::Decode
        };
        let result = if let Some(options) = c.get("tiled_options") {
            policy.execute_tiled(
                op,
                &input,
                q::ImageTileOptions {
                    x: options["tile_x"].as_i64(),
                    y: options["tile_y"].as_i64(),
                    overlap: options["overlap"].as_i64(),
                },
                &mut services,
            )
        } else {
            policy.execute(op, &input, &mut services)
        };
        assert_eq!(
            json!(services.events),
            c["events"],
            "original call and recovery sequence"
        );
        match result {
            Ok(y) => check(
                &V::tensor(y),
                &c["output"],
                Device::Cpu,
                "original image VAE policy output",
            ),
            Err(Error::ReferenceException { class, message }) => {
                assert_eq!(json!(class), c["error"]);
                assert_eq!(json!(message), c["message"]);
            }
            Err(e) => panic!("unexpected {e}"),
        }
    }
}

#[test]
fn qwen_image21_original_c10_oom_type_accelerator_code_and_plain_errors_remain_distinct() {
    use std::ffi::{CStr, CString, c_char};
    unsafe extern "C" {
        fn at_imagine_test_error(kind: i32, code: i32, message: *const c_char);
    }
    let native = kitchen();
    let fixtures: Value =
        serde_json::from_str(include_str!("../../fixtures/qwen-image21-oom.json")).unwrap();
    for c in fixtures["cases"].as_array().unwrap() {
        let kind = c["kind"].as_i64().unwrap() as i32;
        let code = c["code"].as_i64().unwrap() as i32;
        let message = c["message"].as_str().unwrap();
        let expected = c["oom"].as_bool().unwrap();
        let message = CString::new(message).unwrap();
        unsafe {
            at_imagine_test_error(kind, code, message.as_ptr());
        }
        let raw = unsafe { torch_sys::get_and_reset_last_err() };
        assert!(!raw.is_null());
        let text = unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned();
        unsafe {
            libc::free(raw.cast());
        }
        let error: Error = tch::TchError::Torch(text).into();
        match (kind, &error) {
            (
                0,
                Error::ReferenceException {
                    class: "OutOfMemoryError",
                    ..
                },
            ) => (),
            (1, Error::AcceleratorException { code: actual, .. }) => assert_eq!(code, *actual),
            (2, Error::Torch(_)) => (),
            _ => panic!("Lost original native error type: {error:?}"),
        }
        assert_eq!(
            q::image_vae_is_oom(&native, Device::Cuda(0), &error),
            expected
        );
        // An unrelated following tensor operation must remain usable.
        assert_eq!(Tensor::from_slice(&[3_i64]).int64_value(&[0]), 3);
    }
}
