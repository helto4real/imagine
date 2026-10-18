use super::{binding_tests::check, device, kind, kitchen, qwen_text_encoder_tests::data};
use imagine_inference::{execution_value::Value as V, models::qwen_image21 as q, native::Error};
use serde_json::Value;
use std::path::Path;
use tch::{Device, Kind, Tensor};
fn numbers(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap())
        .collect()
}
#[test]
fn qwen_image21_original_pillow_lanczos_alpha_reference_geometry_and_vae_inputs_match() {
    let _native = kitchen();
    let _grad = tch::no_grad_guard();
    let pillow = unsafe {
        q::Pillow::open(Path::new(&std::env::var("IMAGINE_PILLOW_LIBRARY").unwrap())).unwrap()
    };
    let f: Value = serde_json::from_str(include_str!(
        "../../fixtures/qwen-image21-image-preprocess.json"
    ))
    .unwrap();
    assert_eq!(f["resize"].as_array().unwrap().len(), 24);
    assert_eq!(f["nodes"].as_array().unwrap().len(), 28);
    assert_eq!(f["vae_pixels"].as_array().unwrap().len(), 32);
    assert_eq!(f["memory"].as_array().unwrap().len(), 12);
    for c in f["vae_pixels"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let x = data(
            &numbers(&c["shape"]),
            dev,
            kind(c["dtype"].as_str().unwrap()),
            9,
        )
        .g_add_scalar(0.5);
        check(
            &V::tensor(q::vae_crop_pixels(&x).unwrap()),
            &c["cropped"],
            dev,
            "VAE crop/pad",
        );
        check(
            &V::tensor(q::vae_encode_pixels(&x, Device::Cuda(0)).unwrap()),
            &c["processed"],
            dev,
            "VAE transform/cast order",
        );
    }
    for c in f["memory"].as_array().unwrap() {
        let conditions = c["conditions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|refs| {
                refs.as_array()
                    .unwrap()
                    .iter()
                    .map(numbers)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let (required, minimum) =
            q::qwen_sampling_memory_with_references(&numbers(&c["shape"]), &conditions);
        assert_eq!(required.to_bits(), c["required_bits"].as_u64().unwrap());
        assert_eq!(minimum.to_bits(), c["minimum_bits"].as_u64().unwrap());
    }
    for c in f["resize"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let x = data(
            &numbers(&c["shape"]),
            dev,
            kind(c["dtype"].as_str().unwrap()),
            7,
        )
        .g_add_scalar(0.5);
        let out = pillow
            .lanczos(
                &x,
                c["size"][0].as_i64().unwrap(),
                c["size"][1].as_i64().unwrap(),
            )
            .unwrap();
        check(
            &V::tensor(out),
            &c["output"],
            dev,
            &format!("Lanczos {}", c),
        );
    }
    for c in f["nodes"].as_array().unwrap() {
        let dev = device(c["device"].as_str().unwrap());
        let mut ordered = c["specs"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    v[0].as_str()
                        .unwrap()
                        .rsplit('_')
                        .next()
                        .unwrap()
                        .parse::<u64>()
                        .unwrap(),
                    data(&numbers(&v[1]), dev, Kind::Float, 7 + i as i64).g_add_scalar(0.5),
                )
            })
            .collect::<Vec<_>>();
        ordered.sort_by_key(|(n, _)| *n);
        let images = ordered.into_iter().map(|(_, t)| t).collect::<Vec<_>>();
        let mut vae_inputs = vec![];
        let mut encode = |x: &Tensor| -> Result<Tensor, Error> {
            vae_inputs.push(x.shallow_clone());
            Ok(Tensor::zeros(
                [1, 64, x.size()[1] / 16, x.size()[2] / 16],
                (Kind::Float, Device::Cpu),
            ))
        };
        let out = q::prepare_reference_images(
            &pillow,
            &images,
            c["resolution"].as_i64().unwrap(),
            if c["with_vae"] == true {
                Some(&mut encode)
            } else {
                None
            },
        )
        .unwrap();
        assert_eq!(out.vision.len(), c["vision"].as_array().unwrap().len());
        assert_eq!(vae_inputs.len(), c["vae"].as_array().unwrap().len());
        for (t, e) in out.vision.into_iter().zip(c["vision"].as_array().unwrap()) {
            check(&V::tensor(t), e, dev, "vision alpha over white");
        }
        for (t, e) in vae_inputs.into_iter().zip(c["vae"].as_array().unwrap()) {
            check(
                &V::tensor(t),
                e,
                dev,
                "VAE keeps RGBA in original per-image order",
            );
        }
        let latent = Tensor::zeros(
            [1, 64, out.height / 16, out.width / 16],
            (Kind::Float, Device::Cpu),
        );
        check(
            &V::tensor(latent),
            &c["latent"],
            dev,
            "first reference determines latent geometry",
        );
        assert_eq!(
            out.reference_latents.is_empty(),
            c["keep_vision"].as_bool().unwrap()
        );
        assert_eq!(c["matching_positive_negative_images"], true);
    }
    // The C ownership hooks replace original GIL sections; independent callers
    // must retain identical conversion/resampling bytes under concurrent calls.
    let expected = pillow
        .resize_rgb(&[127; 4 * 7 * 11], 11, 7, 4, 19, 13)
        .unwrap();
    let tasks = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                let p = unsafe {
                    q::Pillow::open(Path::new(&std::env::var("IMAGINE_PILLOW_LIBRARY").unwrap()))
                        .unwrap()
                };
                (0..8)
                    .map(|_| p.resize_rgb(&[127; 4 * 7 * 11], 11, 7, 4, 19, 13).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    for task in tasks {
        for actual in task.join().unwrap() {
            assert_eq!(actual, expected);
        }
    }
}
