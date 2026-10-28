//! The released Depth Anything V2 Small weights against the reference capture: the
//! prediction for the synthetic test image, its bicubic resize and the depth map,
//! exact on the captured device.
//!
//! `depth_anything_released <model dir> <fixture>`
use imagine_inference::models::{depth_anything as d, qwen_image21::Pillow};
use sha2::{Digest, Sha256};
use std::path::Path;
use tch::{Device, Tensor};
fn sha(t: &Tensor) -> String {
    let t = t.contiguous().to_device(Device::Cpu);
    let mut out = vec![0u8; t.numel() * t.kind().elt_size_in_bytes()];
    t.f_copy_data_u8(&mut out, t.numel()).unwrap();
    format!("{:x}", Sha256::digest(out))
}
/// The reference's integer test image `[height, width, 3]`.
fn image(height: i64, width: i64, offset: i64) -> Vec<u8> {
    let mut out = Vec::with_capacity((height * width * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            out.push((x * 255 / (width - 1).max(1)) as u8);
            out.push((y * 255 / (height - 1).max(1)) as u8);
            out.push((((x + y) * 7 + offset) % 256) as u8);
        }
    }
    out
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [model, fixture] = &args[..] else {
        return Err("depth_anything_released <model dir> <fixture>".into());
    };
    let f: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture)?)?;
    let r = &f["released"];
    let dev = match r["device"].as_str() {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    let attention = match r["attention"].as_str() {
        Some("sdpa") => d::DinoAttention::Sdpa,
        _ => d::DinoAttention::Eager,
    };
    tch::set_num_threads(2);
    let _grad = tch::no_grad_guard();
    let (h, w, o) = (
        r["height"].as_i64().unwrap(),
        r["width"].as_i64().unwrap(),
        r["offset"].as_i64().unwrap(),
    );
    let pillow = unsafe { Pillow::open(Path::new(&std::env::var("IMAGINE_PILLOW_LIBRARY")?))? };
    let pixels = d::depth_pixels(&pillow, &image(h, w, o), h, w)?.to_device(dev);
    let weights = d::depth_weights(
        Tensor::read_safetensors(Path::new(model).join("model.safetensors"))?,
        dev,
    )?;
    let net = d::DepthAnything::new(d::DepthAnythingConfig::small(attention), weights);
    let predicted = net.forward(&pixels)?;
    let (resized, map) = d::depth_map(&predicted.get(0), h, w)?;
    let mut failures = Vec::new();
    for (what, got, key) in [
        ("prediction", sha(&predicted), "predicted_depth"),
        ("resized", sha(&resized), "resized"),
        ("depth map", sha(&map), "depth_map"),
    ] {
        let ok = r[key]["payload_sha256"].as_str() == Some(got.as_str());
        println!("{what}: {}", if ok { "exact" } else { "DIFFERS" });
        if !ok {
            failures.push(what);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("Differs from the reference: {}", failures.join(", ")).into())
    }
}
