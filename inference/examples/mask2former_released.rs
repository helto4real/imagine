//! The released Mask2Former Swin-L COCO panoptic weights against the reference
//! capture: the processed coffee photo, the class and mask logits, the panoptic
//! segments and their region colors, exact on the captured device.
//!
//! `mask2former_released <model dir> <coffee.rgb> <fixture> [colors.rgb]`
use imagine_inference::models::{mask2former as m, qwen_image21::Pillow};
use sha2::{Digest, Sha256};
use std::path::Path;
use tch::{Device, Tensor};
fn sha(t: &Tensor) -> String {
    let t = t.contiguous().to_device(Device::Cpu);
    let mut out = vec![0u8; t.numel() * t.kind().elt_size_in_bytes()];
    t.f_copy_data_u8(&mut out, t.numel()).unwrap();
    format!("{:x}", Sha256::digest(out))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [model, rgb, fixture, rest @ ..] = &args[..] else {
        return Err("mask2former_released <model dir> <coffee.rgb> <fixture> [colors.rgb]".into());
    };
    let f: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fixture)?)?;
    let r = &f["released"];
    let dev = match r["device"].as_str() {
        Some("cuda") => Device::Cuda(0),
        _ => Device::Cpu,
    };
    // CUDA captures need the native library, which loads torch's CUDA backend.
    let _native = match dev {
        Device::Cuda(_) => Some(unsafe {
            imagine_inference::native::Kitchen::load(Path::new(&std::env::var(
                "IMAGINE_KITCHEN_LIBRARY",
            )?))?
        }),
        _ => None,
    };
    tch::set_num_threads(2);
    let rgb = std::fs::read(rgb)?;
    let (h, w) = (
        r["image"]["height"].as_i64().unwrap(),
        r["image"]["width"].as_i64().unwrap(),
    );
    let mut failures = Vec::new();
    let mut expect = |what: &str, got: String, want: &serde_json::Value| {
        let ok = want.as_str() == Some(got.as_str());
        println!("{what}: {}", if ok { "exact" } else { "DIFFERS" });
        if !ok {
            failures.push(what.to_string());
        }
    };
    expect(
        "image",
        format!("{:x}", Sha256::digest(&rgb)),
        &r["image"]["rgb_sha256"],
    );
    let pillow = unsafe { Pillow::open(Path::new(&std::env::var("IMAGINE_PILLOW_LIBRARY")?))? };
    let pixels = m::segmentation_pixels(&pillow, &rgb, h, w)?.to_device(dev);
    expect(
        "pixel values",
        sha(&pixels),
        &r["pixel_values"]["payload_sha256"],
    );
    let weights = m::segmentation_weights(
        Tensor::read_safetensors(Path::new(model).join("model.safetensors"))?,
        dev,
    )?;
    let net = m::Mask2Former::new(m::Mask2FormerConfig::swin_large(), weights);
    let out = net.forward(&pixels)?;
    expect(
        "class logits",
        sha(&out.class_logits),
        &r["class_logits"]["payload_sha256"],
    );
    expect(
        "mask logits",
        sha(&out.mask_logits),
        &r["mask_logits"]["payload_sha256"],
    );
    let p = m::panoptic(
        &out.class_logits.get(0),
        &out.mask_logits.get(0),
        h,
        w,
        0.8,
        0.8,
        |label| label >= m::COCO_PANOPTIC_THINGS,
    )?;
    expect(
        "segmentation",
        sha(&p.segmentation),
        &r["segmentation"]["payload_sha256"],
    );
    let segments = p
        .segments
        .iter()
        .map(|s| {
            serde_json::json!({"id": s.id, "label": s.label, "fused": s.fused, "score_bits": s.score.to_bits()})
        })
        .collect::<serde_json::Value>();
    let ok = segments == r["segments"];
    println!("segments: {}", if ok { "exact" } else { "DIFFERS" });
    if !ok {
        failures.push("segments".into());
    }
    if let [colors, ..] = rest {
        let palette = [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200]];
        let map = m::region_colors(&p.segmentation, &palette)?;
        let mut out = vec![0u8; map.numel()];
        map.to_device(Device::Cpu)
            .f_copy_data_u8(&mut out, map.numel())?;
        std::fs::write(colors, out)?;
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!("Differs from the reference: {}", failures.join(", ")).into())
    }
}
