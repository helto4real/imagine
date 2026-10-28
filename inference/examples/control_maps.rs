//! MOSS control maps from a photo on CUDA: the Depth Anything V2 Small depth map
//! and the Mask2Former Swin-L region map, as raw RGB next to the given output stem.
//!
//! `control_maps <depth model dir> <region model dir> <photo.rgb> <height> <width> <out stem>`
use imagine_inference::models::{depth_anything as d, mask2former as m, qwen_image21::Pillow};
use std::path::Path;
use tch::{Device, Kind, Tensor};
fn bytes(t: &Tensor) -> Vec<u8> {
    let t = t.contiguous().to_device(Device::Cpu);
    let mut out = vec![0u8; t.numel()];
    t.f_copy_data_u8(&mut out, t.numel()).unwrap();
    out
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [depth_dir, region_dir, photo, h, w, out] = &args[..] else {
        return Err("control_maps <depth dir> <region dir> <photo.rgb> <h> <w> <out stem>".into());
    };
    let (h, w): (i64, i64) = (h.parse()?, w.parse()?);
    let _native = unsafe {
        imagine_inference::native::Kitchen::load(Path::new(&std::env::var(
            "IMAGINE_KITCHEN_LIBRARY",
        )?))?
    };
    let _grad = tch::no_grad_guard();
    let dev = Device::Cuda(0);
    let pillow = unsafe { Pillow::open(Path::new(&std::env::var("IMAGINE_PILLOW_LIBRARY")?))? };
    let rgb = std::fs::read(photo)?;
    let net = d::DepthAnything::new(
        d::DepthAnythingConfig::small(d::DinoAttention::Sdpa),
        d::depth_weights(
            Tensor::read_safetensors(Path::new(depth_dir).join("model.safetensors"))?,
            dev,
        )?,
    );
    let depth = net.forward(&d::depth_pixels(&pillow, &rgb, h, w)?.to_device(dev))?;
    let (_, depth) = d::depth_map(&depth.get(0), h, w)?;
    let depth = depth.unsqueeze(-1).expand([h, w, 3], false);
    std::fs::write(format!("{out}-depth.rgb"), bytes(&depth))?;
    drop(net);
    let net = m::Mask2Former::new(
        m::Mask2FormerConfig::swin_large(),
        m::segmentation_weights(
            Tensor::read_safetensors(Path::new(region_dir).join("model.safetensors"))?,
            dev,
        )?,
    );
    let out_ = net.forward(&m::segmentation_pixels(&pillow, &rgb, h, w)?.to_device(dev))?;
    let p = m::panoptic(
        &out_.class_logits.get(0),
        &out_.mask_logits.get(0),
        h,
        w,
        0.8,
        0.8,
        |label| label >= m::COCO_PANOPTIC_THINGS,
    )?;
    let palette = [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ];
    let colors = m::region_colors(&p.segmentation, &palette)?;
    std::fs::write(
        format!("{out}-regions.rgb"),
        bytes(&colors.to_kind(Kind::Uint8)),
    )?;
    println!(
        "segments {:?}",
        p.segments.iter().map(|s| s.label).collect::<Vec<_>>()
    );
    Ok(())
}
