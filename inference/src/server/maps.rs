//! MOSS control maps from a photo: depth from Depth Anything V2 Small (near is
//! white) and one flat color per panoptic segment from Mask2Former Swin-L COCO
//! panoptic (black where no segment won), both at the photo's size.
use crate::{
    models::{depth_anything as d, mask2former as m, qwen_image21::Pillow},
    native::Error,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path, path::PathBuf};
use tch::{Device, Kind, Tensor};

/// `model.safetensors` of the released Small weights and of the converted Swin-L
/// checkpoint (`inference/tools/convert_mask2former.py`).
const DEPTH_SHA256: &str = "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70";
const REGION_SHA256: &str = "9a34326fd19fdb6b972889cf68eecbdd4a1f2fc17e046ce407494f8240d1c221";
/// Region colors by segment, none black; the colors only identify regions.
const PALETTE: [[u8; 3]; 12] = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 212],
    [0, 128, 128],
    [170, 110, 40],
];

#[derive(Deserialize)]
pub struct MapsRequest {
    /// Model directories holding `model.safetensors` (and Mask2Former's `config.json`).
    pub depth_model: PathBuf,
    pub region_model: PathBuf,
    /// The photo's `[height, width]`.
    pub shape: [i64; 2],
}
#[derive(Serialize, Debug, PartialEq)]
pub struct Region {
    pub color: [u8; 3],
    pub label: String,
    pub share: f32,
}
/// RGB bytes `[height, width, 3]` of both maps and the regions they name.
pub struct Maps {
    pub depth: Vec<u8>,
    pub regions: Vec<u8>,
    pub segments: Vec<Region>,
}
fn native(e: impl std::fmt::Display) -> Error {
    Error::Native(e.to_string())
}
/// The weights after checking the file's pinned hash.
fn verified(path: &Path, expected: &str) -> Result<Vec<(String, Tensor)>, Error> {
    let mut file = std::fs::File::open(path).map_err(native)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(native)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    if format!("{:x}", hasher.finalize()) != expected {
        return Err(Error::Native("Control map model changed".into()));
    }
    Ok(Tensor::read_safetensors(path)?)
}
fn bytes(t: &Tensor) -> Result<Vec<u8>, Error> {
    let t = t.f_contiguous()?.f_to_device(Device::Cpu)?;
    let mut out = vec![0u8; t.numel()];
    t.f_copy_data_u8(&mut out, t.numel())?;
    Ok(out)
}
/// Segments as regions: one per segment id in color order, with the COCO name.
fn regions(
    panoptic: &m::Panoptic,
    names: &serde_json::Value,
    pixels: i64,
) -> Result<Vec<Region>, Error> {
    let counts = Vec::<i64>::try_from(
        panoptic
            .segmentation
            .f_flatten(0, -1)?
            .f_to_kind(Kind::Int64)?
            .f_bincount(None::<Tensor>, 0)?
            .f_to_device(Device::Cpu)?,
    )?;
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for s in &panoptic.segments {
        if !seen.insert(s.id) {
            continue;
        }
        out.push(Region {
            color: PALETTE[(s.id as usize - 1) % PALETTE.len()],
            label: names[s.label.to_string()]
                .as_str()
                .unwrap_or("region")
                .to_owned(),
            share: counts.get(s.id as usize).copied().unwrap_or(0) as f32 / pixels as f32,
        });
    }
    Ok(out)
}
/// Both maps for an RGB photo `[height, width, 3]` on the GPU.
pub fn control_maps(r: &MapsRequest, rgb: &[u8]) -> Result<Maps, Error> {
    let [h, w] = r.shape;
    if h <= 0 || w <= 0 || h > 8192 || w > 8192 || rgb.len() as i64 != h * w * 3 {
        return Err(Error::Native("Invalid control map photo".into()));
    }
    let dev = Device::Cuda(0);
    let _grad = tch::no_grad_guard();
    let pillow = unsafe {
        Pillow::open(&PathBuf::from(
            std::env::var_os("IMAGINE_PILLOW_LIBRARY")
                .ok_or_else(|| Error::Native("Missing Pillow ABI".into()))?,
        ))
    }?;
    // transformers selects SDPA for the backbone; the reference capture records it.
    let depth = {
        let net = d::DepthAnything::new(
            d::DepthAnythingConfig::small(d::DinoAttention::Sdpa),
            d::depth_weights(
                verified(&r.depth_model.join("model.safetensors"), DEPTH_SHA256)?,
                dev,
            )?,
        );
        let predicted = net.forward(&d::depth_pixels(&pillow, rgb, h, w)?.f_to_device(dev)?)?;
        let (_, map) = d::depth_map(&predicted.f_get(0)?, h, w)?;
        bytes(&map.f_unsqueeze(-1)?.f_expand([h, w, 3], false)?)?
    };
    let names: serde_json::Value =
        serde_json::from_slice(&std::fs::read(r.region_model.join("config.json")).map_err(native)?)
            .map_err(native)?;
    let names = &names["id2label"];
    let (regions_rgb, segments) = {
        let net = m::Mask2Former::new(
            m::Mask2FormerConfig::swin_large(),
            m::segmentation_weights(
                verified(&r.region_model.join("model.safetensors"), REGION_SHA256)?,
                dev,
            )?,
        );
        let out = net.forward(&m::segmentation_pixels(&pillow, rgb, h, w)?.f_to_device(dev)?)?;
        let panoptic = m::panoptic(
            &out.class_logits.f_get(0)?,
            &out.mask_logits.f_get(0)?,
            h,
            w,
            0.8,
            0.8,
            |label| label >= m::COCO_PANOPTIC_THINGS,
        )?;
        (
            bytes(&m::region_colors(&panoptic.segmentation, &PALETTE)?)?,
            regions(&panoptic, names, h * w)?,
        )
    };
    Ok(Maps {
        depth,
        regions: regions_rgb,
        segments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_take_palette_colors_names_and_shares_once_per_segment() {
        let panoptic = m::Panoptic {
            segmentation: Tensor::from_slice(&[0i32, 1, 1, 2, 2, 2, 1, 0]).view([2, 4]),
            segments: vec![
                m::Segment {
                    id: 1,
                    label: 0,
                    fused: false,
                    score: 0.9,
                },
                m::Segment {
                    id: 2,
                    label: 122,
                    fused: true,
                    score: 0.95,
                },
                m::Segment {
                    id: 2,
                    label: 122,
                    fused: true,
                    score: 0.85,
                },
            ],
        };
        let names = serde_json::json!({"0": "person", "122": "floor-other-merged"});
        assert_eq!(
            regions(&panoptic, &names, 8).unwrap(),
            [
                Region {
                    color: PALETTE[0],
                    label: "person".into(),
                    share: 3. / 8.,
                },
                Region {
                    color: PALETTE[1],
                    label: "floor-other-merged".into(),
                    share: 3. / 8.,
                },
            ]
        );
    }
}
