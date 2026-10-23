//! Own CPU-only reference RGB/padding comparison; no model/runtime/GPU owner.
use imagine_inference::h3_product::{
    protocol::{Shape, references as wire},
    references,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let (width, height) = (args[2].parse::<u32>()?, args[3].parse::<u32>()?);
    let quality = &args[4];
    let boundary = args[5] == "boundary";
    let rgb = (0..width * height * 3)
        .map(|i| {
            let (x, y, channel) = ((i / 3) % width, (i / 3) / width, i % 3);
            ((x * 13 + y * 17 + channel * 53 + ((x ^ y) % 19) * 11) % 256) as u8
        })
        .collect();
    let input = wire::Input {
        quality: quality.clone(),
        manifest: json!({"images":[{"role":if boundary {"boundary_first_frame"} else {"image_reference"}}]}),
        images: vec![wire::Image { width, height, rgb }],
        ..Default::default()
    };
    let shape = Shape {
        width: 416,
        height: 224,
        frames: 39,
        fps: 24,
    };
    let output = references::images(&input, shape, Path::new(&args[1]))?.remove(0);
    let dimensions = output.size();
    let values = Vec::<f32>::try_from(output.f_contiguous()?.f_view([-1])?)?;
    let actual = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let mut expected = vec![];
    std::io::stdin().read_to_end(&mut expected)?;
    if expected.len() != actual.len() {
        return Err(format!("Expected {} bytes, got {}", expected.len(), actual.len()).into());
    }
    let (mut sum, mut squares, mut maximum) = (0., 0., 0f64);
    for (value, bytes) in values.iter().zip(expected.chunks_exact(4)) {
        let reference = f32::from_le_bytes(bytes.try_into()?);
        let difference = f64::from(*value) - f64::from(reference);
        sum += difference.abs();
        squares += difference * difference;
        maximum = maximum.max(difference.abs());
    }
    let exact = actual == expected;
    println!(
        "{}",
        json!({"shape":dimensions,"dtype":"float32","device":"cpu","values":values.len(),
            "byte_identical":exact,"mae":sum/values.len()as f64,"rmse":(squares/values.len()as f64).sqrt(),
            "max":maximum,"actual_sha256":format!("{:x}",Sha256::digest(&actual)),
            "expected_sha256":format!("{:x}",Sha256::digest(&expected)),"runtime_initialized":false})
    );
    if !exact {
        std::process::exit(1);
    }
    Ok(())
}
