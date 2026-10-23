//! Explicit own-fixture CPU adapter for Reference media preprocessing.
use imagine_inference::{
    h3_product::{protocol as p, reference_video, references},
    native::Error,
};
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};
fn own(path: &Path) -> PathBuf {
    let root = Path::new("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e");
    assert!(path.starts_with(root));
    path.to_path_buf()
}
fn write(path: &Path, tensor: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    let tensor = tensor
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Float)?
        .f_contiguous()?;
    let mut values = vec![0f32; tensor.numel()];
    let count = values.len();
    tensor.f_copy_data(&mut values, count)?;
    std::fs::write(
        own(path),
        values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::set_num_threads(2);
    let path = own(Path::new(&std::env::args().nth(1).unwrap()));
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let shape: p::Shape = serde_json::from_value(data["shape"].clone())?;
    let mut input: p::references::Input = serde_json::from_value(data["references"].clone())?;
    for (image, path) in input
        .images
        .iter_mut()
        .zip(data["image_files"].as_array().unwrap())
    {
        image.rgb = std::fs::read(own(Path::new(path.as_str().unwrap())))?;
    }
    for (audio, path) in input
        .audios
        .iter_mut()
        .zip(data["audio_files"].as_array().unwrap())
    {
        audio.pcm = std::fs::read(own(Path::new(path.as_str().unwrap())))?;
    }
    for (i, video) in input.videos.iter_mut().enumerate() {
        video.rgb = std::fs::read(own(Path::new(data["video_files"][i].as_str().unwrap())))?;
        if let Some(audio) = &mut video.audio {
            audio.pcm = std::fs::read(own(Path::new(
                data["video_audio_files"][i].as_str().unwrap(),
            )))?;
        }
    }
    assert!(input.valid_payload());
    let images = references::images(&input, shape, Path::new(data["pillow"].as_str().unwrap()))?;
    let audios = input
        .audios
        .iter()
        .map(references::audio)
        .collect::<Result<Vec<_>, Error>>()?;
    for (i, tensor) in images.iter().enumerate() {
        write(
            Path::new(data["image_outputs"][i].as_str().unwrap()),
            tensor,
        )?;
    }
    for (i, tensor) in audios.iter().enumerate() {
        write(
            Path::new(data["audio_outputs"][i].as_str().unwrap()),
            tensor,
        )?;
    }
    let mut videos = vec![];
    for (i, video) in input.videos.iter().enumerate() {
        let frames =
            reference_video::prepare(video, shape, Path::new(data["pillow"].as_str().unwrap()))?;
        let qwen = reference_video::qwen(&frames)?;
        let vae = reference_video::vae_input(&frames)?;
        write(
            Path::new(data["video_outputs"][i].as_str().unwrap()),
            &frames,
        )?;
        write(
            Path::new(data["video_qwen_outputs"][i].as_str().unwrap()),
            &qwen,
        )?;
        write(
            Path::new(data["video_vae_outputs"][i].as_str().unwrap()),
            &vae,
        )?;
        videos.push(serde_json::json!({"frames":frames.size(),"qwen":qwen.size(),"vae_shape":vae.size(),"vae_dtype":format!("{:?}",vae.kind()),"vae_stride":vae.stride(),"timestamps":(0..qwen.size()[0]).map(|i|i as f64/2.).collect::<Vec<_>>()}));
    }
    println!(
        "{}",
        serde_json::json!({"images":images.iter().map(Tensor::size).collect::<Vec<_>>(),"videos":videos,"audios":audios.iter().map(Tensor::size).collect::<Vec<_>>(),"protocol":p::VERSION})
    );
    Ok(())
}
