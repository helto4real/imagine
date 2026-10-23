//! Explicit own tensor component test, outside the product worker protocol.
use imagine_inference::h3_product::upscale::{Upscaler, canvas};
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(6..=7).contains(&args.len()) {
        return Err("model input output forward|video factor cpu|cuda [kitchen]".into());
    }
    let model = PathBuf::from(&args[0]);
    let input = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let own = std::path::Path::new(
        "/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e/ui-groups-01",
    );
    if !input.starts_with(own) || !output.starts_with(own) || output.exists() {
        return Err("Own new encrypted component paths required".into());
    }
    let factor: f64 = args[4].parse()?;
    let device = if args[5] == "cuda" {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    tch::set_num_threads(2);
    let _grad = tch::no_grad_guard();
    let runtime = if matches!(device, Device::Cuda(_)) {
        let path = args
            .get(6)
            .ok_or("GPU component requires the existing runtime bridge")?;
        Some(imagine_inference::execution::Runtime::frozen_partial(
            unsafe { imagine_inference::native::Kitchen::load(std::path::Path::new(path)) }?,
        )?)
    } else {
        None
    };
    let values = Tensor::read_safetensors(&input)?;
    let input = &values
        .iter()
        .find(|(key, _)| key == "input")
        .ok_or("Missing own input")?
        .1;
    let mut model = Upscaler::load(&model)?;
    let output_tensor = if args[3] == "video" {
        model.video(input, factor, device)?
    } else {
        model.to(device, Kind::Float)?;
        let input = input.f_to_device(device)?;
        let (w, h) = canvas(
            (input.size()[4] * 16) as u32,
            (input.size()[3] * 16) as u32,
            factor,
        );
        model
            .forward(
                &input,
                factor,
                [input.size()[2], i64::from(h / 16), i64::from(w / 16)],
                true,
            )?
            .f_to_device(Device::Cpu)?
    };
    if let Some(runtime) = &runtime {
        runtime.native.synchronize_device(0)?;
    }
    Tensor::write_safetensors(&[("output", &output_tensor)], &output)?;
    println!(
        "{}",
        serde_json::json!({"output":output,"shape":output_tensor.size(),"dtype":format!("{:?}",output_tensor.kind()),"device":args[5]})
    );
    Ok(())
}
