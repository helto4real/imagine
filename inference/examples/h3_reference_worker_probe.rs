//! Opt-in own-test observer. The product worker always uses run() without this callback.
use imagine_inference::{h3_product, native::Error};
use std::{io::Write, path::PathBuf, sync::OnceLock};
use tch::{Device, Kind, Tensor};
static DESTINATION: OnceLock<PathBuf> = OnceLock::new();
fn observe(name: &str, tensor: &Tensor) -> Result<(), Error> {
    let destination = DESTINATION.get().unwrap();
    let value = tensor
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Float)?
        .f_contiguous()?;
    let mut values = vec![0f32; value.numel()];
    let count = values.len();
    value.f_copy_data(&mut values, count)?;
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let save = |suffix: &str, bytes: &[u8]| -> Result<(), Error> {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination.join(format!("{name}.{suffix}")))
            .and_then(|mut f| f.write_all(bytes))
            .map_err(|e| Error::Native(e.to_string()))
    };
    save("f32", &bytes)?;
    save("json",serde_json::json!({"name":name,"shape":tensor.size(),"dtype":format!("{:?}",tensor.kind()),"samples":values.len(),"finite":values.iter().all(|v|v.is_finite()),"rms":(values.iter().map(|v|f64::from(*v).powi(2)).sum::<f64>()/values.len() as f64).sqrt()}).to_string().as_bytes())?;
    Ok(())
}
fn run() -> Result<(), h3_product::protocol::ErrorCode> {
    let output = PathBuf::from(
        std::env::var_os("IMAGINE_REFERENCE_PROBE_OUTPUT")
            .ok_or(h3_product::protocol::ErrorCode::InvalidRequest)?,
    );
    if !output.starts_with("/run/media/veracrypt1/imagine/.codex-video-night-20261020-a6428d9e")
        || !output.is_dir()
    {
        return Err(h3_product::protocol::ErrorCode::InvalidRequest);
    }
    DESTINATION.set(output).unwrap();
    h3_product::run_observed(observe)
}
fn main() {
    if !h3_product::wire::quiet_entry(run) {
        std::process::exit(1);
    }
}
