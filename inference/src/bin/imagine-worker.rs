//! Private length-bounded pipe protocol; prompts/pixels never become files or argv.
use imagine_inference::{
    checkpoint::{Aimdo, bind_comfy_layouts},
    execution::Runtime,
    native::{Error, Kitchen},
    server::{self, Event, Request},
    triton_compiler::Compiler,
};
use serde_json::{Value, json};
use std::{
    io::{self, Read, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tch::{Device, Kind, Tensor};
const MAX_HEADER: usize = 1024 * 1024;
const MAX_PAYLOAD: usize = 256 * 1024 * 1024;
fn path(key: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(std::env::var_os(key)
        .ok_or("Missing native runtime dependency")?
        .into())
}
fn read(reader: &mut impl Read, max: usize) -> io::Result<Vec<u8>> {
    let mut len = [0; 4];
    reader.read_exact(&mut len)?;
    let n = u32::from_be_bytes(len) as usize;
    if n > max {
        return Err(io::Error::other("Frame exceeds bound"));
    }
    let mut data = vec![0; n];
    reader.read_exact(&mut data)?;
    Ok(data)
}
fn write(out: &mut impl Write, event: Value, data: &[u8]) -> io::Result<()> {
    let json = serde_json::to_vec(&event)?;
    if json.len() > MAX_HEADER || data.len() > MAX_PAYLOAD {
        return Err(io::Error::other("Frame exceeds bound"));
    }
    out.write_all(&(json.len() as u32).to_be_bytes())?;
    out.write_all(&json)?;
    out.write_all(&(data.len() as u32).to_be_bytes())?;
    out.write_all(data)?;
    out.flush()
}
fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let native = unsafe { Kitchen::load(&path("IMAGINE_KITCHEN_LIBRARY")?) }?;
    let cache = path("IMAGINE_WORKER_CACHE")?;
    std::fs::create_dir_all(&cache)?;
    let compiler = Arc::new(unsafe {
        Compiler::open(
            &path("IMAGINE_TRITON_COMPILER_PROFILE")?,
            &cache.join("compiler"),
            &cache.join("work"),
            native.clone(),
        )
    }?);
    let rt = Runtime::frozen(native, compiler)?;
    rt.backends.policy.disable("triton");
    let mut sdpa = rt.native.sdpa_state()?;
    sdpa[..3].copy_from_slice(&[1, 1, 1]);
    rt.native.set_sdpa_state(sdpa)?;
    let mut math = rt.native.comfy_math_state()?;
    math[0] = 1;
    math[1] = 0;
    rt.native.set_comfy_math_state(math)?;
    bind_comfy_layouts(&rt);
    let aimdo = unsafe { Aimdo::load(&path("IMAGINE_AIMDO_LIBRARY")?) }?;
    rt.native.cuda_memory(0)?;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    write(
        &mut output,
        json!({"type":"ready","protocol":1,"attention":"checkpoint_preferences"}),
        &[],
    )?;
    loop {
        let header = match read(&mut input, MAX_HEADER) {
            Ok(v) => v,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let payload = read(&mut input, MAX_PAYLOAD)?;
        let value: Value = serde_json::from_slice(&header)?;
        if value["protocol"] != 1 || value["type"] != "generate" {
            return Err("Invalid frame".into());
        }
        let id = value["id"]
            .as_str()
            .ok_or("Missing request identity")?
            .to_owned();
        let result = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let request: Request = serde_json::from_value(value.clone())?;
            let lengths = value["reference_lengths"]
                .as_array()
                .ok_or("Missing references")?;
            let shapes = value["reference_shapes"]
                .as_array()
                .ok_or("Missing reference shapes")?;
            if shapes.len() != lengths.len() {
                return Err("Reference shape count".into());
            }
            if lengths.len() > 10 {
                return Err("Too many references".into());
            }
            let mut offset = 0usize;
            let mut images = vec![];
            for (i, n) in lengths.iter().enumerate() {
                let n = n.as_u64().ok_or("Invalid reference length")? as usize;
                let end = offset
                    .checked_add(n)
                    .filter(|n| *n <= payload.len())
                    .ok_or("Invalid reference boundary")?;
                let h = shapes[i][0]
                    .as_i64()
                    .filter(|n| *n > 0 && *n <= 8192)
                    .ok_or("Invalid reference height")?;
                let w = shapes[i][1]
                    .as_i64()
                    .filter(|n| *n > 0 && *n <= 8192)
                    .ok_or("Invalid reference width")?;
                if n != h as usize * w as usize * 4 {
                    return Err("Invalid RGBA reference size".into());
                }
                let values = payload[offset..end]
                    .iter()
                    .map(|v| *v as f32 / 255.)
                    .collect::<Vec<_>>();
                images.push(Tensor::f_from_slice(&values)?.f_view([1, h, w, 4])?);
                offset = end;
            }
            if offset != payload.len() {
                return Err("Trailing reference data".into());
            }
            let start = Instant::now();
            let mut observer = |event| {
                let event = match event {
                    Event::Phase(phase) => json!({"type":"phase","id":id,"phase":phase}),
                    Event::Step(i) => {
                        json!({"type":"progress","id":id,"completed":i,"total":request.sampling.steps,"elapsed_seconds":start.elapsed().as_secs_f64()})
                    }
                };
                write(&mut output, event, &[]).map_err(|e| Error::Native(e.to_string()))
            };
            let pixels = server::generate(&rt, &aimdo, &request, &images, &mut observer)?
                .f_get(0)?
                .f_to_device(Device::Cpu)?
                .f_mul_scalar(255.)?
                .f_to_kind(Kind::Uint8)?
                .f_contiguous()?;
            let shape = pixels.size();
            let mut bytes = vec![0u8; pixels.numel()];
            let n = bytes.len();
            pixels.f_copy_data_u8(&mut bytes, n)?;
            if !matches!(shape[2], 3 | 4) {
                return Err("Invalid output channels".into());
            }
            Ok(bytes)
        })();
        match result {
            Ok(pixels) => write(&mut output, json!({"type":"result","id":id}), &pixels)?,
            Err(_) => {
                write(
                    &mut output,
                    json!({"type":"error","id":id,"code":"rust_generation_failed"}),
                    &[],
                )?;
                return Ok(());
            }
        }
    }
}
fn main() {
    // Never print third-party exceptions, file paths or user inputs to a log.
    if serve().is_err() {
        let _ = write(
            &mut io::stdout().lock(),
            json!({"type":"error","id":null,"code":"rust_startup_failed"}),
            &[],
        );
        std::process::exit(1)
    }
}
